import logging
import torch
import os
from clearml import Task
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModel, MarianMTModel
from torch.utils.data import DataLoader

from src.options import Options
from src import utils, dist_utils
from src.data import DistillCollator
from src.distiller import Distiller
from src.data import load_distill_data


logger = logging.getLogger(__name__)


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


if __name__ == "__main__":
    logger.info("START")
    options = Options()
    opt = options.parse()

    torch.manual_seed(opt.seed)

    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    directory_exists = os.path.isdir(opt.output_dir)
    if dist.is_initialized():
        dist.barrier()
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(opt.save_dir, exist_ok=True)
    if not directory_exists and dist_utils.is_main():
        options.print_options(opt)
    if dist.is_initialized():
        dist.barrier()
    utils.init_logger(opt)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Teacher
    teacher_model = AutoModel.from_pretrained(opt.teacher_model_id)
    teacher_model = teacher_model.to(local_rank)
    teacher_tokenizer = AutoTokenizer.from_pretrained(opt.teacher_model_id)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Translator
    translator_model = MarianMTModel.from_pretrained(opt.translator_model_id)
    translator_model = translator_model.to(local_rank)
    translator_tokenizer = AutoTokenizer.from_pretrained(opt.translator_model_id)
    translator_model.eval()
    for param in translator_model.parameters():
        param.requires_grad = False

    # Student
    student_model = Distiller(opt)

    train_dataset, _ = load_distill_data(
        opt,
        teacher_tokenizer,
        translator_tokenizer,
        student_model.tokenizer,
        translator_model,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )

    collator = DistillCollator(opt)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.target_batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
    )

    for batch, _ in train_dataloader:
        text = student_model.tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        tokens = translator_tokenizer(text, return_tensors="pt", padding=True)
        tokens = {k: v.to(local_rank) for k, v in tokens.items()}
        with torch.inference_mode():
            translation = translator_model.generate(
                **tokens, num_beams=4, early_stopping=True
            )
        translation_text = translator_tokenizer.batch_decode(
            translation, skip_special_tokens=True
        )
        eng_tokens = teacher_tokenizer(
            translation_text, return_tensors="pt", padding=True
        )

        eng_tokens = {k: v.to(local_rank) for k, v in eng_tokens.items()}
        with torch.inference_mode():
            teacher_output = teacher_model(**eng_tokens)
            teacher_output = mean_pooling(
                teacher_output[0], eng_tokens["attention_mask"]
            )
        student_output, _ = student_model(**batch)
        print(teacher_output.shape)
        print(student_output.shape)

    with torch.inference_mode():
        print("Inference mode")
