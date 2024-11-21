import logging
import torch
import os
from clearml import Task
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModel, MarianMTModel
from torch.utils.data import DataLoader
import time

from src import beir_utils
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


def eval_model(opt, query_encoder, doc_encoder, tokenizer, step):

    for datasetname in opt.eval_datasets:
        metrics = beir_utils.evaluate_model(
            query_encoder,
            doc_encoder,
            tokenizer,
            dataset=datasetname,
            batch_size=opt.per_gpu_eval_batch_size,
            norm_doc=opt.norm_doc,
            norm_query=opt.norm_query,
            beir_dir=opt.eval_datasets_dir,
            score_function=opt.score_function,
            lower_case=opt.lower_case,
            normalize_text=opt.eval_normalize_text,
        )

        message = []
        if dist_utils.is_main():
            for metric in ["NDCG@10", "Recall@10", "Recall@100"]:
                message.append(f"{datasetname}/{metric}: {metrics[metric]:.2f}")
            logger.info(" | ".join(message))


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
    student_model = student_model.to(local_rank)

    optimizer = torch.optim.AdamW(
        student_model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
    )

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

    loss_fn = torch.nn.MSELoss()

    step = 0
    for batch, _ in train_dataloader:
        step += 1
        start_time = time.time()
        batch = {k: v.to(local_rank) for k, v in batch.items()}
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

        teacher_output = teacher_model(**eng_tokens)
        teacher_output = mean_pooling(teacher_output[0], eng_tokens["attention_mask"])
        student_output, _ = student_model(**batch)
        loss = loss_fn(teacher_output, student_output)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        print(f"Time: {end_time - start_time}")
        print(f"Loss: {loss.item()}")
        optimizer.zero_grad()
        encoder = student_model.get_encoder()
        if (step % 100) == 0:
            # Eval
            eval_model(
                opt,
                query_encoder=encoder,
                doc_encoder=encoder,
                tokenizer=student_model.tokenizer,
                step=step,
            )

    with torch.inference_mode():
        print("Inference mode")
