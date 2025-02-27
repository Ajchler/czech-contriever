# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import time
import sys
import torch
import logging
import json
import numpy as np
import random
import pickle
from clearml import Task
from collections import defaultdict

import torch.distributed as dist
import torch.utils
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, MarianMTModel
from sentence_transformers import SentenceTransformer

from src.options import Options
from src import data, beir_utils, slurm, dist_utils, utils
from src import moco, inbatch
from src.data import build_mask
from src.utils import mean_pooling, load_hf

project_name = os.getenv("PROJECT_NAME", "czechtriever")
task_name = os.getenv("TASK_NAME", "czechtriever-default")
continue_training_env = os.getenv("CONTINUE_TRAINING", "False")
#if continue_training_env.lower() == "true":
#    Task.init(project_name=project_name, task_name=task_name, continue_last_task=True)
#else:
#    Task.init(project_name=project_name, task_name=task_name)

logger = logging.getLogger(__name__)

def compute_sim_matrix(embeddings):
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    sim_matrix = embeddings @ embeddings.T
    return sim_matrix

def gather_all_embeddings(local_embeddings, world_size):
    """Gathers all embeddings from different GPUs to GPU 0."""
    gathered_embeddings = [torch.zeros_like(local_embeddings) for _ in range(world_size)]
    dist.all_gather(gathered_embeddings, local_embeddings)
    return torch.cat(gathered_embeddings, dim=0) if dist.get_rank() == 0 else None

def eval_loss(opt, model, tb_logger, step, val_dataloader, all_docs, scheduler):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        encoder = model.module.get_encoder()
    else:
        encoder = model.get_encoder()

    os.makedirs(os.path.join(opt.output_dir, "logits"), exist_ok=True)

    all_texts_encoded = []
    nbatches = len(all_docs) // opt.per_gpu_eval_batch_size
    for i in range(nbatches):
        curr_docs = all_docs[
            i * opt.per_gpu_eval_batch_size : (i + 1) * opt.per_gpu_eval_batch_size
        ]
        docs_tokens, docs_masks = build_mask(curr_docs)
        with torch.no_grad():
            curr_docs_encoded = encoder(
                input_ids=docs_tokens.cuda(),
                attention_mask=docs_masks.cuda(),
                normalize=opt.eval_normalize_text,
            )

        all_texts_encoded.append(curr_docs_encoded)

    all_texts_encoded = torch.cat(all_texts_encoded, dim=0)

    all_indices = set(range(len(all_texts_encoded)))

    val_loss = 0
    recall_at_k = defaultdict(int)
    K = 10
    total_queries = 0
    sdtq_list = []
    sdtk_list = []

    for i, (batch, indices) in enumerate(val_dataloader):
        indices = list(all_indices - set(indices))
        usable_docs = all_texts_encoded[indices].cuda(non_blocking=True)

        batch = {
            key: value.cuda() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

        with torch.no_grad():
            q = encoder(
                input_ids=batch["q_tokens"],
                attention_mask=batch["q_mask"],
                normalize=opt.eval_normalize_text,
            )
            k = encoder(
                input_ids=batch["k_tokens"],
                attention_mask=batch["k_mask"],
                normalize=opt.eval_normalize_text,
            )

            stdq = torch.std(q, dim=0).mean().item()
            stdk = torch.std(k, dim=0).mean().item()
            sdtk_list.append(stdk)
            sdtq_list.append(stdq)

            l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
            l_neg = torch.einsum("nc,ck->nk", [q, usable_docs.cuda().transpose(0, 1)])

            logits = torch.cat([l_pos, l_neg], dim=1) / opt.temperature
            if i == 0:
                filename = os.path.join(opt.output_dir, "logits", f"step-{step}.pkl")
                with open(filename, "wb") as f:
                    pickle.dump(logits.cpu().numpy(), f)

            labels = torch.zeros(batch["q_tokens"].size(0), dtype=torch.long).cuda()
            loss = torch.nn.functional.cross_entropy(logits, labels)

            val_loss += loss.item()

            # Recall
            _, topk_indices = torch.topk(logits, K, dim=1, largest=True, sorted=False)
            topk_indices = topk_indices.cpu().numpy()

            for j, label in enumerate(labels.cpu().numpy()):
                if label in topk_indices[j]:
                    recall_at_k[min(K, logits.size(1))] += 1
                total_queries += 1

            if (i + 1) % 100 == 0:
                logger.info(f"Validation loss: {loss.item()} at step {i+1}")

            del q, k, l_pos, l_neg, logits, labels, usable_docs
            torch.cuda.empty_cache()

    recall_at_k_value = (
        (recall_at_k[K] * 100) / total_queries if total_queries > 0 else 0
    )

    tb_logger.add_scalar(f"val/recall@{K}", recall_at_k_value, step)
    sdtq = np.mean(sdtq_list)
    stdk = np.mean(sdtk_list)
    tb_logger.add_scalar("val/stdq", sdtq, step)
    tb_logger.add_scalar("val/stdk", stdk, step)
    avg_val_loss = val_loss / len(val_dataloader)
    tb_logger.add_scalar("val/loss", avg_val_loss, step)
    lr = scheduler.get_last_lr()[0]
    tb_logger.add_scalar("val/lr", lr, step)

def train(opt, student_model, teacher_model, student_tokenizer, prompt, optimizer, scheduler, step, local_rank):
    logger.warning(f"Rank {local_rank} is training")
    run_stats = utils.WeightedAvgStats()

    tb_logger = utils.init_tb_logger(opt.output_dir)

    logger.info("Data loading")
    if not dist_utils.is_main():
        if isinstance(student_model, torch.nn.parallel.DistributedDataParallel):
            tokenizer = student_model.module.tokenizer
        else:
            tokenizer = student_model.tokenizer
        collator = data.Collator(opt=opt)

        offsets = []
        cumsums = []

        train_dataset, val_dataset = data.load_data(
            opt, tokenizer, offsets, cumsums, is_main=dist_utils.is_main()
        )
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )
        logger.warning(f"Data loading finished for rank {dist_utils.get_rank()}")

    if not dist_utils.is_main():
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=opt.per_gpu_batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=opt.num_workers,
            collate_fn=collator,
        )

    #if local_rank == 1:
    #    val_dataloader = DataLoader(
    #        val_dataset,
    #        batch_size=opt.per_gpu_eval_batch_size,
    #        num_workers=opt.num_workers_valid,
    #        collate_fn=collator,
    #    )

    epoch = 1

    #if isinstance(student_model, torch.nn.parallel.DistributedDataParallel):
    #    encoder = student_model.module.get_encoder()
    #else:
    #    encoder = student_model.get_encoder()
    #eval_model(
    #    opt,
    #    query_encoder=encoder,
    #    doc_encoder=encoder,
    #    tokenizer=tokenizer,
    #    tb_logger=tb_logger,
    #    step=step,
    #    local_rank=local_rank,
    #)

    if opt.target_batch_size % (opt.per_gpu_batch_size * (dist.get_world_size() - 1) != 0):
        raise ValueError(
            "target_batch_size must be divisible by per_gpu_batch_size * dist.get_world_size()"
        )
    update_freq = opt.target_batch_size // (
        opt.per_gpu_batch_size * (dist.get_world_size() - 1)
    )

    #if local_rank == 1:
    #    eval_loss(
    #        opt,
    #        student_model,
    #        tb_logger,
    #        step,
    #        val_dataloader,
    #        val_dataset.get_passage_from_all_docs(),
    #        scheduler,
    #    )

    while step < opt.total_steps:
        if not dist_utils.is_main():
            logger.warning(f"Start epoch {epoch} for rank {local_rank}")
            student_model.train()
            train_dataset.generate_offset()
            logger.info(f"Start epoch {epoch}")
            if dist.is_initialized():
                sampler.set_epoch(epoch)

            accumulate_steps = 0

            for i, (batch, _) in enumerate(train_dataloader):
                accumulate_steps += 1

                batch = {
                    key: value.cuda() if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }

                # Send inputs to teacher
                queries, q_masks = batch["q_tokens"], batch["q_mask"]

                local_max_len = torch.tensor(queries.shape[1], dtype=torch.int64, device=queries.device)
                global_max_len = local_max_len.clone().detach()
                logger.warning(f"Process {dist.get_rank()} local_max_len: {local_max_len.item()} before reduction")
                dist.all_reduce(global_max_len, op=dist.ReduceOp.MAX, group=global_group)
                logger.warning(f"Process {dist.get_rank()} received max len: {global_max_len.item()}")

                dist.barrier(group=global_group)

                if local_max_len < global_max_len:
                    pad_size = global_max_len - local_max_len
                    pad_tensor = torch.full((queries.shape[0], pad_size), tokenizer.pad_token_id, device=queries.device)
                    queries = torch.cat([queries, pad_tensor], dim=1)

                logger.warning("Sending queries")
                gathered_queries = [torch.zeros_like(queries) for _ in range(dist.get_world_size() - 1)]
                dist.gather(queries, gathered_queries if dist.get_rank() == 0 else None, dst=0)
                logger.warning("Queries sent")

                dist.barrier(group=global_group)

                logger.warning("Calculating loss")
                train_loss, student_embeddings, iter_stats = model(**batch, process_group=student_group, stats_prefix="train")
                encoded_queries = torch.zeros((opt.per_gpu_batch_size, 384)).to(local_rank)
                dist.recv(encoded_queries, src=0)
                print(f"Received encoded queires with shape {encoded_queries.shape}, with values: {encoded_queries}")
                student_sim = compute_sim_matrix(student_embeddings)
                teacher_sim = compute_sim_matrix(encoded_queries)
                aux_loss = torch.nn.functional.mse_loss(student_sim, teacher_sim)
                print(f"Auxiliary loss is {aux_loss}")
                dist.barrier(group=global_group)
                logger.warning("Loss calculated")
                iter_stats["train/loss_contrastive"] = (train_loss.item(), batch["q_tokens"].size(0))
                train_loss = (1 - opt.distill_weight) * train_loss + opt.distill_weight * aux_loss
                train_loss.backward()
                iter_stats["train/loss"] = (train_loss.item(), batch["q_tokens"].size(0))
#
#                if accumulate_steps % update_freq == 0:
#                    run_stats.update(iter_stats)
#                    if step % opt.log_freq == 0:
#                        log = f"{step} / {opt.total_steps}"
#                        for k, v in sorted(run_stats.average_stats.items()):
#                            log += f" | {k}: {v:.3f}"
#                            if tb_logger:
#                                tb_logger.add_scalar(k, v, step)
#                        log += f" | lr: {scheduler.get_last_lr()[0]:0.3g}"
#                        log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"
#
#                        lr = scheduler.get_last_lr()[0]
#                        if tb_logger:
#                            tb_logger.add_scalar("train/lr", lr, step)
#
#                        global_grad_norm = 0
#
#                        for name, param in model.named_parameters():
#                            if param.grad is not None:
#                                norm = param.grad.norm().item()
#                                global_grad_norm += norm**2
#                                if tb_logger:
#                                    tb_logger.add_scalar(f"grad/{name}", norm, step)
#
#                        global_grad_norm = global_grad_norm**0.5
#
#                        if tb_logger:
#                            tb_logger.add_scalar(
#                                "train/global_grad", global_grad_norm, step
#                            )
#
#                        logger.info(log)
#                        run_stats.reset()
#
#                    if opt.clip_gradients:
#                        if opt.max_grad_value is not None:
#                            torch.nn.utils.clip_grad_value_(
#                                model.parameters(), opt.max_grad_value
#                            )
#                        elif opt.max_grad_norm is not None:
#                            torch.nn.utils.clip_grad_norm_(
#                                model.parameters(), opt.max_grad_norm
#                            )
#
#                    optimizer.step()
#                    scheduler.step()
#                    model.zero_grad()
#                    step += 1
#
#                if (step % opt.eval_freq == 0) and (step > 0):
#                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#                        encoder = model.module.get_encoder()
#                    else:
#                        encoder = model.get_encoder()
#                    eval_model(
#                        opt,
#                        query_encoder=encoder,
#                        doc_encoder=encoder,
#                        tokenizer=tokenizer,
#                        tb_logger=tb_logger,
#                        step=step,
#                        local_rank=local_rank,
#                    )
#
#                    if local_rank == 1:
#                        eval_loss(
#                            opt,
#                            model,
#                            tb_logger,
#                            step,
#                            val_dataloader,
#                            val_dataset.get_passage_from_all_docs(),
#                            scheduler,
#                        )
#
#                    if local_rank == 1:
#                        utils.save(
#                            model,
#                            optimizer,
#                            scheduler,
#                            step,
#                            opt,
#                            opt.save_dir,
#                            f"lastlog",
#                        )
#
#                    model.train()
#
#                if local_rank == 1 and step % opt.save_freq == 0 and step > 0:
#                    utils.save(
#                        model,
#                        optimizer,
#                        scheduler,
#                        step,
#                        opt,
#                        opt.save_dir,
#                        f"step-{step}",
#                    )

                if step > opt.total_steps:
                    break
        else:
            logger.warning(f"Start epoch {epoch} for rank {local_rank} (teacher)")
            # TODO: Teacher gathers inputs from all students and encodes them
            while True:
                logger.warning("Teacher process")
                local_max_len = torch.tensor(0, dtype=torch.int64, device="cuda:0")  # Teacher doesn't process queries, so it has len 0
                global_max_len = local_max_len.clone().detach()
                logger.warning(f"Process {dist.get_rank()} local_max_len: {local_max_len.item()} before reduction")
                dist.all_reduce(global_max_len, op=dist.ReduceOp.MAX, group=global_group)
                logger.warning(f"Process {dist.get_rank()} received max len: {global_max_len.item()}")

                dist.barrier(group=global_group)

                gathered_queries = [torch.zeros((opt.per_gpu_batch_size, global_max_len), dtype=torch.long, device="cuda:0") for _ in range(dist.get_world_size())]
                dummy_tensor = torch.zeros((opt.per_gpu_batch_size, global_max_len), dtype=torch.long, device="cuda:0")
                dist.gather(dummy_tensor, gathered_queries, dst=0)  # Gather from student processes

                gathered_queries = torch.cat(gathered_queries, dim=0)
                gathered_queries = gathered_queries[opt.per_gpu_batch_size:]
                texts = student_tokenizer.batch_decode(gathered_queries, skip_special_tokens=True)
                logger.warning("Queries gathered")
                dist.barrier(group=global_group)
                with torch.no_grad():
                    teacher_embeddings = torch.tensor(teacher_model.encode(texts)).to(local_rank)#, prompt)
    
                student_ranks = list(range(1, dist.get_world_size()))
                print(f"Student ranks: {student_ranks}")
                offset = 0
                for student_rank in student_ranks:
                    student_batch_size = opt.per_gpu_batch_size  # Assuming equal batch size
                    student_embedding = teacher_embeddings[offset : offset + student_batch_size].contiguous()
                    dist.send(student_embedding, dst=student_rank)
                    offset += student_batch_size
                dist.barrier(group=global_group)
        epoch += 1


def eval_model(opt, query_encoder, doc_encoder, tokenizer, tb_logger, step, local_rank):

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
                if tb_logger is not None:
                    tb_logger.add_scalar(
                        f"{datasetname}/{metric}", metrics[metric], step
                    )
            logger.info(" | ".join(message))

if __name__ == "__main__":
    logger.info("Start")

    options = Options()
    opt = options.parse()

    torch.manual_seed(opt.seed)

    local_rank = int(os.environ["LOCAL_RANK"])
    logger.warning(f"local rank: {local_rank}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    global_group = dist.new_group(list(range(world_size)))  # Syncs all processes
    student_group = None  # Will be assigned for student processes

    if not dist_utils.is_main():  # Only student processes
        student_group = dist.new_group(list(range(1, world_size)))  # Only rank 1+

    directory_exists = os.path.isdir(opt.output_dir)
    dist.barrier(group=global_group)  # ✅ Ensure all ranks sync

    directory_exists = os.path.isdir(opt.output_dir)
    if dist.is_initialized():
        dist.barrier(group=global_group)
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(opt.save_dir, exist_ok=True)
    if not directory_exists and dist_utils.is_main():
        options.print_options(opt)
    if dist.is_initialized():
        dist.barrier(group=global_group)
    utils.init_logger(opt)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if opt.contrastive_mode == "moco":
        model_class = moco.MoCoDistill
    elif opt.contrastive_mode == "inbatch":
        model_class = inbatch.InBatch
    else:
        raise ValueError(f"contrastive mode: {opt.contrastive_mode} not recognised")

    instruction = 'Given a web search query, retrieve relevant passages that answer the query.'
    prompt = f'<instruct>{instruction}\n<query>'

    student_model = None
    optimizer = None
    scheduler = None
    step = 0

    # First process loads the teacher model (Gemma2)
    if dist_utils.is_main():
        #teacher_model = None
        teacher_model = SentenceTransformer("all-MiniLM-L6-v2")
        ##teacher_model = SentenceTransformer("BAAI/bge-multilingual-gemma2", model_kwargs={"torch_dtype": torch.float16})
        teacher_model = teacher_model.to(local_rank)
    else: # Other processes load student model
        if not directory_exists and opt.model_path == "none":
            student_model = model_class(opt)
            if opt.weight_decay_from_init:
                student_model.init_weights_to_gpu()
            optimizer, scheduler = utils.set_optim(opt, student_model)
            step = 0
            logger.warning(f"Model loaded from rank {local_rank}")
        elif directory_exists:
            model_path = os.path.join(opt.output_dir, "checkpoint", "latest")
            student_model, optimizer, scheduler, opt_checkpoint, step = utils.load(
                model_class,
                model_path,
                opt,
                reset_params=False,
            )
            logger.info(f"Model loaded from {opt.output_dir}")
        else:
            student_model, optimizer, scheduler, opt_checkpoint, step = utils.load(
                model_class,
                opt.model_path,
                opt,
                reset_params=False if opt.continue_training else True,
            )
            if not opt.continue_training:
                step = 0
            logger.info(f"Model loaded from {opt.model_path}")

        logger.info(utils.get_parameters(student_model))

        student_model = student_model.to(local_rank)

        if dist.is_initialized():
            model = torch.nn.parallel.DistributedDataParallel(
                student_model,
                device_ids=[local_rank],
                output_device=local_rank,
                process_group=student_group,
                find_unused_parameters=False,
            )

    logger.warning(f"Before barrier: rank {local_rank}")
    if not dist_utils.is_main():
        dist.barrier(group=student_group)
    logger.warning(f"After barrier: rank {local_rank}")

    student_tokenizer = AutoTokenizer.from_pretrained("models/czert")

    train(
        opt,
        student_model if not dist_utils.is_main() else None,
        teacher_model if dist_utils.is_main() else None, student_tokenizer,
        prompt, optimizer, scheduler, step, local_rank
    )

