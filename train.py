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
from safe_gpu import safe_gpu
from collections import defaultdict

import torch.distributed as dist
import torch.utils
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from src.options import Options
from src import data, beir_utils, slurm, dist_utils, utils
from src import moco, inbatch
from src.data import build_mask

project_name = os.getenv("PROJECT_NAME", "czechtriever")
task_name = os.getenv("TASK_NAME", "czechtriever-default")
Task.init(project_name=project_name, task_name=task_name)

logger = logging.getLogger(__name__)


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


def train(opt, model, optimizer, scheduler, step):

    run_stats = utils.WeightedAvgStats()

    tb_logger = utils.init_tb_logger(opt.output_dir)

    logger.info("Data loading")
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        tokenizer = model.module.tokenizer
    else:
        tokenizer = model.tokenizer
    collator = data.Collator(opt=opt)
    train_dataset, val_dataset = data.load_data(opt, tokenizer)
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
    )
    logger.warning(f"Data loading finished for rank {dist_utils.get_rank()}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.per_gpu_batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=opt.num_workers,
        collate_fn=collator,
    )

    if dist_utils.is_main():
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=opt.per_gpu_eval_batch_size,
            num_workers=opt.num_workers,
            collate_fn=collator,
        )

    epoch = 1

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        encoder = model.module.get_encoder()
    else:
        encoder = model.get_encoder()
    eval_model(
        opt,
        query_encoder=encoder,
        doc_encoder=encoder,
        tokenizer=tokenizer,
        tb_logger=tb_logger,
        step=step,
    )

    if dist_utils.is_main():
        eval_loss(
            opt,
            model,
            tb_logger,
            step,
            val_dataloader,
            val_dataset.get_passage_from_all_docs(),
            scheduler,
        )

    model.train()

    while step < opt.total_steps:
        logger.info(f"Start epoch {epoch}")
        if dist.is_initialized():
            sampler.set_epoch(epoch)

        for i, (batch, _) in enumerate(train_dataloader):
            step += 1

            batch = {
                key: value.cuda() if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            train_loss, iter_stats = model(**batch, stats_prefix="train")
            train_loss.backward()

            run_stats.update(iter_stats)
            if step % opt.log_freq == 0:
                log = f"{step} / {opt.total_steps}"
                for k, v in sorted(run_stats.average_stats.items()):
                    log += f" | {k}: {v:.3f}"
                    if tb_logger:
                        tb_logger.add_scalar(k, v, step)
                log += f" | lr: {scheduler.get_last_lr()[0]:0.3g}"
                log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"

                lr = scheduler.get_last_lr()[0]
                if tb_logger:
                    tb_logger.add_scalar("train/lr", lr, step)

                global_grad_norm = 0

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        norm = param.grad.norm().item()
                        global_grad_norm += norm**2
                        if tb_logger:
                            tb_logger.add_scalar(f"grad/{name}", norm, step)

                global_grad_norm = global_grad_norm**0.5

                if tb_logger:
                    tb_logger.add_scalar("train/global_grad", global_grad_norm, step)

                logger.info(log)
                run_stats.reset()

            if opt.clip_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if step % opt.eval_freq == 0:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    encoder = model.module.get_encoder()
                else:
                    encoder = model.get_encoder()
                eval_model(
                    opt,
                    query_encoder=encoder,
                    doc_encoder=encoder,
                    tokenizer=tokenizer,
                    tb_logger=tb_logger,
                    step=step,
                )

                if dist_utils.is_main():
                    eval_loss(
                        opt,
                        model,
                        tb_logger,
                        step,
                        val_dataloader,
                        val_dataset.get_passage_from_all_docs(),
                        scheduler,
                    )

                if dist_utils.is_main():
                    utils.save(
                        model,
                        optimizer,
                        scheduler,
                        step,
                        opt,
                        opt.output_dir,
                        f"lastlog",
                    )

                model.train()

            if dist_utils.is_main() and step % opt.save_freq == 0:
                utils.save(
                    model,
                    optimizer,
                    scheduler,
                    step,
                    opt,
                    opt.output_dir,
                    f"step-{step}",
                )

            if step > opt.total_steps:
                break
        epoch += 1


def eval_model(opt, query_encoder, doc_encoder, tokenizer, tb_logger, step):

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

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if opt.sge:
        safe_gpu.claim_gpus()

    directory_exists = os.path.isdir(opt.output_dir)
    if dist.is_initialized():
        dist.barrier()
    os.makedirs(opt.output_dir, exist_ok=True)
    if not directory_exists and dist_utils.is_main():
        options.print_options(opt)
    if dist.is_initialized():
        dist.barrier()
    utils.init_logger(opt)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if opt.contrastive_mode == "moco":
        model_class = moco.MoCo
    elif opt.contrastive_mode == "inbatch":
        model_class = inbatch.InBatch
    else:
        raise ValueError(f"contrastive mode: {opt.contrastive_mode} not recognised")

    if not directory_exists and opt.model_path == "none":
        model = model_class(opt)
        if opt.weight_decay_from_init:
            model.init_weights_to_gpu()
        optimizer, scheduler = utils.set_optim(opt, model)
        step = 0
    elif directory_exists:
        model_path = os.path.join(opt.output_dir, "checkpoint", "latest")
        model, optimizer, scheduler, opt_checkpoint, step = utils.load(
            model_class,
            model_path,
            opt,
            reset_params=False,
        )
        logger.info(f"Model loaded from {opt.output_dir}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step = utils.load(
            model_class,
            opt.model_path,
            opt,
            reset_params=False if opt.continue_training else True,
        )
        if not opt.continue_training:
            step = 0
        logger.info(f"Model loaded from {opt.model_path}")

    logger.info(utils.get_parameters(model))

    model = model.to(local_rank)

    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        dist.barrier()

    logger.info("Start training")
    train(opt, model, optimizer, scheduler, step)
