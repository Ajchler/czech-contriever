import torch
import torch.nn as nn
import numpy as np
import math
import random
import transformers
import logging
import torch.distributed as dist

from src import contriever, dist_utils, utils
from src.data import load_distill_data

logger = logging.getLogger(__name__)


class Distiller(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(Distiller, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing
        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
            )
        self.tokenizer = tokenizer
        self.encoder = retriever

    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if ("bert-" in model_id) or ("czert" in model_id):
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder

    def forward(
        self, input_ids, attention_mask, stats_prefix="", iter_stats={}, **kwargs
    ):

        bsz = len(input_ids)
        labels = torch.arange(0, bsz, dtype=torch.long, device=input_ids.device)

        qemb = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            normalize=self.norm_query,
        )

        # gather_fn = dist_utils.gather
        # loss = 0

        # log stats
        # if len(stats_prefix) > 0:
        #    stats_prefix = stats_prefix + "/"
        # iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        # stdq = torch.std(qemb, dim=0).mean().item()
        # iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)

        return qemb, iter_stats
