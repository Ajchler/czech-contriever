# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import glob
import torch
import random
import json
import csv
import numpy as np
import numpy.random
import logging
from collections import defaultdict
import torch.distributed as dist
from datasets import load_dataset

from src import dist_utils
from src.normalize_text import normalize

logger = logging.getLogger(__name__)


def tokenize_jsonl_file(file_path, tokenizer, opt):
    train_docs = []
    if file_path is not None:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = []
            for i, line in enumerate(f):
                line = json.loads(line)
                line = line["text"]
                if opt.normalize_text:
                    line = normalize(line)

                lines.append(line)
                if len(lines) > 100000:
                    tokens = tokenizer.batch_encode_plus(
                        lines,
                        add_special_tokens=False,
                    )["input_ids"]
                    tokens = [torch.tensor(x, dtype=torch.int) for x in tokens]
                    train_docs.extend(tokens)
                    lines = []

        tokens = tokenizer.batch_encode_plus(
            lines,
            add_special_tokens=False,
        )["input_ids"]
        tokens = [torch.tensor(x, dtype=torch.int) for x in tokens]
        train_docs.extend(tokens)
    return train_docs


def load_and_tokenize_datasets(opt, tokenizer):
    datasets = {}
    val_datasets = {}

    for path in opt.train_data:
        if path is not None:
            train_docs = tokenize_jsonl_file(path, tokenizer, opt)
            datasets[path] = Dataset(train_docs, opt.chunk_length, tokenizer, opt)

    for path in opt.valid_data:
        if path is not None:
            val_docs = tokenize_jsonl_file(path, tokenizer, opt)
            val_datasets[path] = Dataset(val_docs, opt.chunk_length, tokenizer, opt)

    dataset = MultiDataset(datasets)
    val_dataset = MultiDataset(val_datasets)
    dataset.set_prob(coeff=opt.sampling_coefficient)
    val_dataset.set_prob(coeff=opt.sampling_coefficient)
    return dataset, val_dataset


def load_data(opt, tokenizer):
    if opt.data_preprocessed:
        datasets = {}
        for path in opt.train_data:
            data = load_dataset_custom(path, opt.loading_mode)
            if data is not None:
                datasets[path] = Dataset(data, opt.chunk_length, tokenizer, opt)
        train_dataset = MultiDataset(datasets)
        train_dataset.set_prob(coeff=opt.sampling_coefficient)

        valid_datasets = {}
        for path in opt.valid_data:
            data = load_dataset_custom(path, opt.loading_mode)
            if data is not None:
                valid_datasets[path] = Dataset(data, opt.chunk_length, tokenizer, opt)
        val_dataset = MultiDataset(valid_datasets)
        val_dataset.set_prob(coeff=opt.sampling_coefficient)
    else:
        # train_dataset, val_dataset = load_and_tokenize_datasets(opt, tokenizer)
        # train_dataset, val_dataset = load_streaming_datasets(opt, tokenizer)
        train_dataset = LazyDataset(opt.train_data[0], tokenizer, opt)

        val_docs = tokenize_jsonl_file(opt.valid_data[0], tokenizer, opt)

        val_dataset = Dataset(val_docs, opt.chunk_length, tokenizer, opt)

    return train_dataset, val_dataset


def load_dataset_custom(data_path, loading_mode):
    files = glob.glob(os.path.join(data_path, "*.p*"))
    files.sort()
    tensors = []
    if loading_mode == "split":
        files_split = list(np.array_split(files, dist_utils.get_world_size()))[
            dist_utils.get_rank()
        ]
        for filepath in files_split:
            try:
                tensors.extend(torch.load(filepath, map_location="cpu"))
            except:
                logger.warning(f"Unable to load file {filepath}")
    elif loading_mode == "full":
        for fin in files:
            tensors.extend(torch.load(fin, map_location="cpu"))
    elif loading_mode == "single":
        tensors.extend(torch.load(files[0], map_location="cpu"))
    if len(tensors) == 0:
        return None
    return tensors


class LazyDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, tokenizer, opt, buffer_size=300000):
        self.path = path
        self.tokenizer = tokenizer
        self.opt = opt
        self.buffer_size = buffer_size
        self.chunk_length = opt.chunk_length

    def _create_pair(self, text):
        tokens = self.tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)
        start_idx = random.randint(0, max(0, tokens.size(0) - self.chunk_length))
        end_idx = start_idx + self.chunk_length
        tokens = tokens[start_idx:end_idx]
        q_tokens = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
        k_tokens = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
        q_tokens = apply_augmentation(q_tokens, self.opt)
        q_tokens = add_bos_eos(
            q_tokens, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        )
        k_tokens = apply_augmentation(k_tokens, self.opt)
        k_tokens = add_bos_eos(
            k_tokens, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        )

        return {"q_tokens": q_tokens, "k_tokens": k_tokens}

    def __iter__(self):
        buffer = []
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                text = json.loads(line)["text"]
                if len(buffer) < self.buffer_size:
                    buffer.append(text)
                else:
                    idx = random.randint(0, len(buffer) - 1)
                    yield (self._create_pair(buffer[idx]), idx)
                    buffer[idx] = text


class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):

        self.datasets = datasets
        self.prob = [1 / len(self.datasets) for _ in self.datasets]
        self.dataset_ids = list(self.datasets.keys())

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets.values()])

    def __getitem__(self, index):
        dataset_idx = numpy.random.choice(range(len(self.prob)), 1, p=self.prob)[0]
        did = self.dataset_ids[dataset_idx]
        index = random.randint(0, len(self.datasets[did]) - 1)
        sample, _ = self.datasets[did][index]
        sample["dataset_id"] = did
        return sample, index

    def generate_offset(self):
        for dataset in self.datasets.values():
            dataset.generate_offset()

    def set_prob(self, coeff=0.0):

        prob = np.array([float(len(dataset)) for _, dataset in self.datasets.items()])
        prob /= prob.sum()
        prob = np.array([p**coeff for p in prob])
        prob /= prob.sum()
        self.prob = prob

    def all_docs(self):
        all_docs = []
        for dataset in self.datasets.values():
            all_docs.extend(dataset.docs)
        return all_docs

    def get_passage_from_all_docs(self):
        all_docs = []
        for dataset in self.datasets.values():
            all_docs.extend(dataset.get_passage_from_all_docs())
        return all_docs


class Dataset(torch.utils.data.Dataset):
    """Monolingual dataset based on a list of paths"""

    def __init__(self, docs, chunk_length, tokenizer, opt):

        self.chunk_length = chunk_length
        self.tokenizer = tokenizer
        self.opt = opt
        self.docs = docs

    def __len__(self):
        # return (self.data.size(0) - self.offset) // self.chunk_length
        return len(self.docs)

    def __getitem__(self, index):

        start_idx = random.randint(
            0, max(0, self.docs[index].size(0) - self.chunk_length)
        )
        end_idx = start_idx + self.chunk_length
        tokens = self.docs[index][start_idx:end_idx]
        q_tokens = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
        k_tokens = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
        q_tokens = apply_augmentation(q_tokens, self.opt)
        q_tokens = add_bos_eos(
            q_tokens, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        )
        k_tokens = apply_augmentation(k_tokens, self.opt)
        k_tokens = add_bos_eos(
            k_tokens, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        )

        return {"q_tokens": q_tokens, "k_tokens": k_tokens}, index

    def get_passage_from_all_docs(self):
        docs = []
        for doc in self.docs:
            start_idx = random.randint(0, max(0, doc.size(0) - self.chunk_length))
            end_idx = start_idx + self.chunk_length
            tokens = doc[start_idx:end_idx]
            q_tokens = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
            q_tokens = apply_augmentation(q_tokens, self.opt)
            q_tokens = add_bos_eos(
                q_tokens, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
            )
            docs.append(q_tokens)

        return docs

    def generate_offset(self):
        # TODO: ASI SMAZAT?
        self.offset = random.randint(0, self.chunk_length - 1)


class Collator(object):
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, batch_and_indices):
        batch_examples, indices = zip(*batch_and_indices)
        batch = defaultdict(list)
        for example in batch_examples:
            for k, v in example.items():
                batch[k].append(v)

        q_tokens, q_mask = build_mask(batch["q_tokens"])
        k_tokens, k_mask = build_mask(batch["k_tokens"])

        batch["q_tokens"] = q_tokens
        batch["q_mask"] = q_mask
        batch["k_tokens"] = k_tokens
        batch["k_mask"] = k_mask

        return batch, indices


def randomcrop(x, ratio_min, ratio_max):

    ratio = random.uniform(ratio_min, ratio_max)
    length = int(len(x) * ratio)
    start = random.randint(0, len(x) - length)
    end = start + length
    crop = x[start:end].clone()
    return crop


def build_mask(tensors):
    shapes = [x.shape for x in tensors]
    maxlength = max([len(x) for x in tensors])
    returnmasks = []
    ids = []
    for k, x in enumerate(tensors):
        returnmasks.append(torch.tensor([1] * len(x) + [0] * (maxlength - len(x))))
        ids.append(torch.cat((x, torch.tensor([0] * (maxlength - len(x))))))
    ids = torch.stack(ids, dim=0).long()
    returnmasks = torch.stack(returnmasks, dim=0).bool()
    return ids, returnmasks


def add_token(x, token):
    x = torch.cat((torch.tensor([token]), x))
    return x


def deleteword(x, p=0.1):
    mask = np.random.rand(len(x))
    x = [e for e, m in zip(x, mask) if m > p]
    return x


def replaceword(x, min_random, max_random, p=0.1):
    mask = np.random.rand(len(x))
    x = [
        e if m > p else random.randint(min_random, max_random) for e, m in zip(x, mask)
    ]
    return x


def maskword(x, mask_id, p=0.1):
    mask = np.random.rand(len(x))
    x = [e if m > p else mask_id for e, m in zip(x, mask)]
    return x


def shuffleword(x, p=0.1):
    count = (np.random.rand(len(x)) < p).sum()
    """Shuffles any n number of values in a list"""
    indices_to_shuffle = random.sample(range(len(x)), k=count)
    to_shuffle = [x[i] for i in indices_to_shuffle]
    random.shuffle(to_shuffle)
    for index, value in enumerate(to_shuffle):
        old_index = indices_to_shuffle[index]
        x[old_index] = value
    return x


def apply_augmentation(x, opt):
    if opt.augmentation == "mask":
        return torch.tensor(maskword(x, mask_id=opt.mask_id, p=opt.prob_augmentation))
    elif opt.augmentation == "replace":
        return torch.tensor(
            replaceword(
                x,
                min_random=opt.start_id,
                max_random=opt.vocab_size - 1,
                p=opt.prob_augmentation,
            )
        )
    elif opt.augmentation == "delete":
        return torch.tensor(deleteword(x, p=opt.prob_augmentation))
    elif opt.augmentation == "shuffle":
        return torch.tensor(shuffleword(x, p=opt.prob_augmentation))
    else:
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        return x


def add_bos_eos(x, bos_token_id, eos_token_id):
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    if bos_token_id is None and eos_token_id is not None:
        x = torch.cat([x.clone().detach(), torch.tensor([eos_token_id])])
    elif bos_token_id is not None and eos_token_id is None:
        x = torch.cat([torch.tensor([bos_token_id]), x.clone().detach()])
    elif bos_token_id is None and eos_token_id is None:
        pass
    else:
        x = torch.cat(
            [
                torch.tensor([bos_token_id]),
                x.clone().detach(),
                torch.tensor([eos_token_id]),
            ]
        )
    return x


# Used for passage retrieval
def load_passages(path):
    if not os.path.exists(path):
        logger.info(f"{path} does not exist")
        return
    logger.info(f"Loading passages from: {path}")
    passages = []
    with open(path) as fin:
        if path.endswith(".jsonl"):
            for k, line in enumerate(fin):
                ex = json.loads(line)
                passages.append(ex)
        else:
            reader = csv.reader(fin, delimiter="\t")
            for k, row in enumerate(reader):
                if not row[0] == "id":
                    ex = {"id": row[0], "title": row[2], "text": row[1]}
                    passages.append(ex)
    return passages
