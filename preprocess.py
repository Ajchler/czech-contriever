# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Author: Vojtěch Eichler
# The code was modified to load jsonl files

import os
import argparse
import json
import torch
import jsonlines

import transformers
from src.normalize_text import normalize


def save(tensor, split_path):
    if not os.path.exists(os.path.dirname(split_path)):
        os.makedirs(os.path.dirname(split_path))
    with open(split_path, "wb") as fout:
        torch.save(tensor, fout)


def apply_tokenizer(path, tokenizer, normalize_text=False):
    alltokens = []
    lines = []

    with jsonlines.open(path) as fin:
        for k, line in enumerate(fin):
            line = line["text"]

            if normalize_text:
                line = normalize(line)

            lines.append(line)
            if len(lines) > 100000:
                tokens = tokenizer.batch_encode_plus(lines, add_special_tokens=False)[
                    "input_ids"
                ]
                tokens = [torch.tensor(x, dtype=torch.int) for x in tokens]
                alltokens.extend(tokens)
                del lines
                lines = []

    tokens = tokenizer.batch_encode_plus(lines, add_special_tokens=False)["input_ids"]
    tokens = [torch.tensor(x, dtype=torch.int) for x in tokens]
    alltokens.extend(tokens)

    return alltokens


def tokenize_file(args):
    filename = os.path.basename(args.datapath)
    savepath = os.path.join(args.outdir, f"{filename}.pkl")
    if os.path.exists(savepath):
        if args.overwrite:
            print(f"File {savepath} already exists, overwriting")
        else:
            print(f"File {savepath} already exists, exiting")
            return
    try:
        if args.tokenizer == "czert":
            tokenizer = transformers.BertTokenizer.from_pretrained("models/czert/")
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                args.tokenizer, local_files_only=True
            )
    except:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.tokenizer, local_files_only=False
        )
    print(f"Encoding {args.datapath}...")
    tokens = apply_tokenizer(
        args.datapath, tokenizer, normalize_text=args.normalize_text
    )

    print(f"Saving at {savepath}...")
    save(tokens, savepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--datapath", type=str)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--normalize_text", action="store_true")

    args, _ = parser.parse_known_args()
    tokenize_file(args)
