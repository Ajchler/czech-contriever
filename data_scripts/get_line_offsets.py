# Author: Vojtěch Eichler
# This script generates line offsets for a JSONL file.

import pickle
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import json
import argparse


def save_offsets(offsets, output_file):
    with open(output_file, "wb") as file:
        pickle.dump(offsets, file)


def find_jsonline_offsets(file_path, tokenizer_path):
    offsets = []
    offset = 0
    cumsum_tokens = 0
    processed_lines = 0

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    with open(file_path, "r") as file:
        for line in file:
            # Calculate the byte offset
            lline = line.encode("utf-8")
            line_json = json.loads(lline)
            offsets.append(
                {
                    "offset": offset,
                    "tokens_before_this_line": cumsum_tokens,
                }
            )
            offset += len(lline)
            cumsum_tokens += len(
                tokenizer(line_json["text"], return_tensors="pt")["input_ids"].squeeze(
                    0
                )
            )
            processed_lines += 1
            if processed_lines % 1000000 == 0:
                print(f"Processed {processed_lines} lines.")

    return offsets


def parse_args():
    parser = argparse.ArgumentParser(description='Generate line offsets for a JSONL file')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to the input JSONL file')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Path to save the offsets')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                      help='Path to the tokenizer model')
    return parser.parse_args()


def main():
    args = parse_args()
    jsonline_offsets = find_jsonline_offsets(args.input_file, args.tokenizer_path)
    save_offsets(jsonline_offsets, args.output_file)


if __name__ == "__main__":
    main()
