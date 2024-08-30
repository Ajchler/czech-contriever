import pickle
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import json


def save_offsets(offsets, output_file):
    with open(output_file, "wb") as file:
        pickle.dump(offsets, file)


def find_jsonline_offsets(file_path):
    offsets = []
    offset = 0
    cumsum_tokens = 0
    processed_lines = 0

    # with open(file_path, "r") as file:
    #     for line in file:
    #         offsets.append(offset)
    #         offset += len(line.encode("utf-8"))  # Calculate byte length of the line

    # Also save length of the line in tokens
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/veichler/repos/czech-contriever/models/czert/"
    )

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


# Example usage:
file_path = "/mnt/data/all-in-one-mosaic/train-portion.jsonl"
jsonline_offsets = find_jsonline_offsets(file_path)
save_offsets(jsonline_offsets, "bar.pkl")
