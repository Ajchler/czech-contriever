# Author: Vojtěch Eichler
# This script creates a token length histogram from a dataset.

import json
from transformers import AutoTokenizer
from collections import Counter
import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Create token length histogram from a dataset')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to the input JSONL file')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Path to save the histogram CSV')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                      help='Path to the tokenizer model')
    parser.add_argument('--batch_size', type=int, default=1000,
                      help='Batch size for processing')
    parser.add_argument('--progress_interval', type=int, default=10000000,
                      help='Interval for saving progress')
    return parser.parse_args()

# Function to read the json-lines file in batches
def read_jsonl_in_batches(file_path, batch_size=100):
    batch = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            batch.append(json.loads(line)['text'])
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

# Tokenize the texts in batches and count token lengths, writing directly to file
def tokenize_and_count_lengths(file_path, tokenizer, batch_size=1000, output_file="token_length_histogram.csv", progress_interval=10000000):
    length_counter = Counter()  # Track token lengths across all batches
    total_lines_processed = 0  # Track the total number of lines processed

    # Process the file in smaller batches
    for i, batch in enumerate(read_jsonl_in_batches(file_path, batch_size)):
        tokenized_batch = tokenizer.batch_encode_plus(batch, padding=False, truncation=False, return_length=True)
        length_counter.update(tokenized_batch['length'])  # Update counter with batch token lengths
        total_lines_processed += len(batch)

        # Save progress after every progress_interval lines
        if total_lines_processed % progress_interval == 0:
            progress_file = f"progress_{total_lines_processed//1000000}M_lines.txt"
            with open(progress_file, "w") as f:
                f.write(f"Processed {total_lines_processed} lines\n")
            print(f"Progress: {total_lines_processed} lines processed. Saved to {progress_file}.")

    # Convert counter to DataFrame and save to CSV
    histogram_df = pd.DataFrame(list(length_counter.items()), columns=['Token Length', 'Count'])
    histogram_df.to_csv(output_file, index=False)
    print(f"Histogram data saved to {output_file}")

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenize_and_count_lengths(
        args.input_file,
        tokenizer,
        batch_size=args.batch_size,
        output_file=args.output_file,
        progress_interval=args.progress_interval
    )

if __name__ == "__main__":
    main()
