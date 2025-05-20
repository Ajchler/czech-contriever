# Author: Vojtěch Eichler

from transformers import AutoTokenizer
import pickle
import struct
import json
from datetime import datetime
import argparse

class TokenizerSaver:
    def __init__(self, tokenizer_name, dataset_path, token_output_path, offsets_output_path, chunk_size=1000):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset_path = dataset_path
        self.token_output_path = token_output_path
        self.offsets_output_path = offsets_output_path
        self.chunk_size = chunk_size
        self.offsets = []
        self.current_offset = 0
        self.processed_lines = 0

    def save_tokens(self):
        with open(self.token_output_path, 'wb') as token_file:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                buffer = []

                for line in f:
                    line_json = json.loads(line)
                    tokens = self.tokenizer(line_json["text"], return_tensors="pt")["input_ids"].squeeze(0)
                    buffer.extend(tokens)

                    while len(buffer) >= self.chunk_size:
                        chunk = buffer[:self.chunk_size]
                        buffer = buffer[self.chunk_size:]

                        for token in chunk:
                            token_file.write(struct.pack('<H', token))  # Write as 2-byte unsigned short

                        self.offsets.append(self.current_offset)
                        self.current_offset += len(chunk) * 2

                    self.processed_lines += 1

                    if self.processed_lines % 100000 == 0:
                        curr_time = datetime.now()
                        print(f"Processed {self.processed_lines} lines at {curr_time}")

                if buffer:
                    for token in buffer:
                        token_file.write(struct.pack('<H', token))
                    self.offsets.append(self.current_offset)

        with open(self.offsets_output_path, 'wb') as offsets_file:
            pickle.dump(self.offsets, offsets_file)

        print(f"Tokenization complete. Tokens saved to {self.token_output_path}. Offsets saved to {self.offsets_output_path}.")

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess text data using a tokenizer')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                      help='Path to the tokenizer model')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to the input dataset file')
    parser.add_argument('--token_output_path', type=str, required=True,
                      help='Path to save the tokenized output')
    parser.add_argument('--offsets_output_path', type=str, required=True,
                      help='Path to save the offsets')
    parser.add_argument('--chunk_size', type=int, default=1000,
                      help='Size of chunks for tokenization')
    return parser.parse_args()

def main():
    args = parse_args()
    saver = TokenizerSaver(
        args.tokenizer_path,
        args.dataset_path,
        args.token_output_path,
        args.offsets_output_path,
        args.chunk_size
    )
    saver.save_tokens()

if __name__ == "__main__":
    main()
