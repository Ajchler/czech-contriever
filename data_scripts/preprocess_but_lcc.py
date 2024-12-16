from transformers import AutoTokenizer
import pickle
import struct
import json
from datetime import datetime

class TokenizerSaver:
    def __init__(self, tokenizer_name, dataset_path, token_output_path, offsets_output_path, chunk_size=1000):
        self.tokenizer = AutoTokenizer.from_pretrained("/storage/brno12-cerit/home/veichler/czech-contriever/models/czert/")
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

tokenizer_name = ""
dataset_path = '/storage/brno12-cerit/home/veichler/train-portion.jsonl'
token_output_path = '/storage/brno12-cerit/home/veichler/foo.bin'
offsets_output_path = '/storage/brno12-cerit/home/veichler/bar.pkl'

saver = TokenizerSaver(tokenizer_name, dataset_path, token_output_path, offsets_output_path)
saver.save_tokens()

