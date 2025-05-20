# Author: Vojtěch Eichler

import random
import json

INPUT_FILE = "all-in-one.mosaic.jsonl"
TRAIN_FILE = "train-portion.jsonl"
VALID_FILE = "valid-portion.jsonl"

validation_samples_n = 32768


with open(INPUT_FILE, "r") as f, open(TRAIN_FILE, "w") as train_f, open(
    VALID_FILE, "w"
) as valid_f:
    valid_indices = set(random.sample(range(0, 177089377), validation_samples_n))
    for i, line in enumerate(f):
        line_dict = json.loads(line)
        line_dict = {"text": line_dict["text"]}
        line = json.dumps(line_dict, ensure_ascii=False) + "\n"
        if i in valid_indices:
            valid_f.write(line)
        else:
            train_f.write(line)
