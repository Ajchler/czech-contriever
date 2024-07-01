import json
import numpy as np

docs = []

with open("periodicals.lm-ready.cs-only.new.jsonl", "r") as f:
    data = f.readlines()
    for d in data:
        d = json.loads(d)
        docs.append(d["text"])

perm = np.random.permutation(len(docs))

rand_docs = [docs[i] for i in perm]

train = perm[: int(0.975 * len(docs))]
valid = perm[int(0.975 * len(docs)) :]

with open("train.jsonl", "w") as f:
    for i in train:
        f.write(json.dumps({"text": docs[i]}, ensure_ascii=False) + "\n")

with open("valid.jsonl", "w") as f:
    for i in valid:
        f.write(json.dumps({"text": docs[i]}, ensure_ascii=False) + "\n")
