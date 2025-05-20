# Author: Vojtěch Eichler

import pandas as pd
import json

df = pd.read_csv("data/dareczech/dev.tsv", sep="\t", index_col=0)

docs = {}

print("Writing dev docs")

with open("BEIR/datasets/dareczech/corpus.jsonl", "w") as f:
    for doc_id, row in df.iterrows():
        docs = {}
        docs["_id"] = doc_id
        docs["title"] = str(row["title"]) if not pd.isna(row["title"]) else ""
        docs["text"] = str(row["doc"])
        f.write(json.dumps(docs, ensure_ascii=False) + "\n")

print("Finished writing dev docs")

print("Writing dev queries")

df2 = df.copy()
df2 = df2.drop_duplicates(subset="query", keep="first")

queries = {}

with open("BEIR/datasets/dareczech/queries.jsonl", "w") as f:
    for query_id, row in df2.iterrows():
        query = {}
        query["_id"] = "q" + str(query_id)
        query["text"] = str(row["query"])
        queries[query["text"]] = query["_id"]
        f.write(json.dumps(query, ensure_ascii=False) + "\n")

print("Finished writing dev queries")

print("Writing dev qrels")

with open("BEIR/datasets/dareczech/qrels/test.tsv", "w") as f:
    f.write("query_id\tdoc_id\tscore\n")
    for docid, row in df.iterrows():
        query_text = str(row["query"])
        query_id = str(queries[query_text])
        score = str(int(2 * float(row["label"])))
        f.write(f"{query_id}\t{docid}\t{score}\n")

print("Finished writing dev qrels")
