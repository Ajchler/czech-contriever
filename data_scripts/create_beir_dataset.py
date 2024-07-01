# Create validation dataset in BEIR format, this script is mostly
# inspired by a script provided by dr. Fajcik.

import json
import os
import random

import jsonlines

DATA_FOLDER = ""
output_dir = ""
per_user_dataset = {}
examples_of_annotation = {}
# I hand-annotated this (MF)
# 13 in total
conflict_resolution = [
    "negative",  # 0
    "positive",  # 1
    "positive",  # 2
    "positive",  # 3
    "positive",  # 4
    "positive",  # 5
    "negative",  # 6
    "negative",  # 7
    "negative",  # 8
    "positive",  # 9
    "negative",  # 10
    "positive",  # 11
    "positive",  # 12
]

for file in os.listdir(DATA_FOLDER):
    with jsonlines.open(os.path.join(DATA_FOLDER, file)) as reader:
        for record in reader:
            if record["user"] not in per_user_dataset:
                per_user_dataset[record["user"]] = {
                    "annotations": {
                        "query": [],
                        "bold_docs": [],
                        "labels": [],
                    },
                }
            # Ignore queries with no annotations
            annotated = len(
                [
                    1
                    for card in record["data"]["cards"]
                    if card["timestamp"] and card["state"] != "neutral"
                ]
            )
            if annotated == 0:
                continue
            if "query_string" not in record:  # only select asymmetric queries
                continue
            # add annotation example
            per_user_dataset[record["user"]]["annotations"]["query"].append(
                record["query_string"]
            )
            docs = []
            labels = []
            for card in record["data"]["cards"]:
                if card["timestamp"] and card["state"] != "neutral":
                    text = card["segments_cz"]
                    label = card["state"]
                    docs.append(text)
                    labels.append(label)

            per_user_dataset[record["user"]]["annotations"]["bold_docs"].append(docs)
            per_user_dataset[record["user"]]["annotations"]["labels"].append(labels)

# merge annotations across users. If some annotatons have same query, merge assigned docs and labels
# def deduplicate_list, while keeping order

clean_dataset = {}
conflict_id = 0
for user, data in per_user_dataset.items():
    for i, query in enumerate(data["annotations"]["query"]):
        if query not in clean_dataset:
            clean_dataset[query] = {
                "docs": [],
                "labels": [],
                "authors": [],
            }

        # if doc not already in docs, add it and its label
        for doc, label in zip(
            data["annotations"]["bold_docs"][i], data["annotations"]["labels"][i]
        ):
            if doc not in clean_dataset[query]["docs"]:
                clean_dataset[query]["docs"].append(doc)
                clean_dataset[query]["labels"].append(label)
                clean_dataset[query]["authors"].append(user)
            else:
                # if doc already in docs, make sure label matches
                idx = clean_dataset[query]["docs"].index(doc)
                if not clean_dataset[query]["labels"][idx] == label:
                    print("conflict", conflict_id, "\n")
                    print(
                        "Warning: label mismatch for doc:\n",
                        doc,
                        "\nin query:",
                        query,
                        "\n first annotation by",
                        clean_dataset[query]["authors"][idx],
                        "second annotation by",
                        user,
                    )
                    print(
                        "\nExisting label:",
                        clean_dataset[query]["labels"][idx],
                        "New label:",
                        label,
                    )
                    print("-------------------")

                    # resolve conflict
                    clean_dataset[query]["labels"][idx] = conflict_resolution[
                        conflict_id
                    ]
                    conflict_id += 1


corpus = {}
queries = {}
qrels = []
seen_docs = []
query_counter = 0
docs_counter = 0
# Create BEIR dataset from clean dataset
for query_string, data in clean_dataset.items():
    docs = data["docs"]
    labels = data["labels"]

    query_id = f"q{query_counter}"
    queries[query_id] = {"text": query_string}
    query_counter += 1

    # docs
    for doc, label in zip(docs, labels):
        if doc not in seen_docs:
            seen_docs.append(doc)
            doc_id = f"d{docs_counter}"
        else:
            doc_id = f"d{seen_docs.index(doc)}"
        docs_counter += 1
        corpus[doc_id] = {"title": "", "text": doc}

        qrels.append((query_id, doc_id, 2 if label == "positive" else 0))

os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "corpus.jsonl"), "w") as f:
    for doc_id, doc_data in corpus.items():
        f.write(
            json.dumps(
                {
                    "_id": doc_id,
                    "title": doc_data["title"],
                    "text": doc_data["text"],
                },
                ensure_ascii=True,
            )
            + "\n"
        )

with open(os.path.join(output_dir, "queries.jsonl"), "w") as f:
    for query_id, query_data in queries.items():
        f.write(
            json.dumps({"_id": query_id, "text": query_data["text"]}, ensure_ascii=True)
            + "\n"
        )

with open(os.path.join(output_dir, "test.tsv"), "w") as f:
    f.write("query_id\tdoc_id\tscore\n")
    for query_id, doc_id, score in qrels:
        f.write(f"{query_id}\t{doc_id}\t{score}\n")
