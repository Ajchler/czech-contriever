# Author: Vojtěch Eichler
# This script parses the DAREczech dataset into BEIR format.
import pandas as pd
import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Parse DAREczech dataset into BEIR format')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to the input TSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the BEIR dataset')
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input_file, sep="\t", index_col=0)

    docs = {}
    os.makedirs(args.output_dir, exist_ok=True)

    print("Writing dev docs")
    with open(os.path.join(args.output_dir, "corpus.jsonl"), "w") as f:
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

    with open(os.path.join(args.output_dir, "queries.jsonl"), "w") as f:
        for query_id, row in df2.iterrows():
            query = {}
            query["_id"] = "q" + str(query_id)
            query["text"] = str(row["query"])
            queries[query["text"]] = query["_id"]
            f.write(json.dumps(query, ensure_ascii=False) + "\n")

    print("Finished writing dev queries")

    print("Writing dev qrels")
    os.makedirs(os.path.join(args.output_dir, "qrels"), exist_ok=True)
    with open(os.path.join(args.output_dir, "qrels", "test.tsv"), "w") as f:
        f.write("query_id\tdoc_id\tscore\n")
        for docid, row in df.iterrows():
            query_text = str(row["query"])
            query_id = str(queries[query_text])
            score = str(int(2 * float(row["label"])))
            f.write(f"{query_id}\t{docid}\t{score}\n")

    print("Finished writing dev qrels")

if __name__ == "__main__":
    main()
