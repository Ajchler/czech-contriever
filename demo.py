from src.contriever import load_retriever
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_path", type=str, default="models/czert")
args = argparser.parse_args()

if __name__ == "__main__":
    retriever, tokenizer, _ = load_retriever("models/czert")

    while True:
        input_text = input("Enter a text: ")
        tokens = tokenizer(input_text, return_tensors="pt", padding=True)
        for start in range(0, tokens["input_ids"].size(1), 128):
            output = retriever(
                input_ids=tokens["input_ids"][:, start : start + 128],
                attention_mask=tokens["attention_mask"][:, start : start + 128],
            )
            # get text chunk from the token ids chunk
            print(f"Embedding: {output}")
            print(f"For this part of the text: {tokenizer.decode(tokens["input_ids"][0, start : start + 128])}")
