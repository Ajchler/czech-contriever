import random
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Split a dataset into train and validation sets')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to the input JSONL file')
    parser.add_argument('--train_file', type=str, required=True,
                      help='Path to save the training set')
    parser.add_argument('--valid_file', type=str, required=True,
                      help='Path to save the validation set')
    parser.add_argument('--validation_samples', type=int, default=32768,
                      help='Number of samples to use for validation')
    parser.add_argument('--total_samples', type=int, required=True,
                      help='Total number of samples in the input file', default=177089377)
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    with open(args.input_file, "r") as f, open(args.train_file, "w") as train_f, open(
        args.valid_file, "w"
    ) as valid_f:
        valid_indices = set(random.sample(range(0, args.total_samples), args.validation_samples))
        for i, line in enumerate(f):
            line_dict = json.loads(line)
            line_dict = {"text": line_dict["text"]}
            line = json.dumps(line_dict, ensure_ascii=False) + "\n"
            if i in valid_indices:
                valid_f.write(line)
            else:
                train_f.write(line)

if __name__ == "__main__":
    main()
