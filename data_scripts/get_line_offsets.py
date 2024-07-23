import pickle


def save_offsets(offsets, output_file):
    with open(output_file, "wb") as file:
        pickle.dump(offsets, file)


def find_jsonline_offsets(file_path):
    offsets = []
    offset = 0

    with open(file_path, "r") as file:
        for line in file:
            offsets.append(offset)
            offset += len(line.encode("utf-8"))  # Calculate byte length of the line

    return offsets


# Example usage:
file_path = "/mnt/data/all-in-one-mosaic/train-portion.jsonl"
jsonline_offsets = find_jsonline_offsets(file_path)
save_offsets(jsonline_offsets, "/mnt/data/all-in-one-mosaic/line-offsets.pkl")
print(jsonline_offsets)
