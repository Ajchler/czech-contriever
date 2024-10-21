import torch
import time
from transformers import AutoTokenizer, AutoModel

import src.simcse
import src.contriever


def evaluate_throughput(
    model, max_seq_length=128, initial_batch_size=1, device="cuda", vocab_size=30522
):
    model.to(device)
    model.eval()

    batch_size = initial_batch_size
    successful_batch_size = batch_size

    # Find maximum batch size
    while True:
        try:
            # Create random inputs (assuming tokenized inputs for embeddings)
            inputs = torch.randint(
                0, vocab_size, (batch_size, max_seq_length), dtype=torch.long
            ).to(
                device
            )  # 30522 for typical vocab size (adjust if needed)

            # Warm-up run to avoid initial overhead
            _ = model(inputs)

            # If successful, increase the batch size
            successful_batch_size = batch_size
            batch_size *= 2  # Double the batch size each time
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Out of memory at batch size: {batch_size}")
                break
            else:
                raise e

    # Use maximum batch size to evaluate throughput
    print(f"Max batch size that fits in memory: {successful_batch_size}")

    # Measure throughput
    inputs = torch.randint(
        0, vocab_size, (successful_batch_size, max_seq_length), dtype=torch.long
    ).to(device)
    torch.cuda.synchronize()
    start_time = time.time()

    # Number of iterations for measuring throughput
    num_iterations = 100

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(inputs)

    torch.cuda.synchronize()
    total_time = time.time() - start_time

    throughput = (num_iterations * successful_batch_size) / total_time
    print(f"Throughput: {throughput:.2f} samples/second")


# Example usage:
# Assume `model` is your embedding model
model, tokenizer, _ = src.simcse.load_simcse()
# model = AutoModel.from_pretrained("facebook/contriever")
# tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
# model = AutoModel.from_pretrained("facebook/mcontriever")
# tokenizer = AutoTokenizer.from_pretrained("facebook/mcontriever")
# model = AutoModel.from_pretrained("Ajchler/czechtriever-demo")
# tokenizer = AutoTokenizer.from_pretrained("Ajchler/czechtriever-demo")

evaluate_throughput(model, vocab_size=tokenizer.vocab_size)
