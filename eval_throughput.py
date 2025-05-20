import torch
import time
from transformers import AutoTokenizer, AutoModel, MarianMTModel

import src.simcse
import src.contriever


def translate_en_cs(cs_sentences, max_seq_len, translate_model, translate_tokenizer, model_tokenizer):
    # Tokenize Czech text with translator's tokenizer
    inputs = {k: v.to("cuda") for k, v in translate_tokenizer(cs_sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len).items()}

    # Translate to English
    translated_tokens = translate_model.generate(**inputs, num_beams=4, early_stopping=True)

    # Decode to English text
    translated_texts = translate_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

    # Tokenize English text with model's tokenizer
    model_inputs = model_tokenizer(translated_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
    input_ids = model_inputs["input_ids"].to("cuda")
    attention_mask = (input_ids != 0).long().to("cuda")
    return input_ids, attention_mask


def get_random_batch(batch_size, vocab_size, min_seq_len=16, max_seq_len=128):
    # Generate random Czech text sequences
    cs_texts = []
    for _ in range(batch_size):
        # Generate random sequence length
        seq_len = torch.randint(min_seq_len, max_seq_len + 1, (1,)).item()
        # Generate random tokens
        tokens = torch.randint(0, vocab_size, (seq_len,))
        # Decode to text
        text = tokenizer.decode(tokens)
        cs_texts.append(text)
    return cs_texts


def process_batch(inputs, max_seq_length, translate_model, translate_tokenizer, model_tokenizer, translate=False):
    if translate:
        # Full pipeline: Czech text -> translate -> model tokenize
        return translate_en_cs(inputs, max_seq_length, translate_model, translate_tokenizer, model_tokenizer)
    else:
        # Direct model processing (no translation needed)
        # Tokenize with model's tokenizer
        model_inputs = model_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
        input_ids = model_inputs["input_ids"].to("cuda")
        attention_mask = (input_ids != 0).long().to("cuda")
        return input_ids, attention_mask


def evaluate_throughput(
    model, max_seq_length=128, min_seq_length=16, initial_batch_size=1, device="cuda", vocab_size=30522, tokenizer=None, translate=False
):
    model.to(device)
    model.eval()

    # Only load translation model if needed
    translate_model = None
    translate_tokenizer = None
    if translate:
        print("Using translation pipeline (Czech -> English)")
        translate_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-cs")
        translate_model.to("cuda")
        translate_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-cs")
    else:
        print("Using direct model processing (no translation)")

    def measure_throughput(batch_size):
        print(f"\nMeasuring throughput with batch size: {batch_size}")
        torch.cuda.empty_cache()

        # Test token counting with a single batch first
        test_inputs = get_random_batch(batch_size, vocab_size, min_seq_length, max_seq_length)
        test_processed_ids, test_attention_mask = process_batch(test_inputs, max_seq_length, translate_model, translate_tokenizer, tokenizer, translate)
        test_tokens = test_attention_mask.sum().item()
        print(f"\nToken count verification:")
        print(f"Batch size: {batch_size}")
        print(f"Total tokens in test batch: {test_tokens}")
        print(f"Average tokens per sequence: {test_tokens/batch_size:.1f}")
        print(f"Expected tokens for batch size 64: {test_tokens * (64/batch_size):.1f}")

        num_iterations = 1000
        total_samples = 0
        total_tokens = 0

        # Track individual batch times for statistics
        pipeline_times = []  # Full pipeline time including translation
        token_counts = []

        print("Starting measurement iterations...")

        # Pre-generate only the Czech input batches
        print("Preparing Czech input batches...")
        all_czech_inputs = []
        for _ in range(num_iterations):
            inputs = get_random_batch(batch_size, vocab_size, min_seq_length, max_seq_length)
            all_czech_inputs.append(inputs)

        print(f"\nBatch preparation complete. Starting full pipeline measurements...")

        # Main measurement loop - now timing the full pipeline including translation
        with torch.no_grad():
            for i, czech_inputs in enumerate(all_czech_inputs):
                # Measure the full pipeline time including translation
                torch.cuda.synchronize()
                start_time = time.time()

                # Process batch (includes translation if translate=True)
                processed_ids, attention_mask = process_batch(czech_inputs, max_seq_length, translate_model, translate_tokenizer, tokenizer, translate)

                # Model inference
                _ = model(input_ids=processed_ids, attention_mask=attention_mask)

                torch.cuda.synchronize()
                pipeline_time = time.time() - start_time
                pipeline_times.append(pipeline_time)

                # Count tokens from the attention mask
                batch_tokens = attention_mask.sum().item()
                token_counts.append(float(batch_tokens))

                total_samples += batch_size
                total_tokens += batch_tokens

                if (i + 1) % 100 == 0:  # Print more frequently since translation makes it slower
                    print(f"Completed {i + 1}/{num_iterations} iterations...")
                    # Print running average throughput
                    current_time = sum(pipeline_times)
                    current_samples = (i + 1) * batch_size
                    current_tokens = sum(token_counts[:i+1])
                    print(f"Running average: {current_samples/current_time:.1f} samples/sec, {current_tokens/current_time:.1f} tokens/sec")

        # Calculate statistics
        total_pipeline_time = sum(pipeline_times)
        avg_pipeline_time = (total_pipeline_time / num_iterations) * 1000  # in milliseconds
        std_pipeline_time = torch.tensor(pipeline_times).std().item() * 1000  # in milliseconds

        # Calculate throughput based on total pipeline time
        samples_per_second = total_samples / total_pipeline_time
        tokens_per_second = total_tokens / total_pipeline_time

        avg_seq_length = total_tokens / total_samples
        std_seq_length = torch.tensor(token_counts, dtype=torch.float).std().item()

        print("\nResults (including translation time):")
        print(f"Full pipeline throughput: {samples_per_second:.2f} ± {samples_per_second * (std_pipeline_time/avg_pipeline_time):.2f} samples/second")
        print(f"Full pipeline token throughput: {tokens_per_second:.2f} ± {tokens_per_second * (std_pipeline_time/avg_pipeline_time):.2f} tokens/second")
        print(f"Average pipeline time per batch: {avg_pipeline_time:.2f} ± {std_pipeline_time:.2f}ms")
        print(f"Average sequence length: {avg_seq_length:.1f} ± {std_seq_length:.1f} tokens")
        print(f"Total samples processed: {total_samples}")
        print(f"Total tokens processed: {total_tokens}")
        print(f"Number of iterations: {num_iterations}")
        print(f"Standard deviation of pipeline times: {std_pipeline_time:.2f}ms ({std_pipeline_time/avg_pipeline_time*100:.1f}% of mean)")

        # Print some additional diagnostics
        print("\nDiagnostics:")
        print(f"Average tokens per batch: {total_tokens/total_samples:.1f}")
        print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        print(f"Memory cached: {torch.cuda.memory_reserved()/1024**2:.1f}MB")

        # Print batch size specific diagnostics
        print(f"\nBatch size {batch_size} diagnostics:")
        print(f"Average tokens per sample: {avg_seq_length:.1f}")
        print(f"Total tokens per batch (avg): {avg_seq_length * batch_size:.1f}")
        print(f"Throughput per token: {1/(tokens_per_second/1000):.3f}ms")
        print(f"Throughput per sample: {1/(samples_per_second/1000):.3f}ms")

        # Print timing distribution
        pipeline_times_ms = torch.tensor(pipeline_times) * 1000
        print(f"\nTiming distribution (ms):")
        print(f"Min: {pipeline_times_ms.min().item():.2f}")
        print(f"25th percentile: {torch.quantile(pipeline_times_ms, 0.25).item():.2f}")
        print(f"Median: {torch.median(pipeline_times_ms).item():.2f}")
        print(f"75th percentile: {torch.quantile(pipeline_times_ms, 0.75).item():.2f}")
        print(f"Max: {pipeline_times_ms.max().item():.2f}")

    # First measure with fixed batch size of 64
    print("\n=== Fixed Batch Size (64) ===")
    measure_throughput(64)

    # Then find and measure with maximum batch size
    print("\n=== Maximum Batch Size ===")
    batch_size = initial_batch_size
    successful_batch_size = batch_size

    # Find maximum batch size
    while True:
        try:
            # Create random inputs with varying sequence lengths
            inputs = get_random_batch(batch_size, vocab_size, min_seq_length, max_seq_length)

            # Warm-up run with full pipeline
            processed_ids, attention_mask = process_batch(inputs, max_seq_length, translate_model, translate_tokenizer, tokenizer, translate)
            _ = model(input_ids=processed_ids, attention_mask=attention_mask)

            successful_batch_size = batch_size
            batch_size *= 2
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Out of memory at batch size: {batch_size}")
                break
            else:
                raise e

    print(f"Max batch size that fits in memory: {successful_batch_size}")
    measure_throughput(successful_batch_size)


def evaluate_throughput2(
    model, max_seq_length=128, min_seq_length=16, initial_batch_size=1, device="cuda", vocab_size=30522, tokenizer=None, translate=False
):
    model.to(device)
    model.eval()

    # Only load translation model if needed
    translate_model = None
    translate_tokenizer = None
    if translate:
        print("Using translation pipeline (Czech -> English)")
        translate_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-cs")
        translate_model.to("cuda")
        translate_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-cs")
    else:
        print("Using direct model processing (no translation)")

    def measure_throughput(batch_size):
        print(f"\nMeasuring throughput with batch size: {batch_size}")
        torch.cuda.empty_cache()

        # Test token counting with a single batch first
        test_inputs = get_random_batch(batch_size, vocab_size, min_seq_length, max_seq_length)
        test_processed_ids, test_attention_mask = process_batch(test_inputs, max_seq_length, translate_model, translate_tokenizer, tokenizer, translate)
        test_tokens = test_attention_mask.sum().item()
        print(f"\nToken count verification:")
        print(f"Batch size: {batch_size}")
        print(f"Total tokens in test batch: {test_tokens}")
        print(f"Average tokens per sequence: {test_tokens/batch_size:.1f}")
        print(f"Expected tokens for batch size 64: {test_tokens * (64/batch_size):.1f}")

        # Rest of the function remains the same...
        num_iterations = 1000
        total_samples = 0
        total_tokens = 0

        # Track individual batch times for statistics
        inference_times = []  # Just model inference time
        token_counts = []

        print("Starting measurement iterations...")

        # Pre-generate all batches to avoid timing data preparation
        print("Preparing input batches...")
        all_inputs = []
        total_tokens_in_batches = 0
        for _ in range(num_iterations):
            inputs = get_random_batch(batch_size, vocab_size, min_seq_length, max_seq_length)
            processed_ids, attention_mask = process_batch(inputs, max_seq_length, translate_model, translate_tokenizer, tokenizer, translate)
            all_inputs.append((processed_ids, attention_mask))
            # Count tokens from the attention mask
            batch_tokens = attention_mask.sum().item()
            token_counts.append(float(batch_tokens))
            total_tokens_in_batches += batch_tokens

            # Print first batch details for debugging
            if _ == 0:
                print(f"\nFirst batch details:")
                print(f"Number of input texts: {len(inputs)}")
                print(f"Average input text length: {sum(len(text) for text in inputs)/len(inputs):.1f} chars")
                print(f"Processed shape: {processed_ids.shape}")
                print(f"Attention mask shape: {attention_mask.shape}")
                print(f"Non-zero tokens in first sequence: {attention_mask[0].sum().item()}")
                print(f"Non-zero tokens in last sequence: {attention_mask[-1].sum().item()}")
                print(f"Total non-zero tokens in batch: {batch_tokens}")
                print(f"Average tokens per sequence: {batch_tokens/batch_size:.1f}")

        print(f"\nBatch preparation complete:")
        print(f"Average tokens per batch: {total_tokens_in_batches/num_iterations:.1f}")
        print(f"Average tokens per sequence: {total_tokens_in_batches/(num_iterations*batch_size):.1f}")

        # Main measurement loop - now only timing the model inference
        with torch.no_grad():
            for i, (processed_ids, attention_mask) in enumerate(all_inputs):
                # Only measure the model inference time
                torch.cuda.synchronize()
                start_time = time.time()

                _ = model(input_ids=processed_ids, attention_mask=attention_mask)

                torch.cuda.synchronize()
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                total_samples += batch_size
                total_tokens += token_counts[i]

                if (i + 1) % 1000 == 0:
                    print(f"Completed {i + 1}/{num_iterations} iterations...")
                    # Print running average throughput
                    current_time = sum(inference_times)
                    current_samples = (i + 1) * batch_size
                    current_tokens = sum(token_counts[:i+1])
                    print(f"Running average: {current_samples/current_time:.1f} samples/sec, {current_tokens/current_time:.1f} tokens/sec")

        # Calculate statistics
        total_inference_time = sum(inference_times)
        avg_inference_time = (total_inference_time / num_iterations) * 1000  # in milliseconds
        std_inference_time = torch.tensor(inference_times).std().item() * 1000  # in milliseconds

        # Calculate throughput based on pure inference time
        samples_per_second = total_samples / total_inference_time
        tokens_per_second = total_tokens / total_inference_time

        avg_seq_length = total_tokens / total_samples
        std_seq_length = torch.tensor(token_counts, dtype=torch.float).std().item()

        print("\nResults:")
        print(f"Pure model inference throughput: {samples_per_second:.2f} ± {samples_per_second * (std_inference_time/avg_inference_time):.2f} samples/second")
        print(f"Pure model token throughput: {tokens_per_second:.2f} ± {tokens_per_second * (std_inference_time/avg_inference_time):.2f} tokens/second")
        print(f"Average inference time per batch: {avg_inference_time:.2f} ± {std_inference_time:.2f}ms")
        print(f"Average sequence length: {avg_seq_length:.1f} ± {std_seq_length:.1f} tokens")
        print(f"Total samples processed: {total_samples}")
        print(f"Total tokens processed: {total_tokens}")
        print(f"Number of iterations: {num_iterations}")
        print(f"Standard deviation of inference times: {std_inference_time:.2f}ms ({std_inference_time/avg_inference_time*100:.1f}% of mean)")

        # Print some additional diagnostics
        print("\nDiagnostics:")
        print(f"Average tokens per batch: {total_tokens/total_samples:.1f}")
        print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        print(f"Memory cached: {torch.cuda.memory_reserved()/1024**2:.1f}MB")

        # Print batch size specific diagnostics
        print(f"\nBatch size {batch_size} diagnostics:")
        print(f"Average tokens per sample: {avg_seq_length:.1f}")
        print(f"Total tokens per batch (avg): {avg_seq_length * batch_size:.1f}")
        print(f"Throughput per token: {1/(tokens_per_second/1000):.3f}ms")
        print(f"Throughput per sample: {1/(samples_per_second/1000):.3f}ms")

        # Print timing distribution
        inference_times_ms = torch.tensor(inference_times) * 1000
        print(f"\nTiming distribution (ms):")
        print(f"Min: {inference_times_ms.min().item():.2f}")
        print(f"25th percentile: {torch.quantile(inference_times_ms, 0.25).item():.2f}")
        print(f"Median: {torch.median(inference_times_ms).item():.2f}")
        print(f"75th percentile: {torch.quantile(inference_times_ms, 0.75).item():.2f}")
        print(f"Max: {inference_times_ms.max().item():.2f}")

    # First measure with fixed batch size of 64
    print("\n=== Fixed Batch Size (64) ===")
    measure_throughput(64)

    # Then find and measure with maximum batch size
    print("\n=== Maximum Batch Size ===")
    batch_size = initial_batch_size
    successful_batch_size = batch_size

    # Find maximum batch size
    while True:
        try:
            # Create random inputs with varying sequence lengths
            inputs = get_random_batch(batch_size, vocab_size, min_seq_length, max_seq_length)

            # Warm-up run
            processed_ids, attention_mask = process_batch(inputs, max_seq_length, translate_model, translate_tokenizer, tokenizer, translate)
            _ = model(input_ids=processed_ids, attention_mask=attention_mask)

            successful_batch_size = batch_size
            batch_size *= 2
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Out of memory at batch size: {batch_size}")
                break
            else:
                raise e

    print(f"Max batch size that fits in memory: {successful_batch_size}")
    measure_throughput(successful_batch_size)


# Example usage:
# For Contriever (needs translation):
model = AutoModel.from_pretrained("facebook/contriever")
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
evaluate_throughput(model, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer, translate=True)

del model, tokenizer

# For other models (no translation needed):
model, tokenizer, _ = src.simcse.load_simcse()
evaluate_throughput2(model, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer, translate=False)

del model, tokenizer

tokenizer = AutoTokenizer.from_pretrained("Ajchler/czechtriever-demo")
model = AutoModel.from_pretrained("Ajchler/czechtriever-demo")
evaluate_throughput2(model, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer, translate=False)
