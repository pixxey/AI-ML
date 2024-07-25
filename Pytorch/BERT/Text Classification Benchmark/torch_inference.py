import time
import torch
from transformers import BertTokenizer, BertModel

# Load the pretrained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Move the model to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create a sample input
sample_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(sample_text, return_tensors="pt")

# Move input tensors to the same device as the model
inputs = {k: v.to(device) for k, v in inputs.items()}

# Define a function to benchmark the model and calculate tokens/second
def benchmark_model(model, inputs, num_runs=100):
    model.eval()
    num_tokens = inputs['input_ids'].shape[1]

    # Warm-up run to load model and data onto GPU first
    with torch.no_grad():
        model(**inputs)

    # Perform inference and record time
    latency = []
    for _ in range(num_runs):
        torch.cuda.synchronize()  # Synchronize GPU before starting timing
        start = time.time()
        with torch.no_grad():
            model(**inputs)
        torch.cuda.synchronize()  # Synchronize GPU after finishing timing
        end = time.time()
        latency.append(end - start)

    avg_time_per_run = sum(latency) / num_runs
    tokens_per_second = (num_tokens * num_runs) / sum(latency)

    print(f"Device: {device}")
    print(f"Average inference time over {num_runs} runs: {avg_time_per_run * 1000:.2f} ms")
    print(f"Tokens processed per second: {tokens_per_second:.2f}")

# Run the benchmark
benchmark_model(model, inputs)
