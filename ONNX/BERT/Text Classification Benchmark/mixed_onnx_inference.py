import time
import numpy as np
import onnxruntime as ort
from transformers import BertTokenizer

# Load the pretrained BERT ONNX model
model_path = "bert_mixed.onnx"  # Update this path to your actual ONNX model path
session = ort.InferenceSession(model_path, providers=['ROCMExecutionProvider'])

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create a sample input
sample_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(sample_text, return_tensors="np")

# Prepare the inputs for the ONNX model
input_feed = {
    'input_ids': inputs['input_ids'],
    'attention_mask': inputs['attention_mask'],
    'token_type_ids': inputs['token_type_ids']
}

# Define a function to benchmark the model and calculate tokens/second
def benchmark_model(session, input_feed, num_runs=100):
    output_name = session.get_outputs()[0].name
    num_tokens = input_feed['input_ids'].shape[1]

    start_time = time.time()
    for _ in range(num_runs):
        session.run([output_name], input_feed)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_run = total_time / num_runs
    tokens_per_second = (num_tokens * num_runs) / total_time

    print(f"Average inference time over {num_runs} runs: {avg_time_per_run * 1000:.2f} ms")
    print(f"Tokens processed per second: {tokens_per_second:.2f}")

# Run the benchmark
benchmark_model(session, input_feed)
