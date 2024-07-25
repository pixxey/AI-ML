import onnx
from onnxconverter_common.auto_mixed_precision import auto_convert_mixed_precision
import numpy as np

# Define the shape of the input tensors for BERT (batch_size, sequence_length)
batch_size = 1
sequence_length = 128

# Create sample input tensors
input_ids = np.random.randint(30522, size=(batch_size, sequence_length)).astype(np.int64)
attention_mask = np.ones((batch_size, sequence_length)).astype(np.int64)  # All tokens are valid
token_type_ids = np.zeros((batch_size, sequence_length)).astype(np.int64)  # All tokens are from the first segment

# Construct dictionary for input tensors
feed_dict = {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'token_type_ids': token_type_ids
}

# Load the FP16 model
model_path_fp16 = "bert_fp16.onnx"
model_fp16 = onnx.load(model_path_fp16)

# Convert the FP16 model to mixed precision
model_mixed = auto_convert_mixed_precision(model_fp16, feed_dict, rtol=0.01, atol=0.001, keep_io_types=True)

# Save the mixed precision model
onnx.save(model_mixed, "bert_mixed.onnx")

print(f"Mixed precision model saved to: bert_mixed.onnx")
