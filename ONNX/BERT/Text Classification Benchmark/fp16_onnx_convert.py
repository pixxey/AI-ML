import onnx
from onnxconverter_common import float16

# Load the BERT model
model_path = "bert.onnx"
model = onnx.load(model_path)

# Convert the model to FP16 (mixed precision)
model_fp16 = float16.convert_float_to_float16(model)

# Save the converted model
onnx.save(model_fp16, "bert_fp16.onnx")

print(f"FP16 model saved to: bert_fp16.onnx")
