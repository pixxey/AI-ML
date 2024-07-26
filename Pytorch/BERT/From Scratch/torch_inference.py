import torch
from transformers import BertTokenizer, BertConfig, BertModel
import torch.nn as nn
import torch.nn.functional as F

class BertModelForClassification(nn.Module):
    def __init__(self, config, num_labels):
        super(BertModelForClassification, self).__init__()
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # pooled_output is the output of the [CLS] token
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return loss, logits

# Load the configuration
config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    type_vocab_size=2
)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the number of labels (e.g., 2 for binary classification)
num_labels = 2

# Initialize the classification model
model = BertModelForClassification(config, num_labels)

# Load the weights from the MLM-trained model
state_dict = torch.load('unsupervised_bert_trained.pth', map_location=torch.device('cpu'))
state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}  # Exclude classifier weights
model.bert.load_state_dict(state_dict, strict=False)

# Move the model to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function to perform inference
def classify_text(model, inputs):
    model.eval()
    with torch.no_grad():
        _, logits = model(**inputs)
    return logits

# List of sample texts
texts = [
    "The movie was fantastic!",
    "The plot was boring and predictable.",
    "The acting was superb.",
    "The film was too long and dull."
]

# Classify each text and map to human-readable labels
label_map = {0: "Negative", 1: "Positive"}

for text in texts:
    inputs = tokenizer(text, return_tensors="pt")
    inputs.pop('token_type_ids', None)  # Remove token_type_ids if present
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to the same device as the model
    
    logits = classify_text(model, inputs)
    predicted_class = torch.argmax(logits, dim=1).item()
    
    print(f"Text: {text}")
    print(f"Predicted class: {predicted_class} ({label_map[predicted_class]})")
    print()
