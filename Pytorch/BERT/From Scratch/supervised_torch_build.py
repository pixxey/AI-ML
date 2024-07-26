import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW

# Step 1: Define and Initialize the BERT Model from Scratch
class BertModelFromScratch(nn.Module):
    def __init__(self, config):
        super(BertModelFromScratch, self).__init__()
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 2)  # Binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(cls_output)
        return logits

# Initialize BERT config from scratch
config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    type_vocab_size=2
)

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModelFromScratch(config)

# Initialize weights
def initialize_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

model.apply(initialize_weights)

# Save the initialized model
torch.save(model.state_dict(), 'bert_from_scratch.pth')
print("Model initialized and saved to bert_from_scratch.pth")

# Step 2: Load and Preprocess Dataset for Training
# Using the IMDb dataset for binary sentiment classification
dataset = load_dataset('imdb')

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)

def convert_to_tensors(batch):
    return {
        'input_ids': torch.tensor(batch['input_ids']),
        'attention_mask': torch.tensor(batch['attention_mask']),
        'labels': torch.tensor(batch['label'])
    }

tensor_dataset = encoded_dataset.map(convert_to_tensors, batched=True, remove_columns=dataset['train'].column_names)

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    labels = torch.stack([torch.tensor(item['labels']) for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

train_loader = DataLoader(tensor_dataset['train'], batch_size=8, shuffle=True, collate_fn=collate_fn)

# Step 3: Train the Model from Scratch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')
print("Training completed!")

# Save the trained model
torch.save(model.state_dict(), 'supervised_bert_trained.pth')
print("Model trained and saved to supervised_bert_trained.pth")
