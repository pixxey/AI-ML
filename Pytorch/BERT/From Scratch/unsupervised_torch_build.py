import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertForMaskedLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW
import random

# Step 1: Define and Initialize the BERT Model from Scratch
class BertModelForPretraining(nn.Module):
    def __init__(self, config):
        super(BertModelForPretraining, self).__init__()
        self.bert = BertForMaskedLM(config)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

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
model = BertModelForPretraining(config)

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

# Step 2: Load and Preprocess Dataset for Pre-training
# Using a small subset of Wikipedia for demonstration
dataset = load_dataset('wikipedia', '20220301.en', split='train[:10000]')  # Load first 10000 examples

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling. """
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) keep the masked input tokens unchanged
    return inputs, labels

def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    input_ids = torch.tensor(tokenized_inputs['input_ids'])
    inputs, labels = mask_tokens(input_ids, tokenizer)
    tokenized_inputs['input_ids'] = inputs
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    labels = torch.stack([torch.tensor(item['labels']) for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

train_loader = DataLoader(encoded_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Step 3: Train the Model from Scratch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        loss, _ = model(input_ids, attention_mask, labels=labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')
print("Training completed!")

# Save the trained model
torch.save(model.state_dict(), 'bert_from_scratch_trained.pth')
print("Model trained and saved to bert_from_scratch_trained.pth")
