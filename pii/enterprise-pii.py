import datetime
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Check for MPS device availability for Apple Silicon Macs
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Initialization and Data Preparation
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
dbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased', config=config).to(device)

# Freeze all the parameters in the DistilBERT model.
for param in dbert_model.parameters():
    param.requires_grad = False

class SensitiveInfoDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_dataset(file_path)
    
    def load_dataset(self, file_path):
        samples = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                samples.append((data['query'], data['gold']))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

# Load the dataset
file_path = 'enterprise_pii_classification.jsonl'  # Update this path accordingly
dataset = SensitiveInfoDataset(file_path, tokenizer)

# Split the dataset
train_dataset, test_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class SensitiveInfoClassifier(nn.Module):
    def __init__(self, dbert_model):
        super(SensitiveInfoClassifier, self).__init__()
        self.dbert_model = dbert_model
        self.classifier = nn.Sequential(
            nn.Linear(dbert_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2)  # Output layer for binary classification
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.dbert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use the [CLS] token's output
        logits = self.classifier(pooled_output)
        return logits

model = SensitiveInfoClassifier(dbert_model).to(device)

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    predictions, true_labels = [], []

    for inputs, attention_mask, labels in tqdm(train_loader, desc="Training"):
        inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predictions += torch.argmax(outputs, dim=1).tolist()
        true_labels += labels.tolist()

    train_accuracy = accuracy_score(true_labels, predictions)
    print(f'Epoch {epoch+1} \t Training Loss: {train_loss / len(train_loader):.3f} \t Training Accuracy: {train_accuracy:.3%}')

    # Validation loop
    model.eval()
    val_loss = 0.0
    predictions, true_labels = [], []
    with torch.no_grad():
        for inputs, attention_mask, labels in tqdm(test_loader, desc="Validation"):
            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(inputs, attention_mask)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            predictions += torch.argmax(outputs, dim=1).tolist()
            true_labels += labels.tolist()

    val_accuracy = accuracy_score(true_labels, predictions)
    print(f'Epoch {epoch+1} \t Validation Loss: {val_loss / len(test_loader):.3f} \t Validation Accuracy: {val_accuracy:.3%}')

print('Training completed')
