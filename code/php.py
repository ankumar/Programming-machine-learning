import datetime
import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Initialization and Data Preparation
embedding_dim = 128
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Load the configuration for DistilBERT
config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
dbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased', config=config)

# Freeze all the parameters in the DistilBERT model.
for param in dbert_model.parameters():
    param.requires_grad = False

# Load your dataset. Assuming it's a CSV file with columns 'conversation', 'model_a', 'model_b', 'winner'
datapath = 'pair_wise.csv'
df = pd.read_csv(datapath)

unique_model_names = list(set(df['model_a'].unique()).union(set(df['model_b'].unique())))
model2id = {model: idx for idx, model in enumerate(unique_model_names)}

class PairWiseDataSet(Dataset):
    def __init__(self, dataframe, tokenizer, model2id, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.model2id = model2id
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['conversation']
        model_a_id = self.model2id[row['model_a']]
        model_b_id = self.model2id[row['model_b']]
        winner_id = self.model2id[row['winner']]
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return input_ids, attention_mask, torch.tensor([model_a_id, model_b_id]), torch.tensor(winner_id)

# Split the dataset
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = PairWiseDataSet(train_df, tokenizer, model2id)
test_dataset = PairWiseDataSet(test_df, tokenizer, model2id)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class PairwiseClassifier(nn.Module):
    def __init__(self, dbert_model, num_models, embedding_dim):
        super(PairwiseClassifier, self).__init__()
        self.dbert_model = dbert_model
        self.model_embeddings = nn.Embedding(num_models, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(dbert_model.config.hidden_size + 2 * embedding_dim, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, num_models)
        )

    def forward(self, input_ids, attention_mask, model_pair_ids):
        outputs = self.dbert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        model_embeddings = self.model_embeddings(model_pair_ids)
        model_embeddings = model_embeddings.view(model_embeddings.size(0), -1)
        combined = torch.cat((pooled_output, model_embeddings), 1)
        logits = self.classifier(combined)
        return logits

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Adjust for Metal GPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")
model = PairwiseClassifier(dbert_model, len(model2id), embedding_dim).to(device)

# Measure time for training
start_time = datetime.datetime.now()

# Training loop
epochs = 5 # Number of epochs
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Define the optimizer
criterion = nn.CrossEntropyLoss()  # Define the loss function
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    train_loop = tqdm(train_loader, position=0, leave=True, desc=f'Epoch {epoch+1}/{epochs} [Training]')
    for inputs, attention_mask, model_pairs, labels in train_loop:
        inputs, attention_mask, model_pairs, labels = inputs.to(device), attention_mask.to(device), model_pairs.to(device), labels.to(device)
        
        optimizer.zero_grad() # Zero the gradients
        outputs = model(inputs, attention_mask, model_pairs) # Forward pass
        loss = criterion(outputs, labels) # Compute the loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights
        
        train_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_predictions += labels.size(0)

        train_loop.set_postfix(loss=(train_loss / (train_loop.n + 1)))

    train_accuracy = correct_predictions / total_predictions
        
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    val_loop = tqdm(test_loader, position=0, leave=True, desc=f'Epoch {epoch+1}/{epochs} [Validation]')
    with torch.no_grad():
        for inputs, attention_mask, model_pairs, labels in val_loop:
            inputs, attention_mask, model_pairs, labels = inputs.to(device), attention_mask.to(device), model_pairs.to(device), labels.to(device)

            outputs = model(inputs, attention_mask, model_pairs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

            val_loop.set_postfix(loss=(val_loss / (val_loop.n + 1)))

    val_accuracy = correct_predictions / total_predictions

    # Print formatted loss and accuracy
    print(f'\nEpoch {epoch+1} \t Training Loss: {train_loss / len(train_loader):.3f} \t Validation Loss: {val_loss / len(test_loader):.3f}')
    print(f'\t Training Accuracy: {train_accuracy:.3%} \t Validation Accuracy: {val_accuracy:.3%}')


# Save the model
torch.save(model.state_dict(), 'pairwise_classifier_model.pth')
print('Model saved to pairwise_classifier_model.pth')

# Measure training time
end_time = datetime.datetime.now()
print(f'Training completed in: {end_time - start_time}')




