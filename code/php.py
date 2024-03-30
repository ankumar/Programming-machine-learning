# Predictive Human Preference - PHP
# This is an attempt to implement this article https://huyenchip.com/2024/02/28/predictive-human-preference.html

import datetime
import pandas as pd
import transformers
import numpy as np
from sentence_transformers import SentenceTransformer

import torch
from torch import nn

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

embedding_dim = 128

pair_wise_words = []
winner = []

tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
config = transformers.DistilBertConfig(dropout=0.2, attention_dropout=0.2)
dbert_pt = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased', config=config)
sentence_transformer_model = SentenceTransformer("all-mpnet-base-v2")

for param in dbert_pt.parameters():
    param.requires_grad = False

datapath = 'pair_wise.csv'
df = pd.read_csv(datapath)

X = df[['conversation', 'model_a', 'model_b']]
X_list=X.values.tolist()

unique_model_names = []
for conv, model_a, model_b in X_list:
    if model_a not in unique_model_names:
        unique_model_names.append(model_a)
    if model_b not in unique_model_names:
        unique_model_names.append(model_b)

print("Number of Unique Models ", len(unique_model_names))

id2model = dict(enumerate(unique_model_names))
model2id = {token: id for id, token in id2model.items()}

print(id2model)
print(model2id)

training_data = []
for elem in X_list:
    # A word_tokens = tokenizer(elem[0], padding='max_length', max_length = 512, truncation=True, return_tensors='pt')["input_ids"]
    sent_emb = sentence_transformer_model.encode(elem[0])
    model_array = np.array([model2id[elem[1]], model2id[elem[2]]])
    model_pair = torch.from_numpy(model_array)
    # A training_data.append((word_tokens, (model_pair)))
    training_data.append((sent_emb, (model_pair)))
    
y_list = []
for winner in df['winner']:
    y_list.append(model2id[winner])

y_pt = torch.Tensor(y_list).long()

# A X_pt_train, X_pt_test, y_pt_train, y_pt_test = train_test_split(training_data, y_pt)

X_pt_train, X_pt_test, y_pt_train, y_pt_test = train_test_split(training_data, y_pt, test_size=0.10, random_state=42, stratify=y_pt)


class PairWiseDataSet(Dataset):
    def __init__(self, X, y):
        self.X_train = X
        self.y_train = y

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

train_data_pt = PairWiseDataSet(X=X_pt_train, y=y_pt_train)
test_data_pt = PairWiseDataSet(X=X_pt_test, y=y_pt_test)

# Get train and test data in form of Dataloader class
train_loader_pt = DataLoader(train_data_pt, batch_size=32)
test_loader_pt = DataLoader(test_data_pt, batch_size=32)

# A device = "cuda" if torch.cuda.is_available() else "cpu"
# A print(f"Using {device} device")

# Adjust for Metal GPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

class PairwiseClassifer(nn.Module):
    def __init__(self):
        super(PairwiseClassifer, self).__init__()
        self.dbert = dbert_pt
        self.embedding = nn.Embedding(len(unique_model_names), embedding_dim) 
        self.model = nn.Sequential(nn.Linear(1024, 768),
                                   nn.Dropout(p=0.2),  # Adding dropout layer with a dropout rate of 0.2
                                   nn.ReLU(),
                                   nn.Linear(768,256),
                                   nn.ReLU(),
                                   nn.Linear(256,128),
                                   nn.ReLU(),
                                   nn.Linear(128,64),
                                   nn.ReLU(),
                                   nn.Linear(64,len(unique_model_names)))
#                                   nn.Sigmoid())

    def forward(self, x):
        inp = x[0]
        inp = torch.squeeze(inp, dim=1)
        f2 = x[1]
        # A inp = self.dbert(input_ids=inp)
        # A inp = inp["last_hidden_state"][:,0,:]
        x = self.embedding(f2)
        x = torch.flatten(x, start_dim=1)
        y = torch.cat((inp, x), dim=1)
        logits = self.model(y)
        return logits
       
model_pt = PairwiseClassifer().to(device)
print(model_pt)

total_params = sum(p.numel() for p in model_pt.parameters())
total_params_trainable = sum(p.numel() for p in model_pt.parameters() if p.requires_grad)
print("Number of parameters: ", total_params)
print("Number of trainable parameters: ", total_params_trainable)

epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_pt.parameters(), lr=0.0001)
from tqdm import tqdm
# Define the dictionary "history" that will collect key performance indicators during training
history = {}
history["epoch"]=[]
history["train_loss"]=[]
history["valid_loss"]=[]
history["train_accuracy"]=[]
history["valid_accuracy"]=[]


# Measure time for training
start_time = datetime.datetime.now()

# Loop on epochs
for e in range(epochs):
    
    # Set mode in train mode
    model_pt.train()
    
    train_loss = 0.0
    train_accuracy = []
    
    # Loop on batches
    for X, y in tqdm(train_loader_pt):
        X = (X[0].to(device), X[1].to(device))  # Pass all elements to device
        y = y.to(device)

        # Get prediction & loss
        prediction = model_pt(X)
        prediction = torch.squeeze(prediction)
        loss = criterion(prediction, y)
        
        # Adjust the parameters of the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        prediction_index = prediction.argmax(axis=1)
        accuracy = (prediction_index==y)
        train_accuracy += accuracy
    
    train_accuracy = (sum(train_accuracy) / len(train_accuracy)).item()
    
    # Calculate the loss on the test data after each epoch
    # Set mode to evaluation (by opposition to training)
    model_pt.eval()
    valid_loss = 0.0
    valid_accuracy = []
    for X, y in test_loader_pt:
        X = (X[0].to(device), X[1].to(device))  # Pass all elements to device
        y = y.to(device)
    
        prediction = model_pt(X)
        loss = criterion(prediction, y)

        valid_loss += loss.item()
        
        
        prediction_index = prediction.argmax(axis=1)
        accuracy = (prediction_index==y)
        valid_accuracy += accuracy
    valid_accuracy = (sum(valid_accuracy) / len(valid_accuracy)).item()
    
    # Populate history
    history["epoch"].append(e+1)
    history["train_loss"].append(train_loss / len(train_loader_pt))
    history["valid_loss"].append(valid_loss / len(test_loader_pt))
    history["train_accuracy"].append(train_accuracy)
    history["valid_accuracy"].append(valid_accuracy)    
        
    print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader_pt) :10.3f} \t\t Validation Loss: {valid_loss / len(test_loader_pt) :10.3f}')
    print(f'\t\t Training Accuracy: {train_accuracy :10.3%} \t\t Validation Accuracy: {valid_accuracy :10.3%}')
    
# Measure time for training
end_time = datetime.datetime.now()
