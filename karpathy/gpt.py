import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformer model components

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weights = q @ k.transpose(-2, -1) * C ** -0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(n_heads, n_embed // n_heads)
        self.ffw = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x
    

class TransformerModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb.unsqueeze(0)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits, None
    
    def generate(self, idx, max_new_tokens):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        return idx

from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset

# Load a tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # To allow for padding with the EOS token

# Define a dataset class
class WikiTextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        super(WikiTextDataset, self).__init__()
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text = open(file_path, encoding="utf-8").read()
        self.tokens = tokenizer(self.text, return_tensors='pt', truncation=True, max_length=self.block_size)['input_ids'][0]

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        return self.tokens[start_idx:end_idx]

# Prepare the dataset
train_dataset = WikiTextDataset('wikitext-2/wiki.train.tokens', tokenizer)
val_dataset = WikiTextDataset('wikitext-2/wiki.valid.tokens', tokenizer)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

from torch.optim import Adam

# Hyperparameters for the Transformer
dropout = 0.1
n_embed = 64
n_heads = 4
block_size = 32
batch_size = 16
n_layers = 4
vocab_size = 10000  # Example vocabulary size

# Initialize the Transformer model
# Adjust for Metal GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")
# model = TransformerModel(vocab_size=vocab_size).to(device)

# Assuming your TransformerModel class is defined as in previous instructions
model = TransformerModel(vocab_size=len(tokenizer)).to(device)

optimizer = Adam(model.parameters(), lr=5e-5)

def train_epoch(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for batch in data_loader:
        inputs = batch.to(device)
        optimizer.zero_grad()
        _, loss = model(inputs, inputs)  # Using inputs as both inputs and targets
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Training loop
epochs = 5  # For demonstration; adjust as needed
for epoch in range(epochs):
    loss = train_epoch(model, train_loader, optimizer)
    print(f'Epoch {epoch+1}, Loss: {loss}')

def generate_text(model, tokenizer, prompt, max_length=50):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad():
        generated_ids = model.generate(input_ids, max_new_tokens=max_length - input_ids.size(1))
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

prompt = "The mysteries of the universe are"
generated_text = generate_text(model, tokenizer, prompt)
print(generated_text)






