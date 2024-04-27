import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch.nn import Parameter
from torch_geometric.utils import add_self_loops, softmax
import torch.nn.functional as F
import json
import numpy as np

# Custom Message Passing Layer for directed graph connections with attention mechanism
class CustomConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomConv, self).__init__(aggr='add')  # Using sum aggregation for messages
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Linear transformation layer for node features
        self.lin = torch.nn.Linear(in_channels, out_channels)

        # Attention coefficients and bias parameters
        self.att = Parameter(torch.Tensor(1, 2 * out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Use Xavier initialization for weights and zero for biases
        torch.nn.init.xavier_uniform_(self.lin.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.att, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, x, edge_index):
        # Add self-loops to the edges to include self-connections
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)  # Apply linear transformation
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)  # Propagate messages

    def message(self, x_i, x_j, edge_index, size_i):
        # Calculate attention coefficients for messages
        z = torch.cat([x_i, x_j], dim=1)
        attention = F.leaky_relu(torch.matmul(z, self.att.t()))
        attention = softmax(attention, edge_index[0], num_nodes=size_i)
        return attention * x_j  # Weight messages by attention coefficients
        
    def update(self, aggr_out):
        # Optional: update nodes with aggregated messages
        return F.relu(aggr_out + self.bias)
    
def load_json_data(filepath):
    # Load JSON data from file
    with open(filepath, 'r') as file:
        data = json.load(file)
    aggregated_data = {}
    # Process data to average measurements for each node
    for item in data:
        row_id = item.pop('_row')
        for k, v in item.items():
            if row_id not in aggregated_data:
                aggregated_data[row_id] = []
            aggregated_data[row_id].append(v)
    for row_id in aggregated_data:
        aggregated_data[row_id] = np.mean(aggregated_data[row_id])
    return aggregated_data

def calculate_differential_features(diseased_data, healthy_data, node_idx):
    # Calculate differential features between diseased and healthy states
    features = torch.zeros(len(node_idx), 1)
    for node, idx in node_idx.items():
        if node in diseased_data and node in healthy_data:
            diseased_level = diseased_data[node] + 1
            healthy_level = healthy_data[node] + 1
            features[idx, 0] = np.log2(healthy_level / diseased_level)
    return features

def load_and_process_data():
    # Load interaction data
    lig_rec = pd.read_csv('/Users/anil/code/python/scMLnet/database/LigRec.txt', sep='\t')
    rec_tf = pd.read_csv('/Users/anil/code/python/scMLnet/database/RecTF.txt', sep='\t')
    tf_target = pd.read_csv('/Users/anil/code/python/scMLnet/database/TFTargetGene.txt', sep='\t')
    diseased_data = load_json_data('/Users/anil/Downloads/t0_diseased.json')
    healthy_data = load_json_data('/Users/anil/Downloads/t1_healthy.json')

    # Initialize node indices and edges
    node_idx = {}
    edges = []
    current_index = 0

    # Function to update node indices and collect graph edges
    def update_indices_and_collect_edges(df, source_col, target_col):
        nonlocal current_index
        for _, row in df.iterrows():
            src, tgt = row[source_col], row[target_col]
            if src not in node_idx:
                node_idx[src] = current_index
                current_index += 1
            if tgt not in node_idx:
                node_idx[tgt] = current_index
                current_index += 1
            edges.append([node_idx[src], node_idx[tgt]])

    # Update indices and edges from interaction data
    update_indices_and_collect_edges(lig_rec, 'Ligand', 'Receptor')
    update_indices_and_collect_edges(rec_tf, 'Receptor', 'TF')
    update_indices_and_collect_edges(tf_target, 'TF', 'Targets')

    # Calculate node features and prepare graph data object
    x = calculate_differential_features(diseased_data, healthy_data, node_idx)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.randn(len(node_idx), 1)  # Placeholder for actual targets

    return Data(x=x, edge_index=edge_index, y=y)

class GNN(torch.nn.Module):
    """
    Graph Neural Network model class.
    This model uses two convolutional layers followed by a fully connected layer.
    """
    def __init__(self, feature_dim, hidden_dim):
        super(GNN, self).__init__()
        self.conv1 = CustomConv(feature_dim, hidden_dim)  # First convolutional layer
        self.conv2 = CustomConv(hidden_dim, hidden_dim)   # Second convolutional layer
        self.out = torch.nn.Linear(hidden_dim, 1)  # Output layer for predictions

    def forward(self, data):
        """
        Forward pass of the model.
        :param data: Data object containing batched graph data.
        :return: Output tensor of predictions.
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))  # Apply ReLU after first conv layer
        x = F.relu(self.conv2(x, edge_index))  # Apply ReLU after second conv layer
        x = self.out(x)  # Apply final transformation
        return x

# Initialize the model and set hyperparameters
feature_dim = 1  # Number of input features for each node
hidden_dim = 32  # Number of units in hidden layers
model = GNN(feature_dim, hidden_dim)
print(model)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Using Adam optimizer with learning rate of 0.01

def train(data):
    """
    Function to train the model.
    :param data: Data object containing batched graph data.
    :return: Loss value as a float.
    """
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear gradients
    out = model(data)  # Perform forward pass
    loss = F.mse_loss(out, data.y)  # Compute mean squared error loss
    loss.backward()  # Perform backpropagation
    optimizer.step()  # Update model parameters
    return loss.item()  # Return the loss value

# Load data
data = load_and_process_data()  # Load and process graph data

if data.y is None:
    raise ValueError("Target values (data.y) are not set.")  # Ensure targets are set

# Training loop
for epoch in range(100):  # Loop over the dataset multiple times
    loss = train(data)  # Train model on data
    print(f"Epoch {epoch+1}, Loss: {loss}")
