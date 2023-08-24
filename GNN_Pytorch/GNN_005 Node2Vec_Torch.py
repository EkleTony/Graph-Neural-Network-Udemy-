# ===================Import libraries==============
import torch
import os.path as osp
import sys

import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import Node2Vec # Import Node2Vec model
from sklearn.manifold import TSNE 

# ===================Import dataset==============
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset)
data = dataset[0] 
#path = "C:/Users/anthonyekle/Desktop"
#dataset = Planetoid(path,  "Cora") # download the dataset
#data = dataset[0] # tensor representation of cura
print("coda: ", data)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# =================== Contruct The Model==============
Node2Vec_model = Node2Vec(
        data.edge_index,
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1,
        q=1,
        sparse=True,
    ).to(device)

# Using loader for batch sizes for scaling purpose
num_workers = 0 if sys.platform.startswith('win') else 4
loader = Node2Vec_model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
optimizer = torch.optim.SparseAdam(list(Node2Vec_model.parameters()), # list of parameters
                                   lr = 0.01 # learning rate
                                   )
# ================Train Function ======================

def train():
        Node2Vec_model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

# =================== Obtain Embedded Representation ==============

for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch: 02d}, Loss: {loss:.4f}')
    
# =================== Plot 2D of Embedded Rep.==============
@torch.no_grad # Deactivate autograd functionality
def plot_points(colors):
    Node2Vec_model.eval() # evaluate the model based on the trained parameters
    z = Node2Vec_model(torch.arange(data.num_nodes, device=device)) # Embedding rep
    z = TSNE(n_components =2 ).fit_transform(z.cpu().numpy())
    y = data.y.cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    for i in range(dataset.num_classes):
        plt.scatter(z[y==i, 0], z[y==i, 1], s=20, color = colors[i])
    plt.axis('off')
    plt.show()
    

colors = [
        '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
        '#ffd700'
    ]
plot_points(colors)

if __name__ == '__main__':
    freeze_support()


# =================== Node Classification==============



