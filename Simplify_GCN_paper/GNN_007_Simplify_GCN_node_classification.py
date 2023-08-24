import os.path as osp
import sys


import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SGConv
import torch.nn.functional as F 

# ===================Import dataset==============

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset)
data = dataset[0]
print("Cora: ", data)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
# =================== Contruct The Simplifying Graph Covn Network Model==============
SGC_model = SGConv(in_channels= data.num_features, # Number of features
                    out_channels = dataset.num_classes, # Dimension of embedding
                    K = 1,
                    cached = True
                    )
# =================== GET Embedding ==================
print("Shape of the original data: ", data.x.shape)
print("Shape of the embedding data: ", SGC_model(data.x, data.edge_index).shape)


#==================== Contruct the NODE Classification Model ==================
class SGCNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SGConv(in_channels = dataset.num_features, # Number of features
                            out_channels = dataset.num_classes, # dimension of embedding
                            K =2, cached = True)
    def forward(self):
        x = self.conv1(data.x, data.edge_index) # applying convolution to data
        return F.log_softmax(x, dim =1)
        
    
# device consideration
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SGC_model, data = SGCNet().to(device), data.to(device)
optimizer = torch.optim.Adam(SGC_model.parameters(), lr=0.2,weight_decay=0.005)

# learning parameters
for i, parameters in SGC_model.named_parameters():
    print(" Parameters {}".format(i))
    print(" Shape: ", parameters.shape)
        

# ============ TRAIN ==================
train_losses = []
train_accuracies = []
val_accuracies = []
test_accuracies = []
def train():
    SGC_model.train()
    optimizer.zero_grad()
    predicted_y = SGC_model()
    true_y = data.y
    losses = F.nll_loss(predicted_y[data.train_mask], true_y[data.train_mask])
    losses.backward()
    optimizer.step()
    train_losses.append(losses.item())  # Store the loss value



# =================TEST Function ====================
@torch.no_grad()
def test():
    SGC_model.eval()   # set the model.training to be false
    logits, accs = SGC_model(), [] # log probability and accuracy llist of model
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs



#===================== Putting all together ================
best_val_acc = 0
test_acc = 0
for epoch in range(1, 101):
    train()
    train_acc, val_acc, tmp_test_acc = test()

    # Append accuracies to lists
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    test_accuracies.append(tmp_test_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc

    log = 'Epoch: {:03d}, Loss: {:.4f}, Train: {:.4f}, val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_losses[-1], train_acc, best_val_acc, test_acc))



#===================== Ploting loss and accuracy ==========
def plot_loss_and_accuracy():
    plt.figure(figsize=(12, 4))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training, Validation, and Test Accuracies')
    plt.legend()

    plt.tight_layout()
    plt.show()
plot_loss_and_accuracy()