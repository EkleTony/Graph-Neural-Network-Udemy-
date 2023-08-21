#===================IMPORT LIBRARY=============
import torch 
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

#============MODEL==========
path = "/Users/anthonyekle/Desktop/TENNESSEE TECH/PHD-WORLD/02 SUMMER-2023/GNN and Anomaly Detection/Codes"
train = datasets.MNIST(path, train=True, download=True, 
                       transform = transforms.Compose([transforms.ToTensor()]))


test = datasets.MNIST(path, train=False, download=True, 
                       transform = transforms.Compose([transforms.ToTensor()]))

#==============DATA Loader =================
train_set = torch.utils.data.DataLoader(train, batch_size =10, shuffle=True)
train_set = torch.utils.data.DataLoader(test, batch_size =10, shuffle=True)


for data in train_set:
    print(data)
    break
  
X, y = data[0][0], data[1][0]

print(y)


print(data[0][0].shape)

#===============Visualization ==========

plt.imshow(data[0][0].view(28,28))
#plt.show()

#=============Balance dataset ==========

total  = 0 
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in train_set:
    Xs, ys = data
    for y in ys:
        counter_dict[int(y)] += 1
        total+=1
        
print(counter_dict)

for i in counter_dict:
    print(f"{i}: {counter_dict[i]/total*100}")
    
print(28*28)