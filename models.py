import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 784
        self.fc1 = nn.Linear(self.input_dim,200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, X):
        X = F.relu(self.fc1(X.reshape(-1,self.input_dim)))
        X = self.fc2(X)
        return X


class cifar10_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 3072
        self.fc1 = nn.Linear(self.input_dim,1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 10)


    def forward(self, X):
        X = F.relu(self.fc1(X.reshape(-1,self.input_dim)))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return X

class CNNMnist(nn.Module):
    def __init__(self,in_channels,out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, out_dim)

    def forward(self, X):
        X = F.relu(F.max_pool2d(self.conv1(X), 2))
        X = F.relu(F.max_pool2d(self.conv2(X), 2))
        X = F.relu(self.fc1(X.reshape(-1,320)))
        X = self.fc2(X)
        return X


class cifar10_CNNMnist(nn.Module):
    def __init__(self,in_channels,out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 10, kernel_size=5)
        self.fc1 = nn.Linear(250, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, X):
        X = F.relu(F.max_pool2d(self.conv1(X), 2))
        X = F.relu(F.max_pool2d(self.conv2(X), 2))
        X = F.relu(self.fc1(X.reshape(-1,250)))
        X = self.fc2(X)
        return X

""" #测试模型
model = CNNMnist(in_channels=1,output_dim=10)
#net = MLP(784,10)
X = torch.randn(size=(2,1,28,28))
X = model(X)
 """