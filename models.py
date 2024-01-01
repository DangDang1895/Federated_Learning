import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
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
