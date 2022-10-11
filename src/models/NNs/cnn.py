import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, history_length=0, n_classes=3):
      super(CNN, self).__init__()
      self.seq = nn.Sequential(
        nn.Conv2d(in_channels=history_length+1, out_channels=16, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
      )
      self.flat = nn.Flatten()
      self.fc1 = nn.Linear(128 * 4 * 4, 128)
      self.fc2 = nn.Linear(128, n_classes)
    
    def forward(self, x):
        x = self.seq(x)
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x