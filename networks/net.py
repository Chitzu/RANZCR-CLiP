import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, l1=256, c1=48, c2=96, d1=0.1):
        super().__init__()
        self.d1 = d1
        self.conv1 = nn.Conv2d(3, c1, 3)
        self.conv2 = nn.Conv2d(c1, c1, 3)
        self.conv3 = nn.Conv2d(c1, c2, 3)
        self.conv4 = nn.Conv2d(c2, c2, 3, stride=2)
        self.flat = nn.Flatten()
        self.batch_norm = nn.BatchNorm1d(c2 * 144)
        self.fc1 = nn.Linear(c2 * 144, l1)
        self.fc2 = nn.Linear(l1, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.dropout(x, self.d1)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.dropout(x, 0.5)
        x = self.flat(x)
        x = nn.functional.relu(self.batch_norm(x))
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

