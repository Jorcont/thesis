from symtable import Class

import torch
from torch import nn

class CNN(nn.Module):

    def __init__(self, pool: bool = True):
        super().__init__()
        self.pooling = pool
        self.conv1_1ch = nn.Conv2d(1,32,3,padding=1)
        self.conv1_3ch = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.norm1 =nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 64)
        self.norm3 = nn.GroupNorm(8, 128)
        self.pool = nn.MaxPool2d(2, 2)
        print("pool:!", self.pooling)
        if self.pooling:
            self.gap = nn.AdaptiveAvgPool2d(1)
        else:
            self.gap = nn.Identity()
        self.name = "SmallCNN"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = self.conv1_1ch(x)
        else:
            x = self.conv1_3ch(x)

        x = self.norm1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.pool(x)
        x = self.gap(x)
        return x

