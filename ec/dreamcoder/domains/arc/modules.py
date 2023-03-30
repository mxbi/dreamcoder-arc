import numpy as np
from numpy.random import default_rng
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, conv1_filters, conv2_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(conv1_filters, conv2_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.conv2 = nn.Conv2d(conv2_filters, conv2_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_filters)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)
        return out

class OneByOneBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, 1)
        self.bn = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class FC(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden=1, hidden_dim=528,
            batch_norm=False):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        for i in range(num_hidden-1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AllConv(nn.Module):
    def __init__(self, residual_blocks=1,
            input_filters=10,
            residual_filters=10,
            conv_1x1s=2,
            output_dim=128,
            conv_1x1_filters=128,
            pooling='max'):
        super().__init__()

        self.conv1 = nn.Conv2d(input_filters, residual_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(residual_filters)

        self.res_blocks = nn.ModuleList([ResBlock(residual_filters,
            residual_filters) for i in range(residual_blocks)])

        filters_1x1 = [residual_filters] + [conv_1x1_filters] * (conv_1x1s - 1) + [output_dim]

        self.conv_1x1s = nn.ModuleList([OneByOneBlock(filters_1x1[i],
            filters_1x1[i+1]) for i in range(conv_1x1s)])

        self.pooling = pooling

    def forward(self, x):
        """
            input: float tensor of shape (batch_size, n_filters, w, h)
            output: float tensor of shape (batch_size, output_dim)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        for block in self.res_blocks:
            x = block(x)

        for c in self.conv_1x1s:
            x = c(x)

        if self.pooling == 'max':
            x = F.max_pool2d(x, kernel_size=x.size()[2:])
        else:
            x = F.avg_pool2d(x, kernel_size=x.size()[2:])

        x = x.squeeze(3).squeeze(2)
        return x
