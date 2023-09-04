import torch
import torch.nn as nn

import numpy as np
import scipy

from models.pe import PE

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, width, num_layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, width))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(width, output_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.layers(x)
        out = torch.sigmoid(out)
        return out

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.pe = PE(num_res=10)
        self.mlp = MLP(42, 3, 256, 9)
    def forward(self, x):
        out = self.pe(x)
        out = self.mlp(out)
        return out

class NERF_W(nn.Module):
    def __init__(self, dataset, imageSize, embeddingSize = 20, useZ = True): # Assume image size is a square
        super(NERF_W, self).__init__()
        self.useZ = useZ

        self.Z = nn.Parameter( torch.zeros( len( dataset. images ), embeddingSize ) ) # requires_grad = True by default.

        self.pe = PE(num_res=10)

        if (self.useZ):
            self.mlp = MLP(42 + embeddingSize, 3, imageSize[0], 9)
        else:
            self.mlp = MLP(42, 3, imageSize[0], 9)

    def forward(self, x, index, interpolate = False, interpWeight = 0.5):
        out = self.pe(x)

        if (self.useZ):
            if (interpolate):
                Z = torch.lerp(torch.min(self.Z), torch.max(self.Z), interpWeight)
                Z = Z.repeat([index.size()[0], 1])
            else:
                Z = self.Z[index]
            out = torch.cat((out, Z), dim=-1)

        out = self.mlp(out)
        return out
