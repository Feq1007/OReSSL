import torch
import torch.nn as nn
import torch.nn.functional as F
from net.mlp import MLP

class BaseModel(nn.Module):
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        self.trunk = MLP(layer_sizes[0])
        self.embedder = MLP(layer_sizes[1])
        self.classifier = MLP(layer_sizes[2])

    def forward(self, x):
        x = self.trunk(x)
        x = self.embedder(x)
        x = self.classifier(x)
        return x