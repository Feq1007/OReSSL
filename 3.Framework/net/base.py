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
        # self.classifier = nn.Sequential(
        #     MLP(layer_sizes[2]),
        #     nn.Softmax()
        # )

    def forward(self, x):
        x = self.trunk(x)
        x = self.embedder(x)
        x = self.classifier(x)
        return x

class BaseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential()


class BaseNoModel(nn.Module):
    def __init__(self):
        super(BaseNoModel, self).__init__()

    def forward(self, x):
        return x

class NoModel(nn.Module):
    def __init__(self):
        super(NoModel, self).__init__()
        self.trunk = BaseNoModel()
        self.embedder = BaseNoModel()
        self.classifier = BaseNoModel()

    def forward(self, x):
        return x