import torch.nn as nn
from .mlp import MLP


class BaseModel(nn.Module):
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        self.trunk = MLP(layer_sizes[0], final_relu=True)
        self.embedder = MLP(layer_sizes[1])
        self.classifier = MLP(layer_sizes[2])
        self.abc_embedding = MLP(layer_sizes[3])
        self.abc_classifier = MLP(layer_sizes[4])

    def forward(self, x):
        x = self.trunk(x)
        x = self.embedder(x)
        return x

    def classify(self, x):
        return self.classifier(x)

    def abc_embedder(self, x):
        return self.abc_embedding(x)

    def abc_classify(self, x):
        return self.abc_classifier(x)


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
