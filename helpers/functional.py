import torch
from torch import nn


def enable_dropout(model: nn.Module):
    """Enable any dropout layer"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


class ScaledTranslatedSigmoid(nn.Module):
    def __init__(self, init_gamma, a):
        super(ScaledTranslatedSigmoid, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
        self.a = a

    def forward(self, x) -> torch.Tensor:
        return torch.sigmoid((x + (self.a * self.gamma)) / self.gamma)

