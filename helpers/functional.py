from torch import nn


def enable_dropout(model: nn.Module):
    """Enable any dropout layer"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
