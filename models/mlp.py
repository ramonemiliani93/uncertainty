from abc import ABCMeta

import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, **kwargs):
        super(MLP, self).__init__()
        # Activations
        self._choices = nn.ModuleDict({
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid()
        })

        # Initialization parameters
        self.num_inputs = kwargs.get('num_inputs', 13)
        self.num_outputs = kwargs.get('num_outputs', 1)
        self.hidden_size = kwargs.get('hidden_size', 50)
        self.activation = kwargs.get('activation', 'sigmoid')

        # Optional parameters
        self.p = kwargs.get('p', 0)

        # One hidden layer mlp model
        self.model = nn.Sequential(
            nn.Linear(self.num_inputs, self.hidden_size),
            self._choices[self.activation],
            nn.Dropout(self.p),
            nn.Linear(self.hidden_size, self.num_outputs),
            nn.Dropout(self.p),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.model(inputs.float())
        return output


if __name__ == '__main__':
    mlp = MLP()
