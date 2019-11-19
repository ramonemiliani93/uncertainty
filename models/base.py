from typing import Tuple
from abc import ABC, abstractmethod

import torch
from torch import nn


class UncertaintyModel(nn.Module, ABC):
    """Base model for NN"""

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of model.
        Args:
            inputs: Batches of data.
        Returns:
            predictions (torch.Tensor): Predictions of the net.
        """
        pass

    @abstractmethod
    def loss(self, *args, **kwargs) -> torch.Tensor:
        """
        Use forward pass of model and targets to predict loss that will be back propagated.
        Args:
            *args: Batches of data, targets as defined by the used dataset.
        Returns:
            loss (torch.Tensor): Error to be used in back propagation.
        """
        pass



