from typing import Tuple
from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseModel(nn.Module, ABC):

    @abstractmethod
    def loss(self, *args) -> torch.Tensor:
        """
        Use forward pass of model and targets to predict loss that will be back propagated.
        Args:
            *args: Batches of data, targets as defined by the used dataset.
        Returns:
            loss (torch.Tensor): Error to be used in back propagation.
        """
        pass

    @abstractmethod
    def predict_uncertainty(self, *args) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use forward pass of model to predict mean and variance.
        Args:
            *args: Batches of data (and targets) as defined by the used dataset.
        Returns:
            loss (Tuple[torch.Tensor, torch.Tensor]): Mean and variance of each measure.
        """
        pass




