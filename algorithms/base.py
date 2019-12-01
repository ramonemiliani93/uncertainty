from typing import Tuple
from abc import ABC, abstractmethod

import torch


class UncertaintyAlgorithm(ABC):

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

    @abstractmethod
    def predict_with_uncertainty(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use forward pass of model to predict mean and variance.
        Args:
            *args: Batches of data (and targets) as defined by the used dataset.
        Returns:
            prediction (Tuple[torch.Tensor, torch.Tensor]): Mean and std of each measurement.
        """
        pass
