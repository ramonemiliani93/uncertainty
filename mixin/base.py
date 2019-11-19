from typing import Tuple
from abc import ABC, abstractmethod

import torch


class UncertaintyMixin(ABC):

    @abstractmethod
    def predict_with_uncertainty(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use forward pass of model to predict mean and variance.
        Args:
            *args: Batches of data (and targets) as defined by the used dataset.
        Returns:
            prediction (Tuple[torch.Tensor, torch.Tensor]): Mean and variance of each measure.
        """
        pass
