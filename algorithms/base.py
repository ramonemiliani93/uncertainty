from typing import Tuple
from abc import ABC, abstractmethod

import torch


class UncertaintyAlgorithm(ABC):
    """Base class for uncertainty algorithms"""
    def __init__(self, **kwargs):
        self.__dict__['kwargs'] = kwargs

    def __setattr__(self, key, value):
        """All str attributes are considered parameters to the algorithm"""
        if isinstance(value, str):
            if value in self.kwargs:
                self.__dict__[key] = self.kwargs.get(value)
            else:
                raise AttributeError('Necessary parameter {} not defined in configuration file.'.format(value))
        else:
            self.__dict__[key] = value

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
