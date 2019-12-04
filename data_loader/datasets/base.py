from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import Dataset


class UncertaintyDataset(Dataset, ABC):
    """Uncertainty dataset."""
    def __init__(self):
        super(UncertaintyDataset, self).__init__()

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def generate_neighbors(self, neighbors: int, **kwargs) -> np.ndarray:
        pass
