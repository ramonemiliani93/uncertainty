from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import Dataset


class UncertaintyDataset(Dataset, ABC):
    """Uncertainty dataset."""
    def __init__(self):
        super(UncertaintyDataset, self).__init__()
        self.neighbors: np.ndarray = None
        self.probabilities: np.ndarray = None

    def generate_probabilities(self, neighbors, psu, ssu):
        assert self.neighbors is not None, "Generate neighbors first"
        probabilities = np.zeros((len(self.neighbors, 1)))
        for i in range(len(self.neighbors)):
            probability = 0
            for j in range(len(self.neighbors)):
                probability += (ssu / neighbors) * np.isin(i, self.neighbors[j, :])
            probability *= psu / len(self.neighbors)
            probabilities[i, 0] = probability
        self.probabilities = probabilities

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def generate_neighbors(self, neighbors: int, **kwargs) -> np.ndarray:
        pass
