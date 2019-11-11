from typing import Tuple
from abc import ABC, abstractmethod

from annoy import AnnoyIndex
import numpy as np
import torch
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


class SineDataset(UncertaintyDataset):
    """ Sine function dataset given by:
        y = x · sin(x) + 0.3 · eps_1 + 0.3 · x · eps_2 , with  eps_1, eps_2 ∼ N(0,1)

    """

    def __init__(self, num_samples: int, domain: Tuple[float, float]):
        """
        Args:
            num_samples: Number of samples to draw from the domain of the function.
            domain: X range of the data to be generated.
        """
        super(SineDataset,  self).__init__()
        self.num_samples = num_samples
        self.domain = domain
        self.samples = np.random.uniform(*self.domain, self.num_samples)
        self.targets = self.function(self.samples)

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @num_samples.setter
    def num_samples(self, value: int):
        if value <= 0:
            raise ValueError("Number of samples has to be a positive integer.")
        self._num_samples = value

    @property
    def domain(self) -> Tuple[float, float]:
        return self._domain

    @domain.setter
    def domain(self, value: Tuple[float, float]):
        low, high = value
        if high <= low:
            raise ValueError("Invalid domain specified.")
        self._domain = low, high

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = torch.tensor(self.samples[item])
        target = torch.tensor(self.targets[item])

        return sample, target

    def __len__(self) -> int:
        return self.num_samples

    def generate_neighbors(self, neighbors, **kwargs):
        # Extract parameters if provided in kwargs.
        dimension = 1  # Length of item vector that will be indexed
        metric = kwargs.get('metric', 'euclidean')
        num_trees = kwargs.get('num_trees', 10)

        # Build tree with the given data.
        t = AnnoyIndex(dimension, metric)
        for i in range(self.num_samples):
            t.add_item(i, [self.samples[i].item()])
        t.build(num_trees)

        # Generate neighbor map array.
        neighbor_map = np.zeros((self.num_samples, neighbors))
        for i in range(self.num_samples):
            nearest_neighbors = t.get_nns_by_item(i, neighbors)
            neighbor_map[i, :] = nearest_neighbors

        return neighbor_map.astype(int)

    @staticmethod
    def function(x: np.ndarray) -> np.ndarray:
        eps_1 = np.random.normal(loc=0, scale=1, size=(len(x)))
        eps_2 = np.random.normal(loc=0, scale=1, size=(len(x)))
        y = x * np.sin(x) + 0.3 * eps_1 + 0.3 * x * eps_2

        return y


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create dataset
    dataset = SineDataset(500, (0, 10))

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Extract all points and plot
    data = [x.numpy() for index in range(len(dataset)) for x in dataset[index]]
    ax.scatter(data[::2], data[1::2])

    plt.title('Sine scatter plot')
    plt.show()
