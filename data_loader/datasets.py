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
        print(self.targets.shape)

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


class TwoMoonDataset(UncertaintyDataset):
    """ Two Moon Dataset

    """

    def __init__(self, num_samples: int = 500):
        """
        Args:
            num_samples: Number of samples to draw from the domain of the function.
        """
        super(TwoMoonDataset,  self).__init__()
        self.num_samples = num_samples
        self.samples = np.random.binomial(n=1, p=0.5, size=self.num_samples)
        self.z, self.v = self.function(self.samples)

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @num_samples.setter
    def num_samples(self, value: int):
        if value <= 0:
            raise ValueError("Number of samples has to be a positive integer.")
        self._num_samples = value

    def __getitem__(self, item):
        sample = torch.tensor(self.samples[item])
        z = torch.tensor(self.z[item])
        v = torch.tensor(self.v[item])

        # v1 = ((v[0], v[1]), (v[0], v[2]), (v[0], v[3]))
        # v2 = ((v[1], v[0]), (v[1], v[2]), (v[1], v[3]))
        # v3 = ((v[2], v[0]), (v[2], v[1]), (v[2], v[3]))
        # v4 = ((v[3], v[0]), (v[3], v[1]), (v[3], v[2]))
        v1 = ((v[1], v[0]), (v[2], v[0]), (v[3], v[0]))
        v2 = ((v[0], v[1]), (v[2], v[1]), (v[3], v[1]))
        v3 = ((v[0], v[2]), (v[1], v[2]), (v[3], v[2]))
        v4 = ((v[0], v[3]), (v[1], v[3]), (v[2], v[3]))

        return sample, z, v1, v2, v3, v4

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
    def sampler(u):
        if u == 1:
            c = [0.5, 0]
            alpha1 = np.random.uniform(*[0, np.pi], 1)[0]
        else:
            c = [-0.5, 0]
            alpha1 = np.random.uniform(*[np.pi, 2*np.pi], 1)[0]
        z = [c[0] + np.cos(alpha1), c[1] + np.sin(alpha1)]
        alpha2 = np.random.uniform(*[0, 2 * np.pi], 1)[0]
        mu = np.random.uniform(*[0, 1], 1)[0]
        z[0] += (mu / 4) * np.cos(alpha2)
        z[1] += (mu / 4) * np.sin(alpha2)

        return z

    @staticmethod
    def get_functions():
        def f1(x):
            return x[0] - x[1] + np.random.normal(1) * np.sqrt(0.03 + (0.05 * (3 + x[0])))

        def f2(x):
            return (x[0]**2) - (1/2) * x[1] + np.random.normal(1) * np.sqrt(0.03 + (0.03 * np.linalg.norm(x)))

        def f3(x):
            return x[0] * x[1] - x[0] + np.random.normal(1) * np.sqrt(0.03 + (0.05 * np.linalg.norm(x)))

        def f4(x):
            return x[0] + x[1] + np.random.normal(1) * np.sqrt(0.03 + (0.03 / (0.2 + np.linalg.norm(x))))

        return f1, f2, f3, f4

    def function(self, samples):
        z = np.array([self.sampler(u) for u in samples])
        f1, f2, f3, f4 = self.get_functions()
        v1 = np.array([f1(x) for x in z.tolist()])
        v2 = np.array([f2(x) for x in z.tolist()])
        v3 = np.array([f3(x) for x in z.tolist()])
        v4 = np.array([f4(x) for x in z.tolist()])
        points = np.array(list(zip(v1, v2, v3, v4)))

        return z, points


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create dataset
    dataset = TwoMoonDataset(500)

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Extract all points and plot
    data = [x.numpy() for index in range(len(dataset)) for x in dataset[index][1]]
    ax.scatter(data[::2], data[1::2])

    plt.title('Two Moon scatter plot (2d)')
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create dataset
    dataset = TwoMoonDataset(500)

    # Create plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # Extract all points and plot
    data = [x.numpy() for index in range(len(dataset)) for x in dataset[index][2][0]]
    ax1.scatter(data[::2], data[1::2])

    data = [x.numpy() for index in range(len(dataset)) for x in dataset[index][2][1]]
    ax2.scatter(data[::2], data[1::2])

    data = [x.numpy() for index in range(len(dataset)) for x in dataset[index][2][2]]
    ax3.scatter(data[::2], data[1::2])

    plt.title('Two Moon scatter plot v1 vs rest')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create dataset
    dataset = TwoMoonDataset(500)

    # Create plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # Extract all points and plot
    data = [x.numpy() for index in range(len(dataset)) for x in dataset[index][3][0]]
    ax1.scatter(data[::2], data[1::2])

    data = [x.numpy() for index in range(len(dataset)) for x in dataset[index][3][1]]
    ax2.scatter(data[::2], data[1::2])

    data = [x.numpy() for index in range(len(dataset)) for x in dataset[index][3][2]]
    ax3.scatter(data[::2], data[1::2])

    plt.title('Two Moon scatter plot v2 vs rest')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create dataset
    dataset = TwoMoonDataset(500)

    # Create plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # Extract all points and plot
    data = [x.numpy() for index in range(len(dataset)) for x in dataset[index][4][0]]
    ax1.scatter(data[::2], data[1::2])

    data = [x.numpy() for index in range(len(dataset)) for x in dataset[index][4][1]]
    ax2.scatter(data[::2], data[1::2])

    data = [x.numpy() for index in range(len(dataset)) for x in dataset[index][4][2]]
    ax3.scatter(data[::2], data[1::2])

    plt.title('Two Moon scatter plot (v3 vs rest)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create dataset
    dataset = TwoMoonDataset(500)

    # Create plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # Extract all points and plot
    data = [x.numpy() for index in range(len(dataset)) for x in dataset[index][5][0]]
    ax1.scatter(data[::2], data[1::2])

    data = [x.numpy() for index in range(len(dataset)) for x in dataset[index][5][1]]
    ax2.scatter(data[::2], data[1::2])

    data = [x.numpy() for index in range(len(dataset)) for x in dataset[index][5][2]]
    ax3.scatter(data[::2], data[1::2])

    plt.title('Two Moon scatter plot (v4 vs rest)')

    plt.tight_layout()
    plt.show()
