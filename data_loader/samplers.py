import numpy as np
from torch.utils.data import Sampler

from data_loader.datasets import UncertaintyDataset


class LocalitySampler(Sampler):
    """
    Locality sampler as proposed by Nicki S. Detlefsen, Martin Jørgensen, and Søren Hauberg on their paper
    Reliable training and estimation of variance networks. The algorithm is:
        Input: N data points, a metric d on feature space R^D, integers m, n, k.
            1. For each data point calculate the k nearest neighbors under the metric d.
            2. Sample m primary sampling units with uniform probability without replacement among all N units.
            3. For each of the primary sampling units sample n secondary sampling units among the primary
               sampling units k nearest neighbors with uniform probability without replacement.
        Output: All secondary sampling units which is a sample of at most m · n points.
                If a new sample is needed repeat from Step 2
    """
    def __init__(self, data_source: UncertaintyDataset, neighbors: int, psu: int, ssu, **kwargs):
        """
        Args:
            data_source (UncertaintyDataset): Uncertainty dataset with implemented generate_neighbors function.
            neighbors (int): Number of nearest neighbors to get for each point (k).
            psu (int): Number of primary units to sample (m).
            ssu (int): Number of secondary units to sample (n).
        """
        super(LocalitySampler, self).__init__(data_source)
        self.data_source = data_source
        self.neighbors = neighbors
        self.psu = psu
        self.ssu = ssu
        self.neighbor_map = self.data_source.generate_neighbors(neighbors=self.neighbors, **kwargs)

    @property
    def neighbors(self) -> int:
        return self._neighbors

    @neighbors.setter
    def neighbors(self, value: int):
        if value <= 0:
            raise ValueError("Number of neighbors should be a positive integer.")
        self._neighbors = value

    @property
    def psu(self) -> int:
        return self._psu

    @psu.setter
    def psu(self, value: int):
        if value <= 0:
            raise ValueError("Number of primary sampling units should be a positive integer.")
        self._psu = value

    @property
    def ssu(self) -> int:
        return self._ssu

    @ssu.setter
    def ssu(self, value: int):
        if value <= 0:
            raise ValueError("Number of secondary sampling units should be a positive integer.")
        if value > self.neighbors:
            raise Warning("Number of secondary sampling units greater than number of neighbors.")
        self._ssu = value

    @property
    def neighbor_map(self) -> np.ndarray:
        return self._neighbor_map

    @neighbor_map.setter
    def neighbor_map(self, value: np.ndarray):
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError("Invalid neighbor map provided, must be an array of integers")
        if len(self.data_source) != len(value):
            raise ValueError("Invalid neighbor map provided, number of rows should match number of samples in dataset.")
        self._neighbor_map = value

    def __iter__(self):
        # Get total number of samples in the dataset and sample primary units without replacement.
        n = len(self.data_source)
        primary_units = np.random.choice(n, self.psu, replace=False)

        # Sample secondary units with replacement
        secondary_units = np.random.choice(self.neighbors, (self.psu, self.ssu), replace=True)

        # Modify sampled indices to generate (row, column) pairs to extract from neighbor map.
        primary_units = primary_units.repeat(self.ssu)
        secondary_units = secondary_units.ravel()

        # Extract neighbors from neighbor map using flat indices from the (row, columns) pairs.
        flat_indices = np.ravel_multi_index((primary_units, secondary_units), self.neighbor_map.shape)
        samples = self.neighbor_map.ravel()[flat_indices].tolist()

        return iter(samples)

    def __len__(self):
        return self.psu * self.ssu


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from data_loader.datasets import SineDataset

    # Create dataset
    dataset = SineDataset(500, (0, 10))

    # Implement data loader with Locality Sampler and extract a batch
    loader = DataLoader(dataset, batch_size=120, sampler=LocalitySampler(dataset, 50, 3, 40))

    # Extract batch
    sample_batch, target_batch = next(iter(loader))

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Extract all points and plot
    data = [x.numpy() for index in range(len(dataset)) for x in dataset[index]]
    ax.scatter(data[::2], data[1::2], s=8)

    # Plot locality sampled points
    ax.scatter(sample_batch.numpy(), target_batch.numpy(),  s=80, facecolors='none', edgecolors='r')

    plt.title('Sine scatter plot')
    plt.show()
