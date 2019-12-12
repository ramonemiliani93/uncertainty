from typing import Tuple

import torch
import numpy as np
import pandas as pd
from annoy import AnnoyIndex

from .base import UncertaintyDataset


class ConcreteDataset(UncertaintyDataset):
    def __init__(self):
        super(ConcreteDataset, self).__init__()

        concrete = pd.read_excel(r"../data/concrete_data.xls")
        self.features = concrete.values[:, :-1]
        self.targets = concrete.values[:, -1:]

        assert len(self.features) == len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = torch.tensor(self.features[idx])
        target = torch.tensor(self.targets[idx])

        if self.probabilities is None:
            probability = torch.tensor([1])
        else:
            probability = torch.tensor(self.probabilities[idx])

        return sample, target, probability

    def __len__(self):
        return len(self.features)

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