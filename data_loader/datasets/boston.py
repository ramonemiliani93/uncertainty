from typing import Tuple

import torch
import numpy as np
from annoy import AnnoyIndex
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from data_loader.datasets.base import UncertaintyDataset


class BostonDataset(UncertaintyDataset):
    def __init__(self, split="train"):
        super(BostonDataset, self).__init__()
        
        boston = load_boston()
        data = boston.data
        targets = boston.target

        features_train, features_test, targets_train, targets_test = train_test_split(data, targets, test_size=0.2)

        if split == "train":
            self.features = features_train
            self.targets = targets_train

        else:
            self.features = features_test
            self.targets = targets_test

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
