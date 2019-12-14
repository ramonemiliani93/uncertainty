from typing import Tuple

import torch
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .base import UncertaintyDataset


class WineWhiteDataset(UncertaintyDataset):
    def __init__(self, split="train"):
        super(WineWhiteDataset, self).__init__()

        wine_white = pd.read_csv("../data/winequality_white.csv", sep=';')
        data = wine_white.values[:, :-1]
        targets = wine_white.values[:, -1:]

        features_train, features_test, targets_train, targets_test = train_test_split(data, targets, test_size=0.2)

        scaler = StandardScaler()
        scaler.fit(features_train)
        features_train = scaler.transform(features_train)
        features_test = scaler.transform(features_test)

        if split == "train":
            self.samples = features_train
            self.targets = targets_train

        else:
            self.samples = features_test
            self.targets = targets_test

        assert len(self.samples) == len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = torch.tensor(self.samples[idx])
        target = torch.tensor(self.targets[idx])

        if self.probabilities is None:
            probability = torch.tensor([1])
        else:
            probability = torch.tensor(self.probabilities[idx])

        return sample, target, probability

    def __len__(self):
        return len(self.samples)

    def generate_neighbors(self, neighbors, **kwargs):
        # Extract parameters if provided in kwargs.
        dimension = 11  # Length of item vector that will be indexed
        metric = kwargs.get('metric', 'euclidean')
        num_trees = kwargs.get('num_trees', 10)

        # Build tree with the given data.
        t = AnnoyIndex(dimension, metric)
        for i in range(self.samples.shape[0]):
            t.add_item(i, [self.samples[i].item()])
        t.build(num_trees)

        # Generate neighbor map array.
        neighbor_map = np.zeros((self.samples.shape[0], neighbors))
        for i in range(self.samples.shape[0]):
            nearest_neighbors = t.get_nns_by_item(i, neighbors)
            neighbor_map[i, :] = nearest_neighbors

        self.neighbor_map = neighbor_map.astype(int)


