from typing import Tuple
from abc import abstractmethod

import torch
from torch import nn
from torch.nn.functional import mse_loss
from algorithms.base import UncertaintyAlgorithm
from helpers.functional import enable_dropout


class MonteCarloDropout(UncertaintyAlgorithm):

    def __init__(self, num_samples: int, p: float, model: nn.Module, **kwargs):
        self.num_samples = num_samples
        self.p = p
        self.model = model(**dict(p=self.p, **kwargs))

    def loss(self, *args, **kwargs) -> torch.Tensor:
        # Set model to train mode
        self.model.train()

        # Forward pass and MSE loss
        data, target = args
        prediction = self.model(data)
        mse = mse_loss(target, prediction)

        return mse

    def predict_with_uncertainty(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Set model to evaluation except dropout layers
        self.model.eval()
        enable_dropout(self.model)

        # Sample multiple times from the ensemble of models
        prediction = []
        with torch.no_grad():
            for i in range(self.num_samples):
                prediction.append(self.model(args[0]))

            # Calculate statistics of the outputs
            prediction = torch.stack(prediction)
            mean = prediction.mean(0).squeeze()
            var = prediction.var(0).squeeze().sqrt() #FIXME

        return mean, var


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    from data_loader.datasets import SineDataset
    from models.mlp import MLP

    algorithm = MonteCarloDropout(model=MLP, p=0.05, num_samples=10000)
    train_loader = DataLoader(SineDataset(500, (0, 10)), batch_size=500)
    optimizer = Adam(algorithm.model.parameters(), lr=1e-2, weight_decay=0)

    for epoch in range(10000):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = algorithm.loss(*data)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss))

    print('Finished Training')
    x = np.linspace(-4, 14, 5000)
    x_tensor = torch.FloatTensor(x).reshape(-1, 1)
    mean, std = algorithm.predict_with_uncertainty(x_tensor)

    plt.plot(x, mean.numpy(), '-', color='gray')
    plt.fill_between(x, mean.numpy() - 2 * std.numpy(), mean.numpy() + 2 * std.numpy(), color='gray', alpha=0.2)

    # Plot real function
    y = x * np.sin(x)
    plt.plot(x, y)

    plt.xlim(-4, 14)
    plt.show()
    print("Finished plotting")