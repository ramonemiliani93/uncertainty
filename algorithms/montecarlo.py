from typing import Tuple

import torch
from torch.nn.functional import mse_loss
from algorithms.base import UncertaintyAlgorithm

from helpers.functional import enable_dropout
from utils import plot_toy_uncertainty

import pdb


class MonteCarloDropout(UncertaintyAlgorithm):

    def __init__(self, **kwargs):
        super(MonteCarloDropout, self).__init__(**kwargs)

        # Algorithm parameters
        self.num_samples: int = 'num_samples'

        # Create model
        model = kwargs.get('model')
        self.model = model(**dict(**kwargs)).float()

    def loss(self, *args, **kwargs) -> torch.Tensor:
        # Forward pass and MSE loss
        data, target, probability = args
        prediction = self.model(data.float())
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
            mean = prediction.mean(0)
            std = prediction.var(0).sqrt()

        return mean, std

    @staticmethod
    def calculate_nll(target, mean, log_variance):
        # Estimate the negative log-likelihood. Here we estimate log of sigma squared for stability in training.
        log_two_pi_term = (torch.ones_like(mean, dtype=torch.float32) * np.pi * 2).log()
        nll = (log_variance / 2 + ((target - mean) ** 2) / (2 * torch.exp(log_variance)) + log_two_pi_term).mean()
        return nll

    @staticmethod
    def get_test_ll(y_test, mean_test, std_test):

        log_variance = (std_test**2).log()
        log_two_pi_term = (torch.ones_like(mean_test, dtype=torch.float32) * np.pi * 2).log()

        nll = (log_variance / 2 + ((y_test - mean_test) ** 2) / (2 * torch.exp(log_variance)) + log_two_pi_term).mean()
        #nll = (log_variance / 2 + ((y_test - mean_test) ** 2) / (2 * torch.exp(log_variance))).mean()
        nll_std = (log_variance / 2 + ((y_test - mean_test) ** 2) / (2 * torch.exp(log_variance))).std()
        nll_var = nll_std ** 2

        return - nll, nll_std, nll_var


if __name__ == '__main__':
    import numpy as np
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    #from dataloader.datasets import BostonDataset
    from dataloader.datasets import SineDataset
    from models.mlp import MLP

    algorithm = MonteCarloDropout(model=MLP, p=0.05, num_samples=10000)
    dict_params = {'num_samples': 500, 'domain': (0, 10)}
    train_loader = DataLoader(SineDataset(**dict_params), batch_size=500)
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
    plot_toy_uncertainty(x, mean, std, train_loader)



