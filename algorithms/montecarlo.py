from typing import Tuple

import torch
from torch.nn.functional import mse_loss
from algorithms.base import UncertaintyAlgorithm
import numpy as np
from helpers.functional import enable_dropout
from utils import plot_toy_uncertainty


class MonteCarloDropout(UncertaintyAlgorithm):

    def __init__(self, **kwargs):
        super(MonteCarloDropout, self).__init__(**kwargs)

        # Algorithm parameters
        self.num_samples: int = 'num_samples'

        # Create model
        model = kwargs.get('model')
        self.model = model(**dict(**kwargs)).float()
        self.dataset = kwargs.get('dataset')

    def loss(self, *args, **kwargs) -> torch.Tensor:
        # Forward pass and MSE loss
        data, target, probability = args
        prediction = self.model(data.float())
        mse = mse_loss(target, prediction)

        return mse

    @staticmethod
    def calculate_nll(target, mean, log_variance):
        # Estimate the negative log-likelihood. Here we estimate log of sigma squared for stability in training.
        log_two_pi_term = (torch.ones_like(mean, dtype=torch.float32) * np.pi * 2).log()
        nll = (log_variance / 2 + ((target - mean) ** 2) / (2 * torch.exp(log_variance)) + log_two_pi_term).mean()
        return nll

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

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def get_test_ll(self):

        x_test = self.dataset.features_test
        y_test = self.dataset.targets_test
        mean_test, std_test = self.predict_with_uncertainty(x_test)
        log_variance_test = (std_test**2).log()
        ll = -self.calculate_nll(y_test, mean_test, log_variance_test)
        return ll


if __name__ == '__main__':
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    # from data_loader.datasets import BostonDataset
    from data_loader.datasets import SineDataset
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



