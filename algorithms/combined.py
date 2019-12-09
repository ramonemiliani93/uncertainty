from typing import Tuple

from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.nn.functional import softplus

from algorithms.base import UncertaintyAlgorithm
from utils import plot_toy_uncertainty


class Combined(UncertaintyAlgorithm):

    def __init__(self, **kwargs):
        super(Combined, self).__init__(**kwargs)

        # Algorithm parameters
        self.init_gamma: int = 'init_gamma'
        self.inducing_points: float = 'inducing_points'
        self.eta: float = 'eta'
        self.switch_modulo: int = 'switch_modulo'
        self.warm_start_it: int = 'warm_start_it'

        # Create models and register parameter
        model = kwargs.get('model')
        self.mean = model(**kwargs)
        self.alpha = model(**kwargs)
        self.beta = model(**kwargs)
        # self.gamma = nn.Parameter(torch.tensor(self.init_gamma))
        self.model = nn.ModuleDict({
            'mean': self.mean,
            'alpha': self.alpha,
            'beta': self.beta,
            # 'gamma': self.gamma
        })

        # Extract dataset and use inducing points to generate clusters
        # dataset = kwargs.get('dataset')
        # _ = [dataset[index][0] for index in range(len(dataset))]

        # Reserved params
        self._current_it = 0

    def loss(self, *args, **kwargs) -> torch.Tensor:
        # Unpack input
        data, target, probability = args

        # Mean-variance split training
        if self._current_it % self.switch_modulo or self._current_it < self.warm_start_it:
            mean = self.mean(data)
            with torch.no_grad():
                alpha = softplus(self.alpha(data))
                beta = softplus(self.beta(data))
        else:
            with torch.no_grad():
                mean = self.mean(data)
            alpha = softplus(self.alpha(data))
            beta = softplus(self.beta(data))

        nll = self.calculate_nll_student_t(target, mean, alpha, beta)
        weighted_nll = (nll * probability).mean()

        # Update current iteration
        self._current_it += 1

        return weighted_nll

    def predict_with_uncertainty(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Set model to evaluation
        self.mean.eval()
        self.alpha.eval()
        self.beta.eval()

        # Sample multiple times from the ensemble of models
        with torch.no_grad():
            mean = self.mean(args[0])
            alpha = self.alpha(args[0])
            beta = self.alpha(args[0])
            variance = beta / alpha

        return mean, variance

    @staticmethod
    def calculate_nll_student_t(target: torch.Tensor, mean: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        # Function derived on appendix B of the paper " Reliable training and estimation of variance networks" by
        # Nicki S. Detlefsen, Martin Jørgensen, and Søren Hauberg.
        mse = 0.5 * (target - mean) ** 2
        nll = - alpha * beta.log() + alpha.lgamma() - (alpha + 0.5).lgamma() + (alpha + 0.5) * (beta + mse).log()

        return nll


if __name__ == '__main__':
    import numpy as np
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    from data_loader.datasets import SineDataset
    from models.mlp import MLP
    params = {'init_gamma': 1.5,
              'inducing_points': 500,
              'eta': 3.5 ** 2,
              'switch_modulo': 2,
              'model': MLP,
              'warm_start_it': 0
              }

    algorithm = Combined(**params)
    kwargs = {'num_samples': 500, 'domain': (0, 10)}
    train_loader = DataLoader(SineDataset(**kwargs), batch_size=500)
    optimizer = Adam(algorithm.model.parameters(), lr=1e-2, weight_decay=0)

    for epoch in range(10000):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = algorithm.loss(*data, it=epoch)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss))

    print('Finished Training')

    x = np.linspace(-4, 14, 5000)
    x_tensor = torch.FloatTensor(x).reshape(-1, 1)
    mean, var = algorithm.predict_with_uncertainty(x_tensor)
    std = np.sqrt(var.squeeze())
    plot_toy_uncertainty(x, mean.squeeze(), std, train_loader)