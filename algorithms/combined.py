from typing import Tuple
from math import pi

from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.nn.functional import softplus

from algorithms import DeepEnsembles
from algorithms.base import UncertaintyAlgorithm
from helpers.functional import ScaledTranslatedSigmoid
from utils import plot_toy_uncertainty


class Combined(UncertaintyAlgorithm):

    def __init__(self, **kwargs):
        super(Combined, self).__init__(**kwargs)

        # Algorithm parameters
        self.init_gamma: float = 'init_gamma'
        self.a: float = 'a'
        self.num_inducing_points: float = 'num_inducing_points'
        self.eta: float = 'eta'
        self.switch_modulo: int = 'switch_modulo'
        self.warm_start_it: int = 'warm_start_it'

        # Create models and register parameter
        model = kwargs.get('model')
        self.mean = model(**kwargs)
        self.alpha = model(**kwargs)
        self.beta = model(**kwargs)
        self.st_sigmoid = ScaledTranslatedSigmoid(self.init_gamma, self.a)
        self.model = nn.ModuleDict({
            'mean': self.mean,
            'alpha': self.alpha,
            'beta': self.beta,
            'st_sigmoid': self.st_sigmoid
        })

        # Extract dataset and use inducing points to generate clusters
        dataset = kwargs.get('dataset')
        x_train = np.stack([dataset[index][0].numpy() for index in range(len(dataset))])
        k_means = KMeans(n_clusters=min(self.num_inducing_points, len(x_train))).fit(x_train)
        self.inducing_points = torch.tensor(k_means.cluster_centers_)

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

        # Bound the ratio of alpha and beta when out of distribution
        delta = torch.min(0)
        nll = self.reparametrized_nll_student_t(target, mean, alpha / beta, 2 * alpha)
        weighted_nll = (nll / probability).mean()

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
            alpha = softplus(self.alpha(args[0]))
            beta = softplus(self.beta(args[0]))
            variance = beta / alpha
            std = variance.sqrt()

        return mean, std

    @staticmethod
    def calculate_nll_student_t(target: torch.Tensor, mean: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        # Function derived on appendix B of the paper " Reliable training and estimation of variance networks" by
        # Nicki S. Detlefsen, Martin Jørgensen, and Søren Hauberg.
        mse = 0.5 * (target - mean) ** 2
        nll = - alpha * beta.log() + alpha.lgamma() - (alpha + 0.5).lgamma() + (alpha + 0.5) * (beta + mse).log()

        return nll

    @staticmethod
    def reparametrized_nll_student_t(target: torch.Tensor, mean: torch.Tensor, lamda: torch.Tensor, nu: torch.Tensor):
        mse = (target - mean) ** 2
        nll = -(nu / 2 + 0.5).lgamma() + (nu / 2).lgamma() - 0.5 * lamda.log() + 0.5 * (pi * nu).log() + (nu / 2 + 0.5) * (1 + lamda * mse / nu).log()

        return nll

if __name__ == '__main__':
    import numpy as np
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from data_loader.samplers import LocalitySampler
    import matplotlib.pyplot as plt
    from data_loader.datasets import SineDataset
    from models.mlp import MLP

    kwargs = {'num_samples': 500, 'domain': (0, 10)}
    dataset = SineDataset(**kwargs)
    params = {'init_gamma': 1.5,
              'a': -6.9077,
              'num_inducing_points': 500,
              'eta': 3.5 ** 2,
              'switch_modulo': 2,
              'model': MLP,
              'warm_start_it': 25000,
              'dataset': dataset
              }

    algorithm = Combined(**params)
    train_loader = DataLoader(dataset, batch_size=500, sampler=LocalitySampler(dataset, neighbors=30, psu=5, ssu=25))
    optimizer = Adam([
        {'params': algorithm.mean.parameters(), 'lr': 1e-2},
        {'params': algorithm.alpha.parameters(), 'lr': 1e-2},
        {'params': algorithm.beta.parameters(), 'lr': 1e-2}
    ], lr=1e-4, weight_decay=0)

    for epoch in range(50000):  # loop over the dataset multiple times

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
    mean, std = algorithm.predict_with_uncertainty(x_tensor)
    plot_toy_uncertainty(x, mean.squeeze(), std.squeeze(), train_loader)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.linspace(-4, 14, 5000)
    true_std = 0.3 * ((1 + x ** 2) ** 0.5)
    ax.plot(x, true_std, label='True std')
    ax.plot(x, std.squeeze())
    plt.title('plots')
    plt.show()