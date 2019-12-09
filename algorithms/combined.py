from typing import Tuple

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

        # Create models and register parameter
        model = kwargs.get('model')
        self.mean = model(**kwargs)
        self.alpha = model(**kwargs)
        self.beta = model(**kwargs)
        self.gamma = nn.Parameter(torch.tensor(self.init_gamma))
        self.model = nn.ModuleDict({
            'mean': self.mean,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma
        })

        # Reserved params
        self._current_it = 0

    def loss(self, *args, **kwargs) -> torch.Tensor:
        # Unpack input
        data, target, probability = args

        # Mean-variance split training
        if self._current_it % 2:
            mean = self.mean(data)
            with torch.no_grad():
                alpha = softplus(self.alpha(data))
                beta = softplus(self.beta(data))
        else:
            with torch.no_grad():
                mean = self.mean(data)
            alpha = softplus(self.alpha(data))
            beta = softplus(self.beta(data))


        return nll

    def predict_with_uncertainty(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Set model to evaluation
        self.mean.eval()
        self.log_variance.eval()

        # Sample multiple times from the ensemble of models
        predictive_mean_list, predictive_log_variance_list = [], []
        with torch.no_grad():
            for i in range(self.num_models):
                predictive_mean = self.mean[i](args[0])
                predictive_mean_list.append(predictive_mean)
                predictive_log_variance = self.log_variance[i](args[0])
                predictive_log_variance_list.append(predictive_log_variance)

            # Stack each of the models
            predictive_mean_ensemble = torch.stack(predictive_mean_list)
            predictive_log_variance_ensemble = torch.stack(predictive_log_variance_list)

            # Compute statistics
            predictive_mean_model = predictive_mean_ensemble.mean(0)
            # print(predictive_log_variance)
            predictive_variance_model = (torch.exp(predictive_log_variance_ensemble)
                                         + predictive_mean_ensemble ** 2).mean(0) - predictive_mean_model ** 2
            preditive_std_model = predictive_variance_model.sqrt()

        return predictive_mean_model, preditive_std_model

    def fgsm_attack(self, data, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()

        # Find max value of the data
        max_value = data.max()

        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_data = data + self.eps * max_value * sign_data_grad

        # Return the perturbed image
        return perturbed_data

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

    params = {'num_models': 1, 'eps': 0.01, 'adversarial': False, 'model': MLP, 'warm_start_it': 15000}
    algorithm = DeepEnsembles(**params)
    kwargs = {'num_samples': 500, 'domain': (0, 10)}
    train_loader = DataLoader(SineDataset(**kwargs), batch_size=500)
    optimizer = Adam(algorithm.model.parameters(), lr=1e-2, weight_decay=0)

    for epoch in range(10000):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = algorithm.loss(*data, warm_start_it=50000, it=epoch)
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
    std = np.sqrt(var)
    plot_toy_uncertainty(x, mean, std, train_loader)