from typing import Tuple

import torch
from torch import nn

from algorithms.base import UncertaintyAlgorithm
from utils import plot_toy_uncertainty


class DeepEnsembles(UncertaintyAlgorithm):

    def __init__(self, **kwargs):
        super(DeepEnsembles, self).__init__(**kwargs)

        # Algorithm parameters
        self.num_models: int = 'num_models'
        self.eps: float = 'eps'
        self.adversarial: bool = 'adversarial'
        self.warm_start_it: int = 'warm_start_it'

        # Create models
        model = kwargs.get('model')
        self.mean = nn.ModuleList([model(**kwargs) for _ in range(self.num_models)])
        self.log_variance = nn.ModuleList([model(**kwargs) for _ in range(self.num_models)])
        self.model = nn.ModuleDict({
            'mean': self.mean,
            'log_variance': self.log_variance
        })

        # Reserved params
        self._current_it = 0

    def loss(self, *args, **kwargs) -> torch.Tensor:
        # Sample one model from the ensemble (workaround to the fact that models should be trained in parallel with
        # different data order, will take longer to converge to same point.)
        index = np.random.randint(0, self.num_models)
        mean_model = self.mean[index]
        log_variance_model = self.log_variance[index]

        # Set model to train mode
        mean_model.train()
        log_variance_model.train()

        # Extract data and set data gradient to true for use in th FGSA
        data, target = args

        if self.adversarial:
            data.requires_grad = True

        # Extract mean and variance from prediction
        predictive_mean = mean_model(data)
        predictive_log_variance = log_variance_model(data)

        # Calculate the loss
        if self._current_it < self.warm_start_it:
            nll = self.calculate_nll(target, predictive_mean, (torch.ones_like(predictive_mean) * 0.001).log())
        else:
            nll = self.calculate_nll(target, predictive_mean, predictive_log_variance)

        # If adversarial training enabled generate sample
        if self.adversarial:  # TODO
            # Calculate gradients of model in backward pass
            nll.backward()

            # Collect data gradients
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = self.fgsm_attack(data, data_grad)
            
            # Forward pass
            prediction = model(perturbed_data)
            mean = prediction[:, 0::2]
            variance = prediction[:, 1::2]
            
            # Add to loss
            nll += self.calculate_nll(target, mean, predictive_log_variance)
            
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
    def calculate_nll(target, mean, log_variance):
        # Estimate the negative log-likelihood. Here we estimate log of sigma squared for stability in training.
        nll = (log_variance / 2 + ((target - mean) ** 2) / (2 * torch.exp(log_variance))).mean()

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