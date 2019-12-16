from typing import Tuple

import torch
from torch import nn
from torch.nn.functional import softplus

from algorithms.base import UncertaintyAlgorithm
from utils import plot_toy_uncertainty


class DeepEnsembles(UncertaintyAlgorithm):

    def __init__(self, **kwargs):
        super(DeepEnsembles, self).__init__(**kwargs)

        # Algorithm parameters
        self.num_models: int = 'num_models'
        self.warm_start_it: int = 'warm_start_it'

        # Create models
        model = kwargs.get('model')
        self.mean = nn.ModuleList([model(**kwargs) for _ in range(self.num_models)])
        self.variance = nn.ModuleList([model(**kwargs) for _ in range(self.num_models)])
        self.model = nn.ModuleDict({
            'mean': self.mean,
            'variance': self.variance
        })

        # Reserved params
        self._current_it = 0

    def loss(self, *args, **kwargs) -> torch.Tensor:
        # Extract data
        data, target, probability = args

        # Iterate through each model and sum nll
        nll = []
        for index in range(self.num_models):
            # Extract mean and variance from prediction
            if self._current_it < self.warm_start_it:
                predictive_mean = self.mean[index](data)
                with torch.no_grad():
                    predictive_variance = softplus(self.variance[index](data))
            else:
                with torch.no_grad():
                    predictive_mean = self.mean[index](data)
                predictive_variance = softplus(self.variance[index](data))

            # Calculate the loss
            nll.append(self.calculate_nll(target, predictive_mean, predictive_variance))

        mean_nll = torch.stack(nll).mean()

        # Update current iteration
        if self.training:
            self._current_it += 1

        return mean_nll

    def predict_with_uncertainty(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Set model to evaluation
        self.mean.eval()
        self.variance.eval()

        # Sample multiple times from the ensemble of models
        predictive_mean_list, predictive_variance_list = [], []
        with torch.no_grad():
            for i in range(self.num_models):
                predictive_mean = self.mean[i](args[0])
                predictive_mean_list.append(predictive_mean)
                predictive_variance = softplus(self.variance[i](args[0]))
                predictive_variance_list.append(predictive_variance)

            # Stack each of the models
            predictive_mean_ensemble = torch.stack(predictive_mean_list)
            predictive_variance_ensemble = torch.stack(predictive_variance_list)

            # Compute statistics
            predictive_mean_model = predictive_mean_ensemble.mean(0)
            predictive_variance_model = (predictive_variance_ensemble
                                         + predictive_mean_ensemble ** 2).mean(0) - predictive_mean_model ** 2
            predictive_std_model = predictive_variance_model.sqrt()

        return predictive_mean_model, predictive_std_model

    def save(self, path):
        state_dict = {}
        for index, model in enumerate(self.mean):
            state_dict['mean_{}'.format(index)] = model.state_dict()
        for index, model in enumerate(self.variance):
            state_dict['variance_{}'.format(index)] = model.state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        checkpoint = torch.load(path, map_location=device)
        for index, model in enumerate(self.mean):
            model.load_state_dict(checkpoint['mean_{}'.format(index)])
        for index, model in enumerate(self.variance):
            model.load_state_dict(checkpoint['variance_{}'.format(index)])

    @staticmethod
    def calculate_nll(target, mean, variance):
        # Estimate the negative log-likelihood. Here we estimate log of sigma squared for stability in training.
        variance = variance + 0.0001
        nll = (variance.log() / 2 + ((target - mean) ** 2) / (2 * variance)).mean()

        return nll


if __name__ == '__main__':
    import numpy as np
    from torch.optim import Adam
    from torch.utils.data import DataLoader

    from data_loader.datasets import SineDataset
    from models.mlp import MLP

    params = {'num_models': 1, 'model': MLP, 'warm_start_it': 5000}
    algorithm = DeepEnsembles(**params)
    algorithm.training = True

    params = {'num_samples': 500, 'domain': (0, 10)}
    train_loader = DataLoader(SineDataset(**params), batch_size=500)

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

    algorithm.training = False
    x = np.linspace(-4, 14, 5000)
    x_tensor = torch.FloatTensor(x).reshape(-1, 1)
    mean, std = algorithm.predict_with_uncertainty(x_tensor)
    plot_toy_uncertainty(x, mean.squeeze(), std.squeeze(), train_loader)