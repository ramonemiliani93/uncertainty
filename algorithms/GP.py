from typing import Tuple

import torch
import numpy as np
import math
from algorithms.base import UncertaintyAlgorithm
import GPy


class GaussianProcess(UncertaintyAlgorithm):

    def __init__(self, **kwargs):
        super(GaussianProcess, self).__init__(**kwargs)

        self.dataset = kwargs.get('dataset')

        # Create model
        model = kwargs.get('model')
        self.model = model(**dict(**kwargs)).float()

    def predict_with_uncertainty(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:

        dim = self.dataset.samples.shape[1]

        X = self.dataset.samples
        y = self.dataset.targets
        y = y.reshape(-1, 1)

        X_test = args[0].numpy()
        y_test = args[1].numpy()

        kern = GPy.kern.RBF(dim, ARD=True)
        model = GPy.models.GPRegression(X, y, kern, normalizer=True)
        model.optimize()

        mu_test, cov_test = model.predict(X_test, full_cov=True)
        var_test = np.diag(cov_test).reshape(-1, 1)
        std_test = var_test ** (1/2)

        print('rmse')
        print(y_test.reshape(-1, 1).shape)
        print(mu_test.reshape(-1, 1).shape)

        rmse = math.sqrt(((y_test.reshape(-1, 1)-mu_test.reshape(-1, 1))**2).mean())

        return torch.tensor(mu_test), torch.tensor(std_test), rmse

    def loss(self, *args, **kwargs) -> torch.Tensor:
        pass

    @staticmethod
    def calculate_nll(target, mean, log_variance):
        # Estimate the negative log-likelihood. Here we estimate log of sigma squared for stability in training.
        log_two_pi_term = 0.5*(torch.ones_like(mean, dtype=torch.float32) * np.pi * 2).log()
        nll = (log_variance / 2 + ((target - mean) ** 2) / (2 * torch.exp(log_variance)) + log_two_pi_term).mean()
        return nll

    @staticmethod
    def get_test_ll(y_test, mean_test, std_test):
        log_variance = (std_test ** 2).log()
        log_two_pi_term = 0.5*(torch.ones_like(mean_test, dtype=torch.float32) * np.pi * 2).log()

        nll = (log_variance / 2 + ((y_test - mean_test) ** 2) / (2 * torch.exp(log_variance)) + log_two_pi_term).mean()
        # nll = (log_variance / 2 + ((y_test - mean_test) ** 2) / (2 * torch.exp(log_variance))).mean()
        nll_std = (log_variance / 2 + ((y_test - mean_test) ** 2) / (2 * torch.exp(log_variance)) + log_two_pi_term).std()
        nll_var = nll_std ** 2

        return - nll, nll_std, nll_var
