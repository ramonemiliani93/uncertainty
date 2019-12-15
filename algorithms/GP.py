import torch
import numpy as np
from algorithms.base import UncertaintyAlgorithm
import GPy


class GaussianProcess(UncertaintyAlgorithm):

    def __init__(self, **kwargs):
        super(GaussianProcess, self).__init__(**kwargs)

        self.dataset = kwargs.get('dataset')

    def predict_with_uncertainty(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:

        dim = self.dataset.samples.shape[1]
        X = self.dataset.samples
        y = self.dataset.targets

        kern = GPy.kern.RBF(dim, ARD=True)
        model = GPy.models.GPRegression(X, y, kern, normalizer=True)
        model.optimize()

        mu_test, cov_test = model.predict(args[0], full_cov=True)
        var_test = np.diag(cov_test).reshape(-1, 1)
        std_test = var_test ** (1/2)

        return mu_test, std_test

    @staticmethod
    def calculate_nll(target, mean, log_variance):
        # Estimate the negative log-likelihood. Here we estimate log of sigma squared for stability in training.
        nll = (log_variance / 2 + ((target - mean) ** 2) / (2 * torch.exp(log_variance))).mean()

        return nll

    def get_test_ll(self, y_test, mean_test, std_test):

        #x_test = self.dataset.features_test
        #y_test = self.dataset.targets_test
        #mean_test, std_test = self.predict_with_uncertainty(x_test)
        log_variance_test = (std_test**2).log()
        ll = -self.calculate_nll(y_test, mean_test, log_variance_test)
        return ll
