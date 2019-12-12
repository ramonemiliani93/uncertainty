from typing import Tuple

import torch
from torch.utils.data import DataLoader

from algorithms.base import UncertaintyAlgorithm
from jax import vmap
import jax.numpy as np
import jax.random as random

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC

from data_loader import SineDataset
from sklearn import preprocessing

from utils import plot_toy_uncertainty
import numpy as onp


def nonlin(x):
    return np.tanh(x)


class BNN(UncertaintyAlgorithm):

    def __init__(self, **kwargs):
        super(BNN, self).__init__(**kwargs)
        dataset = kwargs.get('dataset')

        # bnn parameters
        self.hidden_neurons = kwargs.get('num_hidden')
        self.num_chains = kwargs.get('num_chains')
        self.num_warmup = kwargs.get('num_warmup')
        self.num_samples = kwargs.get('num_samples')
        self.rng_key = kwargs.get('rng_key')
        self.rng_key_predict = kwargs.get('rng_key_predict')
        # duh
        self.scaler = preprocessing.StandardScaler()

        self.x_train = np.array(dataset.samples.reshape(-1, 1))
        self.y_train = dataset.targets.reshape(-1, 1)
        self.y_train = np.array(self.scaler.fit_transform(self.y_train))

        self.d_x = self.x_train.shape[1]
        self.d_y = self.y_train.shape[1]

    def model(self, x_train, y_train, d_h):
        d_x = x_train.shape[1]
        d_y = x_train.shape[1]
        # d_x, d_y = x_train.shape[1], 1

        # sample first layer (we put unit normal priors on all weights)
        w1 = numpyro.sample("w1", dist.Normal(np.zeros((d_x, d_h)), np.ones((d_x, d_h))))  # d_x d_h
        b1 = numpyro.sample("b1", dist.Normal(0, 1), sample_shape=(d_h,))
        z1 = b1 + nonlin(np.matmul(x_train, w1))
        # z1 = nonlin(np.matmul(x_train, w1))   # N d_h  <= first layer of activations

        # sample second layer
        w2 = numpyro.sample("w2", dist.Normal(np.zeros((d_h, d_h)), np.ones((d_h, d_y))))  # d_h d_h
        b2 = numpyro.sample("b2", dist.Normal(0, 1), sample_shape=(d_h,))
        z2 = b2 + nonlin(np.matmul(z1, w2))  # N d_h  <= second layer of activations

        # sample final layer of weights and neural network output
        # w3 = numpyro.sample("w3", dist.Normal(np.zeros((d_h, d_y)), np.ones((d_h, d_y))))  # d_h d_y
        # z3 = np.matmul(z2, w3)  # N d_y  <= output of the neural network

        # we put a prior on the observation noise
        prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
        sigma_obs = 1.0 / np.sqrt(prec_obs)

        # observe data
        numpyro.sample("y_train", dist.Normal(z2, sigma_obs), obs=y_train)

    def predict(self, *args, **kwargs):
        samples, x = args
        model = handlers.substitute(handlers.seed(self.model, self.rng_key), samples)
        # note that y_train will be sampled in the model because we pass y_train=None here
        model_trace = handlers.trace(model).get_trace(x_train=x, y_train=None, d_h=self.hidden_neurons)
        return model_trace['y_train']['value']

    def run_inference(self, *args, **kwargs) -> torch.Tensor:
        if self.num_chains > 1:
            rng_key = random.split(self.rng_key, self.num_chains)
        else:
            rng_key = self.rng_key
        kernel = HMC(self.model)
        mcmc = MCMC(kernel, self.num_warmup, self.num_samples, num_chains=self.num_chains)
        mcmc.run(rng_key, self.x_train, self.y_train, self.hidden_neurons)
        return mcmc.get_samples()

    def predict_with_uncertainty(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # predict Y_test at inputs x_test
        samples, x_test = args
        vmap_args = (samples, random.split(self.rng_key_predict, self.num_samples * self.num_chains))
        predictions = vmap(lambda samples, rng_key: self.predict(samples, x_test))(*vmap_args)
        predictions = predictions[..., 0]
        mean_prediction = np.mean(predictions, axis=0)
        var_prediction = np.var(predictions, axis=0)
        std = np.sqrt(var_prediction)
        return mean_prediction, std

    def loss(self, *args, **kwargs):
        rng_key, rng_key_predict = args
        samples = self.run_inference(self.model, rng_key)
        return samples


if __name__ == '__main__':

    bnn_params = {'num_chains': 1,
                  'num_samples': 2000,
                  'num_warmup': 5000,
                  'num_hidden': 50,
                  'device': 'cpu'}

    dataset_params = kwargs = {'num_samples': 500, 'domain': (0, 10)}
    dataset = SineDataset(**kwargs)
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    bnn_params.update({'rng_key': rng_key,
                       'rng_key_predict': rng_key_predict,
                       'dataset': dataset})
    train_loader = DataLoader(dataset, batch_size=500)
    algorithm = BNN(**bnn_params)
    numpyro.set_platform(bnn_params['device'])
    numpyro.set_host_device_count(bnn_params['num_chains'])

    samples = algorithm.run_inference()
    x_test = np.linspace(-4, 14, 5000)
    x_test = x_test.reshape(-1, 1)
    mean, std = algorithm.predict_with_uncertainty(samples, x_test)
    mean = algorithm.scaler.inverse_transform(onp.array(mean))
    x_test = np.squeeze(x_test)
    mean = torch.tensor(mean).squeeze()
    std = torch.tensor(onp.array(std)).squeeze()
    plot_toy_uncertainty(x_test, mean, std, train_loader)