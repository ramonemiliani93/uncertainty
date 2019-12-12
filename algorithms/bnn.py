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

    def loss(self, *args, **kwargs) -> torch.Tensor:
        pass

    def predict_with_uncertainty(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def __init__(self, **kwargs):
        super(BNN, self).__init__(**kwargs)
        self.dataset = kwargs.get('dataset')
        self.args = kwargs
        self.scaler = preprocessing.StandardScaler()

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
        w2 = numpyro.sample("w2", dist.Normal(np.zeros((d_h, d_h)), np.ones((d_h, d_h))))  # d_h d_h
        b2 = numpyro.sample("b2", dist.Normal(0, 1), sample_shape=(d_h,))
        z2 = b2 + nonlin(np.matmul(z1, w2))  # N d_h  <= second layer of activations

        # sample final layer of weights and neural network output
        w3 = numpyro.sample("w3", dist.Normal(np.zeros((d_h, d_y)), np.ones((d_h, d_y))))  # d_h D_Y
        z3 = np.matmul(z2, w3)  # N D_Y  <= output of the neural network

        # we put a prior on the observation noise
        prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
        sigma_obs = 1.0 / np.sqrt(prec_obs)

        # observe data
        numpyro.sample("y_train", dist.Normal(z3, sigma_obs), obs=y_train)

    def main(self):
        N, d_x, d_h = self.args['num_data'], 1, self.args['num_hidden']
        x_train, y_train = self.dataset.samples, self.dataset.targets
        x_test = np.linspace(-4, 14, 5000)
        x_train = np.expand_dims(x_train, -1)
        y_train = np.expand_dims(y_train, -1)
        x_test = np.expand_dims(x_test, -1)

        y_train = self.scaler.fit_transform(y_train)
        # do inference
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        samples = self.run_inference(self.args, rng_key, x_train, y_train, d_h)

        # predict Y_test at inputs x_test
        vmap_args = (samples, random.split(rng_key_predict, self.args['num_samples'] * self.args['num_chains']))
        predictions = vmap(lambda samples, rng_key: self.predict(self.model, rng_key, samples, x_test, d_h))(*vmap_args)
        predictions = predictions[..., 0]

        # compute mean prediction and confidence interval around median
        mean_prediction = np.mean(predictions, axis=0)
        var_prediction = np.var(predictions, axis=0)
        std_prediction = np.sqrt(var_prediction)

        # plot training data
        # plot mean prediction
        mean_prediction *= self.scaler.scale_
        mean_prediction += self.scaler.mean_

        y_train *= self.scaler.scale_
        y_train += self.scaler.mean_

        return mean_prediction, x_train, y_train, x_test, std_prediction

    def predict(self, model, rng_key, samples, x_train, d_h):
        model = handlers.substitute(handlers.seed(model, rng_key), samples)
        # note that y_train will be sampled in the model because we pass y_train=None here
        model_trace = handlers.trace(model).get_trace(x_train=x_train, y_train=None, d_h=d_h)
        return model_trace['y_train']['value']

    def run_inference(self, args, rng_key, x_train, y_train, d_h):
        if args['num_chains'] > 1:
            rng_key = random.split(rng_key, args['num_chains'])
        kernel = NUTS(self.model)
        mcmc = MCMC(kernel, args['num_warmup'], args['num_samples'], num_chains=args['num_chains'])
        mcmc.run(rng_key, x_train, y_train, d_h)
        return mcmc.get_samples()


if __name__ == '__main__':

    # BNN params
    bnn_params = {'num_chains': 1,
                  'num_data': 500,
                  'num_samples': 2000,
                  'num_warmup': 1000,
                  'num_hidden': 3,
                  'device': 'cpu'}
    # dataset params
    dataset_params = kwargs = {'num_samples': 500, 'domain': (0, 10)}
    dataset = SineDataset(**kwargs)

    # more bnn params
    bnn_params.update({'dataset': dataset})

    train_loader = DataLoader(dataset, batch_size=500)
    algorithm = BNN(**bnn_params)
    mean_prediction, x_train, y_train, x_test, std_prediction = algorithm.main()
    x_test = np.squeeze(x_test)
    mean = torch.from_numpy(onp.array(mean_prediction)).squeeze()
    std = torch.from_numpy(onp.array(std_prediction)).squeeze()

    plot_toy_uncertainty(x_test, mean, std, train_loader)
    x = 1