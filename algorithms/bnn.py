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
    return 1.0 / (1 + np.exp(-x))


class BNN(UncertaintyAlgorithm):

    def __init__(self, **kwargs):
        super(BNN, self).__init__(**kwargs)
        self.dataset = kwargs.get('dataset')
        model = kwargs.get('model')
        self.model = model(**kwargs)
        self.args = kwargs
        self.scaler = preprocessing.StandardScaler()
        x_test = kwargs.get('x_test')
        if len(x_test.shape) == 1:
            x_test = np.expand_dims(x_test, -1)
        self.x_test = x_test

    def loss(self, *args, **kwargs) -> torch.Tensor:
        x_train, y_train = self.dataset.samples, self.dataset.targets

        if len(x_train.shape) == 1:
            x_train = np.expand_dims(x_train, -1)
        if len(y_train.shape) == 1:
            y_train = np.expand_dims(y_train, -1)

        num_hidden = self.args['num_hidden']
        # rescale y
        y_train = self.scaler.fit_transform(y_train)

        # do inference
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        samples = self.run_inference(self.args, rng_key, x_train, y_train, num_hidden)

        # predict Y_test at inputs x_test
        # this shouldn't be here but I couldn't get it to work otherwise
        vmap_args = (samples, random.split(rng_key_predict, self.args['num_samples'] * self.args['num_chains']))
        predictions = vmap(lambda samples, rng_key: self.predict(self.bnn_model, rng_key, samples, self.x_test, num_hidden))(
            *vmap_args)
        self.predictions = predictions[..., 0]

        return torch.tensor([0.0], requires_grad=True)

    def predict_with_uncertainty(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:

        predictions = self.predictions

        # compute mean prediction and confidence interval around median
        mean_prediction = np.mean(predictions, axis=0)
        var_prediction = np.var(predictions, axis=0)
        std_prediction = np.sqrt(var_prediction)

        # rescale
        mean_prediction *= self.scaler.scale_
        mean_prediction += self.scaler.mean_

        # convert to numpy and then torch
        mean_prediction = onp.array(mean_prediction)
        std_prediction = onp.array(std_prediction)

        return torch.from_numpy(mean_prediction), torch.from_numpy(std_prediction)

    def bnn_model(self, x, y, num_hidden):
        d_x = x.shape[1]  # number of features in x
        d_y = 1  #
        # d_x, d_y = x_train.shape[1], 1

        # sample first layer (we put unit normal priors on all weights)
        w1 = numpyro.sample("w1", dist.Normal(np.zeros((d_x, num_hidden)), np.ones((d_x, num_hidden))))  # d_x d_h
        b1 = numpyro.sample("b1", dist.Normal(0, 1), sample_shape=(num_hidden,))
        z1 = nonlin(b1 + np.matmul(x, w1))
        # z1 = nonlin(np.matmul(x_train, w1))   # N d_h  <= first layer of activations

        # sample second layer
        w2 = numpyro.sample("w2", dist.Normal(np.zeros((num_hidden, d_y)), np.ones((num_hidden, d_y))))  # d_h d_h
        b2 = numpyro.sample("b2", dist.Normal(0, 1), sample_shape=(d_y,))
        z2 = b2 + np.matmul(z1, w2)  # N d_h  <= second layer of activations

        # sample final layer of weights and neural network output
        # w3 = numpyro.sample("w3", dist.Normal(np.zeros((d_h, d_y)), np.ones((d_h, d_y))))  # d_h D_Y
        # z3 = np.matmul(z2, w3)  # N D_Y  <= output of the neural network

        # we put a prior on the observation noise
        prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
        sigma_obs = 1.0 / np.sqrt(prec_obs)

        # observe data
        numpyro.sample("y", dist.Normal(z2, sigma_obs), obs=y)

    # THIS METHOD ONLY RUNS WHEN CALLING THE FILE AS A SCRIPT
    def main(self):

        x_train, y_train = self.dataset.samples, self.dataset.targets
        x_test = np.linspace(-4, 14, 5000)
        x_test = np.expand_dims(x_test, -1)

        if len(x_train.shape) == 1:
            x_train = np.expand_dims(x_train, -1)
        if len(y_train.shape) == 1:
            y_train = np.expand_dims(y_train, -1)

        num_hidden = self.args['num_hidden']
        # rescale y
        y_train = self.scaler.fit_transform(y_train)

        # do inference
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        samples = self.run_inference(self.args, rng_key, x_train, y_train, num_hidden)

        # predict Y_test at inputs x_test
        vmap_args = (samples, random.split(rng_key_predict, self.args['num_samples'] * self.args['num_chains']))
        predictions = vmap(lambda samples, rng_key: self.predict(self.bnn_model, rng_key, samples, x_test, num_hidden))(*vmap_args)
        predictions = predictions[..., 0]

        # compute mean prediction and confidence interval around median
        mean_prediction = np.mean(predictions, axis=0)
        var_prediction = np.var(predictions, axis=0)
        std_prediction = np.sqrt(var_prediction)

        # rescale
        mean_prediction *= self.scaler.scale_
        mean_prediction += self.scaler.mean_

        return mean_prediction, x_test, std_prediction

    @staticmethod
    def predict(bnn_model, rng_key, samples, x, num_hidden):
        bnn_model = handlers.substitute(handlers.seed(bnn_model, rng_key), samples)
        # note that y will be sampled in the bnn_model because we pass y=None here
        model_trace = handlers.trace(bnn_model).get_trace(x=x, y=None, num_hidden=num_hidden)
        return model_trace['y']['value']

    def run_inference(self, args, rng_key, x_train, y_train, num_hidden):
        if args['num_chains'] > 1:
            rng_key = random.split(rng_key, args['num_chains'])
        kernel = NUTS(self.bnn_model)
        mcmc = MCMC(kernel, args['num_warmup'], args['num_samples'], num_chains=args['num_chains'])
        mcmc.run(rng_key, x_train, y_train, num_hidden)
        return mcmc.get_samples()


if __name__ == '__main__':

    # BNN params
    bnn_params = {'num_chains': 1,
                  'num_samples': 2000,
                  'num_warmup': 1000,
                  'num_hidden': 50,
                  'device': 'cpu'}

    # dataset params
    dataset_params = kwargs = {'num_samples': 500, 'domain': (0, 10)}
    dataset = SineDataset(**kwargs)

    # more bnn params
    bnn_params.update({'dataset': dataset})

    train_loader = DataLoader(dataset, batch_size=500)
    algorithm = BNN(**bnn_params)

    mean, x_test, std = algorithm.main()

    x_test = np.squeeze(x_test)
    mean = torch.from_numpy(onp.array(mean)).squeeze()
    std = torch.from_numpy(onp.array(std)).squeeze()
    plot_toy_uncertainty(x_test, mean, std, train_loader)