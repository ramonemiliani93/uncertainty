import argparse
import os
import os.path as osp
import numpy as np

import torch.nn

import utils
from utils import instantiate, plot_toy_weather
from data_loader.datasets.weather import WeatherDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='experiments/toy_regression/montecarlo',
                        help="Directory containing params.yml")
    args = parser.parse_args()

    yaml_path = os.path.join('params.yml')
    assert os.path.isfile(yaml_path), "No yaml configuration file found at {}".format(yaml_path)
    params = utils.Params(yaml_path)

    # Instantiate algorithm
    algorithm_module, algorithm_name = params.algorithm['module'], params.algorithm['name']
    algorithm = instantiate(algorithm_module, algorithm_name)
    algorithm_params = params.algorithm['params']

    path = osp.join("saved_model/model.pth")
    algorithm = torch.load(path)

    os.chdir('../../..')

    # Instantiate dataset
    dataset = WeatherDataset()

    X, y, sigma_train = dataset.samples, dataset.targets, dataset.var
    X, y = torch.FloatTensor(X).reshape(-1, 1), torch.FloatTensor(y).reshape(-1, 1)

    size = (X.max() - X.min()) / 50
    x = np.linspace(X.min() - size, X.max() + size, 500).reshape(-1, 1)
    x_tensor = torch.FloatTensor(x)
    mean, sigma = algorithm.predict_with_uncertainty(x_tensor)
    mean, sigma = mean.reshape(-1).numpy(), sigma.reshape(-1).numpy()

    # sigma_train_estimate is an estimate of the std (sigma_train) on the train set (X)
    _, sigma_train_estimate = algorithm.predict_with_uncertainty(X)
    sigma_train_estimate = sigma_train_estimate.reshape(-1).numpy()

    error = np.mean(np.abs(sigma_train - sigma_train_estimate)).round(2)

    # Start plotting
    model_name = args.model_dir.split('/')[-1]
    plot_toy_weather(X, y, x.reshape(-1, 1), mean, sigma, dataset.mu, dataset.var, model_name, error)
