import argparse
import os
import os.path as osp
import numpy as np

import torch.nn

import utils
from utils import instantiate, plot_toy_weather


def run_toy_weather_experiment(algorithm, weather_dataset, model_name):
    # num_samples = len(weather_dataset.std)
    X = weather_dataset.data.index[:].dayofyear
    y = weather_dataset.data.iloc[:, 0]
    sigma_train = np.array(weather_dataset.std)
    mu_train = np.array(weather_dataset.mean)
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
    # error = 0
    # Start plotting
    plot_toy_weather(X, y, x.reshape(-1, 1), mean, sigma, mu_train, sigma_train, model_name, error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='experiments/toy_regression/montecarlo',
                        help="Directory containing params.yml")
    args = parser.parse_args()

    yaml_path = os.path.join(args.model_dir, 'params.yml')
    assert os.path.isfile(yaml_path), "No yaml configuration file found at {}".format(yaml_path)
    params = utils.Params(yaml_path)

    # Instantiate algorithm
    algorithm_module, algorithm_name = params.algorithm['module'], params.algorithm['name']
    algorithm = instantiate(algorithm_module, algorithm_name)
    algorithm_params = params.algorithm['params']

    # Instantiate model
    model_module, model_name = params.model['module'], params.model['name']
    model = instantiate(model_module, model_name)

    # algorithm = torch.load(path)

    # Instantiate dataset
    dataset_module, dataset_name = params.dataset['module'], params.dataset['name']
    dataset = instantiate(dataset_module, dataset_name)
    dataset_params = params.dataset['params']
    dataset = dataset(**dataset_params)

    # x_test = np.linspace(-4, 14, 5000)
    algorithm_params.update({'model': model,
                             'dataset': dataset})
    algorithm = algorithm(**algorithm_params)

    path = osp.join(args.model_dir, "model.pt")

    algorithm.model.load_state_dict(torch.load(path))

    model_name = args.model_dir.split('/')[-1]
    run_toy_weather_experiment(algorithm, dataset, model_name)

