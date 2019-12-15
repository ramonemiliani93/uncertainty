import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import algorithms
from models import MLP
from utils import Params
from data_loader.datasets import WeatherDataset


def plot_max_temperature(days, mean, std, dataset, method=None):
    fig, ax = plt.subplots()

    ax.plot(days, mean, '-', color='black')
    ax.fill_between(days, mean - 2 * std, mean + 2 * std, color='gray', alpha=0.2)

    # Plot real function
    ax.plot(dataset.mean, '--')
    ax.plot(np.array(dataset.mean) + 2 * np.array(dataset.std), color='green')
    ax.plot(np.array(dataset.mean) - 2 * np.array(dataset.std), color='green')

    # Plot train data points
    ax.scatter(dataset.data.index.dayofyear, dataset.data.iloc[:, 0].tolist(), c='r', s=2)

    # Custom legend
    legend_elements = [Line2D([0], [0], color='b', lw=1, linestyle='--'),
                       Line2D([0], [0], color='green', lw=1, linestyle='--'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=6),
                       Line2D([0], [0], color='black', lw=1),
                       Patch(facecolor='grey', edgecolor='grey', alpha=0.2)]
    ax.legend(legend_elements, ['Ground truth mean', 'Ground truth $\pm 2\sigma$',
                                'Training data', '$\mu(x)$', '$\pm 2\sigma(x)$'])

    error = abs(np.array(dataset.std) ** 2 - std ** 2).mean()
    print(np.array(dataset.std) ** 2 - std ** 2)
    if method is not None:
        plt.title('Weather dataset for {} with {:0.2f}'.format(method, error))
    else:
        plt.title('Weather dataset {:0.2f}'.format(error))

    plt.xlim(0, 366)
    plt.ylim(0, 100)
    plt.grid()
    plt.show()


if __name__ == '__main__':

    path = os.path.join('experiments', 'toy_weather')
    experiments = {
        'montecarlo': 'MonteCarloDropout',
        'nn': 'DeepEnsembles',
        'ensembles': 'DeepEnsembles',
        'combined': 'Combined'
    }
    for folder, algorithm in experiments.items():
        params_path = os.path.join(path, folder, 'params.yml')
        params = Params(params_path)

        dataset = WeatherDataset()

        model_path = os.path.join(path, folder, 'model.pt')
        algorithm_class = getattr(algorithms, algorithm)
        algorithm_params = params.algorithm['params']
        algorithm_params.update({'model': MLP, 'dataset': dataset})
        algorithm = algorithm_class(**algorithm_params)
        algorithm.load(model_path)

        days = torch.tensor(np.array(list(range(366))) / 366, dtype=torch.float32).reshape(-1, 1)
        mean, std = algorithm.predict_with_uncertainty(days)

        days = days.numpy().ravel() * 366
        mean = mean.numpy().ravel()
        std = std.numpy().ravel()

        plot_max_temperature(days, mean, std, dataset, method=folder)
        print(model_path)
