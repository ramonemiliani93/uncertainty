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


def plot_max_temperature(days, mean, std, dataset):
    fig, ax = plt.subplots()

    ax.plot(days, mean, '-', color='black')
    ax.fill_between(days, mean - std, mean + std, color='gray', alpha=0.2)

    # Plot real function
    ax.plot(dataset.mean, '--')
    ax.plot(np.array(dataset.mean) + 2 * np.array(dataset.std), color='green')
    ax.plot(np.array(dataset.mean) - 2 * np.array(dataset.std), color='green')

    # # Plot train data points
    ax.scatter(dataset.data.index.dayofyear, dataset.data.iloc[:, 0].tolist(), c='r', s=2)

    # Custom legend
    legend_elements = [Line2D([0], [0], color='b', lw=1, linestyle='--'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=6),
                       Line2D([0], [0], color='black', lw=1),
                       Patch(facecolor='grey', edgecolor='grey', alpha=0.2)]
    ax.legend(legend_elements, ['Ground truth mean', 'Training data', '$\mu(x)$', '$\pm 2\sigma(x)$'])
    # plt.title()
    plt.xlim(0, 366)
    plt.ylim(0, 100)
    plt.grid()
    plt.show()


if __name__ == '__main__':

    path = os.path.join('experiments', 'toy_weather')
    experiments = {
        # 'montecarlo': 'MonteCarloDropout',
        'nn': 'DeepEnsembles'
    }
    for folder, algorithm in experiments.items():
        params_path = os.path.join(path, folder, 'params.yml')
        params = Params(params_path)

        model_path = os.path.join(path, folder, 'model.pt')
        algorithm_class = getattr(algorithms, algorithm)
        algorithm_params = params.algorithm['params']
        algorithm_params.update({'model': MLP})
        algorithm = algorithm_class(**algorithm_params)
        algorithm.load(model_path)

        dataset = WeatherDataset()
        days = torch.tensor(np.array(list(range(366))) / 366, dtype=torch.float32).reshape(-1, 1)
        mean, std = algorithm.predict_with_uncertainty(days)

        days = days.numpy().ravel() * 366
        mean = mean.numpy().ravel()
        std = std.numpy().ravel()

        plot_max_temperature(days, mean, std, dataset)
        print(model_path)
