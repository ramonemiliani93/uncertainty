import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import algorithms
from models import MLP
from utils import Params, get_data_loaders
from data_loader.datasets import SineDataset


def plot_toy_uncertainty(x_test, mean, std, train_loader, figure_path):
    fig, ax = plt.subplots()

    ax.plot(x_test, mean.numpy(), '-', color='black')
    ax.fill_between(x_test, mean.numpy() - 2 * std.numpy(), mean.numpy() + 2 * std.numpy(), color='gray', alpha=0.2)
    # ax.fill_between(x, (mean.numpy() - 2 * std.numpy())[:, 0], (mean.numpy() + 2 * std.numpy())[:, 0], color='gray',
    #               alpha=0.2)
    # Plot real function
    y = x_test * np.sin(x_test)
    ax.plot(x_test, y, '--')

    # Plot train data points
    x_tensor, y_tensor = train_loader.dataset.samples, train_loader.dataset.targets
    x = x_tensor
    y = y_tensor
    ax.scatter(x, y, c='r', s=2)

    # Custom legend
    legend_elements = [Line2D([0], [0], color='b', lw=1, linestyle='--'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=6),
                       Line2D([0], [0], color='black', lw=1),
                       Patch(facecolor='grey', edgecolor='grey', alpha=0.2)]
    ax.legend(legend_elements, ['Ground truth mean', 'Training data', '$\mu(x)$', '$\pm 2\sigma(x)$'])
    plt.title(
        '$y = x \, sin(x) + 0.3 \, \epsilon_1 + 0.3 \, x \, \epsilon_2 \;' + 'where' + '\; \epsilon_1,\,\epsilon_2 \sim \mathcal{N}(0,1)$')
    plt.xlim(-4, 14)
    plt.ylim(-15, 15)
    plt.grid()
    plt.savefig(figure_path, format='pdf', bbox_inches="tight")
    plt.show()


if __name__ == '__main__':

    path = os.path.join('experiments', 'toy_weather')
    experiments = {
        'montecarlo': 'MonteCarloDropout',
        'nn': 'DeepEnsembles',
        'ensembles': 'DeepEnsembles',
        'bnn': 'BNN',
        'combined': 'Combined'
    }
    std_values = {}
    for folder, algorithm in experiments.items():
        params_path = os.path.join(path, folder, 'params.yml')
        params = Params(params_path)

        figure_path = os.path.join(path, folder, 'figure.pdf')
        model_path = os.path.join(path, folder, 'model.pt')
        algorithm_class = getattr(algorithms, algorithm)
        algorithm_params = params.algorithm['params']
        algorithm_params.update({'model': MLP})
        algorithm = algorithm_class(**algorithm_params)
        algorithm.load(model_path)

        # dataset_module, dataset_name = params.dataset['module'], params.dataset['name']
        # dataset = instantiate(dataset_module, dataset_name)
        dataset_params = params.dataset['params']
        dataset = SineDataset(**dataset_params)
        train_loader, _ = get_data_loaders(dataset, params.parameters['batch_size'], sampler=None)

        x_test = torch.FloatTensor(np.linspace(-4, 14, 5000)).reshape(-1, 1)
        x = torch.FloatTensor(x_test).reshape(-1, 1)
        mean, std = algorithm.predict_with_uncertainty(x)
        mean = mean.numpy().ravel()
        std = std.numpy().ravel()
        std_values[algorithm] = std

        plot_toy_uncertainty(x_test, mean, std, train_loader, figure_path)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for k in std_values.keys():
            ax.plot(x, std_values[k], label=k)

        x = np.linspace(0, 10, 5000)
        true_std = 0.3 * ((1 + x ** 2) ** 0.5)
        ax.plot(x, true_std, label='True std')
        plt.title('plots')
        ax.legend()

    plt.show()
