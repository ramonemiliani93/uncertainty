import json
import yaml
import torch
from tensorboardX import SummaryWriter
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader
import importlib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt


class Params:
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(yml_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, yml_path):
        with open(yml_path) as f:
            params = yaml.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def create_summary_writer(log_dir):
    writer = SummaryWriter(logdir=log_dir)
    return writer


def instantiate(module, name):
    module = importlib.import_module(module)
    instance = getattr(module, name)
    return instance


def get_data_loaders(dataset, train_batch_size, val_batch_size):
    train_loader = DataLoader(dataset, batch_size=train_batch_size)
    val_loader = DataLoader(dataset, batch_size=val_batch_size)
    return train_loader, val_loader


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.
    """
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


def create_train_engine(algorithm, optimizer,
                        device=None, non_blocking=False,
                        prepare_batch=_prepare_batch,
                        output_transform=lambda batch, loss: loss.item()):
    if device:
        algorithm.model.to(device)

    def _update(engine, batch):
        algorithm.model.train()
        optimizer.zero_grad()
        batch = prepare_batch(batch, device=device, non_blocking=non_blocking)
        # y_pred = algorithm.model(x)
        loss = algorithm.loss(*batch)
        loss.backward()
        optimizer.step()
        return output_transform(batch, loss)

    return Engine(_update)


def create_supervised_evaluator(algorithm, metrics=None,
                                device=None, non_blocking=False,
                                prepare_batch=_prepare_batch,
                                output_transform=lambda x, y, y_pred: (y_pred, y,)):
    metrics = metrics or {}

    if device:
        algorithm.model.to(device)

    def _inference(engine, batch):
        algorithm.model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            # TODO FIX THIS ASAP
            y_pred_mean, y_pred_var = algorithm.predict_with_uncertainty(x)
            return output_transform(x, y, y_pred_mean)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def plot_toy_uncertainty(x_test, mean, std, train_loader):
    fig, ax = plt.subplots()

    ax.plot(x_test, mean.numpy(), '-', color='black')
    ax.fill_between(x_test, mean.numpy() - 2 * std.numpy(), mean.numpy() + 2 * std.numpy(), color='gray', alpha=0.2)
    # ax.fill_between(x, (mean.numpy() - 2 * std.numpy())[:, 0], (mean.numpy() + 2 * std.numpy())[:, 0], color='gray',
    #               alpha=0.2)
    # Plot real function
    y = x_test * np.sin(x_test)
    ax.plot(x_test, y, '--')

    # Plot train data points
    x_tensor, y_tensor = next(iter(train_loader))
    x = x_tensor.numpy()
    y = y_tensor.numpy()
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
    plt.show()
