import json
import yaml
import torch
from tensorboardX import SummaryWriter
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader
import importlib

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
                        output_transform=lambda x, y, y_pred, loss: loss.item()):
    if device:
        algorithm.model.to(device)

    def _update(engine, batch):
        algorithm.model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = algorithm.model(x)
        loss = algorithm.loss(*(x, y))
        loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

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
            y_pred = algorithm.model(x)
            return output_transform(x, y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine