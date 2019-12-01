import argparse
import os
import os.path as osp
import importlib
import torch
import torch.nn
from torch.utils.data import DataLoader
import logging
import utils
from tensorboardX import SummaryWriter
from ignite.engine import Events  # , create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.utils import convert_tensor
from ignite.engine.engine import Engine, State, Events
import numpy as np
import matplotlib.pyplot as plt

def create_summary_writer(log_dir):
    writer = SummaryWriter(logdir=log_dir)
    return writer


def instantiate(module, name):
    module = importlib.import_module(module)
    instance = getattr(module, name)
    return instance


# TODO
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
        loss = algorithm.loss(*(x, y))  # TODO CHECK
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


def run(model, train_loader, val_loader, optimizer, epochs, log_interval, log_dir):
    writer = create_summary_writer(log_dir)
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    # trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    trainer = create_train_engine(model, optimizer, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'loss': Loss(model.loss)},
                                            device=device)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
              "".format(engine.state.epoch, engine.state.iteration, len(train_loader), engine.state.output))
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_mse = metrics['loss']
        print("Training Results - Epoch: {}   Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_mse))
        writer.add_scalar("training/avg_loss", avg_mse, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_mse = metrics['loss']
        print("Validation Results - Epoch: {}  Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_mse))
        writer.add_scalar("validation/avg_loss", avg_mse, engine.state.epoch)

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='experiments/base_model', help="Directory containing params.yml")
    parser.add_argument('--restore-file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'
    args = parser.parse_args()

    yaml_path = os.path.join(args.model_dir, 'params.yml')
    assert os.path.isfile(yaml_path), "No yaml configuration file found at {}".format(yaml_path)
    params = utils.Params(yaml_path)

    # Instantiate algorithm
    algorithm_module, algorithm_name = params.algorithm['module'], params.algorithm['name']
    algorithm_params = params.algorithm['params']

    algorithm = instantiate(algorithm_module, algorithm_name)

    # Instantiate model
    model_module, model_name = params.model['module'], params.model['name']
    model = instantiate(model_module, model_name)

    algorithm_params.update({'model': model})
    algorithm = algorithm(algorithm_params)

    # model_to_train = model_to_train(algorithm_params)
    # Instantiate optimizer
    optimizer_module, optimizer_name = params.optimizer['module'], params.optimizer['name']
    optimizer = instantiate(optimizer_module, optimizer_name)
    optimizer = optimizer(algorithm.model.parameters(), lr=1e-2)

    # Instantiate dataset
    dataset_module, dataset_name = params.dataset['module'], params.dataset['name']
    dataset_params = params.dataset['params']
    dataset = instantiate(dataset_module, dataset_name)
    dataset = dataset(dataset_params)

    train_loader, _ = get_data_loaders(dataset, train_batch_size=500, val_batch_size=100)
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.parameters['num_epochs']))

    tensorboard_dir = osp.join(args.model_dir, 'tensorboard')
    if not osp.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    run(algorithm, train_loader, train_loader,
        optimizer, params.parameters['num_epochs'], 10000,
        tensorboard_dir)

    x = np.linspace(-4, 14, 5000)
    x_tensor = torch.FloatTensor(x).reshape(-1, 1)
    mean, std = algorithm.predict_with_uncertainty(x_tensor)

    plt.plot(x, mean.numpy(), '-', color='gray')
    plt.fill_between(x, mean.numpy() - 2 * std.numpy(), mean.numpy() + 2 * std.numpy(), color='gray', alpha=0.2)

    # Plot real function
    y = x * np.sin(x)
    plt.plot(x, y)

    plt.xlim(-4, 14)
    plt.show()
    print("Finished plotting")


