import argparse
import os
import os.path as osp

import torch
import torch.nn
import logging

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import utils
from ignite.metrics import Loss
from ignite.engine.engine import Events
import numpy as np
import matplotlib.pyplot as plt
from utils import create_train_engine, create_supervised_evaluator,\
    get_data_loaders, create_summary_writer, instantiate


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
    parser.add_argument('--model-dir', default='experiments/montecarlo', help="Directory containing params.yml")
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
    algorithm = algorithm(**algorithm_params)

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
    mean, std = mean.reshape(-1), std.reshape(-1)
    # Start plotting
    fig, ax = plt.subplots()

    ax.plot(x, mean.numpy(), '-', color='black')
    ax.fill_between(x, mean.numpy() - 2 * std.numpy(), mean.numpy() + 2 * std.numpy(), color='gray', alpha=0.2)
    # ax.fill_between(x, (mean.numpy() - 2 * std.numpy())[:, 0], (mean.numpy() + 2 * std.numpy())[:, 0], color='gray',
     #               alpha=0.2)
    # Plot real function
    y = x * np.sin(x)
    ax.plot(x, y, '--')

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
    print("Finished plotting")

