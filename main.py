import argparse
import os
import os.path as osp

import torch
import torch.nn
import logging

import utils
from ignite.metrics import Average
from ignite.engine.engine import Events
import numpy as np
from utils import create_train_engine, create_supervised_evaluator,\
    get_data_loaders, create_summary_writer, instantiate, plot_toy_uncertainty


def run(model, train_loader, val_loader, optimizer, epochs, log_interval, log_dir, val=False, log=True):
    writer = create_summary_writer(log_dir)
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'
    loss_metric = Average()
    trainer = create_train_engine(model, optimizer, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'loss': loss_metric},
                                            device=device)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        # print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
        #      "".format(engine.state.epoch, engine.state.iteration, len(train_loader), engine.state.output))
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    if log:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            avg_mse = metrics['loss']
            print("Training Results - Epoch: {}   Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_mse))
            writer.add_scalar("training/avg_loss", avg_mse, engine.state.epoch)

    if val:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_mse = metrics['loss']
            # print("Validation Results - Epoch: {}  Avg loss: {:.2f}"
            #      .format(engine.state.epoch, avg_mse))
            writer.add_scalar("validation/avg_loss", avg_mse, engine.state.epoch)

    # kick everything off
    trainer.run(train_loader, max_epochs=epochs)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='experiments/'
                                               'uci_regression/'
                                               'boston/nn', help="Directory containing params.yml")
    parser.add_argument('--restore-file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'
    args = parser.parse_args()

    np.random.seed(42)
    torch.random.manual_seed(42)
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
    model_params = params.model['params']

    # Instantiate optimizer
    optimizer_module, optimizer_name = params.optimizer['module'], params.optimizer['name']
    optimizer = instantiate(optimizer_module, optimizer_name)

    # Instantiate dataset
    dataset_module, dataset_name = params.dataset['module'], params.dataset['name']
    dataset = instantiate(dataset_module, dataset_name)
    dataset_params = params.dataset['params']

    # Instantiate sampler
    sampler_module, sampler_name = params.sampler['module'], params.sampler['name']
    sampler_params = params.sampler['params']

    # Create objects
    dataset = dataset(**dataset_params)
    sampler = None
    if params.sampler['name'] is not None:
        sampler = instantiate(sampler_module, sampler_name)
        sampler = sampler(dataset, **sampler_params)

    x_test = np.linspace(-4, 14, 5000)
    algorithm_params.update({'model': model,
                             'dataset': dataset,
                             'x_test': x_test})
    algorithm_params.update(model_params)
    algorithm = algorithm(**algorithm_params)
    optimizer = optimizer(algorithm.model.parameters(), lr=params.parameters['learning_rate'], weight_decay=1e-4)
    train_loader, _ = get_data_loaders(dataset, params.parameters['batch_size'], sampler=sampler)
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.parameters['num_epochs']))

    tensorboard_dir = osp.join(args.model_dir, 'tensorboard')
    if not osp.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    run(algorithm, train_loader, train_loader,
        optimizer, params.parameters['num_epochs'], 10000,
        tensorboard_dir)

    test_ll = algorithm.get_test_ll()

    model_path = os.path.join(args.model_dir, 'model.pt')
    algorithm.save(model_path)

