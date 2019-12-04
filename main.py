import argparse
import os

import importlib
import torch
import torch.nn
from torch.utils.data import DataLoader
import logging
import utils
from tensorboardX import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.utils import convert_tensor
from ignite.engine.engine import Engine, State, Events


def create_summary_writer(log_dir):
    writer = SummaryWriter(logdir=log_dir)
    return writer


def instantiate(module, name):
    module = importlib.import_module(module)
    instance = getattr(module, name)
    return instance


# TODO
def get_data_loaders(train_batch_size, val_batch_size):
    train_loader = DataLoader()
    val_loader = DataLoader()
    return train_loader, val_loader


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.
    """
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


def create_train_engine(model, optimizer,
                        device=None, non_blocking=False,
                        prepare_batch=_prepare_batch,
                        output_transform=lambda x, y, y_pred, loss: loss.item()):
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = model.loss(*(x, y))  # TODO CHECK
        loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return Engine(_update)


def run(model, loss, optimizer, train_batch_size, val_batch_size, epochs, log_interval, log_dir):
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    writer = create_summary_writer(log_dir)
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    # trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    trainer = create_train_engine(model, optimizer, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'mse': Loss(loss)},
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
        avg_mse = metrics['mse']
        print("Training Results - Epoch: {}   Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_mse))
        writer.add_scalar("training/avg_loss", avg_mse, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_mse = metrics['mse']
        print("Validation Results - Epoch: {}  Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_mse))
        writer.add_scalar("valdation/avg_loss", avg_mse, engine.state.epoch)

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

    # Instantiate model
    model_module, model_name = params.model['module'], params.model['name']
    model_to_train = instantiate(model_module, model_name)

    # Instantiate optimizer
    optimizer_module, optimizer_name = params.optimizer['module'], params.optimizer['name']
    optimizer = instantiate(optimizer_module, optimizer_name)

    # Instantiate loss
    loss_module, loss_name = params.optimizer['module'], params.optimizer['name']
    loss_fn = instantiate(optimizer_module, optimizer_name)

    # Instantiate dataset
    dataset_module, dataset_name = params.dataset['module'], params.dataset['name']

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
