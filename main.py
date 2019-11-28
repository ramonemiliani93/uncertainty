import argparse
import os
import sys
import importlib
import torch
import torch.nn

import logging
import tqdm
import utils
from tensorboardX import SummaryWriter

from data_loader.datasets import SineDataset
from mixin.montecarlo import MonteCarloMixin


def instantiate(module, name):
    module = importlib.import_module(module)
    instance = getattr(module, name)
    return instance


def train(model, optimizer, dataloader, params, metrics):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # Metrics?

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, data in enumerate(dataloader):
            # move to GPU if available
            if torch.cuda.is_available():
                train_batch = train_batch.to(params.device)
                labels_batch = labels_batch.to(params.device)

            loss = model.loss(*data)

            # update the average loss
            metrics['loss'].update(loss.item())

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                for metric in metrics.values():
                    metric(output_batch, labels_batch)

            t.set_postfix(loss='{:05.3f}'.format(metrics['loss'].mean()))
            t.update()

    # Summary of metrics in log
    metrics_string = "".join([metric.__str__(y_labels=params.y_labels) for metric in metrics.values()])
    logging.info("- Train metrics: \n" + metrics_string)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='experiments/base_model', help="Directory containing params.yml")
    parser.add_argument('--restore-file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'
    args = parser.parse_args()

    yaml_path = os.path.join(args.model_dir, 'params.yml')
    assert os.path.isfile(yaml_path), "No json configuration file found at {}".format(yaml_path)
    params = utils.Params(yaml_path)
    # Tensorboard writer
    # writer = SummaryWriter(args.model_dir)

    # Instantiate model
    model_module, model_name = params.model['module'], params.model['name']
    model = instantiate(model_module, model_name)

    # Instantiate optimizer
    optimizer_module, optimizer_name = params.optimizer['module'], params.optimizer['name']
    optimizer = instantiate(optimizer_module, optimizer_name)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    # train(model, train_dl, val_dl, optimizer, metrics, params, args.model_dir,
    #                   args.restore_file, writer)

    # writer.export_scalars_to_json(os.path.join(args.model_dir, "all_scalars.json"))
    # writer.close()

