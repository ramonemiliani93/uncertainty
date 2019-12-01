from typing import Tuple
from abc import abstractmethod

import torch
from torch import nn
from torch.nn.functional import mse_loss
from algorithms.base import UncertaintyAlgorithm
from helpers.functional import enable_dropout


class DeepEnsembles(UncertaintyAlgorithm):

    def __init__(self, model: nn.Module, num_models: int = 5, eps: float = 0.01, adversarial: bool = True, **kwargs):
        # Update to predict mean and variance
        kwargs.update({'num_outputs': 2 * kwargs.get('num_outputs', 1)})
        self.num_models = num_models
        self.eps = eps
        self.adversarial = adversarial
        self.model = nn.ModuleList([model(**kwargs) for i in range(self.num_models)])

    def loss(self, *args, **kwargs) -> torch.Tensor:
        # Sample one model from the ensemble (workaround to the fact that models should be trained in parallel with
        # different data order, will take longer to converge to same point.)
        model = np.random.choice(self.model)

        # Set model to train mode
        model.train()

        # Extract data and set data gradient to true for use in th FGSA
        data, target = args

        if self.adversarial:
            data.requires_grad = True

        # Forward pass through the model
        prediction = model(data)

        # Extract mean and variance from prediction
        mean = prediction[:, 0::2]
        variance = prediction[:, 1::2]

        # Calculate the loss
        nll = self.calculate_nll(target, mean, variance)

        # If adversarial training enabled generate sample
        if self.adversarial:
            # Calculate gradients of model in backward pass
            nll.backward()

            # Collect data gradients
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = self.fgsm_attack(data, data_grad)
            
            # Forward pass
            prediction = model(perturbed_data)
            mean = prediction[:, 0::2]
            variance = prediction[:, 1::2]
            
            # Add to loss
            nll += self.calculate_nll(target, mean, variance)
            
        return nll

    def predict_with_uncertainty(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Set model to evaluation
        self.model.eval()

        # Sample multiple times from the ensemble of models
        mean, variance = [], []
        with torch.no_grad():
            for i in range(self.num_models):
                prediction = self.model[i](args[0])
                mean.append(prediction[:, 0::2])
                variance.append(prediction[:, 1::2])

            # Stack each of the models
            mean = torch.stack(mean)
            variance = torch.stack(variance)

            # Compute statistics
            predicted_mean = mean.mean(0)
            predicted_variance = (torch.exp(variance) + mean ** 2).mean(0) - predicted_mean

        return predicted_mean, predicted_variance

    def fgsm_attack(self, data, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()

        # Find max value of the data
        max_value = data.max()

        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_data = data + self.eps * max_value * sign_data_grad

        # Return the perturbed image
        return perturbed_data

    @staticmethod
    def calculate_nll(target, mean, variance):
        # Estimate the negative log-likelihood. Here we estimate log of sigma squared for stability in training.
        loss = (variance / 2 + ((target - mean) ** 2) / (2 * torch.exp(variance))).sum()

        return loss


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    from data_loader.datasets import SineDataset
    from models.mlp import MLP

    algorithm = DeepEnsembles(model=MLP, adversarial=False)
    train_loader = DataLoader(SineDataset(500, (0, 10)), batch_size=500)
    optimizer = Adam(algorithm.model.parameters(), lr=1e-2, weight_decay=0)

    for epoch in range(50000):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = algorithm.loss(*data)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss))

    print('Finished Training')
    x = np.linspace(-4, 14, 5000)
    x_tensor = torch.FloatTensor(x).reshape(-1, 1)
    mean, var = algorithm.predict_with_uncertainty(x_tensor)
    std = np.sqrt(var)

    plt.plot(x, mean.numpy(), '-', color='gray')
    plt.fill_between(x, (mean.numpy() - 2 * std.numpy())[:, 0], (mean.numpy() + 2 * std.numpy())[:, 0], color='gray', alpha=0.2)

    # Plot real function
    y = x * np.sin(x)
    plt.plot(x, y)

    plt.xlim(-4, 14)
    plt.ylim(-15, 15)
    plt.show()
    print("Finished plotting")