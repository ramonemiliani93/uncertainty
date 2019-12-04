from typing import Tuple
from abc import abstractmethod

import torch
from torch import nn
from torch.nn.functional import mse_loss
from algorithms.base import UncertaintyAlgorithm
from helpers.functional import enable_dropout


class MonteCarloDropout(UncertaintyAlgorithm):

    def __init__(self, **kwargs):
        super(MonteCarloDropout, self).__init__(**kwargs)

        # Algorithm parameters
        self.num_samples: int = 'num_samples'

        # Create model
        model = kwargs.get('model')
        self.model = model(**dict(**kwargs))

    def loss(self, *args, **kwargs) -> torch.Tensor:
        # Set model to train mode
        self.model.train()

        # Forward pass and MSE loss
        data, target = args
        prediction = self.model(data)
        mse = mse_loss(target, prediction)

        return mse

    def predict_with_uncertainty(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # Set model to evaluation except dropout layers
        self.model.eval()
        enable_dropout(self.model)

        # Sample multiple times from the ensemble of models
        prediction = []
        with torch.no_grad():
            for i in range(self.num_samples):
                prediction.append(self.model(args[0]))

            # Calculate statistics of the outputs
            prediction = torch.stack(prediction)
            mean = prediction.mean(0).squeeze()
            std = prediction.var(0).squeeze().sqrt() #FIXME

        return mean, std


if __name__ == '__main__':
    import numpy as np
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    from data_loader.datasets import SineDataset
    from models.mlp import MLP

    algorithm = MonteCarloDropout(model=MLP, p=0.05, num_samples=10000)
    dict_params = {'num_samples': 500, 'domain': (0, 10)}
    train_loader = DataLoader(SineDataset(**dict_params), batch_size=500)
    optimizer = Adam(algorithm.model.parameters(), lr=1e-2, weight_decay=0)

    for epoch in range(10000):  # loop over the dataset multiple times

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
    mean, std = algorithm.predict_with_uncertainty(x_tensor)

    # Start plotting
    fig, ax = plt.subplots()

    ax.plot(x, mean.numpy(), '-', color='black')
    ax.fill_between(x, mean.numpy() - 2 * std.numpy(), mean.numpy() + 2 * std.numpy(), color='gray', alpha=0.2)

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
    plt.title('$y = x \, sin(x) + 0.3 \, \epsilon_1 + 0.3 \, x \, \epsilon_2 \;'+ 'where' + '\; \epsilon_1,\,\epsilon_2 \sim \mathcal{N}(0,1)$')
    plt.xlim(-4, 14)
    plt.ylim(-15, 15)
    plt.grid()
    plt.show()
    print("Finished plotting")