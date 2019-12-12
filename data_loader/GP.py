#from data_loader.datasets import SineDataset
from data_loader.datasets import SineDataset
from torch.utils.data import DataLoader
from utils import plot_toy_uncertainty
import numpy as np
import GPy
import torch


def GP(X,y,X_test):
    dim = X.shape[1]

    kern = GPy.kern.RBF(dim, ARD=True)
    model = GPy.models.GPRegression(X, y, kern, normalizer=True)
    model.optimize()
    print(model)

    # model.plot()

    mu_test, cov_test = model.predict(X_test, full_cov=True)
    var_test = np.diag(cov_test).reshape(-1,1)

    return mu_test, var_test



if __name__ == '__main__':
    dataset = SineDataset(500, (0, 10))
    #test_dataset = SineDataset(500, (0, 10))
    train_dataloader = DataLoader(dataset)

    X = []
    X_test = []
    y = []
    #y_test = []

    for pair in dataset:
        X.append(pair[0].numpy())
        y.append(pair[1].numpy())

    X = np.array(X)
    y = np.array(y)

    #for pair in test_dataset:
        #X_test.append(pair[0].numpy())
        #y_test.append(pair[1].numpy())

    #l = [[X_test[i], y_test[i]] for i in range(len(X_test))]
    #l.sort()

    #X_test = [l[i][0] for i in range(len(l))]
    #X_test = np.array(X_test)
    #print("X_test.shape: ", X_test.shape)
    #y_test = [l[i][1] for i in range(len(l))]
    #y_test = np.array(y_test)
    #print("y_test.shape: ", y_test.shape)

    X_test = np.linspace(-4,14,5000)
    X_test = np.expand_dims(X_test, -1)

    mu_test, var_test = GP(X,y,X_test)

    mu_test = torch.from_numpy(mu_test).squeeze()
    std_test = np.sqrt(var_test)
    std_test = torch.from_numpy(std_test).squeeze()

    #print(mu_test)

    plot_toy_uncertainty(np.squeeze(X_test), mu_test, std_test, train_dataloader)


"""
from datasets import SineDataset
import numpy as np
import GPy

dataset = SineDataset(1000, (0, 10))


X = []
y = []


for pair in dataset:
    X.append(pair[0].numpy())
    y.append(pair[1].numpy())

X = np.array(X)
y = np.array(y)

dim = X.shape[1]

kern = GPy.kern.RBF(dim, ARD = True)

print("y shape: ", y.shape)

model = GPy.models.GPRegression(X, y, kern)

model.optimize()

#model.optimize_restarts(num_restarts = 10) # Always the same value

print(model)

fig = model.plot()
#GPy.plotting.show(fig, filename='Essaie')

GPy.plotting.gpy_plot.data_plots.plot_data(fig)

y_pred, cov = model.predict(X, full_cov=True)

#print(cov.shape)

#print(cov[0:10,0:10])
"""


