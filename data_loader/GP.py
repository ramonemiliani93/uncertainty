from data_loader.datasets import SineDataset
import numpy as np
import GPy

def GP(X,y,X_test):
    dim = X.shape[1]

    kern = GPy.kern.RBF(dim, ARD=True)
    model = GPy.models.GPRegression(X, y, kern, normalizer=True)
    model.optimize()
    print(model)
    model.plot()

    mu_test, cov_test = model.predict(X_test, full_cov=True)
    var_test = np.diag(cov_test).reshape(-1,1)

    return mu_test, var_test



if __name__ == '__main__':
    dataset = SineDataset(500, (0, 10))
    test_dataset = SineDataset(500, (0, 10))

    X = []
    X_test = []
    y = []
    y_test = []

    for pair in dataset:
        X.append(pair[0].numpy())
        y.append(pair[1].numpy())

    for pair in test_dataset:
        X_test.append(pair[0].numpy())
        y_test.append(pair[1].numpy())

    X = np.array(X)
    y = np.array(y)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    mu_test, var_test = GP(X,y,X_test)

    #print("mu_test: ", mu_test)
    print("var_test: ", var_test)

    #mu_and_var = list(zip(X_test, mu_test[0]))
    #print("mu_and_var:", mu_and_var)
    #print("X_test: ", X_test[0:20])
    #print("mu_test[0]: ", mu_test[0][0:20])
    #print("mu_and_var:", mu_and_var)
    #mu_and_var.sort(key=lambda pair: pair[0])
    #print(mu_and_var)














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


