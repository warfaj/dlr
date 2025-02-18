# Combined 2&3 Linear Regression  + SGD on 5d vector -> vector with pytorch grad

import numpy as np
import torch


# Create synthetic data 5 dimensional array
data = np.random.random((10000,5))
# add a dimension of ones for bias weight
data = np.c_[data, np.ones(len(data))]

# target func
def f(x):
    weights = np.random.random((6,5))
    return x @ weights

targets = np.apply_along_axis(f, axis=1, arr=data)


learning_rate = 0.00001
iters = 0
max_iters = 1000
batch_size = 100

weights = torch.rand((6,5), requires_grad=True, dtype=torch.float64)

while iters <= max_iters:
    for i in range(int(len(data)/ batch_size)):
        loss = 0
        for j in range(batch_size):
            x = torch.tensor(data[(i*batch_size) + j], dtype=torch.float64)
            t = torch.tensor(targets[(i*batch_size) + j], dtype=torch.float64)
            loss += 0.5*((t - (x @ weights))**2).sum()
        loss /= batch_size
        loss.backward()
        print("weights gradient: ",weights.grad)
        with torch.no_grad():
            weights -= learning_rate * weights.grad
            weights.grad.zero_()
        print("Loss: ", loss, " iter: ", iters)
    iters += 1
        

print(weights)

