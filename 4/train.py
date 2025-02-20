# 3. Classification where input is a vector, output is categorical (softmax & negative log likelihood loss).

import numpy as np
import torch
import matplotlib.pyplot as plt

DATASET_SIZE = 1000000

# Data is 5 dimensional vec; taget is 3 categories 
data = np.random.random((DATASET_SIZE,5))
#shuffle data (unnecssary since random but good practice)
np.random.shuffle(data)
data = torch.tensor(data,  dtype=torch.float64)
targets = torch.tensor(np.random.randint(0,3, DATASET_SIZE), dtype=torch.int64)

# Initialize weights and hyperparameters
learning_rate = 0.01
iters = 0
max_iters = 100000
batch_size = 100
weights = torch.rand((5,5), requires_grad=True, dtype=torch.float64)
b = torch.rand(5, requires_grad=True, dtype=torch.float64 )
losses = []


# turns R -> [0,1]
def softmax(o):
    exp = torch.exp(o)
    return exp/exp.sum()


# loss is equivalent to the negative log probability (higher likelihoods for correct class are smaller)
def loss(target, output):
    log_soft_max = torch.log(softmax(output))
    return -log_soft_max[torch.arange(log_soft_max.size(0)), target].mean()


def forward(x, w, b):
    assert x.shape[1] == w.shape[0] and x.shape[1] == w.shape[1], "x and w aren't compatible sizes"
    assert w.shape[0] == b.shape[0], "w and b are not the correct size"
    return x @ w + b


# SGD gradient descent on vector linear reg
for i in range(0, data.shape[0], batch_size):
    if iters >= max_iters:
        break
    x = data[i:i+batch_size]
    t = targets[i:i+batch_size]
    loss_val = loss(t, forward(x,weights,b))
    loss_val.backward()
    # detach so weight update is not part of computational graph
    with torch.no_grad():
        weights -= learning_rate * weights.grad
        b -= learning_rate * b.grad
        weights.grad.zero_()
        b.grad.zero_()
    losses.append(loss_val.item())
    iters += 1
        

print(weights, b)

# Plot loss curve
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
