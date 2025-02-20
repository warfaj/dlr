# Combined 3 Linear Regression  + SGD on 5d vector -> vector with pytorch grad

import numpy as np
import torch
import matplotlib.pyplot as plt


# Create synthetic data 5 dimensional array
data = np.random.random((1000000,5))
#shuffle data (unnecssary since random but good practice)
np.random.shuffle(data)

# target func
def f(x):
    weights = np.arange(25).reshape(5,5)
    b = np.array([100, 100, 100, 100, 100])
    return x @ weights + b

targets = np.apply_along_axis(f, axis=1, arr=data)
targets

# Initialize weights and hyperparameters
learning_rate = 0.01
iters = 0
max_iters = 100000
batch_size = 1000
weights = torch.rand((5,5), requires_grad=True, dtype=torch.float64)
b = torch.rand(5, requires_grad=True, dtype=torch.float64 )
losses = []

# explicit forward function
def forward(x, w, b):
    assert x.shape[1] == w.shape[0] and x.shape[1] == w.shape[1], "x and w aren't compatible sizes"
    assert w.shape[0] == b.shape[0], "w and b are not the correct size"
    return x @ w + b

# SGD gradient descent on vector linear reg
for i in range(0, data.shape[0], batch_size):
    if iters >= max_iters:
        break
    x = torch.tensor(data[i:i+batch_size], dtype=torch.float64)
    t = torch.tensor(targets[i:i+batch_size], dtype=torch.float64)
    loss = 0.5*((t - (forward(x,weights,b)))**2).mean()
    loss.backward()
    # detach so weight update is not part of computational graph
    with torch.no_grad():
        weights -= learning_rate * weights.grad
        b -= learning_rate * b.grad
        weights.grad.zero_()
        b.grad.zero_()
    losses.append(loss.item())
    iters += 1
        

print(weights, b)

# Plot loss curve
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.grid(True)
