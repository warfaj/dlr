# Combined 2&3 Linear Regression  + SGD on 5d vector -> vector with pytorch grad

import numpy as np
import torch
import matplotlib.pyplot as plt


# Create synthetic data 5 dimensional array
data = np.random.random((100000,5))
# add a dimension of ones for bias weight
data = np.c_[data, np.ones(len(data))]

#shuffle data (unnecssary since random but good practice)
np.random.shuffle(data)

# target func
def f(x):
    weights = np.arange(30).reshape(6,5)
    return x @ weights

targets = np.apply_along_axis(f, axis=1, arr=data)
targets

# Initialize weights and hyperparameters
learning_rate = 0.01
iters = 0
max_iters = 10000
batch_size = 100
weights = torch.rand((6,5), requires_grad=True, dtype=torch.float64)
losses = []

# SGD gradient descent on vector linear reg
for i in range(0, data.shape[0], batch_size):
    if iters >= max_iters:
        break
    x = torch.tensor(data[i:i+batch_size], dtype=torch.float64)
    t = torch.tensor(targets[i:i+batch_size], dtype=torch.float64)
    loss = 0.5*((t - (x @ weights))**2).mean()
    loss.backward()
    print("weights gradient: ",weights.grad)
    # detach so weight update is not part of computational graph
    with torch.no_grad():
        weights -= learning_rate * weights.grad
        weights.grad.zero_()
    print("Loss: ", loss, " iter: ", iters)
    losses.append(loss.item())
    iters += 1
        

print(weights)

# Plot loss curve
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
