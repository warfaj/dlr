# Combined 5 & 6 arbitrary depth feed forward  + SGD on 5d vector -> vector with pytorch grad

import numpy as np
import torch
import matplotlib.pyplot as plt
import feed_forward as ff

# Set fixed seed 
np.random.seed(42)
torch.manual_seed(42)

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
losses = []

network = ff.BasicFeedForwardNetwork(2,learning_rate, 5,5,5)

# SGD gradient descent on vector linear reg
for i in range(0, data.shape[0], batch_size):
    if iters >= max_iters:
        break
   
    x = torch.tensor(data[i:i+batch_size], dtype=torch.float64)
    t = torch.tensor(targets[i:i+batch_size], dtype=torch.float64)

    o = network.forward(x)

    loss = network.squared_loss(o, t)

    loss.backward()
    network.backward()

    losses.append(loss.item())
    iters += 1

network.print_weights()

# Plot loss curve
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
