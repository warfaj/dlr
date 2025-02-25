
# 6 arbitrary depth feed forward network using pytorch object oriented components
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import BasicFeedForwardNetwork

# Set fixed seed 
np.random.seed(42)
torch.manual_seed(42)

# Use available accelerator 
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# Create synthetic data 5 dimensional array
data = np.random.random((1000000,5))
#shuffle data (unnecssary since random but good practice)
np.random.shuffle(data)

# target func
def f(x):
    weights = np.arange(25).reshape(5,5)
    b = np.array([1, 1, 1, 1, 1])
    return np.cos(x @ weights)+ b

targets = np.apply_along_axis(f, axis=1, arr=data)
targets

# Initialize weights and hyperparameters
learning_rate = 0.0001
iters = 0
batch_size = 1000
epochs = 30
losses = []


model = BasicFeedForwardNetwork(2, 5,5,5).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss(reduction='mean')

model.train()

# SGD gradient descent on vector linear reg
for _ in range(epochs):
    for i in range(0, data.shape[0], batch_size):

        x = torch.tensor(data[i:i+batch_size], dtype=torch.float32).to(device)
        t = torch.tensor(targets[i:i+batch_size], dtype=torch.float32).to(device)

        o = model(x)

        loss = loss_fn(o, t)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        iters += 1

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
plt.savefig("graph.png", bbox_inches="tight")
