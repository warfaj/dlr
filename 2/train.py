# Combined 2 Linear Regression  + SGD on 5d vector -> vector with pytorch grad

import numpy as np
import torch
import matplotlib.pyplot as plt

# Set fixed seed to compare exercise 1 & 2
np.random.seed(42)

# Target function
def f(x):
    return 10*x - 12

# Create data (randomized and whitened)
input = np.arange(100000)
np.random.shuffle(input) 
input = (input-input.mean())/input.std()
targets = f(input)

# Initialize weights and hyperparameters
weight = torch.tensor(0.1, requires_grad=True, dtype=torch.float64)
b = torch.tensor(0.1, requires_grad=True, dtype=torch.float64 )
iters = 0
max_iters = 1000
batch_size = 100
learning_rate = 0.001
losses = []

# SGD gradient descent on vector linear reg
for i in range(0, input.shape[0], batch_size):
    if iters >= max_iters:
        break
    x = torch.tensor(input[i:i+batch_size], dtype=torch.float64)
    t = torch.tensor(targets[i:i+batch_size], dtype=torch.float64)
    o = (x * weight + b)
    loss = (0.5*(t - o)**2).mean()
    print(x,t,loss, o)
    loss.backward()
    print("weights gradient: ",weight.grad)
    print("b grad", b.grad)
    # detach so weight update is not part of computational graph
    with torch.no_grad():
        weight -= learning_rate * weight.grad
        b -= learning_rate * b.grad
        weight.grad.zero_()
        b.grad.zero_()
    print("Loss: ", loss, " iter: ", iters)
    losses.append(loss.item())
    iters += 1
        

print(b, weight)

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
