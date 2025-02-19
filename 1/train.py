# Linear Regression  + SGD on single scalar -> single scalar

import numpy as np
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
w = np.array([0.1, 0.1])
iters = 0
max_iters = 1000
batch_size = 100
learning_rate = 0.001
losses = []

input.dtype
# Process each batch 
for i in range(0, input.shape[0], batch_size):
    if iters > max_iters: 
        break
    x = input[i : i+batch_size ]
    target = targets[i: i+batch_size]
    output = w[1]*x+w[0]
    loss = (0.5*(target - output)**2).mean()
    print(x,target,loss, output)
    gradient_w0 = (target-output)
    gradient_w1 = (target - output)* x
    print(gradient_w0, gradient_w1)
    print("loss: ", np.mean(loss))
    print("prev weights: ", w)
    w[0] += learning_rate * np.mean(gradient_w0)
    w[1] += learning_rate * np.mean(gradient_w1)
    print("new weights: ",w)
    losses.append(loss.item())
    iters += 1
    

print(w)

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




