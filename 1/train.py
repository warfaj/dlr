# Linear Regression  + SGD on single scalar -> single scalar

import numpy as np

# Target function
def f(x):
    return 10*x - 12

# Create data (randomized and whitened)
input = np.arange(100000)
np.random.shuffle(input) 
input.mean()
input = (input-input.mean())/input.std()
targets = f(input)


# Initialize weights and hyperparameters
w = np.array([0.1, 0.1])
iters = 0
max_iters = 100
batch_size = 2
learning_rate = 0.0001

# Process each batch 
for i in range(0, input.shape[0], batch_size):
    if iters > max_iters: 
        break
    x = input[i : i+batch_size ]
    target = targets[i: i+batch_size]
    output = w[1]*x+w[0]
    loss = 0.5*(target - output)**2
    gradient_w0 = (target-output)
    gradient_w1 = (target - output)* x
    print("loss: ", np.mean(loss))
    print("prev weights: ", w)
    w[0] += learning_rate * np.mean(gradient_w0)
    w[1] += learning_rate * np.mean(gradient_w1)
    print("new weights: ",w)
    iters += 1

print(w)
            




