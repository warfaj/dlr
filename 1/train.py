# Linear Regression  + SGD on single scalar -> single scalar

import numpy as np

def f(x):
    return 10*x - 12

vectorized_fx = np.vectorize(f)

x = np.arange(100)
data = np.stack([x,vectorized_fx(x)])

w = np.array([0.1, 0.1])

iters = 0
max_iters = 1000

batch_size = 1
learning_rate = 0.0001

while iters <= max_iters:
    for i in range(int(len(data[0])/batch_size)):
        gradient = np.zeros(2, dtype=np.float64)
        loss = 0
        for j in range(batch_size):
            x = data[0][(i*batch_size) + j]
            t = data[1][(i*batch_size) + j]
            output = w[1] * x + w[0]
            loss += 0.5*(t - output)**2 /batch_size
            gradient[0] += learning_rate * (t - output) / batch_size
            gradient[1] += learning_rate * (t - output) * x/batch_size
        print("calculating batch")
        print("loss: ", loss)
        print("gradient: ", gradient)
        print("prev weights: ", w)
        w +=  gradient
        print("new weights: ",w)
    iters += 1

print(w)
            




