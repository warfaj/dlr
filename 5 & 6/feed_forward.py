import numpy as np
import torch
import matplotlib.pyplot as plt

# Simple FeedForward network that assumes same shape across hidden layers

class BasicFeedForwardLayer:

    def __init__(self, input_width,  layer_width , learning_rate):
        self.weights = torch.rand((layer_width, input_width), requires_grad=True, dtype=torch.float64)
        self.biases = torch.rand(layer_width, requires_grad=True, dtype=torch.float64)
        self.learning_rate = learning_rate
    
    def relu(self, x):
        zero = torch.zeros_like(x)
        return torch.max(x,zero)

    def backward(self):
        with torch.no_grad():
            self.weights -= self.learning_rate * self.weights.grad
            self.biases -= self.learning_rate * self.biases.grad
            self.weights.grad.zero_()
            self.biases.grad.zero_()
    
    def print_weights(self):
        print(self.weights, self.biases)

class BasicFeedForwardOutputLayer(BasicFeedForwardLayer):

     def forward(self, x):
        return x @ self.weights + self.biases

class BasicFeedForwardHiddenLayer(BasicFeedForwardLayer):

    def forward(self, x):
        o = x @ self.weights + self.biases
        return self.relu(o)


# Basic FFN makes a fully connected network
class BasicFeedForwardNetwork:
    
    def __init__(self, n_hidden_layers, learning_rate, input_width, hidden_width, output_width):
        self.hidden_layers = [ BasicFeedForwardHiddenLayer(hidden_width, input_width, learning_rate) for _ in range(n_hidden_layers)]
        self.output_layer = BasicFeedForwardOutputLayer(hidden_width, output_width, learning_rate)
        self.loss = 0

    def forward(self, x):
        current = x
        for layer in self.hidden_layers:
            current = layer.forward(current)
        return self.output_layer.forward(current)

    def backward(self):
        self.output_layer.backward()
        for layer in self.hidden_layers:
            layer.backward()
    
    def squared_loss(self, output, target):
        return (0.5*(target -output)**2).mean()

    def print_weights(self):
        for hidden_layer in self.hidden_layers:
            hidden_layer.print_weights()
        
        self.output_layer.print_weights()
