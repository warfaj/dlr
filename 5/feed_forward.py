import numpy as np
import torch
import matplotlib.pyplot as plt

def relu(x):
    zero = torch.zeros_like(x)
    return torch.max(x,zero)


# Simple FeedForward network that assumes same shape across hidden layers

class BasicFeedForwardLayer:

    def __init__(self, input_width,  layer_width, non_linearity=None):
        self.weights = torch.rand((layer_width, input_width), requires_grad=True, dtype=torch.float64)
        self.biases = torch.rand(layer_width, requires_grad=True, dtype=torch.float64)
        self.non_linearity = non_linearity

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        o = x @ self.weights + self.biases
        if self.non_linearity == 'relu':
            return relu(o)
        return o

    def print_weights(self):
        print(self.weights, self.biases)


# Basic FFN makes a fully connected network
class BasicFeedForwardNetwork:
    
    def __init__(self, n_hidden_layers, input_width, hidden_width, output_width):
        self.hidden_layers = [ BasicFeedForwardLayer(hidden_width, input_width, non_linearity='relu') for _ in range(n_hidden_layers)]
        self.output_layer = BasicFeedForwardLayer(hidden_width, output_width)
        self.loss = 0

    def forward(self, x):
        current = x
        for layer in self.hidden_layers:
            current = layer(current)
        return self.output_layer(current)

    def __call__(self, x):
        return self.forward(x)
    
    def get_layers(self):
        return [self.output_layer, *self.hidden_layers]

    def print_weights(self):
        for hidden_layer in self.hidden_layers:
            hidden_layer.print_weights()
        
        self.output_layer.print_weights()
