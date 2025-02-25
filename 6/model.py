import torch.nn as nn
import torch.nn.functional as F



# Simple FeedForward network that assumes same shape across hidden layers using Pytorch
class BasicFeedForwardNetwork(nn.Module):

    def __init__(self, n_hidden_layers, input_width, hidden_width, output_width ):
        super().__init__()
        
        layers = []

        in_length = input_width
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(input_width, hidden_width))
            layers.append(nn.ReLU())
            in_length = hidden_width
        layers.append(nn.Linear(in_length, output_width))

        self.network = nn.Sequential(*layers)

    
    def forward(self, x):
        return self.network(x)
    


        
