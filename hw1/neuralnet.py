import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Neural network with linear hidden layers
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_layers, num_classes, hidden_size):
        assert hidden_size >= 1
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        # input layer
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False)])
        # hidden layers
        for i in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        # output layer
        self.layers.append(nn.Linear(hidden_size, num_classes, bias=False))
        #print(self.layers)

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            #print(i, out.size())
        return out

    def num_parameters(self):
        return self.input_size*self.hidden_size + self.num_layers*(self.hidden_size*self.hidden_size) + self.hidden_size*self.num_classes


