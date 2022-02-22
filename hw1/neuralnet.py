import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def get_grad_norm(model):
    grad_all = 0.0
    for p in model.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all += grad
    return grad_all ** 0.5

# helper function for updating model based on loss of given x inputs and correct y outputs
# returns loss from criterion
def update_model(x_i, y_i, model, optimizer, criterion, no_unsqueeze=False, no_float=False, grad_norm=False, device=None):
    if not no_float:
        output = model(x_i.float())
    else:
        output = model(x_i)

    if not no_unsqueeze:
        y_i = y_i.unsqueeze(1)

    if not no_float:
        loss = criterion(output, y_i.float())
    else:
        loss = criterion(output, y_i)

    if grad_norm:
        loss = criterion(torch.tensor([get_grad_norm(model)],dtype=torch.float32,requires_grad=True)
                        ,torch.tensor([0],dtype=torch.float32,requires_grad=True))


    #print(loss)
    out_loss = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    return out_loss

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


