import matplotlib.pyplot as plt
import random
import pylab as pl
import numpy as np
from torch.utils.data import Dataset
import math

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from neuralnet import NeuralNet, update_model
from dataset_objs import FunctionDataset, TimeSeriesDataset


#
# hyper parameters
# 
input_size = 1
output_size = 1
batch_size = 100
learning_rate = 0.001
num_epochs = 10
training_times = 1
function_dataset = FunctionDataset()


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.MSELoss()

xy_train = [x for x in function_dataset.x_y]
#print(len(xy_train))
training_set = TimeSeriesDataset([x[0] for x in xy_train], [y[1] for y in xy_train])

#xy_test = [x for x in function_dataset.x_y[int(.9*len(function_dataset.x_y)):]]
#print(len(xy_test))
#testing_set = TimeSeriesDataset([x[0] for x in xy_test], [y[1] for y in xy_test])

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)

#model
num_layers = 4
hidden_size = 22
model = NeuralNet(input_size, num_layers=num_layers, hidden_size=hidden_size, num_classes=output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print("model.num_parameters()",model.num_parameters(),num_layers,hidden_size)

def get_grad_norm(model):
    grad_all = 0.0
    for p in model.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all += grad
    return grad_all ** 0.5

class GradNorm(nn.Module):
    def __init__(self, model):
        super(GradNorm,self).__init__()
        self.model = model
    def forward(self, outputs, labels):
        return torch.tensor(get_grad_norm(self.model),dtype=torch.float32,device=device,requires_grad=True)

# Training
n_total_steps = len(train_loader)
mean_epoch_loss = []
batch_loss = []
grad_norm_list = [get_grad_norm(model)]
for epoch in range(num_epochs):
    if epoch >= num_epochs/2:
        criterion = GradNorm(model)
    loss_list = []
    for i, (x_i, y_i) in enumerate(train_loader):
        x_i = x_i.reshape(-1, input_size).to(device)
        y_i = y_i.to(device)
        loss_list.append(update_model(x_i, y_i, model, optimizer, criterion))
        grad_norm_list.append(get_grad_norm(model))

        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss_list[-1]:.4f}')

    batch_loss.extend(loss_list)
    mean_epoch_loss.append(sum(loss_list)/len(loss_list))
    print()

fig, ax = plt.subplots(3,1)
fig.tight_layout()
fig.set_size_inches(14,13.5)
ax[0].plot(range(len(batch_loss)), batch_loss,label='model loss',color='green')
ax[0].set_yscale('log')
ax[0].set_title('Model Loss')
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('Loss')
ax[0].legend()

with torch.no_grad():
    X = torch.from_numpy(np.array(training_set.X).astype(np.float32)).to(device)
    X = X.reshape(len(X),1)
    y_predicted = model(X)

ax[1].plot(range(len(grad_norm_list)), grad_norm_list,label='model grad norm',color='green')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Grad Norm')

ax[2].plot(function_dataset.x, function_dataset.y, color='black',label=r"$\frac{\mathrm{sin}(5 \pi x)}{5 \pi x}$")
ax[2].xaxis.set_ticks_position('bottom')
ax[2].yaxis.set_ticks_position('left')
ax[2].set_xlim([0.01,1])
ax[2].set_ylim([-.25,1])
print(y_predicted.cpu())
ax[2].plot(training_set.X, y_predicted.cpu(), color='green', label='model_2')
ax[2].legend()
plt.savefig('./figures/part1_2grad_zero')
plt.show()
