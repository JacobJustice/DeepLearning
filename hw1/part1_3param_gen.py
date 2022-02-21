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

#
# hyper parameters
# 
input_size = 28*28
output_size = 10
batch_size = 200
learning_rate = 0.001
num_epochs = 1

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size*50, 
                                          shuffle=False)

def get_grad_norm(model):
    grad_all = 0.0
    for p in model.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        grad_all += grad
    return grad_all ** 0.5

#model
models = []
optimizers = []
num_models = 15
for i in range(num_models):
    num_layers = 4+(1*(i//5))
    hidden_size = 5+3*i
    models.append(NeuralNet(input_size, num_layers=num_layers, hidden_size=hidden_size, num_classes=output_size).to(device))
    optimizers.append(torch.optim.Adam(models[i].parameters(), lr=learning_rate))
#print("model.num_parameters()",model.num_parameters(),num_layers,hidden_size)

# Training
n_total_steps = len(train_loader)
train_epoch_loss_list = []# [[] for x in range(len(models))]
test_epoch_loss_list = []# [[] for x in range(len(models))]
train_acc_list = []# [[] for x in range(len(models))]
test_acc_list = []# [[] for x in range(len(models))]
for epoch in range(num_epochs):
    train_loss_list = [[] for x in range(len(models))]
    test_loss_list = [[] for x in range(len(models))]
    for m in range(len(models)):
        for i, (x_i, y_i) in enumerate(train_loader):
            x_i = x_i.reshape(-1, input_size).to(device)
            y_i = y_i.to(device)
            train_loss_list[m].append(update_model(x_i, y_i, models[m], optimizers[m], criterion, no_unsqueeze=True, no_float=True))
            test_loss_list[m].append(criterion(models[m](x_i), y_i).item())
            if (i+1) % 300 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {train_loss_list[m][-1]:.4f}')

# compute final loss and accuracy
for m in range(len(models)):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in train_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = models[m](images)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
        train_acc_list.append(n_correct/n_samples)

        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = models[m](images)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
        test_acc_list.append(n_correct/n_samples)

    train_epoch_loss_list.append(sum(train_loss_list[m])/len(train_loss_list[m]))
    test_epoch_loss_list.append(sum(test_loss_list[m])/len(test_loss_list[m]))

#colors = ['red','blue','green','orange','pink','magenta','brown','lime','purple','cornflowerblue'
#,'darkorange','seagreen','midnightblue','crimson','rosybrown','olive','yellow','slategray','teal','thistle']

fig, ax = plt.subplots(2,1)
fig.tight_layout()
fig.set_size_inches(10,8)
ax[0].scatter([m.num_parameters() for m in models], train_epoch_loss_list,label='train loss',color='blue')
ax[0].scatter([m.num_parameters() for m in models], test_epoch_loss_list,label='test loss',color='orange')
ax[0].set_yscale('log')
ax[0].set_title('Model Loss')
ax[0].set_xlabel('Number of Parameters')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].scatter([m.num_parameters() for m in models], train_acc_list,label='train acc',color='blue')
ax[1].scatter([m.num_parameters() for m in models], test_acc_list,label='test acc',color='orange')
ax[1].set_title('Model Accuracy')
ax[1].set_xlabel('Number of Parameters')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

plt.savefig('./figures/part1_3param_gen.png')
plt.show()
