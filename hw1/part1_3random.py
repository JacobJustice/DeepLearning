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
#from dataset_objs import FunctionDataset, TimeSeriesDataset


#
# hyper parameters
# 
input_size = 28*28
output_size = 10
batch_size = 200
learning_rate = 0.001
num_epochs = 100

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)
print(train_dataset)
random.shuffle(train_dataset.targets)
print(train_dataset.targets)

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
num_layers = 4
hidden_size = 22
model = NeuralNet(input_size, num_layers=num_layers, hidden_size=hidden_size, num_classes=output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print("model.num_parameters()",model.num_parameters(),num_layers,hidden_size)

# Training
n_total_steps = len(train_loader)
mean_epoch_loss = []
train_batch_loss = []
grad_norm_list = [get_grad_norm(model)]
test_loss_list = []
test_acc_list = []
for epoch in range(num_epochs):
    loss_list = []
    for i, (x_i, y_i) in enumerate(train_loader):
        x_i = x_i.reshape(-1, input_size).to(device)
        y_i = y_i.to(device)
        loss_list.append(update_model(x_i, y_i, model, optimizer, criterion, no_unsqueeze=True, no_float=True))
#        print(len(test_loss_list))
        grad_norm_list.append(get_grad_norm(model))
        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss_list[-1]:.4f}')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_loss_list.append(criterion(outputs, labels).item())

            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()


    train_batch_loss.extend(loss_list)
    mean_epoch_loss.append(sum(loss_list)/len(loss_list))
    print()

fig, ax = plt.subplots(1,1)
fig.tight_layout()
fig.set_size_inches(14,13.5)
print(mean_epoch_loss)
print(test_loss_list)
ax.plot(range(len(mean_epoch_loss)), mean_epoch_loss,label='model loss',color='green')
ax.plot(range(len(test_loss_list)), test_loss_list,label='model loss',color='orange')
ax.set_yscale('log')
ax.set_title('Model Loss')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.legend()

#with torch.no_grad():
#    X = torch.from_numpy(np.array(training_set.X).astype(np.float32)).to(device)
#    X = X.reshape(len(X),1)
#    y_predicted = model(X)

#ax[1].plot(range(len(grad_norm_list)), grad_norm_list,label='model grad norm',color='green')
#ax[1].set_xlabel('Iteration')
#ax[1].set_ylabel('Grad Norm')
plt.savefig('./figures/part1_3random')
plt.show()
