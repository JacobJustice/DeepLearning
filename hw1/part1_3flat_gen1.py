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
# Create 2 models that are the interpolation between two existing models. Plot their
#

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()

#
# hyper parameters
# 
input_size = 28*28
output_size = 10
batch_size = 1024
num_epochs = 100

# MNIST dataset 
train_dataset_1 = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset_1 = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader_1
train_loader_1  = torch.utils.data.DataLoader(dataset=train_dataset_1, 
                                           batch_size=batch_size, 
                                           shuffle=True)
# Data loader_2
train_loader_2  = torch.utils.data.DataLoader(dataset=train_dataset_1, 
                                           batch_size=batch_size//16, 
                                           shuffle=True)

test_loader  = torch.utils.data.DataLoader(dataset=test_dataset_1, 
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

#model_1
num_layers = 4
hidden_size = 22
model_1 = NeuralNet(input_size, num_layers=num_layers, hidden_size=hidden_size, num_classes=output_size).to(device)
optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=.001)
print("model_1.num_parameters()",model_1.num_parameters(),num_layers,hidden_size)

#model 2
num_layers = 4
hidden_size = 22
model_2 = NeuralNet(input_size, num_layers=num_layers, hidden_size=hidden_size, num_classes=output_size).to(device)
optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=.0001)
print("model.num_parameters()",model_2.num_parameters(),num_layers,hidden_size)

# Training Model 1
n_total_steps = len(train_loader_1)
mean_epoch_loss_1 = []
train_batch_loss_1 = []
test_loss_list_1 = []
train_acc_list_1 = []
test_acc_list_1 = []
for epoch in range(num_epochs):
    loss_list_1 = []
    for i, (x_i, y_i) in enumerate(train_loader_1):
        x_i = x_i.reshape(-1, input_size).to(device)
        y_i = y_i.to(device)
        loss_list_1.append(update_model(x_i, y_i, model_1, optimizer_1, criterion, no_unsqueeze=True, no_float=True))

        if (i+1) % 10 == 0:
            print (f'Model 1 Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss_list_1[-1]:.4f}')

#    with torch.no_grad():
#        n_correct = 0
#        n_samples = 0
#        for images, labels in test_loader:
#            images = images.reshape(-1, 28*28).to(device)
#            labels = labels.to(device)
#            outputs_1 = model_1(images)
#            test_loss_list_1.append(criterion(outputs_1, labels).item())
#
#            _, predicted = torch.max(outputs.data, 1)
#            n_samples += labels.size(0)
#            n_correct += (predicted == labels).sum().item()
#        test_acc_list_1.append(n_correct/n_samples)


    train_batch_loss_1.extend(loss_list_1)
    mean_epoch_loss_1.append(sum(loss_list_1)/len(loss_list_1))
    print()

n_total_steps = len(train_loader_2)
mean_epoch_loss_2 = []
train_batch_loss_2 = []
test_loss_list_2 = []
train_acc_list_2 = []
test_acc_list_2 = []
for epoch in range(num_epochs):
    loss_list_2 = []
    for i, (x_i, y_i) in enumerate(train_loader_2):
        x_i = x_i.reshape(-1, input_size).to(device)
        y_i = y_i.to(device)
        loss_list_2.append(update_model(x_i, y_i, model_2, optimizer_2, criterion, no_unsqueeze=True, no_float=True))

        if (i+1) % 100 == 0:
            print (f'Model 2 Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss_list_2[-1]:.4f}')

    train_batch_loss_2.extend(loss_list_2)
    mean_epoch_loss_2.append(sum(loss_list_2)/len(loss_list_2))

def interpolate_models(model_1, model_2, alpha):
    out_model = NeuralNet(input_size, num_layers=num_layers, hidden_size=hidden_size, num_classes=output_size)
    for out_layer, layer_1, layer_2 in zip(out_model.layers, model_1.layers,model_2.layers):
        if hasattr(out_layer, 'weight'):
            out_layer.weight.data = (1-alpha)*layer_1.weight.data + alpha*layer_2.weight.data
    #print(out_model.layers[3].weight.data)
    return out_model

mean_test_loss_list = []
mean_train_loss_list = []
test_acc_list = []
train_acc_list = []
alphas = np.linspace(-1,2,35)
for alpha in alphas:
    print(alpha)
    model_3 = interpolate_models(model_1,model_2,alpha).to(device)

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        train_loss_list = []
        for images, labels in train_loader_1:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model_3(images)
            train_loss_list.append(criterion(outputs, labels).item())

            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
        mean_train_loss_list.append(sum(train_loss_list)/len(train_loss_list))
        train_acc_list.append(n_correct/n_samples)
        print('train_acc_list',train_acc_list)

        n_correct = 0
        n_samples = 0
        test_loss_list = []
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model_3(images)
            test_loss_list.append(criterion(outputs, labels).item())

            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
        mean_test_loss_list.append(sum(test_loss_list)/len(test_loss_list))
        test_acc_list.append(n_correct/n_samples)
        print('test_acc_list',test_acc_list)

fig, ax = plt.subplots(1,1)
fig.tight_layout()
fig.set_size_inches(8,6)

par = ax.twinx()
par.set_ylabel('accuracy')
ax.set_xlabel('alpha')
ax.set_ylabel('cross_entropy')

p1, = ax.plot(alphas, mean_train_loss_list, label="train", color='blue')
p2, = ax.plot(alphas, mean_test_loss_list, label="test", color='blue',linestyle='dashed')
ax.legend()
p3, = par.plot(alphas, train_acc_list, label="train", color='red')
p4, = par.plot(alphas, test_acc_list, label="test", color='red',linestyle='dashed')


#with torch.no_grad():
#    X = torch.from_numpy(np.array(training_set.X).astype(np.float32)).to(device)
#    X = X.reshape(len(X),1)
#    y_predicted = model(X)

#ax[1].plot(range(len(grad_norm_list)), grad_norm_list,label='model grad norm',color='green')
#ax[1].set_xlabel('Iteration')
#ax[1].set_ylabel('Grad Norm')
plt.savefig('./figures/part1_3flat_gen1')
plt.show()
