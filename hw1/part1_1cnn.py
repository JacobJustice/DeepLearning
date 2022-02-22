import matplotlib.pyplot as plt
import random
import pylab as pl
import numpy as np
from torch.utils.data import Dataset
import math
from neuralnet import update_model

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#
# hyper parameters
# 
input_size = 1
output_size = 1
batch_size = 200
learning_rate = 0.001
num_epochs = 500

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                                          batch_size=batch_size, 
                                          shuffle=False)

#examples = iter(test_loader)
#example_data, example_targets = examples.next()

#for i in range(6):
#    plt.subplot(2,3,i+1)
#    plt.imshow(example_data[i][0], cmap='gray')

#print(example_data.shape)

model_1 = nn.Sequential(
	nn.Conv2d(1, 6, 5),
	nn.ReLU(),
	nn.MaxPool2d(2, 2),
	nn.Conv2d(6,16,5),
	nn.ReLU(),
	nn.MaxPool2d(2, 2),
	nn.Flatten(),
	nn.Linear(16*4*4, 120),
	nn.ReLU(),
	nn.Linear(120,84),
	nn.ReLU(),
	nn.Linear(84,10)
	).to(device)

model_2 = nn.Sequential(
	nn.Conv2d(1, 6, 5),
	nn.ReLU(),
	nn.MaxPool2d(2, 2),

	nn.Conv2d(6,16,5),
	nn.ReLU(),
	nn.MaxPool2d(2, 2),

	nn.Conv2d(16,10,3),
	nn.ReLU(),
	nn.MaxPool2d(2, 2),
	nn.Flatten(),

	nn.Linear(10*1*1, 60),
	nn.ReLU(),
	nn.Linear(60,42),
	nn.ReLU(),
	nn.Linear(42,10)
	).to(device)
conv1 = nn.Conv2d(1, 6, 5)
relu = nn.ReLU()
pool = nn.MaxPool2d(2, 2)
fc1 = nn.Linear(10*1*1, 120)
fc2 = nn.Linear(84,10)

criterion = nn.CrossEntropyLoss()
optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=learning_rate)
optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=learning_rate)

# Training
n_total_steps = len(train_loader)
mean_epoch_loss_1 = []
mean_epoch_loss_2 = []
mean_epoch_test_acc_1 = []
mean_epoch_test_acc_2 = []
mean_epoch_train_acc_1 = []
mean_epoch_train_acc_2 = []
#mean_epoch_loss_3 = []
for epoch in range(num_epochs):
    loss_list_1 = []
    loss_list_2 = []
#    loss_list_3 = []
    for i, (x_i, y_i) in enumerate(train_loader):
        x_i = x_i.to(device)
        y_i = y_i.to(device)
        loss_list_1.append(update_model(x_i, y_i, model_1, optimizer_1, criterion))
        loss_list_2.append(update_model(x_i, y_i, model_2, optimizer_2, criterion))
#        loss_list_3.append(update_model(x_i, y_i, model_3, optimizer_3, criterion))
   
    with torch.no_grad():
        n_correct_1 = 0
        n_correct_2 = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs_1 = model_1(images)
            outputs_2 = model_2(images)
            # max returns (value ,index)
            _, predicted_1 = torch.max(outputs_1.data, 1)
            _, predicted_2 = torch.max(outputs_2.data, 1)
            n_samples += labels.size(0)
            n_correct_1 += (predicted_1 == labels).sum().item()
            n_correct_2 += (predicted_2 == labels).sum().item()
        test_acc_1 = n_correct_1 / n_samples
        test_acc_2 = n_correct_2 / n_samples
        mean_epoch_test_acc_1.append(test_acc_1)
        mean_epoch_test_acc_2.append(test_acc_2)
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs_1 = model_1(images)
            outputs_2 = model_2(images)
            # max returns (value ,index)
            _, predicted_1 = torch.max(outputs_1.data, 1)
            _, predicted_2 = torch.max(outputs_2.data, 1)
            n_samples += labels.size(0)
            n_correct_1 += (predicted_1 == labels).sum().item()
            n_correct_2 += (predicted_2 == labels).sum().item()
        train_acc_1 = n_correct_1 / n_samples
        train_acc_2 = n_correct_2 / n_samples
        mean_epoch_train_acc_1.append(train_acc_1)
        mean_epoch_train_acc_2.append(train_acc_2)
    mean_epoch_loss_1.append(sum(loss_list_1)/len(loss_list_1))
    mean_epoch_loss_2.append(sum(loss_list_2)/len(loss_list_2))

#    mean_epoch_loss_3.append(sum(loss_list_3)/len(loss_list_3))
    print()

fig, ax = plt.subplots(2,1)
fig.tight_layout()
fig.set_size_inches(14,13.5)
ax[0].plot(range(len(mean_epoch_loss_1)), mean_epoch_loss_1,label='model_1 loss',color='red')
ax[0].plot(range(len(mean_epoch_loss_2)), mean_epoch_loss_2,label='model_2 loss',color='green')
#ax[0].plot(range(len(mean_epoch_loss_3)), mean_epoch_loss_3,label='model_3 loss',color='blue')
ax[0].set_yscale('log')
ax[0].set_title('Model Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(range(len(mean_epoch_test_acc_1)), mean_epoch_test_acc_1,label='model_1 acc test',color='red')
ax[1].plot(range(len(mean_epoch_test_acc_2)), mean_epoch_test_acc_2,label='model_2 acc test',color='green')
ax[1].plot(range(len(mean_epoch_train_acc_1)), mean_epoch_train_acc_1,label='model_1 acc train',color='orange')
ax[1].plot(range(len(mean_epoch_train_acc_2)), mean_epoch_train_acc_2,label='model_2 acc train',color='blue')
ax[1].set_title('Model Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

plt.savefig('figures/part1_1cnn.png')
plt.show()
