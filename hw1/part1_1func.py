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
from neuralnet import NeuralNet
from dataset_objs import FunctionDataset, TimeSeriesDataset

#
# hyper parameters
# 
input_size = 1
output_size = 1
batch_size = 200
learning_rate = 0.001
num_epochs = 1000
function_dataset = FunctionDataset()


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.MSELoss()

xy_train = [x for x in function_dataset.x_y[:
    #int(.9*len(function_dataset.x_y))
    ]]
#print(len(xy_train))
training_set = TimeSeriesDataset([x[0] for x in xy_train], [y[1] for y in xy_train])

#xy_test = [x for x in function_dataset.x_y[int(.9*len(function_dataset.x_y)):]]
#print(len(xy_test))
#testing_set = TimeSeriesDataset([x[0] for x in xy_test], [y[1] for y in xy_test])

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)

# helper function for updating model based on loss of given x inputs and correct y outputs
# returns loss from criterion
def update_model(x_i, y_i, model, optimizer, criterion):
    output = model(x_i.float())
    y_i = y_i.unsqueeze(1)
    loss = criterion(output, y_i.float())
    out_loss = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 10 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {out_loss:.4f}')

    return out_loss


#model_1
num_layers = 20
hidden_size = 10
model_1 = NeuralNet(input_size, num_layers=num_layers, hidden_size=hidden_size, num_classes=output_size).to(device)
optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=learning_rate)
print("model_1.num_parameters()",model_1.num_parameters(),num_layers,hidden_size)

#model_2
num_layers = 4
hidden_size = 22
model_2 = NeuralNet(input_size, num_layers=num_layers, hidden_size=hidden_size, num_classes=output_size).to(device)
optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=learning_rate)
print("model_2.num_parameters()",model_2.num_parameters(),num_layers,hidden_size)

#model_3
num_layers = 80
hidden_size = 5
model_3 = NeuralNet(input_size, num_layers=num_layers, hidden_size=hidden_size, num_classes=output_size).to(device)
optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=learning_rate)
print("model_3.num_parameters()",model_3.num_parameters(),num_layers,hidden_size)

# Training
n_total_steps = len(train_loader)
mean_epoch_loss_1 = []
mean_epoch_loss_2 = []
mean_epoch_loss_3 = []
for epoch in range(num_epochs):
    loss_list_1 = []
    loss_list_2 = []
    loss_list_3 = []
    for i, (x_i, y_i) in enumerate(train_loader):
        x_i = x_i.reshape(batch_size, input_size).to(device)
        y_i = y_i.to(device)
        loss_list_1.append(update_model(x_i, y_i, model_1, optimizer_1, criterion))
        loss_list_2.append(update_model(x_i, y_i, model_2, optimizer_2, criterion))
        loss_list_3.append(update_model(x_i, y_i, model_3, optimizer_3, criterion))

    mean_epoch_loss_1.append(sum(loss_list_1)/len(loss_list_1))
    mean_epoch_loss_2.append(sum(loss_list_2)/len(loss_list_2))
    mean_epoch_loss_3.append(sum(loss_list_3)/len(loss_list_3))
    print()

fig, ax = plt.subplots(2,1)
fig.tight_layout()
fig.set_size_inches(8,6)
ax[0].plot(range(len(mean_epoch_loss_1)), mean_epoch_loss_1,label='model_1 loss',color='red')
ax[0].plot(range(len(mean_epoch_loss_2)), mean_epoch_loss_2,label='model_2 loss',color='green')
ax[0].plot(range(len(mean_epoch_loss_3)), mean_epoch_loss_3,label='model_3 loss',color='blue')
ax[0].set_yscale('log')
ax[0].set_title('Model Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()

with torch.no_grad():
    X = torch.from_numpy(np.array(training_set.X).astype(np.float32)).to(device)
    X = X.reshape(len(X),1)
    y_predicted_1 = model_1(X)
    y_predicted_2 = model_2(X)
    y_predicted_3 = model_3(X)

ax[1].plot(function_dataset.x, function_dataset.y, color='black',label=r"$\frac{\mathrm{sin}(5 \pi x)}{5 \pi x}$")
ax[1].xaxis.set_ticks_position('bottom')
ax[1].yaxis.set_ticks_position('left')
ax[1].set_xlim([0.01,1])
ax[1].set_ylim([-.25,1])
ax[1].plot(training_set.X, y_predicted_1.cpu(), color='red', label='model_1')
ax[1].plot(training_set.X, y_predicted_2.cpu(), color='green', label='model_2')
ax[1].plot(training_set.X, y_predicted_3.cpu(), color='blue', label='model_3')
ax[1].legend()
plt.savefig('figures/part1_1.png',dpi=200)
plt.show()
