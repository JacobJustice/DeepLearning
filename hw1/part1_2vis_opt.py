import matplotlib.pyplot as plt
import pandas as pd
import random
import pylab as pl
import numpy as np
from torch.utils.data import Dataset
import math
from pprint import pprint
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
num_models = 8
learning_rate = 0.005
num_epochs = 47
N_epochs = 3 # after N epochs record parameters

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


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

def collect_params(model,first_layer=False):
    params = []
    for p in model.parameters():
        #print(p)
        params.extend(list(p.flatten().detach().cpu().numpy()))
        if first_layer:
            break
    return params

#model
num_layers = 4
hidden_size = 22
model = NeuralNet(input_size,num_layers,hidden_size,output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training
n_total_steps = len(train_loader)
mean_epoch_loss = []
train_batch_loss = []
test_loss_list = []
test_acc_list = []
model_params_loss = [[] for x in range(num_models)]
first_layer_loss = [[] for x in range(num_models)]
# train num_models times
for t in range(num_models):
    for epoch in range(num_epochs):
        loss_list = []
        for i, (x_i, y_i) in enumerate(train_loader):
            x_i = x_i.reshape(-1, 28*28).to(device)
            y_i = y_i.reshape(batch_size).to(device)
            loss_list.append(update_model(x_i, y_i, model, optimizer, criterion,no_unsqueeze=True,no_float=True))
            if (i+1) % 100 == 0:
                print('model',t)
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss_list[-1]:.4f}')

        # Accuracy
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
        accuracy = n_correct/n_samples
        print()
        if epoch % N_epochs == 0:
            print('recording params',epoch,N_epochs)
            first_layer_loss[t].append((collect_params(model,first_layer=True),accuracy))
            model_params_loss[t].append((collect_params(model),accuracy))

#print("MODEL PARAMS")
#for p in model_params_loss:
#    print(p)
colors = ['b','g','r','c','m','y','darkorange','limegreen','grey','brown']
print("converting to numpy")
model_params =[[y[0] for y in x] for x in model_params_loss]
model_acc = [[y[1]*100 for y in x] for x in model_params_loss]
fl_model_params =[[y[0] for y in x] for x in first_layer_loss]
fl_model_acc = [[y[1]*100 for y in x] for x in first_layer_loss]
fig, ax = plt.subplots(1,1)
fig.tight_layout()
fig.set_size_inches(8,6)
# plot whole model
for i, model in enumerate(model_params):
    df_params = pd.DataFrame(model)
    #print(df_params)
    df_params = StandardScaler().fit_transform(df_params)
    df_acc = pd.DataFrame(model_acc[i])
    pca = PCA(n_components=2)

    print('df_params\n',df_params)
    print('df_acc\n',df_acc)
    p_components = pca.fit_transform(df_params)
    principal_df_params = pd.DataFrame(data=p_components,columns=['pc 1', 'pc 2'])
    print(principal_df_params)

    for j, x,y in zip(range(len(principal_df_params['pc 1'])), principal_df_params['pc 1'], principal_df_params['pc 2']):
#        print(i,j)
        ax.annotate(str(int(model_acc[i][j])), xy=(x,y), color=colors[i], fontsize='small',
        horizontalalignment='center', verticalalignment='center')
    ax.scatter(principal_df_params['pc 1'], principal_df_params['pc 2'],label='model_'+str(i),color=colors[i],marker='None')
    plt.savefig('./figures/part1_2vis_opt_whole_model.png')

# plot first layer
fig, ax = plt.subplots(1,1)
fig.tight_layout()
fig.set_size_inches(8,6)
for i, model in enumerate(fl_model_params):
    df_params = pd.DataFrame(model)
    #print(df_params)
    df_params = StandardScaler().fit_transform(df_params)
    df_acc = pd.DataFrame(fl_model_acc[i])
    pca = PCA(n_components=2)

    print('df_params\n',df_params)
    print('df_acc\n',df_acc)
    p_components = pca.fit_transform(df_params)
    principal_df_params = pd.DataFrame(data=p_components,columns=['pc 1', 'pc 2'])
    print(principal_df_params)

    for j, x,y in zip(range(len(principal_df_params['pc 1'])), principal_df_params['pc 1'], principal_df_params['pc 2']):
        ax.annotate(str(int(fl_model_acc[i][j])), xy=(x,y), color=colors[i], fontsize='small',
        horizontalalignment='center', verticalalignment='center')
    ax.scatter(principal_df_params['pc 1'], principal_df_params['pc 2'],label='model_'+str(i),color=colors[i],marker='None')
    plt.savefig('./figures/part1_2vis_opt_first_layer.png')#with torch.no_grad():
