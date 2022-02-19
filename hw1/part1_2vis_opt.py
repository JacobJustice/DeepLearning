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
batch_size = 400
num_models = 8
learning_rate = 0.001
num_epochs = 3
N_epochs = 1 # after N epochs record parameters

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

def collect_params(model):
    params = []
    for p in model.parameters():
        #print(p)
        params.extend(list(p.flatten().detach().cpu().numpy()))
    return params

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
test_loss_list = []
test_acc_list = []
model_params_loss = []
# train 8 times
for t in range(num_models):
    for epoch in range(num_epochs):
        loss_list = []
        for i, (x_i, y_i) in enumerate(train_loader):
            x_i = x_i.reshape(-1, input_size).to(device)
            y_i = y_i.to(device)
            loss_list.append(update_model(x_i, y_i, model, optimizer, criterion, no_unsqueeze=True, no_float=True))
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
        if epoch % N_epochs == 0:
            print('recording params',epoch,N_epochs)
            model_params_loss.append((collect_params(model),mean_epoch_loss[-1]))

#print("MODEL PARAMS")
#for p in model_params_loss:
#    print(p)

print("converting to numpy")
model_params =[x[0] for x in model_params_loss]
model_loss = [x[1] for x in model_params_loss]
df_params = pd.DataFrame(model_params)
print(df_params)
df_params = StandardScaler().fit_transform(df_params)
df_loss = pd.DataFrame(model_loss)
pca = PCA(n_components=2)

print(df_params)
print(df_loss)
p_components = pca.fit_transform(df_params)
principal_df_params = pd.DataFrame(data=p_components,columns=['pc 1', 'pc 2'])
print(principal_df_params)
#df.to_csv('model_params.csv')

fig, ax = plt.subplots(2,1)
fig.tight_layout()
fig.set_size_inches(14,13.5)
ax[0].scatter(principal_df_params['pc 1'], principal_df_params['pc 2'],label='model loss',color='green')
ax[0].set_yscale('log')
ax[0].set_title('Model Loss')
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('Loss')
ax[0].legend()

#with torch.no_grad():
#    X = torch.from_numpy(np.array(training_set.X).astype(np.float32)).to(device)
#    X = X.reshape(len(X),1)
#    y_predicted = model(X)

plt.savefig('./figures/part1_2vis_opt.png')
#plt.show()
