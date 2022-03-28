import matplotlib.pyplot as plt
import random
import time
from convert_caption import convert_caption, construct_caption, make_dataset
import random
import pylab as pl
import numpy as np
from torch.utils.data import Dataset
import math
from gen_json import generate_reverse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ExponentialLR
from lstm_objs import Decoder_LSTM, VideoDataset
import os
import json

# Sequential ( LSTM -> Softmax )k
torch.set_default_dtype(torch.double)
# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# load word_tokens
with open('./word_tokens.json') as fp:
    word_tokens = json.load(fp)
    reverse_word_tokens = generate_reverse(word_tokens)

# load dataset
print('loading dataset...')
training_data = make_dataset('./data/training_data/feat/','./data/training_label.json')
#random.seed(10)
#random.shuffle(training_data)
training_X = [x for x, y in training_data]
training_Y = [y for x, y in training_data]
training_set = VideoDataset(training_X,training_Y)
#val_X = [x for x, y in training_data[int(.9*len(training_data)):]]
#val_Y = [y for x, y in training_data[int(.9*len(training_data)):]]
#val_set = VideoDataset(val_X,val_Y)

batch_size = 100
learning_rate = 0.001
num_epochs = 1000
train_loader = torch.utils.data.DataLoader(dataset=training_set
                                            , batch_size=batch_size
                                            , shuffle=True
                                            , pin_memory=True)

#val_loader = torch.utils.data.DataLoader(dataset=val_set
#                                            , batch_size=batch_size
#                                            , shuffle=True
#                                            , pin_memory=True)


print("done loading data!")

print("setting up model...")
# setup model, optimizer etc.
n_total_steps = len(train_loader)
output_size = len(word_tokens.keys())
model = Decoder_LSTM(4096, 128, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ExponentialLR(optimizer,gamma=.999)
print("done setting up model!")


def validate():
    step_loss = []
    for i, (x, y_list) in enumerate(val_loader):
        x = x.to(device)
        y = torch.Tensor(np.array([np.argmax(word_vec) for word_vec in convert_caption(y_,word_tokens, reverse_word_tokens) for y_ in y_list]))
        print('y',y)
        y = y.to(device)
        model.zero_grad()

        y_out = model(x)

        loss = criterion(y_out, y)
        out_loss = loss.item()
        step_loss.append(out_loss)
        print(i,len(val_loader),out_loss)

    return sum(step_loss)/len(step_loss)

epoch_loss = []
for epoch in range(num_epochs):
    step_loss = []
    for i, (x, y_list) in enumerate(train_loader):
        x = x.to(device)
        #print(y_list)
        y = torch.Tensor(np.array([convert_caption(y_,word_tokens, reverse_word_tokens) for y_ in y_list[1]]))
#        print('y',y,y.shape)
        y = y.to(device)
#        print(x)
#        print(x.shape)
        model.zero_grad()

        y_out = model(x)

        loss = criterion(y_out, y)
        out_loss = loss.item()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            prob_out = F.softmax(y_out[0], dim=1)
            step_loss.append(out_loss)
            test_caption = construct_caption(prob_out.detach().cpu(),word_tokens)
            print ()
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {out_loss:.4f}')
            print ('file : ', y_list[0][0])
            print ('y    :', y_list[1][0])
            print ('y_hat:', test_caption)
            print ('# of words: ',len(test_caption.split()))
           
#    print('validating epoch...')
#    val_loss = validate()
#    print('validation complete!', val_loss)
    scheduler.step()
    print('new lr:',scheduler.get_last_lr())
    epoch_loss.append(sum(step_loss)/len(step_loss))
        
       

print("saving model...")
torch.save(model, 'model_'+str(int(time.time()))+'.model')
print("done saving model!")
