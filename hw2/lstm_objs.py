import matplotlib.pyplot as plt
import time
from convert_caption import convert_caption
import random
import pylab as pl
import numpy as np
from torch.utils.data import Dataset
import math
from gen_json import generate_reverse
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json

class Decoder_LSTM(nn.Module):
    def __init__(self,input_dim, hidden_dim, vocab_size):
        print(input_dim, hidden_dim, vocab_size)
        super(Decoder_LSTM, self).__init__()
        # lstm outputs hidden state of size hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)

        # linear layer maps from LSTM hidden size to the vocab space
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.linear(out)
        return out

class VideoDataset(Dataset):
    def __init__(self,X,Y):
        assert len(X) == len(Y)
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        _x = self.X[index]
        _y = self.Y[index]
        return _x, _y


