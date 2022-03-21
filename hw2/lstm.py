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

import os
import json


a = np.load('./data/training_data/feat/a-cek0mvXxE_15_25.avi.npy')

# pick from distribution : [.01, .2, .001, .04 ...]

# Sequential ( LSTM -> Softmax )

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
