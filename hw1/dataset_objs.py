import numpy as np
from torch.utils.data import Dataset
import math

class FunctionDataset(Dataset):
    def __init__(self):
        self.x = np.linspace(-5,5,2000)
        x_term = 5*math.pi*self.x
        self.y = np.array(list(map(math.sin,x_term)))/x_term
        self.x_y = [(x,y) for x, y in zip(self.x, self.y)]

    def __len__(self):
        return len(self.x_y)

    def __getitem__(self, index):
        _x = self.x_y[index][0]
        _y = self.x_y[index][1]

        return _x, _y

class TimeSeriesDataset(Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        _x = self.X[index]
        _y = self.Y[index]
        return _x, _y


