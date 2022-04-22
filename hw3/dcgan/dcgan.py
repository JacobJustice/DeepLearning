import torch
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np

class Dataset(Dataset):
    def __init__(self,X,Y,transform,device='cuda:0'):
        assert len(X) == len(Y)
        self.X = X/255
        self.X = self.X.astype(np.float32)
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        _x = self.X[index]
        _y = self.Y[index]
        return _x, _y

class Generator(nn.Module):
    def __init__(self, input_size=100, gen_features=32) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(input_size, gen_features*8
                                        , kernel_size=4
                                        , stride=1
                                        , padding=0
                                        , bias=False)
        self.bn1 = nn.BatchNorm2d(gen_features*8)

        self.conv2 = nn.ConvTranspose2d(gen_features*8, gen_features*4
                                        , kernel_size=4
                                        , stride=2
                                        , padding=1
                                        , bias=False)
        self.bn2 = nn.BatchNorm2d(gen_features*4)

        self.conv3 = nn.ConvTranspose2d(gen_features*4, gen_features*2
                                        , kernel_size=4
                                        , stride=2
                                        , padding=1
                                        , bias=False)
        self.bn3 = nn.BatchNorm2d(gen_features*2)

        self.conv4 = nn.ConvTranspose2d(gen_features*2, gen_features
                                        , kernel_size=4
                                        , stride=2
                                        , padding=1
                                        , bias=False)
        self.bn4 = nn.BatchNorm2d(gen_features)

        self.conv5 = nn.ConvTranspose2d(gen_features, 3
                                       , kernel_size=4
                                       , stride=1
                                       , padding=1
                                       , bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = torch.tanh(self.conv5(x))
        # x = torch.tanh(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, num_channels=3, input_size=100, hidden_size=32) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, hidden_size,
            4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(hidden_size, hidden_size*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_size*2)

        self.conv3 = nn.Conv2d(hidden_size*2, hidden_size*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_size*4)

        self.conv4 = nn.Conv2d(hidden_size*4, hidden_size*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(hidden_size*8)

        self.conv5 = nn.Conv2d(hidden_size*8, 1, 2, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

        x = torch.sigmoid(self.conv5(x))

        return x

def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)