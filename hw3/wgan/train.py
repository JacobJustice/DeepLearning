import torch
import matplotlib.pyplot as plt
from torch import _pin_memory, nn 
from torch import optim
from torchvision import transforms
import torchvision.utils as vutils
import pickle
import os
import sys
from utils import load_cifar_10_data
from pprint import pprint
from wgan import Generator, Discriminator, Dataset, weights_init
import pylab
import numpy as np

NUM_EPOCHS=500
BATCH_SIZE=100
LEARNING_RATE=.00005
Z_VECTOR = 100

cifar_10_dir = '../cifar-10-python/cifar-10-batches-py'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
    load_cifar_10_data(cifar_10_dir)

transform = transforms.ToTensor()
print(type(train_data))

dset = Dataset(np.concatenate((train_data, test_data)), np.concatenate((train_labels, test_labels)), transform)
#print(dset.X)
dataloader = torch.utils.data.DataLoader(dataset=dset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            )


criterion = nn.BCELoss()

fixed_noise = torch.randn(64, Z_VECTOR, 1, 1, device=device)

# generator
netG = Generator().to(device)
netG.apply(weights_init)
print(netG)
optimizerG = optim.RMSprop(netG.parameters(), lr=LEARNING_RATE)

# discriminator
netD = Discriminator().to(device)
netD.apply(weights_init)
print(netD)
optimizerD = optim.RMSprop(netD.parameters(), lr=LEARNING_RATE)

real_label = torch.FloatTensor([1]).to(device)
fake_label = torch.FloatTensor([-1]).to(device)
img_list = []
G_losses = []
D_losses = []

iters = 0
print("Starting Training")
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(dataloader, start=0):
        real_data=data[0].to(device)
        #print(real_data)
        b_size = real_data.size(0)

        netD.zero_grad()
        output = netD(real_data).view(-1)
        errD_real = output.mean(0).view(1)
        errD_real.backward(real_label)
        D_x = output.mean().item()

        # gen fake_data
        noise = torch.randn(b_size, Z_VECTOR, 1, 1, device=device)
        fake_data = netG(noise)

        output = netD(fake_data.detach()).view(-1)
        errD_fake = output.mean(0).view(-1)
        errD_fake.backward(fake_label)
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()

        output = netD(fake_data).view(-1)
        errG = output.mean().mean(0).view(1)
        errG.backward(real_label)
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i%50==0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, NUM_EPOCHS, i, len(dataloader),
                     errD.item(), -errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on a fixed noise.
        if (iters % 100 == 0) or ((epoch == NUM_EPOCHS-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake_data = netG(fixed_noise).detach().cpu()
            #img_list.append()
            img_grid = vutils.make_grid(fake_data, padding=2, normalize=True)
            plt.imshow(img_grid.permute(1,2,0))
            plt.savefig('myimage.png')

        iters += 1

    # Save the model.
    if epoch % 2 == 0:
        torch.save({
            'generator' : netG.state_dict(),
            'discriminator' : netD.state_dict(),
            'optimizerG' : optimizerG.state_dict(),
            'optimizerD' : optimizerD.state_dict(),
            }, 'model/model_epoch_{}.pth'.format(epoch))
