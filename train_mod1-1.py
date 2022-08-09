'''
1stage opamp train NN
Used for predicting OPAMP MOSFET sizes depending on performances
2022 Junsang Yoo

train NN model 1 takes performances as input and opamp size parameters as output
loss function compares predicted opamp size parameters and ground truth opamp size parameters that is in dataset csv file
in 1-1, normalization for input and output layer and loss calculation is considered
'''
# todo: some consistancy to variable name

## imports
import numpy as np
import os
import matplotlib.pyplot as plt

# pyspice import
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from PySpice.Spice.Library import SpiceLibrary

# import simulation function
from simulation import simulate

# pytorch import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(dev)

## Dataset Loader
# os.chdir(os.path.dirname(__file__))     # change directory to where current file is
spice_library = SpiceLibrary('./')      # use TSMC 180nm library file (measured)

class LoadData(Dataset):
    def __init__(self, filename):
        with open(filename, 'r') as f:
            # name of each data
            keys=f.readline()
            keys=keys[:-2].split(',')
            self.param_names=keys[:6]
            self.perf_names=keys[6:]

            # dataset
            rawdata=f.read()
            rawdata=rawdata.split()
            data=[]
            for line in rawdata:
                data.append(line[:-1].split(','))
            data=np.array(data,dtype=np.float32)
            self.X=data[:,6:]       # performances
            self.Y=data[:,:6]       # circuit parameters (sizes)

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        x=torch.FloatTensor(self.X[idx], device=dev)
        y=torch.FloatTensor(self.Y[idx], device=dev)
        return x,y

    def getrand(self,num):
        # get 'num' amount of random data
        l=len(self.X)
        randidx=np.arange(l)
        np.random.shuffle(randidx)
        randx=torch.FloatTensor(self.X[randidx[:num]], device=dev)
        randy=torch.FloatTensor(self.Y[randidx[:num]], device=dev)
        return randx,randy

# Split Dataset Into Train, Valid, Test
# (train : valid : test) = (6 : 2 : 2)
data=LoadData('dataset.csv')
ntrain=len(data)//10*6; nvalid=len(data)//10*2; ntest=len(data)-ntrain-nvalid
traind, validd, testd = random_split(data, [ntrain,nvalid,ntest])

# Load Dataset
BATCH=256
traindata=DataLoader(traind, batch_size=BATCH, shuffle=True)
validdata=DataLoader(validd, batch_size=100)
testdata=DataLoader(testd, batch_size=100)
print(f'data splitted. ntrain : nvalid : ntest = 6 : 2 : 2 = {ntrain} : {nvalid} : {ntest}')

## Normalization
'''
Normalization process
*****GBP should be dealt in logarithmic scale?*****
Normalization layer before input layer and after output layer when model definition
uses mean and std values found in this section
'''
x1000=data.getrand(1000)[0]; x1000[:,-1]=torch.log10(x1000[:,-1])   # log10 GBP before normalization
xstd=torch.std(x1000,axis=0)
xmean=torch.mean(x1000,axis=0)
y1000=data.getrand(1000)[1]
ystd=torch.std(y1000,axis=0)
ymean=torch.mean(y1000,axis=0)
# normx1000=(x1000-xmean)/xstd
# normy1000=(y1000-ymean)/ystd

## Neural Net
class MODEL(nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()
        self.hidden=nn.Sequential(
            nn.Linear(in_features=17, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            )
        self.outlayer=nn.Linear(in_features=16, out_features=6)

    def forward(self, x):
        x[:,-1]=torch.log10(x[:,-1])    # log10 GBP before normalization
        in_norm=(x-xmean)/xstd
        out = self.hidden(in_norm)
        out = self.outlayer(out)
        out_denorm=out*ystd+ymean
        return out_denorm

model=MODEL()
model.to(dev)
torch.cuda.empty_cache()

## Train
EPOCH=50
opt = torch.optim.Adam(model.parameters())
loss_fn = nn.MSELoss().to(dev)
trainloss=[]
validloss=[]

# Iterations
for epoch in range(EPOCH):
    model.train()
    meanloss=[]

    # train
    for batch_idx, (batch_x, batch_y) in enumerate(traindata):
        opt.zero_grad()         # clear gradient
        batch_x=batch_x.to(dev); batch_y=batch_y.to(dev)
        pred=model(batch_x)
        npred=(pred-ymean)/ystd; nbatch_y=(batch_y-ymean)/ystd  # normalization before loss
        loss=loss_fn(npred, nbatch_y)
        loss.backward()         # Backward Propagation: update NN weight and bias
        opt.step()              # next step for optimizer

        # to keep data
        meanloss.append(loss.detach().numpy())
    trainloss.append(np.mean(meanloss))

    # validation
    for batch_idx, (batch_x, batch_y) in enumerate(validdata):
        batch_x=batch_x.to(dev); batch_y=batch_y.to(dev)
        pred=model(batch_x)
        npred=(pred-ymean)/ystd; nbatch_y=(batch_y-ymean)/ystd  # normalization before loss
        loss=loss_fn(npred, nbatch_y)

        # to keep data
        meanloss.append(loss.detach().numpy())
    validloss.append(np.mean(meanloss))

## Test
for batch_idx, (batch_x, batch_y) in enumerate(testdata):
    batch_x=batch_x.to(dev); batch_y=batch_y.to(dev)
    pred=model(batch_x)
    npred=(pred-ymean)/ystd; nbatch_y=(batch_y-ymean)/ystd  # normalization before loss
    loss=loss_fn(npred,nbatch_y)
    meanloss.append(loss.detach().numpy())

testloss=np.mean(meanloss)

print('\nbatch_x:')
print(batch_x[:10])
print('\npred:')
print(pred[:10])
print('\nbatch_y:')
print(batch_y[:10])

## Results
plt.plot(trainloss)
plt.show()