import numpy as np
import scipy.io as sio
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import UNetSAR

import pickle
import os
import matplotlib.pyplot as plt
import json

# config = {
#     "batch_size": 2,
#     "learning_rate": .1,
#     "max_epochs": 10
# }

# -------
# 1d training data
# ---------
# inputs = np.random.rand(5,1,1000) # 5 training samples with 1 channel in 1000-dim space
# y =np.random.rand(1000,5)  # 5 training examples classified as a 0 or 1
# inputs, y = torch.Tensor(inputs), torch.Tensor(y)

# -------
# 2d arc training data
# ---------
# note that some arc grids have different sizes
    # 0a938d79.json is 10x25
with open("../../../../data/ARC/data/training/0a938d79.json", "rb") as f:
    d = json.load(f)
inputs = np.array(d["train"][0]["input"]) # get one training example
inputs = inputs[np.newaxis, np.newaxis, :, :] # adds the batch dimension and channel dim
inputs = torch.Tensor(inputs)
# print(inputs.shape)
# ----------
# to help batch it 
# ----------
# trainloader = torch.utils.data.DataLoader(zip(inputs, y), batch_size=config["batch_size"])
# print('Loaded Training Dataset')

# ----------
# do a run
# ----------
net = UNetSAR()
out = net(inputs)
print(out.shape)

# ----------
# debug layer by layer
# ----------
# layer1_out = net.enc_b1(net.enc_conv1(inputs))
# print(layer1_out.shape)
# layer2_out = net.enc_b2(net.enc_conv2(layer1_out))
# print(layer2_out)

# ----------
# plot
# ----------
# plot the input and plot the output
# plt.plot(inputs[0,0,:]); plt.show()
# plt.plot(out.detach().numpy()[0,0,:]); plt.show()


