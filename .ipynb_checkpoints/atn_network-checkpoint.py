# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

# ATN_c for cifar10
class ATN_c(nn.Module):

    def __init__(self):
        super(ATN_c, self).__init__()
        # Convolute operation 3 times
        # (batch_size, 3, 32, 32) -> (batch_size, 12, 16, 16)
        #self.conv1 = nn.Conv2d(3, 3, 3)
        self.conv1 = nn.Conv2d(3, 12, 4, stride=2, padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        # (batch_size, 12, 16, 16) -> (batch_size, 24, 8, 8)
        #self.conv2 = nn.Conv2d(3, 3, 3)
        self.conv2 = nn.Conv2d(12, 24, 4, stride=2, padding=1)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        # (batch_size, 24, 8, 8) -> (batch_size, 48, 4, 4)
        #self.conv3 = nn.Conv2d(3, 3, 3)
        self.conv3 = nn.Conv2d(24, 48, 4, stride=2, padding=1)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        # Deconvolute 2 times
        # (batch_size, 48, 4, 4) -> (batch_size, 24, 8, 8)
        #self.deconv1 = nn.ConvTranspose2d(3, 3, 3)
        self.deconv1 = nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1)
        torch.nn.init.kaiming_normal_(self.deconv1.weight)
        
        # (batch_size, 24, 8, 8) -> (batch_size, 12, 16, 16)
        #self.deconv2 = nn.ConvTranspose2d(3, 3, 3)
        self.deconv2 = nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1)
        torch.nn.init.kaiming_normal_(self.deconv2.weight)
        
        # (batch_size, 12, 16, 16) -> (batch_size, 3, 32, 32)
        #self.deconv3 = nn.ConvTranspose2d(3, 3, 3)
        self.deconv3 = nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1)
        torch.nn.init.kaiming_normal_(self.deconv3.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        perturbed_x = torch.tanh(self.deconv3(x))
        return perturbed_x
        
        #return x
