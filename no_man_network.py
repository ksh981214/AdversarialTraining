# Numpy
import numpy as np

import random

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from config import config



class img_encoder(nn.Module):
    def __init__(self):
        super(img_encoder,self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),# [batch, 12, 16, 16]
            nn.Conv2d(12, 24, 4, stride=2, padding=1)# [batch, 24, 8, 8]
        )
        
    def forward(self,x):
        return self.encoder(x)
    
class img_decoder(nn.Module):
    def __init__(self):
        super(img_decoder,self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(        
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1)  # [batch, 3, 32, 32]
        )
        
    def forward(self,x):
        #residual block
        x = 6*x
        return self.decoder(x)
    
    
class label_encoder_feature_integration(nn.Module):
    def __init__(self):
        super(label_encoder_feature_integration, self).__init__()
        #Input [batch, 34, 8, 8]
        self.conv= nn.Conv2d(34,24,1,stride=1,padding=0)
        self.target_label = np.zeros(config.num_classes)
        #Output [batch,24,8,8]
        
    def forward(self, img_feature,t):
        #t: target label
        #img_feature: [batch,24,8,8]
        #T: [batch,num_class,8,8]
        batch_size = img_feature.shape[0]
        H = img_feature.shape[2]
        W = img_feature.shape[3]
        
        #print(H,W)
        self.target_label[t] = 1
        T = torch.Tensor(self.target_label)
        T = T.expand(batch_size,H, W,config.num_classes)
        T = T.reshape(batch_size,config.num_classes, H, W)

        M= torch.cat((torch.Tensor(img_feature), T), dim =1) #Output [batch,34,8,8]
        
        M_dot= self.conv(M) #[batch,24,8,8]
        return M_dot
#MANc
class Multi_target_Adversarial_Network(nn.Module):
    def __init__(self):
        super(Multi_target_Adversarial_Network, self).__init__()
        self.img_encoder = img_encoder()
        self.img_decoder = img_decoder()
        self.label_encoder_feature_integration=label_encoder_feature_integration()
    
    def forward(self, x, t):
        
        M = self.img_encoder.forward(x) #32 x 3 x 32 x 32
        #print("M_size: ", M.size())
        M_dot = self.label_encoder_feature_integration.forward(M,t)
        #print("M_dot_size: ", M_dot.size())
        adv_img = self.img_decoder(M_dot)
        #print("adv_img_size: ", adv_img.size())
        return adv_img #tensor
        