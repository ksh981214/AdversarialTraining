<<<<<<< HEAD
import torch
import torch.nn as nn
from utils import reranking

from config import config

class Loss(torch.autograd.Function):
    # def __init__():
    # parameters
    # y: label --> (batch_size, # of classes)
    # y_hat: result of forward propagation --> (batch_size, # of classes)
    def L2_loss(y_hat, y):
        loss = torch.mean((y_hat - y)**2)
        return loss

    def ATN_loss(x_hat, x, y_hat, y, t, alpha=config.ALPHA, beta=config.BETA):
        criterion = nn.MSELoss()
        # perceptual similarity loss
        p_s_loss = beta * criterion(x_hat, x)
        # wrong classification loss
        #w_c_loss = criterion(y_hat, reranking(y, t, alpha))
        # total loss
        #loss = p_s_loss + w_c_loss
        return p_s_loss
=======
import torch
import torch.nn as nn
from utils import reranking

from config import config

class Loss(torch.autograd.Function):
    # def __init__():
    # parameters
    # y: label --> (batch_size, # of classes)
    # y_hat: result of forward propagation --> (batch_size, # of classes)
    def L2_loss(y_hat, y):
        loss = torch.mean((y_hat - y)**2)
        return loss

    def ATN_loss(x_hat, x, y_hat, y, t, alpha=config.ALPHA, beta=config.BETA):
        criterion = nn.MSELoss()
        # perceptual similarity loss
        p_s_loss = beta * criterion(x_hat, x)
        # wrong classification loss
        w_c_loss = criterion(y_hat, reranking(y, t, alpha))
        # total loss
        loss = p_s_loss + w_c_loss
        return loss
>>>>>>> a2f6790205a8572af9fa97268fe35fbbe4837e6d
