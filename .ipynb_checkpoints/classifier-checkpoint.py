import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

from model import VGG
from config import config

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    #npimg = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

class Classifier():
  
  def __init__(self, device, weight_path):
    self.device = device
    #self.transform = transform
    self.net = VGG('VGG16').to(device)
    self.net.load_state_dict(
        torch.load(weight_path, map_location=device)
    )
    self.net.eval()
  
  def plot_prediction(self, image, label):
    with torch.no_grad():

        x= image
        y = self.net(x.unsqueeze(0))
        y = nn.functional.softmax(y, dim=1).cpu()[0]
        fig = plt.figure(figsize=(10, 2))
        fig.add_subplot(1, 2, 1)
        plt.title(config.classes[label])
        plt.imshow(np.transpose(image.numpy(),(1,2,0)))
        fig.add_subplot(1, 2, 2)
        plt.title('probs')
        plt.barh(np.arange(len(y)), y)
        plt.xlim(0, 1)
        plt.yticks(np.arange(len(y)), config.classes)
        plt.show()
  
  def predict(self, image):
    with torch.no_grad():
        x = image
        y = self.net(x.unsqueeze(0))
        y = nn.functional.softmax(y, dim=1).cpu()[0] #1,10
    
        #print("y's sum: {}".format(np.sum(y)))
        return y