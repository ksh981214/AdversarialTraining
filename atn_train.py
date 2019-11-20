#import numpy as np

from loss import Loss
from utils import map_label_to_target

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

import torch.optim as optim

import matplotlib.pyplot as plt

from config import config

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
def imsave(img, label):
    img = img / 2 + 0.5     # unnormalize
    img = img.detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray(img,'RGB')
    name = label + ".png"
    img.save(name)
    #img.show()
<<<<<<< HEAD

def atn_train(trg_model, atn, trainloader, num_epoch=config.EPOCH_NUM):    
=======
    
    
def freeze_model(net):
    for param in net.parameters():
        param.requires_grad = False

def atn_train(trg_model, atn, trainloader, num_epoch=config.EPOCH_NUM):    
    
    #freeze_model(trg_model)
>>>>>>> a2f6790205a8572af9fa97268fe35fbbe4837e6d
    
    optimizer = optim.RMSprop(
        atn.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)
    
    for epoch in range(num_epoch):   # 데이터셋을 수차례 반복합니다.
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후
            inputs, labels = data
            
            inputs = inputs.to(config.device) #tensor, 32, 3, 32, 32
            labels = labels.to(config.device) #tensor, 32
            #print("inputs's size: {}".format(inputs.size()))
            #print("labels's size: {}".format(labels.size()))
            
            # set labels to one-hot
#            labels = labels_to_onehot(labels, num_classes)
            # map labels to target
            targets = map_label_to_target(labels, t=config.T) #list, 32
            #print("labels: {}".format(labels))
            #print("targets: {}".format(targets))
            #print("targets's size: {}".format(len(targets)))
            
        
            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            atn_outputs = atn(inputs) #tensor, 32, 3, 32, 32
            #print("atn_output's shape: {}".format(atn_outputs.shape))
            #print("atn_output's type: {}".format(type(atn_outputs)))
            
            
            #print(config.classes[labels[0]])
            #imshow(inputs[0])
            
            #print(atn_outputs[0])
            #imshow(atn_outputs[0])
            
            # victim forward (y_hat)
            normal_outputs = torch.zeros(config.BATCH_SIZE, config.num_classes, dtype=torch.float32) 
            #print("normal_outputs's size : {}".format(normal_outputs.shape))
            for idx, _input in enumerate(inputs):
                normal_outputs[idx,:] = trg_model.predict(_input).squeeze()
<<<<<<< HEAD
                #imshow(_input)
=======
                imshow(_input)
>>>>>>> a2f6790205a8572af9fa97268fe35fbbe4837e6d
                #imsave(_input, config.classes[labels[idx]])
                #trg_model.plot_prediction(_input,labels[idx])
                
                
            fooled_outputs = torch.zeros(config.BATCH_SIZE, config.num_classes, dtype=torch.float32)
            #print("fooled_outputs's size : {}".format(fooled_outputs.shape))
            for idx, atn_output in enumerate(atn_outputs):
                fooled_outputs[idx,:] = trg_model.predict(atn_output).squeeze()
<<<<<<< HEAD
                #print(config.classes[labels[idx]])
                #imshow(inputs[idx])
=======
>>>>>>> a2f6790205a8572af9fa97268fe35fbbe4837e6d
                #imshow(atn_output)
                #trg_model.plot_prediction(atn_output, labels[idx])
            
            # preprocess target
            # get loss
            loss = Loss.ATN_loss(x_hat=atn_outputs, x=inputs,
                                 y_hat=fooled_outputs, y=normal_outputs, t=targets, alpha=config.ALPHA, beta=config.BETA)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.data
<<<<<<< HEAD
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                print(config.classes[labels[0]])
                imshow(inputs[0])
                imshow(atn_outputs[0])
=======
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
>>>>>>> a2f6790205a8572af9fa97268fe35fbbe4837e6d
                

    print('Finished Training')