import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from config import config

import random

import matplotlib.pyplot as plt
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def CrossEntropy(yHat, y):
    if y == 1:
        return -np.log(yHat)
    else:
        return -np.log(1 - yHat)

def atn_train(trg_model, atn, trainloader, num_epoch=config.num_epoch, lr=config.lr, momentum = config.momentum):    
    cls_loss = None
    re_loss = nn.MSELoss()
    
    optimizer = optim.SGD(atn.parameters(), lr=lr, momentum=momentum)
    
    for epoch in range(num_epoch):   # 데이터셋을 수차례 반복합니다.
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후
            inputs, labels = data
            
            #print(len(inputs))

            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후

            t = [random.randint(0, config.num_classes-1) for _ in range(config.batch_size)]
            #print(t)
            x= inputs
            x_dot_batch = atn.forward(x,t)

            #Reconstruction Loss with x, x'
            reconstruction_loss = re_loss(x, x_dot_batch)
            
            #Classification Loss with label, k(want to this target)
            outputs=[]
            
            for idx,x_dot in enumerate(x_dot_batch):
                #print(x_dot.size()) #3,32,32
                outputs.append(trg_model.predict(x_dot)) #argmax를 뱉어냄
                if i % 10==0:
                    trg_model.plot_prediction(x_dot, labels[idx])
            #print(outputs)
            #outputs = torch.Tensor(outputs)
            #t = torch.Tensor(t)
            #print(outputs)
            #print(t)
            classification_loss=0
            for i in range(config.batch_size):
                classification_loss += CrossEntropy(outputs[t[i]],t[i])
            
            classification_loss /= config.batch_size
            #classification_loss = torch.Tensor(classification_loss)
            
            #print(classification_loss)
            #print(reconstruction_loss)
            
            loss = classification_loss + config.alpha * reconstruction_loss
            #print(loss)
            loss.backward()
            optimizer.step()

            # 통계를 출력합니다.
            print("i: %d : , loss: %f " % (i, loss))
#             running_loss += loss.item()
#             if i % 10 == 0:    # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/10))
#                 running_loss = 0.0

    print('Finished Training')