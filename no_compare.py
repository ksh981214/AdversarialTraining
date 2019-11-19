import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np

from config import config

def compare(net, testloader, advloader): 
    dataiter = iter(testloader)
    test_images, test_labels = dataiter.next()
    
    dataiter = iter(advloader)
    adv_images, adv_labels = dataiter.next()
    
    
    classes = config.classes

#     # 이미지를 출력합니다.
#     imshow(torchvision.utils.make_grid(images))
#     print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
#     outputs = net(images)
#     _, predicted = torch.max(outputs, 1)

#     print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                                   for j in range(4)))
    
    test_correct = 0
    test_total = 0
    
    adv_correct = 0
    adv_total = 0
    with torch.no_grad():
        for test_data, adv_data in zip(testloader,advloader):
            test_images, test_labels = test_data
            adv_images, adv_labels = adv_data
            
            test_outputs = net(test_images)
            adv_outputs = net(adv_images)
            
            _, test_predicted = torch.max(test_outputs.data, 1)
            _, adv_predicted = torch.max(adv_outputs.data, 1)
            
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_labels).sum().item()
            
            adv_total += adv_labels.size(0)
            adv_correct += (adv_predicted == adv_labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * test_correct / test_total))
    print('Accuracy of the network on the 10000 adv_ images: %d %%' % (
        100 * adv_correct / adv_total))
    
    test_class_correct = list(0. for i in range(10))
    test_class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                test_class_correct[label] += c[i].item()
                test_class_total[label] += 1
                
    adv_class_correct = list(0. for i in range(10))
    adv_class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in advloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                adv_class_correct[label] += c[i].item()
                adv_class_total[label] += 1


    for i in range(10):
        print('Test: Accuracy of %5s : %2d %%' % (
            classes[i], 100 * test_class_correct[i] / test_class_total[i]))
        
    for i in range(10):
        print('Adv: Accuracy of %5s : %2d %%' % (
            classes[i], 100 * adv_class_correct[i] / adv_class_total[i]))
        
        
#if __name__ == "__main__":
#    main()