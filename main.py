from preprocess import Preprocess
from atn_network import Multi_target_Adversarial_Network
from classifier import Classifier
from atn_train import atn_train

import torchvision.transforms as transforms

from config import config

def main():
    preprocess = Preprocess()
    #net = Net()
    

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.491, 0.483, 0.481),
            (0.263, 0.258, 0.256)
        )
    ])
    trg_model = Classifier(config.device, config.weight_path, transform)
    
    atn = Multi_target_Adversarial_Network()
    
    atn_train(trg_model, atn, preprocess.train_dataloader, num_epoch=config.num_epoch, lr=config.lr, momentum = config.momentum)
    
    
#     train(net, preprocess.trainloader)
    
#     #before pgd attack
#     #test(net, preprocess.testloader)
    
#     _, adv_loader = PGD_attack(net, preprocess.testloader)
    
#     #after pgd attack
#     #test(net, adv_loader) 
    
#     compare(net, preprocess.testloader, adv_loader)
    
if __name__ =="__main__":
    main()