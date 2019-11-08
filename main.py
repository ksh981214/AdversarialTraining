from preprocess import Preprocess
from train import train
from test import test
from compare import compare
from model import simpel_Net
from model import VGG

from pgd_attack import PGD_attack



def main():
    preprocess = Preprocess()
    
    #net = Net()
    net = VGG('VGG16')
    
    
    train(net, preprocess.trainloader)
    
    #before pgd attack
    #test(net, preprocess.testloader)
    
    _, adv_loader = PGD_attack(net, preprocess.testloader)
    
    #after pgd attack
    #test(net, adv_loader) 
    
    compare(net, preprocess.testloader, adv_loader)
    
if __name__ =="__main__":
    main()