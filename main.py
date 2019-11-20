<<<<<<< HEAD
from preprocess import Preprocess
from atn_network import ATN_c
from classifier import Classifier
from atn_train import atn_train


from config import config

def main():
    preprocess = Preprocess()

    trg_model = Classifier(config.device, config.weight_path)
    
    atn = ATN_c()
    
    atn_train(trg_model, atn, preprocess.train_dataloader, num_epoch=config.EPOCH_NUM)
    
    
if __name__ =="__main__":
=======
from preprocess import Preprocess
#from atn_network import Multi_target_Adversarial_Network
from atn_network import ATN_c
from classifier import Classifier
from atn_train import atn_train

import torchvision.transforms as transforms

from config import config

def main():
    preprocess = Preprocess()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.491, 0.483, 0.481),
            (0.263, 0.258, 0.256)
        )
    ])
    trg_model = Classifier(config.device, config.weight_path, transform)
    
    #atn = Multi_target_Adversarial_Network()
    atn = ATN_c()
    
    atn_train(trg_model, atn, preprocess.train_dataloader, num_epoch=config.EPOCH_NUM)
    
    
if __name__ =="__main__":
>>>>>>> a2f6790205a8572af9fa97268fe35fbbe4837e6d
    main()