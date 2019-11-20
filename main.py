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
    main()