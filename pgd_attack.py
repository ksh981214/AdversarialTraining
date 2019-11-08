import torchattacks
from torchattacks import PGD

from config import config

import torch

def PGD_attack(net, test_loader, eps = config.eps, alpha = config.alpha, iters = config.iters, batch_size = config.batch_size, num_workers = config.num_workers):
    pgd_attack = PGD(net, eps=eps, alpha=alpha, iters=iters)
    
    # If you want to reduce the space of dataset, set 'to_unit8' as True
    # If you don't want to know about accuaracy of the model, set accuracy as False
    pgd_attack.save(data_loader=test_loader, file_name="data/cifar10_pgd.pt", accuracy=True)
    
    # When scale=True it automatically tansforms images to [0, 1]
    adv_data = pgd_attack.load(file_name="data/cifar10_pgd.pt", scale=True)
    adv_loader = torch.utils.data.DataLoader(adv_data, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    
    return adv_data, adv_loader