class config():
    batch_size = 32
    num_workers = 2
    num_epoch = 1
    
    lr=0.001
    momentum = 0.9
    
    #pgd
    eps=0.1
    alpha = 2/255
    iters = 5
    
    num_classes = 10
    
    '''
    Larger alpha --> better reconstruction quality
    smaller alpha --> get higher attack success rate
    '''
    alpha = 1.0
    
    #classifier
    device='cpu'
    weight_path = 'vgg16_e080_91.39.pth'
    
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')