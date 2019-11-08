class config():
    batch_size = 4
    num_workers = 2
    num_epoch = 1
    
    lr=0.001
    momentum = 0.9
    
    #pgd
    eps=0.1
    alpha = 2/255
    iters = 5
    
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')