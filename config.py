class config():
    #classifier
    device='cpu'
    weight_path = 'vgg16_e086_90.62.pth'
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    num_classes = 10
    
    '''
    Larger alpha --> better reconstruction quality
    smaller alpha --> get higher attack success rate
    '''
    EPOCH_NUM = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    # ATN
    ALPHA = 1.5
    BETA = 0.01
    T = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 6,
        6: 7,
        7: 8,
        8: 9,
        9: 0
    }