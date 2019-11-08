import torch
import torchvision
import torchvision.transforms as transforms


from config import config

class Preprocess():
    '''
    torchvision 데이터셋의 출력(output)은 [0, 1] 범위를 갖는 PILImage 이미지입니다. 이를 [-1, 1]의 범위로 정규화된 Tensor로 변환합니다
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
    http://groups.csail.mit.edu/vision/TinyImages/ 
    '''
    def __init__(self, batch_size = config.batch_size, num_workers = config.num_workers):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=self.transform)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=self.transform)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

        #if PGD_attack:
        #    pgd = PGD()
#def main():
#    p = Preprocess()
    
    
#if __name__ == "__main__":
#    main()