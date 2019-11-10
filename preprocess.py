import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from config import config

class Preprocess():
    '''
    torchvision 데이터셋의 출력(output)은 [0, 1] 범위를 갖는 PILImage 이미지입니다. 이를 [-1, 1]의 범위로 정규화된 Tensor로 변환합니다
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
    http://groups.csail.mit.edu/vision/TinyImages/ 
    '''
    def __init__(self, batch_size = config.batch_size, num_workers = config.num_workers):
        print("Start getting Data....")
        '''
        get train/valid data
        '''
        indices = list(range(50000))
        train_sampler = torch.utils.data.SubsetRandomSampler(indices[:40000])
        valid_sampler = torch.utils.data.SubsetRandomSampler(indices[40000:])
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.489, 0.480, 0.477),(0.265, 0.259, 0.256))])
        
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.489, 0.480, 0.477),(0.265, 0.259, 0.256))])

        self.train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,transform=train_transform)

        self.valid_dataset = datasets.CIFAR10(
          root='./data', train=True, download=True,
          transform=valid_transform
      )

        self.train_dataloader = torch.utils.data.DataLoader(
          dataset=self.train_dataset,
          batch_size=batch_size,
          sampler=train_sampler,
          drop_last=True
      )

        self.valid_dataloader = torch.utils.data.DataLoader(
          dataset=self.train_dataset,
          batch_size=batch_size,
          sampler=valid_sampler,
      )
        
        
        '''
        get test data
        '''
        test_transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(
              (0.491, 0.483, 0.481),
              (0.263, 0.258, 0.256)
          )
      ])

        self.test_dataset = datasets.CIFAR10(
          root='./data', train=False, download=True,
          transform=test_transform
      )

        self.test_dataloader = torch.utils.data.DataLoader(dataset=self.test_dataset)
        
        print("Finish getting Data....")