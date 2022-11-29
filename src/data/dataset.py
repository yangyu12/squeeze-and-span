import os
import torchvision.datasets as tvdatasets



def is_image_file(x):
    _, ext = os.path.splitext(x)
    return ext in ['.jpg', '.png', '.webp'] # TODO: add more

__all__ = [
    "CIFAR10", "CIFAR100", "STL10", 
    "NUM_CLASSES" 
]

NUM_CLASSES = {
    "CIFAR10": 10, 
    "CIFAR100": 100, 
    "STL10": 10, 
}

def CIFAR10(data_dir, split='train', transform=None):
    train_ = split in ['train', 'train_knn']
    return tvdatasets.CIFAR10(data_dir, train_, transform)

def CIFAR100(data_dir, split='train', transform=None):
    train_ = split in ['train', 'train_knn']
    return tvdatasets.CIFAR100(data_dir, train_, transform)

def STL10(data_dir, split='train', transform=None):
    split_name = {
        'train': 'train+unlabeled',
        'train_knn': 'train',
        'val': 'test'
    }[split]
    return tvdatasets.STL10(data_dir, split_name, transform=transform)
