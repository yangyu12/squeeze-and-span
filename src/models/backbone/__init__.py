import torch.nn as nn
from torchvision.models import *


__all__ = ["resnet18cifar", 
           "resnet18",  "resnet50", "resnet101", 
]

def resnet18cifar(
    num_classes, 
    zero_init_residual=True,
    **kwargs,
):
    r18_model = resnet18(num_classes=num_classes, zero_init_residual=zero_init_residual)
    r18_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # change conv1 from 7x7 to 3x3
    r18_model.maxpool = nn.Identity()  # remove the first max pooling layer
    return r18_model
