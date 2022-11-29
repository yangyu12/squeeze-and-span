# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import sys
import math

import torch
import torch.nn as nn
from PIL import ImageFilter
from torchvision.transforms import functional as F


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class GaussianBlur_tensor(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.], use_cuda=True):
        self.sigma = sigma
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.gaussian_filter = torch.nn.Conv2d(in_channels=3,
                                                   out_channels=3,
                                                   kernel_size=5,
                                                   groups=3,
                                                   bias=False).cuda()
        else:
            self.gaussian_filter = torch.nn.Conv2d(in_channels=3,
                                                   out_channels=3,
                                                   kernel_size=5,
                                                   groups=3,
                                                   bias=False)

        self.gaussian_filter.weight.requires_grad = False

    def __call__(self, x):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        ori_x_shape = x.shape
        x_cord = torch.arange(5)
        x_grid = x_cord.repeat(5).view(5, 5)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        try:
            x.shape[3]
        except:
            x = x.unsqueeze(0)
        x_padding = torch.zeros(
            [x.shape[0], x.shape[1], x.shape[2] + 4,
             x.shape[3] + 4]).to(x.device)
        x_padding[:, :, 2:-2, 2:-2] = x
        # else:
        #     x_padding = torch.zeros([x.shape[0], x.shape[1] + 4, x.shape[2] + 4]).cuda()
        #     x_padding[:, 2:-2, 2:-2] = x
        #     x_padding = x_padding.unsqueeze(0)
        mean = (5 - 1) / 2.
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, 5, 5)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)
        if self.use_cuda:
            self.gaussian_filter.weight.data = gaussian_kernel.cuda()
        else:
            self.gaussian_filter.weight.data = gaussian_kernel
        x = self.gaussian_filter(x_padding)
        try:
            ori_x_shape.shape[3]
        except:
            x = x.squeeze(0)
        return x
