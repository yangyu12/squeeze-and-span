import os
import pickle
import re
import sys
import json
import numpy as np
import torch.nn.functional as F

import torch


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path("./src")


class PretrainedDiscriminator(torch.nn.Module):
    """
    Generator wrapper for saving inner features for output.
    Suitable for StyleGAN2
    """
    def __init__(self,
        gan_pkl,                                   #
        regex               = r'b32|b16|b8|b4.fc$',   # TODO: Match all the convolutional layers  r'.*synthesis\.b(\d+)\.conv$'
        batch_size          = None,
    ):
        super().__init__()

        with open(gan_pkl, 'rb') as f:
            self.D = pickle.load(f)['D']

        if batch_size is not None:
            self.D.b4.mbstd.group_size = batch_size

        # Default
        # b32
        # b32.fromrgb
        # b32.conv0
        # b32.conv1
        # b16
        # b16.conv0
        # b16.conv1
        # b8
        # b8.conv0
        # b8.conv1
        # b4
        # b4.mbstd
        # b4.conv
        # b4.fc
        # b4.out

        # extract intermediate features with hooks.
        # Reference: https://medium.com/the-owl/using-forward-hooks-to-extract-intermediate-layer-outputs-from-a-pre-trained-model-in-pytorch-1ec17af78712
        # fix bug in data parallel
        # https://discuss.pytorch.org/t/aggregating-the-results-of-forward-backward-hook-on-nn-dataparallel-multi-gpu/28981/8
        # https://github.com/kazuto1011/grad-cam-pytorch/issues/13#issuecomment-486071455
        self.inner_features = []
        self.fhooks = []
        self.f_shape = None
        for name, mod in self.D.named_modules():
            if re.fullmatch(regex, name):
                # print(name)
                self.fhooks.append(mod.register_forward_hook(self.forward_hook))

    def forward_hook(self, module, input, output):
        # TODO: seems like we are not using same feature maps as DatasetGAN
        if isinstance(output, torch.Tensor):
            self.inner_features.append(output.float())
        else: # StyleGAN2 synthesis block typically return 2 tensors
            assert len(output) > 1
            # channels = [x.size(1) for x in output]
            # max_ch_idx = channels.index(max(channels))
            self.inner_features.append(output[0].float())

    def forward(self, x):
        self.inner_features = []
        out = self.D(x, None)
        if self.f_shape is None:
            self.f_shape = [x.shape[1:] for x in self.inner_features]
        return [x.mean(dim=[2,3]) if x.ndim==4 else x for x in self.inner_features]
    