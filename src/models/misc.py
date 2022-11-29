import math
import os
import shutil
import sys

import numpy as np
import torch


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename), 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, init_lr, epoch, max_epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / max_epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    # identify whether pretrain model is trained under DDP
    pretrained_model_distributed = list(state_dict_pre.keys())[0].startswith('module.')
    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        if pretrained_model_distributed:
            k_pre = 'module.backbone.' + k[len('module.'):] if k.startswith('module.') else 'module.' + k
        else:
            k_pre = 'backbone.' + k[len('module.'):] if k.startswith('module.') else 'backbone.' + k
        
        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)


# exponential moving average
# https://github.com/lucidrains/byol-pytorch/blob/caa65d7c19a80a3611f4049c638ecd8401f8f3c4/byol_pytorch/byol_pytorch.py#L59
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def update_moving_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)


def modify_checkpoint_keys(model, checkpoint):
    """
    Modify checkpoint state_dict keys according to model type in order to correctly load pretrained weights
    """
    add_prefix = isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    if add_prefix:
        if not list(checkpoint['state_dict'].keys())[0].startswith('module.'):
            param_keys = list(checkpoint['state_dict'].keys())
            for k in param_keys:
                checkpoint['state_dict'][f'module.{k}'] = checkpoint['state_dict'][k]
                checkpoint['state_dict'].pop(k)
    else:   # remove 'module.' prefix
        if list(checkpoint['state_dict'].keys())[0].startswith('module.'):
            param_keys = list(checkpoint['state_dict'].keys())
            for k in param_keys:
                checkpoint['state_dict'][k[len('module.'):]] = checkpoint['state_dict'][k]
                checkpoint['state_dict'].pop(k)
    return checkpoint
