# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from enum import Enum
import torch.nn as nn
from collections import OrderedDict

#----------------------------------------------------------------------------

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(
        self, base_encoder, dim=2048, pred_dim=512, num_proj_layers=2, disable_downsample=True,
        use_rotation_head=False
    ):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        backbone = base_encoder(num_classes=dim, zero_init_residual=True)
        prev_dim = backbone.fc.weight.shape[1]

        # build a 3-layer projector
        prev_dim = backbone.fc.weight.shape[1]
        projector = []
        for _ in range(num_proj_layers-1):
            projector += [nn.Linear(prev_dim, prev_dim, bias=False),
                          nn.BatchNorm1d(prev_dim),
                          nn.ReLU(inplace=True),]
        projector += [backbone.fc, nn.BatchNorm1d(dim, affine=False)]
        backbone.fc = nn.Sequential(*projector) # output layer
        backbone.fc[-2].bias.requires_grad = False # hack: not use bias as it is followed by BN
        self.backbone = backbone

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True), # hidden layer
                                       nn.Linear(pred_dim, dim)) # output layer

        # rotation head
        self.use_rotation_head = use_rotation_head
        if self.use_rotation_head:
            self.rot_classifier = nn.Linear(prev_dim, 4)

    def forward(self, *xs):
        """
        Input:
            xs:= x1, x2, ...: views of images
        Output:
            outs:= p1, p2, ..., z1, z2, ...: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        ps = []
        zs = []
        if self.use_rotation_head:
            rot_outputs = []
        for x in xs:
            # compute features for each view
            z = self.backbone(x)
            if self.use_rotation_head:
                raise NotImplementedError 
                rot_outputs.append(self.rot_classifier(x))
            # z = self.projector(x)   # NxC
            p = self.predictor(z)   # NxC
            ps.append(p)
            zs.append(z.detach())

        outs = ps + zs
        if self.use_rotation_head:
            raise NotImplementedError
            outs = outs + rot_outputs
        return tuple(outs)

#----------------------------------------------------------------------------
