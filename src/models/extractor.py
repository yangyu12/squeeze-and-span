# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

#----------------------------------------------------------------------------

class ProjExtractor(nn.Module):
    """
    A plain representation extrator.
    """
    def __init__(self, 
        base_encoder, 
        dim                 = 2048, 
        num_proj_layers     = 2,
    ):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super().__init__()
        self.dim = dim
        
        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        backbone = base_encoder(num_classes=dim, zero_init_residual=True)
        if isinstance(backbone.fc, nn.Sequential):
            prev_dim = backbone.fc[0].weight.shape[1]
        else:
            prev_dim = backbone.fc.weight.shape[1]
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # build projector
        layers = []
        for _ in range(num_proj_layers - 1):
            layers += [
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Linear(prev_dim, dim, bias=False))
        self.projector = nn.Sequential(*layers)

    def forward(self, x):
        """xxx
        """
        # forward
        x = self.backbone(x)
        x = self.projector(x)
        return x

#----------------------------------------------------------------------------
