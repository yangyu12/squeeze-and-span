import math
from functools import partial

import torch
import torch.nn.functional as F

#----------------------------------------------------------------------------

class CompactRepr(torch.nn.Module):
    """Convert generator features into compact representations"""
    def __init__(self,
        input_shapes,              # The shapes of the multi-resolution feature maps
        tmp_channels,              # The number of channels of the propagated feature map
        output_channels,           # The number of classes
        num_layers      = 3,       # The number of fc layers used to predict pixel labels
        fuse_bn_relu    = False,
        use_factor      = False,
        act             = 'relu',   # relu, gelu
        norm            = 'bn',     # bn, gn
    ):
        super().__init__()
        self.input_channels = [x[0] for x in input_shapes]
        self.num_features = len(self.input_channels)
        # added
        self.output_channels = output_channels

        # Configure activation function, normalization layer
        which_act = {'relu': partial(torch.nn.ReLU, inplace=True),
                     'gelu': torch.nn.GELU}[act]
        which_norm = {'bn': torch.nn.BatchNorm1d,}[norm]

        # Lateral layer producing feature map to be propagated
        for i in range(self.num_features):
            channels = self.input_channels[i]
            fc = torch.nn.Linear(channels, tmp_channels)
            setattr(self, f'lateral{i}', fc)
        self.fuse_factor = 1. / math.sqrt(self.num_features) if use_factor else 1.

        # MLP head
        head = [which_norm(tmp_channels), which_act()] if fuse_bn_relu else []
        for i in range(num_layers):
            last_layer = (i == num_layers - 1)
            if last_layer:  # last layer
                head.append(torch.nn.Linear(tmp_channels, output_channels, bias=True))
                break
            else:
                head.append(torch.nn.Linear(tmp_channels, tmp_channels, bias=False))
            head.append(which_norm(tmp_channels))
            head.append(which_act())
        self.to_repr = torch.nn.Sequential(*head)

    def forward(self, x, masks=None):
        # feature fusion
        all_features = x
        x = None
        for i in range(self.num_features):
            y = all_features[i]
            assert y.size(1) == self.input_channels[i]

            # aggregate the features (crop if masks is not None)
            if masks is None and y.dim() == 4:  # Global average pooling
                y = torch.mean(y, dim=[2, 3])
            elif y.dim() == 4:
                m = masks[i]  # (B, 1, H, W)
                y = torch.sum(y * m, dim=[2, 3]) / torch.sum(m, dim=[2, 3])  

            # forward the processing layer
            lateral = getattr(self, f'lateral{i}')
            y = self.fuse_factor * lateral(y)
            if x is None:
                x = y
            else:
                x = x.add_(y)

        # MLP head
        return self.to_repr(x)

#----------------------------------------------------------------------------

class MLP(torch.nn.Module):
    """Convert generator features into compact representations"""
    def __init__(self,
        input_shapes,              # The shapes of the multi-resolution feature maps
        tmp_channels,              # The number of channels of the propagated feature map
        output_channels,           # The number of classes
        num_layers      = 3,       # The number of fc layers used to predict pixel labels
        act             = 'relu',   # relu, gelu
        norm            = 'bn',     # bn, gn
    ):
        super().__init__()
        self.input_channels = [x[0] for x in input_shapes]
        self.num_features = len(self.input_channels)
        # added
        self.output_channels = output_channels

        # Configure activation function, normalization layer
        which_act = {'relu': partial(torch.nn.ReLU, inplace=True),
                     'gelu': torch.nn.GELU}[act]
        which_norm = {'bn': torch.nn.BatchNorm1d,}[norm]

        # MLP head
        head = []
        in_channels = sum(self.input_channels)
        for i in range(num_layers):
            if i == num_layers - 1:  # last layer
                head.append(torch.nn.Linear(in_channels, output_channels, bias=True))
                break
            else:
                head.append(torch.nn.Linear(in_channels, tmp_channels, bias=False))
                in_channels = tmp_channels
            head.append(which_norm(tmp_channels))
            head.append(which_act())
        self.to_repr = torch.nn.Sequential(*head)

    def forward(self, x):
        # MLP head
        return self.to_repr(x)

#----------------------------------------------------------------------------

class CompactReprWithPredictor(torch.nn.Module):
    """Convert generator features into compact representations"""
    def __init__(self,
        input_shapes,              # The shapes of the multi-resolution feature maps
        dim,                       # 
        pred_dim        = 512,     # The number hidden uints in predictor MLP 
        num_proj_layers = 2,       # The number of projector layers (2 for CIFAR, 3 for ImageNet)
        # fuse_bn_relu    = False,
        # use_factor      = False,
        act             = 'relu',   # relu, gelu
        norm            = 'bn',     # bn, gn
        # feat_choices    = None
    ):
        super().__init__()
        self.input_channels = [x[0] for x in input_shapes]
        self.num_features = len(self.input_channels)

        # if feat_choices is None:
        #     feat_choices = [True] * self.num_features
        # self.feat_choices = feat_choices
        # self.num_chosen_features = sum(self.feat_choices)

        # Configure activation function, normalization layer
        which_act = {'relu': partial(torch.nn.ReLU, inplace=True),
                     'gelu': torch.nn.GELU}[act]
        which_norm = {'bn': torch.nn.BatchNorm1d,}[norm]

        # Projector
        # prev_dim = sum([c for c, choice in zip(self.input_channels, self.feat_choices) if choice])
        prev_dim = sum(self.input_channels)
        assert num_proj_layers > 1
        proj_layers = []
        for _ in range(num_proj_layers - 1):
            proj_layers += [
                torch.nn.Linear(prev_dim, prev_dim, bias=False),
                which_norm(prev_dim),
                which_act(),
            ]
        proj_layers += [
            torch.nn.Linear(prev_dim, dim, bias=False),
            which_norm(dim, affine=False)
        ]
        self.projector = torch.nn.Sequential(*proj_layers) # output layer

        # Predictor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(dim, pred_dim, bias=False),
            which_norm(pred_dim),
            which_act(), # hidden layer
            torch.nn.Linear(pred_dim, dim)
        ) # output layer
        
    def forward(self, inputs, masks=None):
        # Concate the global average pooling of every layer output
        if masks is None:
            # x = torch.cat(
            #     [
            #         torch.mean(xx, dim=[2, 3]) # TODO: change to adaptive pool 2d
            #         for xx, choice in zip(x, self.feat_choices) if choice
            #     ], dim=1
            # )
            x = torch.cat([torch.mean(x, dim=[2, 3]) for x in inputs], dim=1)
        else:
            x = torch.cat(
                [torch.sum(x * m, dim=[2, 3]) / torch.sum(m, dim=[2, 3]) 
                 for x, m in zip(inputs, masks)], 
            dim=1)

        # Projection
        z = self.projector(x)

        # Prediction
        p = self.predictor(z)

        return p, z.detach()

#----------------------------------------------------------------------------

class MultiPrototypes(torch.nn.Module):
    def __init__(self,
        in_channels,                        # 
        num_prototypes  = [30, 30, 30],     # 
    ):
        super().__init__()
        self.num_heads = len(num_prototypes)

        for h, K in enumerate(num_prototypes):
            proto = torch.nn.Parameter(torch.randn(K, in_channels), requires_grad=True)
            self.register_parameter(f"proto{h}", proto)
        
    def forward(self, x):
        """We assue x is unnormalized features"""
        x = F.normalize(x, dim=1, p=2)

        outs = []
        for h in range(self.num_heads):
            proto = getattr(self, f"proto{h}")
            proto = F.normalize(proto, dim=1, p=2)
            logit = torch.mm(x, proto.t())  # (N, K)
            outs.append(logit)
        return outs

#----------------------------------------------------------------------------

class AggregatedCompactRepr(torch.nn.Module):
    """Aggregate features together"""
    # TODO: use resnet block instead of vanilla conv
    def __init__(self,
        input_shapes,              # The shapes of the multi-resolution feature maps
        dim,                       # 
        pred_dim        = 512,     # The number hidden uints in predictor MLP 
        num_proj_layers = 2,       # The number of projector layers (2 for CIFAR, 3 for ImageNet)
        # fuse_bn_relu    = False,
        # use_factor      = False,
        act             = 'relu',   # relu, gelu
        norm            = 'bn',     # bn, gn
        input_image     = False
    ):
        super().__init__()
        self.input_channels = [x[0] for x in input_shapes]
        self.num_features = len(self.input_channels)

        self.input_image = input_image
        if self.input_image:
            self.layer_image = torch.nn.Sequential(
                torch.nn.Conv2d(3, self.input_channels[-1], 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(self.input_channels[-1]),
                torch.nn.ReLU(inplace=True),
            )
        # Downsample block
        for i in range(self.num_features):
            if i < self.num_features - 1:
                if (not input_image) and i == 0:
                    self.add_module(f'layer{i}', torch.nn.Sequential(
                        torch.nn.Conv2d(self.input_channels[i], self.input_channels[i + 1], 3, stride=2, padding=1, bias=False),
                        torch.nn.BatchNorm2d(self.input_channels[-1]),
                        torch.nn.ReLU(inplace=True),
                    ))
                else:
                    self.add_module(f'layer{i}', torch.nn.Sequential(
                        torch.nn.Conv2d(self.input_channels[i] * 2, self.input_channels[i + 1], 3, stride=2, padding=1, bias=False),
                        torch.nn.BatchNorm2d(self.input_channels[-1]),
                        torch.nn.ReLU(inplace=True),
                    ))                    
            else:
                self.add_module(f'layer{i}', torch.nn.Sequential(
                    torch.nn.Conv2d(self.input_channels[i] * 2, self.input_channels[i], 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(self.input_channels[-1]),
                    torch.nn.ReLU(inplace=True),
                ))
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # Configure activation function, normalization layer
        which_act = {'relu': partial(torch.nn.ReLU, inplace=True),
                     'gelu': torch.nn.GELU}[act]
        which_norm = {'bn': torch.nn.BatchNorm1d,}[norm]
        
        # Projector
        prev_dim = self.input_channels[-1]
        assert num_proj_layers > 1
        proj_layers = []
        for _ in range(num_proj_layers - 1):
            proj_layers += [
                torch.nn.Linear(prev_dim, prev_dim, bias=False),
                which_norm(prev_dim),
                which_act(),
            ]
        proj_layers += [
            torch.nn.Linear(prev_dim, dim, bias=False),
            which_norm(dim, affine=False)
        ]
        self.projector = torch.nn.Sequential(*proj_layers) # output layer

        # Predictor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(dim, pred_dim, bias=False),
            which_norm(pred_dim),
            which_act(), # hidden layer
            torch.nn.Linear(pred_dim, dim)
        ) # output layer

    def forward(self, x, img=None):
        out = None
        if self.input_image:
            assert img is not None
            out = self.layer_image(img)
        else:
            out = x[-1]
        
        for i, xx in enumerate(x[::-1]):
            if i > 0 or self.input_image:
                out = torch.cat((out, xx), dim=1)
            layer = getattr(self, f'layer{i}')
            out = layer(out)

        out = self.avgpool(out).flatten(1, 3)
        
        # Projection
        z = self.projector(out)

        # Prediction
        p = self.predictor(z)

        return p, z.detach()

#----------------------------------------------------------------------------

class VanillaCompactRepr(torch.nn.Module):
    """0-Paramed; dim = 2048"""
    # def __init__(self, feat_choices=None):
    def __init__(self):
        super().__init__()
        # self.feat_choices = feat_choices
        # self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs, masks=None):
        # if self.feat_choices is None:
        #     self.feat_choices = [True] * len(x)
        # out = [
        #     self.avgpool(feat).flatten(1, 3)
        #     for feat, choice in zip(x, self.feat_choices) if choice
        # ]
        def avgpool(x):
            if x.ndim == 4:
                x = x.mean(dim=[2,3])
            return x
        
        def weighted_sum(x, m):
            return torch.sum(x * m, dim=[2, 3]) / torch.sum(m, dim=[2, 3])
        
        if masks is None:
            out = [avgpool(x) for x in inputs]
        else:
            out = [weighted_sum(x, m) for x, m in zip(inputs, masks)]
        return torch.cat(out, dim=1)

#----------------------------------------------------------------------------

DENSE_ANNOTATOR_VARIANTS = ["DenseRepr", "VanillaDenseRepr"]

class DenseRepr(torch.nn.Module):
    """Convert generator features into dense representations"""
    def __init__(self,
        input_shapes,              # The shapes of the multi-resolution feature maps
        tmp_channels,              # The number of channels of the propagated feature map
        output_channels,           # The number of classes
        num_layers      = 3,       # The number of 3x3 conv layers used to predict pixel labels
        stride          = 1,
        fuse_bn_relu    = False,
        use_factor      = False,
        act             = 'relu', # relu, gelu
        norm            = 'bn', # bn, gn
    ):
        super().__init__()
        self.input_channels = [x[0] for x in input_shapes]
        self.num_features = len(self.input_channels)
        # added
        self.output_channels = output_channels

        # Configure activation function, normalization layer
        which_act = {'relu': partial(torch.nn.ReLU, inplace=True),
                     'gelu': torch.nn.GELU}[act]
        which_norm = {'bn': torch.nn.BatchNorm2d,
                      'gn': lambda n: torch.nn.GroupNorm(8, n)}[norm]

        # Lateral layer producing feature map to be propagated
        for i in range(self.num_features):
            channels = self.input_channels[i]
            conv = torch.nn.Conv2d(channels, tmp_channels, kernel_size=1)
            setattr(self, f'lateral{i}', conv)
        self.fuse_factor = 1. / math.sqrt(self.num_features) if use_factor else 1.

        # Segmentation head
        head = [which_norm(tmp_channels), which_act()] if fuse_bn_relu else []
        for i in range(num_layers):
            last_layer = (i == num_layers - 1)
            stride_this_layer = stride if i == 0 else 1
            if last_layer:  # last layer
                head.append(torch.nn.Conv2d(tmp_channels, output_channels, kernel_size=3,
                                            stride=stride_this_layer, padding=1, bias=True))
                break
            else:
                head.append(torch.nn.Conv2d(tmp_channels, tmp_channels, kernel_size=3,
                                            stride=stride_this_layer, padding=1, bias=False))
            head.append(which_norm(tmp_channels))
            head.append(which_act())
        self.to_repr = torch.nn.Sequential(*head)

    def forward(self, x):
        # FPN
        all_features = x
        x = None
        for i in range(self.num_features):
            assert all_features[i].size(1) == self.input_channels[i]
            lateral = getattr(self, f'lateral{i}')
            y = self.fuse_factor * lateral(all_features[i])
            if x is None:
                x = y
            else:
                if x.size(2) != y.size(2) or x.size(3) != y.size(3):
                    x = torch.nn.functional.interpolate(x, size=(y.size(2), y.size(3)),
                                                        mode='bilinear', align_corners=False)
                x = x.add_(y)

        # Segmentation head
        return self.to_repr(x)

#----------------------------------------------------------------------------

class VanillaDenseRepr(torch.nn.Module):
    """Convert generator features into dense representations"""
    def __init__(self,
        input_shapes,              # The shapes of the multi-resolution feature maps
        output_channels=None,           # The number of classes
        num_proj_layers=3,          # 
        tmp_channels=512,
    ):
        super().__init__()
        self.input_channels = [x[0] for x in input_shapes]
        self.num_features = len(self.input_channels)
        self.concat_dim = sum(self.input_channels)
        # added
        self.output_channels = output_channels
        
        self.max_size = input_shapes[-1][-2:]
        

        # Lateral layer producing feature map to be propagated
        for i in range(self.num_features):
            self.add_module(
                f'upsample{i}', 
                torch.nn.Upsample(
                    scale_factor=(self.max_size[-1] / input_shapes[i][-1]), 
                    mode='bilinear', align_corners=True
                ),
            )

        if output_channels is not None:
            projector = []
            for i in range(num_proj_layers):
                # if i < num_proj_layers - 1:
                projector.extend([
                    torch.nn.Conv2d(self.concat_dim if i == 0 else tmp_channels, tmp_channels, kernel_size=1),
                    torch.nn.BatchNorm2d(tmp_channels),
                    torch.nn.ReLU(inplace=True)
                ])
                # else:
                #     projector.extend([
                #         torch.nn.Conv2d(64, self.output_channels, kernel_size=1),
                #         torch.nn.BatchNorm2d(self.output_channels), 
                #     ])
            self.projector = torch.nn.Sequential(*projector)
            if num_proj_layers == 0:
                tmp_channels = self.concat_dim
            self.to_repr = torch.nn.Conv2d(tmp_channels, self.output_channels, 1)


    def forward(self, x):
        # Mere upsample
        all_features = x
        B = all_features[0].size(0)
        out = torch.empty(B, self.concat_dim, self.max_size[0], self.max_size[1]).to(x[0].device)
        for i in range(self.num_features):
            assert all_features[i].size(1) == self.input_channels[i]
            
            upsampler = self.__getattr__(f'upsample{i}')
            out[:, sum(self.input_channels[:i]):sum(self.input_channels[:(i + 1)]), ...] = upsampler(all_features[i])  # (B, C_i, H, W)        

        # Linear Head
        if self.output_channels is not None:
            out = self.projector(out)
            out = self.to_repr(out)
        return out
