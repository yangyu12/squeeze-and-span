import pickle
import re
import sys

import torch


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path("./src")  

#----------------------------------------------------------------------------

class SynDataGenerator(torch.nn.Module):
    """
    Generator wrapper for saving inner features for output.
    Suitable for StyleGAN2
    """
    def __init__(self,
        generator_pkl,                                   #
        regex               = r'.*synthesis\.b(\d+)$',   # TODO: Match all the convolutional layers  r'.*synthesis\.b(\d+)\.conv$'
    ):
        super().__init__()

        try:
            with open(generator_pkl, 'rb') as f:
                self.G = pickle.load(f)['G_ema']  
        except ModuleNotFoundError:
            with open(generator_pkl, 'rb') as f:
                self.G = legacy.load_network_pkl(f)['G_ema']
        
        self.synthesis = self.G.synthesis
        self.mapping = self.G.mapping

        # Default
        # ---
        # b4    conv1           (512, 4, 4)
        # b8    conv0/conv1     (512, 8, 8)
        # b16   conv0/conv1     (512, 16, 16)
        # b32   conv0/conv1     (512, 32, 32)
        # ------------------------------------
        # b64   conv0/conv1     (512, 64, 64)
        # b128  conv0/conv1     (256, 128, 128)
        # b256  conv0/conv1     (128, 256, 256)
        # b512  conv0/conv1     (64, 512, 512)
        # -------------------------------------
        # f_dim = 512 * (1 + 2 + 2 + 2 + 2) + 256 * 2 + 128 * 2 + 64 * 2 = 5504

        # Copy some attributes for convenience
        self.z_dim = self.G.z_dim
        self.c_dim = self.G.c_dim
        self.w_dim = self.G.w_dim
        self.num_ws = self.G.num_ws
        self.img_resolution = self.G.img_resolution
        self.img_channels = self.G.img_channels

        # extract intermediate features with hooks.
        # Reference: https://medium.com/the-owl/using-forward-hooks-to-extract-intermediate-layer-outputs-from-a-pre-trained-model-in-pytorch-1ec17af78712
        # fix bug in data parallel
        # https://discuss.pytorch.org/t/aggregating-the-results-of-forward-backward-hook-on-nn-dataparallel-multi-gpu/28981/8
        # https://github.com/kazuto1011/grad-cam-pytorch/issues/13#issuecomment-486071455
        self.inner_features = []
        self.fhooks = []
        self.f_shape = None
        for name, mod in self.G.named_modules():
            if re.fullmatch(regex, name):
                # print(name)
                self.fhooks.append(mod.register_forward_hook(self.forward_hook))

    def forward_hook(self, module, input, output):
        # TODO: seems like we are not using same feature maps as DatasetGAN
        if isinstance(output, torch.Tensor):
            if output.ndim == 3: # w variable
                output = output[:, 0, :]  # deduplicate 
            self.inner_features.append(output.float())
        else: # StyleGAN2 synthesis block typically return 2 tensors
            assert len(output) > 1
            channels = [x.size(1) for x in output]
            max_ch_idx = channels.index(max(channels))
            self.inner_features.append(output[max_ch_idx].float())

    def forward(self, z, c, noise_mode='random'):
        self.inner_features = []
        out = self.G(z, c, noise_mode=noise_mode)
        if self.f_shape is None:
            self.f_shape = [x.shape[1:] for x in self.inner_features]
        return out, self.inner_features
    
    def forward_w(self, w, noise_mode='random'):
        self.inner_features = []
        out = self.G.synthesis(w, noise_mode=noise_mode)
        if self.f_shape is None:
            self.f_shape = [x.shape[1:] for x in self.inner_features]
        return out, self.inner_features

#----------------------------------------------------------------------------
