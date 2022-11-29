import argparse
from collections import OrderedDict
import os
import re
import sys
import math
from turtle import pos
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
import umap
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models.squeeze_module import CompactRepr, VanillaCompactRepr
from src.models.misc import modify_checkpoint_keys
from src.models.generator import SynDataGenerator
from src.models.discriminator import PretrainedDiscriminator
from src.models.simsiam import SimSiam, SimSiamNew
from src.models.extractor import ProjExtractor
from src.utils.logger import DummyLogger, set_logger
from src.models.wrn import WideResNet
# from src.models.vicreg import invariance_loss, covariance_loss, variance_loss

# add dnnlib and torch_utils to PYTHONPATH
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path("./src")

logger = set_logger(__name__, to_console=False)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training with StyleGAN2-ada')
parser.add_argument('--output-dir', metavar='OUTPUT_DIR', default="output", 
                    help='path to logging output')

# plot generative representation umap specific configs:
parser.add_argument('--gpath', type=str,
                    help='path to the pretrained stylegan-ada pickle')
parser.add_argument('--cpath', type=str,
                    help='path to a pretrained classifier')

#----------------------------------------------------------------------------

def main():
    args = parser.parse_args()
    compare_representation(args)

#----------------------------------------------------------------------------

def compare_representation(args):
    # Setup
    # Construct data loader
    # https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=AoliFX5AnBJ0
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    syn_transform = transforms.Compose([
        transforms.Normalize(mean=-1.0, std=2.0), # convert [-1, 1] to [0, 1]
        normalize
    ])
    
    # create generative model, i.e. stylegan2-ada
    print("=> creating stylegan2-ada")
    G = SynDataGenerator(args.gpath).eval().requires_grad_(False).cuda()
    G(torch.empty([1, G.z_dim], device="cuda"), torch.empty([1, G.c_dim], device="cuda"))  # get feature shapes
    assert G.f_shape is not None
    print(f"Generator: \n{str(G)}\n") # print G
    
    # create discriminator
    D = PretrainedDiscriminator(args.gpath).eval().requires_grad_(False).cuda()
    print(f"Discriminator: \n{str(D)}\n")

    # load classifier
    print("=> creating classifier WRN")
    classifier = WideResNet(1, 10)
    print(classifier)

    # 
    assert os.path.isfile(args.cpath)
    if os.path.isfile(args.cpath):
        print("=> loading classifier from '{}'".format(args.cpath))
        checkpoint = torch.load(args.cpath, map_location="cpu")
        # rename moco pre-trained keys
        state_dict = checkpoint['net'] # ['state_dict']
        accepted_keys = list(classifier.state_dict().keys())
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith('module'):
                # remove prefix
                state_dict[k[len("module."):]] = state_dict[k]
            if not k in accepted_keys:
                # delete renamed or unused k
                del state_dict[k]

        msg = classifier.load_state_dict(state_dict, strict=False)
        print("Missing keys: ", msg.missing_keys)
        print("=> loaded pre-trained model '{}'".format(args.cpath))
    else:
        raise KeyError("=> no checkpoint found at '{}'".format(args.cpath))
    classifier = classifier.eval().requires_grad_(False).cuda()

    # 
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0"
    
    data = {
        "Dfeat": [],
        "Latent": [],
        "Gfeat": [],
    }
    Y = []

    # set module to eval model and collect all feature representations
    num_samples     = 50000
    batch_size      = 16
    with torch.no_grad():
        for _ in tqdm(range(num_samples // batch_size), desc="Collecting features"):
            # Generate data and assign pseudo labels
            z = torch.randn([batch_size, G.z_dim], device=device)
            # latent
            wplus = G.mapping(z, None)
            latent = wplus.flatten(start_dim=1)
            
            # gfeat
            syn_image, gfeat = G.forward_w(wplus) # NCHW, float32, dynamic range [-1, +1]
            gfeat = torch.cat([x.mean(dim=[2, 3]) for x in gfeat], dim=1)
            
            # dfeat
            dfeat = D(syn_image)
            dfeat = torch.cat([x.mean(dim=[2, 3]) if x.ndim==4 else x for x in dfeat], dim=1)
            
            # label
            syn_image = syn_transform(syn_image)
            y = classifier(syn_image).max(dim=1).indices
            # 
            data["Dfeat"].append(dfeat.cpu())
            data["Latent"].append(latent.cpu())
            data["Gfeat"].append(gfeat.cpu())
            Y.append(y.cpu())
            
    data = {k: torch.cat(v, dim=0).numpy() for k, v in data.items()}
    Y = torch.cat(Y, dim=0)
    num_classes = len(torch.unique(Y))
    Y = Y.numpy()
        
    def plot_umap(X, filename):
        X = umap.UMAP(n_components=2).fit_transform(X)

        # passing to dataframe
        df = pd.DataFrame()
        df["feat_1"] = X[:, 0]
        df["feat_2"] = X[:, 1]
        df["Y"] = Y
        plt.figure(figsize=(9, 9))
        ax = sns.scatterplot(
            x="feat_1",
            y="feat_2",
            hue="Y",
            palette=sns.color_palette("hls", num_classes),
            data=df,
            legend="full",
            alpha=0.3,
        )
        ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
        ax.tick_params(left=False, right=False, bottom=False, top=False)

        # manually improve quality of imagenet umaps
        anchor = (0.5, 1.35)

        plt.legend(loc="upper center", bbox_to_anchor=anchor, ncol=math.ceil(num_classes / 10))
        plt.tight_layout()

        # save plot locally as well
        plt.savefig(filename)
        print(f'plot saved at {filename}')
        plt.close()
        
    for k, v in data.items():
        plot_umap(v, os.path.join(args.output_dir, f"{k}_umap.pdf"))

#----------------------------------------------------------------------------

if __name__ == '__main__':
    main()

#----------------------------------------------------------------------------


