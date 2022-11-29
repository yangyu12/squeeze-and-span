import argparse
import json
import os
import sys
import time
from copy import deepcopy
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from src.data.transform import TwoCropsTransform
import src.data.dataset as datasets 
from models.squeeze_module import CompactRepr, VanillaCompactRepr
from src.models.knn import KNNEvaluator
from src.models.misc import adjust_learning_rate, save_checkpoint
from src.torch_utils.misc import InfiniteSampler
from src.models.extractor import ProjExtractor
import src.models.backbone as models
from src.models.generator import SynDataGenerator
from src.utils.logger import DummyLogger, set_logger
from src.utils.meters import AverageMeter, ProgressMeter
from src.torch_utils.misc import strip_ddp
from src.engine import main, setup_env, get_transform, module_to_gpu


# add dnnlib and torch_utils to PYTHONPATH
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path("./src")

logger = set_logger(__name__, to_console=False)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__dict__
    if not name.startswith("__")
    and callable(datasets.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training with StyleGAN2-ada')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset-name', default='CIFAR10', choices=dataset_names)
parser.add_argument('--output-dir', metavar='OUTPUT_DIR', default="output", 
                    help='path to logging output')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total iterations to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial (base) learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--save-freq', default=20, type=int, 
                    metavar='SAVE_FREQ', help='save frequency wrt round (default: 20)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency wrt iter (default: 10)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='auto', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# vicreg specific configs:
parser.add_argument('-i', '--input-scale', default=32, type=int, metavar='N',
                    help='The input scale which the representation learning'
                         'network is working on')
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--num-proj-layers', default=2, type=int,
                    help='number of projection layer (default: 5)')
parser.add_argument('--a_depth', default=3, type=int, help='')
parser.add_argument('--a_hidden', default=2048, type=int, help='')

# distillation loss
parser.add_argument('--sim_loss_weight', default=25.0, type=float)
parser.add_argument('--var_loss_weight', default=25.0, type=float)
parser.add_argument('--cov_loss_weight', default=1.0, type=float)

# StyleGAN-ada specific configs:
parser.add_argument('--syn_ratio', type=float, help='If specified, train network with real data as well', default=1.)
parser.add_argument('--gpath', type=str, help='path to pre-trained GAN pickle')
parser.add_argument('--regex', type=str, help='', default=r'.*synthesis\.b(\d+)$')

# MISC specific configs
parser.add_argument('--augment-syn-data', action='store_true',
                    help='augment synthetic data')

# knn specific configs
parser.add_argument('--disable-knn', action='store_true', help='')
parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor')

#----------------------------------------------------------------------------

def main_worker(gpu, ngpus_per_node, args):
    # Setup
    init_lr                 = setup_env(gpu, ngpus_per_node, args)
    args.syn_batch_size     = int(args.batch_size * args.syn_ratio)
    args.real_batch_size    = args.batch_size - args.syn_batch_size
    assert args.syn_batch_size > 0
    args.logger.info(f"synthetic batch size: {args.syn_batch_size}, real batch size: {args.real_batch_size}")
    
    # create synthetic data generator
    syn_transform = get_transform(args.input_scale, real_image=False, aug=args.augment_syn_data, 
                                  cifar=args.dataset_name in ["CIFAR10", "CIFAR100"])
    args.logger.info("=> creating stylegan2-ada")
    assert os.path.isfile(args.gpath), f"File {args.gpath} does not exist."
    G = SynDataGenerator(args.gpath, regex=args.regex).eval().requires_grad_(False).cuda(args.gpu)
    G(torch.empty([1, G.z_dim], device=args.gpu), None)  # get feature shapes
    args.logger.info(f"Generator: \n{str(G)}\n") # print G

    # create real data iterator
    train_dataset = datasets.__dict__[args.dataset_name](
        args.data, 'train', TwoCropsTransform(
            get_transform(args.input_scale, 
                          cifar=args.dataset_name in ["CIFAR10", "CIFAR100"])
        )
    )
    train_sampler = InfiniteSampler(train_dataset, rank=args.rank, num_replicas=args.world_size)
    if args.real_batch_size > 0:
        train_iterator = iter(torch.utils.data.DataLoader(
            train_dataset, batch_size=args.real_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False, 
        ))
    else:
        train_iterator = None
    iters_per_epoch = len(train_dataset) // (args.batch_size * ngpus_per_node)

    # create model
    args.logger.info("=> creating model '{}'".format(args.arch))
    outdim = args.dim
    if args.a_depth == 0:
        outdim = sum([x[0] for x in G.f_shape])
    model = ProjExtractor(
        models.__dict__[args.arch], outdim, 
        num_proj_layers=args.num_proj_layers, 
    )
    model = module_to_gpu(model, args.distributed, args.gpu)
    args.logger.info(f"Model: \n{str(model)}\n") # print model

    # create annotator
    args.logger.info("=> creating annotator")
    if args.a_depth > 0:
        A = CompactRepr(
            input_shapes=G.f_shape, tmp_channels=args.a_hidden, 
            output_channels=args.dim, num_layers=args.a_depth,    
        )
    else:
        A = VanillaCompactRepr()
    if args.freeze_a:
        A = A.cuda(args.gpu).eval().requires_grad_(False)
    else:
        A = module_to_gpu(A, args.distributed, args.gpu)
    args.logger.info(f"Annotator: \n{str(A)}\n") # print A

    # Construct optimizer
    optim_params = [p for p in model.parameters() if p.requires_grad] + \
                   [p for p in A.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        optim_params, init_lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )

    model.train().requires_grad_(True)
    if not args.freeze_a:
        A.train().requires_grad_(True)

    # knn evaluator
    if not args.disable_knn:
        knn_evaluator = KNNEvaluator(args.dataset_name, args.data, args.input_scale, args.workers, args.knn_k, args.knn_t)
    
    # training loop
    for epoch in range(args.start_epoch, args.epochs):
        model.train().requires_grad_(True)
        if not args.freeze_a:
            A.train().requires_grad_(True)

        adjust_learning_rate(optimizer, init_lr, epoch, args.epochs)

        # training
        train(
            G, A, train_iterator, model, optimizer, syn_transform,
            epoch, iters_per_epoch, args
        )

        if ((not args.multiprocessing_distributed) or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0)) and (epoch + 1) % args.save_freq == 0:
            # evaluate
            if not args.disable_knn:
                model.eval()
                A.eval()
                with torch.no_grad():
                    backbone = deepcopy(strip_ddp(model).backbone).eval().requires_grad_(False)
                    knn_acc = knn_evaluator.evaluate(backbone)
                args.logger.info("\t".join([f"Epoch: [{epoch:8d}]", f"KNN Acc: {knn_acc:6.3f}%"]))
                args.tb_writer.add_scalar('Val/KNN_Acc', knn_acc, iters_per_epoch * (epoch + 1))

                # record
                jsonl_line = json.dumps(dict(knn_acc=knn_acc, epoch=epoch, timestamp=time.time()))
                with open(os.path.join(args.run_dir, f'metric-knn-acc.jsonl'), 'at') as f:
                    f.write(jsonl_line + '\n')

            # save
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, 
                filename=os.path.join(args.run_dir, 'S_epoch{:04d}.pth.tar'.format(epoch))
            )
            save_checkpoint(
                {
                    'state_dict': A.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, 
                filename=os.path.join(args.run_dir, 'A_epoch{:04d}.pth.tar'.format(epoch))
            )

#----------------------------------------------------------------------------

def train(
    G, 
    A,
    train_iterator,
    model, 
    optimizer, 
    transform,
    epoch, 
    iters_per_epoch, 
    args,
):
    batch_time  = AverageMeter('Time', ':6.3f')
    data_time   = AverageMeter('Data', ':6.3f')
    losses      = AverageMeter('Loss', ':.4f')
    losses_sim  = AverageMeter('Loss_sim', ':.4f')
    losses_var  = AverageMeter('Loss_var', ':.4f')
    losses_cov  = AverageMeter('Loss_cov', ':.4f')
    progress    = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, losses, losses_sim, losses_var, losses_cov],
        prefix="Epoch: [{}]".format(epoch)
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(iters_per_epoch):
        # measure data loading time
        syn_image, zg = generate(G, A, transform, args.rank, args.syn_batch_size)  # syn image
        if train_iterator is not None:
            real_images, _ = next(train_iterator)
            for j in range(len(real_images)):
                real_images[j] = real_images[j].cuda(args.gpu, non_blocking=False)
        data_time.update(time.time() - end)

        # synthetic branch
        zf = model(syn_image)

        # compute loss
        sim_loss = F.mse_loss(zg, zf)
        var_loss_g = variance_loss(zg)
        var_loss_f = variance_loss(zf)
        cov_loss_g = covariance_loss(zg)
        cov_loss_f = covariance_loss(zf)

        # 
        loss = args.sim_loss_weight * sim_loss + \
               args.var_loss_weight * var_loss_f + \
               args.cov_loss_weight * cov_loss_f
        losses.update(loss.item(), args.batch_size)
        losses_sim.update(sim_loss.item(), args.batch_size)
        losses_var.update((var_loss_g + var_loss_f).item(), args.batch_size)
        losses_cov.update((cov_loss_g + cov_loss_f).item(), args.batch_size)

        # real branch
        if train_iterator is not None:
            zr1 = model(real_images[0])
            zr2 = model(real_images[1])

            sim_loss_r = F.mse_loss(zr1, zr2)
            var_loss_r1 = variance_loss(zr1)
            var_loss_r2 = variance_loss(zr2)
            cov_loss_r1 = covariance_loss(zr1)
            cov_loss_r2 = covariance_loss(zr2)

            loss_real = args.sim_loss_weight * sim_loss_r + \
                        args.var_loss_weight * (var_loss_r1 + var_loss_r2) + \
                        args.cov_loss_weight * (cov_loss_r1 + cov_loss_r2)
            
            loss = args.syn_ratio * loss + (1 - args.syn_ratio) * loss_real

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            args.logger.info(progress.display(i))
            args.tb_writer.add_scalar('Train/Loss', loss.item(), iters_per_epoch * epoch + i)
            args.tb_writer.add_scalar('Train/Loss/invariance', sim_loss.item(), iters_per_epoch * epoch + i)
            args.tb_writer.add_scalar('Train/Loss/variance_f', var_loss_f.item(), iters_per_epoch * epoch + i)
            args.tb_writer.add_scalar('Train/Loss/variance_g', var_loss_g.item(), iters_per_epoch * epoch + i)
            args.tb_writer.add_scalar('Train/Loss/covariance_f', cov_loss_f.item(), iters_per_epoch * epoch + i)
            args.tb_writer.add_scalar('Train/Loss/covariance_g', cov_loss_g.item(), iters_per_epoch * epoch + i)

#----------------------------------------------------------------------------

def variance_loss(z: torch.Tensor) -> torch.Tensor:
    """Computes variance loss given batch of projected features z.

    Args:
        z (torch.Tensor): NxD Tensor containing projected features from view 1.

    Returns:
        torch.Tensor: variance regularization loss.
    """

    eps = 1e-4
    std_z = torch.sqrt(z.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z))
    return std_loss

#----------------------------------------------------------------------------

def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Computes covariance loss given batch of projected features z.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.

    Returns:
        torch.Tensor: covariance regularization loss.
    """

    N, D = z.size()

    z = z - z.mean(dim=0)
    cov_z = (z.T @ z) / (N - 1)

    diag = torch.eye(D, device=z.device)
    cov_loss = cov_z[~diag.bool()].pow_(2).sum() / D
    return cov_loss

#----------------------------------------------------------------------------

def generate(
    G, 
    A,
    transform, 
    rank            = None, 
    batch_size      = 512,
):
    with torch.no_grad():
        z = torch.randn([batch_size, G.z_dim], device=f'cuda:{rank}') # latent codes
        syn_image, feat = G(z, None) # NCHW, float32, dynamic range [-1, +1]

        # multi-view augmentation of input data
        syn_image = syn_image.clamp(min=-1., max=1.)
        syn_image = transform(syn_image)
        syn_image = syn_image.contiguous()

    # squeezed representation
    c = A(feat) 
    return syn_image, c

#----------------------------------------------------------------------------

if __name__ == '__main__':
    args = parser.parse_args()
    main(args, main_worker)

#----------------------------------------------------------------------------
