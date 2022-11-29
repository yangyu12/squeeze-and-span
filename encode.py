import argparse
import json
import os
import sys
import time
from copy import deepcopy
import lpips

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import src.data.dataset as datasets 
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
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
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
parser.add_argument('--num-proj-layers', default=2, type=int,
                    help='number of projection layer (default: 2)')

# autoencoding loss
parser.add_argument('--latent_loss_weight', default=1.0, type=float)

# StyleGAN-ada specific configs:
parser.add_argument('--syn_ratio', type=float, help='If specified, train network with real data as well', default=1.0)
parser.add_argument('--gpath', type=str, help='path to pre-trained GAN pickle')

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
    args.logger.info(f"synthetic batch size: {args.syn_batch_size}, real batch size: {args.real_batch_size}")
    
    # create synthetic data generator
    args.logger.info("=> creating stylegan2-ada")
    if args.syn_batch_size > 0:
        G = SynDataGenerator(args.gpath, regex="").eval().requires_grad_(False).cuda(args.gpu)
        G(torch.empty([1, G.z_dim], device=args.gpu), None)  # get feature shapes
        args.logger.info(f"Generator: \n{str(G)}\n") # print G
    else:
        G = None

    # create real data iterator
    train_dataset = datasets.__dict__[args.dataset_name](
        args.data, 'train', transforms.Compose([
            transforms.Resize(args.input_scale),  
            transforms.CenterCrop(args.input_scale),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
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

    # create encoder
    args.logger.info("=> creating model '{}'".format(args.arch))
    model = ProjExtractor(
        models.__dict__[args.arch], G.w_dim * G.num_ws, 
        num_proj_layers=args.num_proj_layers, 
    )
    model = module_to_gpu(model, args.distributed, args.gpu)
    args.logger.info(f"Model: \n{str(model)}\n") # print model

    # Construct optimizer
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        optim_params, init_lr,
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    model.train().requires_grad_(True)

    # knn evaluator
    if not args.disable_knn:
        knn_evaluator = KNNEvaluator(args.dataset_name, args.data, args.input_scale, args.workers, args.knn_k, args.knn_t)
    
    # training loop
    for epoch in range(args.start_epoch, args.epochs):
        model.train().requires_grad_(True)
        adjust_learning_rate(optimizer, init_lr, epoch, args.epochs)

        # training
        train(
            G, train_iterator, model, optimizer, 
            epoch, iters_per_epoch, args
        )

        if ((not args.multiprocessing_distributed) or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0)) and (epoch + 1) % args.save_freq == 0:
            # evaluate
            if not args.disable_knn:
                model.eval()
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
                filename=os.path.join(args.run_dir, 'encoder_epoch{:04d}.pth.tar'.format(epoch))
            )

#----------------------------------------------------------------------------

def train(
    G, 
    train_iterator,
    model, 
    optimizer, 
    epoch, 
    iters_per_epoch, 
    args,
):
    batch_time  = AverageMeter('Time', ':6.3f')
    data_time   = AverageMeter('Data', ':6.3f')
    losses      = AverageMeter('Loss', ':.4f')
    losses_latent      = AverageMeter('Loss_w', ':.4f')
    losses_image_syn      = AverageMeter('Loss_x_s', ':.4f')
    losses_image_real      = AverageMeter('Loss_x_r', ':.4f')
    progress    = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, losses, losses_latent, losses_image_syn, losses_image_real],
        prefix="Epoch: [{}]".format(epoch)
    )
    
    # construct lpips loss
    lpips_fn = lpips.LPIPS(net='alex', spatial=False).eval().cuda(args.gpu)

    # switch to train mode
    model.train()
    end = time.time()
    for i in range(iters_per_epoch):
        # measure data loading time
        if args.syn_batch_size > 0:
            syn_image, wplus = generate(G, args.rank, args.syn_batch_size)  # syn image
        if args.real_batch_size > 0:
            real_images, _ = next(train_iterator)
            real_images = real_images.cuda(args.gpu, non_blocking=False)
        data_time.update(time.time() - end)

        # synthetic branch
        if args.syn_batch_size > 0:
            pred_w = model(syn_image)  # TODO: wrap with leaky relu
            pred_w = pred_w.reshape(args.syn_batch_size, G.num_ws, -1)
            loss_latent = F.mse_loss(pred_w, wplus)
            rec_image = G.synthesis(pred_w)
            loss_image_s = F.l1_loss(rec_image, syn_image) + lpips_fn(rec_image, syn_image).mean()
            loss_syn = args.latent_loss_weight * loss_latent + loss_image_s
        else:
            loss_syn = 0.        

        # real branch
        if args.real_batch_size > 0:
            pred_wr = model(real_images)
            pred_wr = pred_wr.reshape(args.real_batch_size, G.num_ws, -1)
            rec_image_r = G.synthesis(pred_wr)
            loss_image_r = F.l1_loss(rec_image_r, real_images) + lpips_fn(rec_image_r, real_images).mean()
        else:
            loss_image_r = 0.
        
        # overall loss
        loss = args.syn_ratio * loss_syn + (1 - args.syn_ratio) * loss_image_r
        
        # record loss
        losses.update(loss.item(), args.syn_batch_size)
        if args.syn_batch_size > 0:
            losses_latent.update(loss_latent.item(), args.syn_batch_size)
            losses_image_syn.update(loss_image_s.item(), args.syn_batch_size)
        if args.real_batch_size > 0:
            losses_image_real.update(loss_image_r.item(), args.real_batch_size)

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
            if args.syn_batch_size > 0:
                args.tb_writer.add_scalar('Train/Loss/latent', loss_latent.item(), iters_per_epoch * epoch + i)
                args.tb_writer.add_scalar('Train/Loss/image_syn', loss_image_s.item(), iters_per_epoch * epoch + i)
            if args.real_batch_size > 0:
                args.tb_writer.add_scalar('Train/Loss/image_real', loss_image_r.item(), iters_per_epoch * epoch + i)

#----------------------------------------------------------------------------

def generate(
    G, 
    rank            = None, 
    batch_size      = 512,
):
    with torch.no_grad():
        z = torch.randn([batch_size, G.z_dim], device=f'cuda:{rank}') # latent codes
        wplus = G.mapping(z, None)
        syn_image = G.synthesis(wplus)

    return syn_image, wplus

#----------------------------------------------------------------------------

if __name__ == '__main__':
    args = parser.parse_args()
    main(args, main_worker)

#----------------------------------------------------------------------------
