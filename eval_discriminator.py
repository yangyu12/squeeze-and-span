#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import random
import re
import sys
import time
import warnings
from collections import OrderedDict
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# import torchvision.datasets as datasets
# import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from src.models.misc import adjust_learning_rate, sanity_check, save_checkpoint
from src.models.simsiam import SimSiam
import src.models.backbone as models
from src.models.discriminator import PretrainedDiscriminator
import src.data.dataset as datasets
from src.utils.logger import DummyLogger, set_logger
from src.utils.meters import AverageMeter, ProgressMeter

from src.engine import find_free_tcp_port

logger = set_logger(__name__, to_console=False)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__dict__
    if not name.startswith("__")
    and callable(datasets.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Linear Evaluation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output-dir', metavar='OUTPUT_DIR', 
                    help='path to logging output')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-i', '--input-scale', default=224, type=int, metavar='N',
                    help='The input scale which the representation learning'
                         'network is working on')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
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

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')
parser.add_argument('--regex', default='b32|b16|b8|b4.fc$', type=str,
                    help='')
parser.add_argument('--lars', action='store_true',
                    help='Use LARS')
parser.add_argument('--num-classes', default=-1, type=int,
                    help="number of classes to be predicted")

# dataset
parser.add_argument('--dataset-name', default='CIFAR10', choices=dataset_names)

best_acc1 = 0

#----------------------------------------------------------------------------

def main():
    args = parser.parse_args()

    # Pick output directory.
    prev_run_dirs = []
    if (args.output_dir is None) or (not os.path.isdir(args.output_dir)):
        args.output_dir = os.path.dirname(args.pretrained)
    args.output_dir = os.path.join(args.output_dir, "eval_cls")
    if os.path.isdir(args.output_dir):
        prev_run_dirs = [x for x in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_desc = args.dataset_name
    args.run_dir = os.path.join(args.output_dir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)
    os.makedirs(args.run_dir)
    logger = set_logger(__name__, os.path.join(args.run_dir, "log.txt"), to_console=True)

    # Print options.
    logger.info(
        "\n" 
        "Training options: \n" 
        f"{json.dumps(vars(args), indent=2)}\n" 
        "\n"
        f'Output directory:   {args.run_dir}\n'
        "\n"
    )

    # Create output directory.
    logger.info('Creating output directory...')
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.run_dir, 'command.sh'), 'wt') as command_file:
        command_file.write('python ')
        command_file.write(' '.join(sys.argv))
        command_file.write('\n')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Launch processes.
    logger.info('Launching processes...')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "auto":
        port = find_free_tcp_port()
        args.dist_url = f"tcp://localhost:{port}"
    elif args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(0, ngpus_per_node, args)

#----------------------------------------------------------------------------

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    args.logger = DummyLogger()
    args.tb_writer = DummyLogger()
    if not args.multiprocessing_distributed:
        args.logger = logger
        args.tb_writer = SummaryWriter(args.run_dir)
    elif args.gpu == 0:
        args.logger = set_logger(f"{__name__}", log_path=os.path.join(args.run_dir, "log.txt"), to_console=True)
        args.logger.propagate = False
        args.tb_writer = SummaryWriter(args.run_dir)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
        
    # load from pre-trained, before DistributedDataParallel constructor
    D = PretrainedDiscriminator(args.pretrained, args.regex, batch_size=args.batch_size).eval().requires_grad_(False).cuda()
    D(torch.empty((1, 3, args.input_scale, args.input_scale), device=f"cuda:{args.gpu}"))

    # create model
    args.logger.info("=> creating model '{}'".format(args.arch))
    if args.num_classes == -1:
        args.num_classes = datasets.NUM_CLASSES[args.dataset_name]
    in_channels = sum([x[0] for x in D.f_shape])
    model = nn.Linear(in_channels, args.num_classes)
    # freeze all layers but the last fc
    # if not args.finetune:
    #     for name, param in model.named_parameters():
    #         if name not in ['fc.weight', 'fc.bias']:
    #             param.requires_grad = False
    # init the fc layer
    model.weight.data.normal_(mean=0.0, std=0.01)
    model.bias.data.zero_()
    args.logger.info(model)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2, len(parameters)  # fc.weight, fc.bias

    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.lars:
        args.logger.info("=> use LARS optimizer.")
        from apex.parallel.LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            args.logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            args.logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # if args.dataset_name in ["CIFAR10", "CIFAR100"]:
    #     normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
    #                                     std=[0.2023, 0.1994, 0.2010])
    # else:
    #     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_scale),  # (default) 224 / 96
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(args.input_scale),  # (default) 256 / 128
        transforms.CenterCrop(args.input_scale),
        transforms.ToTensor(),
        normalize,
    ])    
    train_dataset = datasets.__dict__[args.dataset_name](
        args.data, 'train_knn', train_transform
    )
    val_dataset = datasets.__dict__[args.dataset_name](
        args.data, 'val', val_transform
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args.epochs)

        # train for one epoch
        train(train_loader, D, model, criterion, optimizer, epoch, args)

        if (not epoch == 0) and (not (epoch + 1) % 5 == 0) and (not epoch == args.epochs - 1):
            continue

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, D, model, criterion, args)
        args.tb_writer.add_scalar('Val/Acc', acc1, len(train_loader) * (epoch + 1))

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # record the results
        if args.gpu == 0 or (not args.multiprocessing_distributed):
            jsonl_line = json.dumps(dict(top1_acc=acc1.item(), top5_acc=acc5.item(), epoch=epoch, timestamp=time.time()))
            with open(os.path.join(args.run_dir, f'metric-acc.jsonl'), 'at') as f:
                f.write(jsonl_line + '\n')

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename=os.path.join(args.run_dir, 'checkpoint.pth.tar'))
            if epoch == args.start_epoch:
                args.logger.info("=> loading '{}' for sanity check".format(args.pretrained))
                # sanity_check(model.state_dict(), args.pretrained)
                args.logger.info("=> sanity check passed.")

def train(train_loader, D, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            inputs = D(images)
            inputs = [x.mean(dim=[2,3]) if x.ndim==4 else x for x in inputs]
            inputs = torch.cat(inputs, dim=1)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(inputs)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            args.logger.info(progress.display(i))
            args.tb_writer.add_scalar('Train/Loss', loss.item(), len(train_loader) * epoch + i)

def validate(val_loader, D, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            with torch.no_grad():
                inputs = D(images)
                inputs = [x.mean(dim=[2,3]) if x.ndim==4 else x for x in inputs]
                inputs = torch.cat(inputs, dim=1)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(inputs)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                args.logger.info(progress.display(i))

        # TODO: this should also be done with the ProgressMeter
        args.logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
