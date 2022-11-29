import argparse
import json
import os
import random
import re
import sys
import warnings
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from src.data.transform import GaussianBlur_tensor, GaussianBlur

from src.utils.logger import DummyLogger, set_logger

#----------------------------------------------------------------------------

def find_free_tcp_port():
    """
    Find the free port that can be used for Rendezvous on the local machine.
    We use this for 1 machine training where the port is automatically detected.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

#----------------------------------------------------------------------------

def main(args, main_worker):    
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(args.output_dir):
        prev_run_dirs = [x for x in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_desc = datetime.now().strftime("%Y%m%d-%H_%M_%S")
    args.run_dir = os.path.join(args.output_dir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)
    os.makedirs(args.run_dir)
    logger = set_logger(__name__, os.path.join(args.run_dir, "log.txt"), to_console=True)

    # Print options.
    logger.info(
        "\nTraining options: \n" 
        f"{json.dumps(vars(args), indent=2)}\n\n" 
        f"Output directory:   {args.run_dir}\n\n"
    )

    # Create output directory.
    logger.info('Creating output directory...')
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.run_dir, 'command.sh'), 'wt') as command_file:
        command_file.write('python ')
        command_file.write(' '.join(sys.argv))
        command_file.write('\n')

    # Launch processes.
    logger.info('Launching processes...')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

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
    if ngpus_per_node > 1 and args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if args.gpu is None:
            args.gpu = 0
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

#----------------------------------------------------------------------------

def setup_env(
    gpu, 
    ngpus_per_node, 
    args
):
    # Setup
    args.gpu        = gpu
    cudnn.benchmark = True
    init_lr         = args.lr * args.batch_size / 256  # infer learning rate before changing batch size
    # iters_per_epoch = args.num_images // args.batch_size  # TODO: infer iterations per epoch before changing batch size (drop_last=True)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed and args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    # suppress printing if not master
    args.logger = DummyLogger()
    args.tb_writer = DummyLogger()
    if not args.multiprocessing_distributed:
        args.logger = logger = set_logger(__name__, to_console=False)
        args.tb_writer = SummaryWriter(args.run_dir)
    elif args.gpu == 0:
        args.logger = set_logger(f"{__name__}", log_path=os.path.join(args.run_dir, "log.txt"), to_console=True)
        args.logger.propagate = False
        args.tb_writer = SummaryWriter(args.run_dir)

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
    return init_lr

#----------------------------------------------------------------------------

def get_transform(
    input_scale,
    real_image  = True,
    train       = True,
    cifar       = False,
    aug         = True
):
    if cifar:
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1994, 0.2010])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    if train:
        if aug:
            transform_list = [
                # transforms.Normalize(mean=-1.0, std=2.0), # convert [-1, 1] to [0, 1]
                transforms.RandomResizedCrop(input_scale, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                # Gaussian blur
                transforms.RandomHorizontalFlip(),
                # To tensor
                normalize
            ]
        else:
            transform_list = [normalize]
        if aug and not cifar:
            blur_fn = GaussianBlur if real_image else GaussianBlur_tensor
            transform_list.insert(3, transforms.RandomApply([blur_fn([.1, 2.])], p=0.5))
        if real_image:
            transform_list.insert(-1, transforms.ToTensor())
        else:
            transform_list.insert(0, transforms.Normalize(mean=-1.0, std=2.0))
    else:
        transform_list = [
            transforms.Resize(input_scale),  
            transforms.CenterCrop(input_scale),
            transforms.ToTensor(),
            normalize
        ]
    
    return transforms.Compose(transform_list)

#----------------------------------------------------------------------------

def module_to_gpu(model, distributed, gpu):
    has_parameters = len([x for x in model.parameters() if x.requires_grad]) > 0

    if distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # A = torch.nn.SyncBatchNorm.convert_sync_batchnorm(A)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if gpu is not None:
            model.cuda(gpu)
            if has_parameters:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            if has_parameters:
                model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model.cuda(gpu)
    return model

#----------------------------------------------------------------------------
