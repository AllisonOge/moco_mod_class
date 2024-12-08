"""
Script to run all ssl training experiments

code is an adaptation of the repo https://github.com/facebookresearch/moco-v3
"""
from typing import Any
import builtins
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import shutil
import random
import warnings
import math
import torch
import torch.distributed as dist
from moco_models import MoCo_MobileNet
import moco_optimizer as moco_opt
from torchvision.transforms import Compose, Lambda
import transforms as RF_txfms
from dataset import MoCo_H5_Dataset
import moco_dataloader as moco_loader
import torch.backends.cudnn as cudnn
import time
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='SSL Training')
parser.add_argument('--arch', "-a", default='mobilenetv3_small_050', type=str,
                    choices=['mobilenetv3_small_050', 'mobilenetv3_small_075',
                             'mobilenetv3_small_100', 'mobilenetv3_large_100'],
                    help='model architecture: ' +
                    ' | '.join(['mobilenetv3_small_050', 'mobilenetv3_small_075',
                                'mobilenetv3_small_100', 'mobilenetv3_large_100']) +
                    ' (default: mobilenetv3_small_050)')
parser.add_argument('--data', metavar='DIR', required=True,
                    help='path to dataset')
parser.add_argument('--output-dir', default=None, metavar='DIR',
                    help='path to save checkpoint')
parser.add_argument('--optimizer', default='lars', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer: ' +
                    ' | '.join(['lars', 'adamw']) +
                    ' (default: lars)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--num-workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

parser.add_argument('--aug-rate', '-r', type=float, metavar='NAME', default=0.2,
                    choices=[.1, .2, .3, .4, .5, .6, .7, .8, .9],
                    help='zero masking augmentation rate')


parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument("--master-gpu", default=0, type=int,
                    help="Master GPU for multi-GPU training")

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a '
                         'half-cycle cosine schedule')
parser.add_argument('--moco-t', default=1.0, type=float,
                    help='softmax temperature (default: 1.0)')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # adjust output dir if not set
    if not args.output_dir:
        args.output_dir = f'./saved_models/{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    if "RANK" in os.environ:
        args.rank = int(os.environ["RANK"])

    # automatically set the GPU for each process
    if "LOCAL_RANK" in os.environ:
        args.gpu = int(os.environ["LOCAL_RANK"])

    # distributed mode is enabled by default if WORLD_SIZE > 1
    args.distributed = int(os.environ.get("WORLD_SIZE", "0")) > 1

    # Run the main_worker directly since torchrun will manage process creation
    main_worker(args.gpu, args)


def main_worker(gpu: int, args: Any):
    """
    :param gpu: gpu id
    :param args: arguments
    """
    args.gpu = gpu
    # suprress printing if not the main process
    if args.distributed and args.gpu != args.master_gpu:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        # distributed settings are handled by torchrun
        dist.init_process_group(backend=args.dist_backend)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = MoCo_MobileNet(args.arch, dim=args.moco_dim,
                           mlp_dim=args.moco_mlp_dim)

    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 4096
    print(f"Learning rate adjusted to {args.lr}")

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(
                args.batch_size / int(os.environ['WORLD_SIZE']))
            print(f"Batch size adjusted to {args.batch_size}")
            ngpus_per_node = torch.cuda.device_count()
            args.num_workers = int(
                (args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[gpu])

    if args.optimizer == 'lars':
        optimizer = moco_opt.LARS(model.parameters(), args.lr,
                                  weight_decay=args.weight_decay,
                                  momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                      weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler()
    summary_writer = SummaryWriter() if args.gpu == args.master_gpu else None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(
                    args.resume, map_location=loc, weights_only=True)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # follow zero masking augmentation recipe: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9930826
    # with some modifications
    augmentation1 = Compose([
        Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
        RF_txfms.RandomZeroMasking(
            max_rate=args.aug_rate, dim=-1),
    ])

    augmentation2 = Compose([
        Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
        RF_txfms.RandomZeroMasking(
            max_rate=args.aug_rate, dim=-1),
    ])

    train_dataset = MoCo_H5_Dataset(args.data, data_txfm=moco_loader.MoCoTransform(base_transform1=augmentation1,
                                                                                   base_transform2=augmentation2))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    tracker = Tracker('loss', mode='min')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        loss = train(train_loader, model, optimizer,
                     scaler, summary_writer, epoch, args)

        if not args.distributed or (args.distributed
                                    and args.rank == args.master_gpu):  # only the master GPU saves checkpoint
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"Saving checkpoint to {args.output_dir}")
            is_best = tracker.operator(loss, tracker.best)
            tracker.best = min(tracker.best, loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best=is_best, filename=os.path.join(args.output_dir, 'checkpoint_%04d.pth.tar' % epoch))

    if args.rank == args.master_gpu:
        summary_writer.close()


def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    for i, views in enumerate(train_loader):  # ignore labels and indices
        # save some views for visualization
        if i == 0 and args.rank == args.master_gpu:
            summary_writer.add_images(
                'views', plot_data(views, 10), global_step=epoch * iters_per_epoch + i)
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        if args.gpu is not None:
            views[0] = views[0].cuda(args.gpu, non_blocking=True)
            views[1] = views[1].cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.autocast(device_type='cuda'):
            loss = model(views[0], views[1], moco_m)

        losses.update(loss.item(), views[0].size(0))
        if args.rank == args.master_gpu:
            summary_writer.add_scalar(
                "loss", loss.item(), epoch * iters_per_epoch + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print("Saving best model")
        parent_dir = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(
            parent_dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Tracker:
    def __init__(self, metric, mode='auto'):
        self.metric = metric
        self.mode = mode
        self.mode_dict = {
            'auto': np.less if 'loss' in metric else np.greater,
            'min': np.less,
            'max': np.greater
        }
        self.operator = self.mode_dict[mode]

        self._best = np.inf if 'loss' in metric else -np.inf

    @property
    def best(self):
        return self._best

    @best.setter
    def best(self, value):
        self._best = value


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch -
                                                       args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch /
                                  args.epochs)) * (1. - args.moco_m)
    return m


def plot_data(data, num_images):
    """Plot the IQ data"""
    images = []
    view1 = data[0]
    view2 = data[1]
    for i in range(num_images):
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))

        axs[0].plot(view1[i][0].detach().cpu().numpy().T)
        axs[1].plot(view2[i][1].detach().cpu().numpy().T)

        axs[0].set_xlabel('Samples')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title('View 1')
        axs[1].set_xlabel('Samples')
        axs[1].set_ylabel('Amplitude')
        axs[1].set_title('View 2')
        plt.tight_layout()

        # Convert the plot to a tensor
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, :3]

        plt.close(fig)

        # Convert to torch tensor and permute to (C, H, W)
        image_tensor = torch.tensor(image).permute(2, 0, 1)
        images.append(image_tensor)
    return torch.stack(images)


if __name__ == "__main__":
    main()
