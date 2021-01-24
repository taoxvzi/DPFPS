# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import models
import utils
import copy

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='Implement DPSS by PyTorch')

parser.add_argument('--data-dir', type=str, default='./data', help='path to dataset')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ILSVRC2012'],
                    help='training dataset (default: cifar10)')
parser.add_argument('--save-dir', type=str, default='./snapshot', help='Folder to save checkpoints.')
parser.add_argument('--model', type=str, default='', metavar='PATH', help='import model(default: none)')
parser.add_argument('--arch', type=str, metavar='ARCH', default='vggsmall',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--lambda21', type=float, default=0.1, help='L21 regularization')
parser.add_argument('--flops', action='store_true', help='use pruned flops')
parser.add_argument('--pr', type=float, default=0.94, help='pruned ratio')
parser.add_argument('-stsr', action='store_true', help='stop progressive regulation')

parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--lr', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')



def main():
    global args
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # initial
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Data loading code
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4824, 0.4467], [0.2471, 0.2435, 0.2616])
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        num_classes = 10
        input_pix = 32

    elif args.dataset == 'cifar100':
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        num_classes = 100
        input_pix = 32
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.data_dir, 'train'),
                                 transforms.Compose([
                                     # transforms.Scale(256),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                 ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.data_dir, 'val'),
                                 transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                 ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        num_classes = 1000
        input_pix = 224

    # create model
    model = models.__dict__[args.arch]()
    if args.cuda:
        model.cuda()
    pruned_model = torch.load(args.model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint

    cudnn.benchmark = True
    test_accuracy = validate(val_loader, pruned_model, criterion)

    params_before = utils.print_model_param_nums(model)
    flops_before = utils.count_model_param_flops(model, input_pix)
    params_after = utils.print_model_param_nums(pruned_model.module)
    flops_after = utils.count_model_param_flops(pruned_model.module, input_pix)
    print('Before pruning: \n'
          'params: {}\t''flops: {}\n'
          'After pruning: \n'
          'params: {}\t''flops: {}\n'
          'params_ratio: {pratio:.2f}%\t''flops_ratio: {fratio:.2f}%\n'
          'params_rate: {prate:.2f}\t''flops_rate: {frate:.2f}\n'
          'Prec@1: {top1:.4f}\n'
          'Prec@5: {top5:.4f}'.format(
        params_before, flops_before, params_after, flops_after,
        pratio=(params_before - params_after) * 100. / params_before,
        fratio=(flops_before - flops_after) * 100. / flops_before,
        prate=params_before / params_after,
        frate=flops_before / flops_after,
        top1=test_accuracy[0],
        top5=test_accuracy[1]))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():  # operations inside don't track history
        for i, (data, target) in enumerate(val_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            # compute output
            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
