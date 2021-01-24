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
parser.add_argument('--lambda21', type=float, default=0.01, help='L21 regularization')
parser.add_argument('--flops', action='store_true', help='use pruned flops')
parser.add_argument('--pr', type=float, default=0.94, help='pruned ratio')
parser.add_argument('-stsr', action='store_true', help='stop progressive regulation')

parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
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

best_prec1 = 0


def main():
    global args, best_prec1, log, log1, log2, log3
    args = parser.parse_args()
    args.save_time = args.arch + '_experiment_' + args.dataset + '_' + '%.4f' % (args.lambda21) + '_' + str(args.flops) + '_flops_' + '%.3f' % (
    args.pr) + '_' + '%d' % (args.epochs)
    args.save_dir = os.path.join('./snapshot/', args.dataset + '_experiment_' + args.arch + '_' + 'lambda21' + '_' + '%d' % (args.epochs))
    args.save_time = args.save_time + '_dpss_sigmoid_filter_sum_context_add_scratch_' + time.strftime("%Y%m%d%H%M", time.localtime())
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, '{}.log'.format(args.save_time)), 'w')
    log1 = open(os.path.join(args.save_dir, '{}_rate_epoch.txt'.format(args.save_time)), 'w')
    log2 = open(os.path.join(args.save_dir, '{}_sparsity_ratio_epoch.txt'.format(args.save_time)), 'w')
    log3 = open(os.path.join(args.save_dir, '{}_sparsity_allocation_ratio_epoch.txt'.format(args.save_time)), 'w')
    model_save_name = os.path.join(args.save_dir, args.save_time)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print_log('Parameters setting: \n''epochs: {}\t''lr: {}\n'.format(args.epochs, args.lr), log)

    # initial
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Data loading code
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_dir, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Pad(4),
                                 transforms.RandomCrop(32),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.4914, 0.4824, 0.4467], [0.2471, 0.2435, 0.2616])
                             ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4824, 0.4467], [0.2471, 0.2435, 0.2616])
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        num_classes = 10
        input_pix = 32

    elif args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data_dir, train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.Pad(4),
                                  transforms.RandomCrop(32),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
                              ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
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

    # multi-gpu
    gpu_num = torch.cuda.device_count()
    print_log('GPU NUM: {:2d}'.format(gpu_num), log)
    model = torch.nn.DataParallel(model, list(range(gpu_num))).cuda()
    # if gpu_num > 1:
    #     model = torch.nn.DataParallel(model, list(range(gpu_num))).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            model.load_state_dict(torch.load(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    method = DPSS(model, args.lambda21, args.pr)
    model_pruning = copy.deepcopy(model)
    params_before = utils.print_model_param_nums(model_pruning)
    flops_before = utils.count_model_param_flops(model_pruning, input_pix)
    validate(val_loader, model, criterion)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    next_ratio = 1.
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        model_pruning = copy.deepcopy(model)
        method1 = DPSS(model_pruning, args.lambda21, args.pr)
        # method1.adjust_scale_coe(next_ratio)
        method1.channel_prune()
        print_log(str(method1.layer_sparsity_ratio), log2)
        params_pruning = utils.print_model_param_nums(method1.model.module)
        flops_pruning = utils.count_model_param_flops(method1.model.module, input_pix)
        pruned_params = 1 - params_pruning / params_before
        pruned_flops = 1 - flops_pruning / flops_before
        if not args.flops:
            next_ratio = get_new_scale(args.pr, pruned_params, copy.deepcopy(model), input_pix, params_before, next_ratio)
        else:
            next_ratio = get_new_scale(args.pr, pruned_flops, copy.deepcopy(model), input_pix, flops_before, next_ratio, args.flops)
        method.adjust_scale_coe(next_ratio)
        model_pruning1 = copy.deepcopy(model)
        method2 = DPSS(model_pruning1, args.lambda21, args.pr)
        method2.adjust_scale_coe(next_ratio)
        method2.channel_prune()
        print_log(str(method2.layer_sparsity_allocation_ratio), log3)
        print(next_ratio)
        train_loss = train(train_loader, model, criterion, optimizer, epoch, method, True)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(model, model_save_name, is_best)

        print_log('{}\t''{trn_loss:.4f}\t''{tst_acc:.4f}\t''{pratio:.4f}\t''{fratio:.4f}\t'.format(epoch, trn_loss=train_loss, tst_acc=prec1, pratio=(params_before - params_pruning) / params_before, fratio=(flops_before - flops_pruning) / flops_before), log1)
        print_log('{}\t''{trn_loss:.4f}\t''{tst_acc:.4f}\t''{pratio:.4f}\t''{fratio:.4f}\t'.format(epoch, trn_loss=train_loss, tst_acc=prec1, pratio=(params_before - params_pruning) / params_before, fratio=(flops_before - flops_pruning) / flops_before), log)

    print_log('Get new model\n', log)
    model = torch.load(model_save_name + '_best.pth.tar')
    if args.cuda:
        model.cuda()
    method2 = DPSS(model, args.lambda21, args.pr)
    method2.channel_prune()
    model = method2.model
    if args.cuda:
        model.cuda()
    validate(val_loader, model, criterion)
    save_checkpoint(model, model_save_name + '_pruned', False)
    params_after = utils.print_model_param_nums(model.module)
    flops_after = utils.count_model_param_flops(model.module, input_pix)
    print_log('Before pruning: \n'
          'params: {}\t''flops: {}\n'
          'After pruning: \n'
          'params: {}\t''flops: {}\n'
          'params_ratio: {pratio:.2f}%\t''flops_ratio: {fratio:.2f}%\n'
          'params_rate: {prate:.2f}\t''flops_rate: {frate:.2f}\n'
          'Prec@1: {top1:.4f}'.format(
        params_before, flops_before, params_after, flops_after,
        pratio=(params_before - params_after) * 100. / params_before,
        fratio=(flops_before - flops_after) * 100. / flops_before,
        prate=params_before / params_after,
        frate=flops_before / flops_after,
        top1=best_prec1), log)
    log.close()
    log1.close()
    log2.close()
    log3.close()


def get_new_scale(dest_pr, input_pr, input_model, input_pix, input_before, input_ratio=1., flops=False):
    e_pr = dest_pr
    v_pr = input_pr
    get_ratio = input_ratio

    if abs(v_pr - e_pr) < 0.002:
        get_ratio = 1.0
    else:
        model_pruning = copy.deepcopy(input_model)
        method1 = DPSS(model_pruning, args.lambda21, args.pr)
        method1.adjust_scale_coe(get_ratio)
        method1.channel_prune()
        model_pruning = method1.model
        params_pruning = utils.print_model_param_nums(model_pruning.module)
        flops_pruning = utils.count_model_param_flops(model_pruning.module, input_pix)
        if not flops:
            v_pr = 1 - params_pruning / input_before
        else:
            v_pr = 1 - flops_pruning / input_before
        while v_pr > e_pr:
            get_ratio += 0.005
            model_pruning = copy.deepcopy(input_model)
            method1 = DPSS(model_pruning, args.lambda21, args.pr)
            method1.adjust_scale_coe(get_ratio)
            method1.channel_prune()
            model_pruning = method1.model
            params_pruning = utils.print_model_param_nums(model_pruning.module)
            flops_pruning = utils.count_model_param_flops(model_pruning.module, input_pix)
            if not flops:
                v_pr = 1 - params_pruning / input_before
            else:
                v_pr = 1 - flops_pruning / input_before
        else:
            while v_pr < e_pr:
                get_ratio -= 0.005
                model_pruning = copy.deepcopy(input_model)
                method1 = DPSS(model_pruning, args.lambda21, args.pr)
                method1.adjust_scale_coe(get_ratio)
                method1.channel_prune()
                model_pruning = method1.model
                params_pruning = utils.print_model_param_nums(model_pruning.module)
                flops_pruning = utils.count_model_param_flops(model_pruning.module, input_pix)
                if not flops:
                    v_pr = 1 - params_pruning / input_before
                else:
                    v_pr = 1 - flops_pruning / input_before

    return get_ratio


def train(train_loader, model, criterion, optimizer, epoch, method, method_flag=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    s_e = sparse_coefficent(epoch, args)
    method.sparse_coefficent_value(s_e, args.stsr)
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

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

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if i % 5 == 0:
        if method_flag:
            method.model = model
            method.model_weight_update()
            model = method.model
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t''lambda: {s_e:.5f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, s_e=s_e), log)
    # method.print_index()

    return losses.avg


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
                print_log('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5), log)

    print_log(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5), log)

    return top1.avg


def save_checkpoint(state, save_name, is_best):
    torch.save(state, save_name + '_model.pth.tar')
    if is_best:
        shutil.copyfile(save_name + '_model.pth.tar', save_name + '_best.pth.tar')


def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset == 'ILSVRC2012':
        epoch_step = [30, 60, 90]
    else:
        # epoch_step = [60, 120, 160]
        epoch_step = [0.5*args.epochs, 0.75*args.epochs]
        # epoch_step = [120, 160]
    if epoch in epoch_step:
        # args.lr *= 0.2
        args.lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


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


def sparse_coefficent(epoch, args):
    if args.dataset == 'ILSVRC2012':
        epoch_step = [30, 60, 90]
    else:
        # epoch_step = [60, 120, 160]
        epoch_step = [0.5*args.epochs, 0.75*args.epochs]
    s_e = 1.
    if epoch < epoch_step[0]:
        s_e = 1 / (1 + math.exp(15 - 30 * (epoch + 1) / epoch_step[0]))


    return s_e


class DPSS:
    def __init__(self, model, lad21, prt):
        self.model = model
        if args.arch == 'vggsmall':
            self.list1 = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
        elif args.arch == 'vggvariant':
            # self.list1 = [0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40, 43, 46, 49]
            self.list1 = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
        else:
            if args.dataset == 'ILSVRC2012':
                self.list1 = ['layer1', 'layer2', 'layer3', 'layer4']
            else:
                self.list1 = ['layer1', 'layer2', 'layer3']
        self.lambda_L21 = lad21
        self.prune_ratio = prt
        self.scale_coe = 1.
        self.s_c_v = 1.
        self.prune_thr = 1e-2
        self.layer_sparsity_ratio = []
        self.layer_sparsity_allocation_ratio = []

    def adjust_scale_coe(self, scale_coe):
        self.scale_coe = scale_coe

    def sparse_coefficent_value(self, value, stsr_flag):
        if not stsr_flag:
            self.s_c_v = value

    def get_update_index(self, conv_layer, next_conv_layer):
        weight_copy = conv_layer.weight.data.clone()
        weight_copy_next = next_conv_layer.weight.data.clone()
        c_l21_norm = torch.norm(torch.norm(torch.norm(weight_copy.data, 2, 3), 2, 2), 2, 1)
        first_layer_index = torch.reshape(torch.nonzero(c_l21_norm.gt(torch.max(c_l21_norm) * self.prune_thr)), (-1,))
        num = min(int(first_layer_index.shape[0] * self.scale_coe), conv_layer.out_channels)
        # void zero
        num = max(num, 1)
        t = torch.mul(weight_copy, conv_layer.weight.grad)
        # t = torch.where(t > 0., torch.zeros_like(t).cuda(), t)
        # print(t)
        t1 = torch.mul(weight_copy_next, next_conv_layer.weight.grad)
        # t1 = torch.where(t1 > 0., torch.zeros_like(t1).cuda(), t1)
        # print(t1)
        t_sum = torch.sum(t, (3, 2, 1))
        if isinstance(next_conv_layer, nn.Conv2d):
            t1_sum = torch.sum(t1, (3, 2, 0))
        else:
            t1_sum = torch.sum(t1, 0)
        # mul_t_t1 = torch.mul(t_l2, t1_l2)
        add_t_t1 = torch.abs(torch.add(t_sum, t1_sum))
        # print(t_l2)
        # print(t1_l2)
        # print(mul_t_t1)
        # print(add_t_t1)
        _, index = torch.topk(add_t_t1, conv_layer.out_channels - num, largest=False)
        return index, t_sum, t1_sum
    def print_index(self):
        if 'vgg' in args.arch:
            for i, layer_index in enumerate(self.list1):
                if i < len(self.list1) - 1:
                    print('feature: {0}\n'.format(layer_index), self.get_update_index(self.model.module._modules['feature'][layer_index], self.model.module._modules['feature'][self.list1[i + 1]])[1])
                    print('feature: {0}\n'.format(self.list1[i + 1]), self.get_update_index(self.model.module._modules['feature'][layer_index], self.model.module._modules['feature'][self.list1[i + 1]])[2])
                else:
                    print('feature: {0}\n'.format(layer_index), self.get_update_index(self.model.module._modules['feature'][layer_index], self.model.module._modules['classifier1'])[1])
                    print('feature: classifier1\n',
                          self.get_update_index(self.model.module._modules['feature'][layer_index],
                                                self.model.module._modules['classifier1'])[2])
        else:
            for layer_index, layer_index_value in enumerate(self.list1):
                for j in range(self.model.module.layernum[layer_index]):
                    print('{}''{:}.conv1\n'.format(layer_index_value, j), self.get_update_index(self.model.module._modules[layer_index_value][j].conv1, self.model.module._modules[layer_index_value][j].conv2)[1])
                    print('{}''{:}.conv2\n'.format(layer_index_value, j),
                          self.get_update_index(self.model.module._modules[layer_index_value][j].conv1,
                                                self.model.module._modules[layer_index_value][j].conv2)[2])
                    if args.arch in ['resnet50', 'resnet101']:
                        print('{}''{:}.conv2\n'.format(layer_index_value, j),
                              self.get_update_index(self.model.module._modules[layer_index_value][j].conv2,
                                                    self.model.module._modules[layer_index_value][j].conv3)[1])
                        print('{}''{:}.conv3\n'.format(layer_index_value, j),
                              self.get_update_index(self.model.module._modules[layer_index_value][j].conv2,
                                                    self.model.module._modules[layer_index_value][j].conv3)[2])

    def L21_regularizer(self, conv_layer, index):
        weight_copy = conv_layer.weight.data[index].clone()
        if weight_copy.shape[0] > 0:
            c_l21_norm = torch.norm(torch.norm(torch.norm(weight_copy.data, 2, 3), 2, 2), 2, 1)
            conv_layer.weight.data[index] = torch.reshape(torch.reshape(weight_copy, (weight_copy.shape[0], -1)).t().mul(
            torch.where(c_l21_norm > args.lr * self.lambda_L21 * self.s_c_v, torch.div(c_l21_norm - args.lr * self.lambda_L21 * self.s_c_v, c_l21_norm),
                        torch.zeros(c_l21_norm.shape).cuda())).t(), weight_copy.shape)
        return conv_layer.weight.data

    def L21_next_regularizer(self, conv_layer, index):
        weight_copy = conv_layer.weight.data[:, index].clone()
        if weight_copy.shape[1] > 0:
            if isinstance(conv_layer, nn.Conv2d):
                c_l21_norm = torch.norm(torch.norm(torch.norm(weight_copy.data, 2, 3), 2, 2), 2, 0)
            else:
                c_l21_norm = torch.norm(weight_copy.data, 2, 0)
            conv_layer.weight.data[:, index] = torch.transpose(torch.reshape(torch.reshape(torch.transpose(weight_copy, 0, 1), (weight_copy.shape[1], -1)).t().mul(
            torch.where(c_l21_norm > args.lr * self.lambda_L21 * self.s_c_v, torch.div(c_l21_norm - args.lr * self.lambda_L21 * self.s_c_v, c_l21_norm),
                        torch.zeros(c_l21_norm.shape).cuda())).t(), torch.transpose(weight_copy, 0, 1).shape), 0, 1)
        return conv_layer.weight.data

    def model_weight_update(self):
        if 'vgg' in args.arch:
            for i, layer_index in enumerate(self.list1):
                if self.lambda_L21 > 1e-6:
                    if i < len(self.list1) - 1:
                        update_index = self.get_update_index(self.model.module._modules['feature'][layer_index], self.model.module._modules['feature'][self.list1[i + 1]])[0]
                        self.model.module._modules['feature'][layer_index].weight.data = self.L21_regularizer(self.model.module._modules['feature'][layer_index], update_index)
                        self.model.module._modules['feature'][self.list1[i + 1]].weight.data = self.L21_next_regularizer(
                            self.model.module._modules['feature'][self.list1[i + 1]], update_index)
                    else:
                        update_index = self.get_update_index(self.model.module._modules['feature'][layer_index], self.model.module._modules['classifier1'])[0]
                        self.model.module._modules['feature'][layer_index].weight.data = self.L21_regularizer(self.model.module._modules['feature'][layer_index], update_index)
                        self.model.module._modules['classifier1'].weight.data = self.L21_next_regularizer(
                            self.model.module._modules['classifier1'], update_index)

        else:
            for layer_index, layer_index_value in enumerate(self.list1):
                for j in range(self.model.module.layernum[layer_index]):
                    if self.lambda_L21 > 1e-6:
                        update_index = self.get_update_index(self.model.module._modules[layer_index_value][j].conv1, self.model.module._modules[layer_index_value][j].conv2)[0]
                        self.model.module._modules[layer_index_value][j].conv1.weight.data = self.L21_regularizer(self.model.module._modules[layer_index_value][j].conv1, update_index)
                        self.model.module._modules[layer_index_value][j].conv2.weight.data = self.L21_next_regularizer(
                            self.model.module._modules[layer_index_value][j].conv2, update_index)
                    if args.arch in ['resnet50', 'resnet101']:
                        if self.lambda_L21 > 1e-6:
                            update_index = self.get_update_index(self.model.module._modules[layer_index_value][j].conv2, self.model.module._modules[layer_index_value][j].conv3)[0]
                            self.model.module._modules[layer_index_value][j].conv2.weight.data = self.L21_regularizer(self.model.module._modules[layer_index_value][j].conv2, update_index)
                            self.model.module._modules[layer_index_value][j].conv3.weight.data = self.L21_next_regularizer(
                                self.model.module._modules[layer_index_value][j].conv3, update_index)

    def row_prune(self, layer_p, BN_layer_p, layer1_p, pretrained=False):
        c_l21_norm = torch.norm(torch.norm(torch.norm(layer_p.weight.data, 2, 3), 2, 2), 2, 1)
        if isinstance(layer1_p, nn.Conv2d):
            c_l21_norm += torch.norm(torch.norm(torch.norm(layer1_p.weight.data, 2, 3), 2, 2), 2, 0)
        else:
            c_l21_norm += torch.norm(layer1_p.weight.data, 2, 0)
        first_layer_index = torch.reshape(torch.nonzero(c_l21_norm.gt(torch.max(c_l21_norm) * self.prune_thr)), (-1,))
        self.layer_sparsity_ratio.append(1.-first_layer_index.shape[0]/c_l21_norm.shape[0])
        num = min(int(first_layer_index.shape[0] * self.scale_coe), layer_p.out_channels)
        num = max(num, 1)
        self.layer_sparsity_allocation_ratio.append(1. - num / layer_p.out_channels)
        _, first_layer_index = torch.topk(c_l21_norm, num, largest=True)
        # l21_norm = torch.norm(torch.norm(torch.norm(layer_p.weight.data, 2, 1), 2, 1), 2, 1)
        # tk = int(layer_p.weight.data.shape[0] * (1. - self.prune_ratio))
        # _, first_layer_index = torch.topk(l21_norm, tk, largest=True)
        first_layer_p = nn.Conv2d(in_channels=layer_p.in_channels, out_channels=first_layer_index.shape[0], kernel_size=layer_p.kernel_size, \
                               stride=layer_p.stride, padding=layer_p.padding, bias=layer_p.bias)
        first_layer_p.weight.data = layer_p.weight.data[first_layer_index]
        if isinstance(layer1_p, nn.Conv2d):
            last_layer_p = nn.Conv2d(in_channels=first_layer_index.shape[0], \
                                out_channels=layer1_p.out_channels, kernel_size=layer1_p.kernel_size, stride=layer1_p.stride, padding=layer1_p.padding, bias=layer1_p.bias)
            last_layer_p.weight.data = layer1_p.weight.data[:,first_layer_index]
        elif isinstance(layer1_p, nn.Linear):
            last_layer_p = nn.Linear(first_layer_index.shape[0], layer1_p.out_features)
            last_layer_p.weight.data = layer1_p.weight.data[:,first_layer_index]
            last_layer_p.bias.data = layer1_p.bias.data
        bn_layer_p = nn.BatchNorm2d(first_layer_index.shape[0])
        bn_layer_p.weight.data = BN_layer_p.weight.data[first_layer_index].clone()
        bn_layer_p.bias.data = BN_layer_p.bias.data[first_layer_index].clone()
        bn_layer_p.running_mean = BN_layer_p.running_mean[first_layer_index].clone()
        bn_layer_p.running_var = BN_layer_p.running_var[first_layer_index].clone()
        return first_layer_p, bn_layer_p, last_layer_p

    def channel_prune(self):
        if 'vgg' in args.arch:
            for i, layer_index in enumerate(self.list1):
                if i < len(self.list1) - 1:
                    self.model.module._modules['feature'][layer_index], self.model.module._modules['feature'][layer_index + 1], self.model.module._modules['feature'][self.list1[i + 1]] = \
                self.row_prune(self.model.module._modules['feature'][layer_index], self.model.module._modules['feature'][layer_index + 1], self.model.module._modules['feature'][self.list1[i + 1]])
                else:
                    self.model.module._modules['feature'][layer_index], self.model.module._modules['feature'][layer_index + 1], self.model.module._modules['classifier1'] = \
                    self.row_prune(self.model.module._modules['feature'][layer_index], self.model.module._modules['feature'][layer_index + 1], self.model.module._modules['classifier1'])
        else:
            for layer_index, layer_index_value in enumerate(self.list1):
                for j in range(self.model.module.layernum[layer_index]):
                    self.model.module._modules[layer_index_value][j].conv1, \
                    self.model.module._modules[layer_index_value][j].bn1, \
                    self.model.module._modules[layer_index_value][j].conv2 = self.row_prune(
                        self.model.module._modules[layer_index_value][j].conv1,
                        self.model.module._modules[layer_index_value][j].bn1,
                        self.model.module._modules[layer_index_value][j].conv2)
                    if args.arch in ['resnet50', 'resnet101']:
                        self.model.module._modules[layer_index_value][j].conv2, \
                        self.model.module._modules[layer_index_value][j].bn2, \
                        self.model.module._modules[layer_index_value][j].conv3 = self.row_prune(
                            self.model.module._modules[layer_index_value][j].conv2,
                            self.model.module._modules[layer_index_value][j].bn2,
                            self.model.module._modules[layer_index_value][j].conv3)

if __name__ == '__main__':
    main()
