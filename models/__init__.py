#!/usr/bin/env python
# encoding: utf-8
"""
@author: taoxvzi
@contact: ruanxiaofenghit@163.com
@file: __init__.py
@time: 2019/5/20 9:47
@desc:
"""
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .resnet_cifar import resnet20, resnet32, resnet44, resnet56, resnet110
from .vgg import vggsmall, vggvariant
from .vgg_uniform import vgg_uniform
from .densenet import densenet
from .mobilenetv2 import mobilenetv2cifar, mobilenetv2cifar100, mobilenetv2
from .preactresnet import preactresnet18, preactresnet34, preactresnet50, preactresnet101, preactresnet152