'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_best import *

import logging

from mmcv.cnn import (VGG, xavier_init, constant_init, kaiming_init,
                      normal_init)
from mmcv.runner import load_checkpoint
from ..registry import BACKBONES

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(True)
        self.attention = AttentionLayer(in_planes, in_planes, True, True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.attention(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

@BACKBONES.register_module
class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    # our [0,      1,     2,      3,   4,        5,   6,   7,  8,   9,   10,   11,      12]
    # real[1,      2,     3,      4,   5,        6,   7,   8,  9,   10,  11,   12,      13]
    def __init__(self, out_indices=(10, 12, 14, 16, 18, 20),
                 frozen_stages=-1,
                 use_gn=False,
                 bn_eval=True,
                 bn_frozen=False):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, 1000)
        self.relu = nn.ReLU(True)
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_eval = bn_eval
        self.use_gn = use_gn
        self.bn_frozen = bn_frozen
        self.out_indices = out_indices

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        layers.append(BasicConv(1024, 256, kernel_size=1))
        layers.append(BasicConv(256, 512, kernel_size=3, stride=2, padding=1))
        layers.append(BasicConv(512, 128, kernel_size=1))
        layers.append(BasicConv(128, 256, kernel_size=3, stride=2, padding=1))
        layers.append(BasicConv(256, 128, kernel_size=1))
        layers.append(BasicConv(128, 256, kernel_size=3))
        layers.append(BasicConv(256, 128, kernel_size=1))
        layers.append(BasicConv(128, 256, kernel_size=3))
        return nn.Sequential(*layers)

    def forward(self, x):
        outs = []
        out = self.relu(self.bn1(self.conv1(x)))
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if i in self.out_indices:
                outs.append(out)
        return tuple(outs)

    def train(self, mode=True):
        super(MobileNet, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        '''
        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            for param in self.layers.parameters():
                param.requires_grad = False
            self.bn1.eval()
            self.bn1.weights.requires_grad = False
            self.bn1.bias.requires_grad = False
        '''
    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1)        

        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')
'''
def mobilenet(pretrained=False, **kwargs):
    """Constructs a MobileNet model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNet()
    return model
'''