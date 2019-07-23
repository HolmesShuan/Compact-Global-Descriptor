'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_best import *

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


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    # cfg = [45, (90,2), 90, (179,2), 179, (358,2), 358, 358, 358, 358, 358, (717,2), 717]

    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)
        # self.conv1 = nn.Conv2d(3, 22, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(22)
        # self.layers = self._make_layers(in_planes=22)
        # self.linear = nn.Linear(717, num_classes)
        self.relu = nn.ReLU(True)
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def mobilenet(pretrained=False, **kwargs):
    """Constructs a MobileNet model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNet()
    return model
