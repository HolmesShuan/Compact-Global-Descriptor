import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler
import math
import torch.nn.functional as F
'''
class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, nonlinear=True):
        super(AttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.softmax = nn.Softmax(dim=1)

        self.w0 = nn.Parameter(torch.ones(in_channels,1), requires_grad=True)
        self.w1 = nn.Parameter(torch.ones(in_channels,1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(in_channels,1), requires_grad=True)
        
        self.bias0 = nn.Parameter(torch.zeros(1,in_channels,1,1), requires_grad=True)
        self.bias1 = nn.Parameter(torch.zeros(1,in_channels,1,1), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(1,in_channels,1,1), requires_grad=True)

        nn.init.xavier_uniform_(self.w0)
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

        # self.tanh = nn.Tanh()

    def forward(self, x):
        b, c, _, _ = x.size()
        x0 = self.avg_pool(x).view(b, c, 1, 1)
        x1 = self.max_pool(x).view(b, c, 1, 1)
        
        x0_s = self.softmax(x0) # b ,c ,1 ,1 

        y0 = torch.matmul(x0.view(b,c), self.w0).view(b, 1, 1, 1)
        y1 = torch.matmul(x1.view(b,c), self.w1).view(b, 1, 1, 1)

        y0_s = torch.tanh(y0*x0_s + self.bias0) # b ,c ,1 ,1 
        y1_s = torch.tanh(y1*x0_s + self.bias1) # b ,c ,1 ,1 

        y2 = torch.matmul(y1_s.view(b,c), self.w2).view(b, 1, 1, 1)
        y2_s = torch.tanh(y2*y0_s + self.bias2).view(b, c, 1, 1)

        z = x*(y2_s+1)
        return z
'''
class SqueezeLayer(nn.Module):
    def __init__(self, in_channels, bias_term):
        super(SqueezeLayer, self).__init__()
        self.weight = nn.Parameter(torch.ones(in_channels,1), requires_grad=True)
        self.bias_term = bias_term
        if self.bias_term == True:
            self.bias = nn.Parameter(torch.zeros(in_channels,1,1), requires_grad=True)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.matmul(x.view(b,c,c), self.weight).view(b, c, 1, 1)
        if self.bias_term == True:
            return y + self.bias
        else:
            return y

class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, nonlinear=True):
        super(AttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.softmax = nn.Softmax(dim=1)

        self.bias_term = bias
        self.nonlinear = nonlinear

        if self.nonlinear == True:
            self.squeeze0 = nn.Sequential(
                SqueezeLayer(in_channels, bias_term=self.bias_term),
                nn.Tanh()
            )
            self.squeeze1 = nn.Sequential(
                SqueezeLayer(in_channels, bias_term=self.bias_term),
                nn.Tanh()
            )
            self.squeeze2 = nn.Sequential(
                SqueezeLayer(in_channels, bias_term=self.bias_term),
                nn.Tanh()
            )
        else:
            self.squeeze0 = nn.Sequential(
                SqueezeLayer(in_channels, bias_term=self.bias_term)
            )
            self.squeeze1 = nn.Sequential(
                SqueezeLayer(in_channels, bias_term=self.bias_term)
            )
            self.squeeze2 = nn.Sequential(
                SqueezeLayer(in_channels, bias_term=self.bias_term),
            )

    def forward(self, x):
        b, c, _, _ = x.size()
        x0 = self.avg_pool(x).view(b, c, 1, 1)
        x1 = self.max_pool(x).view(b, c, 1, 1)
        x0_s = self.softmax(x0)
        x1_s = self.softmax(x1)
        y0 = torch.matmul(x0_s.view(b,c,1), x0.view(b,1,c)).view(b, c, c, 1)
        y1 = torch.matmul(x0_s.view(b,c,1), x1.view(b,1,c)).view(b, c, c, 1)
        att0 = self.squeeze0(y0)
        att1 = self.squeeze1(y1)
        y2 = torch.matmul(att0.view(b,c,1), att1.view(b,1,c)).view(b, c, c, 1)
        att2 = self.squeeze2(y2)
        z = x*(att2+1)
        return z
