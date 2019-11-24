import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler
import math
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler
import math
import torch.nn.functional as F

class ChannelAttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionLayer, self).__init__()
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

class SpatialAttentionLayer(nn.Module):
    def __init__(self, spatial_size):
        super(SpatialAttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.softmax = nn.Softmax(dim=1)

        self.width_w0 = nn.Parameter(torch.ones(spatial_size,1), requires_grad=True)
        self.width_w1 = nn.Parameter(torch.ones(spatial_size,1), requires_grad=True)
        self.width_w2 = nn.Parameter(torch.ones(spatial_size,1), requires_grad=True)
        
        self.width_bias0 = nn.Parameter(torch.zeros(1,spatial_size), requires_grad=True)
        self.width_bias1 = nn.Parameter(torch.zeros(1,spatial_size), requires_grad=True)
        self.width_bias2 = nn.Parameter(torch.zeros(1,spatial_size), requires_grad=True)

        self.height_w0 = nn.Parameter(torch.ones(spatial_size,1), requires_grad=True)
        self.height_w1 = nn.Parameter(torch.ones(spatial_size,1), requires_grad=True)
        self.height_w2 = nn.Parameter(torch.ones(spatial_size,1), requires_grad=True)
        
        self.height_bias0 = nn.Parameter(torch.zeros(1,spatial_size), requires_grad=True)
        self.height_bias1 = nn.Parameter(torch.zeros(1,spatial_size), requires_grad=True)
        self.height_bias2 = nn.Parameter(torch.zeros(1,spatial_size), requires_grad=True)

        nn.init.xavier_uniform_(self.width_w0)
        nn.init.xavier_uniform_(self.width_w1)
        nn.init.xavier_uniform_(self.width_w2)
        nn.init.xavier_uniform_(self.height_w0)
        nn.init.xavier_uniform_(self.height_w1)
        nn.init.xavier_uniform_(self.height_w2)

    def forward(self, x):
        b, c, h, w = x.size()
        # print(x.size())
        x_spatial_max = torch.max(x,1)[0] # b, h, w
        x_spatial_mean = torch.mean(x,1) # b, h, w

        x_width_max = torch.max(x_spatial_max,1)[0] # b, w
        x_width_mean = torch.mean(x_spatial_mean,1) # b, w

        x_height_max = torch.max(x_spatial_max,2)[0] # b, h
        x_height_mean = torch.mean(x_spatial_mean,2) # b, h

        x0_w_s = self.softmax(x_width_mean) # b, w

        y0_w = torch.matmul(x_width_mean, self.width_w0) # b, 1
        y1_w = torch.matmul(x_width_max, self.width_w1) # b, 1

        y0_w_t = torch.tanh(y0_w*x0_w_s + self.width_bias0) # b, w
        y1_w_t = torch.tanh(y1_w*x0_w_s + self.width_bias1) # b ,w

        y2_w = torch.matmul(y1_w_t, self.width_w2) # b, 1
        y2_w_t = y2_w*y0_w_t + self.width_bias2
        # y2_w_t = torch.tanh(y2_w*y0_w_t + self.width_bias2) # b, w
 
        x0_h_s = self.softmax(x_height_mean) # b, w

        y0_h = torch.matmul(x_height_mean, self.height_w0) # b, 1
        y1_h = torch.matmul(x_height_max, self.height_w1) # b, 1

        y0_h_t = torch.tanh(y0_h*x0_h_s + self.height_bias0) # b, w
        y1_h_t = torch.tanh(y1_h*x0_h_s + self.height_bias1) # b ,w

        y2_h = torch.matmul(y1_h_t, self.height_w2) # b, 1
        y2_h_t = y2_h*y0_h_t + self.height_bias2
        # y2_h_t = torch.tanh(y2_h*y0_h_t + self.height_bias2) # b, w

        spatial = torch.tanh(torch.matmul(y2_h_t.view(b,h,1), y2_w_t.view(b,1,w))).unsqueeze(1)

        z = x*(spatial+1)
        return z

class AttentionLayer(nn.Module):
    def __init__(self, in_channels, spatial_size):
        super(AttentionLayer, self).__init__()
        self.spatial_att = SpatialAttentionLayer(spatial_size)
        self.channel_att = ChannelAttentionLayer(in_channels)

    def forward(self, x):
        x = self.channel_att(x)
        # x = self.spatial_att(x)
        return x
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
                # nn.Sigmoid()
                nn.Tanh()
            )
            self.squeeze1 = nn.Sequential(
                SqueezeLayer(in_channels, bias_term=self.bias_term),
                # nn.Sigmoid()
                nn.Tanh()
            )
            self.squeeze2 = nn.Sequential(
                SqueezeLayer(in_channels, bias_term=self.bias_term),
                # nn.Sigmoid()
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
                # nn.Sigmoid()
                nn.Tanh()
            )
            self.squeeze1 = nn.Sequential(
                SqueezeLayer(in_channels, bias_term=self.bias_term),
                # nn.Sigmoid()
                nn.Tanh()
            )
            self.squeeze2 = nn.Sequential(
                SqueezeLayer(in_channels, bias_term=self.bias_term),
                # nn.Sigmoid()
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
'''
