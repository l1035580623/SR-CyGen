import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# conv for DFF
class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out


# deConv for DFF
class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


# 基于反投影的特征融合块
class DFFBlock(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DFFBlock, self).__init__()
        self.num_ft = num_ft - 1
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.up_convs.append(
                DeconvBlock(num_filter//(2**i), num_filter//(2**(i+1)), kernel_size, stride, padding, bias, activation)
            )
            self.down_convs.append(
                ConvBlock(num_filter//(2**(i+1)), num_filter//(2**i), kernel_size, stride, padding, bias, activation)
            )

    def forward(self, ft_l, ft_h_list):
        ft_fusion = ft_l
        for i in range(len(ft_h_list)):
            ft = ft_fusion
            for j in range(self.num_ft - i):
                ft = self.up_convs[j](ft)
            ft = ft - ft_h_list[i]
            for j in range(self.num_ft - i):
                ft = self.down_convs[self.num_ft - i - j - 1](ft)
            ft_fusion = ft_fusion + ft

        return ft_fusion


# 基于反投影的上采样模块
class UBPBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, bias=True, activation='prelu', norm=None):
        super(UBPBlock, self).__init__()

        self.up_conv1 = DeconvBlock(num_filter, num_filter // 2, kernel_size, stride, padding, bias, activation)
        self.up_conv2 = DeconvBlock(num_filter, num_filter // 2, kernel_size, stride, padding, bias, activation)
        self.down_conv = ConvBlock(num_filter // 2, num_filter, kernel_size, stride, padding, bias, activation)

        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, 1, 1, 0, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter // 2, num_filter // 2, 1, 1, 0, bias=bias)

    def forward(self, x):
        fm1 = self.up_conv1(x)
        fm2 = self.down_conv(fm1) - self.conv1(x)
        fm3 = self.up_conv2(fm2) + self.conv2(fm1)
        return fm3


# 基于反投影的下采样模块
class DBPBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, bias=True, activation='prelu', norm=None):
        super(DBPBlock, self).__init__()

        self.down_conv1 = ConvBlock(num_filter, num_filter * 2, kernel_size, stride, padding, bias, activation)
        self.down_conv2 = ConvBlock(num_filter, num_filter * 2, kernel_size, stride, padding, bias, activation)
        self.up_conv = DeconvBlock(num_filter * 2, num_filter, kernel_size, stride, padding, bias, activation)

        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, 1, 1, 0, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter * 2, num_filter * 2, 1, 1, 0, bias=bias)

    def forward(self, x):
        fm1 = self.down_conv1(x)
        fm2 = self.up_conv(fm1) - self.conv1(x)
        fm3 = self.down_conv2(fm2) + self.conv2(fm1)
        return fm3