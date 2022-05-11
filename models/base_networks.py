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


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x