import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, t, stride_dw=1):
        super(Bottleneck, self).__init__()
        self.stride_dw=stride_dw
        self.t=t
        self.in_channels=in_channels
        self.out_channels=out_channels
        if self.t==1:
            self.dw1 = nn.Sequential(OrderedDict([
                ('1',nn.Conv2d(self.in_channels * self.t, self.in_channels * self.t, 3, stride=1, padding=1, groups=self.in_channels * self.t)),
                ('bn', nn.BatchNorm2d(self.in_channels* self.t, )),
                ('relu', nn.LeakyReLU(inplace=True))

            ]))
            self.conv2 = nn.Sequential(OrderedDict([
                ('1', nn.Conv2d(self.in_channels* self.t, self.out_channels, 1, stride=1, padding=0)),
                ('bn', nn.BatchNorm2d(self.out_channels))]))
        else:
            self.conv1 = nn.Sequential(OrderedDict([
            ('1', nn.Conv2d(self.in_channels, self.in_channels *self.t, 1, stride=1, padding=0)),
            ('bn', nn.BatchNorm2d(self.in_channels * self.t, )),
            ('relu', nn.LeakyReLU(inplace=True))]))
            self.dw1 = nn.Sequential(OrderedDict([
            ('1', nn.Conv2d(self.in_channels * self.t, self.in_channels * self.t, 3, stride=self.stride_dw, padding=1, groups=self.in_channels * self.t)),
            ('bn', nn.BatchNorm2d(self.in_channels * self.t, )),
            ('relu', nn.LeakyReLU(inplace=True))

            ]))
            self.conv2 = nn.Sequential(OrderedDict([
            ('1', nn.Conv2d(self.in_channels * self.t, self.out_channels, 1, stride=1, padding=0)),
            ('bn', nn.BatchNorm2d(self.out_channels))]))

    def forward(self, input):
        if self.t ==1:
            out = self.dw1(input)
            out = self.conv2(out)
        else:
            out = self.conv1(input)
            out = self.dw1(out)
            out = self.conv2(out)
        if self.out_channels==self.in_channels and self.stride_dw==1:
            out = out + input
        return out


class MobileNetV2(nn.Module):
    def __init__(self,inp_chanel=3,inp_BN=32):
        super(MobileNetV2, self).__init__()
        self.layers=[]
        self.last_channel=1280
        self.conv1 = nn.Sequential(OrderedDict([
            ('1', nn.Conv2d(inp_chanel, inp_BN, 3, stride=1, padding=1)),
            ('bn', nn.BatchNorm2d(32)),
            ('relu', nn.LeakyReLU(inplace=True))]))
        cfg_BottleNeck = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 1],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        for t,oup,n,s in cfg_BottleNeck:
            for times in range(n):
                if times==0:
                    self.layers.append(Bottleneck(inp_BN,oup,t,stride_dw=s))
                else :
                    self.layers.append(Bottleneck(inp_BN, oup, t, stride_dw=1))
                inp_BN=oup

        self.conv2 = nn.Sequential(OrderedDict([
                ('1',nn.Conv2d(320, 1280, 1, stride=1, padding=0)),
                ('bn', nn.BatchNorm2d(1280)),
                ('relu', nn.LeakyReLU(inplace=True))]))
        self.poo1 = nn.AvgPool2d(4)
        self.poo2 = nn.AvgPool2d(2)
        self.conv3 = nn.Sequential(OrderedDict([
                ('1',nn.Conv2d(1280, 10, 1, stride=1, padding=0)),
                ('bn', nn.BatchNorm2d(10)),
                ('relu', nn.LeakyReLU(inplace=True))]))
        self.classfier = nn.Linear(490, 2)
        self.layers=nn.Sequential(*self.layers)

        # self._initialize_weights()
    def forward(self, input):
        x= self.conv1(input)
        x=self.poo2(x)
        x=self.layers(x)
        x=self.conv2(x)
        x=self.poo1(x)
        x=self.conv3(x)
        x = x.view(x.size(0), -1)

        x=self.classfier(x)
        return x
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             n = m.weight.size(1)
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()