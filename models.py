# -*- coding: UTF-8 -*-
"""
@Project ：bls-torch 
@File    ：models.py
@IDE     ：PyCharm 
@Author  ：Chunyu Lei
@Date    ：2021/10/29 21:09 
"""
import torch
import torch.nn as nn

from layers.feature import Feature
from layers.enhancement import Enhancement
from layers.output import Output

"""
FashionMNIST:
seed = 1
N1:10 N2:10 N3:11000
train_acc:94.9%
test_acc: 89.5%
"""


class BLS(nn.Module):
    def __init__(self, in_features, N1, N2, N3, out_features):
        super(BLS, self).__init__()
        self.fl = nn.ModuleList()
        for i in range(N2):
            self.fl.add_module('Z{}'.format(i + 1), Feature(in_features, N1))

        self.el = Enhancement(N1 * N2, N3, s=0.8)
        self.tanh = nn.Tanh()

        self.ol = Output(N1 * N2 + N3, out_features, c=2 ** -30)

    def forward(self, *args):
        x, y = None, None
        if self.training:
            x, y = args
        else:
            x = args[0]

        feature_out = None
        for idx, m in enumerate(self.fl):
            if idx == 0:
                feature_out = m(x)
            else:
                feature_out = torch.hstack([feature_out, m(x)])

        enhancement_out = self.tanh(self.el(feature_out))
        input_of_output_layer = torch.hstack([feature_out, enhancement_out])

        if self.training:
            output = self.ol(input_of_output_layer, y)
        else:
            output = self.ol(input_of_output_layer)

        return output


class CascadeFeatureLayer(nn.Module):
    def __init__(self, in_features, hidden_size, num_layers):
        super(CascadeFeatureLayer, self).__init__()
        self.modulelist = nn.ModuleList()
        for i in range(num_layers):
            self.modulelist.add_module('Z{}'.format(i + 1), Feature(
                in_features=in_features if i == 0 else hidden_size,
                out_features=hidden_size
            ))

    def forward(self, x):
        ret = None
        for idx, m in enumerate(self.modulelist):
            x = m(x)
            if idx == 0:
                ret = x
            else:
                ret = torch.hstack([ret, x])
        return ret

"""
FashionMNIST:
seed = 1
N1:10 N2:4 N3:7800
train_acc: 92.5%
test_acc: 88.6%
"""
class CFBLS(nn.Module):
    def __init__(self, in_features, out_features, N1, N2, N3):
        super(CFBLS, self).__init__()

        self.fl = nn.ModuleList()
        for i in range(N2):
            self.fl.add_module('CF{}'.format(i + 1), CascadeFeatureLayer(
                in_features=in_features,
                hidden_size=N1,
                num_layers=2
            ))

        self.el = Enhancement(N1 * N2 * 2, N3, s=0.8)

        self.tanh = nn.Tanh()

        self.ol = Output(N1 * N2 * 2 + N3, out_features, c=2 ** -30)

    def forward(self, *args):
        x, y = None, None
        if self.training:
            x, y = args
        else:
            x = args[0]

        feature_out = None
        for idx, m in enumerate(self.fl):
            res = m(x)
            if idx == 0:
                feature_out = res
            else:
                feature_out = torch.hstack([feature_out, res])

        enhancement_out = self.tanh(self.el(feature_out))

        input_of_output_layer = torch.hstack([feature_out, enhancement_out])

        if self.training:
            output = self.ol(input_of_output_layer, y)
        else:
            output = self.ol(input_of_output_layer)
        return output

"""
FashionMNIST
seed = 1
train acc: 93.7%
test acc: 90.0%
"""
class CCFBLS(nn.Module):
    def __init__(self):
        super(CCFBLS, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3), (1, 1), 1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.el = Enhancement(1568, 1000)
        self.tanh = nn.Tanh()
        self.ol = Output(4704 + 1000, 10, c=2 ** -30)

    def forward(self, *args):
        if self.training:
            x, y = args
        else:
            x = args[0]
        z1 = self.conv1(x)
        z2 = self.conv2(z1)

        z1, z2 = z1.view(z1.shape[0], -1), z2.view(z2.shape[0], -1)

        eout = self.tanh(self.el(z2))

        oin = torch.hstack([torch.hstack([z1, z2]), eout])
        if self.training:
            output = self.ol(oin, y)
        else:
            output = self.ol(oin)
        return output
