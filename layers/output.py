# -*- coding: UTF-8 -*-
"""
@Project ：BLS-torch
@File    ：output.py
@IDE     ：PyCharm 
@Author  ：Chunyu Lei
@Date    ：2021/10/28 21:48 
"""
import torch
from torch import Tensor
from torch.nn import init
from torch.nn import Module
from torch.nn.parameter import Parameter


# 输出层
class Output(Module):
    """
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        c: regularization coefficient
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, c: float) -> None:
        super(Output, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c

        self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.weight)

    def forward(self, *args):
        """
        前馈计算输出层
        :param args: tensor tuple (x, y) training / x testing
        :return: tensor
        """
        if self.training:
            x, y = args
            weight = self.pinv(x, self.c) @ y  # 岭回归计算输出层权重
            self.weight.data = weight
        else:
            x = args[0]

        return x @ self.weight

    @staticmethod
    def pinv(A, reg):
        return (reg * torch.eye(A.shape[1], device=A.device) + A.T @ A).inverse() @ A.T

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )
