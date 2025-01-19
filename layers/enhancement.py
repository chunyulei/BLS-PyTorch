# -*- coding: UTF-8 -*-
"""
@Project ：BLS-torch
@File    ：enhancement.py
@IDE     ：PyCharm 
@Author  ：Chunyu Lei
@Date    ：2021/10/28 21:48 
"""
import torch

from torch import Tensor
from torch.nn import init
from torch.nn import Module
from torch.nn.parameter import Parameter


# 增强层
class Enhancement(Module):
    """
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: shrink coefficient
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, s: float = 0.8, bias: bool = True) -> None:
        super(Enhancement, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.bias = bias

        if self.bias:
            self.weight = Parameter(torch.Tensor(self.in_features + 1, self.out_features), requires_grad=False)
        else:
            raise Exception('to be implemented')

        self.parameterOfShrink = Parameter(torch.Tensor(1), requires_grad=False)
        self.reset_parameters()

    @staticmethod
    def orth(weight: Tensor) -> Tensor:
        r = torch.linalg.matrix_rank(weight)
        u, s, v = torch.svd(weight)
        return u[:, :r]

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        in_features, out_features = self.weight.shape[0] - 1, self.weight.shape[1]
        if in_features >= out_features:
            self.weight.data = self.orth(self.weight)
        else:
            self.weight.data = self.orth(self.weight.T).T

    def forward(self, input: Tensor) -> Tensor:
        """
        前馈计算增强层输出
        :param input: 输入张量
        :return: 输出张量
        """
        input_with_bias = torch.hstack(
            [input, 0.1 * torch.ones(input.shape[0], 1, dtype=torch.float64, device=input.device)])

        if self.bias:
            out = input_with_bias @ self.weight
        else:
            raise Exception('to be implemented')
        # 训练时计算缩减系数，测试时采用训练时计算出的缩减系数
        if self.training:
            self.parameterOfShrink.data = self.s / torch.max(out).unsqueeze(dim=0)
        else:
            pass
        out = out * self.parameterOfShrink
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, s={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.s
        )
