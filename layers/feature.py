# -*- coding: UTF-8 -*-
"""
@Project ：BLS-torch
@File    ：feature.py
@IDE     ：PyCharm 
@Author  ：Chunyu Lei
@Date    ：2021/10/28 21:47 
"""
import torch

from torch import Tensor
from torch.nn import Module
from torch.nn import init
from torch.nn.parameter import Parameter


# 特征层
class Feature(Module):
    """
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Feature, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        if bias:
            self.weight = Parameter(torch.Tensor(in_features + 1, out_features), requires_grad=False)
        else:
            raise Exception('to be implemented')

        self.max_ = Parameter(torch.Tensor(1, 10), requires_grad=False)  # normalization parameters
        self.min_ = Parameter(torch.Tensor(1, 10), requires_grad=False)  # normalization parameters

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        """
        前馈计算特征层输出（训练时，先使用 SAE优化权重；测试时，直接前馈计算特征层输出）
        :param input: 输入张量
        :return: 输出张量
        """
        input_with_bias = torch.hstack(
            [input, 0.1 * torch.ones(input.shape[0], 1, dtype=torch.float64, device=input.device)])

        # training stage
        if self.training:
            # 使用SAE优化特征层权重
            if self.bias:
                z = input_with_bias @ self.weight
            else:
                raise Exception('error: undefined method')

            z_min, z_max = torch.min(z, dim=0, keepdim=True).values, torch.max(z, dim=0, keepdim=True).values
            z_scaled = (z - z_min) / (z_max - z_min)

            weight = self.sparse_bls(z_scaled, input_with_bias).T

            self.weight.data = weight  # update feature layer weight

        # compute sparse feature
        if self.bias:
            out = input_with_bias @ self.weight
        else:
            raise Exception('error: undefined method')

        # 训练时直接归一化，测试时采用训练时计算出的最大最小值进行"归一化"
        if self.training:
            self.max_.data = torch.max(out, dim=0, keepdim=True).values
            self.min_.data = torch.min(out, dim=0, keepdim=True).values
        out = (out - self.min_) / (self.max_ - self.min_)
        return out

    @staticmethod
    def sparse_bls(a, b):
        lam = 0.001
        iterations = 50
        aa = a.T @ a
        m = a.shape[1]
        n = b.shape[1]
        x1 = torch.zeros([m, n], dtype=torch.float64, device=a.device)
        wk = x1
        ok = x1
        uk = x1
        l1 = (aa + torch.eye(m, dtype=torch.float64, device=a.device)).inverse()
        l2 = (l1 @ a.T) @ b
        for i in range(iterations):
            ck = l2 + (l1 @ (ok - uk))
            ok = torch.maximum(ck + uk - lam,
                               torch.zeros_like(ck + uk - lam, dtype=torch.float64, device=a.device)) - torch.maximum(
                -ck - uk - lam, torch.zeros_like(-ck - uk - lam, dtype=torch.float64, device=a.device))
            uk = uk + ck - ok
            wk = ok
        return wk

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
