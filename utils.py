# -*- coding: UTF-8 -*-
"""
@Project ：BLS-torch
@File    ：utils.py
@IDE     ：PyCharm 
@Author  ：Chunyu Lei
@Date    ：2021/10/28 19:37 
"""
import os
import shutil
import psutil

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable


# 用于计算平均值
# 比如每个batch里面计算精度的话要算一下均值，迭代过程同样需要维护一个总的均值
class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# 用于计算top-k accuracy
# 比如常用的top-1和top-5
def accuracy(output, target, topk=(1,)):
    """
    计算分类准确率
    :param output: 模型输出
    :param target: 标签（非one-hot形式）
    :param topk: top-k
    :return: accuracy
    """
    target = torch.argmax(target, dim=1)

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
