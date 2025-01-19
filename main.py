# -*- coding: UTF-8 -*-
"""
@Project ：BLS-torch
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：Chunyu Lei
@Date    ：2021/10/27 22:51 
"""
import time
import utils
import torch

import numpy as np
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import BLS, CFBLS

# set random seed
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

# 加载训练数据
train_data = datasets.MNIST('./data/', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
# 加载测试数据
test_data = datasets.MNIST('./data/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

# 是否使用GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device: {}'.format(device))

img_shape = (1, 28, 28)
num_class = 10

# 模型实例化
net = BLS(
    in_features=img_shape[0] * img_shape[1] * img_shape[2],
    out_features=num_class,
    N1=5,
    N2=5,
    N3=100
).type(torch.float64).to(device)

# 打印网络结构
print(net)

train_acc, test_acc = None, None

# 模型训练
net.train()
time_start = time.time()
for batch_i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)

    batch_size, _, img_h, img_w = images.shape
    images = images.view(batch_size, -1)
    # Z-Score normalization
    mu, sigma = torch.mean(images, dim=1, keepdim=True), torch.std(images, dim=1, keepdim=True)
    images = (images - mu) / sigma
    train_images = images.double()

    train_labels = F.one_hot(labels, num_class).double()

    out = net(train_images, train_labels)
    train_acc = utils.accuracy(out, train_labels)
time_end = time.time()
train_time = time_end - time_start

# 模型评估
net.eval()
time_start = time.time()
for batch_i, (images, labels) in enumerate(test_loader):
    images, labels = images.to(device), labels.to(device)

    batch_size, _, img_h, img_w = images.shape
    images = images.view(batch_size, -1)
    # Z-Score normalization
    mu, sigma = torch.mean(images, dim=1, keepdim=True), torch.std(images, dim=1, keepdim=True)
    images = (images - mu) / sigma
    test_images = images.double()

    test_labels = F.one_hot(labels, num_class).double()

    out = net(test_images, test_labels)
    test_acc = utils.accuracy(out, test_labels)
time_end = time.time()
test_time = time_end - time_start

print('--------------------------------------------------')
print('| train acc: {:> 10.3}% | train time:{:>10.3f}s |'.format(train_acc[0], train_time))
print('| test acc:  {:> 10.3}% | test time: {:>10.3f}s |'.format(test_acc[0], test_time))
print('--------------------------------------------------')


"""特征重要性分析"""
import sage
import torch.nn as nn
import matplotlib.pyplot as plt

# Add activation at output
model_activation = nn.Sequential(net, nn.Softmax(dim=1))

# Move test data to numpy
test_np = test_images.cpu().data.numpy()
Y_test_np = torch.argmax(test_labels, dim=1).cpu().data.numpy()

"""分组重要性 4*4(~2 mins)"""
# Feature groups
width = 4
num_superpixels = 28 // width
groups = []
for i in range(num_superpixels):
    for j in range(num_superpixels):
        img = np.zeros((28, 28), dtype=int)
        img[width*i:width*(i+1), width*j:width*(j+1)] = 1
        img = img.reshape((784,))
        groups.append(np.where(img)[0])

# Setup and calculate
imputer = sage.GroupedMarginalImputer(model_activation, test_np[:128], groups)
estimator = sage.PermutationEstimator(imputer, 'cross entropy')
sage_values = estimator(test_np, Y_test_np, batch_size=128, thresh=0.05)

# Plot
plt.figure(figsize=(6, 6))
m = np.max(np.abs(sage_values.values))
plt.imshow(- sage_values.values.reshape(7, 7),
           cmap='seismic', vmin=-m, vmax=m)
plt.xticks([])
plt.yticks([])
plt.savefig('vis_1.png')

"""分组重要性 2*2(~8 mins)"""
# Feature groups
width = 2
num_superpixels = 28 // width
groups = []
for i in range(num_superpixels):
    for j in range(num_superpixels):
        img = np.zeros((28, 28), dtype=int)
        img[width*i:width*(i+1), width*j:width*(j+1)] = 1
        img = img.reshape((784,))
        groups.append(np.where(img)[0])
# Setup and calculate
imputer = sage.GroupedMarginalImputer(model_activation, test_np[:128], groups)
estimator = sage.PermutationEstimator(imputer, 'cross entropy')
sage_values = estimator(test_np, Y_test_np, batch_size=128, thresh=0.05)

# Plot
plt.figure(figsize=(6, 6))
m = np.max(np.abs(sage_values.values))
plt.imshow(- sage_values.values.reshape(14, 14),
           cmap='seismic', vmin=-m, vmax=m)
plt.xticks([])
plt.yticks([])
plt.savefig('vis2.png')

"""像素级重要性 1*1"""
# Setup and calculate
imputer = sage.MarginalImputer(model_activation, test_np[:128])
estimator = sage.PermutationEstimator(imputer, 'cross entropy')
sage_values = estimator(test_np, Y_test_np, batch_size=128, thresh=0.05)

# Plot
plt.figure(figsize=(6, 6))
m = np.max(np.abs(sage_values.values))
plt.imshow(- sage_values.values.reshape(28, 28),
           cmap='seismic', vmin=-m, vmax=m)
plt.xticks([])
plt.yticks([])
plt.savefig('vis3.png')
