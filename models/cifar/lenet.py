'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
import sys
import time
import math 
from src.muppet.quant_layers import QuantConv2d, QuantLinear

__all__ = ['lenet']

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = QuantConv2d(3, 6, kernel_size=5)
        self.conv2 = QuantConv2d(6, 16, kernel_size=5)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.classifier1 = QuantLinear(16*5*5, 120)
        self.classifier2 = QuantLinear(120, 84)
        self.classifier3 = QuantLinear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool2d(x) 
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = x.view(x.size(0), -1)
        x = self.classifier1(x)
        x = self.relu(x)
        x = self.classifier2(x)
        x = self.relu(x)
        x = self.classifier3(x)        
        return x

def lenet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = LeNet(**kwargs)
    return model
