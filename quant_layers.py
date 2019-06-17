import torch.nn as nn
import torch
import sys
import src.muppet.quantize as quantize


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', _bitWidth=8, _SFHolder=None):

        self.bitWidth = _bitWidth
        self.sfHolder = _SFHolder
        self.quantizer = quantize.Quantizer()
        self.prevLayer = None
        self.weightSF = 0

        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        result = super().forward(input)
        return forward(self, result)

class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, _bitWidth=8, _SFHolder=None):

        self.bitWidth = _bitWidth
        self.sfHolder = _SFHolder
        self.quantizer = quantize.Quantizer()
        self.prevLayer = None
        self.weightSF = 0

        super(QuantLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        result = super().forward(input)
        return forward(self, result)

def forward(self, result):
    if self.bitWidth != -1:
        result.data, sf = self.quantizer.quantize_inputs(result.data, self.bitWidth)
    return result

class SFHolder(object):
    def __init__(self):
        self.sf = {'': 0}

