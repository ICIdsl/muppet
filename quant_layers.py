import torch.nn as nn
import torch
import sys
import src.muppet.quantize as quantize


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', _bitWidth=8, _SFHolder=None):

        self.bitWidth = _bitWidth
        self.sfHolder = _SFHolder
        self.prevLayer = None
        self.weightSF = 0

        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def setup_quantizer(self, quantizer):
        self.quantizer = quantizer

    def forward(self, input):
        result = super().forward(input)
        return forward(self, result)

class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, _bitWidth=8, _SFHolder=None):

        self.bitWidth = _bitWidth
        self.sfHolder = _SFHolder
        self.prevLayer = None
        self.weightSF = 0

        super(QuantLinear, self).__init__(in_features, out_features, bias)
    
    def setup_quantizer(self, quantizer):
        self.quantizer = quantizer

    def forward(self, input):
        result = super().forward(input)
        return forward(self, result)

class QuantAvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, _bitWidth=8, _SFHolder=None):

        self.bitWidth = _bitWidth
        self.sfHolder = _SFHolder
        self.prevLayer = None
        self.weightSF = 0

        super(QuantAvgPool2d, self).__init__(kernel_size, stride, padding, ceil_mode, count_include_pad)
    
    def setup_quantizer(self, quantizer):
        self.quantizer = quantizer

    def forward(self, input):
        result = super().forward(input)
        return forward(self, result)

class QuantAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size, _bitWidth=8, _SFHolder=None):

        self.bitWidth = _bitWidth
        self.sfHolder = _SFHolder
        self.prevLayer = None
        self.weightSF = 0

        super(QuantAdaptiveAvgPool2d, self).__init__(output_size)
    
    def setup_quantizer(self, quantizer):
        self.quantizer = quantizer

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

