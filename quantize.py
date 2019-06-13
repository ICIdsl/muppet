import sys
import torch
import copy
import math

class Quantizer(object):
    def quantize_inputs(self, inputs, bitWidth):
        if isinstance(inputs, torch.Tensor): 
            tmp = inputs.clone()
        else: 
            raise TypeError

        scaleMat, scaleFac = self.scale(tmp, bitWidth)
        scaleMat.mul_(pow(2, -scaleFac))

        return scaleMat, scaleFac
         
    def scale(self, scaled, bitWidth):

        if ((torch.max(scaled) == 0) and (torch.min(scaled) == 0)):
            return scaled.round(), 0

        val = 1 << (bitWidth-1)
        maxVal = (val - 1) + 0.5
        minVal = (-val) - 0.5

        # check if values are outside representable range (minVal -> maxVal)
        if (torch.max(scaled).item() == float("inf") or torch.min(scaled).item() == float("-inf") or 0):
            raise ValueError

        rangeBest = min(abs(maxVal/torch.max(scaled)), abs(minVal/torch.min(scaled)))
        sf = math.floor(math.log2(rangeBest))
        scaled.mul_(pow(2, sf))

        return scaled.round(), sf
