import sys
import torch
import copy
import math
import random as rand
import time
import csv

class Quantizer(object):
    def __init__(self, roundMeth):
        self.roundMeth = roundMeth

    def log(self, line):
        with open('quant.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(line)

    def quantize_inputs(self, inputs, bitWidth, loc=None):
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

        if self.roundMeth == 'Simple': 
            scaled, sf = self.simpleRound(scaled, maxVal, minVal)
        elif self.roundMeth == 'Stochastic':
            scaled, sf = self.stochRound(scaled, maxVal, minVal)
        else:
            raise ValueError("Rounding method should be one of 'Simple' or 'Stochastic'")

        return scaled, sf

    def findSfAndScale(self, scaled, maxVal, minVal):
        rangeBest = min(abs(maxVal/torch.max(scaled)), abs(minVal/torch.min(scaled)))
        # floor returns int value closest to zero
        #  on the way up, dont want to over estimate
        #  on they way down, dont want to underestimate
        sf = math.floor(math.log2(rangeBest))
        scaled.mul_(pow(2, sf))
        
        return scaled, sf

    def stochRound(self, scaled, maxVal, minVal):
        scaled, sf = self.findSfAndScale(scaled, maxVal, minVal)

        # add values in rand -0.5 -> 0.5 to potentially tip rounding in a certain direction
        mod = torch.FloatTensor(scaled.size()).cuda(scaled.device)
        mod.uniform_(-0.5, 0.5)
        scaled.add_(mod)

        return scaled.round(), sf

    def simpleRound(self, scaled, maxVal, minVal):
        scaled, sf = self.findSfAndScale(scaled, maxVal, minVal)
        return scaled.round(), sf
