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
            return self.round(scaled, bitWidth).float(), 0


        val = 1 << (bitWidth-1)
        maxVal = (val - 1)
        minVal = (-val)

        sf = 0
        
        complete = 0
        # check if values are outside representable range (minVal -> maxVal)
        if (torch.max(scaled).item() == float("inf") or torch.min(scaled).item() == float("-inf")):
            raise ValueError
        if (self.round(torch.max(scaled)) > maxVal) or (self.round(torch.min(scaled)) < minVal):
            test = scaled.clone()
            rangeBest = min(maxVal/torch.max(test), minVal/torch.min(test))
            testsf = int(math.log2(rangeBest))
            test.mul_(pow(2, testsf))

            while not complete:
                # as soon as the values of scaled are no longer outside the representable range, finish
                if not ((self.round(torch.max(scaled)) > maxVal) or (self.round(torch.min(scaled)) < minVal)):
                    maximum = torch.max(scaled).int()
                    minimum = torch.min(scaled).int()
                    complete = 1
                # as long as the values of scaled are outside the representable range, halve the value of each element of scaled
                else:
                    scaled.mul_(0.5)
                    sf -= 1
            print('--max:', torch.max(scaled), torch.max(test))
            print('min:', torch.min(scaled), torch.min(test))
            print('sf:', sf, testsf)
            sys.exit()

        else:
            rangeBest = min(abs(maxVal/torch.max(scaled)), abs(minVal/torch.min(scaled)))

            maxVal = maxVal/2
            minVal = minVal/2
                
            if rangeBest != 0:
                sf = int(math.log2(rangeBest))
                scaled.mul_(pow(2, sf))
                if (self.round(torch.max(scaled).sub_(0.5), maxVal) <= maxVal) and (self.round(torch.min(scaled), minVal) >= minVal):
                    sf += 1
                    scaled.mul_(2)
            else:
                sf = 0

            maximum = torch.max(scaled).int()
            minimum = torch.min(scaled).int()

        return self.round(scaled, val, maximum, minimum).float(), sf

    def round(self, val, bitVal=math.inf, maximum=0, minimum=0):
        if (maximum == (bitVal-1) or minimum == (-bitVal)):
            return val.int()
        else:
            return val.round().int()
