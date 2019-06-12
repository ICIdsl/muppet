import sys
import torch

class Scaler(object):
    def __init__(self, model, _quantizer, bitWidth):
        modules = model._modules
        self.layers = modules['module']._modules
        self.quantizer = _quantizer
        self.weightsSF = {}
        self.inputSF = 0
        self.bitWidth = bitWidth

    def register_hooks(self):
        for k,v in self.layers.items():
            if 'conv' in k or 'relu' in k or 'classifier' in k:
                v.register_backward_hook(self.backward_quantize_hook)


    def backward_quantize_hook(self, module, grad_input, grad_output):
        if (isinstance(grad_output, tuple)):
            for i in range(len(grad_input)):
                if grad_input[i] is not None:
                    grad_input[i].data, _ = self.quantizer.quantize_inputs(grad_input[i].data, self.bitWidth)
        else:
            grad_input.data, _ = self.quantizer.quantize_inputs(grad_output.data, self.bitWidth)
        return grad_input
