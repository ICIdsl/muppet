import sys
import torch

class Scaler(object):
    # def __init__(self, model, _quantizer, bitWidth):
    def __init__(self, model, _quantizer, params):
        modules = model._modules
        self.layers = modules['module']._modules
        self.quantizer = _quantizer
        self.weightsSF = {}
        self.inputSF = 0
        # self.bitWidth = bitWidth
        self.params = params

        self.update_model_precision(model)
    
    def update_model_precision(self, model):
        layers = model._modules['module']._modules
        for k,v in layers.items():
            if 'conv' in k or 'classifier' in k:
                v.bitWidth = self.params.bitWidth

    def register_hooks(self):
        for k,v in self.layers.items():
            if 'conv' in k or 'relu' in k or 'classifier' in k:
                v.register_backward_hook(self.backward_quantize_hook)

    def backward_quantize_hook(self, module, grad_input, grad_output):
        if self.params.dataType != 'Float':
            if (isinstance(grad_output, tuple)):
                for i in range(len(grad_input)):
                    if grad_input[i] is not None:
                        grad_input[i].data, _ = self.quantizer.quantize_inputs(grad_input[i].data, self.params.bitWidth)
            else:
                grad_input.data, _ = self.quantizer.quantize_inputs(grad_output.data, self.params.bitWidth)
        
        return grad_input

