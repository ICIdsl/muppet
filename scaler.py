import sys
import torch

import src.muppet.quant_layers as ql

class UniversalIterator(object):
    def __init__(self, layers):
        self.layersIterator = layers.items().__iter__()
        self.quantLayers = []
        self.index = 0 

    def __iter__(self):
        return self 
    
    def __next__(self):
        if self.index == len(self.quantLayers):
            k,v = next(self.layersIterator)
            while('Quant' not in str(v)):
                k,v = next(self.layersIterator)
            
            topEntity = v 
            self.quantLayers = []
            self.index = 0
            self.parse_description(topEntity)
            quantLayer = self.quantLayers[self.index]
            self.index += 1
        else:
            quantLayer = self.quantLayers[self.index]
            self.index += 1
        
        return quantLayer 
    
    def parse_description(self, v):
        name = v.__class__.__name__
        if name == 'QuantConv2d':
            self.quantLayers.append(v)
        elif name == 'QuantLinear':
            self.quantLayers.append(v)
        elif name == 'QuantAdaptiveAvgPool2d':
            self.quantLayers.append(v)
        elif name == 'QuantAvgPool2d':
            self.quantLayers.append(v)
        else :
            for k,v in v._modules.items():
                if 'Quant' in str(v):
                    self.parse_description(v)

class Scaler(object):
    # def __init__(self, model, _quantizer, bitWidth):
    def __init__(self, model, _quantizer, params):
    #{{{
        modules = model._modules
        self.layers = modules['module']._modules
        self.quantizer = _quantizer
        self.weightsSF = {}
        self.inputSF = 0
        self.params = params

        # self.quantIterator = eval(params.arch.capitalize() + 'Iterator')
        self.quantIterator = UniversalIterator
        
        # setup quantiser
        prevLayer = ''
        # for v in self.quantIterator(self.layers):
        #     v.setup_quantizer(self.quantizer)
        #     v.prevLayer = prevLayer
        #     v.sfHolder = None
        #     prevLayer = str(v)

        for k,v in model.named_modules():
            classes = tuple([ql.__dict__[x] for x in ql.__dict__['__all__']])
            if isinstance(v, classes):
                v.setup_quantizer(self.quantizer)
                v.prevLayer = prevLayer
                v.sfHolder = None
                prevLayer = str(v)
        
        self.update_model_precision(model)
    #}}}
    
    def update_model_precision(self, model):
        for k,v in model.named_modules():
            classes = tuple([ql.__dict__[x] for x in ql.__dict__['__all__']])
            if isinstance(v, classes):
                v.bitWidth = self.params.bitWidth

    def register_hooks(self):
        for v in self.quantIterator(self.layers):
            v.register_backward_hook(self.backward_quantize_hook)

    def backward_quantize_hook(self, module, grad_input, grad_output):
        if self.params.dataType != 'Float':
            if (isinstance(grad_output, tuple)):
                for i in range(len(grad_input)):
                    if grad_input[i] is not None:
                        grad_input[i].data, _ = self.quantizer.quantize_inputs(grad_input[i].data, self.params.bitWidth, "backward-{}".format(module))
            else:
                grad_input.data, _ = self.quantizer.quantize_inputs(grad_output.data, self.params.bitWidth, "Backward-{}".format(module))
        
        return grad_input

