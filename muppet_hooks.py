import sys

class HookRegister(object):

    def quantise(self, input, output):
        # output = quantiser.quantise_inputs(output, params.bitWidth)
        output = input

    def register_hooks(self, model):
        modules = model._modules
        layers = modules['module']._modules
        for k,v in layers.items():
            if 'conv' in k or 'relu' in k or 'classifier' in k:
                v.register_forward_hook(self.quantise)
                v.register_backward_hook(self.quantise)


