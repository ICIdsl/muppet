import torch
import src.muppet.quantize as quantize
import sys
import copy
from torch.optim.optimizer import required


class QuantSGD(torch.optim.Optimizer):
    def __init__(self, params, _quantizer, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        self.quantizer = _quantizer

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov, fpWeights=[])
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(QuantSGD, self).__init__(params, defaults)

        for group in self.param_groups:
            for i in range(len(group['params'])):
                group['fpWeights'].append(copy.deepcopy(group['params'][i]))

    def __setstate__(self, state):
        super(QuantSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, params, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            fpWeights = group['fpWeights']
            weights = group['params']

            for i in range(len(weights)):
                p = weights[i]
                fp = fpWeights[i]

                if p.grad is None:
                    continue

                # quantize the gradients
                if params.dataType != 'Float':
                    p.grad.data, _ = self.quantizer.quantize_inputs(p.grad.data, params.bitWidth, "optimizer-grad-{}".format(p.grad.data.shape))
                
                d_p = p.grad.data
                
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # if still in dynamic fixed point, quantize updated FP32 weights to target precision for upcoming forward pass
                if params.dataType == 'Float':
                    p.data.add_(-group['lr'], d_p)
                else:
                    fp.data.add_(-group['lr'], d_p)
                    p.data, _ = self.quantizer.quantize_inputs(fp.data, params.bitWidth, "optimizer-data-{}".format(p.data.shape))
            
        return loss
