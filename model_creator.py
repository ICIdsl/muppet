import src.muppet.models as models
import src.muppet.quant_sgd as qsgd

import src.model_creator as mcSrc

import copy
import sys
import os

import torch

class ModelCreator(mcSrc.ModelCreator):   
    def setup_model(self, params, quantizer):
        model = self.read_model(params)
        model = self.transfer_to_gpu(params, model)
        model = self.load_pretrained(params, model)
        criterion = self.setup_criterion(params)
        optimiser = self.setup_optimiser(params, model, quantizer)
    
        return (model, criterion, optimiser)
    
    def setup_criterion(self, params):
        return torch.nn.CrossEntropyLoss()

    def setup_optimiser(self, params, model, quantizer):
        # use quantized SGD optimizer to quantize the gradients and update the model appropriately
        opt = qsgd.QuantSGD(model.parameters(), quantizer, lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)

        if params.resume == True or params.branch == True or params.evaluate == True:
            path = params.pretrained.split('/')
            epoch = path[-1].split('-')
            masterCopy = '-'.join([epoch[0], 'fpMasterCopy.pth.tar'])
            masterCopyPath = os.path.join('/'.join(path[:-1]), masterCopy)

            if os.path.exists(masterCopyPath):
                fp32Weights = torch.load(masterCopyPath)
                opt.param_groups[0]['fpWeights'] = fp32Weights
            
        return opt

    def transfer_to_gpu(self, params, model):
        model = torch.nn.DataParallel(model, device_ids=params.gpuList)
        device = 'cuda:' + str(params.gpuList[0])
        model.cuda(device)
        return model

    def read_model(self, params):
        if params.dataset == 'cifar10' : 
            import models.cifar as models 
            num_classes = 10
    
        elif params.dataset == 'cifar100' : 
            import models.cifar as models 
            num_classes = 100
    
        else : 
            import models.imagenet as models 
            num_classes = 1000
    
        print("Creating Quantized Model %s" % params.arch)
        
        if params.arch.endswith('resnet'):
            if 'cifar' in params.dataset:
                model = models.__dict__[params.arch](
                            num_classes=num_classes,
                            depth=params.depth
                        )
            else:
                if params.depth == 18:
                    model = models.__dict__['resnet18'](pretrained=False, progress=False)
        elif 'googlenet' in params.arch:
            if params.evaluate == False and params.dataset == 'imagenet':
                model = models.__dict__[params.arch](num_classes=num_classes, aux_logits=True)
            else:
                model = models.__dict__[params.arch](num_classes=num_classes)
            
        else:
            model = models.__dict__[params.arch](num_classes=num_classes)

        return model
