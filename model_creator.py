import src.muppet.models as models
import src.muppet.quant_sgd as qsgd

import src.model_creator as mcSrc
import copy
import torch

class ModelCreator(mcSrc.ModelCreator):   
    def setup_model(self, params, quantizer):
        model = self.read_model(params)
        model = self.transfer_to_gpu(params, model)
        model = self.load_pretrained(params, model)
        criterion = self.setup_criterion()
        optimiser = self.setup_optimiser(params, model, quantizer)
    
        return (model, criterion, optimiser)

    def setup_optimiser(self, params, model, quantizer):
        return qsgd.QuantSGD(model.parameters(), quantizer, lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)

    def transfer_to_gpu(self, params, model):
        gpu_list = [int(x) for x in params.gpu_id.split(',')]
        
        model = torch.nn.DataParallel(model, gpu_list)
        model = model.cuda()
        
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
            model = models.__dict__[params.arch](
                        num_classes=num_classes,
                        depth=params.depth
                    )
        else:
            model = models.__dict__[params.arch](num_classes=num_classes)

        return model
