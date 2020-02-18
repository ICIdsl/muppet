import sys
import time
from tqdm import tqdm

import torch
import torch.autograd

import src.utils as utils
import src.training as trainingSrc
import src.muppet.quantize as quantizerSrc

class Trainer(trainingSrc.Trainer):
    def __init__(self, quantizer):
        self.quantizer = quantizer 
    
    def update_lr(self, params, optimiser) : 
        if params.runMuppet:
            return
        
        # update learning rate
        if params.lr_schedule != [] : 
            # get epochs to change at and lr at each of those changes
            # ::2 gets every other element starting at 0 
            change_epochs = params.lr_schedule[::2]
            new_lrs = params.lr_schedule[1::2]
            epoch = params.curr_epoch
    
            if epoch in change_epochs : 
                new_lr = new_lrs[change_epochs.index(epoch)]
                if new_lr == -1 :
                    params.lr *= params.gamma
                else : 
                    params.lr = new_lr
             
            for param_group in optimiser.param_groups : 
                param_group['lr'] = params.lr
    
        return params

    def train(self, model, criterion, optimiser, inputs, targets, params): 

        outputs = model(inputs)
        
        if 'googlenet' in params.arch:
            if params.evaluate == False and params.dataset == 'imagenet':
                finalOp = outputs[0]
                auxOp2 = outputs[1]
                auxOp1 = outputs[2]
                loss1 = criterion(finalOp, targets)
                loss2 = criterion(auxOp2, targets)
                loss3 = criterion(auxOp1, targets)
                loss = loss1 + 0.3*loss2 + 0.3*loss3
                prec1, prec5 = utils.accuracy(finalOp.data, targets.data) 
        
            else:
                loss = criterion(outputs, targets)
                prec1, prec5 = utils.accuracy(outputs.data, targets.data) 
        else:
            loss = criterion(outputs, targets)
            prec1, prec5 = utils.accuracy(outputs.data, targets.data) 

        model.zero_grad() 
        loss.backward() 

        optimiser.step(params)

        return (loss.item(), prec1.item(), prec5.item())

    def batch_iter(self, model, criterion, optimiser, train_loader, params, losses, top1, top5):
        model.train()
        
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)-1, desc='epoch', leave=False): 
            # move inputs and targets to GPU
            device = 'cuda:'+str(params.gpuList[0])
            if params.use_cuda: 
                inputs, targets = inputs.cuda(device, non_blocking=True), targets.cuda(device, non_blocking=True)
            
            if params.dataType != 'Float':
                for i in range(len(inputs)):
                    inputs[i], _ = self.quantizer.quantize_inputs(inputs[i].data, params.bitWidth, "inputs")
            
            # train model
            loss, prec1, prec5 = self.train(model, criterion, optimiser, inputs, targets, params)

            losses.update(loss) 
            top1.update(prec1) 
            top5.update(prec5)
    
    def train_network(self, params, tbx_writer, checkpointer, train_loader, test_loader, valLoader, model, criterion, optimiser, inferer, policy, scaler):
        print('Epoch,\tLR,\tTrain_Loss,\tTrain_Top1,\tTrain_Top5,\tTest_Loss,\tTest_Top1,\tTest_Top5,\tVal_Loss,\tVal_Top1,\tVal_Top5,\tDataType,\tBitWidth')
        
        for epoch in tqdm(range(params.start_epoch, params.epochs), desc='training', leave=False) : 
            params.curr_epoch = epoch
            state = self.update_lr(params, optimiser)
    
            losses = utils.AverageMeter()
            top1 = utils.AverageMeter()
            top5 = utils.AverageMeter()

            # iterate over the batches in the epoch
            self.batch_iter(model, criterion, optimiser, train_loader, params, losses, top1, top5)

            params.train_loss = losses.avg        
            params.train_top1 = top1.avg        
            params.train_top5 = top5.avg        
            
            # get val and test loss
            params.test_loss, params.test_top1, params.test_top5 = inferer.test_network(params, test_loader, model, criterion, optimiser)
            params.val_loss, params.val_top1, params.val_top5 = inferer.test_network(params, valLoader, model, criterion, optimiser)
            
            if params.runMuppet:
                policy.update(model)
                if policy.check_violation(epoch, tqdm, checkpointer):
                    policy.change_precision(scaler, model, optimiser)
                    tqdm.write("GD violation detected, precision changed to {}".format(params.bitWidth))
                if policy.check_stopping_condition(optimiser):
                    tqdm.write("Ending training")
                    return
            
            checkpointer.save_checkpoint(model.state_dict(), optimiser.state_dict(), params)
            
            tqdm.write("{},\t{},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{},\t\t{}".format(epoch, params.lr, params.train_loss, params.train_top1, params.train_top5, params.test_loss, params.test_top1, params.test_top5, params.val_loss, params.val_top1, params.val_top5, params.dataType, params.bitWidth))
