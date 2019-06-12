import sys
import time
from tqdm import tqdm

import torch.autograd

import src.utils as utils
import src.training as trainingSrc
import src.muppet.quantize as quantizerSrc

class Trainer(trainingSrc.Trainer):
    def __init__(self):
        self.quantizer = quantizerSrc.Quantizer()

    def train(self, model, criterion, optimiser, inputs, targets, params): 
        model.train()
        
        outputs = model(inputs) 
        loss = criterion(outputs, targets)
        # loss.data, _ = self.quantizer.quantize_inputs(loss.data, params.bitWidth)
    
        prec1, prec5 = utils.accuracy(outputs.data, targets.data) 
    
        model.zero_grad() 
        loss.backward() 
        
        optimiser.step(params)
    
        return (loss, prec1, prec5)

    def batch_iter(self, model, criterion, optimiser, train_loader, params, losses, top1, top5):
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)-1, desc='epoch', leave=False): 
            # move inputs and targets to GPU
            if params.use_cuda: 
                inputs, targets = inputs.cuda(), targets.cuda()
            for i in range(len(inputs)):
                inputs[i], _ = self.quantizer.quantize_inputs(inputs[i].data, params.bitWidth)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            
            # train model
            loss, prec1, prec5 = self.train(model, criterion, optimiser, inputs, targets, params)
            
            losses.update(loss) 
            top1.update(prec1) 
            top5.update(prec5)
    

    
    def train_network(self, params, tbx_writer, checkpointer, train_loader, test_loader, valLoader, model, criterion, optimiser, inferer):
        print('Epoch,\tLR,\tTrain_Loss,\tTrain_Top1,\tTrain_Top5,\tTest_Loss,\tTest_Top1,\tTest_Top5,\tVal_Loss,\tVal_Top1,\tVal_Top5')
        
        for epoch in tqdm(range(params.start_epoch, params.epochs), desc='training', leave=False) : 
            params.curr_epoch = epoch
            state = self.update_lr(params, optimiser)
    
            losses = utils.AverageMeter()
            top1 = utils.AverageMeter()
            top5 = utils.AverageMeter()
            
            self.batch_iter(model, criterion, optimiser, train_loader, params, losses, top1, top5)
            
            params.train_loss = losses.avg        
            params.train_top1 = top1.avg        
            params.train_top5 = top5.avg        
    
            # get test loss
            params.test_loss, params.test_top1, params.test_top5 = inferer.test_network(params, test_loader, model, criterion, optimiser)
            params.val_loss, params.val_top1, params.val_top5 = inferer.test_network(params, valLoader, model, criterion, optimiser)
            
            checkpointer.save_checkpoint(model.state_dict(), optimiser.state_dict(), params)
            
            tqdm.write("{},\t{},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f}".format(epoch, params.lr, params.train_loss, params.train_top1, params.train_top5, params.test_loss, params.test_top1, params.test_top5, params.val_loss, params.val_top1, params.val_top5))
