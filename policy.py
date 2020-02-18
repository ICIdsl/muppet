import sys
import math
import torch
import time

class Policy(object):
    def __init__(self, params):
        self.params = params
        self.threshold = self._threshold()
        self.fp32Count = 0
        self.precIndex = 0

    def _threshold(self):
        return [1 + 1.5*math.exp(-0.1*x) for x in range(self.params.epochs)]

    def update_calculation(self, layer):
        layerName = layer[0]
        layerGrad = layer[1].grad.data

        gradNorm = torch.pow(torch.norm(layerGrad,2),2)
        if layerName in self.params.sumOfNorms:
            self.params.sumOfNorms[layerName].add_(gradNorm)
        else:
            self.params.sumOfNorms[layerName] = gradNorm

        if layerName in self.params.sumOfGrads:
            self.params.sumOfGrads[layerName].add_(layerGrad)
        else:
            self.params.sumOfGrads[layerName] = layerGrad

    def update(self, model):
        for param in model.named_parameters():
            self.update_calculation(param)

    def calculate_mean_gd(self):
        self.params.meanGD = 0
        numGD = 0
        for layer, val in self.params.sumOfGrads.items():
            self.params.meanGD += (self.params.sumOfNorms[layer] / torch.pow(torch.norm(self.params.sumOfGrads[layer],2),2))
            numGD += 1
        self.params.meanGD = self.params.meanGD / numGD
        self.params.sumOfNorms = {}
        self.params.sumOfGrads = {}

    # check if policy has been violated
    def check_violation(self, epoch, tqdm, cp): 
        if self.params.precEpochSchedule == []:
            # ensure that the resolution has been met, and if not there is no need to check the violation
            if ((epoch+1) % self.params.policyResolution) != 0 or self.params.dataType == 'Float':
                return False 

            self.calculate_mean_gd()
            
            if self.params.meanGD >= self.params.maxGD:
                self.params.maxGD = self.params.meanGD 
            else:
                if (self.params.maxGD / self.params.meanGD) > self.threshold[epoch]:
                    self.params.gdViolations += 1
            
            self.params.threshold = self.threshold[epoch]

            tqdm.write("meanGD = {}, maxGD = {}, ratio = {}, threshold = {}, gdViolations = {}".format(self.params.meanGD, self.params.maxGD, (self.params.maxGD / self.params.meanGD), self.threshold[epoch], self.params.gdViolations))

            if self.params.gdViolations >= self.params.policyPatience:
                self.params.gdViolations = 0
                return True
            
            return False
        else:
            if epoch in self.params.precEpochSchedule:
                return True
            else:
                return False
    
    def copy_fp32_model(self, optimiser):
        for group in optimiser.param_groups:
            fpWeights = group['fpWeights']
            weights = group['params']
            for i in range(len(weights)):
                weights[i].data = fpWeights[i].data
        
    def change_precision(self, scaler, model, optimiser):
    #{{{
        self.precIndex += 1 
        self.params.bitWidth = self.params.precSchedule[self.precIndex]
        self.params.maxGD = 0

        # check to see if it is the final precision change to FP32
        if self.params.bitWidth == -1:
            self.params.dataType = 'Float'
            self.copy_fp32_model(optimiser)

        scaler.update_model_precision(model)
    #}}}

    def check_stopping_condition(self, optimiser):
    #{{{
        if self.params.dataType == 'Float':
            # check if the minimum number of FP32 epochs has passed
            if (self.fp32Count+1) % self.params.fp32EpochsPerLR == 0:
                # check if the final LR has been passed yet
                if self.params.lr > self.params.minLR:
                    for group in optimiser.param_groups: 
                        group['lr'] *= self.params.gamma 
                        self.params.lr *= self.params.gamma
                else:
                    return True
           
            self.fp32Count += 1
            return False
        else:
            return False
    #}}}
