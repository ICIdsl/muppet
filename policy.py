import sys
import math

class Policy(object):
    def __init__(self, params):
        # self.meanGD = params.meanGD
        # self.maxGD = params.maxGD
        # self.sumOfNorms = params.sumOfNorms
        # self.sumOfGrads = params.sumOfGrads
        # self.gdViolations = params.gdViolations
        # self.resolution = params.policyResolution
        self.params = params
        self.threshold = self._threshold()
        self.fp32Count = 0

    def _threshold(self):
        return [1 + 1.5*math.exp(-0.1*x) for x in range(self.params.epochs)]

    def update_calculation(self, layer):
        layerName = layer[0]
        layerGrad = layer[1].grad.data
        
        gradNorm = torch.pow(torch.norm(grad,2),2)
        if layerName in self.params.sumOfNorms:
            self.params.sumOfNorms[layer].add_(gradNorm)
        else:
            self.params.sumOfNorms[layer] = gradNorm

        if layerName in self.params.sumOfGrads:
            self.params.sumOfGrads[layer].add_(layerGrad)
        else:
            self.params.sumOfGrads[layer] = layerGrad

    def update(self, model):
        for param in model.named_parameters():
            self.update_calculation(param)

    def calculate_mean_gd(self):
        self.params.meanGD = 0
        numGd = 0
        for layer in self.params.sumOfGrads:
            self.params.meanGD += (self.params.sumOfNorms[layer] / torch.pow(torch.norm(self.params.sumOfGrads[layer],2),2))
            numGD += 1
        self.params.meanGD = self.params.meanGD / numGD
        self.params.sumOfNorms = {}
        self.params.sumOfGrads = {}

    def check_violation(self, epoch): 
        if ((epoch+1) % self.params.resolution) != 0:
            return False 

        self.calculate_mean_gd()
        
        if self.params.meanGD >= self.params.maxGD:
            self.params.maxGD = self.params.meanGD 
        else:
            if (self.params.maxGD / self.params.meanGD) > self.threshold[epoch]:
                self.params.gdViolations += 1
        
        if self.params.gdViolations >= self.params.patience:
            self.params.gdViolations = 0
            return True
        
        return False
    
    def copy_fp32_model(self, optimiser):
        for group in optimiser.param_groups:
            fpWeights = group['fpWeights']
            weights = group['params']
            for i in range(len(weights)):
                weights[i].data = fpWeights[i].data
        
    def change_precision(self, scaler, model, optimiser):
        if (self.params.bitWidth < self.params.lowPrecLimit) and self.params.dataType != 'Float':
            # change bitWidth in params which ensures, 
            # weights, inputs, and gradients get quantised correctly
            self.params.bitWidth += 2
            self.params.maxGD = 0

            # call scaler's update_model_precision to ensure
            # layer outputs are quantised correctly
            scaler.update_model_precision(model)
        else:
            self.params.bitWidth = -1
            self.params.dataType = 'Float'
            self.params.maxGD = 0
            
            scaler.update_model_precision(model)
            
            if self.fp32Count == 0: 
                self.copy_fp32_model(optimiser)
    
    def check_stopping_condition(self, optimiser):
        if self.params.dataType == 'Float':
            if (self.fp32Count+1) % self.params.fp32EpochsPerLR == 0:
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
        

                
                    











