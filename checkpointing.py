import src.checkpointing as cpSrc
import sys

class Checkpointer(cpSrc.Checkpointer):
    def __init__(self, params, configFile):
        super().__init__(params, configFile)
        
        self.headers += ['DataType', 'BitWidth']

        if params.runMuppet:
            self.headers += ['MeanGD', 'MaxGD', \
                             'Ratio', 'Threshold', 'GDViolations']

    def setup_values(self, params):
        super().setup_values(params)
        
        self.values += [params.dataType, params.bitWidth]

        if params.runMuppet:
            meanGD = params.meanGD.item() if 'torch' in str(type(params.meanGD)) else params.meanGD
            maxGD = params.maxGD.item() if 'torch' in str(type(params.maxGD)) else params.maxGD
            ratio = (params.maxGD / params.meanGD).item() if 'torch' in str(type(params.meanGD)) else (params.maxGD / params.meanGD)
            
            self.values += [meanGD, maxGD, ratio, params.threshold, params.gdViolations]

    
