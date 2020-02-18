import src.checkpointing as cpSrc
import sys
import subprocess
import os
import torch

class Checkpointer(cpSrc.Checkpointer):
    def __init__(self, params, configFile):
        super().__init__(params, configFile)
        
        self.headers += ['DataType', 'BitWidth']

        # include additional headers relevant to MuPPET
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

    def save_checkpoint(self, model_dict, optimiser_dict, params) : 
        if params.printOnly == True:
            return

        # create directory if not done already, this way empty directory is never created
        if self.created_dir == False : 
            self.__create_dir(self.root)
            self.__create_log(self.root)
        
        # copy config file into root dir
        cmd = 'cp ' + self.configFile + ' ' + self.root
        subprocess.check_call(cmd, shell=True)         

        # write to log file
        self.setup_values(params)
        line = [str(x) for x in self.values]
        line = ',\t'.join(line)
        line += '\n'
        with open(self.logfile, 'a') as f :
            f.write(line)

        # create checkpoints to store
        modelpath = os.path.join(self.root, str(params.curr_epoch) + '-model' + '.pth.tar')
        statepath = os.path.join(self.root, str(params.curr_epoch) + '-state' + '.pth.tar')
        fpWeightsPath = os.path.join(self.root, str(params.curr_epoch) + '-fpMasterCopy' + '.pth.tar')
        
        # store checkpoints
        torch.save(model_dict, modelpath) 
        torch.save(params.get_state(), statepath)
        torch.save(optimiser_dict['param_groups'][0]['fpWeights'], fpWeightsPath)
            
        # store best model separately 
        if params.val_top1 >= params.bestValidLoss:
            params.bestValidLoss = params.val_top1
            bestModelPath = os.path.join(self.root, 'best-model' + '.pth.tar')
            bestStatePath = os.path.join(self.root, 'best-state' + '.pth.tar')
            bestFp32Path = os.path.join(self.root, 'best-fp32' + '.pth.tar')
            torch.save(model_dict, bestModelPath) 
            torch.save(params.get_state(), bestStatePath)
            torch.save(optimiser_dict['param_groups'][0]['fpWeights'], bestFp32Path)
    

    def restore_state(self, params): 
        res = super().restore_state(params)
        if params.branch == True : 
            file_to_load = params.pretrained.replace('model', 'state')        
            device = 'cuda:' + str(params.gpuList[0])
            prev_state_dict = torch.load(file_to_load, map_location=device)

            res.maxGD = prev_state_dict['maxGD']
            res.meanGD = prev_state_dict['meanGD']

        return res

