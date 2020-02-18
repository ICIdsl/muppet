from __future__ import print_function

import argparse
import configparser as cp
import sys

import src.param_parser as ppSrc

class Params(ppSrc.Params): 
    def __init__(self, config_file) : 
        # import generic configuration paramters 
        super().__init__(config_file)
        self.minLR = config_file.getfloat('training_hyperparameters', 'min_lr')

        # muppet attributes 
        self.runMuppet = config_file.getboolean('muppet_hyperparameters', 'run_muppet')
        self.bitWidth = config_file.getint('muppet_hyperparameters', 'bit_width')
        self.dataType = config_file.get('muppet_hyperparameters', 'data_type')
        self.roundMeth = config_file.get('muppet_hyperparameters', 'round_meth')
        self.policyResolution = config_file.getint('muppet_hyperparameters', 'policy_resolution')
        self.policyPatience = config_file.getint('muppet_hyperparameters', 'policy_patience')
        self.fp32EpochsPerLR = config_file.getint('muppet_hyperparameters', 'fp32_epochs_per_lr')
        self.precEpochSchedule = config_file.get('muppet_hyperparameters', 'prec_epoch_schedule', fallback = 'undefined')
        
        self.precSchedule = config_file.get('muppet_hyperparameters', 'prec_schedule', fallback='undefined')
        if self.runMuppet:
            if self.precSchedule == 'undefined' or self.precSchedule == '':
                raise ValueError('Precision Schedule not defined')
            else:
                self.precSchedule = [int(x) for x in self.precSchedule.split()]
                assert self.precSchedule[0] == self.bitWidth, 'specified bitwidth ({}) and initial precision ({}) in prec schedule should match'.format(self.bitWidth, self.precSchedule[0])

            if self.precEpochSchedule == 'undefined' or self.precEpochSchedule == '':
                self.precEpochSchedule = [] 
            else:
                self.precEpochSchedule = [int(x) for x in self.precEpochSchedule.split()]
                assert len(self.precEpochSchedule) == len(self.precSchedule)-1, 'Number of precision switching points ({}) does not match the number of precisions-1 ({})'.format(len(self.precEpochSchedule), len(self.precSchedule))
                    

        if self.dataType == "Float":
            self.bitWidth = -1

        self.meanGD = 1
        self.maxGD = 0
        self.gdViolations = 0
        self.sumOfNorms = {}
        self.sumOfGrads = {}
        self.quantised = (self.bitWidth != 'Float')
        self.threshold = -1
        
def parse_command_line_args() : 
    parser = argparse.ArgumentParser(description='PyTorch Pruning')

    # Command line vs Config File
    parser.add_argument('--config-file', default='None', type=str, help='config file with training parameters')
    
    args = parser.parse_args()

    return args
