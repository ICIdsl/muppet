from __future__ import print_function

import argparse
import configparser as cp

import src.param_parser as ppSrc

class Params(ppSrc.Params): 
    def __init__(self, config_file) : 
        super().__init__(config_file)
        self.minLR = config_file.getfloat('training_hyperparameters', 'min_lr')

        # muppet attributes 
        self.runMuppet = config_file.getboolean('muppet_hyperparameters', 'run_muppet')
        self.bitWidth = config_file.getint('muppet_hyperparameters', 'bit_width')
        self.dataType = config_file.get('muppet_hyperparameters', 'data_type')
        self.roundMeth = config_file.get('muppet_hyperparameters', 'round_meth')
        self.policyResolution = config_file.getint('muppet_hyperparameters', 'policy_resolution')
        self.policyPatience = config_file.getint('muppet_hyperparameters', 'policy_patience')
        self.lowPrecLimit = config_file.getint('muppet_hyperparameters', 'low_prec_limit')
        self.fp32EpochsPerLR = config_file.getint('muppet_hyperparameters', 'fp32_epochs_per_lr')

        if self.dataType == "Float":
            self.bitWidth = -1

        self.meanGD = 1
        self.maxGD = 0
        self.gdViolations = 0
        self.sumOfNorms = {}
        self.sumOfGrads = {}
        self.quantised = (self.bitWidth != 'Float')
        self.sub_classes = []
        self.threshold = -1
        
def parse_command_line_args() : 
    parser = argparse.ArgumentParser(description='PyTorch Pruning')

    # Command line vs Config File
    parser.add_argument('--config-file', default='None', type=str, help='config file with training parameters')
    
    args = parser.parse_args()

    return args
