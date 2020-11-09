import os
import random
import sys

srcDir = os.path.split(os.getcwd())
ptDir = os.path.split(srcDir[0])
sys.path.append(ptDir[0])
import src.muppet.app as applic

import torch
import torch.cuda
import torch.multiprocessing as mp

import argparse

import getpass

def parse_command_line_args() : 
    parser = argparse.ArgumentParser(description='PyTorch Pruning')
    parser.add_argument('--config-file', default='None', type=str, help='config file with training parameters')
    args = parser.parse_args()
    return args

def main() : 
    # parse config
    print('==> Parsing Config File')
    args = parse_command_line_args()
    
    if args.config_file != 'None' : 
        app = applic.Application(args.config_file)
    else : 
        raise ValueError('Need to specify config file with parameters')

    app.main()

if __name__ == '__main__': 
    main()
