import sys

import src.app as appSrc
import src.muppet.model_creator as mcSrc
import src.muppet.param_parser as ppSrc
import src.muppet.scaler as scaleSrc
import src.muppet.training as trainingSrc
import src.muppet.quantize as quantizeSrc
import src.muppet.quant_layers as quantLayersSrc
import src.muppet.policy as policySrc
import src.muppet.checkpointing as checkpointingSrc

# import src.checkpointing as checkpointingSrc
import src.input_preprocessor as preprocSrc
import src.inference as inferenceSrc

import configparser as cp

import torch

class Application(appSrc.Application):
    def setup_param_checkpoint(self, configFile):
        config = cp.ConfigParser() 
        config.read(configFile)
        self.params = ppSrc.Params(config)
        self.checkpointer = checkpointingSrc.Checkpointer(self.params, configFile)
        self.setup_params()

    def setup_others(self):
        self.preproc = preprocSrc.Preproc()
        self.mc = mcSrc.ModelCreator()
        self.quantizer = quantizeSrc.Quantizer(self.params.roundMeth)
        self.trainer = trainingSrc.Trainer(self.quantizer)
        self.inferer = inferenceSrc.Inferer()
        self.policy = policySrc.Policy(self.params) 

    def setup_model(self):
        print('==> Setting up quantized Model')
        self.model, self.criterion, self.optimiser = self.mc.setup_model(self.params, self.quantizer)
        self.scaler = scaleSrc.Scaler(self.model, self.quantizer, self.params)
        self.scaler.register_hooks()
        self.sfHolder = quantLayersSrc.SFHolder()
        
    def run_training(self):
        # train model 
        print('==> Performing Training')
        self.trainer.train_network(self.params, None, self.checkpointer, self.train_loader, self.test_loader, self.valLoader, self.model, self.criterion, self.optimiser, self.inferer, self.policy, self.scaler)
