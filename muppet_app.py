import src.app as appSrc
import src.muppet.muppet_mc as mcSrc
import src.muppet.muppet_param_parser as ppSrc
import src.muppet.muppet_hooks as hookRegisterSrc

import src.checkpointing as checkpointingSrc
import src.input_preprocessor as preprocSrc
import src.training as trainingSrc
import src.inference as inferenceSrc

import configparser as cp

class Application(appSrc.Application):
    # test comment 
    def __init__(self, configFile):
        super().__init__(configFile)
    
    def setup_param_checkpoint(self, configFile):
        config = cp.ConfigParser() 
        config.read(configFile)
        self.params = ppSrc.Params(config)
        self.checkpointer = checkpointingSrc.Checkpointer(self.params, configFile)
        self.setup_params()

    def setup_others(self):
        self.preproc = preprocSrc.Preproc()
        self.mc = mcSrc.ModelCreator()
        self.trainer = trainingSrc.Trainer()
        self.inferer = inferenceSrc.Inferer()
        self.hookRegister = hookRegisterSrc.HookRegister()

    def setup_model(self):
        print('==> Setting up Model')
        self.model, self.modelQuant, self.criterion, self.optimiser = self.mc.setup_model(self.params)
        self.hookRegister.register_hooks(self.modelQuant)


