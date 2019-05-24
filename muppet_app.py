import src.app as appSrc
import src.muppet.muppet_mc as mcSrc

class MuppetApplication(appSrc.Application):
    
    def __init__(self, configFile):
        super().__init__(configFile)

    def setup_others(self):
        self.preproc = preprocSrc.Preproc()
        self.mc = mcSrc.ModelCreator()
        self.trainer = trainingSrc.Trainer()
        self.inferer = inferenceSrc.Inferer()

    def setup_model(self):
        print('==> Setting up Model')
        self.model, self.modelQuant, self.criterion, self.optimiser = self.mc.setup_model(self.params)


