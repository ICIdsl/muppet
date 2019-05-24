import src.model_creator as mcSrc
import copy

class MuppetModelCreator(mcSrc.ModelCreator):   
    
    def setup_model(self, params):
        model = self.read_model(params)
        model,modelQuant = self.transfer_to_gpu(params, model)
        model = self.load_pretrained(params, model)
        criterion = self.setup_criterion()
        optimiser = self.setup_optimiser(params, model)
    
        return (model, modelQuant, criterion, optimiser)

    def transfer_to_gpu(self, params, model):
        gpu_list = [int(x) for x in params.gpu_id.split(',')]
        modelQuant = copy.deepcopy(model)
        
        model = torch.nn.DataParallel(model, gpu_list)
        model = model.cuda()
        
        modelQuant = torch.nn.DataParallel(modelQuant, gpu_list)
        modelQuant = modelQuant.cuda()

        return model, modelQuant


    
