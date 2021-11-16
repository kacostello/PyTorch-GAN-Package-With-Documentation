import matplotlib.pyplot as plt





class SuperTrainer:
    def __init__(self, totrain, models={}, in_functions={}, loss_functions={}, opts={}):
        """SuperTrainer object, the base class for all GAN trainer objects.
        switch is a subclass of the Switch object, and returns the designation for which model to train - implementation depends on the specific subclass
        Models is a dictionary containing the pytorch model objects, and is of the format {designation: model}
        In_functions is a dictionary containing the functions which create input data for the models, and is of the format {model designation: function}
        Loss functions is formatted the same as in_functions"""
        self.totrain = totrain
        self.models = models
        self.in_functions = in_functions
        self.loss_functions = loss_functions
        self.optimizers = opts
        self.stats = {}  # Dictionary to keep track of the stats we want to save. Of the format {stat_name:stat_dict}
        self.stats["losses"] = {}  #Dictionary to keep track of the model losses over time Of the format {model designation: [loss0, loss1, ..., lossn]}

    def train(self, n_epochs, n_batch):
        raise NotImplementedError("Not implemented!")

    def eval(self, model, in_dat):
        self.models[model].eval()
        out = self.models[model](in_dat)
        self.models[model].train()
        return out

    def loss_by_epoch(self, model):  # TODO: format the graph nicely
        model_loss = self.stats["losses"][model]
        plt.plot(model_loss)
        plt.show()

    def epochs_trained(self, model):
        return self.stats["epochs_trained"][model]

    def total_epochs_trained(self):
        total = 0
        for model in self.list_models():
            total += self.epochs_trained(model)
        return total

    def list_models(self):
        return [n for n in self.models]  # Kinda scuffed code to get a list of a dict's keys. do not remember the actual way
