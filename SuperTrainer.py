import matplotlib.pyplot as plt


class Switch:
    def __init__(self):
        pass

    def switch(self):
        raise NotImplementedError("Not implemented!")


class SuperTrainer:
    def __init__(self, switch, models={}, in_functions={}, loss_functions={}, opts={}):
        """SuperTrainer object, the base class for all GAN trainer objects.
        switch is a subclass of the Switch object, and returns the designation for which model to train - implementation depends on the specific subclass
        Models is a dictionary containing the pytorch model objects, and is of the format {designation: model}
        In_functions is a dictionary containing the functions which create input data for the models, and is of the format {model designation: function}
        Loss functions is formatted the same as in_functions"""
        self.switch = switch
        self.models = models
        self.in_functions = in_functions
        self.loss_functions = loss_functions
        self.optimizers = opts
        self.losses = {}  # Dictionary to keep track of the losses over time of each model. Of the format {model designation: [loss0, loss1, ..., lossn]}

    def train(self, n_epochs, n_batch):
        raise NotImplementedError("Not implemented!")

    def eval(self, model, in_dat):
        return self.models[model](in_dat)

    def loss_by_epoch(self, model):  # TODO: format the graph nicely
        model_loss = self.losses[model]
        plt.plot(model_loss)
        plt.show()
