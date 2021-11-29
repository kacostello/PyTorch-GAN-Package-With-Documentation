import matplotlib.pyplot as plt
import os
import torch
import pickle


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
        self.stats["losses"] = {}  # Dictionary to keep track of the model losses over time Of the format {model designation: [loss0, loss1, ..., lossn]}

    def __eq__(self, other):
        try:
            # Check if all models have equal state dicts
            # TODO: This doesn't actually guarantee that each model is strictly *equal*, but it's good enough for checking if save/load works
            for model_name in self.list_models():
                assert self.models[model_name].state_dict() == other.models[model_name].state_dict()
            assert self.stats == other.stats
            assert self.in_functions == other.in_functions
            assert self.loss_functions == other.loss_functions
            for opt_name in self.list_opts():
                assert self.optimizers[opt_name].state_dict() == other.optimizers[opt_name].state_dict()
            assert self.list_opts() == other.list_opts()
            assert self.list_models() == other.list_models()
        except:
            return False
        finally:
            return True

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
        return [n for n in
                self.models]  # Kinda scuffed code to get a list of a dict's keys. do not remember the actual way

    def list_opts(self):
        return [n for n in self.optimizers]

    def save_model_state_dicts(self, path):
        """Saves the model state dicts to folder at specified location. Creates folder if it does not exist.
        Saves state dicts in format path/model_name.pt for each model_name in self.models.
        WARNING: Will overwrite existing state_dicts, if present."""
        if not os.path.isdir(path):
            os.mkdir(path)

        for model_name in self.list_models():
            torch.save(self.models[model_name].state_dict(), path + "\\" + model_name + ".pt")

    def save_opt_state_dicts(self, path):
        """Saves the optimizer state dicts to folder at specified location. Creates folder if it does not exist.
        Saves state dicts in format path/optimizer_name.pto for each optimizer_name in self.optimizers.
        WARNING: Will overwrite existing state_dicts, if present."""
        if not os.path.isdir(path):
            os.mkdir(path)

        for opt_name in self.list_opts():
            torch.save(self.optimizers[opt_name].state_dict(), path + "\\" + opt_name + ".pto")

    def load_model_state_dicts(self, path):
        """Loads model state dicts from specified folder. Throws ValueError if path does not exist,
        or does not have *all* necessary state dicts. Requires models to already be instantiated with the
        same structure as saved!
        WARNING: the torch.load() function uses pickle. *Only attempt to load state_dicts from a trusted source.*"""
        try:
            assert os.path.isdir(path)
        except AssertionError:
            raise ValueError("No folder at " + path)

        try:
            for model_name in self.list_models():
                assert os.path.isfile(path + "\\" + model_name + ".pt")
        except AssertionError:
            raise ValueError("Not all models have an associated state_dict at " + path)

        for model_name in self.list_models():
            self.models[model_name].load_state_dict(torch.load(path + "\\" + model_name + ".pt"))

    def load_opt_state_dicts(self, path):
        """Loads optimizer state dicts from specified folder. Throws ValueError if path does not exist,
        or does not have *all* necessary state dicts. Requires optimizers to already be instantiated with the
        same structure as saved!
        WARNING: the torch.load() function uses pickle. *Only attempt to load state_dicts from a trusted source.*"""
        try:
            assert os.path.isdir(path)
        except AssertionError:
            raise ValueError("No folder at " + path)

        try:
            for opt_name in self.list_opts():
                assert os.path.isfile(path + "\\" + opt_name + ".pto")
        except AssertionError:
            raise ValueError("Not all optimizers have an associated state_dict at " + path)

        for opt_name in self.list_opts():
            self.optimizers[opt_name].load_state_dict(torch.load(path + "\\" + opt_name + ".pto"))

    def save_trainer_stats_dict(self, path):
        """Saves trainer stats dict into specified folder. Creates folder if it does not exist.
        Saves state dict in format path/trainer_stats.ts. Not (yet) guaranteed to be compatible across different
        versions!
        WARNING: Will overwrite the existing trainer_stats file, if it exists."""
        if not os.path.isdir(path):
            os.mkdir(path)

        f = open(path + "\\trainer_stats.ts", "wb")
        pickle.dump(self.stats, f)
        f.close()

    def save_loss_functions(self, path):
        """Saves loss functions dict into specified folder. Creates folder if it does not exist.
        Saves state dict in format path/loss_functions.ts.
        WARNING: Will overwrite the existing loss_functions file, if it exists."""
        if not os.path.isdir(path):
            os.mkdir(path)

        f = open(path + "\\loss_functions.ts", "wb")
        pickle.dump(self.loss_functions, f)
        f.close()

    def save_in_functions(self, path):
        """Saves in functions dict into specified folder. Creates folder if it does not exist.
        Saves state dict in format path/in_functions.ts.
        WARNING: Will overwrite the existing in_functions file, if it exists."""
        if not os.path.isdir(path):
            os.mkdir(path)

        f = open(path + "\\in_functions.ts", "wb")
        pickle.dump(self.in_functions, f)
        f.close()

    def save_to_train(self, path):
        """Saves to_train object into specified folder. Creates folder if it does not exist.
        Saves state dict in format path/to_train.ts.
        WARNING: Will overwrite the existing to_train file, if it exists."""
        if not os.path.isdir(path):
            os.mkdir(path)

        f = open(path + "\\to_train.ts", "wb")
        pickle.dump(self.totrain, f)
        f.close()

    def load_trainer_state_dict(self, path):
        """Loads trainer state dict from specified folder. Throws ValueError if path does not exist, or if
        path/trainer_stats.ts does not exist. Requires the trainer object to already be instantiated. Will
        overwrite the current trainer stats dict.
        WARNING: This load function uses pickle. *Only attempt to load from a trusted source.*"""
        try:
            assert os.path.isdir(path)
        except AssertionError:
            raise ValueError("No folder at " + path)

        try:
            assert os.path.isfile(path + "\\trainer_stats.ts")
        except AssertionError:
            raise ValueError("Cannot detect trainer stats dict at " + path)

        f = open(path + "\\trainer_stats.ts", "rb")
        self.stats = pickle.load(f)
        f.close()

    def load_loss_functions(self, path):
        """Loads loss_functions dict from specified folder. Throws ValueError if path does not exist, or if
        path/loss_functions.ts does not exist. Requires the trainer object to already be instantiated. Will
        overwrite the current loss_functions dict. Note that this requires the loss functions to be defined
        in the current scope, or otherwise defined within a built-in module (if you're using a custom loss
        function, make sure to import the file where it's defined!). If you're using pytorch's built-in loss functions,
        and you have pytorch properly installed, this should work without importing pytorch.
        WARNING: This load function uses pickle. *Only attempt to load from a trusted source.*"""
        try:
            assert os.path.isdir(path)
        except AssertionError:
            raise ValueError("No folder at " + path)

        try:
            assert os.path.isfile(path + "\\loss_functions.ts")
        except AssertionError:
            raise ValueError("Cannot detect loss_functions dict at " + path)

        f = open(path + "\\loss_functions.ts", "rb")
        self.loss_functions = pickle.load(f)
        f.close()

    def load_in_functions(self, path):
        """Loads in_functions dict from specified folder. Throws ValueError if path does not exist, or if
        path/in_functions.ts does not exist. Requires the trainer object to already be instantiated. Will
        overwrite the current in_functions dict. Note that this requires the in functions to be defined
        in the current scope, or otherwise defined within a built-in module (if you're using a custom
        function, make sure to import the file where it's defined!).
        WARNING: This load function uses pickle. *Only attempt to load from a trusted source.*"""
        try:
            assert os.path.isdir(path)
        except AssertionError:
            raise ValueError("No folder at " + path)

        try:
            assert os.path.isfile(path + "\\in_functions.ts")
        except AssertionError:
            raise ValueError("Cannot detect in_functions dict at " + path)

        f = open(path + "\\in_functions.ts", "rb")
        self.in_functions = pickle.load(f)
        f.close()

    def load_to_train(self, path):
        """Loads totrain object from specified folder. Throws ValueError if path does not exist, or if
        path/to_train.ts does not exist. Requires the trainer object to already be instantiated. Will
        overwrite the current totrain object. Note that this requires the totrain object to be defined
        in the current scope, or otherwise defined within a built-in module (if you're using a custom
        object, make sure to import the file where it's defined!).
        WARNING: This load function uses pickle. *Only attempt to load from a trusted source.*"""
        try:
            assert os.path.isdir(path)
        except AssertionError:
            raise ValueError("No folder at " + path)

        try:
            assert os.path.isfile(path + "\\to_train.ts")
        except AssertionError:
            raise ValueError("Cannot detect in_functions dict at " + path)

        f = open(path + "\\to_train.ts", "rb")
        self.totrain = pickle.load(f)
        f.close()

    def soft_save(self, path):
        """Saves all model state_dicts, the trainer's stat dict, all optimizer state_dicts,
        all in_functions, and all loss_functions.
        Will overwrite previously saved dicts in same location!"""
        self.save_model_state_dicts(path)
        self.save_trainer_stats_dict(path)
        self.save_loss_functions(path)
        self.save_in_functions(path)
        self.save_opt_state_dicts(path)
        self.save_to_train(path)

    def soft_load(self, path):
        """Loads all model state_dicts and the trainer's stat dict.
        Note that this requires the loss functions and in functions to be defined in the current scope, or otherwise
        defined within a built-in module (if you're using a custom loss function, make sure to import the file where
        it's defined!). If you're using pytorch's built-in loss functions, and you have pytorch properly installed, you
        shouldn't need to import pytorch.
        WARNING: This load function uses pickle. *Only attempt to load from a trusted source.*"""
        self.load_model_state_dicts(path)
        self.load_trainer_state_dict(path)
        self.load_in_functions(path)
        self.load_loss_functions(path)
        self.load_opt_state_dicts(path)
        self.load_to_train(path)
