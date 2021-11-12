import SuperTrainer


class TwoFiveSwitch(SuperTrainer.Switch):
    """Implementation of Switch which follows a simple 2-5 ratio rule: Train G for 2 epochs, and D for 5."""

    def __init__(self):
        SuperTrainer.Switch.__init__(self)
        self.state = 0

    def switch(self):
        if self.state < 2:
            self.state += 1
            return "G"
        else:
            self.state += 1
            if self.state >= 7:
                self.state = 0
            return "D"


class SimpleGANTrainer(SuperTrainer.SuperTrainer):
    def __init__(self, generator, discriminator, latent_space_function, random_from_dataset, g_loss, d_loss, g_opt,
                 d_opt, sw=None):
        """Class to train a simple GAN.
        Generator and discriminator are torch model objects
        Latent_space_function(n) is a function which returns an array of n points from the latent space
        Random_from_dataset is a function which returns an array of n points from the real dataset"""
        if sw is None:
            self.switch = TwoFiveSwitch()
        else:
            self.switch = sw
        self.dataset = random_from_dataset
        self.latent_space = latent_space_function
        SuperTrainer.SuperTrainer.__init__(self, sw, models={"G": generator, "D": discriminator},
                                           in_functions={"G": self.generator_input,
                                                         "D": self.discriminator_input},
                                           loss_functions={"G": g_loss, "D": d_loss},
                                           opts={"G": g_opt, "D": d_opt})
        self.losses = {"G": [], "D": []}

    def train(self, n_epochs, n_batch):
        for epoch in range(n_epochs):
            sw = self.switch.switch()  # Determine which model to train - sw will either be "D" or "G"

            # Both input functions return the tuple (dis_in, labels)
            # generator_in returns (gen_out, labels) - this data is passed through D and used to train G
            # discriminator_in returns (dis_in, labels) - this is used to train D directly
            # For other GAN types: input functions can return whatever makes the most sense for your specific type of GAN
            # (so controllable GAN, for instance, might want to return a classification vector as well)
            dis_in, y = self.in_functions[sw](n_batch)
            if sw == "G":  # If we're training the generator, we should temporarily put the discriminator in eval mode
                self.models["D"].eval()
            mod_pred = self.models["D"](dis_in)
            self.models["D"].train()
            mod_loss = self.loss_functions[sw](mod_pred, y)

            # Logging for visualizers (currently only loss_by_epoch)
            self.losses[sw].append(mod_loss.item())

            # Pytorch training steps
            self.optimizers[sw].zero_grad()
            mod_loss.backward()
            self.optimizers[sw].step()

    def eval_generator(self, in_dat):
        return self.eval("G", in_dat)

    def eval_discriminator(self, in_dat):
        return self.eval("D", in_dat)

    def get_g_loss_fn(self):
        return self.loss_functions["G"]

    def get_g_opt_fn(self):
        return self.optimizers["G"]

    def get_d_loss_fn(self):
        return self.loss_functions["D"]

    def get_d_opt_fn(self):
        return self.optimizers["D"]

    def loss_by_epoch_g(self):
        self.loss_by_epoch("G")

    def loss_by_epoch_d(self):
        self.loss_by_epoch("D")

    def discriminator_input(self, n_batch):
        gen_in = self.latent_space(int(n_batch / 2))
        gen_out = self.models["G"](gen_in)
        dis_in = gen_out + self.dataset(int(n_batch / 2))
        y = [0 for n in range(int(n_batch / 2))] + [1 for n in range(int(n_batch / 2))]
        return dis_in, y

    def generator_input(self, n_batch):
        gen_in = self.latent_space(n_batch)
        gen_out = self.models["G"](gen_in)
        y = [1 for n in range(n_batch)]
        return gen_out, y
