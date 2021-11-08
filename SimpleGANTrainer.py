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
                 d_opt):
        """Class to train a simple GAN.
        Generator and discriminator are torch model objects
        Latent_space_function(n) is a function which returns an array of n points from the latent space
        Random_from_dataset is a function which returns an array of n points from the real dataset"""
        sw = TwoFiveSwitch()
        self.dataset = random_from_dataset
        self.latent_space = latent_space_function
        SuperTrainer.SuperTrainer.__init__(self, sw, models={"G": generator, "D": discriminator},
                                           in_functions={"G": SimpleGANTrainer.generator_input,
                                                         "D": SimpleGANTrainer.discriminator_input},
                                           loss_functions={"G": g_loss, "D": d_loss},
                                           opts={"G": g_opt, "D": d_opt})
        self.losses = {"G": [], "D": []}

    def train(self, n_epochs, n_batch):
        for epoch in range(n_epochs):
            sw = self.switch.switch()
            if sw == "G":
                gen_pred, y = self.in_functions["G"](n_batch, self.models["G"], self.models["D"], self.latent_space)
                gen_loss = self.loss_functions["G"](gen_pred, y)
                self.losses["G"].append(gen_loss.item())
                self.optimizers["G"].zero_grad()
                gen_loss.backward()
                self.optimizers["G"].step()
            else:
                dis_in, y = self.in_functions["D"](n_batch, self.models["G"], self.latent_space, self.dataset)
                dis_pred = self.models["D"](dis_in)
                dis_loss = self.loss_functions["D"](dis_pred, y)

                self.losses["D"].append(dis_loss.item())
                self.optimizers["D"].zero_grad()
                dis_loss.backward()
                self.optimizers["D"].step()

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

    @staticmethod
    def discriminator_input(n_batch, generator, lat_space, from_dat):
        gen_in = lat_space(int(n_batch / 2))
        gen_out = generator(gen_in)
        dis_in = gen_out + from_dat(int(n_batch / 2))
        y = [0 for n in range(int(n_batch / 2))] + [1 for n in range(int(n_batch / 2))]
        return dis_in, y

    @staticmethod
    def generator_input(n_batch, generator, discriminator, lat_space):
        gen_in = lat_space(n_batch)
        gen_out = generator(gen_in)
        dis_out = discriminator(gen_out)
        y = [1 for n in range(n_batch)]
        return dis_out, y
