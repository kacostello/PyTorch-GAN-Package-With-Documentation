## About WassersteinGANTrainer.py

WassersteinGANTrainer has these attributes:
Import Statements:
import SuperTrainer
import ToTrain
import torch
import math
import torch.optim as optim
SuperTrainer and ToTrain are both different classes in the package, while the torch, math, and optim imports are necessary imports that help to implement different functions and optimization algorithms in PyTorch.

class WassersteinGANTrainer(SuperTrainer.SuperTrainer):
This is the class definition that creates a class to train the Wasserstein GAN. In this class, the Generator and the Discriminator are torch model objects. The Latent Space function (Latent_space_function) is a function which returns an array of n points from the real dataset.

optim.RMSprop(generator.parameters(), g_lr):
This function implements the RMS prop algorithm for the Generator. More information can be found here. 

optim.RMSprop(discriminator.parameters(), d_lr):
This function implements the RMS prop algorithm for the Discriminator. More information can be found here.


	# Both input functions return the tuple (dis_in, labels)
	# generator_in returns (gen_out, labels) - this data is passed through D and used to train G
	# discriminator_in returns (dis_in, labels) - this is used to train D directly
	# For other GAN types: input functions can return whatever makes the most sense for your specific GAN
	# (so controllable GAN, for instance, might want to return a classification vector as well)

	def train(self, n_epochs, n_batch):
        all_dists = []
        for epoch in range(n_epochs):
            tt = self.totrain.next(self)  # Determine which model to train - sw will either be "D" or "G"

            dis_in, y = self.in_functions[tt](n_batch)
            if tt == "G":  # If we're training the generator, we should temporarily put the discriminator in eval mode
                self.models["D"].eval()
            mod_pred = self.models["D"](dis_in)
            self.models["D"].train()
            mod_loss = self.loss_functions[tt](mod_pred, y)

            # Logging for visualizers
            self.stats["losses"][tt].append(mod_loss.item())
            self.stats["epochs_trained"][tt] += 1

            y_flat = y.cpu().numpy().flatten()  # Calculate fPr, recall, precision
            mod_pred_flat = mod_pred.cpu().detach().numpy().flatten()
            fP = 0
            fN = 0
            tP = 0
            tN = 0
            for i in range(len(y_flat)):
                if y_flat[i] == 0:
                    if mod_pred_flat[i] > self.d_thresh:
                        fP += 1
                    else:
                        tN += 1
                else:
                    if mod_pred_flat[i] > self.d_thresh:
                        tP += 1
                    else:
                        fN += 1

            if fP + tN > 0:
                self.stats["d_fpr"].append(fP / (fP + tN))
            if tP + fP > 0:
                self.stats["d_precision"].append(tP / (tP + fP))
            if tP + fN > 0:
                self.stats["d_recall"].append(tP / (tP + fN))

            # Pytorch training steps
            self.optimizers[tt].zero_grad()
            mod_loss.backward()
            self.optimizers[tt].step()

            if tt == "D":
                for p in self.models["D"].parameters():
                    p.data.clamp_(-0.01, 0.01)

            w_dists = self.all_Wasserstein_dists(self.eval_generator(self.latent_space(256)), self.dataset(256))
            w_dist_mean = torch.mean(w_dists)
            all_dists.append(w_dist_mean)
            # print(w_dist_mean)
        print(len(all_dists))
        plt.title('Wasserstein GAN Training Over Time')
        plt.xlabel('Batches')
        plt.ylabel('Wasserstein Distance Mean')
        plt.plot(all_dists)
        plt.show()

The above function is the training loop to train the Wasserstein GAN. The train takes in self, the number of epochs, and the batch size as arguments. Next, there is a for loop that is repeated for each epoch in the total number of epochs. Inside of the loop, the input functions return the corresponding tuple. After returning the corresponding tuple, the function then checks if the Generator or the Discriminator is being trained. If the Generator is being trained, then the Discriminator should be put in eval mode for a temporary amount of time. Otherwise, the training continues and the loss function is generated. The logging for visualizers also takes place. After the logging, there are some PyTorch training steps to occur, inducing having the self optimizers call the zero_grad() function. Then, there is some backward propagation and stepping. If the Discriminator is being trained, some of the data gets clamped.

  	# This function evaluates the Generator.
	
	def eval_generator(self, in_dat):
        return self.eval("G", in_dat)


	#This function evaluates the Discriminator.

	def eval_discriminator(self, in_dat):
        return self.eval("D", in_dat)

 
	# This function obtains the loss function for the Generator.

	def get_g_loss_fn(self):
        return self.loss_functions["G"]


	# This function obtains the optimizer for the Generator.

	def get_g_opt_fn(self):
        return self.optimizers["G"]


	# This function obtains the loss function for the Discriminator.
	
	def get_d_loss_fn(self):
        return self.loss_functions["D"]


	# This function obtains the optimizer for the Discriminator.

	def get_d_opt_fn(self):
        return self.optimizers["D"]

 
	# This function calculates the loss by each epoch for the Generator.

	def loss_by_epoch_g(self):
        self.loss_by_epoch("G")

 
	# This function calculates the loss by each epoch for the Discriminator.
	
	def loss_by_epoch_d(self):
        self.loss_by_epoch("D")

  
	# This function returns the input from the Discriminator.

	def discriminator_input(self, n_batch):
        gen_in = self.latent_space(math.ceil(n_batch / 2))
        self.models["G"].eval()
        gen_out = self.models["G"](gen_in)
        self.models["G"].train()
        dis_in = torch.cat((gen_out, self.dataset(int(n_batch / 2))))
        y = torch.tensor([[0] for n in range(math.ceil(n_batch / 2))] + [[1] for n in range(int(n_batch / 2))]).float()
        return dis_in, y


	# This function returns the input from the Generator.

	def generator_input(self, n_batch):
        gen_in = self.latent_space(n_batch)
        gen_out = self.models["G"](gen_in)
        y = torch.tensor([[1] for n in range(n_batch)]).float()
        return gen_out, y
