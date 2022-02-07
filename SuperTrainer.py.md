## About SuperTrainer.py

SuperTrainer has these attributes:

* ToTrain: ToTrain object, which determines which model gets trained on a given epoch. Can be as simple or complex as needed, and can be fed whatever input makes sense for the specific type of GAN. ToTrain objects all have a .next(trainer) function, which takes as an argument the trainer object and returns the string representation of the next model to train.
* Model names: Not explicitly stored, these are the strings used to refer to each model in each dictionary. in the SimpleGAN implementation, these are "D" and "G" for the discriminator and generator respectively.
* Models: Dictionary of the form {model name: pytorch model}. In the SimpleGAN implementation, this is {"D":discriminator, "G":generator}
* in_functions: Dictionary of the functions which returns the data to be fed into a model, and the labels for that data. Each function takes in whatever it needs to generate this information, and outputs [model_in, true]. In the SimpleGAN implementation, this is {"D":discriminator_input, "G":generator_input}. In the SimpleGAN implementation, both in_functions return data to be fed into the discriminator.
	* Example: generator_input takes in batch size, generator model object, and the latent space function. It creates a batch of the latent space, and feeds this batch to the generator. The generator wants to always fool the discriminator, so the labels are all 1. The function then returns [generator_out, labels]. After calling this function, the training loop feeds generator_out to the discriminator, and then feeds (discriminator_out, labels) to the generator's loss function. By default, the label for fake data is 0 and real data is 1.
* loss_functions: Dictionary of the PyTorch loss functions. These functions take in the (model_output, labels) pair, and returns the PyTorch object used for model training.
* opts: Dictionary of the PyTorch optimizers for each model. These are the optimizer objects used for training with PyTorch.
* SuperTrainer's Stats Dictionary: The stats dictionary mapping names of statistics with dictionaries containing those statistics. It is of the format {stat_name:stat_dict}
* losses: Dictionary of the record of the numerical loss values over every training epoch. After training is complete, it should be of the structure {model_name:[loss_epoch_0, loss_epoch_1, ..., loss_epoch_n]}. Used for the loss_by_epoch visualization.
* epochs_trained: Dictionary of the number of epochs each model has been trained. After training is complete, it should be of the structure {model_name:epochs_model_trained}

		# Training loop process for Simple GAN:
		

		for each epoch:
			x = self.totrain.next(self)  # x = "G" or "D", following a 2:5 rule
			if x == "G":
				dis_in, labels = in_functions[x](batch_size, generator, latent_space_function)  # generator_input()
			else:
				dis_in, labels = in_functions[x](batch_size, generator, latent_space_function, from_dataset_function)  # discriminator_input() 
			predicted = models["D"](dis_in)
			loss = loss_functions[x](predicted, labels)  # g_loss() or d_loss()
			# add numerical loss to losses[x]

			opts[x].zero_grad()  # g_opt or d_opt.zero_grad()
			loss.backward()
        		self.opts[x].step()
