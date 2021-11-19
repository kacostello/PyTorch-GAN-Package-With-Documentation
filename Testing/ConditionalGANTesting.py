from SimpleGANTrainer import SimpleGANTrainer
from ToTrain import TwoFiveRule
import torch
import torch.nn as nn
import numpy as np
import GetData

def lat_space(batch_size):
    return torch.randint(0, 2, size=(batch_size, 7)).float()

def batch_from_data(batch_size=16):
    # Obtain some entries
    num_rows = wine_data.shape[0]
    rand_indices = np.random.choice(num_rows, size=batch_size, replace=False)
    data = wine_data[rand_indices, :]

    # Get related labels
    labels = wine_labels[rand_indices, :]

    return torch.tensor(data).float()


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(num_inputs, num_inputs)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(num_inputs, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))

# Data imports
wine_data, wine_labels = GetData.wineData()
num_inputs = np.shape(wine_data)[1]
num_classes = 11
print(num_classes)

gen = Generator()
dis = Discriminator()

gen_opt = torch.optim.Adam(gen.parameters(), lr=0.001)
dis_opt = torch.optim.Adam(dis.parameters(), lr=0.001)

gen_loss = nn.BCELoss()
dis_loss = nn.BCELoss()

sw = TwoFiveRule()

gan = SimpleGANTrainer(gen, dis, lat_space, batch_from_data, gen_loss, dis_loss, gen_opt, dis_opt, sw)
gan.train(7000, 16)
print(gan.eval_generator(lat_space(16)))
gan.loss_by_epoch_d()

