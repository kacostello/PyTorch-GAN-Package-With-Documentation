from WassersteinGANTrainer import WassersteinGANTrainer
from ToTrain import TwoFiveRule
import torch
import torch.nn as nn
import math
import numpy as np


def lat_space(batch_size, device="cpu"):
    return torch.randint(0, 2, size=(batch_size, 7), device=device).float()


def list_from_num(num):
    return [int(x) for x in list(bin(num))[2:]]


def batch_from_data(batch_size=16, device="cpu"):
    max_int = 128
    # Get the number of binary places needed to represent the maximum number
    max_length = int(math.log(max_int, 2))

    # Sample batch_size number of integers in range 0-max_int
    sampled_integers = np.random.randint(0, int(max_int / 2), batch_size)

    # create a list of labels all ones because all numbers are even
    labels = [1] * batch_size

    # Generate a list of binary numbers for training.
    data = [list_from_num(int(x * 2)) for x in sampled_integers]
    data = [([0] * (max_length - len(x))) + x for x in data]

    return torch.tensor(data, device=device).float()


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(7, 7)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(7, 4),
            nn.ReLU(inplace = True),
            nn.Linear(4, 1))

    def forward(self, x):
        return self.disc(x)


gen = Generator()
dis = Discriminator()

def genLoss(prediction, label):
    return -torch.mean(prediction)

def disLoss(prediction, label):
    pred_fake = prediction[label == 0]
    pred_real = prediction[label == 1]
    return -(torch.mean(pred_real) - torch.mean(pred_fake))

gen_loss = genLoss
dis_loss = disLoss

sw = TwoFiveRule()

device = "cuda"

gan = WassersteinGANTrainer(gen, dis, lat_space, batch_from_data, gen_loss, dis_loss, 0.0001, 0.0002, device, sw)
gan.train(1000, 2000)
print(gan.eval_generator(lat_space(16, device)))
