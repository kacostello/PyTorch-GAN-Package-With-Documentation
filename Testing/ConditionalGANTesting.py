from SimpleGANTrainer import SimpleGANTrainer
from ToTrain import TwoFiveRule
import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import GetData

def lat_space(batch_size):
    data = torch.rand(batch_size, num_inputs)
    labels = torch.randint(0, num_classes, size=(batch_size, 1))
    labels = func.one_hot(labels, num_classes=num_classes)
    labels = labels.reshape(batch_size, num_classes)
    return torch.cat((data, labels), 1)

def batch_from_data(batch_size=16):
    # Obtain some entries
    num_rows = wine_data.shape[0]
    rand_indices = np.random.choice(num_rows, size=batch_size, replace=False)
    data = wine_data[rand_indices, :]

    # Get related labels
    labels = wine_labels[rand_indices]
    labels = to_one_hot(labels)

    # Combine data and labels
    data_labels = np.concatenate((data, labels), axis=1)

    return torch.tensor(data_labels)

def to_one_hot(labels):
    oneHot = np.zeros((labels.shape[0], num_classes)).astype(int)
    oneHot[np.arange(labels.size), labels.astype(int)] = 1
    return oneHot

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.embedding_label = nn.Embedding(num_classes, 1)
        self.dense_layer = nn.Linear(num_inputs + num_classes, num_inputs)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        data = x[:, 0:num_inputs]
        labels = x[:, num_inputs:].int()
        gan_batch_size = labels.size(dim=0)
        gen_input = torch.cat((data, self.embedding_label(labels).reshape(gan_batch_size, num_classes)), 1)
        new_data = self.activation(self.dense_layer(gen_input))
        return torch.cat((new_data, labels), 1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.embedding_label = nn.Embedding(num_classes, 1)
        self.dense = nn.Linear(num_inputs + num_classes, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        data = x[:, 0:num_inputs]
        labels = x[:, num_inputs:].int()
        gan_batch_size = labels.size(dim=0)
        dis_input = torch.cat((data, self.embedding_label(labels).reshape(gan_batch_size, num_classes)), 1)
        return self.activation(self.dense(dis_input.float()))

# Data imports
wine_data, wine_labels = GetData.wineData()
num_inputs = np.shape(wine_data)[1]
num_classes = 11

gen = Generator()
dis = Discriminator()

gen_opt = torch.optim.Adam(gen.parameters(), lr=0.001)
dis_opt = torch.optim.Adam(dis.parameters(), lr=0.001)

gen_loss = nn.BCELoss()
dis_loss = nn.BCELoss()

sw = TwoFiveRule()

gan = SimpleGANTrainer(gen, dis, lat_space, batch_from_data, gen_loss, dis_loss, gen_opt, dis_opt, sw)
gan.train(7000, 16)
output = gan.eval_generator(lat_space(16))
out_data = output[:, 0:num_inputs]
out_labels = output[:, num_inputs:].int()
out_labels = torch.argmax(out_labels, dim=1)
print(out_data)
print(out_labels)
gan.loss_by_epoch_d()

