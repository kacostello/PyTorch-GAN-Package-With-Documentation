# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:31:33 2021

@author: Kyle Costello
"""

#Import Statments and GPU Logic

import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Get statisitcs about the data
    # Data Volume
    # Data Dimensions
    # Image or numerical?
    
# Hyperparameters
    # lr
    # Batch Size
    # D/G ratio
    
class Hyperparameter:
    batch_size = 64
    epochs = 20
    latent_size = 32
    n_discriminator = 5
    discriminator_size = 1024
    generator_size = 1024
    discriminator_hidden_size = 1024
    gp_lambda = 10.0
        
hp = Hyperparameter()
    
#Generator
    # __init__()
    # forward()
    # get_noise()
    # get_loss()
    # Save Model()
    # Load Model()
    # initialize
    # optimizer
def get_generator_block(X_dim, h_dim):
    torch.nn.Sequential(
        torch.nn.Linear(X_dim, h_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(h_dim, 1),
)
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_embedding = nn.Sequential(
            nn.Linear(hp.latent_size, hp.generator_size),
        )
        self.tcnn = nn.Sequential(
        nn.ConvTranspose2d(hp.generator_size, hp.generator_size, 4, 1, 0),
        nn.BatchNorm2d(hp.generator_size),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(hp.generator_size, hp.generator_size / 2, 3, 2, 1),
        nn.BatchNorm2d(hp.generator_size / 2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(hp.generator_size // 2, hp.generator_size / 4, 4, 2, 1),
        nn.BatchNorm2d(hp.generator_size / 4),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(hp.generator_size / 4, 1, 4, 2, 1),
        nn.Tanh()
        )
        
        
    def forward(self, latent_space):
        vec_latent_space = self.latent_embedding(latent_space).reshape(-1, hp.generator_size, 1, 1)
        return self.tcnn(vec_latent_space)
    
# Discriminator
    # __init__()
    # forward()
    # get_loss()
    # Save Model()
    # Load Model()
    # initialize
    # optimizer
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cnn_net = nn.Sequential(
        nn.Conv2d(1, hp.discriminator_size / 4, 3, 2),
        nn.InstanceNorm2d(hp.discriminator_size / 4, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(hp.discriminator_size / 4, hp.discriminator_size / 2, 3, 2),
        nn.InstanceNorm2d(hp.discriminator_size / 2, affine=True),
        nn.LeakyReLU(0.2, inplace=True),   
        nn.Conv2d(hp.discriminator_size / 2, hp.discriminator_size, 3, 2),
        nn.InstanceNorm2d(hp.discriminator_size, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Flatten(),
        )
        self.critic_net = nn.Sequential(
        nn.Linear(hp.discriminator_size * 4, hp.discriminator_hidden_size),
        nn.LeakyReLU(0.2, inplace=True),   
        nn.Linear(hp.discriminator_hidden_size, 1),
        )
    
    def forward(self, image):
        cnn_features = self.cnn_net(image)
        return self.critic_net(cnn_features)
    
# Training Loop
    # Alternate training
    # Record training stats

# Compile Training Report
    # Send training stats and visualization to disc 
    
# Sample Data 
    # Generate a sample of fake data
    
# Evaluate generation quality
    # Sample Data and then visualize
    # Compile report and send to disc
