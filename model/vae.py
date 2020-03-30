#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt

The code in this file is based part on the source code of:
- Simon Legaive (simon.leglaive@centralesupelec.fr)
- in “A recurrent variational autoencoder for speech enhancement” ICASSP, 2020
"""

import torch
from torch import nn
# my_seed = 0
import numpy as np
# np.random.seed(my_seed)
import torch
# torch.manual_seed(my_seed)

class VAE(nn.Module):

    '''
    VAE model class
    x: input data
    z: latent variables
    y: output data
    hidden_dim_enc: python list, the dimensions of hidden layers for encoder,
                        its reverse is the dimensions of hidden layers for decoder
    '''

    def __init__(self, x_dim=None, z_dim=None,
                 hidden_dim_enc=None, batch_size=None,
                 activation=None):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hidden_dim_enc = hidden_dim_enc
        self.hidden_dim_dec = list(reversed(hidden_dim_enc))
        self.batch_size = batch_size
        self.activation = activation
        self.model = None
        self.history = None

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        self.y_dim = self.x_dim

        self.build()
        
    def build(self):
        # Define the encode layers (without activation)
        for n, dim in enumerate(self.hidden_dim_enc):
            if n == 0:
                self.encoder_layers.append(nn.Linear(self.x_dim, dim))
            else:
                self.encoder_layers.append(nn.Linear(self.hidden_dim_enc[n-1], dim))

        # Define the bottleneck layer (the latent variable space)
        self.latent_mean_layer = nn.Linear(self.hidden_dim_enc[-1], self.z_dim)
        self.latent_logvar_layer = nn.Linear(self.hidden_dim_enc[-1], self.z_dim)

        # Define the decode layers (without activation)
        for n, dim in enumerate(self.hidden_dim_dec):
            if n == 0:
                self.decoder_layers.append(nn.Linear(self.z_dim, dim))
            else:
                self.decoder_layers.append(nn.Linear(self.hidden_dim_dec[n-1], dim))

        # Output
        self.output_layer = nn.Linear(self.hidden_dim_dec[-1], self.y_dim)

    def encode(self, x):
        # print('shape of x: {}'.format(x.shape)) # used for debug only
        for layer in self.encoder_layers:
            x = self.activation(layer(x))

        mean = self.latent_mean_layer(x)
        logvar = self.latent_logvar_layer(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        for layer in self.decoder_layers:
            z = self.activation(layer(z))
        return torch.exp(self.output_layer(z))

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar, z

    def get_info(self):
        info = []
        info.append("----- Encoder -----")
        for layer in self.encoder_layers:
            info.append(str(layer))
            info.append(str(self.activation))
        
        info.append("----- Bottleneck -----")
        info.append(str(self.latent_mean_layer))
        info.append(str(self.latent_logvar_layer))
        
        info.append("----- Decoder -----")
        for layer in self.decoder_layers:
            info.append(str(layer))
            info.append(str(self.activation))
        info.append(str(self.output_layer))

        return info
    
def loss_function(recon_x, x, mu, logvar):
    recon = torch.sum(x/recon_x - torch.log(x/recon_x)-1)
    KLD = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp())
    return recon + KLD 

if __name__ == '__main__':
    x_dim = 513
    z_dim = 16
    hidden_dim_enc = [128]
    batch_size = 128
    activation = eval('torch.tanh')
    device = 'cpu'
    vae = VAE(x_dim = x_dim,
              z_dim = z_dim,
              hidden_dim_enc = hidden_dim_enc,
              batch_size = batch_size,
              activation = activation).to(device)
    model_finfo = vae.get_info()
    for i in model_finfo:
        print(i)
    