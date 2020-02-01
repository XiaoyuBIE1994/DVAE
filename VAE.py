#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import torch
from torch import nn

class VAE(nn.Module):

    def __init__(self, input_dim = None,
                       latent_dim = None,
                       hidden_dim_encoder = None,
                       batch_size = None,
                       activation = None):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim_encoder = hidden_dim_encoder
        self.hidden_dim_decoder = list(reversed(hidden_dim_encoder))
        self.batch_size = batch_size
        self.activation = activation
        self.model = None
        self.history = None

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.MoudleList()

        self.output_dim = self.input_dim

        self.build()
        
    def build(self):
        # Define the encode layers (without activation)
        for n, dim in enumerate(self.hidden_dim_encoder):
            if n == 0:
                self.encoder_layers.append(nn.Linear(self.input_dim, dim))
            else:
                self.encoder_layers.append(nn.Linear(self.hidden_dim_encoder[n-1], dim))

        # Define the bottleneck layer (the latent variable space)
        self.latent_mean_layer = nn.Linear(self.hidden_dim_encoder[-1], self.latent_dim)
        self.latent_logvar_layer = nn.Linear(self.hidden_dim_encoder[-1], self.latent_dim)

        # Define the decode layers (without activation)
        for n, dim in enumerate(self.hidden_dim_decoder):
            if n == 0:
                self.decoder_layers.append(nn.Linear(self.latent_dim, dim))
            else:
                self.decoder_layers.append(nn.Linear(self.hidden_dim_decoder[n-1], dim))

        # Output
        self.output_layer = nn.Linear(self.hidden_dim_decoder[-1], self.output_dim)

    def encode(self, x):
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

    def print_model(self):
        print("----- Encoder -----")
        for layer in self.encoder_layers:
            print(layer)
            print(self.activation)
        
        print("----- Bottleneck -----")
        print(self.latent_mean_layer)
        print(self.latent_logvar_layer)
        
        print("----- Decoder -----")
        for layer in self.decoder_layers:
            print(layer)
            print(self.activation)
        print(self.output_layer)
    