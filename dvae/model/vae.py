#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

The code in this file is based on:
- "Auto-Encoding Variational Bayes" ICLR, 2014, Diederik P. Kingma and Max Welling
"""

from torch import nn
import torch
from collections import OrderedDict


def build_VAE(cfg, device='cpu'):

    ### Load parameters
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Inference and generation
    dense_x_z = [] if cfg.get('Network', 'dense_x_z') == '' else [int(i) for i in cfg.get('Network', 'dense_x_z').split(',')]

    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')

    # Build model
    model = VAE(x_dim=x_dim, z_dim=z_dim,
                dense_x_z=dense_x_z, activation=activation,
                dropout_p=dropout_p, beta=beta, device=device).to(device)

    return model


    
class VAE(nn.Module):

    '''
    VAE model class
    x: input data
    z: latent variables
    y: output data
    hidden_dim_enc: python list, the dimensions of hidden layers for encoder,
                        its reverse is the dimensions of hidden layers for decoder
    '''

    def __init__(self, x_dim=None, z_dim=16,
                 dense_x_z=[128], activation='tanh',
                 dropout_p = 0, beta=1, device='cpu'):

        super().__init__()
        ### General parameters for VAE
        self.x_dim = x_dim
        self.y_dim = self.x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        self.device = device
        ### Inference
        self.dense_x_z = dense_x_z
        ### Generation
        self.dense_z_x = list(reversed(dense_x_z))
        ### Beta-loss
        self.beta = beta
        
        self.build()
        

    def build(self):

        ###################
        #### Inference ####
        ###################
        # 1. x_t to z_t
        dic_layers = OrderedDict()
        if len(self.dense_x_z) == 0:
            dim_x_z = self.dim_x
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_x_z = self.dense_x_z[-1]
            for n in range(len(self.dense_x_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_z[n-1], self.dense_x_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_x_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_x_z, self.z_dim)
        self.inf_logvar = nn.Linear(dim_x_z, self.z_dim)

        ######################
        #### Generation x ####
        ######################
        # 1. z_t to x_t
        dic_layers = OrderedDict()
        if len(self.dense_z_x) == 0:
            dim_z_x = self.dim_z
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_z_x = self.dense_z_x[-1]
            for n in range(len(self.dense_z_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.z_dim, self.dense_z_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_z_x[n-1], self.dense_z_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_z_x = nn.Sequential(dic_layers)
        self.gen_out = nn.Linear(dim_z_x, self.y_dim)


    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return torch.addcmul(mean, eps, std) 


    def inference(self, x):

        x_z = self.mlp_x_z(x)
        z_mean = self.inf_mean(x_z)
        z_logvar = self.inf_logvar(x_z)
        z = self.reparameterization(z_mean, z_logvar)

        return z, z_mean, z_logvar

    
    def generation_x(self, z):

        z_x = self.mlp_z_x(z)
        y = self.gen_out(z_x)

        return y


    def forward(self, x):
        
        # need input:  (seq_len, batch_size, x_dim)
        # main part
        self.z, self.z_mean, self.z_logvar = self.inference(x)
        y = self.generation_x(self.z)
        
        self.z_mean_p = torch.zeros_like(self.z_mean)
        self.z_logvar_p = torch.zeros_like(self.z_logvar)

        return y


    def get_info(self):

        info = []
        info.append("----- Inference -----")
        for layer in self.mlp_x_z:
            info.append(str(layer))
        
        info.append("----- Bottleneck -----")
        info.append(str(self.inf_mean))
        info.append(str(self.inf_logvar))
        
        info.append("----- Decoder -----")
        for layer in self.mlp_z_x:
            info.append(str(layer))
        info.append(str(self.gen_out))

        return info
    
    
if __name__ == '__main__':
    x_dim = 513
    device = 'cpu'
    vae = VAE(x_dim = x_dim).to(device)
    model_info = vae.get_info()
    for i in model_info:
        print(i)
    