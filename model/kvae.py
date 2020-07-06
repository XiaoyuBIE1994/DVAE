#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt

The code in this file is based on:
- “A Disentanagled Recognition and Nonlinear Dynamics Model for Unsupervised Learning” NIPS, 2017, Macro Fraccaro et al.

Not include:
- different learning target (alpha first, then KF params, finally total params)
- no imputation
"""

from torch import nn
import torch
from collection import OrderedDict

class KVAE(nn.Module):

    def __init__(self, x_dim, z_dim=16, activation='tanh',
                 dropout_p=0, device='cpu'):

        super().__init__()
        ## General parameters
        self.x_dim = x_dim
        self.y_dim = x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemError('Wrong activation type')
        self.device = device
        # VAE
        
        # LGSSM


        self.build()

    def build():
        pass

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)


    def inference():
        pass


    def generation():
        pass


    def kf_smoother(self):
        pass

    
    def forward():
        pass

    def get_info():
        pass


if __name__ == '__main__':

    x_dim = 513
    z_dim = 16
    device = 'cpu'

    kvae = KVAE(x_dim, z_dim).to(device)
    
    x = torch.ones([2,513,3])
    y, z_mean, z_logvar, z_mean_p, z_logvar_p, z = kvae.forward(x)

    def loss_function(recon_x, x, mu, logvar, mu_prior, logvar_prior):
        recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
        KLD = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), logvar_prior.exp()))
        return recon + KLD

    loss = loss_function(y,x,z_mean,z_logvar,z_mean_p,z_logvar_p)/6

    print(loss)
