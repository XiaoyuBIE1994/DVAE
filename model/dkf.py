#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt

The code in this file is based on:
- “Deep Kalman Filter” arXiv, 2015, Rahul G.Krishnan et al.
- "Structured Inference Networks for Nonlinear State Space Models" AAAI, 2017, Rahul G.Krishnan et al.

"""


from torch import nn
import numpy as np
import torch
from collections import OrderedDict


class DKF(nn.Module):

    def __init__(self, x_dim, z_dim=32, batch_size=16,
                 device='cpu'):

        super().__init__()

    def build(self):
        pass


    def forward(self):
        pass

    def get_info(self):
        pass


if __name__ == '__main__':
    x_dim = 513
    z_dim = 16
    device = 'cpu'
    dkf = DKF(x_dim=x_dim, z_dim=z_dim).to(device)
    # model_info = vrnn.get_info()
    # for i in model_info:
    #     print(i)

    x = torch.ones((2,513,3))
    y, mean, logvar, mean_prior, logvar_prior, z = vrnn.forward(x)
    def loss_function(recon_x, x, mu, logvar, mu_prior=None, logvar_prior=None):
        if mu_prior is None:
            mu_prior = torch.zeros_like(mu)
        if logvar_prior is None:
            logvar_prior = torch.zeros_like(logvar)
        recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
        KLD = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), logvar_prior.exp()))
        return recon + KLD

    print(loss_function(y,x,mean,logvar,mean_prior,logvar)/6)