#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt

The code in this file is based on:
- “Deep Kalman Filter” arXiv, 2015, Rahul G.Krishnan et al.
- "Structured Inference Networks for Nonlinear State Space Models" AAAI, 2017, Rahul G.Krishnan et al.

DKS refers to the deep kalman smoother in the second paper, it is the one with the inference model
that respects the stucture of the true posterior 

To have consistant expression comparing with other models we change some functions' name:
Emissino Function -> Generation
Gated Transition Fucntion -> Prior
"""


from torch import nn
from torch.nn import functional as F
import numpy as np
import torch
from collections import OrderedDict


class DKS(nn.Module):

    def __init__(self, x_dim, z_dim=16, activation='tanh',
                 dense_z_x=[128,128],
                 dim_RNN_g=128, num_RNN_g=1, bidir_g=False,
                 dense_z_z=[128,128],
                 dropout_p = 0, device='cpu'):

        super().__init__()
        ### General parameters for storn        
        self.x_dim = x_dim
        self.y_dim = x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        self.device = device
        ### Generation
        self.dense_z_x = dense_z_x
        ### Inference 
        self.dim_RNN_g = dim_RNN_g
        self.num_RNN_g = num_RNN_g
        self.bidir_g = bidir_g
        ### Prior
        self.dense_z_z = dense_z_z

        self.build()

    def build(self):
        
        ####################
        #### Generation ####
        ####################
        dic_layers = OrderedDict()
        for n in range(len(self.dense_z_x)):
            if n == 0:
                dic_layers['linear'+str(n)] = nn.Linear(self.z_dim, self.dense_z_x[n])
            else:
                dic_layers['linear'+str(n)] = nn.Linear(self.dense_z_x[n-1], self.dense_z_x[n])
            dic_layers['activation'+str(n)] = self.activation
            dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_z_x = nn.Sequential(dic_layers)
        self.gen_logvar = nn.Linear(self.dense_z_x[-1], self.y_dim)

        ###################
        #### Inference ####
        ###################
        # 1. RNN of g_t
        self.rnn_g = nn.LSTM(self.x_dim, self.dim_RNN_g, self.num_RNN_g, bidirectional=self.bidir_g)
        # 2. dense layer of z_tm1
        self.mlp_z_z = nn.Linear(self.z_dim, self.dim_RNN_g)
        # 3. Infer z
        self.inf_mean = nn.Linear(self.dim_RNN_g, self.z_dim)
        self.inf_logvar = nn.Linear(self.dim_RNN_g, self.z_dim)

        ###############
        #### Prior ####
        ###############
        # 1. Gating Unit
        dic_layers = OrderedDict()
        dic_layers['linear1'] = nn.Linear(self.z_dim, self.z_dim)
        dic_layers['ReLU'] = nn.ReLU()
        dic_layers['linear2'] = nn.Linear(self.z_dim, self.z_dim)
        dic_layers['Sigmoid'] = nn.Sigmoid()
        self.mlp_gate = nn.Sequential(dic_layers)
        # 2. Proposed mean
        dic_layers = OrderedDict()
        dic_layers['linear1'] = nn.Linear(self.z_dim, self.z_dim)
        dic_layers['ReLU'] = nn.ReLU()
        dic_layers['linear2'] = nn.Linear(self.z_dim, self.z_dim)
        self.mlp_z_prop = nn.Sequential(dic_layers)
        # 3. Prior
        self.prior_mean = nn.Linear(self.z_dim, self.z_dim)
        self.prior_logvar = nn.Linear(self.z_dim, self.z_dim)
    

    def generation(self, z):
        
        z_x = self.mlp_z_x(z)
        log_y = self.gen_logvar(z_x)
        y = torch.exp(log_y)
        
        return y
    
    
    def prior(self, z_tm1):

        gate = self.mlp_gate(z_tm1)
        z_prop = self.mlp_z_prop(z_tm1)
        mean_prior = (1 - gate) * self.prior_mean(z_tm1) + gate * z_prop
        var_prior = F.softplus(self.prior_logvar(F.relu(z_prop)))
        logvar_prior = torch.log(var_prior) # consistant with other models

        return mean_prior, logvar_prior


    def inference(self, x):
        
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # 1. Create variable holder and send to GPU if needed
        z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        mean_prior = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        logvar_prior = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_t = torch.zeros((batch_size, self.z_dim)).to(self.device)

        if self.bidir_g:
            g, _ = self.rnn_g(x)
            g = g.view(seq_len, batch_size, 2, self.dim_RNN_g)
            g_forward = g[:,:,0,:]
            g_backward = g[:,:,1,:]
            for t in range(seq_len):
                g_t = torch.tanh(self.mlp_z_z(z_t) + g_forward[t,:,:] + g_backward[t,:,:]) / 3
                z_mean[t,:,:] = self.inf_mean(g_t)
                z_logvar[t,:,:] = self.inf_logvar(g_t)
                mean_prior[t,:,:], logvar_prior[t,:,:] = self.prior(z_t)
                z_t = self.reparatemize(z_mean[t,:,:], z_logvar[t,:,:]) # 为什么 z[t,:,:] = z_t 互换就会报错
                z[t,:,:] = z_t
        else:
            g, _ = self.rnn_g(torch.flip(x, [0]))
            g = torch.flip(g, [0])
            for t in range(seq_len):
                g_t = torch.tanh(self.mlp_z_z(z_t) + g[t,:,:]) / 2
                z_mean[t,:,:] = self.inf_mean(g_t)
                z_logvar[t,:,:] = self.inf_logvar(g_t)
                mean_prior[t,:,:], logvar_prior[t,:,:] = self.prior(z_t)
                z_t = self.reparatemize(z_mean[t,:,:], z_logvar[t,:,:])
                z[t,:,:] = z_t

        return z, z_mean, z_logvar, mean_prior, logvar_prior


    def reparatemize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)


    def forward(self, x):
        # train input: (batch_size, x_dim, seq_len)
        # test input:  (seq_len, x_dim) 
        # need input:  (seq_len, batch_size, x_dim)
        if len(x.shape) == 3:
            x = x.permute(-1, 0, 1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)

        # main part 
        z, z_mean, z_logvar, mean_prior, logvar_prior = self.inference(x)
        y = self.generation(z)

        # y/z dimension:    (seq_len, batch_size, y/z_dim)
        # output dimension: (batch_size, y/z_dim, seq_len)
        z = torch.squeeze(z)
        y = torch.squeeze(y)
        mean = torch.squeeze(z_mean)
        logvar = torch.squeeze(z_logvar)
        mean_prior = torch.squeeze(mean_prior)
        logvar_prior = torch.squeeze(logvar_prior)

        if len(z.shape) == 3:
            z = z.permute(1,-1,0)
        if len(y.shape) == 3:    
            y = y.permute(1,-1,0)

        return y, mean, logvar, mean_prior, logvar_prior, z 


    def get_info(self):
        info = []
        info.append("----- Inference -----")
        info.append('>>>> RNN g')
        info.append(self.rnn_g)
        info.append('>>>> dense layer of z_tm1')
        info.append(self.mlp_z_z)

        info.append("----- Bottleneck -----")
        info.append(self.inf_mean)
        info.append(self.inf_logvar)

        info.append("----- Generation -----")
        for layer in self.mlp_z_x:
            info.append(layer)
        info.append(self.gen_logvar)
        
        info.append("----- Prior -----")
        info.append('>>>> Gating unit')
        for layer in self.mlp_gate:
            info.append(layer)
        info.append('>>>> Proposed mean')
        for layer in self.mlp_z_prop:
            info.append(layer)
        info.append(self.prior_mean)
        info.append(self.prior_logvar)

        return info


if __name__ == '__main__':
    x_dim = 513
    z_dim = 16
    device = 'cpu'
    dks = DKS(x_dim=x_dim, z_dim=z_dim).to(device)
    # model_info = vrnn.get_info()
    # for i in model_info:
    #     print(i)

    x = torch.ones((2,513,3))
    y, mean, logvar, mean_prior, logvar_prior, z = dks.forward(x)
    def loss_function(recon_x, x, mu, logvar, mu_prior=None, logvar_prior=None):
        if mu_prior is None:
            mu_prior = torch.zeros_like(mu)
        if logvar_prior is None:
            logvar_prior = torch.zeros_like(logvar)
        recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
        KLD = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), logvar_prior.exp()))
        return recon + KLD

    print(loss_function(y,x,mean,logvar,mean_prior,logvar)/6)