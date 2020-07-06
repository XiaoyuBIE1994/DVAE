#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt

The code in this file is based on:
- “Deep Kalman Filter” arXiv, 2015, Rahul G.Krishnan et al.
- "Structured Inference Networks for Nonlinear State Space Models" AAAI, 2017, Rahul G.Krishnan et al.

DMM refers to the deep Markov model in the second paper,
with only forward RNN in inference, it's a Deep Kalman Filter (DKF),
with only backwrad RNN in inference, it's a Deep Kalman Smoother (DKS),
with bi-directional RNN in inference, it's a ST-LR

To have consistant expression comparing with other models we change some functions' name:
Emissino Function -> Generation
Gated Transition Fucntion -> Prior
"""


from torch import nn
import torch
from collections import OrderedDict


class DMM(nn.Module):

    def __init__(self, x_dim, z_dim=16, activation='tanh',
                 dense_x_g=[],
                 dim_RNN_g=128, num_RNN_g=1, bidir_g=False,
                 dense_z_x=[128,128],
                 dropout_p = 0, device='cpu'):

        super().__init__()
        ### General parameters  
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
        ### Inference
        self.dense_x_g = dense_x_g
        self.dim_RNN_g = dim_RNN_g
        self.num_RNN_g = num_RNN_g
        self.bidir_g = bidir_g
        ### Generation x
        self.dense_z_x = dense_z_x

        self.build()

    def build(self):
    
        ###################
        #### Inference ####
        ###################
        # 1. x_t to g_t
        dic_layers = OrderedDict()
        if len(self.dense_x_g) == 0:
            dim_x_g = self.x_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_x_g = self.dense_x_g[-1]
            for n in range(len(self.dense_x_g)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x_g[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_g[n-1], self.dense_x_g[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_x_g = nn.Sequential(dic_layers)
        self.rnn_g = nn.LSTM(dim_x_g, self.dim_RNN_g, self.num_RNN_g, bidirectional=self.bidir_g)
        # 2. g_t and z_tm1 to z_t
        dic_layers = OrderedDict()
        dic_layers['linear'] = nn.Linear(self.z_dim, self.dim_RNN_g)
        dic_layers['activation'] = nn.Tanh()
        self.mlp_z_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(self.dim_RNN_g, self.z_dim)
        self.inf_logvar = nn.Linear(self.dim_RNN_g, self.z_dim)

        ######################
        #### Generation z ####
        ######################
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
        self.prior_logvar = nn.Sequential(nn.ReLU(),
                                          nn.Linear(self.z_dim, self.z_dim),
                                          nn.Softplus())
        
        ######################
        #### Generation x ####
        ######################
        dic_layers = OrderedDict()
        if len(self.dense_z_x) == 0:
            dim_z_x = self.z_dim
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
        self.gen_logvar = nn.Linear(dim_z_x, self.y_dim)


    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return torch.addcmul(mean, eps, std)


    def inference(self, x):
        
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # Create variable holder and send to GPU if needed
        z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_t = torch.zeros((batch_size, self.z_dim)).to(self.device)

        # 1. x_t to g_t, g_t and z_tm1 to z_t
        x_g = self.mlp_x_g(x)
        if self.bidir_g:
            g, _ = self.rnn_g(x_g)
            g = g.view(seq_len, batch_size, 2, self.dim_RNN_g)
            g_forward = g[:,:,0,:]
            g_backward = g[:,:,1,:]
            for t in range(seq_len):
                g_t = (self.mlp_z_z(z_t) + g_forward[t,:,:] + g_backward[t,:,:]) / 3
                z_mean[t,:,:] = self.inf_mean(g_t)
                z_logvar[t,:,:] = self.inf_logvar(g_t)
                z_t = self.reparameterization(z_mean[t,:,:], z_logvar[t,:,:]) 
                z[t,:,:] = z_t
        else:
            g, _ = self.rnn_g(torch.flip(x_g, [0]))
            g = torch.flip(g, [0])
            for t in range(seq_len):
                g_t = (self.mlp_z_z(z_t) + g[t,:,:]) / 2
                z_mean[t,:,:] = self.inf_mean(g_t)
                z_logvar[t,:,:] = self.inf_logvar(g_t)
                z_t = self.reparameterization(z_mean[t,:,:], z_logvar[t,:,:])
                z[t,:,:] = z_t

        return z, z_mean, z_logvar
    
    
    def generation_z(self, z_tm1):

        gate = self.mlp_gate(z_tm1)
        z_prop = self.mlp_z_prop(z_tm1)
        z_mean_p = (1 - gate) * self.prior_mean(z_tm1) + gate * z_prop
        z_var_p = self.prior_logvar(z_prop)
        z_logvar_p = torch.log(z_var_p) # consistant with other models

        return z_mean_p, z_logvar_p


    def generation_x(self, z):
        
        # 1. z_t to y_t
        log_y = self.mlp_z_x(z)
        log_y = self.gen_logvar(log_y)
        y = torch.exp(log_y)
        
        return y
    

    def forward(self, x):
        
        # train input: (batch_size, x_dim, seq_len)
        # test input:  (seq_len, x_dim) 
        # need input:  (seq_len, batch_size, x_dim)
        if len(x.shape) == 3:
            x = x.permute(-1, 0, 1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)

        # main part 
        z, z_mean, z_logvar = self.inference(x)
        z_mean_p, z_logvar_p = self.generation_z(z)
        y = self.generation_x(z)
        
        # y/z dimension:    (seq_len, batch_size, y/z_dim)
        # output dimension: (batch_size, y/z_dim, seq_len)
        z = torch.squeeze(z)
        y = torch.squeeze(y)
        z_mean = torch.squeeze(z_mean)
        z_logvar = torch.squeeze(z_logvar)
        z_mean_p = torch.squeeze(z_mean_p)
        z_logvar_p = torch.squeeze(z_logvar_p)

        if len(z.shape) == 3:
            z = z.permute(1,-1,0)
        if len(y.shape) == 3:    
            y = y.permute(1,-1,0)

        return y, z_mean, z_logvar, z_mean_p, z_logvar_p, z


    def get_info(self):
        
        info = []
        info.append("----- Inference -----")
        info.append('>>>> x_t to g_t')
        for layer in self.mlp_x_g:
            info.append(layer)
        info.append(self.rnn_g)
        info.append('>>>> mlp for z_tm1')
        info.append(self.mlp_z_z)

        info.append("----- Bottleneck -----")
        info.append(self.inf_mean)
        info.append(self.inf_logvar)

        info.append("----- Generation x -----")
        for layer in self.mlp_z_x:
            info.append(layer)
        info.append(self.gen_logvar)
        
        info.append("----- Generation z -----")
        info.append('>>>> Gating unit')
        for layer in self.mlp_gate:
            info.append(layer)
        info.append('>>>> Proposed mean')
        for layer in self.mlp_z_prop:
            info.append(layer)
        info.append('>>>> Prior mean and logvar')
        info.append(self.prior_mean)
        info.append(self.prior_logvar)

        return info


if __name__ == '__main__':
    x_dim = 513
    z_dim = 16
    device = 'cpu'
    dmm = DMM(x_dim=x_dim, z_dim=z_dim).to(device)

    x = torch.ones((2,513,3))
    y, z_mean, z_logvar, z_mean_p, z_logvar_p, z = dmm.forward(x)

    def loss_function(recon_x, x, mu, logvar, mu_prior, logvar_prior):
        recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
        KLD = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), logvar_prior.exp()))
        return recon + KLD

    loss = loss_function(y,x,z_mean,z_logvar,z_mean_p,z_logvar_p)/6

    print(loss)