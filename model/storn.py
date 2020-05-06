#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt

The code in this file is based on:
- “Learning Stochastic Recurrent Networks” ICLR, 2015, Justin Bayer et al.

Note:
In the original paper, the input of inference is ￼x_tm1 in order to do prediction
Here, we use x_t￼ instead to have a faire comparison with other models
"""



from torch import nn
import numpy as np
import torch
from collections import OrderedDict


class STORN(nn.Module):

    def __init__(self, x_dim, z_dim=16, activation = 'tanh',
                 dense_zx_h=[128], dense_h_x=[128],
                 dim_RNN_h=128, num_RNN_h=1,             
                 dense_x_g=[128], dense_g_z=[128],
                 dim_RNN_g=128, num_RNN_g=1,
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
        self.dense_zx_h = dense_zx_h
        self.dense_h_x = dense_h_x
        self.dim_RNN_h = dim_RNN_h
        self.num_RNN_h = num_RNN_h
        ### Inference
        self.dense_x_g = dense_x_g
        self.dense_g_z = dense_g_z
        self.dim_RNN_g = dim_RNN_g
        self.num_RNN_g = num_RNN_g

        self.build()


    def build(self):

        ####################
        #### Generation ####
        ####################
        # 1. MLP from z_t and x_tm1 to h_t
        dic_layers = OrderedDict()
        for n in range(len(self.dense_zx_h)):
            if n == 0:
                dic_layers['linear'+str(n)] = nn.Linear(self.z_dim+self.x_dim, self.dense_zx_h[n])
            else:
                dic_layers['linear'+str(n)] = nn.Linear(self.dense_zx_h[n-1], self.dense_zx_h[n])
            dic_layers['activation'+str(n)] = self.activation
            dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_zx_h = nn.Sequential(dic_layers)
        # 2. RNN, output h_t
        self.rnn_h = nn.LSTM(self.dense_zx_h[-1], self.dim_RNN_h, self.num_RNN_h)
        # 3. MLP from h_t to y_t (log-variance of x_t)
        dic_layers = OrderedDict()
        for n in range(len(self.dense_h_x)):
            if n == 0:
                dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN_h, self.dense_h_x[n])
            else:
                dic_layers['linear'+str(n)] = nn.Linear(self.dense_h_x[n-1], self.dense_h_x[n])
            dic_layers['activation'+str(n)] = self.activation
            dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_h_x = nn.Sequential(dic_layers)
        self.gen_logvar = nn.Linear(self.dense_h_x[-1], self.y_dim)
        
        ###################
        #### Inference ####
        ###################
        # 1. MLP from x_t to RNN_g
        dic_layers = OrderedDict()
        for n in range(len(self.dense_x_g)):
            if n == 0:
                dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x_g[n])
            else:
                dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_g[n-1], self.dense_x_g[n])
            dic_layers['activation'+str(n)] = self.activation
            dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_x_g = nn.Sequential(dic_layers)
        # 2. RNN, output g_t
        self.rnn_g = nn.LSTM(self.dense_x_g[-1], self.dim_RNN_g, self.num_RNN_g)
        # 3. MLP from g_t to z_t
        dic_layers = OrderedDict()
        for n in range(len(self.dense_g_z)):
            if n == 0:
                dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN_g, self.dense_g_z[n])
            else:
                dic_layers['linear'+str(n)] = nn.Linear(self.dense_g_z[n-1], self.dense_g_z[n])
            dic_layers['activation'+str(n)] = self.activation
            dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_g_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(self.dense_g_z[-1], self.z_dim)
        self.inf_logvar = nn.Linear(self.dense_g_z[-1], self.z_dim)


    def generation(self, z, x_tm1):
        
        zx = torch.cat((z, x_tm1), 2)
        batch_size = zx.shape[1]

        # 1. From z_t and x_tm1 to h_t
        zx_h = self.mlp_zx_h(zx)
        h, _ = self.rnn_h(zx_h)

        # 2. From h_t to y_t
        h_x = self.mlp_h_x(h)
        log_y = self.gen_logvar(h_x)

        # 3. Transform log-variance to variance
        y = torch.exp(log_y)

        return torch.squeeze(y)


    def inference(self, x):

        # 1. From x_t to g_t
        g, _ = self.rnn_g(self.mlp_x_g(x))

        # 2. From g_t to z_t
        g_z = self.mlp_g_z(g)
        z_mean = self.inf_mean(g_z)
        z_logvar = self.inf_logvar(g_z)
        z = self.reparatemize(z_mean, z_logvar)

        return z, z_mean, z_logvar


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

        seq_len = x.shape[0]
        batch_size = x.shape[1]
        x_dim = x.shape[2]

        # create x_tm1
        x_0 = torch.zeros(1, batch_size, x_dim).to(self.device)
        x_tm1 = torch.cat((x_0, x[:-1,:,:]), 0)

        # main part
        z, z_mean, z_logvar = self.inference(x)
        y = self.generation(z, x_tm1)

        # y/z dimension:    (seq_len, batch_size, y/z_dim)
        # output dimension: (batch_size, y/z_dim, seq_len)
        z = torch.squeeze(z)
        y = torch.squeeze(y)
        mean = torch.squeeze(z_mean)
        logvar = torch.squeeze(z_logvar)

        if len(z.shape) == 3:
            z = z.permute(1,-1,0)
        if len(y.shape) == 3:    
            y = y.permute(1,-1,0)
        
        return y, mean, logvar, z


    def get_info(self):

        info = []
        info.append("----- Inference -----")
        for layer in self.mlp_x_g:
            info.append(str(layer))
        info.append(self.rnn_g)
        for layer in self.mlp_g_z:
            info.append(str(layer))
        
        info.append("----- Bottleneck -----")
        info.append('mean: ' + str(self.inf_mean))
        info.append('logvar: ' + str(self.inf_logvar))

        info.append("----- Generation -----")
        for layer in self.mlp_zx_h:
            info.append(str(layer))
        info.append(self.rnn_h)
        for layer in self.mlp_h_x:
            info.append(str(layer))
        info.append('Output: ' + str(self.gen_logvar))

        return info

if __name__ == '__main__':
    x_dim = 513
    device = 'cpu'
    storn = STORN(x_dim=x_dim).to(device)
    model_info = storn.get_info()

    x = torch.ones((2,513,3))
    y, mean, logvar, z = storn.forward(x)
    print(y.shape)
    print(mean.shape)
    print(logvar.shape)
    print(z.shape)

    def loss_vlb(recon_x, x, mu, logvar, mu_prior=None, logvar_prior=None, batch_size=32, seq_len=50):
        if mu_prior is None:
            mu_prior = torch.zeros_like(mu).to(device)
        if logvar_prior is None:
            logvar_prior = torch.zeros(logvar.shape).to(device)
        recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
        KLD = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), logvar_prior.exp()))
        return (recon + KLD) / (batch_size * seq_len)

    print(loss_vlb(y, x, mean, logvar))

