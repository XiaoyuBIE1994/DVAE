#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

The code in this file is based on:
- “Learning Stochastic Recurrent Networks” ICLR, 2015, Justin Bayer et al.

Note:
In the original paper, the input of inference is ￼x_tm1 in order to do prediction
Here, we use x_t￼ instead to have a comparable experiment with other models
"""



from torch import nn
import torch
from collections import OrderedDict


def build_STORN(cfg, device='cpu'):

    ### Load parameters for STORN
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Inference
    dense_x_g = [] if cfg.get('Network', 'dense_x_g') == '' else [int(i) for i in cfg.get('Network', 'dense_x_g').split(',')]
    dim_RNN_g = cfg.getint('Network', 'dim_RNN_g')
    num_RNN_g = cfg.getint('Network', 'num_RNN_g')
    dense_g_z = [] if cfg.get('Network', 'dense_g_z') == '' else [int(i) for i in cfg.get('Network', 'dense_g_z').split(',')]
    # Generation
    dense_z_h = [] if cfg.get('Network', 'dense_z_h') == '' else [int(i) for i in cfg.get('Network', 'dense_z_h').split(',')]
    dense_xtm1_h = [] if cfg.get('Network', 'dense_xtm1_h') == '' else [int(i) for i in cfg.get('Network', 'dense_xtm1_h').split(',')]
    dim_RNN_h = cfg.getint('Network', 'dim_RNN_h')
    num_RNN_h = cfg.getint('Network', 'num_RNN_h')
    dense_h_x = [] if cfg.get('Network', 'dense_h_x') == '' else [int(i) for i in cfg.get('Network', 'dense_h_x').split(',')]
    
    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')

    # Build model
    model = STORN(x_dim=x_dim, z_dim=z_dim, activation=activation,
                 dense_x_g=dense_x_g, dense_g_z=dense_g_z,
                 dim_RNN_g=dim_RNN_g, num_RNN_g=num_RNN_g,
                 dense_z_h=dense_z_h, dense_xtm1_h=dense_xtm1_h,
                 dense_h_x=dense_h_x,
                 dim_RNN_h=dim_RNN_h, num_RNN_h=num_RNN_h,
                 dropout_p=dropout_p, beta=beta, device=device).to(device)

    return model

    

class STORN(nn.Module):

    def __init__(self, x_dim, z_dim=16, activation = 'tanh',
                 dense_x_g=[128], dense_g_z=[128],
                 dim_RNN_g=128, num_RNN_g=1,
                 dense_z_h=[128], dense_xtm1_h=[128], dense_h_x=[128],
                 dim_RNN_h=128, num_RNN_h=1,
                 dropout_p = 0, beta=1, device='cpu'):

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
        ### Inference
        self.dense_x_g = dense_x_g
        self.dense_g_z = dense_g_z
        self.dim_RNN_g = dim_RNN_g
        self.num_RNN_g = num_RNN_g
        ### Generation x
        self.dense_z_h = dense_z_h
        self.dense_xtm1_h = dense_xtm1_h
        self.dense_h_x = dense_h_x
        self.dim_RNN_h = dim_RNN_h
        self.num_RNN_h = num_RNN_h
        ### Beta-loss
        self.beta = beta

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
        # 2. g_t, forward recurrence
        self.rnn_g = nn.LSTM(dim_x_g, self.dim_RNN_g, self.num_RNN_g)
        # 3. g_t to z_t
        dic_layers = OrderedDict()
        if (len(self.dense_g_z)) == 0:
            dim_g_z = self.dim_RNN_g
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_g_z = self.dense_g_z[-1]
            for n in range(len(self.dense_g_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN_g, self.dense_g_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_g_z[n-1], self.dense_g_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_g_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_g_z, self.z_dim)
        self.inf_logvar = nn.Linear(dim_g_z, self.z_dim)
   
        ######################
        #### Generation x ####
        ######################
        # 1. z_t to h_t
        dic_layers = OrderedDict()
        if len(self.dense_z_h) == 0:
            dim_z_h = self.z_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_z_h = self.dense_z_h[-1]
            for n in range(len(self.dense_z_h)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.z_dim, self.dense_z_h[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_z_h[n-1], self.dense_z_h[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_z_h = nn.Sequential(dic_layers)
        # 2. x_tm1 to h_t
        dic_layers = OrderedDict()
        if len(self.dense_xtm1_h) == 0:
            dim_xtm1_h = self.x_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_xtm1_h = self.dense_xtm1_h[-1]
            for n in range(len(self.dense_xtm1_h)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_xtm1_h[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_xtm1_h[n-1], self.dense_xtm1_h[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_xtm1_h = nn.Sequential(dic_layers)
        # 3. h_t, forward recurrence
        self.rnn_h = nn.LSTM(dim_z_h+dim_xtm1_h, self.dim_RNN_h, self.num_RNN_h)
        # 4. h_t to x_t
        dic_layers = OrderedDict()
        if len(self.dense_h_x) == 0:
            dim_h_x = self.dim_RNN_h
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_h_x = self.dense_h_x[-1]
            for n in range(len(self.dense_h_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN_h, self.dense_h_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_h_x[n-1], self.dense_h_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_h_x = nn.Sequential(dic_layers)
        self.gen_out = nn.Linear(dim_h_x, self.y_dim) 


    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return torch.addcmul(mean, eps, std)
        
        
    def inference(self, x):

        # 1. From x_t to g_t
        x_g = self.mlp_x_g(x)
        g, _ = self.rnn_g(x_g)

        # 2. From g_t to z_t
        g_z = self.mlp_g_z(g)
        z_mean = self.inf_mean(g_z)
        z_logvar = self.inf_logvar(g_z)
        z = self.reparameterization(z_mean, z_logvar)
        # z = z_mean

        return z, z_mean, z_logvar


    def generation_x(self, z, x_tm1):
        
        
        # 1. From z_t and x_tm1 to h_t
        z_h = self.mlp_z_h(z)
        xtm1_h = self.mlp_xtm1_h(x_tm1)
        zx_h = torch.cat((z_h, xtm1_h), -1)
        h, _ = self.rnn_h(zx_h)

        # 2. From h_t to y_t
        h_x = self.mlp_h_x(h)
        y = self.gen_out(h_x)

        return y


    def forward(self, x):

        # need input:  (seq_len, batch_size, x_dim)
        _, batch_size, x_dim = x.shape

        # main part
        self.z, self.z_mean, self.z_logvar = self.inference(x)
        # print(self.z_mean[62])
        x_0 = torch.zeros(1, batch_size, x_dim).to(self.device)
        x_tm1 = torch.cat((x_0, x[:-1,:,:]), 0)
        y = self.generation_x(self.z, x_tm1)

        self.z_mean_p = torch.zeros_like(self.z_mean)
        self.z_logvar_p = torch.zeros_like(self.z_logvar)
        # print(y[62, 0, -25:-20].exp())
        return y

        
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

        info.append("----- Generation x -----")
        for layer in self.mlp_z_h:
            info.append(str(layer))
        for layer in self.mlp_xtm1_h:
            info.append(str(layer))
        info.append(self.rnn_h)
        for layer in self.mlp_h_x:
            info.append(str(layer))
        info.append('Output: ' + str(self.gen_out))

        info.append("----- Generation z -----")
        info.append('>>>> zero-mean, unit-variance Gaussian')

        return info


if __name__ == '__main__':
    x_dim = 513
    device = 'cpu'
    storn = STORN(x_dim=x_dim).to(device)
    model_info = storn.get_info()

    x = torch.ones((2,513,3))
    y, z_mean, z_logvar, z_mean_p, z_logvar_p, z = storn.forward(x)

    def loss_function(recon_x, x, mu, logvar, mu_prior, logvar_prior):
        recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
        KLD = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), logvar_prior.exp()))
        return recon + KLD

    loss = loss_function(y,x,z_mean,z_logvar,z_mean_p,z_logvar_p)/6

    print(loss)

