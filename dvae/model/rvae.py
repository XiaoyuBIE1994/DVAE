#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

The code in this file is based on:
- “A recurrent variational autoencoder for speech enhancement” ICASSP, 2020, Simon Legaive

"""

from torch import nn
import torch
from collections import OrderedDict


def build_RVAE(cfg, device='cpu'):

    ### Load special paramters for RVAE
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Inference
    dense_x_gx = [] if cfg.get('Network', 'dense_x_gx') == '' else [int(i) for i in cfg.get('Network', 'dense_x_gx').split(',')]
    dim_RNN_g_x = cfg.getint('Network', 'dim_RNN_g_x')
    num_RNN_g_x = cfg.getint('Network', 'num_RNN_g_x')
    bidir_g_x = cfg.getboolean('Network', 'bidir_g_x')
    dense_z_gz = [] if cfg.get('Network', 'dense_z_gz') == '' else [int(i) for i in cfg.get('Network', 'dense_z_gz').split(',')]
    dim_RNN_g_z = cfg.getint('Network', 'dim_RNN_g_z')
    num_RNN_g_z = cfg.getint('Network', 'num_RNN_g_z')
    dense_g_z = [] if cfg.get('Network', 'dense_g_z') == '' else [int(i) for i in cfg.get('Network', 'dense_g_z').split(',')]
    # Generation
    dense_z_h = [] if cfg.get('Network', 'dense_z_h') == '' else [int(i) for i in cfg.get('Network', 'dense_z_h').split(',')]
    dim_RNN_h = cfg.getint('Network', 'dim_RNN_h')
    num_RNN_h = cfg.getint('Network', 'num_RNN_h')
    bidir_h = cfg.getboolean('Network', 'bidir_h')
    dense_h_x = [] if cfg.get('Network', 'dense_h_x') == '' else [int(i) for i in cfg.get('Network', 'dense_h_x').split(',')]

    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')

    # Build model
    model = RVAE(x_dim=x_dim, z_dim=z_dim, activation=activation,
                 dense_x_gx=dense_x_gx,
                 dim_RNN_g_x=dim_RNN_g_x, num_RNN_g_x=num_RNN_g_x,
                 bidir_g_x=bidir_g_x, 
                 dense_z_gz=dense_z_gz,
                 dim_RNN_g_z=dim_RNN_g_z, num_RNN_g_z=num_RNN_g_z,
                 dense_g_z=dense_g_z,
                 dense_z_h=dense_z_h,
                 dim_RNN_h=dim_RNN_h, num_RNN_h=num_RNN_h,
                 bidir_h=bidir_h,
                 dense_h_x=dense_h_x,
                 dropout_p=dropout_p, beta=beta, device=device).to(device)

    return model


    
class RVAE(nn.Module):

    def __init__(self, x_dim, z_dim=16, activation = 'tanh',
                 dense_x_gx=[], dim_RNN_g_x=128, num_RNN_g_x=1, bidir_g_x=False,
                 dense_z_gz=[], dim_RNN_g_z=128, num_RNN_g_z=1,
                 dense_g_z=[128],
                 dense_z_h=[], dim_RNN_h=128, num_RNN_h=1, bidir_h=False, dense_h_x=[],
                 dropout_p = 0, beta=1, device='cpu'):
                 
        super().__init__()

        ### General parameters for rvae
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
        self.dense_x_gx = dense_x_gx
        self.dim_RNN_g_x = dim_RNN_g_x
        self.num_RNN_g_x = num_RNN_g_x
        self.bidir_g_x = bidir_g_x
        self.dense_z_gz = dense_z_gz
        self.dim_RNN_g_z = dim_RNN_g_z
        self.num_RNN_g_z = num_RNN_g_z
        self.dense_g_z = dense_g_z
        ### Generation
        self.dense_z_h = dense_z_h
        self.dim_RNN_h = dim_RNN_h
        self.num_RNN_h = num_RNN_h
        self.bidir_h = bidir_h
        self.dense_h_x = dense_h_x
        ### Beta-loss
        self.beta = beta
        
        
        self.build()

    def build(self):
        
        ###################
        #### Inference ####
        ###################
        # 1. x_t to g_t^x
        dic_layers = OrderedDict()
        if len(self.dense_x_gx) == 0:
            dim_x_gx = self.x_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_x_gx = self.dense_x_gx[-1]
            for n in range(len(self.dense_x_gx)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x_gx[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_gx[n-1], self.dense_x_gx[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_x_gx = nn.Sequential(dic_layers)
        self.rnn_g_x = nn.LSTM(dim_x_gx, self.dim_RNN_g_x, self.num_RNN_g_x,
                               bidirectional=self.bidir_g_x)
        # 2. z_tm1 to g_t^z
        dic_layers = OrderedDict()
        if len(self.dense_z_gz) == 0:
            dim_z_gz = self.z_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_z_gz = self.dense_z_gz[-1]
            for n in range(len(self.dense_z_gz)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.z_dim, self.dense_z_gz[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_z_gz[n-1], self.dense_z_gz[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_z_gz = nn.Sequential(dic_layers)
        self.rnn_g_z  = nn.LSTM(dim_z_gz, self.dim_RNN_g_z, self.num_RNN_g_z)

        # 3. g_t^x and g_t^z to z
        num_dir_x = 2 if self.bidir_g_x else 1
        dic_layers = OrderedDict()
        if len(self.dense_g_z) == 0:
            dim_g_z = self.dim_RNN_g_z + num_dir_x * self.dim_RNN_g_x
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_g_z = self.dense_g_z[-1]
            for n in range(len(self.dense_g_z)):
                if n == 0: 
                    dic_layers['linear'+str(n)] = nn.Linear(num_dir_x * self.dim_RNN_g_x + self.dim_RNN_g_z, self.dense_g_z[n])
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
        # 2. h_t, recurrence
        self.rnn_h = nn.LSTM(dim_z_h, self.dim_RNN_h, self.num_RNN_h,
                             bidirectional=self.bidir_h)

        # 3. h_t to x_t
        num_dir_h = 2 if self.bidir_h else 1
        dic_layers = OrderedDict()
        if len(self.dense_h_x) == 0:
            dim_h_x = num_dir_h * self.dim_RNN_h
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_h_x = self.dense_h_x[-1]
            for n in range(len(self.dense_h_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(num_dir_h * self.dim_RNN_h, self.dense_h_x[n])
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
    
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # Create variable holder and send to GPU if needed
        z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_t = torch.zeros(batch_size, self.z_dim).to(self.device)
        g_z_t = torch.zeros(self.num_RNN_g_z, batch_size, self.dim_RNN_g_z).to(self.device)
        c_z_t = torch.zeros(self.num_RNN_g_z, batch_size, self.dim_RNN_g_z).to(self.device)
        
        # 1. x_t to g_t^x
        x_gx = self.mlp_x_gx(x)
        g_x_inverse, _ = self.rnn_g_x(torch.flip(x_gx, [0]))
        g_x = torch.flip(g_x_inverse, [0])
        
        # 2. z_t to g_t^x, g_t^x and g_t^z to z
        for t in range(0, seq_len):

            z_gz = self.mlp_z_gz(z_t).unsqueeze(0)
            _, (g_z_t, c_z_t) = self.rnn_g_z(z_gz, (g_z_t, c_z_t))
            # Get output of the last layer
            # g_z_t.view(num_layers, num_directions, batch, hidden_size)
            g_z_t_last = g_z_t.view(self.num_RNN_g_z, 1, batch_size, self.dim_RNN_g_z)[-1,:,:,:]
            # delete the first two dimension (both are 1)
            g_z_t_last = g_z_t_last.view(batch_size, self.dim_RNN_g_z)
            # concatenate g_x and g_z for time step n
            concat_xz = torch.cat([g_x[t, :,:], g_z_t_last], -1)
            # From x_t and g_z_t to z_t
            g_z = self.mlp_g_z(concat_xz)
            z_mean_t = self.inf_mean(g_z)
            z_logvar_t = self.inf_logvar(g_z)
            z_t = self.reparameterization(z_mean_t, z_logvar_t)
            # z_t = z_mean_t
            # Infer z_t
            z_mean[t,:,:] = z_mean_t
            z_logvar[t,:,:] = z_logvar_t
            z[t,:,:] = z_t

        return z, z_mean, z_logvar

    
    def generation_x(self, z):

        # 1. z_t to h_t
        z_h = self.mlp_z_h(z)

        # 2. h_t recurrence
        h, _ = self.rnn_h(z_h)

        # 3. h_t to y_t
        hx = self.mlp_h_x(h)
        y = self.gen_out(hx)

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
        info.append('>>>> x_t to g_t^x')
        for layer in self.mlp_x_gx:
            info.append(layer)
        info.append(str(self.rnn_g_x))
        info.append('>>>> z_tm1 to g_t_z')
        for layer in self.mlp_z_gz:
            info.append(layer)
        info.append(str(self.rnn_g_z))
        info.append('>>>> g_t^x and g_t_z to z_t')
        for layer in self.mlp_g_z:
            info.append(str(layer))

        info.append("----- Bottleneck -----")
        info.append(str(self.inf_mean))
        info.append(str(self.inf_logvar))

        info.append("----- Generation x -----")
        info.append('>>>> z_t to h_t')
        for layer in self.mlp_z_h:
            info.append(layer)
        info.append(str(self.rnn_h))
        info.append('>>>> h_t to x_t')
        for layer in self.mlp_h_x:
            info.append(layer)
        info.append(str(self.gen_out))
        
        return info

if __name__ == '__main__':

    x_dim = 513
    z_dim = 16
    device = 'cpu'
    rvae = RVAE(x_dim = x_dim, z_dim = z_dim).to(device)
    model_info = rvae.get_info()
    for i in model_info:
        print(i)

    x = torch.ones((2,513,3))
    y, z_mean, z_logvar, z_mean_p, z_logvar_p, z = rvae.forward(x)

    def loss_function(recon_x, x, mu, logvar, mu_prior, logvar_prior):
        recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
        KLD = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), logvar_prior.exp()))
        return recon + KLD

    loss = loss_function(y,x,z_mean,z_logvar,z_mean_p,z_logvar_p)/6

    print(loss)