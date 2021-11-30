#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

The code in this file is based on:
- “Sequential Neural Models with Stochastic Layers” NIPS, 2016, Macro Fraccaro et al.

"""


from torch import nn
import torch
from collections import OrderedDict


def build_SRNN(cfg, device='cpu'):

    ### Load parameters for SRNN
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Deterministic
    dense_x_h = [] if cfg.get('Network', 'dense_x_h') == '' else [int(i) for i in cfg.get('Network', 'dense_x_h').split(',')]
    dim_RNN_h = cfg.getint('Network', 'dim_RNN_h')
    num_RNN_h = cfg.getint('Network', 'num_RNN_h')
    # Inference
    dense_hx_g = [] if cfg.get('Network', 'dense_hx_g') == '' else [int(i) for i in cfg.get('Network', 'dense_hx_g').split(',')]
    dim_RNN_g = cfg.getint('Network', 'dim_RNN_g')
    num_RNN_g = cfg.getint('Network', 'num_RNN_g')
    dense_gz_z = [] if cfg.get('Network', 'dense_gz_z') == '' else [int(i) for i in cfg.get('Network', 'dense_gz_z').split(',')]
    # Prior
    dense_hz_z = [] if cfg.get('Network', 'dense_hz_z') == '' else [int(i) for i in cfg.get('Network', 'dense_hz_z').split(',')]
    # Generation
    dense_hz_x = [] if cfg.get('Network', 'dense_hz_x') == '' else [int(i) for i in cfg.get('Network', 'dense_hz_x').split(',')]

    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')

    # Build model
    model = SRNN(x_dim=x_dim, z_dim=z_dim, activation=activation,
                 dense_x_h=dense_x_h,
                 dim_RNN_h=dim_RNN_h, num_RNN_h=num_RNN_h,
                 dense_hx_g=dense_hx_g,
                 dim_RNN_g=dim_RNN_g, num_RNN_g=num_RNN_g,
                 dense_gz_z=dense_gz_z,
                 dense_hz_x=dense_hz_x,
                 dense_hz_z=dense_hz_z,
                 dropout_p=dropout_p, beta=beta, device=device).to(device)

    return model


    
class SRNN(nn.Module):

    def __init__(self, x_dim, z_dim=16, activation = 'tanh',
                 dense_x_h=[], dim_RNN_h=128, num_RNN_h=1,
                 dense_hx_g=[], dim_RNN_g=128, num_RNN_g=1,
                 dense_gz_z=[128,128],
                 dense_hz_z=[128,128],
                 dense_hz_x=[128,128],
                 dropout_p = 0, beta=1, device='cpu'):

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
        ### Deterministic RNN (forward)
        self.dense_x_h = dense_x_h
        self.dim_RNN_h = dim_RNN_h
        self.num_RNN_h = num_RNN_h
        ### Inference
        self.dense_hx_g = dense_hx_g
        self.dim_RNN_g = dim_RNN_g
        self.num_RNN_g = num_RNN_g
        self.dense_gz_z = dense_gz_z
        ### Generation z
        self.dense_hz_z = dense_hz_z
        ### Generation x
        self.dense_hz_x = dense_hz_x
        ### Beta-loss
        self.beta=beta
        
        self.build()


    def build(self):
        
        #######################
        #### Deterministic ####
        #######################
        # 1. x_tm1 -> h_t
        dic_layers = OrderedDict()
        if len(self.dense_x_h) == 0:
            dim_x_h = self.x_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_x_h = self.dense_x_h[-1]
            for n in range(len(self.dense_x_h)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x_h[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_h[n-1], self.dense_x_h[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_x_h = nn.Sequential(dic_layers)

        # 2. h_t, forward recurrence
        self.rnn_h = nn.LSTM(dim_x_h, self.dim_RNN_h, self.num_RNN_h)

        ###################
        #### Inference ####
        ###################
        # 1. h_t x_t -> g_t
        dic_layers = OrderedDict()
        if len(self.dense_hx_g) == 0:
            dim_hx_g = self.dim_RNN_h + self.x_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hx_g = self.dense_hx_g[-1]
            for n in range(len(self.dense_hx_g)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN_h+self.x_dim, self.dense_hx_g[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_hx_g[n-1], self.dense_hx_g[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_hx_g = nn.Sequential(dic_layers)

        # 2. g_t, backward recurrence
        self.rnn_g = nn.LSTM(dim_hx_g, self.dim_RNN_g, self.num_RNN_g)

        # 3. g_t z_tm1 -> z_t, inference
        dic_layers = OrderedDict()
        if len(self.dense_gz_z) == 0:
            dim_gz_z = self.dim_RNN_g + self.z_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_gz_z = self.dense_gz_z[-1]
            for n in range(len(self.dense_gz_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN_g+self.z_dim, self.dense_gz_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_gz_z[n-1], self.dense_gz_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_gz_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_gz_z, self.z_dim)
        self.inf_logvar = nn.Linear(dim_gz_z, self.z_dim)


        ######################
        #### Generation z ####
        ######################
        # 1. h_t z_tm1 -> z_t
        dic_layers = OrderedDict()
        if len(self.dense_hz_z) == 0:
            dim_hz_z = self.dim_RNN_h + self.z_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hz_z = self.dense_hz_z[-1]
            for n in range(len(self.dense_hz_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN_h+self.z_dim, self.dense_hz_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_hz_z[n-1], self.dense_hz_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_hz_z = nn.Sequential(dic_layers)    
        self.prior_mean = nn.Linear(dim_hz_z, self.z_dim)
        self.prior_logvar = nn.Linear(dim_hz_z, self.z_dim)


        ######################
        #### Generation x ####
        ######################
        # 1. h_t z_t -> x_t
        dic_layers = OrderedDict()
        if len(self.dense_hz_x) == 0:
            dim_hz_x = self.dim_RNN_h + self.z_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hz_x = self.dense_hz_x[-1]
            for n in range(len(self.dense_hz_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN_h+self.z_dim, self.dense_hz_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_hz_x[n-1], self.dense_hz_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_hz_x = nn.Sequential(dic_layers)    
        self.gen_out = nn.Linear(dim_hz_x, self.y_dim)


    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return torch.addcmul(mean, eps, std)
    
    
    def deterministic_h(self, x_tm1):

        x_h = self.mlp_x_h(x_tm1)
        h, _ = self.rnn_h(x_h)

        return h
    
    
    def inference(self, x, h):

        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # Create variable holder and send to GPU if needed
        z_mean = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        z_logvar = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        z = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        z_t = torch.zeros(batch_size, self.z_dim).to(self.device)

        # 1. From h_t and x_t to g_t
        hx_g = torch.cat((h, x), -1)
        hx_g = self.mlp_hx_g(hx_g)
        g_inverse, _ = self.rnn_g(torch.flip(hx_g, [0]))
        g = torch.flip(g_inverse, [0])

        # 2. From g_t and z_tm1 to z_t
        for t in range(seq_len):
            # z_t here is z[t,:,:] in the last loop (or a zero tensor)
            # so it refers to z_tm1 actually
            gz_z = torch.cat((g[t,:,:], z_t), -1)
            gz_z = self.mlp_gz_z(gz_z)
            z_mean[t,:,:] = self.inf_mean(gz_z)
            z_logvar[t,:,:] = self.inf_logvar(gz_z)
            z_t = self.reparameterization(z_mean[t,:,:], z_logvar[t,:,:])
            # z_t = z_mean[t,:,:]
            z[t,:,:] = z_t
        
        return z, z_mean, z_logvar
    

    def generation_z(self, h, z_tm1):

        hz_z = torch.cat((h, z_tm1), -1)
        hz_z = self.mlp_hz_z(hz_z)
        z_mean_p = self.prior_mean(hz_z)
        z_logvar_p = self.prior_logvar(hz_z)

        return z_mean_p, z_logvar_p


    def generation_x(self, z, h):

        # 1. z_t and h_t to y_t
        hz_x = torch.cat((h, z), -1)
        hz_x = self.mlp_hz_x(hz_x)
        y = self.gen_out(hz_x)

        return y

    
    def forward(self, x):

        # need input:  (seq_len, batch_size, x_dim)
        _, batch_size, x_dim = x.shape
        # main part
        x_0 = torch.zeros(1, batch_size, x_dim).to(self.device)
        x_tm1 = torch.cat((x_0, x[:-1, :, :]), 0)
        h = self.deterministic_h(x_tm1)
        self.z, self.z_mean, self.z_logvar = self.inference(x, h)
        z_0 = torch.zeros(1, batch_size, self.z_dim).to(self.device)
        z_tm1 = torch.cat((z_0, self.z[:-1, :, :]), 0)
        self.z_mean_p, self.z_logvar_p = self.generation_z(h, z_tm1)
        y = self.generation_x(self.z, h)

        return y


    def get_info(self):
        
        info = []
        info.append('----- Inference -----')
        info.append('>>>> From x_tm1 to h_t')
        for layer in self.mlp_x_h:
            info.append(str(layer))
        info.append('>>>> Forward RNN to generate h_t')
        info.append(str(self.rnn_h))
        info.append('>>>> From h_t and x_t to g_t')
        for layer in self.mlp_hx_g:
            info.append(str(layer))
        info.append('>>>> Backward RNN to generate g_t')
        info.append(str(self.rnn_g))
        info.append('>>>> From z_tm1 and g_t to z_t')
        for layer in self.mlp_gz_z:
            info.append(str(layer))
        
        info.append("----- Bottleneck -----")
        info.append('mean: ' + str(self.inf_mean))
        info.append('logvar: ' + str(self.inf_logvar))

        info.append('----- Generation x -----')
        info.append('>>>> From h_t and z_t to x_t')
        for layer in self.mlp_hz_x:
            info.append(str(layer))
        info.append(str(self.gen_out))

        info.append('----- Generation z -----')
        info.append('>>>> From h_t and z_tm1 to z_t')
        for layer in self.mlp_hz_z:
            info.append(str(layer))
        info.append('prior mean: ' + str(self.prior_mean))
        info.append('prior logvar: ' + str(self.prior_logvar))

        return info


if __name__ == '__main__':

    x_dim = 513
    z_dim = 16
    device = 'cpu'
    srnn = SRNN(x_dim=x_dim, z_dim=z_dim).to(device)
    model_info = srnn.get_info()
    for i in model_info:
        print(i)

    x = torch.ones((2,513,3))
    y, z_mean, z_logvar, z_mean_p, z_logvar_p, z = srnn.forward(x)

    def loss_function(recon_x, x, mu, logvar, mu_prior, logvar_prior):
        recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
        KLD = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), logvar_prior.exp()))
        return recon + KLD

    loss = loss_function(y,x,z_mean,z_logvar,z_mean_p,z_logvar_p)/6

    print(loss)
