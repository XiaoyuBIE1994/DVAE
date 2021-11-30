#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

The code in this file is based on:
- “Deep Kalman Filter” arXiv, 2015, Rahul G.Krishnan et al.
- "Structured Inference Networks for Nonlinear State Space Models" AAAI, 2017, Rahul G.Krishnan et al.

DKF refers to the deep Markov model in the second paper, which has two possibilities:
- with only backwrad RNN in inference, it's a Deep Kalman Smoother (DKS),
- with bi-directional RNN in inference, it's a ST-LR

To have consistant expression comparing with other models we change some functions' name:
Emissino Function -> Generation
Gated Transition Fucntion -> Prior
"""


from torch import nn
import torch
from collections import OrderedDict


def build_DKF(cfg, device='cpu'):

    ### Load parameters
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Inference
    dense_x_gx = [] if cfg.get('Network', 'dense_x_gx') == '' else [int(i) for i in cfg.get('Network', 'dense_x_gx').split(',')]
    dim_RNN_gx = cfg.getint('Network', 'dim_RNN_gx')
    num_RNN_gx = cfg.getint('Network', 'num_RNN_gx')
    bidir_gx = cfg.getboolean('Network', 'bidir_gx')
    dense_ztm1_g = [] if cfg.get('Network', 'dense_ztm1_g') == '' else [int(i) for i in cfg.get('Network', 'dense_ztm1_g').split(',')]
    dense_g_z = [] if cfg.get('Network', 'dense_g_z') == '' else [int(i) for i in cfg.get('Network', 'dense_g_z').split(',')]
    # Generation
    dense_z_x = [] if cfg.get('Network', 'dense_z_x') == '' else [int(i) for i in cfg.get('Network', 'dense_z_x').split(',')]

    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')

    # Build model
    model = DKF(x_dim=x_dim, z_dim=z_dim, activation=activation,
                dense_x_gx=dense_x_gx, dim_RNN_gx=dim_RNN_gx, 
                num_RNN_gx=num_RNN_gx, bidir_gx=bidir_gx,
                dense_ztm1_g=dense_ztm1_g, dense_g_z=dense_g_z,
                dense_z_x=dense_z_x,
                dropout_p=dropout_p, beta=beta, device=device).to(device)

    return model



class DKF(nn.Module):

    def __init__(self, x_dim, z_dim=16, activation='tanh',
                 dense_x_gx=[], dim_RNN_gx=128, num_RNN_gx=1, bidir_gx=False,
                 dense_ztm1_g=[], dense_g_z=[],
                 dense_z_x=[128,128],
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
        ### Inference
        self.dense_x_gx = dense_x_gx
        self.dim_RNN_gx = dim_RNN_gx
        self.num_RNN_gx = num_RNN_gx
        self.bidir_gx = bidir_gx
        self.dense_ztm1_g = dense_ztm1_g
        self.dense_g_z = dense_g_z
        ### Generation x
        self.dense_z_x = dense_z_x
        ### Beta-loss
        self.beta = beta

        self.build()


    def build(self):
    
        ###################
        #### Inference ####
        ###################
        # 1. x_t to g_tˆx
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
        self.rnn_gx = nn.LSTM(dim_x_gx, self.dim_RNN_gx, self.num_RNN_gx, bidirectional=self.bidir_gx)
        # 2. g_tˆx and z_tm1 to g_t
        dic_layers = OrderedDict()
        if len(self.dense_ztm1_g) == 0:
            dic_layers['linear_last'] = nn.Linear(self.z_dim, self.dim_RNN_gx)
            dic_layers['activation_last'] = self.activation
            dic_layers['dropout_last'+str(n)] = nn.Dropout(p=self.dropout_p)
        else:
            for n in range(len(self.dense_ztm1_g)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.z_dim, self.dense_ztm1_g[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_ztm1_g[n-1], self.dense_ztm1_g[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
            dic_layers['linear_last'] = nn.Linear(self.dense_ztm1_g[-1], self.dim_RNN_gx)
            dic_layers['activation_last'] = self.activation
            dic_layers['dropout_last'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_ztm1_g = nn.Sequential(dic_layers)
        # 3. g_t to z_t
        dic_layers = OrderedDict()
        if len(self.dense_g_z) == 0:
            dim_g_z = self.dim_RNN_gx
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_g_z = self.dense_g_z[-1]
            for n in range(len(self.dense_g_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN_gx, self.dense_g_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_g_z[n-1], self.dense_g_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_g_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_g_z, self.z_dim)
        self.inf_logvar = nn.Linear(dim_g_z, self.z_dim)

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
        self.gen_out = nn.Linear(dim_z_x, self.y_dim)


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
        x_g = self.mlp_x_gx(x)
        if self.bidir_gx:
            g, _ = self.rnn_gx(x_g)
            g = g.view(seq_len, batch_size, 2, self.dim_RNN_gx)
            g_forward = g[:,:,0,:]
            g_backward = g[:,:,1,:]
            for t in range(seq_len):
                g_t = (self.mlp_ztm1_g(z_t) + g_forward[t,:,:] + g_backward[t,:,:]) / 3
                g_z = self.mlp_g_z(g_t)
                z_mean[t,:,:] = self.inf_mean(g_z)
                z_logvar[t,:,:] = self.inf_logvar(g_z)
                z_t = self.reparameterization(z_mean[t,:,:], z_logvar[t,:,:]) 
                # z_t = z_mean[t,:,:]
                z[t,:,:] = z_t
        else:
            g, _ = self.rnn_gx(torch.flip(x_g, [0]))
            g = torch.flip(g, [0])
            for t in range(seq_len):
                g_t = (self.mlp_ztm1_g(z_t) + g[t,:,:]) / 2
                g_z = self.mlp_g_z(g_t)
                z_mean[t,:,:] = self.inf_mean(g_z)
                z_logvar[t,:,:] = self.inf_logvar(g_z)
                z_t = self.reparameterization(z_mean[t,:,:], z_logvar[t,:,:])
                # z_t = z_mean[t,:,:]
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
        y = self.mlp_z_x(z)
        y = self.gen_out(y)
        
        return y
    

    def forward(self, x):
        
        # need input:  (seq_len, batch_size, x_dim)
        _, batch_size, _ = x.shape
        self.z, self.z_mean, self.z_logvar = self.inference(x)
        z_0 = torch.zeros(1, batch_size, self.z_dim).to(self.device)
        z_tm1 = torch.cat([z_0, self.z[:-1, :,:]], 0)
        self.z_mean_p, self.z_logvar_p = self.generation_z(z_tm1)
        y = self.generation_x(self.z)

        return y


    def get_info(self):
        
        info = []
        info.append("----- Inference -----")
        info.append('>>>> x_t to g_t^x')
        for layer in self.mlp_x_gx:
            info.append(layer)
        info.append(self.rnn_gx)
        info.append('>>>> z_tm1 to g_x')
        info.append(self.mlp_ztm1_g)
        info.append('>>>> g_x to z_t')
        for layer in self.mlp_g_z:
            info.append(layer)

        info.append("----- Bottleneck -----")
        info.append(self.inf_mean)
        info.append(self.inf_logvar)

        info.append("----- Generation x -----")
        for layer in self.mlp_z_x:
            info.append(layer)
        info.append(self.gen_out)
        
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
    dkf = DKF(x_dim=x_dim, z_dim=z_dim).to(device)

    x = torch.ones((2,513,3))
    y, z_mean, z_logvar, z_mean_p, z_logvar_p, z = dkf.forward(x)

    def loss_function(recon_x, x, mu, logvar, mu_prior, logvar_prior):
        recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
        KLD = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), logvar_prior.exp()))
        return recon + KLD

    loss = loss_function(y,x,z_mean,z_logvar,z_mean_p,z_logvar_p)/6

    print(loss)
