#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

The code in this file is based on:
- “A Recurrent Latent Variable Model for Sequential Data” ICLR, 2015, Junyoung Chung et al.
"""

from torch import nn
import torch
from collections import OrderedDict


def build_VRNN(cfg, device='cpu'):

    ### Load parameters for VRNN
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Feature extractor
    dense_x = [] if cfg.get('Network', 'dense_x') == '' else [int(i) for i in cfg.get('Network', 'dense_x').split(',')]
    dense_z = [] if cfg.get('Network', 'dense_z') == '' else [int(i) for i in cfg.get('Network', 'dense_z').split(',')]
    # Dense layers
    dense_hx_z = [] if cfg.get('Network', 'dense_hx_z') == '' else [int(i) for i in cfg.get('Network', 'dense_hx_z').split(',')]
    dense_hz_x = [] if cfg.get('Network', 'dense_hz_x') == '' else [int(i) for i in cfg.get('Network', 'dense_hz_x').split(',')]
    dense_h_z = [] if cfg.get('Network', 'dense_h_z') == '' else [int(i) for i in cfg.get('Network', 'dense_h_z').split(',')]
    # RNN
    dim_RNN = cfg.getint('Network', 'dim_RNN')
    num_RNN = cfg.getint('Network', 'num_RNN')

    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')

    # Build model
    model = VRNN(x_dim=x_dim, z_dim=z_dim, activation=activation,
                 dense_x=dense_x, dense_z=dense_z,
                 dense_hx_z=dense_hx_z, dense_hz_x=dense_hz_x, 
                 dense_h_z=dense_h_z,
                 dim_RNN=dim_RNN, num_RNN=num_RNN,
                 dropout_p= dropout_p, beta=beta, device=device).to(device)

    return model


    
class VRNN(nn.Module):

    def __init__(self, x_dim, z_dim=16, activation = 'tanh',
                 dense_x=[128], dense_z=[128],
                 dense_hx_z=[128], dense_hz_x=[128], dense_h_z=[128],
                 dim_RNN=128, num_RNN=1,
                 dropout_p = 0, beta=1, device='cpu'):

        super().__init__()
        ### General parameters
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        self.y_dim = self.x_dim
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        self.device = device
        ### Feature extractors
        self.dense_x = dense_x
        self.dense_z = dense_z
        ### Dense layers
        self.dense_hx_z = dense_hx_z
        self.dense_hz_x = dense_hz_x
        self.dense_h_z = dense_h_z
        ### RNN
        self.dim_RNN = dim_RNN
        self.num_RNN = num_RNN
        ### Beta-loss
        self.beta = beta

        self.build()

    def build(self):

        ###########################
        #### Feature extractor ####
        ###########################
        # x
        dic_layers = OrderedDict()
        if len(self.dense_x) == 0:
            dim_feature_x = self.x_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_feature_x = self.dense_x[-1]
            for n in range(len(self.dense_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x[n-1], self.dense_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.feature_extractor_x = nn.Sequential(dic_layers)
        # z
        dic_layers = OrderedDict()
        if len(self.dense_z) == 0:
            dim_feature_z = self.z_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_feature_z = self.dense_z[-1]
            for n in range(len(self.dense_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.z_dim, self.dense_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_z[n-1], self.dense_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.feature_extractor_z = nn.Sequential(dic_layers)
        
        ######################
        #### Dense layers ####
        ######################
        # 1. h_t, x_t to z_t (Inference)
        dic_layers = OrderedDict()
        if len(self.dense_hx_z) == 0:
            dim_hx_z = self.dim_RNN + dim_feature_x
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hx_z = self.dense_hx_z[-1]
            for n in range(len(self.dense_hx_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x[-1] + self.dim_RNN, self.dense_hx_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_hx_z[n-1], self.dense_hx_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_hx_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_hx_z, self.z_dim)
        self.inf_logvar = nn.Linear(dim_hx_z, self.z_dim)
        
        # 2. h_t to z_t (Generation z)
        dic_layers = OrderedDict()
        if len(self.dense_h_z) == 0:
            dim_h_z = self.dim_RNN
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_h_z = self.dense_h_z[-1]
            for n in range(len(self.dense_h_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN, self.dense_h_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_h_z[n-1], self.dense_h_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_h_z = nn.Sequential(dic_layers)
        self.prior_mean = nn.Linear(dim_h_z, self.z_dim)
        self.prior_logvar = nn.Linear(dim_h_z, self.z_dim)

        # 3. h_t, z_t to x_t (Generation x)
        dic_layers = OrderedDict()
        if len(self.dense_hz_x) == 0:
            dim_hz_x = self.dim_RNN + dim_feature_z
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hz_x = self.dense_hz_x[-1]
            for n in range(len(self.dense_hz_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN + dim_feature_z, self.dense_hz_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_hz_x[n-1], self.dense_hz_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_hz_x = nn.Sequential(dic_layers)
        self.gen_out = nn.Linear(dim_hz_x, self.y_dim)
        
        ####################
        #### Recurrence ####
        ####################
        self.rnn = nn.LSTM(dim_feature_x+dim_feature_z, self.dim_RNN, self.num_RNN)


    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return torch.addcmul(mean, eps, std)


    def generation_x(self, feature_zt, h_t):

        dec_input = torch.cat((feature_zt, h_t), 2)
        dec_output = self.mlp_hz_x(dec_input)
        y_t = self.gen_out(dec_output)

        return y_t
        

    def generation_z(self, h):

        prior_output = self.mlp_h_z(h)
        mean_prior = self.prior_mean(prior_output)
        logvar_prior = self.prior_logvar(prior_output)

        return mean_prior, logvar_prior


    def inference(self, feature_xt, h_t):

        enc_input = torch.cat((feature_xt, h_t), 2)
        enc_output = self.mlp_hx_z(enc_input)
        mean_zt = self.inf_mean(enc_output)
        logvar_zt = self.inf_logvar(enc_output)

        return mean_zt, logvar_zt


    def recurrence(self, feature_xt, feature_zt, h_t, c_t):

        rnn_input = torch.cat((feature_xt, feature_zt), -1)
        _, (h_tp1, c_tp1) = self.rnn(rnn_input, (h_t, c_t))

        return h_tp1, c_tp1


    def forward(self, x):

        # need input:  (seq_len, batch_size, x_dim)
        seq_len, batch_size, _ = x.shape

        # create variable holder and send to GPU if needed
        self.z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        self.z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        y = torch.zeros((seq_len, batch_size, self.y_dim)).to(self.device)
        self.z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        h = torch.zeros((seq_len, batch_size, self.dim_RNN)).to(self.device)
        z_t = torch.zeros(batch_size, self.z_dim).to(self.device)
        h_t = torch.zeros(self.num_RNN, batch_size, self.dim_RNN).to(self.device)
        c_t = torch.zeros(self.num_RNN, batch_size, self.dim_RNN).to(self.device)

        # main part
        feature_x = self.feature_extractor_x(x)
        for t in range(seq_len):
            feature_xt = feature_x[t,:,:].unsqueeze(0)
            h_t_last = h_t.view(self.num_RNN, 1, batch_size, self.dim_RNN)[-1,:,:,:]
            mean_zt, logvar_zt = self.inference(feature_xt, h_t_last)
            z_t = self.reparameterization(mean_zt, logvar_zt)
            # z_t = mean_zt
            feature_zt = self.feature_extractor_z(z_t)
            y_t = self.generation_x(feature_zt, h_t_last)
            self.z_mean[t,:,:] = mean_zt
            self.z_logvar[t,:,:] = logvar_zt
            self.z[t,:,:] = torch.squeeze(z_t)
            y[t,:,:] = torch.squeeze(y_t)
            h[t,:,:] = torch.squeeze(h_t_last)
            h_t, c_t = self.recurrence(feature_xt, feature_zt, h_t, c_t) # recurrence for t+1 
        self.z_mean_p, self.z_logvar_p  = self.generation_z(h)
        
        return y

        
    def get_info(self):

        info = []
        info.append("----- Feature extractor -----")
        for layer in self.feature_extractor_x:
            info.append(str(layer))
        for layer in self.feature_extractor_z:
            info.append(str(layer))
        info.append("----- Inference -----")
        for layer in self.mlp_hx_z:
            info.append(str(layer))
        info.append(str(self.inf_mean))
        info.append(str(self.inf_logvar))
        info.append("----- Generation x -----")
        for layer in self.mlp_hz_x:
            info.append(str(layer))
        info.append(str(self.gen_out))
        info.append("----- Recurrence -----")
        info.append(str(self.rnn))
        info.append("----- Generation z -----")
        for layer in self.mlp_h_z:
            info.append(str(layer))
        info.append(str(self.prior_mean))
        info.append(str(self.prior_logvar))

        return info


if __name__ == '__main__':
    x_dim = 513
    z_dim = 16
    device = 'cpu'
    vrnn = VRNN(x_dim=x_dim, z_dim=z_dim).to(device)
    model_info = vrnn.get_info()
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
