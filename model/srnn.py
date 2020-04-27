#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt

The code in this file is based on:
- “Sequential Neural Models with Stochastic Layers” NIPS, 2016, Macro Fraccaro et al.

Remark:
- In the original paper, x_tm1 is directly used to generate h_t, in order to have a fair
  comparison with other models, we add a dense layer between the forward RNN and x_t, which
  can be considered as a feature extractor, and so as for using h_t and x_t to generate g_t,
  by default it is set to be an identity layer
"""


from torch import nn
import numpy as np
import torch
from collections import OrderedDict


class SRNN(nn.Module):

    def __init__(self, x_dim, z_dim=32, activation = 'tanh',
                 dense_x_h=[], dense_hx_g=[], dense_gz_z=[128,128],
                 dense_hz_x=[128,128], dense_hz_z=[128,128],
                 dim_RNN_h=128, num_RNN_h=1,
                 dim_RNN_g=128, num_RNN_g=1,
                 dropout_p = 0, device='cpu'):

        super().__init__()
        ### General parameters for storn        
        self.x_dim = x_dim
        self.y_dim = self.x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        self.device = device
        ### Dense layers
        self.dense_x_h = dense_x_h
        self.dense_hx_g = dense_hx_g
        self.dense_gz_z = dense_gz_z
        self.dense_hz_x = dense_hz_x
        self.dense_hz_z = dense_hz_z
        self.dense_hx_g = dense_hx_g 
        ### RNN
        # Forward RNN for h_t
        self.dim_RNN_h = dim_RNN_h
        self.num_RNN_h = num_RNN_h
        # Backward RNN for g_t
        self.dim_RNN_g = dim_RNN_g
        self.num_RNN_g = num_RNN_g
        
        

        self.build()


    def build(self):
        
        #####################
        #### Dense Layer ####
        #####################
        # 1. x_tm1 -> rnn_h, encoder/decoder
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
        self.denselayer_x_h = nn.Sequential(dic_layers)

        # 2. h_t x_t -> rnn_g, encoder
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
        self.denselayer_hx_g = nn.Sequential(dic_layers)
            
        # 3. g_t z_tm1 -> z_t, encoder
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
        self.denselayer_gz_z = nn.Sequential(dic_layers)
        self.enc_mean = nn.Linear(dim_gz_z, self.z_dim)
        self.enc_logvar = nn.Linear(dim_gz_z, self.z_dim)

        # 4. h_t z_t -> x_t
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
        self.denselayer_hz_x = nn.Sequential(dic_layers)    
        self.dec_logvar = nn.Linear(dim_hz_x, self.y_dim)

        # 5. h_t z_tm1 -> z_t
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
        self.denselayer_hz_z = nn.Sequential(dic_layers)    
        self.prior_mean = nn.Linear(dim_hz_z, self.z_dim)
        self.prior_logvar = nn.Linear(dim_hz_z, self.z_dim)
        
        #############
        #### RNN ####
        #############
        # 1. Forward RNN to generate h_t from x_tm1
        self.rnn_h = nn.LSTM(dim_x_h, self.dim_RNN_h, self.num_RNN_h)
        # 2. Backward RNN to generate g_t from h_t and x_t
        self.rnn_g = nn.LSTM(dim_hx_g, self.dim_RNN_g, self.num_RNN_g)


    def forward(self, x):
        # case1: input x is (batch_size, x_dim, seq_len)
        #        we want to change it to (seq_len, batch_size, x_dim)
        # case2: shape of x is (seq_len, x_dim) but we need 
        #        (seq_len, batch_size, x_dim)
        if len(x.shape) == 3:
            x = x.permute(-1, 0, 1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        x_dim = x.shape[2]

        # reate variable holder and send to GPU if needed
        h_0_h = torch.zeros(self.num_RNN_h, batch_size, self.dim_RNN_h).to(self.device)
        c_0_h = torch.zeros(self.num_RNN_h, batch_size, self.dim_RNN_h).to(self.device)
        h_0_g = torch.zeros(self.num_RNN_g, batch_size, self.dim_RNN_g).to(self.device)
        c_0_g = torch.zeros(self.num_RNN_g, batch_size, self.dim_RNN_g).to(self.device)
        z_0 = torch.zeros(batch_size, self.z_dim).to(self.device)
        z = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        mean_enc = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        logvar_enc = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        mean_prior = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)
        logvar_prior = torch.zeros(seq_len, batch_size, self.z_dim).to(self.device)

        # 1. Generate h_t from x_tm1
        x_0 = torch.zeros(1, batch_size, x_dim).to(self.device)
        x_tm1 = torch.cat((x_0, x[1:,:,:]), 0)
        x_h = self.denselayer_x_h(x_tm1)
        h, _ = self.rnn_h(x_h, (h_0_h, c_0_h))

        # 2. Generate g_t from h_t and x_t 
        hx_g = self.denselayer_hx_g(torch.cat((h, x), -1))
        g_inverse, _ = self.rnn_g(torch.flip(hx_g, [0]), (h_0_g, c_0_g))
        g = torch.flip(g_inverse, [0])

        # 3. Infer z_t from g_t and z_tm1 (Inference/Encoder)
        for t in range(0, seq_len):
            if t == 0:
                tmp_enc = self.denselayer_gz_z(torch.cat((h[0, :, :], z_0), -1))
                mean_enc[t,:,:] = self.enc_mean(tmp_enc)
                logvar_enc[t,:,:] = self.enc_logvar(tmp_enc)
                z[t,:,:] = self.reparatemize(mean_enc[t,:,:], logvar_enc[t,:,:])
            else:
                tmp_enc = self.denselayer_gz_z(torch.cat((h[t, :, :], z[t, :, :]), -1))
                mean_enc[t,:,:] = self.enc_mean(tmp_enc)
                logvar_enc[t,:,:] = self.enc_logvar(tmp_enc)
                z[t,:,:] = self.reparatemize(mean_enc[t,:,:], logvar_enc[t,:,:])

        # 4. Generate x_t from z_t and h_t (Generation/Decoder)
        tmp_dec = self.denselayer_hz_x(torch.cat((z, h), -1))
        dec_logvar = self.dec_logvar(tmp_dec)
        y = torch.exp(dec_logvar)

        # 5. Generate prior of z_t from h_t and z_tm1 (Prior)
        for t in range(0, seq_len):
            if t == 0:
                tmp_prior = self.denselayer_hz_z(torch.cat((h[0, :, :], z_0), -1))
                mean_prior[t, :, :] = self.prior_mean(tmp_prior)
                logvar_prior[t, :, :] = self.prior_logvar(tmp_prior)
            else:
                tmp_prior = self.denselayer_hz_z(torch.cat((h[t, :, :], z[t-1, :, :]), -1))
                mean_prior[t, :, :] = self.prior_mean(tmp_prior)
                logvar_prior[t, :, :] = self.prior_logvar(tmp_prior)
        
        if len(z.shape) == 3:
            z = z.permute(1,-1,0)
        if len(y.shape) == 3:    
            y = y.permute(1,-1,0)
        
        mean = torch.squeeze(mean_enc)
        logvar = torch.squeeze(logvar_enc)
        mean_prior = torch.squeeze(mean_prior)
        logvar_prior = torch.squeeze(logvar_prior)

        return y, mean, logvar, mean_prior, logvar_prior, z 

    def reparatemize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)


    def get_info(self):
        info = []
        info.append('----- Encoder -----')
        info.append('>>>> Dense layer x_tm1 to h_t')
        for layer in self.denselayer_x_h:
            info.append(str(layer))
        info.append('>>>> Forward RNN to generate h_t')
        info.append(str(self.rnn_h))
        info.append('>>>> Dense layer h_t and x_t to g_t')
        for layer in self.denselayer_hx_g:
            info.append(str(layer))
        info.append('>>>> Backward RNN to generate g_t')
        info.append(str(self.rnn_g))
        info.append('>>>> Dense layer z_tm1 and g_t to z_t')
        for layer in self.denselayer_gz_z:
            info.append(str(layer))
        info.append(str(self.enc_mean))
        info.append(str(self.enc_logvar))
        info.append('----- Decoder -----')
        info.append('>>>> Dense layer h_t and z_t to x_t')
        for layer in self.denselayer_hz_x:
            info.append(str(layer))
        info.append(str(self.dec_logvar))
        info.append('----- Prior -----')
        info.append('>>>> Dense layer h_t and z_tm1 to z_t')
        for layer in self.denselayer_hz_z:
            info.append(str(layer))
        info.append(str(self.prior_mean))
        info.append(str(self.prior_logvar))

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
    y, mean, logvar, mean_prior, logvar_prior, z = srnn.forward(x)

    def loss_function(recon_x, x, mu, logvar, mu_prior=None, logvar_prior=None, batch_size=32, seq_len=50):
        if mu_prior is None:
            mu_prior = torch.zeros_like(mu)
        if logvar_prior is None:
            logvar_prior = torch.zeros_like(logvar)
        recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
        KLD = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), logvar_prior.exp()))
        return (recon + KLD) / (batch_size * seq_len)

    print(loss_function(y,x,mean,logvar,mean_prior,logvar)/6)