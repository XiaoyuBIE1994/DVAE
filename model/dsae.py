#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt

The code in this file is based on:
- “Disentangled Sequential Autoencoder” ICML, 2018, Yingzhen Li, Stephan Mandt

"""
from torch import nn
import numpy as np
import torch
from collections import OrderedDict

class DSAE(nn.Module):

    def __init__(self, x_dim, z_dim=16, v_dim=16,activation='tanh',
                 dense_vz_x=[128,128],
                 dim_RNN_gv=128, num_RNN_gv=1, bidir_gv=True,
                 dim_RNN_gx=128, num_RNN_gx=1, bidir_gx=True,
                 dim_RNN_total=128, num_RNN_total=1, bidir_total=False,
                 dim_RNN_prior=16, num_RNN_prior=1, bidir_prior=False,
                 dropout_p=0, device='cpu'):

        super().__init__()
        ## General parameters
        self.x_dim = x_dim
        self.y_dim = x_dim
        self.z_dim = z_dim
        self.v_dim = v_dim
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemError('Wrong activation type!')
        self.device = device
        # Generation
        self.dense_vz_x = dense_vz_x
        # Inference
        self.dim_RNN_gv = dim_RNN_gv
        self.num_RNN_gv = num_RNN_gv
        self.bidir_gv = bidir_gv
        self.dim_RNN_gx = dim_RNN_gx
        self.num_RNN_gx = num_RNN_gx
        self.bidir_gx = bidir_gx
        self.dim_RNN_total = dim_RNN_total
        self.num_RNN_total = num_RNN_total
        self.bidir_total = bidir_total
        #### Prior
        self.dim_RNN_prior = dim_RNN_prior
        self.num_RNN_prior = num_RNN_prior
        self.bidir_prior = bidir_prior

        self.build()

    def build(self):

        ####################
        #### Generation ####
        ####################
        dic_layer = OrderedDict()
        for n in range(len(self.dense_vz_x)):
            if n == 0:
                dic_layer['linear'+str(n)] = nn.Linear(self.v_dim + self.z_dim, self.dense_vz_x[n])
            else:
                dic_layer['linear'+str(n)] = nn.Linear(self.dense_vz_x[n-1], self.dense_vz_x[n])
            dic_layer['activation'+str(n)] = self.activation
            dic_layer['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_vz_x = nn.Sequential(dic_layer)
        self.gen_logvar = nn.Linear(self.dense_vz_x[-1], self.y_dim)

        ##################
        ### Inference ####
        ##################
        # content v
        self.rnn_gv = nn.LSTM(self.x_dim, self.dim_RNN_gv, self.num_RNN_gv, bidirectional=self.bidir_gv)
        self.inf_v_mean = nn.Linear(2*self.dim_RNN_gv, self.v_dim)
        self.inf_v_logvar = nn.Linear(2*self.dim_RNN_gv, self.v_dim)
        # dynamic z
        self.rnn_gx = nn.LSTM(self.x_dim+self.v_dim, self.dim_RNN_gx, self.num_RNN_gx, bidirectional=self.bidir_gx)
        self.rnn_total = nn.RNN(self.dim_RNN_gx*2, self.dim_RNN_total, self.num_RNN_total, bidirectional=self.bidir_total)
        self.inf_z_mean = nn.Linear(self.dim_RNN_total, self.z_dim)
        self.inf_z_logvar = nn.Linear(self.dim_RNN_total, self.z_dim)

        ###############
        #### Prior ####
        ###############
        self.rnn_prior = nn.LSTM(self.z_dim, self.dim_RNN_prior, self.num_RNN_prior, bidirectional=self.bidir_prior)
        self.prior_mean = nn.Linear(self.dim_RNN_prior, self.z_dim)
        self.prior_logvar = nn.Linear(self.dim_RNN_prior, self.z_dim)

    def generation(self, v, z):
        
        seq_len = z.shape[0]
        batch_size = z.shape[1]
        z_dim = z.shape[2]
        v_dim = v.shape[-1]

        # concatenate v and z_t
        v_expand = v.expand(seq_len, batch_size, v_dim)
        vz_cat = torch.cat((v_expand, z), -1)

        # mlp to output y
        log_y = self.gen_logvar(self.mlp_vz_x(vz_cat))
        y = torch.exp(log_y)

        return y


    def prior(self, z):
        
        z_p, _ = self.rnn_prior(z)
        z_mean_p = self.prior_mean(z_p)
        z_logvar_p = self.prior_logvar(z_p)

        return z_mean_p, z_logvar_p


    def inference(self, x):

        seq_len = x.shape[0]
        batch_size = x.shape[1]
        x_dim = x.shape[2]

        # 1. Generate global latent representation v
        _, (_v, _) = self.rnn_gv(x)
        _v = _v.view(self.num_RNN_gv, 2, batch_size, self.dim_RNN_gv)[-1,:,:,:]
        _v = torch.cat((_v[0,:,:], _v[1,:,:]), -1)
        v_mean = self.inf_v_mean(_v)
        v_logvar = self.inf_v_logvar(_v)
        v = self.reparatemize(v_mean, v_logvar)

        # 2. Generate dynamic latent representation z
        v_dim = v.shape[-1]
        v_expand = v.expand(seq_len, batch_size, v_dim)
        vx_cat = torch.cat((v_expand, x), -1)
        _z, _ = self.rnn_gx(vx_cat)
        _z, _ = self.rnn_total(_z)
        z_mean = self.inf_z_mean(_z)
        z_logvar = self.inf_z_logvar(_z)
        z = self.reparatemize(z_mean, z_logvar)
        
        return z, z_mean, z_logvar, v


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
        z, z_mean, z_logvar, v = self.inference(x)
        y = self.generation(v, z)
        z_mean_p, z_logvar_p = self.prior(z)

        # y/z dimension:    (seq_len, batch_size, y/z_dim)
        # output dimension: (batch_size, y/z_dim, seq_len)
        z = torch.squeeze(z)
        y = torch.squeeze(y)
        mean = torch.squeeze(z_mean)
        logvar = torch.squeeze(z_logvar)
        z_mean_p = torch.squeeze(z_mean_p)
        z_logvar_p = torch.squeeze(z_logvar_p)

        if len(z.shape) == 3:
            z = z.permute(1,-1,0)
        if len(y.shape) == 3:    
            y = y.permute(1,-1,0)

        return y, mean, logvar, z_mean_p, z_logvar_p, z


    def get_info(self):
        info = []
        info.append('----- Inference ----')
        info.append('>>>> Content v')
        info.append(self.rnn_gv)
        info.append(self.inf_v_mean)
        info.append(self.inf_v_logvar)
        info.append('>>>> Dynamics z')
        info.append(self.rnn_gx)
        info.append(self.rnn_total)
        info.append(self.inf_z_mean)
        info.append(self.inf_z_logvar)

        info.append('----- Generation -----')
        for layer in self.mlp_vz_x:
            info.append(layer)
        info.append(self.gen_logvar)

        info.append('----- Prior -----')
        info.append(self.rnn_prior)
        info.append(self.prior_mean)
        info.append(self.prior_logvar)

        return info
if __name__ == '__main__':
    x_dim = 513
    z_dim = 16
    device = 'cpu'

    dsae = DSAE(x_dim, z_dim).to(device)
    
    x = torch.ones([2,513,3])
    y, mean, logvar, z_mean_p, z_logvar_p, z = dsae.forward(x)

    def loss_vlb(recon_x, x, mu, logvar, mu_prior=None, z_logvar_p=None, batch_size=32, seq_len=50):
        recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
        KLD = -0.5 * torch.sum(logvar - z_logvar_p - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), z_logvar_p.exp()))
        return (recon + KLD) / (batch_size * seq_len)

    print(loss_vlb(y,x,mean,logvar,z_mean_p,z_logvar_p))