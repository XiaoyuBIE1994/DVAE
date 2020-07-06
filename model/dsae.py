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
import torch
from collections import OrderedDict

class DSAE(nn.Module):

    def __init__(self, x_dim, z_dim=16, v_dim=16,activation='tanh',
                 dense_x=[],
                 dim_RNN_gv=128, num_RNN_gv=1,
                 dense_gv_v=[], dense_xv_gxv=[],
                 dim_RNN_gxv=128, num_RNN_gxv=1,
                 dense_gxv_gz=[],
                 dim_RNN_gz=128, num_RNN_gz=1,
                 dim_RNN_prior=16, num_RNN_prior=1,
                 dense_vz_x=[128,128],
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
        # Inference
        self.dense_x = dense_x
        self.dim_RNN_gv = dim_RNN_gv
        self.num_RNN_gv = num_RNN_gv
        self.dense_gv_v = dense_gv_v
        self.dense_xv_gxv = dense_xv_gxv
        self.dim_RNN_gxv = dim_RNN_gxv
        self.num_RNN_gxv = num_RNN_gxv
        self.dense_gxv_gz = dense_gxv_gz
        self.dim_RNN_gz = dim_RNN_gz
        self.num_RNN_gz = num_RNN_gz
        #### Generation z
        self.dim_RNN_prior = dim_RNN_prior
        self.num_RNN_prior = num_RNN_prior
        # Generation x
        self.dense_vz_x = dense_vz_x

        self.build()


    def build(self):

       
        ##################
        ### Inference ####
        ##################
        ### feature x
        dic_layers = OrderedDict()
        if len(self.dense_x) == 0:
            dim_x = self.x_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_x = self.dense_x[-1]
            for n in range(len(self.dense_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x[n-1], self.dense_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_x = nn.Sequential(dic_layers)
        ### content v
        # 1. g_t^v, bi-directional recurrencce
        self.rnn_g_v = nn.LSTM(dim_x, self.dim_RNN_gv, self.num_RNN_gv, bidirectional=True)
        # 2. g_t^v -> v
        dic_layers = OrderedDict()
        if len(self.dense_gv_v) == 0:
            dim_gv_v = 2*self.dim_RNN_gv
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_gv_v = self.dense_gv_v[-1]
            for n in range(len(self.dense_gv_v)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(2*self.dim_RNN_gv, self.dense_gv_v[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_gv_v[n-1], self.dense_gv_v[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_gv_v = nn.Sequential(dic_layers)
        self.inf_v_mean = nn.Linear(dim_gv_v, self.v_dim)
        self.inf_v_logvar = nn.Linear(dim_gv_v, self.v_dim)
        ### dynamic z
        # 1. feature_x and v -> g_t^xv
        dic_layers = OrderedDict()
        if len(self.dense_xv_gxv) == 0:
            dim_xv_gxv = dim_x + self.v_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_xv_gxv = self.dense_xv_gxv[-1]
            for n in range(len(self.dense_xv_gxv)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(dim_x + self.v_dim, self.dense_xv_gxv[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_xv_gxv[n-1], self.dense_xv_gxv[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_xv_gxv = nn.Sequential(dic_layers)
        # 2. g_t^xv, bidirectional recurrence
        self.rnn_g_xv = nn.LSTM(dim_xv_gxv, self.dim_RNN_gxv, self.num_RNN_gxv, bidirectional=True)
        # 3. g_t^xv -> g_t^z
        dic_layers = OrderedDict()
        if len(self.dense_gxv_gz) == 0:
            dim_gxv_gz = 2*self.dim_RNN_gxv
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_gxv_gz = self.dense_gxv_gz[-1]
            for n in range(len(self.dense_gxv_gz)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(2*self.dim_RNN_gxv, self.dense_gxv_gz[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_gxv_gz[n-1], self.dense_gxv_gz[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_gxv_gz = nn.Sequential(dic_layers)
        # 4. g_t^z, forward recurrence
        self.rnn_g_z = nn.RNN(dim_gxv_gz, self.dim_RNN_gz, self.num_RNN_gz, bidirectional=False)
        # 5. g_t^z -> z_t
        self.inf_z_mean = nn.Linear(self.dim_RNN_gz, self.z_dim)
        self.inf_z_logvar = nn.Linear(self.dim_RNN_gz, self.z_dim)

        ######################
        #### generation z ####
        ######################
        self.rnn_prior = nn.LSTM(self.z_dim, self.dim_RNN_prior, self.num_RNN_prior, bidirectional=False)
        self.prior_mean = nn.Linear(self.dim_RNN_prior, self.z_dim)
        self.prior_logvar = nn.Linear(self.dim_RNN_prior, self.z_dim)

        ######################
        #### Generation x ####
        ######################
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


    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)


    def inference(self, x):

        seq_len = x.shape[0]
        batch_size = x.shape[1]
        x_dim = x.shape[2]

        # 1. Feature x
        feature_x = self.mlp_x(x)

        # 2. Generate content v
        _, (_v, _) = self.rnn_g_v(feature_x)
        _v = _v.view(self.num_RNN_gv, 2, batch_size, self.dim_RNN_gv)[-1,:,:,:]
        _v = torch.cat((_v[0,:,:], _v[1,:,:]), -1)
        _v = self.mlp_gv_v(_v)
        v_mean = self.inf_v_mean(_v)
        v_logvar = self.inf_v_logvar(_v)
        v = self.reparameterization(v_mean, v_logvar)

        # 2. Generate dynamic latent representation z
        v_dim = v.shape[-1]
        v_expand = v.expand(seq_len, batch_size, v_dim)
        xv_cat = torch.cat((feature_x, v_expand), -1)
        g_xv = self.mlp_xv_gxv(xv_cat)
        g_xv, _ = self.rnn_g_xv(g_xv)
        g_z, _ = self.rnn_g_z(g_xv)
        z_mean = self.inf_z_mean(g_z)
        z_logvar = self.inf_z_logvar(g_z)
        z = self.reparameterization(z_mean, z_logvar)
        
        return z, z_mean, z_logvar, v


    def generation_z(self, z):
        
        z_p, _ = self.rnn_prior(z)
        z_mean_p = self.prior_mean(z_p)
        z_logvar_p = self.prior_logvar(z_p)

        return z_mean_p, z_logvar_p


    def generation_x(self, v, z):
        
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
        z_mean_p, z_logvar_p = self.generation_z(z)
        y = self.generation_x(v, z)

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
        info.append('----- Inference ----')
        info.append('>>>> Feature x')
        for layer in self.mlp_x:
            info.append(layer)
        info.append('>>>> Content v')
        info.append(self.rnn_g_v)
        for layer in self.mlp_gv_v:
            info.append(layer)
        info.append(self.inf_v_mean)
        info.append(self.inf_v_logvar)
        info.append('>>>> Dynamics z')
        for layer in self.mlp_xv_gxv:
            info.append(layer)
        info.append(self.rnn_g_xv)
        for layer in self.mlp_gxv_gz:
            info.append(layer)
        info.append(self.rnn_g_z)
        info.append(self.inf_z_mean)
        info.append(self.inf_z_logvar)

        info.append('----- Generation x -----')
        for layer in self.mlp_vz_x:
            info.append(layer)
        info.append(self.gen_logvar)

        info.append('----- Generation z -----')
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
    y, z_mean, z_logvar, z_mean_p, z_logvar_p, z = dsae.forward(x)

    def loss_function(recon_x, x, mu, logvar, mu_prior, logvar_prior):
        recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
        KLD = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), logvar_prior.exp()))
        return recon + KLD

    loss = loss_function(y,x,z_mean,z_logvar,z_mean_p,z_logvar_p)/6

    print(loss)