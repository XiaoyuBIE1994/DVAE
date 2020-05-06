#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt

The code in this file is based part on the source code of:
- Simon Legaive (simon.leglaive@centralesupelec.fr)
- in “A recurrent variational autoencoder for speech enhancement” ICASSP, 2020
"""

from torch import nn
# my_seed = 0
import numpy as np
# np.random.seed(my_seed)
import torch
# torch.manual_seed(my_seed)
from collections import OrderedDict

class RVAE(nn.Module):
    """
    Input:
        x_dim: input dimension, e.g. number of frequency bins
        z_dim: dimensions of latent variables
        batch_size: batch size for training
        
        bidir_g_x: boolen, true if the RNN of x is bi-directional
        dim_RNN_g_x: dimension of hidden state for the RNN of x
        num_RNN_g_x: number of LSTMs of x
        
        rec_over_z: boolen, true if z_t depend on former state
        dim_RNN_g_z: dimension of hidden state for the RNN of z
        num_RNN_g_z: number of LSTMs of z

        dense_inference: python list, indicate the dimensions of hidden dense layers of encoder

        num_RNN_h: number of LSTMs of decoder
        bidir_h: boolen, true if the RNN for decoder is bi-directional
    """
    def __init__(self, x_dim, z_dim=16, activation = 'tanh',
                 bidir_g_x=False, dim_RNN_g_x=128, num_RNN_g_x=1,
                 rec_over_z=True, dim_RNN_g_z=128, num_RNN_g_z=1,
                 dense_inference=[128],
                 bidir_h=False, dim_RNN_h=128, num_RNN_h=1, 
                 dropout_p = 0, device='cpu'):
                 
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
        ### Generation
        self.bidir_h = bidir_h
        self.dim_RNN_h = dim_RNN_h
        self.num_RNN_h = num_RNN_h

        ### Inference
        # RNN of x
        self.bidir_g_x = bidir_g_x
        self.dim_RNN_g_x = dim_RNN_g_x
        self.num_RNN_g_x = num_RNN_g_x
        # RNN of z
        self.rec_over_z = rec_over_z
        self.dim_RNN_g_z = dim_RNN_g_z
        self.num_RNN_g_z = num_RNN_g_z
        # dense layer 
        self.dense_inference = dense_inference
        
        self.build()

    def build(self):
        
        ####################
        #### Generation ####
        ####################
        # 1. Define the LSTM procesing the latent variables
        self.rnn_h = nn.LSTM(self.z_dim, self.dim_RNN_h, self.num_RNN_h,
                               bidirectional=self.bidir_h)

        # 2. Define the linear layer outputing the log-variance
        if self.bidir_h:
            self.gen_logvar = nn.Linear(2*self.dim_RNN_h, self.y_dim)
        else:
            self.gen_logvar = nn.Linear(self.dim_RNN_h, self.y_dim)

        ###################
        #### Inference ####
        ###################
        # 1. RNN of x
        self.rnn_g_x = nn.LSTM(self.x_dim, self.dim_RNN_g_x, self.num_RNN_g_x,
                               bidirectional=self.bidir_g_x)
        # 2. RNN of z
        if self.rec_over_z:
            self.rnn_g_z  = nn.LSTM(self.z_dim, self.dim_RNN_g_z, self.num_RNN_g_z)

        # 3. MLP to infer z
        if self.bidir_g_x:
            num_dir_x = 2
        else:
            num_dir_x = 1
        
        dic_layers = OrderedDict()
        for n in range(len(self.dense_inference)):
            if n == 0: 
                if self.rec_over_z:
                    tmp_dense_dim = num_dir_x * self.dim_RNN_g_x + self.dim_RNN_g_z
                else:
                    tmp_dense_dim = num_dir_x * self.dim_RNN_g_x
                dic_layers['linear'+str(n)] = nn.Linear(tmp_dense_dim, self.dense_inference[n])

            else:
                dic_layers['linear'+str(n)] = nn.Linear(self.dense_inference[n-1], self.dense_inference[n])
            dic_layers['activation'+str(n)] = self.activation
            dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)

        self.mlp_inference = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(self.dense_inference[-1], self.z_dim)
        self.inf_logvar = nn.Linear(self.dense_inference[-1], self.z_dim)
        

    def generation(self, z):

        # 1. Generate h from latent variable
        h, _ = self.rnn_h(z)

        # 2. From h_t to y_t
        log_y = self.gen_logvar(h)
        y = torch.exp(log_y)

        return y


    def inference(self, x):
    
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # 1. Create variable holder and send to GPU if needed
        z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_t = torch.zeros(batch_size, self.z_dim).to(self.device)
        g_z_t = torch.zeros(self.num_RNN_g_z, batch_size, self.dim_RNN_g_z).to(self.device)
        c_z_t = torch.zeros(self.num_RNN_g_z, batch_size, self.dim_RNN_g_z).to(self.device)
        
        # 2. rnn over x, return g_x
        g_x, _ = self.rnn_g_x(torch.flip(x, [0]))
        g_x = torch.flip(g_x, [0])
        
        if self.rec_over_z:
            for n in range(0, seq_len):
                if n > 0:
                    _, (g_z_t, c_z_t) = self.rnn_g_z(z_t.unsqueeze(0), (g_z_t, c_z_t))
                
                # Get output of the last layer
                # g_z_t.view(num_layers, num_directions, batch, hidden_size)
                g_z_t_last = g_z_t.view(self.num_RNN_g_z, 1, batch_size, self.dim_RNN_g_z)[-1,:,:,:]
                # delete the first two dimension (both are 1)
                g_z_t_last = g_z_t.view(batch_size, self.dim_RNN_g_z)
                # concatenate g_x and g_z for time step n
                concat_xz = torch.cat([g_x[n, :,:], g_z_t_last], -1)
                # From x_t and g_z_t to z_t
                infer = self.mlp_inference(concat_xz)
                inf_mean_n = self.inf_mean(infer)
                inf_logvar_n = self.inf_logvar(infer)
                z_t = self.reparatemize(inf_mean_n, inf_logvar_n)

                z_mean[n,:,:] = inf_mean_n
                z_logvar[n,:,:] = inf_logvar_n
                z[n,:,:] = z_t

        else:
            g_z = self.mlp_inference(g_x)
            z_mean = self.inf_mean(g_z)
            z_logvar = self.inf_logvar(g_z)
            z = self.reparatemize(z_mean, z_logvar)

        return z_mean, z_logvar, z


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
        mean, logvar, z = self.inference(x)
        y = self.generation(z)

        # y/z dimension:    (seq_len, batch_size, y/z_dim)
        # output dimension: (batch_size, y/z_dim, seq_len)
        z = torch.squeeze(z)
        y = torch.squeeze(y)
        mean = torch.squeeze(mean)
        logvar = torch.squeeze(logvar)
        
        if len(z.shape) == 3:
            z = z.permute(1,-1,0)
        if len(y.shape) == 3:    
            y = y.permute(1,-1,0)

        return y, mean, logvar, z


    def get_info(self):
        info = []
        info.append("----- Inference -----")
        info.append('>>>> RNN over x')
        info.append(str(self.rnn_g_x))
        if self.rec_over_z:
            info.append('>>>> RNN over z')
            info.append(str(self.rnn_g_z))
        else:
            info.append('>>>> No RNN over z')
        info.append('>>>> Dense layer in encoder')
        for layer in self.mlp_inference:
            info.append(str(layer))

        info.append("----- Bottleneck -----")
        info.append(str(self.inf_mean))
        info.append(str(self.inf_logvar))

        info.append("----- Generation -----")
        info.append('>>>> RNN for decoder')
        info.append(str(self.rnn_h))
        info.append('>>>> Dense layer to generate log-variance')
        info.append(str(self.gen_logvar))
        
        return info

if __name__ == '__main__':
    x_dim = 513
    z_dim = 16
    batch_size = 32
    device = 'cpu'
    rvae = RVAE(x_dim = x_dim,
               z_dim = z_dim,
               batch_size = batch_size).to(device)
    model_finfo = rvae.get_info()
    for i in model_finfo:
        print(i)

