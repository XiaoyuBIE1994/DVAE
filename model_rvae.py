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
        
        bidir_enc_x: boolen, true if the RNN of x is bi-directional
        h_dim_x: dimension of hidden state for the RNN of x
        num_LSTM_x: number of LSTMs of x
        
        rec_over_z: boolen, true if z_n depend on former state
        h_dim_z: dimension of hidden state for the RNN of z
        num_LSTM_z: number of LSTMs of z

        hidden_dim_enc: python list, indicate the dimensions of hidden dense layers of encoder

        num_LSTM_dec: number of LSTMs of decoder
        bidir_dec: boolen, true if the RNN for decoder is bi-directional
    """
    def __init__(self, x_dim, z_dim=16, batch_size=16,
                 bidir_enc_x=False, h_dim_x=128, num_LSTM_x=1,
                 rec_over_z=True, h_dim_z=128, num_LSTM_z=1,
                 hidden_dim_enc=[128],
                 bidir_dec=False, h_dim_dec=128, num_LSTM_dec=1, 
                 device='cpu'):
                 
        super().__init__()
        # General parameters for rvae
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size
        # Encoder: part RNN of x
        self.bidir_enc_x = bidir_enc_x
        self.h_dim_x = h_dim_x
        self.num_LSTM_x = 1
        # Encoder: part RNN of z
        self.rec_over_z = rec_over_z
        self.h_dim_z = h_dim_z
        self.num_LSTM_z = 1
        # Encoder: part dense layer 
        self.hidden_dim_enc = hidden_dim_enc
        # Decoer
        self.bidir_dec = bidir_dec
        self.h_dim_dec = h_dim_dec
        self.num_LSTM_dec = num_LSTM_dec
        # Training device 
        self.device = device

        self.y_dim = self.x_dim
        self.build()

    def build(self):

        ###### Encoder #####
        
        # 1. Define the RNN block for x (input data)
        self.enc_rnn_x = nn.LSTM(self.x_dim, self.h_dim_x, self.num_LSTM_x,
                                 bidirectional = self.bidir_enc_x)
        # 2. Define the RNN block for z (previous latent variables)
        if self.rec_over_z:
            self.enc_rnn_z  = nn.LSTM(self.z_dim, self.h_dim_z, self.num_LSTM_z)

        # 3. Define the dense layer fusing the output of two above-mentioned LSTM blocks
        if self.bidir_enc_x:
            num_directions_x = 2
        else:
            num_directions_x = 1
        
        self.dict_enc_dense = OrderedDict()

        for n, hidden_layer_dim in enumerate(self.hidden_dim_enc):
            if n == 0: # the first layer
                if self.rec_over_z:
                    tmp_dense_dim = num_directions_x * self.h_dim_x + self.h_dim_z
                else:
                    tmp_dense_dim = num_directions_x * self.h_dim_x
                self.dict_enc_dense['linear'+str(n)] = nn.Linear(tmp_dense_dim, hidden_layer_dim)

            else:
                self.dict_enc_dense['linear'+str(n)] = nn.Linear(self.hidden_dim_enc[n-1], self.hidden_dim_enc[n])
            self.dict_enc_dense['tanh'+str(n)] = nn.Tanh()

        self.enc_dense = nn.Sequential(self.dict_enc_dense)

        # 4. Define the linear layer for mean value
        self.enc_mean = nn.Linear(self.hidden_dim_enc[-1], self.z_dim)

        # 5. Define the linear layer for the log-variance
        self.enc_logvar = nn.Linear(self.hidden_dim_enc[-1], self.z_dim)

        ##### Decoder #####
        # 1. Define the LSTM procesing the latent variables
        self.dec_rnn = nn.LSTM(self.z_dim, self.h_dim_dec, self.num_LSTM_dec,
                               bidirectional = self.bidir_dec)

        # 2. Define the linear layer outputing the log-variance
        if self.bidir_dec:
            self.dec_logvar = nn.Linear(2*self.h_dim_dec, self.y_dim)
        else:
            self.dec_logvar = nn.Linear(self.h_dim_dec, self.y_dim)

            
    def encode(self, x):
        print('shape of x: {}'.format(x.shape)) # used for debug only

        # case1: input x is (batch_size, x_dim, seq_len)
        #        we want to change it to (seq_len, batch_size, x_dim)
        # case2: shape of x is (seq_len, x_dim) but we need 
        #        (seq_len, batch_size, x_dim)
        if len(x.shape) == 3:
            x = x.permute(-1, 0, 1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)

        # print('shape of input: {}'.format(x.shape)) # used for debug only
        # input('stop')
        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # create variable holder and send to GPU if needed
        all_enc_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        all_enc_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_n = torch.zeros(batch_size, self.z_dim).to(self.device)
        h_z_n = torch.zeros(self.num_LSTM_z, batch_size, self.h_dim_z).to(self.device)
        c_z_n = torch.zeros(self.num_LSTM_z, batch_size, self.h_dim_z).to(self.device)
        if self.bidir_enc_x:
            h0_x = torch.zeros(self.num_LSTM_x*2, batch_size, self.h_dim_x).to(self.device)
            c0_x = torch.zeros(self.num_LSTM_x*2, batch_size, self.h_dim_x).to(self.device)
        else:
            h0_x = torch.zeros(self.num_LSTM_x, batch_size, self.h_dim_x).to(self.device)
            c0_x = torch.zeros(self.num_LSTM_x, batch_size, self.h_dim_x).to(self.device)
        
        # rnn over x, return h_x
        h_x, _ = self.enc_rnn_x(torch.flip(x, [0]), (h0_x, c0_x))
        h_x = torch.flip(h_x, [0])
        
        if self.rec_over_z:
            for n in range(0, seq_len):
                if n > 0:
                    # Forward recurrence over z
                    # the input of nn.LSTM should be of shape (sea_len, batch_size, z_dim)
                    # so we have to add one dimension to z_n at index 0
                    _, (h_z_n, c_z_n) = self.enc_rnn_z(z_n.unsqueeze(0), (h_z_n, c_z_n))
                
                # Get output of the last layer
                # h_z_n.view(num_layers, num_directions, batch, hidden_size)
                h_z_n_last = h_z_n.view(self.num_LSTM_z, 1, batch_size, self.h_dim_z)[-1, :,:,:]
                # delete the first two dimension (both are 1)
                h_z_n_last = h_z_n.view(batch_size, self.h_dim_z)
                # concatenate h_s and h_z for time step n
                h_xz = torch.cat([h_x[n, :,:], h_z_n_last], 1)

                # encoder
                enc = self.enc_dense(h_xz)
                enc_mean_n = self.enc_mean(enc)
                enc_logvar_n = self.enc_logvar(enc)

                # sampling
                z_n = self.reparatemize(enc_mean_n, enc_logvar_n)

                # store values over time step
                all_enc_mean[n,:,:] = enc_mean_n
                all_enc_logvar[n,:,:] = enc_logvar_n
                z[n,:,:] = z_n

        else:
            # encoder
            enc = self.enc_dense(h_x)
            all_enc_mean = self.enc_mean(enc)
            all_enc_logvar = self.enc_logvar(enc)

            # sampling
            z = self.reparatemize(all_enc_mean, all_enc_logvar)

        return (torch.squeeze(all_enc_mean), torch.squeeze(all_enc_logvar), torch.squeeze(z))


    def reparatemize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        
        if len(z.shape) == 2:
            z = z.unsqueeze(1)
        
        batch_size = z.shape[1]

        # reset initial states

        if self.bidir_dec:
            h0_dec = torch.zeros(self.num_LSTM_dec*2, 
                                 batch_size, self.h_dim_dec).to(self.device)
            c0_dec = torch.zeros(self.num_LSTM_dec*2, 
                                 batch_size, self.h_dim_dec).to(self.device)
        else:
            h0_dec = torch.zeros(self.num_LSTM_dec, 
                                 batch_size, self.h_dim_dec).to(self.device)
            c0_dec = torch.zeros(self.num_LSTM_dec, 
                                 batch_size, self.h_dim_dec).to(self.device)

        # apply LSTM block to the input sequence of latent variable
        h_dec, _ = self.dec_rnn(z, (h0_dec, c0_dec))

        # output layer
        log_y = self.dec_logvar(h_dec)

        # tansform log-variance to variance
        y = torch.exp(log_y)


        # y is (seq_len, batch_size, y_dim), we want to change back to
        # (batch_size, y_dim, seq_len)
        if len(y.shape) == 3:    
            y = y.permute(1,-1,0)

        return torch.squeeze(y)

    def forward(self, x):
        mean, logvar, z = self.encode(x)
        # z is (seq_len, batch_size, z_dim), we want to change back to
        # (batch_size, z_dim, seq_len)
        y = self.decode(z)
        if len(z.shape) == 3:
            z = z.permute(1,-1,0)
        return y, mean, logvar, z


    def get_info(self):
        info = []
        info.append("----- Encoder -----")
        info.append('>>>> RNN over x')
        info.append(str(self.enc_rnn_x))
        if self.rec_over_z:
            info.append('>>>> RNN over z')
            info.append(str(self.enc_rnn_z))
        else:
            info.append('>>>> No RNN over z')
        info.append('>>>> Dense layer in encoder')
        for layer in self.enc_dense:
            info.append(str(layer))

        info.append("----- Bottleneck -----")
        info.append(str(self.enc_mean))
        info.append(str(self.enc_logvar))

        info.append("----- Decoder -----")
        info.append('>>>> RNN for decoder')
        info.append(str(self.dec_rnn))
        info.append('>>>> Dense layer to generate log-variance')
        info.append(str(self.dec_logvar))
        
        return info

if __name__ == '__main__':
    x_dim = 513
    z_dim = 16
    batch_size = 32
    device = 'cpu'
    vae = RVAE(x_dim = x_dim,
               z_dim = z_dim,
               batch_size = batch_size).to(device)
    vae.print_model()

