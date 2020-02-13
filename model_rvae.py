#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import torch
from torch import nn
from collections import OrderedDict

class RVAE(nn.Module):
    """
    Input:
        input_dim: input dimension, e.g. number of frequency bins
        h_dim: internal hidden representation dimensions (output of LSTMs and dense layers)
        z_dim: dimensions of latent variables
        num_LSTM: number of recerrent layers in the LSTM blocks
        num_dense_enc: number of dense layers in the encoder
        bidir_enc_s: boolen, if the rnn for s is bi-directional
        bidir_dec: boolen, if the rnn for decoder is bi-directional
        rec_over_z: boolen, if it needs a rnn to generate z
    """
    def __init__(self, input_dim, h_dim = 128, z_dim = 16, batch_size = 16,
                 num_LSTM = 1, num_dense_enc = 1, bidir_enc_s = False,
                 bidir_dec = False, rec_over_z = True, device = 'cpu'):
        super(RVAE, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.num_LSTM = num_LSTM
        self.num_dense_enc = num_dense_enc
        self.bidir_enc_s = bidir_enc_s
        self.bidir_dec = bidir_dec
        self.rec_over_z = rec_over_z
        self.device = device

        self.build()

    def build(self):

        ###### Encoder #####
        
        # 1. Define the RNN block for s (data input)
        self.enc_rnn_s = nn.LSTM(self.input_dim, self.h_dim, self.num_LSTM,
                                 bidirection = self.bidir_enc_s)
        # 2. Define the RNN block for z (previous latent variables)
        if self.rec_over_z:
            self.enc_rnn_z  = nn.LSTM(self.z_dim, self.h_dim, self.num_LSTM)

        # 3. Define the dense layer fusing the output of two above-mentioned LSTM blocks
        if self.bidir_enc_s:
            num_directions_s = 2
        else:
            num_directions_s = 1
        
        self.dict_enc_dense = OrderedDict()

        for n in range(self.num_dense_enc):
            if n == 0: # the first layer
                if self.rec_over_z:
                    # 
                    tmp_input_dim = num_direction_s * self.h_dim + self.h_dim
                else:
                    tmp_input_dim = num_directions_s * self.h_dim
                self.dict_enc_dense['linear'+str(n)] = nn.Linear(tmp_input_dim, self.h_dim)

            else:
                self.dict_enc_dense['linear'+str(n)] = nn.Linear(self.h_dim, self.h_dim)
            self.dict_enc_dense['tanh'+str(n)] = nn.Tanh()

        self.enc_dense = nn.Sequential(self.dict_enc_dense)

        # 4. Define the linear layer for mean value
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)

        # 5. Define the linear lyaer for the log-variance
        self.enc_logvar = nn.Linear(self.h_dim, self.z_dim)

        ##### Decoder #####
        # 1. Define the LSTM procesing the latent variables
        self.dec_rnn = nn.LSTM(self.z_dim, self.h_dim, self.num_LSTM,
                               bidirectional = self.bidir_dec)

        # 2. Define the linear layer outputing the log-variance
        if self.bidir_dec:
            self.dec_logvar = nn.Linear(2*self.h_dim, self.input_dim)
        else:
            self.dec_logvar = nn.Linear(self.h_dim, self.input_dim)

            
    def encode(self, s):
        # shape of s is (sequence_len, input_dim) but we need 
        # (sequence_len, batch_size, input_dim), so we need to 
        # one dimension in axis 1
        if len(s.shape) == 2:
            s = s.unsqueeze(1)

        seq_len = s.shape[0]
        batch_size = s.shape[1]

        # create variable holder and send to GPU if needed
        all_enc_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        all_enc_logvar = torch.zeros((seq_len. batch_size, self.z_dim)).to(self.device)
        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_n = torch_zeros(batch_size, self.z_dim).to(self.device)
        h_z_n = torch.zeros(self.num_LSTM, batch_size, self.h_dim).to(self.device)
        c_z_n = torch.zeros(self.num_LSTM, batch_size, self.h_dim).to(self.device)
        if self.bidir_enc_s:
            h0 = torch.zeros(self.num_LSTM*2, batch_size, self.h_dim).to(self.device)
            c0 = torch.zeros(self.num_LSTM*2, batch_size, self.h_dim).to(self.device)
        else:
            h0 = torch.zeros(self.num_LSTM, batch_size, self.h_dim).to(self.device)
            c0 = torch.zeros(self.num_LSTM, batch_size, self.h_dim).to(self.device)
        h_s, _ = self.enc_rnn_s(torch_flip(s, [0]), (h0, c0))
        h_s = torch.flip(h_s, [0])
        
        if self.rec_over_z:
            for n in range(0, seq_len):
                if n > 0:
                    _, (h_z_n, c_z_n) = self.enc_rnn_z(z_n_unsequeeze(0), (h_z_n, c_z_n))
                
                h_z_n_last = h_z_n.view(self.num_LSTM, 1, batch_size, self.h_dim)[-1, :,:,:]
                h_z_n_last = h_z_n.view(batch_size, self.h_dim)
                # concatenate h_s and h_z for time step n
                h_sz = torch.cat([h_s[n, :,:], h_z_n_last], 1)

                # encoder
                enc = self.enc_dense(h_sz)
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
            enc = self.dense(h_s)
            all_enc_mean = self.enc_mean(enc)
            all_enc_logvar = self.enc_logvar(enc)

            # sampling
            z = self.(all_enc_mean, all_enc_logvar)

        return (torch,squeeze(all_enc_mean), torch.squeeze(all_enc_logvar), torch.squeeze(z))


    def reparatemize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        
        if len(z.shape) == 2:
            z = z.unsqueeze(1)
        
        batch_size = z.shape(1)

        # reset initial states

        if self.bidir_dec:
            h0 = torch.zeros(self.num_LSTM*2, batch_size, self.h_dim).to(self.device)
            c0 = torch.zeros(self.num_LSTM*2, batch_size, self.h_dim).to(self.device)
        else:
            h0 = torch.zeros(self.num_LSTM, batch_size, self.h_dim).to(self.device)
            c0 = torch.zeros(self.num_LSTM, batch_size, self.h_dim).to(self.device)

        # apply LSTM block to the input sequence of latent variable
        x, _ = self.dec_rnn(z, h0, c0)

        # output layer
        x = self.dec_logvar(x0)

        # tansform log-variance to variance
        x = torc.exp(x)

        return torch.sequeeze(x)

    def forward(self):
        mena, logvar, z = self.reparatemize(self, mean, logvar)
        return self.decode(z), mean, logvar, z

