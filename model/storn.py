#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt

The code in this file is based on:
- “Learning Stochastic Recurrent Networks” ICLR, 2015

To campare log-parameterization and softplus, one should change the last layer
in the build() function and take care of the output in the decode() function and reparameterize() function,
the get_info() should also be adapted
"""



from torch import nn
import numpy as np
import torch
from collections import OrderedDict


class STORN(nn.Module):

    def __init__(self, x_dim, z_dim=16, batch_size=16, activation = 'tanh',
                 bidir_enc=False, h_dim_enc=128, num_LSTM_enc=1,
                 hidden_dim_enc_pre=[128], hidden_dim_enc_post=[128],
                 bidir_dec=False, h_dim_dec=128, num_LSTM_dec=1,
                 hidden_dim_dec_pre=[128], hidden_dim_dec_post=[128],
                 dropout_p = 0,
                 device='cpu'):
        super().__init__()
        ### General parameters for storn        
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        self.device = device
        ### Encoder parameters
        self.bidir_enc = bidir_enc
        self.h_dim_enc = h_dim_enc
        self.num_LSTM_enc = num_LSTM_enc
        self.hidden_dim_enc_pre = hidden_dim_enc_pre
        self.hidden_dim_enc_post = hidden_dim_enc_post
        ### Decoder parameters
        self.bidir_dec = bidir_dec
        self.h_dim_dec = h_dim_dec
        self.num_LSTM_dec = num_LSTM_dec
        self.hidden_dim_dec_pre = hidden_dim_dec_pre
        self.hidden_dim_dec_post = hidden_dim_dec_post
        self.y_dim = self.x_dim

        self.build()

    def build(self):

        #################
        #### Encoder ####
        #################
        # 1. Whether apply bi-directional over x (not indicate in the paper)
        if self.bidir_enc:
            num_dir_enc = 2
        else:
            num_dir_enc = 1
        # 2. Define the dense layer before RNN block
        dic_layers = OrderedDict()
        for n in range(len(self.hidden_dim_enc_pre)):
            if n == 0:
                dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.hidden_dim_enc_pre[n])
            else:
                dic_layers['linear'+str(n)] = nn.Linear(self.hidden_dim_enc_pre[n-1], self.hidden_dim_enc_pre[n])
            dic_layers['activation'+str(n)] = self.activation
            dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.enc_dense_PreRNN = nn.Sequential(dic_layers)
        # 3. Define the RNN block
        self.enc_rnn = nn.LSTM(self.hidden_dim_enc_pre[-1], self.h_dim_enc, self.num_LSTM_enc,
                               bidirectional = self.bidir_enc)
        # 4. Define the dense layer after RNN block
        dic_layers = OrderedDict()
        for n in range(len(self.hidden_dim_enc_post)):
            if n == 0:
                dic_layers['linear'+str(n)] = nn.Linear(num_dir_enc*self.h_dim_enc, self.hidden_dim_enc_post[n])
            else:
                dic_layers['linear'+str(n)] = nn.Linear(self.hidden_dim_enc_post[n-1], self.hidden_dim_enc_post[n])
            dic_layers['activation'+str(n)] = self.activation
            dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.enc_dense_PostRNN = nn.Sequential(dic_layers)
        # 5. Define the linear layer for mean and log-variance
        self.enc_mean = nn.Linear(self.hidden_dim_enc_post[-1], self.z_dim)
        self.enc_logvar = nn.Linear(self.hidden_dim_enc_post[-1], self.z_dim) # log-params
        # self.enc_sigma = nn.Sequential(nn.Linear(self.hidden_dim_enc_post[-1], self.z_dim),
        #                                 nn.Softplus()) # softplus   
        # self.enc_var = nn.Sequential(nn.Linear(self.hidden_dim_enc_post[-1], self.z_dim),
        #                                 nn.Softplus()) # log-params + softplus
        
        
        #################
        #### Decoder ####
        #################
        # 1. Whether apply bi-directional over z (not indicate in the paper)
        if self.bidir_dec:
            num_dir_dec = 2
        else:
            num_dir_dec = 1
        # 2. Define the dense layer before RNN block
        dic_layers = OrderedDict()
        for n in range(len(self.hidden_dim_dec_pre)):
            if n == 0:
                dic_layers['linear'+str(n)] = nn.Linear(self.x_dim+self.z_dim, self.hidden_dim_dec_pre[n])
            else:
                dic_layers['linear'+str(n)] = nn.Linear(self.hidden_dim_dec_pre[n-1], self.hidden_dim_dec_pre[n])
            dic_layers['activation'+str(n)] = self.activation
            dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.dec_dense_PreRNN = nn.Sequential(dic_layers)
        # 3. Define the RNN block
        self.dec_rnn = nn.LSTM(self.hidden_dim_dec_pre[-1], self.h_dim_dec, self.num_LSTM_dec,
                               bidirectional = self.bidir_dec)
        # 4. Define the dense layer after RNN block
        dic_layers = OrderedDict()
        for n in range(len(self.hidden_dim_dec_post)):
            if n == 0:
                dic_layers['linear'+str(n)] = nn.Linear(num_dir_dec*self.h_dim_dec, self.hidden_dim_dec_post[n])
            else:
                dic_layers['linear'+str(n)] = nn.Linear(self.hidden_dim_dec_post[n-1], self.hidden_dim_dec_post[n])
            dic_layers['activation'+str(n)] = self.activation
            dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.dec_dense_PostRNN = nn.Sequential(dic_layers)
        # 5. Define the linear layer for output (variance only)
        self.dec_logvar = nn.Linear(self.hidden_dim_dec_post[-1], self.y_dim) # log-params
        # self.dec_sigma = nn.Sequential(nn.Linear(self.hidden_dim_dec_post[-1], self.y_dim),
        #                                nn.Softplus()) # softplus
        # self.dec_logvar = nn.Sequential(nn.Linear(self.hidden_dim_dec_post[-1], self.y_dim),
        #                                 nn.Softplus()) # log-params + softplus
        

    def encode(self, x):
        # print('shape of x: {}'.format(x.shape)) # used for debug only

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
        x_dim = x.shape[2]

        # create x_tm1 for decoder
        x_0 = torch.zeros(1, batch_size, x_dim).to(self.device)
        x_tm1 = torch.cat((x_0, x[1:,:,:]), 0)

        # create variable holder and send to GPU if needed
        if self.bidir_enc:
            h0_enc = torch.zeros(self.num_LSTM_enc*2, batch_size, self.h_dim_enc).to(self.device)
            c0_enc = torch.zeros(self.num_LSTM_enc*2, batch_size, self.h_dim_enc).to(self.device)
        else:
            h0_enc = torch.zeros(self.num_LSTM_enc, batch_size, self.h_dim_enc).to(self.device)
            c0_enc = torch.zeros(self.num_LSTM_enc, batch_size, self.h_dim_enc).to(self.device)

        # 1. Linear layers before RNN block
        x_PreRNN = self.enc_dense_PreRNN(x)

        # 2. RNN with input x_PreRNN, return x_PostRNN
        h_x, _ = self.enc_rnn(x_PreRNN, (h0_enc, c0_enc))

        # 3. Linear layer after RNN block
        x_PostRNN = self.enc_dense_PostRNN(h_x)

        # 4. output mean and logvar
        all_enc_mean = self.enc_mean(x_PostRNN)
        all_enc_logvar = self.enc_logvar(x_PostRNN)
        # all_enc_sigma = self.enc_sigma(x_PostRNN)

        return (all_enc_mean, all_enc_logvar, x_tm1)
        # return (all_enc_mean, all_enc_sigma, x_tm1)

    def reparatemize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    # def reparatemize(self, mean, std):
    #     eps = torch.randn_like(std)
    #     return eps.mul(std).add_(mean)
    
    def decode(self, dec_input):
        
        batch_size = dec_input.shape[1]

        # create variable holder and send to GPU if needed
        if self.bidir_dec:
            h0_dec = torch.zeros(self.num_LSTM_enc*2, batch_size, self.h_dim_dec).to(self.device)
            c0_dec = torch.zeros(self.num_LSTM_enc*2, batch_size, self.h_dim_dec).to(self.device)
        else:
            h0_dec = torch.zeros(self.num_LSTM_enc, batch_size, self.h_dim_dec).to(self.device)
            c0_dec = torch.zeros(self.num_LSTM_enc, batch_size, self.h_dim_dec).to(self.device)

        # 1. Linear layers before RNN block
        y_PreRNN = self.dec_dense_PreRNN(dec_input)

        # 2. RNN with input x_PreRNN, return x_PostRNN
        h_y, _ = self.dec_rnn(y_PreRNN, (h0_dec, c0_dec))

        # 3. Linear layer after RNN block
        y_PostRNN = self.dec_dense_PostRNN(h_y)

        # 4. output mean and logvar
        log_y = self.dec_logvar(y_PostRNN)

        # 5. transform log-variance to variance
        y = torch.exp(log_y)
        # y = self.dec_sigma(y_PostRNN)

        return torch.squeeze(y)

    def forward(self, x):
        mean, logvar, x_tm1 = self.encode(x)
        z = self.reparatemize(mean, logvar)
        dec_input = torch.cat((z, x_tm1), 2)
        y = self.decode(dec_input)
        z = torch.squeeze(z)
        # y/z is (seq_len, batch_size, y/z_dim), we want to change back to
        # (batch_size, y/z_dim, seq_len)
        if len(z.shape) == 3:
            z = z.permute(1,-1,0)
        if len(y.shape) == 3:    
            y = y.permute(1,-1,0)
        return torch.squeeze(y), torch.squeeze(mean), torch.squeeze(logvar), torch.squeeze(z)

    def get_info(self):
        info = []
        info.append("----- Encoder -----")
        info.append('>>>> Dense before RNN')
        for layer in self.enc_dense_PreRNN:
            info.append(str(layer))
        info.append('>>>> RNN')
        info.append(self.enc_rnn)
        info.append('>>>> Dense after RNN')
        for layer in self.enc_dense_PostRNN:
            info.append(str(layer))
        
        info.append("----- Bottleneck -----")
        info.append('mean: ' + str(self.enc_mean))
        info.append('logvar: ' + str(self.enc_logvar))
        # info.append('logvar: ' + str(self.enc_sigma))

        info.append("----- Decoder -----")
        info.append('>>>> Dense before RNN')
        for layer in self.dec_dense_PreRNN:
            info.append(str(layer))
        info.append('>>>> RNN')
        info.append(self.dec_rnn)
        info.append('>>>> Dense after RNN')
        for layer in self.dec_dense_PostRNN:
            info.append(str(layer))
        info.append('Output: ' + str(self.dec_logvar))
        # info.append('Output: ' + str(self.dec_sigma))

        return info

if __name__ == '__main__':
    x_dim = 513
    z_dim = 16
    batch_size = 32
    device = 'cpu'
    storn = STORN(x_dim=x_dim, z_dim=z_dim, batch_size=batch_size).to(device)
    model_info = storn.get_info()
    # for i in model_info:
    #     print(i)

    x = torch.ones((2,513,3))
    y, mean, logvar, z = storn.forward(x)
    print(x.shape)
    print(y.shape)
    print(mean.shape)
    print(logvar.shape)
    print(z.shape)
    print(y[0,:5,0])
    def loss(recon_x, x, mu, logvar, mu_prior=None, logvar_prior=None):
        if mu_prior is None:
            mu_prior = torch.zeros_like(mu)
        if logvar_prior is None:
            logvar_prior = torch.zeros_like(logvar)
        recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
        KLD = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), logvar_prior.exp()))
        return recon + KLD

    print(loss(y,x,mean,logvar)/6)

