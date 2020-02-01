#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 by Inria
Authored by Simon Leglaive (simon.leglaive@inria.fr)
License agreement in LICENSE.txt
"""

import torch
from torch import nn
from collections import OrderedDict

class VAE(nn.Module):

    """
    Feed-forward fully-connected VAE
    """

    def __init__(self, input_dim=None, latent_dim=None,
                 hidden_dim_encoder=None, batch_size=None, activation=None):

        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim_encoder = hidden_dim_encoder
        self.batch_size = batch_size
        self.activation = activation
        self.model = None
        self.history = None

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        self.build()

    def build(self):

        """ Build the network according to the attributes of the class.
        """

        #### Define encoder ####

        for n, dim in enumerate(self.hidden_dim_encoder):
            if n==0:
                self.encoder_layers.append(nn.Linear(self.input_dim, dim))
            else:
                self.encoder_layers.append(nn.Linear(self.hidden_dim_encoder[n-1], dim))

        #### Define bottleneck layer ####

        self.latent_mean_layer = nn.Linear(self.hidden_dim_encoder[-1], self.latent_dim)
        self.latent_logvar_layer = nn.Linear(self.hidden_dim_encoder[-1], self.latent_dim)

        #### Define decoder ####

        hidden_dim_decoder = list(reversed(self.hidden_dim_encoder))
        for n, dim in enumerate(hidden_dim_decoder):
            if n==0:
                self.decoder_layers.append(nn.Linear(self.latent_dim, dim))
            else:
                self.decoder_layers.append(nn.Linear(hidden_dim_decoder[n-1], dim))


        self.output_layer = nn.Linear(hidden_dim_decoder[-1], self.input_dim)

    def encode(self, x):

        for layer in self.encoder_layers:
            x = self.activation(layer(x))

        mean = self.latent_mean_layer(x)
        logvar = self.latent_logvar_layer(x)
        z = self.reparameterize(mean, logvar)

        return mean, logvar, z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):

        for layer in self.decoder_layers:
            z = self.activation(layer(z))

        return torch.exp(self.output_layer(z))

    def forward(self, x):
        mean, logvar, z = self.encode(x)
        return self.decode(z), mean, logvar, z

    def print_model(self):

        print('------ Encoder -------')
        for layer in self.encoder_layers:
            print(layer)
            print(self.activation)

        print('------ Bottleneck -------')
        print(self.latent_mean_layer)
        print(self.latent_logvar_layer)

        print('------ Decoder -------')
        for layer in self.decoder_layers:
            print(layer)
            print(self.activation)
        print(self.output_layer)



class VAE_Decoder(nn.Module):

    def __init__(self, vae):

        super(VAE_Decoder, self).__init__()

        self.latent_dim = vae.latent_dim
        self.hidden_dim_decoder = list(reversed(vae.hidden_dim_encoder))
        self.batch_size = vae.batch_size
        self.activation = vae.activation
        self.decoder_layers = nn.ModuleList()
        self.build()

    def build(self, vae):

         #### Define decoder ####

        for n, layer in enumerate(vae.decoder_layers):
            self.decoder_layers.append(layer)

        self.output_layer = vae.output_layer

    def forward(self, z):

        for n, layer in enumerate(self.decoder_layers):
            z = self.activation(layer(z))

        return torch.exp(self.output_layer(z))


class RVAE(nn.Module):

    """
    Reccurrent (uni- or bi-directional LSTM) VAE
    """

    def __init__(self, input_dim, h_dim=128, z_dim=16, batch_size=16,
                 num_LSTM=1, num_dense_enc=1, bidir_enc_s=False,
                 bidir_dec=False, rec_over_z=True, device='cpu'):

        super(RVAE, self).__init__()

        self.input_dim = input_dim # input dim. (e.g. num of frequency bins)
        self.h_dim = h_dim # internal hidden representation dimensions
        # (i.e. output of LSTMs and dense layers)
        self.z_dim = z_dim # latent dimension
#        self.batch_size = batch_size # batch size
        self.num_LSTM = num_LSTM # num. of recurrent layers in the LSTM blocks
        # Setting num_layers=2 would mean stacking two LSTMs together.

        self.num_dense_enc = num_dense_enc # num. of dense layers in the encoder

        self.bidir_enc_s = bidir_enc_s # indicates if the encoder LSTM blocks
        # processing the data input sequence sequence should be directional

        self.bidir_dec = bidir_dec # indicates if the decoder LSTM blocks
        # processing the latent variables sequence should be directional

        self.rec_over_z = rec_over_z # indicates if there should be a
        # recurrence for generating the latent variables in the encoding part

        self.device = device

        # Build the network
        self.build()

    def build(self):

        ########## Encoder ##########

        # 1. Define LSTM blocks processing the data input sequence
        self.enc_rnn_s = nn.LSTM(self.input_dim, self.h_dim, self.num_LSTM,
                                 bidirectional=self.bidir_enc_s)

        # 2. Define LSTM blocks handling the posterior dependencies between
        # the latent variables at different time steps
        if self.rec_over_z: # only relevant if recurrence over latent variables
            self.enc_rnn_z = nn.LSTM(self.z_dim, self.h_dim, self.num_LSTM)

        # 3. Define the dense layers fusing the output of two above-mentioned
        # LSTM blocks
        if self.bidir_enc_s:
            num_directions_s = 2
        else:
            num_directions_s = 1
        # Ordered dictionary describing these internal dense layers:
        # * The first layer takes as input the concatenation of the outputs of
        # the two LSTM blocks in the encoder. Its input size is therefore
        # num_directions_s*h_dim + h_dim.
        # * We arbitrarly choose that the first layer has an ouput dimension
        # equal to h_dim, as for the LSTM blocks.
        # * We arbitrarly choose that all the following layers have also an
        # an ouput dimension equal to h_dim.
        # * All these layers are non-linear and use a tanh activation function
        self.dict_enc_dense = OrderedDict()

        for n in range(self.num_dense_enc):
            if n==0: # first layer

                if self.rec_over_z:
                    # we have to consider the output of the LSTM blocks
                    # handling the posterior dependencies between
                    # the latent variables at different time steps
                    tmp_input_dim = num_directions_s*self.h_dim + self.h_dim
                else:
                    # we do not have to
                    tmp_input_dim = num_directions_s*self.h_dim

                self.dict_enc_dense['linear'+str(n)] = nn.Linear(tmp_input_dim,
                                    self.h_dim)
            else:  # following layers

                self.dict_enc_dense['linear'+str(n)] = nn.Linear(self.h_dim,
                                    self.h_dim)

            self.dict_enc_dense['tanh'+str(n)] = nn.Tanh()

        self.enc_dense = nn.Sequential(self.dict_enc_dense)

        # 4. Define the linear layer outputing the mean
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)

        # 5. Define the linear layer outputing the log-variance
        self.enc_logvar = nn.Linear(self.h_dim, self.z_dim)

        ########## Decoder ##########

        # 1. Define LSTM blocks processing the latent variables input sequence
        self.dec_rnn = nn.LSTM(self.z_dim, self.h_dim, self.num_LSTM,
                               bidirectional=self.bidir_dec)

        # 2. Define the linear layer outputing the log-variance
        if self.bidir_dec:
            self.dec_logvar = nn.Linear(2*self.h_dim, self.input_dim)
        else:
            self.dec_logvar = nn.Linear(self.h_dim, self.input_dim)


    def encode(self, s):

        if len(s.shape) == 2:
            # shape is (sequence_len, input_dim) but we need
            # (sequence_len, batch_size, input_dim)
            s = s.unsqueeze(1) # add a dimension in axis 1

        seq_len = s.shape[0]
        batch_size = s.shape[1]

        all_enc_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        # shape (seq_len, batch_size, z_dim)

        all_enc_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        # shape (seq_len, batch_size, z_dim, seq_len)

        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        # shape (seq_len, batch_size, z_dim, seq_len)

        z_n = torch.zeros(batch_size, self.z_dim).to(self.device)  # shape (batch_size, z_dim)

        h_z_n = torch.zeros(self.num_LSTM, batch_size, self.h_dim).to(self.device)
        # h_z_n is the hidden state of an LSTM, cf. nn.LSTM
        # h_z_n is of shape (num_layers*num_directions, batch_size, hidden_size)

        c_z_n = torch.zeros(self.num_LSTM, batch_size, self.h_dim).to(self.device)
        # c_z_n is the cell state of an LSTM, cf. nn.LSTM
        # c_z_n is of shape (num_layers*num_directions, batch_size, hidden_size)

        # backward recurrence on the input sequence with reseted initial states
        if self.bidir_enc_s:
            h0 = torch.zeros(self.num_LSTM*2, batch_size, self.h_dim).to(self.device)
            c0 = torch.zeros(self.num_LSTM*2, batch_size, self.h_dim).to(self.device)
        else:
            h0 = torch.zeros(self.num_LSTM, batch_size, self.h_dim).to(self.device)
            c0 = torch.zeros(self.num_LSTM, batch_size, self.h_dim).to(self.device)
        h_s, _ = self.enc_rnn_s(torch.flip(s,[0]), (h0, c0))
        # h_s is of shape (seq_len, batch_size, hidden_size)
        h_s = torch.flip(h_s,[0]) # h_s[n] is a function of s[n], ..., s[N-1]

        if self.rec_over_z:

            for n in range(0, seq_len):

                if n > 0:
                    # forward recurrence over z
                     # the input of nn.LSTM should be of shape (seq_len, batch_size, z_dim),
                     # so we have to add a dimension to z_n at index 0 using unsqueeze
                    _, (h_z_n,c_z_n) = self.enc_rnn_z( z_n.unsqueeze(0),
                        (h_z_n, c_z_n) )


                # Get the output of the last layer
                # h_z_n.view(num_layers, num_directions, batch, hidden_size)
                h_z_n_last = h_z_n.view(self.num_LSTM, 1,
                                        batch_size, self.h_dim)[-1,:,:,:]
                # reshape it to (batch_size, num_directions*hidden_size)
                h_z_n_last = h_z_n_last.view(batch_size, self.h_dim)

                # concatenate h_s and h_z for time step nvae
                h_sz = torch.cat([h_s[n,:,:], h_z_n_last], 1)
                # h_z[-1] is of shape (batch_size, hidden_size) and corresponds
                # to the output of the last LSTM layer.
                # h_z[-1] is equivalent to h_z.view(num_layers, num_directions, batch, hidden_size)[-1,0,:,:]

                # encoder
                enc = self.enc_dense(h_sz) # shape (batch_size, h_dim)
                enc_mean_n = self.enc_mean(enc) # shape (batch_size, z_dim)
                enc_logvar_n = self.enc_logvar(enc) # shape (batch_size, z_dim)

                # sampling
                z_n = self.sample_enc(enc_mean_n, enc_logvar_n)

                # store values over time
                all_enc_logvar[n,:,:] = enc_logvar_n
                all_enc_mean[n,:,:] = enc_mean_n
                z[n,:,:] = z_n

        else:

            # encoder
            enc = self.enc_dense(h_s) # shape (batch_size, h_dim)
            all_enc_mean = self.enc_mean(enc) # shape (batch_size, z_dim)
            all_enc_logvar = self.enc_logvar(enc) # shape (batch_size, z_dim)

            # sampling
            z = self.sample_enc(all_enc_mean, all_enc_mean)

        return (torch.squeeze(all_enc_mean), torch.squeeze(all_enc_logvar),
                torch.squeeze(z))

    def decode(self, z):

        if len(z.shape) == 2:
            # shape is (sequence_len, input_dim) but we need
            # (sequence_len, batch_size, input_dim)
            z = z.unsqueeze(1) # add a dimension in axis 1

        batch_size = z.shape[1]

        # reset initial states
        if self.bidir_dec:
            h0 = torch.zeros(self.num_LSTM*2, batch_size, self.h_dim).to(self.device)
            c0 = torch.zeros(self.num_LSTM*2, batch_size, self.h_dim).to(self.device)
        else:
            h0 = torch.zeros(self.num_LSTM, batch_size, self.h_dim).to(self.device)
            c0 = torch.zeros(self.num_LSTM, batch_size, self.h_dim).to(self.device)

        # apply LSTM block to the input sequence of latent variables
        x, _ = self.dec_rnn(z, (h0, c0))

        # output layer
        x = self.dec_logvar(x)

        # transform log-variance to variance
        x = torch.exp(x)

        return torch.squeeze(x)

    def forward(self, s):
        mean, logvar, z = self.encode(s)
        return self.decode(z), mean, logvar, z

    def sample_enc(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

class RVAE_Decoder(nn.Module):

    def __init__(self, vae):

        super(RVAE_Decoder, self).__init__()

        self.input_dim = vae.input_dim # input dim. (e.g. num of frequency bins)
        self.h_dim = vae.h_dim # internal hidden representation dimensions
        # (i.e. output of LSTMs and dense layers)
        self.z_dim = vae.z_dim # latent dimension
        # self.batch_size = batch_size # batch size
        self.num_LSTM = vae.num_LSTM # num. of recurrent layers in the LSTM blocks
        # Setting num_layers=2 would mean stacking two LSTMs together.

        self.bidir_dec = vae.bidir_dec # indicates if the decoder LSTM blocks
        # processing the latent variables sequence should be directional

        self.device = vae.device

        # Build the network
        self.build()

    def build(self, vae):

        # 1. Define LSTM blocks processing the latent variables input sequence
        self.dec_rnn = vae.dec_rnn

        # 2. Define the linear layer outputing the log-variance
        self.dec_logvar = vae.dec_logvar


    def forward(self, z):

        batch_size = z.shape[1]

        # reset initial states
        if self.bidir_dec:
            h0 = torch.zeros(self.num_LSTM*2, batch_size, self.h_dim).to(self.device)
            c0 = torch.zeros(self.num_LSTM*2, batch_size, self.h_dim).to(self.device)
        else:
            h0 = torch.zeros(self.num_LSTM, batch_size, self.h_dim).to(self.device)
            c0 = torch.zeros(self.num_LSTM, batch_size, self.h_dim).to(self.device)

        # apply LSTM block to the input sequence of latent variables
        x, _ = self.dec_rnn(z, (h0, c0))

        # output layer
        x = self.dec_logvar(x)

        # transform log-variance to variance
        x = torch.exp(x)

        return x
