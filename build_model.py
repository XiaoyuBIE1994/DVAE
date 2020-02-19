#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import sys
import os
import socket
import datetime
import torch

import librosa
from configparser import ConfigParser
import backup_simon.speech_dataset
from model_vae import VAE


# Re-write configure class, enable to distinguish betwwen upper and lower letters
class myconf(ConfigParser):
    def __init__(self,defaults=None):
        ConfigParser.__init__(self,defaults=None)
    def optionxform(self, optionstr):
        return optionstr


class BuildBasic():

    """
    Basical class for model building, include fundamental functions for all VAE/RVAE models
    """

    def __init__(self, cfg = myconf()):
        # Load config parser
        self.cfg = cfg
        self.model_name = self.cfg.get('Network', 'name')
        self.dataset_name = self.cfg.get('DataFrame', 'dataset_name')

        # Get host name and date
        self.hostname = socket.gethostname()
        print('HOSTNAME: ' + self.hostname)
        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")
        print(self.date)

        # Read STFT parameters
        self.wlen_sec = self.cfg.getfloat('STFT', 'wlen_sec')
        self.hop_percent = self.cfg.getfloat('STFT', 'hop_percent')
        self.fs = self.cfg.getint('STFT', 'fs')
        self.zp_percent = self.cfg.getfloat('STFT', 'zp_percent')
        self.trim = self.cfg.getboolean('STFT', 'trim')
        self.verbose = self.cfg.getboolean('STFT', 'verbose')

        # Load training parameters
        self.lr = self.cfg.getfloat('Training', 'lr')
        self.epochs = self.cfg.getint('Training', 'epochs')
        self.batch_size = self.cfg.getint('Training', 'batch_size')
        self.optimization  = self.cfg.get('Training', 'optimization')
        self.early_stop_patience = self.cfg.getint('Training', 'early_stop_patience')
        self.save_frequency = self.cfg.getint('Training', 'save_frequency')

        # Create directory (saved_model) if not exist
        self.local_hostname = self.cfg.get('User', 'local_hostname')
        if self.hostname == self.local_hostname:
            self.path_prefix = self.cfg.get('Path', 'path_local')
        # ===== develop on Mac, temporarily =====
        elif self.hostname == 'MacPro-BIE.local': 
            self.path_prefix = '/Users/xiaoyu/WorkStation/saved_model'
        # ===== develop on Mac, temporarily =====
        else: 
            self.path_prefix = self.cfg.get('Path', 'path_cluster')
        if not(os.path.isdir(self.path_prefix)):
            print('No saved directory exists, new one will be generated')
            os.makedirs(self.path_prefix)
            print('All training results will be saved in: ' + self.path_prefix)

        # Choose to use gpu or cpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Device for training: ' + self.device)


class BuildFFNN(BuildBasic):

    """
    Feed-forward fully-connected VAE
    Implementation of FFNN in rvae
    """

    def __init__(self, cfg=myconf()):
        
        super(BuildFFNN, self).__init__(cfg)

        # Load network parameters
        self.x_dim = self.cfg.getint('Network', 'x_dim')
        self.z_dim = self.cfg.getint('Network','z_dim')
        self.hidden_dim_encoder = [int(i) for i in self.cfg.get('Network', 'hidden_dim_encoder').split(',')] # this parameter is a python list
        self.activation = eval(self.cfg.get('Network', 'activation'))

        # Create directory for this training
        dir_name = self.dataset_name + '_' + self.date  + '_FFNN_VAE_z_dim=' + str(self.z_dim)
        self.save_dir = os.path.join(self.path_prefix, dir_name)
        if not(os.path.isdir(self.save_dir)):
            os.makedirs(self.save_dir)
        print('In this experiment, result will be saved in: ' + self.save_dir)
        

    def build_net(self):

        # Init VAE model
        print('===== Init VAE =====')
        self.model = VAE(x_dim = self.x_dim,
                         z_dim = self.z_dim,
                         hidden_dim_encoder = self.hidden_dim_encoder,
                         batch_size = self.batch_size,
                         activation = self.activation).to(self.device)
        
        # Define optimizer (Adam by default)
        if self.optimization == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return self.model, self.optimizer

    def loss_function(self, recon_x, x, mu, logvar):
        recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
        KLD = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp())
        return recon + KLD

    def build_dataloader(self):
        # Find data set
        #train_data_dir, val_data_dir = find_dataset()
        self.train_data_dir = self.cfg.get('Path', 'train_data_dir')
        self.val_data_dir = self.cfg.get('Path', 'val_data_dir')
        
        # List all the data with certain suffix
        self.data_suffix = self.cfg.get('DataFrame', 'suffix')
        self.train_file_list = librosa.util.find_files(self.train_data_dir, ext=self.data_suffix)
        self.val_file_list = librosa.util.find_files(self.val_data_dir, ext=self.data_suffix)
        
        # Generate dataloader for pytorch
        self.num_workers = self.cfg.getint('DataFrame', 'num_workers')
        self.shuffle_file_list = self.cfg.get('DataFrame', 'shuffle_file_list')
        self.shuffle_samples_in_batch = self.cfg.get('DataFrame', 'shuffle_samples_in_batch')

        print('===== Instranciate training dataloader =====')
        train_dataset = SpeechDatasetFrames(file_list = self.train_file_list,
                                            wlen_sec = self.wlen_sec,
                                            hop_percent = self.hop_percent,
                                            fs = self.fs,
                                            zp_percent = self.zp_percent,
                                            trim = self.trim,
                                            verbose = self.verbose,
                                            batch_size = self.batch_size,
                                            shuffle_file_list = self.shuffle_file_list,
                                            name = self.dataset_name)
        self.train_num = train_dataset.num_samples
        print('===== Instanciate validation dataloader =====')
        val_dataset = SppechDatasetFrames(file_list = self.val_fil_list,
                                          wlen_sec = self.wlen_sec,
                                          hop_percent = self.hop_percent,
                                          fs = self.fs,
                                          zp_percent = self.zp_percent,
                                          trim = self.trim,
                                          verbose = self.verbose,
                                          batch_size = self.batch_size,
                                          shuffle_file_list = self.shuffle_file_list,
                                          name = self.dataset_name)
        self.val_num = val_dataset = val_dataset.num_samples
        print('===== Create training dataloader =====')
        self.train_dataloader = torch.data.DataLoader(train_dataloader, 
                                                      batch_size=self.batch_size,
                                                      shuffle=self.shuffle_samples_in_batch,
                                                      num_workers = self.num_workers)
        print('===== Create validation dataloader =====')
        self.val_dataloader = torch.data.DataLoader(val_dataset, 
                                                    batch_size=self.batch_size,
                                                    shuffle=self.shuffle_samples_in_batch,
                                                    num_workers = self.num_workers)
        return self.train_dataloader, self.val_dataloader, self.train_num, self.val_num

class BuildRNN(BuildBasic):
    pass

def build_model(config_file='config_default.ini'):
    cfg = myconf()
    cfg.read(config_file)
    model_name = cfg.get('Network', 'name')
    if model_name == 'VAE-FFNN':
        model = BuildFFNN(cfg)
    
    return model


if __name__ == '__main__':
    model_class = build_model('config_rvae-ffnn.ini')
    model, optimizer = model_class.build_net()
    model.print_model()
