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
from torch.utils import data

import librosa
from configparser import ConfigParser
from logger import get_logger
from model.vae import VAE
from model.dmm import DMM
from model.storn import STORN
from model.vrnn import VRNN
from model.srnn import SRNN
from model.rvae import RVAE
from model.dsae import DSAE
from model.kvae import KVAE


from backup_simon.speech_dataset import *



# Re-write configure class, enable to distinguish betwwen upper and lower letters
class myconf(ConfigParser):
    def __init__(self,defaults=None):
        ConfigParser.__init__(self, defaults=None)
    def optionxform(self, optionstr):
        return optionstr


class BuildBasic():

    """
    Basical class for model building, including:
    - read common paramters for different models
    - define data loader
    - define loss function as a class member
    """

    def __init__(self, cfg = myconf(), training=True):

        # 1. Load config parser
        self.cfg = cfg
        self.model_name = self.cfg.get('Network', 'name')
        self.dataset_name = self.cfg.get('DataFrame', 'dataset_name')

        # 2. Get host name and date
        self.hostname = socket.gethostname()
        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")

        # 3. Get logger type
        self.logger_type = self.cfg.getint('User', 'logger_type')

        # 4. Load STFT parameters
        self.wlen_sec = self.cfg.getfloat('STFT', 'wlen_sec')
        self.hop_percent = self.cfg.getfloat('STFT', 'hop_percent')
        self.fs = self.cfg.getint('STFT', 'fs')
        self.zp_percent = self.cfg.getint('STFT', 'zp_percent')
        self.trim = self.cfg.getboolean('STFT', 'trim')
        self.verbose = self.cfg.getboolean('STFT', 'verbose')

        # 5. Load training parameters
        self.use_cuda = self.cfg.getboolean('Training', 'use_cuda')
        self.lr = self.cfg.getfloat('Training', 'lr')
        self.epochs = self.cfg.getint('Training', 'epochs')
        self.batch_size = self.cfg.getint('Training', 'batch_size')
        self.sequence_len = self.cfg.getint('DataFrame','sequence_len')
        self.optimization  = self.cfg.get('Training', 'optimization')
        self.early_stop_patience = self.cfg.getint('Training', 'early_stop_patience')
        self.save_frequency = self.cfg.getint('Training', 'save_frequency')

        # 6. Create saved_model directory if not exist, and find dataset
        self.saved_root = self.cfg.get('User', 'saved_root')
        self.train_data_dir = self.cfg.get('User', 'train_data_dir')
        self.val_data_dir = self.cfg.get('User', 'val_data_dir')

        # 7. Choose to use gpu or cpu
        self.device = 'cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu'

        # 8. Get model tag, used in loss figure and evaluation table
        if self.cfg.has_option('Network', 'tag'):
            self.tag = self.cfg.get('Network', 'tag')
        else:
            self.tag = '{}'.format(self.model_name)

        # 9. Training/Evaluation
        self.training = training
        if self.training:
             # Create directory for results
            self.z_dim = self.cfg.getint('Network','z_dim')
            self.filename = "{}_{}_{}_z_dim={}".format(self.dataset_name, 
                                                   self.date, 
                                                   self.tag, 
                                                   self.z_dim)
        
            self.save_dir = os.path.join(self.saved_root, self.filename)
            if not(os.path.isdir(self.save_dir)):
                os.makedirs(self.save_dir)

            # Create logger
            log_file = os.path.join(self.save_dir, 'log.txt')
            logger = get_logger(log_file, self.logger_type)
            for log in self.get_basic_info():
                logger.info(log)
            logger.info('In this experiment, result will be saved in: ' + self.save_dir)
            self.logger = logger


    # only used for sequential vae modles
    def build_dataloader(self):
        # List all the data with certain suffix
        self.data_suffix = self.cfg.get('DataFrame', 'suffix')
        self.train_file_list = librosa.util.find_files(self.train_data_dir, ext=self.data_suffix)
        self.val_file_list = librosa.util.find_files(self.val_data_dir, ext=self.data_suffix)
        # Generate dataloader for pytorch
        self.num_workers = self.cfg.getint('DataFrame', 'num_workers')
        self.shuffle_file_list = self.cfg.get('DataFrame', 'shuffle_file_list')
        self.shuffle_samples_in_batch = self.cfg.get('DataFrame', 'shuffle_samples_in_batch')

        # Instantiate dataloader
        train_dataset = SpeechDatasetSequences(file_list=self.train_file_list, sequence_len=self.sequence_len,
                                               wlen_sec=self.wlen_sec, hop_percent=self.hop_percent, fs=self.fs,
                                               zp_percent=self.zp_percent, trim=self.trim, verbose=self.verbose,
                                               batch_size=self.batch_size, shuffle_file_list=self.shuffle_file_list,
                                               name=self.dataset_name)
        val_dataset = SpeechDatasetSequences(file_list=self.val_file_list, sequence_len=self.sequence_len,
                                             wlen_sec=self.wlen_sec, hop_percent=self.hop_percent, fs=self.fs,
                                             zp_percent=self.zp_percent, trim=self.trim, verbose=self.verbose,
                                             batch_size=self.batch_size, shuffle_file_list=self.shuffle_file_list,
                                             name=self.dataset_name)
        train_num = train_dataset.num_samples
        val_num = val_dataset.num_samples

        # Create dataloader
        train_dataloader = data.DataLoader(train_dataset, batch_size=self.batch_size, 
                                           shuffle=self.shuffle_samples_in_batch,
                                           num_workers = self.num_workers)
        val_dataloader = data.DataLoader(val_dataset, batch_size=self.batch_size,
                                         shuffle=self.shuffle_samples_in_batch,
                                         num_workers = self.num_workers)

        return train_dataloader, val_dataloader, train_num, val_num


    def get_basic_info(self):
        basic_info = []
        basic_info.append('HOSTNAME: ' + self.hostname)
        basic_info.append('Time: ' + self.date)
        basic_info.append('Training results will be saved in: ' + self.save_dir)
        basic_info.append('Device for training: ' + self.device)
        if self.device == 'cuda':
            basic_info.append('Cuda verion: {}'.format(torch.version.cuda))
        basic_info.append('Model name: {}'.format(self.model_name))
        return basic_info



class BuildVAE(BuildBasic):

    def __init__(self, cfg=myconf(), training=True):
        
        super().__init__(cfg, training)

        ### Load parameters
        # General
        self.x_dim = self.cfg.getint('Network', 'x_dim')
        self.z_dim = self.cfg.getint('Network','z_dim')
        self.activation = self.cfg.get('Network', 'activation')
        self.dropout_p = self.cfg.getfloat('Network', 'dropout_p')
        # Inference and generation
        self.dense_x_z = [int(i) for i in self.cfg.get('Network', 'dense_x_z').split(',')]

        self.build()


    def build(self):

        # Build model
        self.model = VAE(x_dim=self.x_dim, z_dim=self.z_dim,
                         dense_x_z=self.dense_x_z, activation=self.activation,
                         dropout_p=self.dropout_p, device=self.device).to(self.device)
        
        # Print model information
        if self.training:
            self.logger.info('===== Init VAE =====')
            for log in self.model.get_info():
                self.logger.info(log)

        # Init optimizer (Adam by default)
        if self.optimization == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def build_dataloader(self):
        # List all the data with certain suffix
        self.data_suffix = self.cfg.get('DataFrame', 'suffix')
        self.train_file_list = librosa.util.find_files(self.train_data_dir, ext=self.data_suffix)
        self.val_file_list = librosa.util.find_files(self.val_data_dir, ext=self.data_suffix)
        # Generate dataloader for pytorch
        self.num_workers = self.cfg.getint('DataFrame', 'num_workers')
        self.shuffle_file_list = self.cfg.get('DataFrame', 'shuffle_file_list')
        self.shuffle_samples_in_batch = self.cfg.get('DataFrame', 'shuffle_samples_in_batch')

        # Instantiate dataloader
        train_dataset = SpeechDatasetFrames(file_list=self.train_file_list,
                                            wlen_sec=self.wlen_sec, hop_percent=self.hop_percent, fs=self.fs,
                                            zp_percent=self.zp_percent, trim=self.trim, verbose=self.verbose,
                                            batch_size=self.batch_size, shuffle_file_list=self.shuffle_file_list,
                                            name=self.dataset_name)
        val_dataset = SpeechDatasetFrames(file_list=self.val_file_list,
                                          wlen_sec=self.wlen_sec, hop_percent=self.hop_percent, fs=self.fs,
                                          zp_percent=self.zp_percent, trim=self.trim, verbose=self.verbose,
                                          batch_size=self.batch_size, shuffle_file_list=self.shuffle_file_list,
                                          name=self.dataset_name)
        train_num = train_dataset.num_samples
        val_num = val_dataset.num_samples

        # Create dataloader
        train_dataloader = data.DataLoader(train_dataset, batch_size=self.batch_size, 
                                           shuffle=self.shuffle_samples_in_batch,
                                           num_workers = self.num_workers)
        val_dataloader = data.DataLoader(val_dataset, batch_size=self.batch_size,
                                         shuffle=self.shuffle_samples_in_batch,
                                         num_workers = self.num_workers)

        return train_dataloader, val_dataloader, train_num, val_num




class BuildDMM(BuildBasic):

    def __init__(self, cfg=myconf(), training=True):

        super().__init__(cfg, training)

        ### Load parameters
        # General
        self.x_dim = self.cfg.getint('Network', 'x_dim')
        self.z_dim = self.cfg.getint('Network','z_dim')
        self.activation = self.cfg.get('Network', 'activation')
        self.dropout_p = self.cfg.getfloat('Network', 'dropout_p')
        # Inference
        self.dense_x_gx = [int(i) for i in self.cfg.get('Network', 'dense_x_gx').split(',')]
        self.dim_RNN_gx = self.cfg.getint('Network', 'dim_RNN_gx')
        self.num_RNN_gx = self.cfg.getint('Network', 'num_RNN_gx')
        self.bidir_gx = self.cfg.getboolean('Network', 'bidir_gx')
        self.dense_ztm1_g = [int(i) for i in self.cfg.get('Network', 'dense_ztm1_g').split(',')]
        self.dense_g_z = [int(i) for i in self.cfg.get('Network', 'dense_g_z').split(',')]
        # Generation
        self.dense_z_x = [int(i) for i in self.cfg.get('Network', 'dense_z_x').split(',')]

        self.build()


    def build(self):

        # Build model
        self.model = DMM(x_dim=self.x_dim, z_dim=self.z_dim, activation=self.activation,
                         dense_x_gx=self.dense_x_gx, dim_RNN_gx=self.dim_RNN_gx, 
                         num_RNN_gx=self.num_RNN_gx, bidir_gx=self.bidir_gx,
                         dense_ztm1_g=self.dense_ztm1_g, dense_g_z=self.dense_g_z,
                         dense_z_x=self.dense_z_x,
                         dropout_p=self.dropout_p, device=self.device).to(self.device)
        
        # Print model information
        if self.training:
            self.logger.info('===== Init DMM =====')
            for log in self.model.get_info():
                self.logger.info(log)
            
        # Init optimizer (Adam by default)
        if self.optimization == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)



class BuildSTORN(BuildBasic):

    def __init__(self, cfg=myconf(), training=True):

        super().__init__(cfg, training)

        ### Load parameters for STORN
        # General
        self.x_dim = self.cfg.getint('Network', 'x_dim')
        self.z_dim = self.cfg.getint('Network','z_dim')
        self.activation = self.cfg.get('Network', 'activation')
        self.dropout_p = self.cfg.getfloat('Network', 'dropout_p')
        # Inference
        self.dense_x_g = [int(i) for i in self.cfg.get('Network', 'dense_x_g').split(',')]
        self.dim_RNN_g = self.cfg.getint('Network', 'dim_RNN_g')
        self.num_RNN_g = self.cfg.getint('Network', 'num_RNN_g')
        self.dense_g_z = [int(i) for i in self.cfg.get('Network', 'dense_g_z').split(',')]
        # Generation
        self.dense_z_h = [int(i) for i in self.cfg.get('Network', 'dense_z_h').split(',')]
        self.dense_xtm1_h = [int(i) for i in self.cfg.get('Network', 'dense_xtm1_h').split(',')]
        self.dim_RNN_h = self.cfg.getint('Network', 'dim_RNN_h')
        self.num_RNN_h = self.cfg.getint('Network', 'num_RNN_h')
        self.dense_h_x = [int(i) for i in self.cfg.get('Network', 'dense_h_x').split(',')]
        
        ### Beta-vae
        self.beta = self.cfg.getfloat('Training', 'beta')

        self.build()
    

    def build(self):

        # Build model
        self.model = STORN(x_dim=self.x_dim, z_dim=self.z_dim, activation=self.activation,
                           dense_x_g=self.dense_x_g, dense_g_z=self.dense_g_z,
                           dim_RNN_g=self.dim_RNN_g, num_RNN_g=self.num_RNN_g,
                           dense_z_h=self.dense_z_h, dense_xtm1_h=self.dense_xtm1_h,
                           dense_h_x=self.dense_h_x,
                           dim_RNN_h=self.dim_RNN_h, num_RNN_h=self.num_RNN_h,
                           dropout_p=self.dropout_p, beta=self.beta, device=self.device).to(self.device)
        
        # Print model information
        if self.training:
            self.logger.info('===== Init STORN =====')
            for log in self.model.get_info():
                self.logger.info(log)
            
        # Init optimizer (Adam by default)
        if self.optimization == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)



class BuildVRNN(BuildBasic):

    def __init__(self, cfg=myconf(), training=True):

        super().__init__(cfg, training)

        ### Load parameters for VRNN
        # General
        self.x_dim = self.cfg.getint('Network', 'x_dim')
        self.z_dim = self.cfg.getint('Network','z_dim')
        self.activation = self.cfg.get('Network', 'activation')
        self.dropout_p = self.cfg.getfloat('Network', 'dropout_p')
        # Feature extractor
        self.dense_x = [int(i) for i in self.cfg.get('Network', 'dense_x').split(',')]
        self.dense_z = [int(i) for i in self.cfg.get('Network', 'dense_z').split(',')]
        # Dense layers
        self.dense_hx_z = [int(i) for i in self.cfg.get('Network', 'dense_hx_z').split(',')]
        self.dense_hz_x = [int(i) for i in self.cfg.get('Network', 'dense_hz_x').split(',')]
        self.dense_h_z = [int(i) for i in self.cfg.get('Network', 'dense_h_z').split(',')]
        # RNN
        self.dim_RNN = self.cfg.getint('Network', 'dim_RNN')
        self.num_RNN = self.cfg.getint('Network', 'num_RNN')

        self.build()


    def build(self):
        
        # Build model
        self.model = VRNN(x_dim=self.x_dim, z_dim=self.z_dim, 
                          activation=self.activation,
                          dense_x=self.dense_x, dense_z=self.dense_z,
                          dense_hx_z=self.dense_hx_z, dense_hz_x=self.dense_hz_x, 
                          dense_h_z=self.dense_h_z,
                          dim_RNN=self.dim_RNN, num_RNN=self.num_RNN,
                          dropout_p= self.dropout_p,
                          device=self.device).to(self.device)

        # Print model information
        if self.training:
            self.logger.info('===== Init VRNN =====')
            for log in self.model.get_info():
                self.logger.info(log)
            
        # Init optimizer (Adam by default)
        if self.optimization == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)



class BuildSRNN(BuildBasic):

    def __init__(self, cfg=myconf(), training=True):

        super().__init__(cfg, training)

        ### Load parameters for SRNN
        # General
        self.x_dim = self.cfg.getint('Network', 'x_dim')
        self.z_dim = self.cfg.getint('Network','z_dim')
        self.activation = self.cfg.get('Network', 'activation')
        self.dropout_p = self.cfg.getfloat('Network', 'dropout_p')
        # Deterministic
        self.dense_x_h = [int(i) for i in self.cfg.get('Network', 'dense_x_h').split(',')]
        self.dim_RNN_h = self.cfg.getint('Network', 'dim_RNN_h')
        self.num_RNN_h = self.cfg.getint('Network', 'num_RNN_h')
        # Inference
        self.dense_hx_g = [int(i) for i in self.cfg.get('Network', 'dense_hx_g').split(',')]
        self.dim_RNN_g = self.cfg.getint('Network', 'dim_RNN_g')
        self.num_RNN_g = self.cfg.getint('Network', 'num_RNN_g')
        self.dense_gz_z = [int(i) for i in self.cfg.get('Network', 'dense_gz_z').split(',')]
        # Prior
        self.dense_hz_z = [int(i) for i in self.cfg.get('Network', 'dense_hz_z').split(',')]
        # Generation
        self.dense_hz_x = [int(i) for i in self.cfg.get('Network', 'dense_hz_x').split(',')]

        self.build()
 

    def build(self):

        # Build model
        self.model = SRNN(x_dim=self.x_dim, z_dim=self.z_dim, activation=self.activation,
                          dense_x_h=self.dense_x_h,
                          dim_RNN_h=self.dim_RNN_h, num_RNN_h=self.num_RNN_h,
                          dense_hx_g=self.dense_hx_g,
                          dim_RNN_g=self.dim_RNN_g, num_RNN_g=self.num_RNN_g,
                          dense_gz_z=self.dense_gz_z,
                          dense_hz_x=self.dense_hz_x,
                          dense_hz_z=self.dense_hz_z,
                          dropout_p=self.dropout_p,
                          device=self.device).to(self.device)
        
        # Print model information
        if self.training:
            self.logger.info('===== Init SRNN =====')
            for log in self.model.get_info():
                self.logger.info(log)
            
        # Init optimizer (Adam by default)
        if self.optimization == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)



class BuildRVAE(BuildBasic):

    def __init__(self, cfg = myconf(), training=True):

        super().__init__(cfg, training)

        ### Load special paramters for RVAE
        # General
        self.x_dim = self.cfg.getint('Network', 'x_dim')
        self.z_dim = self.cfg.getint('Network','z_dim')
        self.activation = self.cfg.get('Network', 'activation')
        self.dropout_p = self.cfg.getfloat('Network', 'dropout_p')
        # Inference
        self.dense_x_gx = [int(i) for i in self.cfg.get('Network', 'dense_x_gx').split(',')]
        self.dim_RNN_g_x = self.cfg.getint('Network', 'dim_RNN_g_x')
        self.num_RNN_g_x = self.cfg.getint('Network', 'num_RNN_g_x')
        self.bidir_g_x = self.cfg.getboolean('Network', 'bidir_g_x')
        self.dense_z_gz = [int(i) for i in self.cfg.get('Network', 'dense_z_gz').split(',')]
        self.dim_RNN_g_z = self.cfg.getint('Network', 'dim_RNN_g_z')
        self.num_RNN_g_z = self.cfg.getint('Network', 'num_RNN_g_z')
        self.dense_g_z = [int(i) for i in self.cfg.get('Network', 'dense_g_z').split(',')]
        # Generation
        self.dense_z_h = [int(i) for i in self.cfg.get('Network', 'dense_z_h').split(',')]
        self.dim_RNN_h = self.cfg.getint('Network', 'dim_RNN_h')
        self.num_RNN_h = self.cfg.getint('Network', 'num_RNN_h')
        self.bidir_h = self.cfg.getboolean('Network', 'bidir_h')
        self.dense_h_x = [int(i) for i in self.cfg.get('Network', 'dense_h_x').split(',')]

        self.build()
    

    def build(self):
        
        # Build model
        self.model = RVAE(x_dim=self.x_dim, z_dim=self.z_dim, activation=self.activation,
                          dense_x_gx=self.dense_x_gx,
                          dim_RNN_g_x=self.dim_RNN_g_x, num_RNN_g_x=self.num_RNN_g_x,
                          bidir_g_x=self.bidir_g_x, 
                          dense_z_gz=self.dense_z_gz,
                          dim_RNN_g_z=self.dim_RNN_g_z, num_RNN_g_z=self.num_RNN_g_z,
                          dense_g_z=self.dense_g_z,
                          dense_z_h=self.dense_z_h,
                          dim_RNN_h=self.dim_RNN_h, num_RNN_h=self.num_RNN_h,
                          bidir_h=self.bidir_h,
                          dense_h_x=self.dense_h_x,
                          dropout_p=self.dropout_p, device=self.device).to(self.device)

        # Print model information
        if self.training:
            self.logger.info('===== Init RVAE =====')
            for log in self.model.get_info():
                self.logger.info(log)
            
        # Init optimizer (Adam by default)
        if self.optimization == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)



class BuildDSAE(BuildBasic):
    
    def __init__(self, cfg=myconf(), training=True):

        super().__init__(cfg, training)

        ### Load special parameters for DSAE
        # General
        self.x_dim = self.cfg.getint('Network', 'x_dim')
        self.z_dim = self.cfg.getint('Network','z_dim')
        self.v_dim = self.cfg.getint('Network','v_dim')
        self.activation = self.cfg.get('Network', 'activation')
        self.dropout_p = self.cfg.getfloat('Network', 'dropout_p')
        # Inference
        self.dense_x = [int(i) for i in self.cfg.get('Network', 'dense_x').split(',')]
        self.dim_RNN_gv = self.cfg.getint('Network', 'dim_RNN_gv')
        self.num_RNN_gv = self.cfg.getint("Network", 'num_RNN_gv')
        self.dense_gv_v = [int(i) for i in self.cfg.get('Network', 'dense_gv_v').split(',')]
        self.dense_xv_gxv = [int(i) for i in self.cfg.get('Network', 'dense_xv_gxv').split(',')]
        self.dim_RNN_gxv = self.cfg.getint('Network', 'dim_RNN_gxv')
        self.num_RNN_gxv = self.cfg.getint('Network', 'num_RNN_gxv')
        self.dense_gxv_gz = [int(i) for i in self.cfg.get('Network', 'dense_gxv_gz').split(',')]
        self.dim_RNN_gz = self.cfg.getint('Network', 'dim_RNN_gz')
        self.num_RNN_gz = self.cfg.getint('Network', 'num_RNN_gz')
        # Prior
        self.dim_RNN_prior = self.cfg.getint('Network', 'dim_RNN_prior')
        self.num_RNN_prior = self.cfg.getint('Network', 'num_RNN_prior')
        # Generation
        self.dense_vz_x = [int(i) for i in self.cfg.get('Network', 'dense_vz_x').split(',')]
        
        self.build()


    def build(self):

        # Build model
        self.model = DSAE(x_dim=self.x_dim, z_dim=self.z_dim, v_dim=self.v_dim, 
                          activation=self.activation,
                          dense_x=self.dense_x,
                          dim_RNN_gv=self.dim_RNN_gv, num_RNN_gv=self.num_RNN_gv,
                          dense_gv_v=self.dense_gv_v, dense_xv_gxv=self.dense_xv_gxv,
                          dim_RNN_gxv=self.dim_RNN_gxv, num_RNN_gxv=self.num_RNN_gxv,
                          dense_gxv_gz=self.dense_gxv_gz,
                          dim_RNN_gz=self.dim_RNN_gz, num_RNN_gz=self.num_RNN_gz,
                          dim_RNN_prior=self.dim_RNN_prior, num_RNN_prior=self.num_RNN_prior,
                          dense_vz_x=self.dense_vz_x,
                          dropout_p=self.dropout_p, device=self.device).to(self.device)
        
        # Print model information
        if self.training:
            self.logger.info('==== Init DSAE ====')
            for log in self.model.get_info():
                self.logger.info(log)

        # Init optimizer (Adam by default):
        if self.optimization == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)



class BuildKVAE(BuildBasic):

    def __init__(self, cfg=myconf(), training=True):

        super().__init__(cfg, training)

        ### Load special parameters for KVAE
        # General
        self.x_dim = self.cfg.getint('Network', 'x_dim')
        self.a_dim = self.cfg.getint('Network', 'a_dim')
        self.z_dim = self.cfg.getint('Network', 'z_dim')
        self.activation = self.cfg.get('Network', 'activation')
        self.dropout_p = self.cfg.getfloat('Network', 'dropout_p')
        self.scale_reconstruction = self.cfg.getfloat('Network', 'scale_reconstruction')
        # VAE
        self.dense_x_a = [int(i) for i in self.cfg.get('Network', 'dense_x_a').split(',')]
        self.dense_a_x = [int(i) for i in self.cfg.get('Network', 'dense_a_x').split(',')]
        # LGSSM
        self.init_kf_mat = self.cfg.getfloat('Network', 'init_kf_mat')
        self.noise_transition = self.cfg.getfloat('Network', 'noise_transition')
        self.noise_emission = self.cfg.getfloat('Network', 'noise_emission')
        self.init_cov = self.cfg.getfloat('Network', 'init_cov')
        # Dynamics
        self.K = self.cfg.getint('Network', 'K')
        self.dim_RNN_alpha = self.cfg.getint('Network', 'dim_RNN_alpha')
        self.num_RNN_alpha = self.cfg.getint('Network', 'num_RNN_alpha')
        # Training set
        self.scheduler_training = self.cfg.getboolean('Training', 'scheduler_training')
        self.only_vae_epochs = self.cfg.getint('Training', 'only_vae_epochs')
        self.kf_update_epochs = self.cfg.getint('Training', 'kf_update_epochs')
        self.lr_tot = self.cfg.getfloat('Training', 'lr_tot')
        
        self.build()


    def build(self):

        # Build model
        self.model = KVAE(x_dim=self.x_dim, a_dim=self.a_dim, z_dim=self.z_dim, activation=self.activation,
                          dense_x_a=self.dense_x_a, dense_a_x=self.dense_a_x,
                          init_kf_mat=self.init_kf_mat, noise_transition=self.noise_transition,
                          noise_emission=self.noise_emission, init_cov=self.init_cov,
                          K=self.K, dim_RNN_alpha=self.dim_RNN_alpha, num_RNN_alpha=self.num_RNN_alpha,
                          dropout_p=self.dropout_p, scale_reconstruction = self.scale_reconstruction,
                          device=self.device).to(self.device)
        
        # Print model information
        if self.training:
            self.logger.info('==== Init KVAE ====')
            for log in self.model.get_info():
                self.logger.info(log)

        # Init optimizer (Adam by default):
        if self.optimization == 'adam':
            self.optimizer_vae = torch.optim.Adam(self.model.vars_vae, lr=self.lr)
            self.optimizer_vae_kf = torch.optim.Adam(self.model.vars_vae+self.model.vars_kf, lr=self.lr)
            self.optimizer_net = torch.optim.Adam(self.model.vars_vae+self.model.vars_alpha, lr=self.lr)
            self.optimizer_lgssm =  torch.optim.Adam(self.model.vars_kf+self.model.vars_alpha, lr=self.lr)
            self.optimizer_all = torch.optim.Adam(self.model.vars_vae+self.model.vars_kf+self.model.vars_alpha, lr=self.lr_tot)

        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)



def build_model(config_file='config_default.ini', training=True):

    if not os.path.isfile(config_file):
        raise ValueError('Invalid config file path')    
    cfg = myconf()
    cfg.read(config_file)
    model_name = cfg.get('Network', 'name')

    if model_name == 'VAE':
        model_class = BuildVAE(cfg, training)
    elif model_name == 'DMM':
        model_class = BuildDMM(cfg, training)
    elif model_name == 'STORN':
        model_class = BuildSTORN(cfg, training)
    elif model_name == 'VRNN':
        model_class = BuildVRNN(cfg, training)
    elif model_name == 'SRNN':
        model_class = BuildSRNN(cfg, training)
    elif model_name == 'RVAE':
        model_class = BuildRVAE(cfg, training)
    elif model_name == 'DSAE':
        model_class = BuildDSAE(cfg, training)
    elif model_name == 'KVAE':
        model_class = BuildKVAE(cfg, training)

    return model_class


if __name__ == '__main__':
    model_class = build_model('config/cfg_kvae.ini', False)
    model = model_class.model
    print(model_class.lr_tot)
