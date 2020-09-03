#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""


import os
import shutil
import socket
import datetime
import pickle
import numpy as np
import torch
import librosa
import soundfile as sf
import speechmetrics
from utils import myconf, get_logger, rmse_frame, SpeechSequencesFull, SpeechSequencesRandom
from build_model import build_VAE, build_DKF, build_KVAE, build_STORN, build_VRNN, build_SRNN, build_RVAE, build_DSAE



class LearningAlgorithm():

    """
    Basical class for model building, including:
    - read common paramters for different models
    - define data loader
    - define loss function as a class member
    """

    def __init__(self, config_file='config_default.ini'):

        # Load config parser
        self.config_file = config_file
        if not os.path.isfile(self.config_file):
            raise ValueError('Invalid config file path')    
        self.cfg = myconf()
        self.cfg.read(self.config_file)
        self.model_name = self.cfg.get('Network', 'name')
        self.dataset_name = self.cfg.get('DataFrame', 'dataset_name')

        # Get host name and date
        self.hostname = socket.gethostname()
        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")
        
        # Load STFT parameters
        wlen_sec = self.cfg.getfloat('STFT', 'wlen_sec')
        hop_percent = self.cfg.getfloat('STFT', 'hop_percent')
        fs = self.cfg.getint('STFT', 'fs')
        zp_percent = self.cfg.getint('STFT', 'zp_percent')
        wlen = wlen_sec * fs
        wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # pwoer of 2
        hop = np.int(hop_percent * wlen)
        nfft = wlen + zp_percent * wlen
        win = torch.sin(torch.arange(0.5, wlen+0.5) / wlen * np.pi)

        STFT_dict = {}
        STFT_dict['fs'] = fs
        STFT_dict['wlen'] = wlen
        STFT_dict['hop'] = hop
        STFT_dict['nfft'] = nfft
        STFT_dict['win'] = win
        STFT_dict['trim'] = self.cfg.getboolean('STFT', 'trim')
        STFT_dict['verbose'] = self.cfg.getboolean('STFT', 'verbose')
        self.STFT_dict = STFT_dict

        # Load model parameters
        self.use_cuda = self.cfg.getboolean('Training', 'use_cuda')
        self.device = 'cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu'

        # Build model
        self.build_model()


    def build_model(self):

        if self.model_name == 'VAE':
            self.model = build_VAE(cfg=self.cfg, device=self.device)
        elif self.model_name == 'DKF':
            self.model = build_DKF(cfg=self.cfg, device=self.device)
        elif self.model_name == 'KVAE':
            self.model = build_KVAE(cfg=self.cfg, device=self.device)
        elif self.model_name == 'STORN':
            self.model = build_STORN(cfg=self.cfg, device=self.device)
        elif self.model_name == 'VRNN':
            self.model = build_VRNN(cfg=self.cfg, device=self.device)
        elif self.model_name == 'SRNN':
            self.model = build_SRNN(cfg=self.cfg, device=self.device)
        elif self.model_name == 'RVAE':
            self.model = build_RVAE(cfg=self.cfg, device=self.device)
        elif self.model_name == 'DSAE':
            self.model = build_DSAE(cfg=self.cfg, device=self.device)
        

    def init_optimizer(self):

        # Load 
        self.optimization  = self.cfg.get('Training', 'optimization')
        lr = self.cfg.getfloat('Training', 'lr')
        
        # Init optimizer (Adam by default)
        if self.model_name=='KVAE':
            lr_tot = self.cfg.getfloat('Training', 'lr_tot')
            if self.optimization == 'adam':
                self.optimizer_vae = torch.optim.Adam(self.model.vars_vae, lr=lr)
                self.optimizer_net = torch.optim.Adam(self.model.vars_vae+self.model.vars_alpha, lr=lr)
                self.optimizer_lgssm =  torch.optim.Adam(self.model.vars_kf+self.model.vars_alpha, lr=lr)
                self.optimizer_vae_kf = torch.optim.Adam(self.model.vars_vae+self.model.vars_kf, lr=lr_tot)
                self.optimizer_all = torch.optim.Adam(self.model.vars_vae+self.model.vars_kf+self.model.vars_alpha, lr=lr_tot)
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        else:
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)


    def build_dataloader(self, train_data_dir, val_data_dir, sequence_len, batch_size, STFT_dict, use_random_seq=False):

        # List all the data with certain suffix
        data_suffix = self.cfg.get('DataFrame', 'suffix')
        train_file_list = librosa.util.find_files(train_data_dir, ext=data_suffix)
        val_file_list = librosa.util.find_files(val_data_dir, ext=data_suffix)
        # Generate dataloader for pytorch
        num_workers = self.cfg.getint('DataFrame', 'num_workers')
        shuffle_file_list = self.cfg.get('DataFrame', 'shuffle_file_list')
        shuffle_samples_in_batch = self.cfg.get('DataFrame', 'shuffle_samples_in_batch')

        # Instantiate dataloader
        if use_random_seq:
            train_dataset = SpeechSequencesRandom(file_list=train_file_list, sequence_len=sequence_len,
                                                  STFT_dict=self.STFT_dict, shuffle=shuffle_file_list, name=self.dataset_name)
            val_dataset = SpeechSequencesRandom(file_list=val_file_list, sequence_len=sequence_len,
                                                STFT_dict=self.STFT_dict, shuffle=shuffle_file_list, name=self.dataset_name)
        else:
            train_dataset = SpeechSequencesFull(file_list=train_file_list, sequence_len=sequence_len,
                                                STFT_dict=self.STFT_dict, shuffle=shuffle_file_list, name=self.dataset_name)
            val_dataset = SpeechSequencesFull(file_list=val_file_list, sequence_len=sequence_len,
                                              STFT_dict=self.STFT_dict, shuffle=shuffle_file_list, name=self.dataset_name)

        train_num = train_dataset.__len__()
        val_num = val_dataset.__len__()

        # Create dataloader
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                       shuffle=shuffle_samples_in_batch,
                                                       num_workers = num_workers)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                     shuffle=shuffle_samples_in_batch,
                                                     num_workers = num_workers)

        return train_dataloader, val_dataloader, train_num, val_num


    def get_basic_info(self):

        basic_info = []
        basic_info.append('HOSTNAME: ' + self.hostname)
        basic_info.append('Time: ' + self.date)
        basic_info.append('Device for training: ' + self.device)
        if self.device == 'cuda':
            basic_info.append('Cuda verion: {}'.format(torch.version.cuda))
        basic_info.append('Model name: {}'.format(self.model_name))
        
        return basic_info


    def train(self):

        torch.autograd.set_detect_anomaly(True)

        # Create directory for results
        saved_root = self.cfg.get('User', 'saved_root')
        z_dim = self.cfg.getint('Network','z_dim')
        tag = self.cfg.get('Network', 'tag')
        filename = "{}_{}_{}_z_dim={}".format(self.dataset_name, self.date, tag, z_dim)
        save_dir = os.path.join(saved_root, filename)
        if not(os.path.isdir(save_dir)):
            os.makedirs(save_dir)

        # Save the model configuration
        save_cfg = os.path.join(save_dir, 'config.ini')
        shutil.copy(self.config_file, save_cfg)

        # Create logger
        log_file = os.path.join(save_dir, 'log.txt')
        logger_type = self.cfg.getint('User', 'logger_type')
        logger = get_logger(log_file, logger_type)

        # Check if gpu is available on cluster (To be removed in the future)
        if 'gpu' in self.hostname and self.device == 'cpu':
            logger.error('GPU unavailable on cluster, training stop')
            return

        # Print basical infomation
        for log in self.get_basic_info():
            logger.info(log)
        logger.info('In this experiment, result will be saved in: ' + save_dir)

        # Print model infomation (optional)
        if self.cfg.getboolean('User', 'print_model'):
            for log in self.model.get_info():
                logger.info(log)

        # Init optimizer
        self.init_optimizer()

        # Load training parameters
        batch_size = self.cfg.getint('Training', 'batch_size')
        sequence_len = self.cfg.getint('DataFrame','sequence_len')
        use_random_seq = self.cfg.getboolean('DataFrame','use_random_seq')
        epochs = self.cfg.getint('Training', 'epochs')
        early_stop_patience = self.cfg.getint('Training', 'early_stop_patience')
        save_frequency = self.cfg.getint('Training', 'save_frequency')

        # Create data loader
        train_data_dir = self.cfg.get('User', 'train_data_dir')
        val_data_dir = self.cfg.get('User', 'val_data_dir')
        loader = self.build_dataloader(train_data_dir=train_data_dir, val_data_dir=val_data_dir,
                                       sequence_len=sequence_len, batch_size=batch_size,
                                       STFT_dict=self.STFT_dict, use_random_seq=use_random_seq)
        train_dataloader, val_dataloader, train_num, val_num = loader
        log_message = 'Training samples: {}'.format(train_num)
        logger.info(log_message)
        log_message = 'Validation samples: {}'.format(val_num)
        logger.info(log_message)

        # Create python list for loss
        train_loss = np.zeros((epochs,))
        val_loss = np.zeros((epochs,))
        train_recon = np.zeros((epochs,))
        train_KLD = np.zeros((epochs,))
        val_recon = np.zeros((epochs,))
        val_KLD = np.zeros((epochs,))
        best_val_loss = np.inf
        cpt_patience = 0
        cur_best_epoch = epochs
        best_state_dict = self.model.state_dict()

        # Define optimizer (might use different training schedule)
        optimizer = self.optimizer

        # Train with mini-batch SGD
        for epoch in range(epochs):

            start_time = datetime.datetime.now()
            self.model.train()

            # Batch training
            for batch_idx, batch_data in enumerate(train_dataloader):
                
                batch_data = batch_data.to(self.device)
                recon_batch_data = self.model(batch_data)

                loss_tot, loss_recon, loss_KLD = self.model.loss
                optimizer.zero_grad()
                loss_tot.backward()
                optimizer.step()

                train_loss[epoch] += loss_tot.item()
                train_recon[epoch] += loss_recon.item()
                train_KLD[epoch] += loss_KLD.item()
                
            # Validation
            for batch_idx, batch_data in enumerate(val_dataloader):

                batch_data = batch_data.to(self.device)
                recon_batch_data = self.model(batch_data)

                loss_tot, loss_recon, loss_KLD = self.model.loss
                
                val_loss[epoch] += loss_tot.item()
                val_recon[epoch] += loss_recon.item()
                val_KLD[epoch] += loss_KLD.item()

            # Loss normalization
            train_loss[epoch] = train_loss[epoch]/ train_num
            val_loss[epoch] = val_loss[epoch] / val_num
            train_recon[epoch] = train_recon[epoch] / train_num 
            train_KLD[epoch] = train_KLD[epoch]/ train_num
            val_recon[epoch] = val_recon[epoch] / val_num 
            val_KLD[epoch] = val_KLD[epoch] / val_num
            
            # Early stop patiance
            if val_loss[epoch] < best_val_loss:
                best_val_loss = val_loss[epoch]
                cpt_patience = 0
                best_state_dict = self.model.state_dict()
                cur_best_epoch = epoch
            else:
                cpt_patience += 1

            # Training time
            end_time = datetime.datetime.now()
            interval = (end_time - start_time).seconds / 60
            log_message = 'Epoch: {} train loss: {:.4f} val loss {:.4f} training time {:.2f}m'.format(epoch, train_loss[epoch], val_loss[epoch], interval)
            logger.info(log_message)

            # Stop traning if early-stop triggers
            if cpt_patience == early_stop_patience:
                logger.info('Early stop patience achieved')
                break

            # Save model parameters regularly
            if epoch % save_frequency == 0:
                save_file = os.path.join(save_dir, self.model_name + '_epoch' + str(cur_best_epoch) + '.pt')
                torch.save(self.model.state_dict(), save_file)
        
        # Save the final weights of network with the best validation loss
        train_loss = train_loss[:epoch+1]
        val_loss = val_loss[:epoch+1]
        train_recon = train_recon[:epoch+1]
        train_KLD = train_KLD[:epoch+1]
        val_recon = val_recon[:epoch+1]
        val_KLD = val_KLD[:epoch+1]
        save_file = os.path.join(save_dir, self.model_name + '_final_epoch' + str(cur_best_epoch) + '.pt')
        torch.save(best_state_dict, save_file)
        
        # Save the training loss and validation loss
        loss_file = os.path.join(save_dir, 'loss_model.pckl')
        with open(loss_file, 'wb') as f:
            pickle.dump([train_loss, val_loss, train_recon, train_KLD, val_recon, val_KLD], f)


        # Save the loss figure
        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_loss, label='training loss')
        plt.plot(val_loss, label='validation loss')
        plt.legend(fontsize=16, title=self.model_name, title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_{}.png'.format(tag))
        plt.savefig(fig_file)

        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_recon, label='Reconstruction')
        plt.plot(train_KLD, label='KL Divergence')
        plt.legend(fontsize=16, title='{}: Training'.format(self.model_name), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_train_{}.png'.format(tag))
        plt.savefig(fig_file) 

        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(val_recon, label='Reconstruction')
        plt.plot(val_KLD, label='KL Divergence')
        plt.legend(fontsize=16, title='{}: Validation'.format(self.model_name), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_val_{}.png'.format(tag))
        plt.savefig(fig_file)


    def generate(self, audio_orig, audio_recon=None, state_dict_file=None):
        
        # Define generated 
        if audio_recon == None:
            print('Generated audio file will be saved in the same directory as reference audio')
            audio_dir, audio_file = os.path.split(audio_orig)
            file_name, file_ext = os.path.splitext(audio_file)
            audio_recon = os.path.join(audio_dir, file_name+'_recon'+file_ext)
        
        # Load model state
        if state_dict_file != None:
            self.model.load_state_dict(torch.load(state_dict_file, map_location=self.device))

        # Read STFT parameters
        fs = self.STFT_dict['fs']
        nfft = self.STFT_dict['nfft']
        hop = self.STFT_dict['hop']
        wlen = self.STFT_dict['wlen']
        win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)
        trim = self.STFT_dict['trim']
        
        # Read original audio file
        x, fs_x = sf.read(audio_orig)
        if fs != fs_x:
            raise ValueError('Unexpected sampling rate')

        # Silence triming
        if trim:
            x, _ = librosa.effects.trim(x, top_db=30)

        # Scaling
        scale = np.max(np.abs(x))
        x = x / scale

        # STFT
        X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)

        # Prepare data input        
        data_orig = np.abs(X) ** 2 # (x_dim, seq_len)
        data_orig = torch.from_numpy(data_orig.astype(np.float32)).to(self.device) 

        # Reconstruction
        with torch.no_grad():
            data_recon = self.model(data_orig).to('cpu').detach().numpy()

        # Re-synthesis
        X_recon = np.sqrt(data_recon) * np.exp(1j * np.angle(X))
        x_recon = librosa.istft(X_recon, hop_length=hop, win_length=wlen, window=win)
        
        # Wrtie audio file
        scale_norm = 1 / (np.maximum(np.max(np.abs(x_recon)), np.max(np.abs(x)))) * 0.9
        sf.write(audio_recon, scale_norm*x_recon, fs_x)

    
    def eval(self, audio_ref, audio_est, metric='all', state_dict_file=None):
        
        # Load model state
        if state_dict_file != None:
            self.model.load_state_dict(torch.load(state_dict_file, map_location=self.device))

        if metric  == 'rmse':
            eval_func = rmse_frame()
            score = eval_func(audio_est, audio_ref)
            return score
        elif metric == 'pesq':
            eval_func = speechmetrics.load('pesq', window=None)
            score = eval_func(audio_est, audio_ref)['pesq']
            return score
        elif metric == 'stoi':
            eval_func = speechmetrics,load('stoi', window=None)
            score = eval_func(audio_est, audio_ref)['stoi']
            return score
        elif metric == 'all':
            eval_rmse = rmse_frame()
            eval_pesq = speechmetrics.load('pesq', window=None)
            eval_stoi = speechmetrics.load('stoi', window=None)
            score_rmse = eval_rmse(audio_est, audio_ref)
            score_pesq = eval_pesq(audio_est, audio_ref)['pesq']
            score_stoi = eval_stoi(audio_est, audio_ref)['stoi']
            return score_rmse, score_pesq, score_stoi
        else:
            raise ValueError('Evaluation only support: rmse, pesq, stoi, all')


    def test(self, data_dir, state_dict_file=None, print_results=True):
        """
        Apply re-synthesis for all audio files in a given directory, and return evaluation results
        All generated audio files in the same root as data_dir, named audio_dir + '_{}_recon'.format(tag)
        One could use state_dict_file to load preserved model state, otherwise it will use model state in cache
        Attention: if there already exist a folder with the same name, it will be deleted
        """

        # Remove '/' in the end if exist
        if data_dir[-1] == '/':
            data_dir = data_dir[:-1]
        else:
            data_dir = data_dir

        # Find all audio files
        data_suffix = self.cfg.get('DataFrame', 'suffix')
        audio_list = librosa.util.find_files(data_dir, ext=data_suffix)

        # Create re-synthesis folder
        tag = self.cfg.get('Network', 'tag')
        root, audio_dir = os.path.split(self.data_dir)
        recon_dir = os.path.join(root, audio_dir + '_{}_recon-utt'.format(self.tag))
        if os.path.isdir(recon_dir):
            shutil.rmtree(recon_dir)
        os.mkdir(recon_dir)

        # Load model state, avoid repeatly loading model state
        if state_dict_file == None:
            print('Use model state in cache...')
        else:
            print('Model state loading...')
            self.model.load_state_dict(torch.load(state_dict_file, map_location=self.device))

        # Create score list
        list_score_rmse = []
        list_score_pesq = []
        list_score_stoi = []

        # Loop over all audio files
        for audio_file in audio_list:

            # Define reconstructed file path
            root, file = os.path.split(audio_file)
            audio_file_recon = os.path.join(recon_dir, 'recon_'+file)

            # Reconstruction
            self.generate(audio_orig=audio_file, audio_recon=audio_file_recon, state_dict_file=None)

            # Evaluation
            score_rmse, score_pesq, score_stoi = self.eval(audio_ref=audio_file, audio_est=audio_file_recon, 
                                                           metric='all', state_dict_file=None)
            
            list_score_rmse.append(score_rmse)
            list_score_pesq.append(score_pesq)
            list_score_stoi.append(score_stoi)

        return list_score_rmse, list_score_pesq, list_score_stoi

        