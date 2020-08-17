#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import os
import sys
import shutil
import pickle
import numpy as np
import speechmetrics
import soundfile as sf
import librosa
import torch
from build_model import build_model
from backup_simon.speech_dataset import *

        
def rmse_frame():

    def get_result(path_to_estimate_file, path_to_reference):
        x_est, _ = sf.read(path_to_estimate_file)
        x_ref, _ = sf.read(path_to_reference)
        # align
        len_x = len(x_est)
        x_ref = x_ref[:len_x]
        # scaling
        x_est_ = np.expand_dims(x_est, axis=1)
        scale = np.linalg.lstsq(x_est_, x_ref, rcond=None)[0][0]
        x_est_scaled = scale * x_est
        return np.sqrt(np.square(x_est_scaled, x_ref).mean())

    return get_result

    

class Evaluate():

    def __init__(self, model_dir, data_dir):
        
        self.model_dir = model_dir

        if data_dir[-1] == '/':
            self.data_dir = data_dir[:-1]
        else:
            self.data_dir = data_dir

        self.build()


    def build(self):

        # Find config file and training weight
        for file in os.listdir(self.model_dir):
            print(file)
            if '.ini' in file:
                self.cfg_file = os.path.join(model_dir, file)
            if 'final_epoch' in file:
            # if 'KVAE_epoch310' in file:
                self.weight_file = os.path.join(model_dir, file)

        # Find all audio files
        self.audio_list = librosa.util.find_files(self.data_dir, ext='wav')

        # Create model class
        self.model_class = build_model(self.cfg_file, training=False)
        self.model = self.model_class.model
        self.cfg = self.model_class.cfg
        use_cuda = self.cfg.getboolean('Training', 'use_cuda')
        self.local_device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
        
        # Create evaluate metric
        self.eval_rmse = rmse_frame()
        self.eval_pesq = speechmetrics.load('pesq', window=None)
        self.eval_stoi = speechmetrics.load('stoi', window=None)
        self.score_rmse = []
        self.score_pesq = []
        self.score_stoi = []

        # Load weight
        self.model.load_state_dict(torch.load(self.weight_file, map_location=self.local_device))
        self.model.eval()

        # Load STFT parameters
        self.wlen_sec = self.cfg.getfloat('STFT', 'wlen_sec') # windows lenght in seconds
        self.hop_percent = self.cfg.getfloat('STFT', 'hop_percent')
        self.fs = self.cfg.getint('STFT', 'fs')
        self.zp_percent = self.cfg.getint('STFT', 'zp_percent')
        self.trim = self.cfg.getboolean('STFT', 'trim')
        self.verbose = self.cfg.getboolean('STFT', 'verbose')
        # number of fft is supposed to be a power of 2 (num of rows in STFT Matrix is nfft/2 + 1)
        self.nfft = np.int(np.power(2, np.ceil(np.log2(self.wlen_sec * self.fs))))
        # window length <= nfft (equal to nfft by default)
        self.wlen = self.nfft
        # hop: number of audio samples between adjacent STFT columns
        self.hop = np.int(self.hop_percent * self.wlen)
        # a vector or array of length nfft (sin function, 0 ~ pi)
        self.win = np.sin(np.arange(0.5, self.wlen+0.5) / self.wlen * np.pi)

        # Others
        self.sequence_len = self.cfg.getint('DataFrame','sequence_len')


    def evaluate(self):

         # Create re-synthesis folder
        tag = self.model_class.tag
        root, audio_dir = os.path.split(self.data_dir)
        self.recon_dir = os.path.join(root, audio_dir + '_{}_recon'.format(tag))
        if os.path.isdir(self.recon_dir):
            c = input("{} already exists, press 'e' to exist, or delete old one and continue".format(self.recon_dir))
            if c == 'e':
                return
            else:
                shutil.rmtree(self.recon_dir)
                os.mkdir(self.recon_dir)
        else:
            os.mkdir(self.recon_dir)


        # Loop over audio files
        for audio_file in self.audio_list:

            # Define reconstruction file path
            root, file = os.path.split(audio_file)
            file_orig = os.path.join(self.recon_dir, 'orig_'+file)
            file_recon = os.path.join(self.recon_dir, 'recon_'+file)

            # Read audio file and do STFT
            x, fs_x = sf.read(audio_file)
            scale = np.max(np.abs(x))
            x = x / scale
            X = librosa.stft(x, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen, window=self.win)

            # Prepare data input
            data_orig = np.abs(X) ** 2 # (x_dim, seq_len)
            data_orig = torch.from_numpy(data_orig.astype(np.float32)).to(self.local_device)

            # Reconstruction
            with torch.no_grad():
                data_recon = self.model(data_orig).to(self.local_device).detach().numpy()

            # Re-synthesis
            X_recon = np.sqrt(data_recon) * np.exp(1j * np.angle(X))
            x_recon = librosa.istft(X_recon, hop_length=self.hop, win_length=self.wlen, window=self.win)
            x_orig = x

            # Write audio file
            scale_norm = 1 / (np.maximum(np.max(np.abs(x_recon)), np.max(np.abs(x_orig)))) * 0.9
            librosa.output.write_wav(file_orig, scale_norm*x_orig, fs_x)
            librosa.output.write_wav(file_recon, scale_norm*x_recon, fs_x)

            # Evaluation
            self.score_rmse.append(self.eval_rmse(file_recon, file_orig))
            self.score_pesq.append(self.eval_pesq(file_recon, file_orig)['pesq'])
            self.score_stoi.append(self.eval_stoi(file_recon, file_orig)['stoi'])
            self.eval_score = {'rmse': self.score_rmse,
                               'pesq': self.score_pesq,
                               'stoi': self.score_stoi}

            # Print and save results 
        self.print_save()
        

    def evaluate_seq(self):

         # Create re-synthesis folder
        tag = self.model_class.tag
        root, audio_dir = os.path.split(self.data_dir)
        self.recon_dir = os.path.join(root, audio_dir + '_{}_recon-seq'.format(tag))
        if os.path.isdir(self.recon_dir):
            c = input("{} already exists, press 'e' to exist, or delete old one and continue".format(self.recon_dir))
            if c == 'e':
                return
            else:
                shutil.rmtree(self.recon_dir)
                os.mkdir(self.recon_dir)
        else:
            os.mkdir(self.recon_dir)

        # Loop over audio files
        for audio_file in self.audio_list:

            # Read audio file and do STFT
            x, fs_x = sf.read(audio_file)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')
            scale = np.max(np.abs(x))
            x = x / scale

            # Remove silence
            if self.trim:
                x, _ = librosa.effects.trim(x, top_db=30)

            # Evaluation sequence to sequence
            x = np.pad(x, int(self.nfft // 2), mode='reflect')
            X = librosa.stft(x, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen, window=self.win)
            # n_seq = (1 + int((len(x) - self.wlen) / self.hop) )//self.sequence_len
            n_seq = X.shape[1] // self.sequence_len

            for i in range(n_seq):
                sample = X[:, i*self.sequence_len:(i+1)*self.sequence_len]
                data_orig = np.abs(sample) ** 2
                data_orig = torch.from_numpy(data_orig.astype(np.float32))
                
                with torch.no_grad():
                    data_recon = self.model(data_orig).to(self.local_device).detach().numpy()
                
                X_recon = np.sqrt(data_recon) * np.exp(1j * np.angle(sample))
                x_recon = librosa.istft(X_recon, hop_length=self.hop, win_length=self.wlen, window=self.win)
                x_orig = librosa.istft(sample, hop_length=self.hop, win_length=self.wlen, window=self.win)
                
                # Define reconstruction file path
                root, file = os.path.split(audio_file)
                file_orig = os.path.join(self.recon_dir, 'orig_{}_'.format(i) + file)
                file_recon = os.path.join(self.recon_dir, 'recon_{}_'.format(i) + file)

                # Save files
                scale_norm = 1 / (np.maximum(np.max(np.abs(x_recon)), np.max(np.abs(x_orig)))) * 0.9
                librosa.output.write_wav(file_orig, scale_norm*x_orig, fs_x)
                librosa.output.write_wav(file_recon, scale_norm*x_recon, fs_x)

                # Evaluation
                self.score_rmse.append(self.eval_rmse(file_recon, file_orig))
                self.score_pesq.append(self.eval_pesq(file_recon, file_orig)['pesq'])
                self.score_stoi.append(self.eval_stoi(file_recon, file_orig)['stoi'])
                self.eval_score = {'rmse': self.score_rmse,
                                   'pesq': self.score_pesq,
                                   'stoi': self.score_stoi}
        
        # Print and save results 
        self.print_save()


    def print_save(self):

        # Print results
        array_rmse = np.array(self.score_rmse)
        array_pesq = np.array(self.score_pesq)
        array_stoi = np.array(self.score_stoi)
        print("===== RMSE =====")
        print("mean: {:.4f}".format(array_rmse.mean()))
        print("min: {:.4f}".format(array_rmse.min()))
        print("max: {:.4f}".format(array_rmse.max()))
        print("===== PESQ =====")
        print("mean: {:.4f}".format(array_pesq.mean()))
        print("min: {:.4f}".format(array_pesq.min()))
        print("max: {:.4f}".format(array_pesq.max()))
        print("===== STOI =====")
        print("mean: {:.4f}".format(array_stoi.mean()))
        print("min: {:.4f}".format(array_stoi.min()))
        print("max: {:.4f}".format(array_stoi.max()))
        
        # Save evaluation
        save_file = os.path.join(self.recon_dir, 'evaluation.pckl')
        with open(save_file, 'wb') as f:
            pickle.dump([self.score_rmse, self.score_pesq, self.score_stoi], f)



if __name__ == '__main__':

    if len(sys.argv) == 3:
        model_dir = sys.argv[1]
        data_dir = sys.argv[2]
        ev = Evaluate(model_dir, data_dir)
        ev.evaluate_seq()
        
    else:
        print("Please follow: evaluate model_dir data_dir")
    
