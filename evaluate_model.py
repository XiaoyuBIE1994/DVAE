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
            if '.ini' in file:
                self.cfg_file = os.path.join(model_dir, file)
            if 'final_epoch' in file:
                self.weight_file = os.path.join(model_dir, file)

        # Find all audio files
        self.audio_list = librosa.util.find_files(self.data_dir, ext='wav')

        # Create model class
        self.model_class = build_model(self.cfg_file, training=False)
        self.model = model_class.model
        self.cfg = model_class.cfg
        self.local_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
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
        self.wlen_sec = cfg.getfloat('STFT', 'wlen_sec') # windows lenght in seconds
        self.hop_percent = cfg.getfloat('STFT', 'hop_percent')
        self.fs = cfg.getint('STFT', 'fs')
        self.zp_percent = cfg.getint('STFT', 'zp_percent')
        # number of fft is supposed to be a power of 2 (num of rows in STFT Matrix is nfft/2 + 1)
        self.nfft = np.int(np.power(2, np.ceil(np.log2(self.wlen_sec * self.fs))))
        # window length <= nfft (equal to nfft by default)
        self.wlen = nfft
        # hop: number of audio samples between adjacent STFT columns
        self.hop = np.int(self.hop_percent * wlen)
        # a vector or array of length nfft (sin function, 0 ~ pi)
        self.win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)
        # others
        



    def evaluate(self):

         # Create re-synthesis folder
        tag = model_class.tag

        root, audio_dir = os.path.split(self.data_dir)
        recon_dir = os.path.join(root, audio_dir + '_{}_recon'.format(tag))
        if os.path.isdir(recon_dir):
            c = input("{} already exists, press 'e' to exist, or delete old one and continue".format(recon_dir))
            if c == 'e':
                return
            else:
                shutil.rmtree(recon_dir)
                os.mkdir(recon_dir)
        else:
            os.mkdir(recon_dir)


        # Loop over audio files
        for audio_file in self.audio_list:

            # Define reconstruction file path
            root, file = os.path.split(audio_file)
            file_orig = os.path.join(recon_dir, 'orig_'+file)
            file_recon = os.path.join(recon_dir, 'recon_'+file)

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
                data_recon = model(data_orig).to('cpu').detach().numpy()

            # Re-synthesis
            X_recon = np.sqrt(data_recon) * np.exp(1j * np.angle(X))
            x_recon = librosa.istft(X_recon, hop_length=self.hop, win_length=self.wlen, window=self.win)
            x_orig = x

            # Write audio file
            scale_norm = 1 / (np.maximum(np.max(np.abs(x_recon)), np.max(np.abs(x_orig)))) * 0.9
            librosa.output.write_wav(file_orig, scale_norm*x_orig, fs_x)
            librosa.output.write_wav(file_recon, scale_norm*x_recon, fs_x)

            # Evaluation
            score_rmse.append(eval_rmse(file_recon, file_orig))
            score_pesq.append(eval_pesq(file_recon, file_orig)['pesq'])
            score_stoi.append(eval_stoi(file_recon, file_orig)['stoi'])
            self.eval_score = {'rmse': score_rmse,
                               'pesq': score_pesq,
                               'stoi': score_stoi}

        
        # Print results
        array_rmse = np.array(score_rmse)
        array_pesq = np.array(score_pesq)
        array_stoi = np.array(score_stoi)
        print("===== RMSE =====")
        print("mean: {}".format(array_rmse.mean()))
        print("min: {}".format(array_rmse.min()))
        print("max: {}".format(array_rmse.max()))
        print("===== PESQ =====")
        print("mean: {}".format(array_pesq.mean()))
        print("min: {}".format(array_pesq.min()))
        print("max: {}".format(array_pesq.max()))
        print("===== STOI =====")
        print("mean: {}".format(array_stoi.mean()))
        print("min: {}".format(array_stoi.min()))
        print("max: {}".format(array_stoi.max()))
        
        # Save evaluation
        save_file = os.path.join(self.recon_dir, 'evaluation.pckl')
        with open(save_file, 'wb') as f:
            pickle.dump([score_rmse, score_pesq, score_stoi], f)


    def evaluate_seq(self):

         # Create re-synthesis folder
        tag = model_class.tag

        root, audio_dir = os.path.split(self.data_dir)
        recon_dir = os.path.join(root, audio_dir + '_{}_recon-seq'.format(tag))
        if os.path.isdir(recon_dir):
            c = input("{} already exists, press 'e' to exist, or delete old one and continue".format(recon_dir))
            if c == 'e':
                return
            else:
                shutil.rmtree(recon_dir)
                os.mkdir(recon_dir)
        else:
            os.mkdir(recon_dir)

        # Instanciate dataloader
        self.sequence_len = self.cfg.getint('DataFrame','sequence_len')
        self.trim = self.cfg.getboolean('STFT', 'trim')
        self.verbose = self.cfg.getboolean('STFT', 'verbose')
        self.dataset_name = self.cfg.get('DataFrame', 'dataset_name')

        eval_dataset = SpeechDataSequences(file_list=self.audio_list, sequence_len=self.sequence_len,
                                           wlen_sec=self.wlen_sec, hop_percent=self.hop_percent, fs=self.fs,
                                           zp_percent=self.zp_percent, trim=self.trim, verbose=self.verbose,
                                           batch_size=1, shuffle_file_list=False,
                                           name=self.dataset_name)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers = 1)

        # Loop over audio files
        for data in eval_dataloader:

            # Define reconstruction file path
            root, file = os.path.split(audio_file)
            file_orig = os.path.join(recon_dir, 'orig_'+file)
            file_recon = os.path.join(recon_dir, 'recon_'+file)

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
                data_recon = model(data_orig).to('cpu').detach().numpy()

            # Re-synthesis
            X_recon = np.sqrt(data_recon) * np.exp(1j * np.angle(X))
            x_recon = librosa.istft(X_recon, hop_length=self.hop, win_length=self.wlen, window=self.win)
            x_orig = x

            # Write audio file
            scale_norm = 1 / (np.maximum(np.max(np.abs(x_recon)), np.max(np.abs(x_orig)))) * 0.9
            librosa.output.write_wav(file_orig, scale_norm*x_orig, fs_x)
            librosa.output.write_wav(file_recon, scale_norm*x_recon, fs_x)

            # Evaluation
            score_rmse.append(eval_rmse(file_recon, file_orig))
            score_pesq.append(eval_pesq(file_recon, file_orig)['pesq'])
            score_stoi.append(eval_stoi(file_recon, file_orig)['stoi'])
            self.eval_score = {'rmse': score_rmse,
                               'pesq': score_pesq,
                               'stoi': score_stoi}

        
        # Print results
        array_rmse = np.array(score_rmse)
        array_pesq = np.array(score_pesq)
        array_stoi = np.array(score_stoi)
        print("===== RMSE =====")
        print("mean: {}".format(array_rmse.mean()))
        print("min: {}".format(array_rmse.min()))
        print("max: {}".format(array_rmse.max()))
        print("===== PESQ =====")
        print("mean: {}".format(array_pesq.mean()))
        print("min: {}".format(array_pesq.min()))
        print("max: {}".format(array_pesq.max()))
        print("===== STOI =====")
        print("mean: {}".format(array_stoi.mean()))
        print("min: {}".format(array_stoi.min()))
        print("max: {}".format(array_stoi.max()))
        
        # Save evaluation
        save_file = os.path.join(self.recon_dir, 'evaluation.pckl')
        with open(save_file, 'wb') as f:
            pickle.dump([score_rmse, score_pesq, score_stoi], f)



if __name__ == '__main__':

    if len(sys.argv) == 3:
        model_dir = sys.argv[1]
        data_dir = sys.argv[2]
        ev = Evaluate(model_dir, data_dir)
        ev.evaluate()
        
    else:
        print("Please follow: evaluate model_dir data_dir")
    
