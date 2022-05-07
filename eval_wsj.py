#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inria.fr)
License agreement in LICENSE.txt
"""

import os
import sys
import argparse
from tqdm import tqdm
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from dvae.learning_algo import LearningAlgorithm
from dvae.learning_algo_ss import LearningAlgorithm_ss
from dvae.utils.eval_metric import compute_median, EvalMetrics

torch.manual_seed(0)
np.random.seed(0)

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # Basic config file
        self.parser.add_argument('--ss', action='store_true', help='schedule sampling')
        self.parser.add_argument('--cfg', type=str, default=None, help='config path')
        self.parser.add_argument('--saved_dict', type=str, default=None, help='trained model dict')
        # Dataset
        self.parser.add_argument('--test_dir', type=str, default='./data/clean_speech/wsj0_si_et_05', help='test dataset')
        # Restuls directory
        self.parser.add_argument('--ret_dir', type=str, default='./data/tmp', help='tmp dir for audio reconstruction')
    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        params = vars(self.opt)
        return params

params = Options().get_params()

if params['ss']:
    learning_algo = LearningAlgorithm_ss(params=params)
else:
    learning_algo = LearningAlgorithm(params=params)
learning_algo.build_model()
dvae = learning_algo.model
dvae.load_state_dict(torch.load(params['saved_dict'], map_location='cpu'))
eval_metrics = EvalMetrics(metric='all')
dvae.eval()
cfg = learning_algo.cfg
print('Total params: %.2fM' % (sum(p.numel() for p in dvae.parameters()) / 1000000.0))

list_rmse = []
list_sisdr = []
list_pesq = []
list_estoi = []

file_list = librosa.util.find_files(test_dir, ext='wav')

if not os.path.isdir(params['ret_dir']):
    os.makedirs(params['ret_dir'])

for audio_file in tqdm(file_list):
    
    root, file = os.path.split(audio_file)
    filename, _ = os.path.splitext(file)
    recon_audio = os.path.join(params['ret_dir'], 'recon_{}.wav'.format(filename))
    orig_audio = os.path.join(params['ret_dir'], 'orig_{}.wav'.format(filename))

    # STFT parameters
    wlen_sec = cfg.getfloat('STFT', 'wlen_sec')
    hop_percent = cfg.getfloat('STFT', 'hop_percent')
    fs = cfg.getint('STFT', 'fs')
    zp_percent = cfg.getint('STFT', 'zp_percent')
    wlen = wlen_sec * fs
    wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # pwoer of 2
    hop = np.int(hop_percent * wlen)
    nfft = wlen + zp_percent * wlen
    win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)
    trim = cfg.getboolean('STFT', 'trim')

    x, fs_x = sf.read(audio_file)

    if trim:
        x, _ = librosa.effects.trim(x, top_db=30)
    
    #####################
    # Scaling on waveform
    #####################
    # scale = 1
    scale = np.max(np.abs(x)) # normalized by Max(|x|)
    x = x / scale

    # STFT
    X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)

    # Prepare data input        
    data_orig = np.abs(X) ** 2 # (x_dim, seq_len)
    data_orig = torch.from_numpy(data_orig.astype(np.float32)).to(dvae.device) 
    data_orig = data_orig.permute(1,0).unsqueeze(1) #  (x_dim, seq_len) => (seq_len, 1, x_dim)

    # Reconstruction
    with torch.no_grad():
        data_recon = torch.exp(dvae(data_orig))

    data_recon = data_recon.to('cpu').detach().squeeze().permute(1,0).numpy()

    # Re-synthesis
    X_recon = np.sqrt(data_recon) * np.exp(1j * np.angle(X))
    x_recon = librosa.istft(X_recon, hop_length=hop, win_length=wlen, window=win)
    
    # Wrtie audio file
    scale_norm = 1 / (np.maximum(np.max(np.abs(x_recon)), np.max(np.abs(x)))) * 0.9
    sf.write(recon_audio, scale_norm*x_recon, fs_x)
    sf.write(orig_audio, scale_norm*x, fs_x)

    rmse, sisdr, pesq, estoi = eval_metrics.eval(audio_est=recon_audio, audio_ref=orig_audio)
    
    # print('File: {}, rmse: {:.4f}, pesq: {:.4f}, estoi: {:.4f}'.format(filename, rmse, pesq, estoi))

    list_rmse.append(rmse)
    list_sisdr.append(sisdr)
    list_pesq.append(pesq)
    list_estoi.append(estoi)

np_rmse = np.array(list_rmse)
np_sisdr = np.array(list_sisdr)
np_pesq = np.array(list_pesq)
np_estoi = np.array(list_estoi)

print('Re-synthesis finished')
print('RMSE: {:.4f}'.format(np.mean(np_rmse)))
print('SI-SDR: {:.4f}'.format(np.mean(np_sisdr)))
print('PESQ: {:.4f}'.format(np.mean(np_pesq)))
print('ESTOI: {:.4f}'.format(np.mean(np_estoi)))

rmse_median, rmse_ci = compute_median(np_rmse)
sisdr_median, sisdr_ci = compute_median(np_sisdr)
pesq_median, pesq_ci = compute_median(np_pesq)
estoi_median, estoi_ci = compute_median(np_estoi)

print("Median evaluation")
print('median rmse score: {:.4f} +/- {:.4f}'.format(rmse_median, rmse_ci))
print('median sisdr score: {:.4f} +/- {:.4f}'.format(sisdr_median, sisdr_ci))
print('median pesq score: {:.4f} +/- {:.4f}'.format(pesq_median, pesq_ci))
print('median estoi score: {:.4f} +/- {:.4f}'.format(estoi_median, estoi_ci))