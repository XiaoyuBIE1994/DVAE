#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import os
import sys
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
from build_model import build_model

def find_file(dir_result):
    
    for file in os.listdir(dir_result):
        if '.ini' in file:
            cfg_file = os.path.join(dir_result, file)
        if 'final_epoch' in file:
            weight_file = os.path.join(dir_result, file)

    return cfg_file, weight_file

def resynthesis(cfg_file, weight_file, audio_file_list, tag):
    
    model_class = build_model(cfg_file)
    model = model_class.model
    cfg = model_class.cfg
    local_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model.load_state_dict(torch.load(weight_file, map_location=local_device))
    model.eval()
    
    if local_device == 'cuda':
        model.cuda()
    
    # Load STFT parameters
    wlen_sec = cfg.getfloat('STFT', 'wlen_sec') # windows lenght in seconds
    hop_percent = cfg.getfloat('STFT', 'hop_percent')
    fs = cfg.getint('STFT', 'fs')
    zp_percent = cfg.getint('STFT', 'zp_percent')
    trim = cfg.getboolean('STFT', 'trim')
    verbose = cfg.getboolean('STFT', 'verbose')
    
    # number of fft is supposed to be a power of 2 (num of rows in STFT Matrix is nfft/2 + 1)
    nfft = np.int(np.power(2, np.ceil(np.log2(wlen_sec * fs))))
    # window length <= nfft (equal to nfft by default)
    wlen = nfft
    # hop: number of audio samples between adjacent STFT columns
    hop = np.int(hop_percent * wlen)
    # a vector or array of length nfft (sin function, 0 ~ pi)
    win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)

    data_orig_list = []
    
    for num, file in enumerate(audio_file_list):
        
        x, fs_x = sf.read(file)
        x = x / np.max(np.abs(x)) 
        X = librosa.stft(x, n_fft=nfft, hop_length=hop,
                            win_length=wlen, window=win) # stft

        # x_dim: nfft/2 + 1
        # seq_len: number of frames
        data_orig = np.abs(X) ** 2 # (x_dim, seq_len)
        data_orig = torch.from_numpy(data_orig.astype(np.float32))
        data_orig = data_orig.to(local_device)

        # print('shape of data_orig: {}'.format(data_orig.shape)) # used for debug only
        data_orig = data_orig.T # (seq_len, x_dim), adapted to torch layers

        with torch.no_grad():
            data_recon, mean, logvar, z = model(data_orig)
            mean = mean.detach().numpy()
            data_recon = data_recon.detach().numpy()
            data_orig = data_orig.detach().numpy()

        # change back to (x_dim, seq_len), used for result display
        mean = mean.T
        data_orig = data_orig.T
        data_recon = data_recon.T

        # print('shape of mean: {}'.format(mean.shape)) # used for debug only

        plt.figure()
        plt.subplot(311)
        librosa.display.specshow(librosa.power_to_db(data_orig), y_axis='log', sr=fs, hop_length=hop)
        plt.set_cmap('jet')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Original spectrogram')

        plt.subplot(312)
        plt.imshow(mean, origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title('mean value')

        plt.subplot(313)
        librosa.display.specshow(librosa.power_to_db(data_recon), y_axis='log', sr=fs, hop_length=hop)
        plt.set_cmap('jet')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Reconstructed spectrogram')

        X_recon = np.sqrt(data_recon) * np.exp(1j * np.angle(X))
        x_recon = librosa.istft(X_recon, hop_length=hop, win_length=wlen, window=win)
        x_origin = x

        scale = 1 / (np.maximum(np.max(np.abs(x_recon)), np.max(np.abs(x_origin)))) * 0.9

        dir_name, file_name = os.path.split(weight_file)
        dir_name = os.path.join(dir_name, 'test')

        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        
        orig_file = os.path.join(dir_name, '{}_origin-{}.wav'.format(tag, num))
        recon_file = os.path.join(dir_name, '{}}_recon-{}.wav'.format(tag, num))
        plot_file = os.path.join(dir_name, '{}_resynthesis-{}.png'.format(tag, num))

        # librosa.output.write_wav(orig_file, scale*x_origin, fs)
        # librosa.output.write_wav(recon_file, scale*x_recon, fs)
        # plt.savefig(plot_file)

        print('=====> {} file {} reconstruction finished'.format(num, file_name))

###
# Test over 5 males audios and 5 female audios
###
def test(dir_result):
    cfg_file, weight_file = find_file(dir_result)    
    root_dir = '/Users/xiaoyu/WorkStation/Data/clean_speech/wsj0_si_et_05'
    male_audios = ['440/440c020a.wav', '440/440c020b.wav', '440/440c020c.wav', '440/440c020d.wav', '440/440c020e.wav']
    female_audios = ['441/441c020a.wav', '441/441c020b.wav', '441/441c020c.wav', '441/441c020d.wav', '441/441c020e.wav']

    # male
    audiofile_list = []
    for file in male_audios:
        audio_file = os.path.join(root_dir, file)
        audiofile_list.append(audio_file)

    resynthesis(cfg_file, weight_file, audiofile_list, tag='man_440')

    # female
    audiofile_list = []
    for file in female_audios:
        audio_file = os.path.join(root_dir, file)
        audiofile_list.append(audio_file)

    resynthesis(cfg_file, weight_file, audiofile_list, tag='women_441')


if __name__ == '__main__':
    if len(sys.argv) == 2:
        dir_result = sys.argv[1]
        test(dir_result)
    else:
        print('Please indicate the directory of results')
        
