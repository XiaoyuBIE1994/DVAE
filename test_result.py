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

def resynthesis(cfg_file, weight_file, audio_file_list, local_device='cpu'):
    
    model_class = build_model(cfg_file)
    model = model_class.model
    cfg = model_class.cfg

    
    model.load_state_dict(torch.load(weight_file, map_location=local_device))
    model.eval()
    
    if local_device == 'cuda':
        model.cuda()
    
    # Load STFT parameters
    wlen_sec = cfg.getfloat('STFT', 'wlen_sec')
    hop_percent = cfg.getfloat('STFT', 'hop_percent')
    fs = cfg.getint('STFT', 'fs')
    zp_percent = cfg.getint('STFT', 'zp_percent')
    trim = cfg.getboolean('STFT', 'trim')
    verbose = cfg.getboolean('STFT', 'verbose')
    
    wlen = np.int(np.power(2, np.ceil(np.log2(wlen_sec * fs))))
    nfft = wlen
    hop = np.int(hop_percent * wlen)
    win = np.sin(np.arange(0.5, wlen-0.5+1) / wlen * np.pi)

    data_orig_list = []
    
    for num, file in enumerate(audio_file_list):
        x, fs_x = sf.read(file)
        x = x / np.max(np.abs(x))
        X = librosa.stft(x, n_fft=nfft, hop_length=hop,
                            win_length=wlen, window=win) # stft

        data_orig = np.abs(X) ** 2
        data_orig = torch.from_numpy(data_orig.astype(np.float32))
        data_orig = data_orig.to(local_device)

        # print('shape of data_orig: {}'.format(data_orig.shape)) # used for debug only
        data_orig = data_orig.T # ffnn only

        with torch.no_grad():
            data_recon, mean, logvar, z = model(data_orig)
            mean = mean.detach().numpy()
            data_recon = data_recon.detach().numpy()
            data_orig = data_orig.detach().numpy()

        data_orig = data_orig.T  # ffnn only
        data_recon = data_recon.T # ffnn only

        # print('shape of mean: {}'.format(mean.shape)) # used for debug only

        plt.figure()
        plt.subplot(311)
        librosa.display.specshow(librosa.power_to_db(data_orig), y_axis='log', sr=fs, hop_length=hop)
        plt.set_cmap('jet')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Original spectrogram')

        plt.subplot(312)
        plt.imshow(mean.T, origin='lower')
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
        
        orig_file = os.path.join(dir_name, 'man_440_origin-{}.wav'.format(num))
        recon_file = os.path.join(dir_name, 'man_440_recon-{}.wav'.format(num))
        plot_file = os.path.join(dir_name, 'man_440_resynthesis-{}.png'.format(num))
        
        # orig_file = os.path.join(dir_name, 'women_441_origin-{}.wav'.format(num))
        # recon_file = os.path.join(dir_name, 'women_441_recon-{}.wav'.format(num))
        # plot_file = os.path.join(dir_name, 'women_441_resynthesis-{}.png'.format(num))

        librosa.output.write_wav(orig_file, scale*x_origin, fs)
        librosa.output.write_wav(recon_file, scale*x_recon, fs)
        plt.savefig(plot_file)

        print('=====> {} file {} reconstruction finished'.format(num, file_name))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        dir_result = sys.argv[1]
        cfg_file, weight_file = find_file(dir_result)

        # file_list = librosa.util.find_files(data_dir, ext='wav')
        # audiofile_list = random.sample(file_list, 5)

        audiofile_list = []
        
        data_dir = '/Users/xiaoyu/WorkStation/Data/clean_speech/wsj0_si_et_05/440'
        file_list = ['440c020a.wav', '440c020b.wav', '440c020c.wav', '440c020d.wav', '440c020e.wav']

        # data_dir = '/Users/xiaoyu/WorkStation/Data/clean_speech/wsj0_si_et_05/441'
        # file_list = ['441c020a.wav', '441c020b.wav', '441c020c.wav', '441c020d.wav', '441c020e.wav']

        for file in file_list:
            audio_file = os.path.join(data_dir, file)
            audiofile_list.append(audio_file)

        resynthesis(cfg_file, weight_file, audiofile_list)

