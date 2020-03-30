#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'evaluation'))
import random
import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
from build_model import build_model
from evaluation.write_eval import write_eval, write_eval_latex
from evaluation.loss_mse import rmse_td, rmse_fd, rmse_frame
from evaluation.loss_stoi import stoi


# find config file path and network weight file in a directory
def find_file(train_dir):
    
    cfg_file = None
    weight_file = None

    for file in os.listdir(train_dir):
        if '.ini' in file:
            cfg_file = os.path.join(train_dir, file)
        if 'final' in file:
            weight_file = os.path.join(train_dir, file)

    return cfg_file, weight_file
    

# resynthesis an audio with pre-trained weight 
def resynthesis(cfg_file, weight_file, audio_file_list):
    
    # Create model class
    model_class = build_model(cfg_file)
    model = model_class.model
    cfg = model_class.cfg
    local_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load weight
    model.load_state_dict(torch.load(weight_file, map_location=local_device))
    model.eval()
    if local_device == 'cuda':
        model.cuda()
    
    # Load STFT parameters
    wlen_sec = cfg.getfloat('STFT', 'wlen_sec') # windows lenght in seconds
    hop_percent = cfg.getfloat('STFT', 'hop_percent')
    fs = cfg.getint('STFT', 'fs')
    zp_percent = cfg.getint('STFT', 'zp_percent')
    # number of fft is supposed to be a power of 2 (num of rows in STFT Matrix is nfft/2 + 1)
    nfft = np.int(np.power(2, np.ceil(np.log2(wlen_sec * fs))))
    # window length <= nfft (equal to nfft by default)
    wlen = nfft
    # hop: number of audio samples between adjacent STFT columns
    hop = np.int(hop_percent * wlen)
    # a vector or array of length nfft (sin function, 0 ~ pi)
    win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)
    # print('win shape: {}'.format(win.shape))

    # Create a dictionary for evaluation results
    eval_dic = {'rmse_td': [], 'rmse_fd':[], 
                'rmse_td_frame': [], 'rmse_fd_frame':[],
                'stoi':[]}
    fig = plt.figure()
    
    # Loop over audio files
    for num, file in enumerate(audio_file_list):
        
        # Allocate tag
        root_dir, filename = os.path.split(file)
        if filename[:3] == '440':
            tag = 'male_440'
        elif filename[:3] == '441':
            tag = 'female_441'

        # Read audio file and transform via stft
        x, fs_x = sf.read(file)
        x = x / np.max(np.abs(x))
        X = librosa.stft(x, n_fft=nfft, hop_length=hop,
                            win_length=wlen, window=win) # stft
        
        # x_dim: nfft/2 + 1
        # seq_len: number of frames
        data_orig = np.abs(X) ** 2 # (x_dim, seq_len)
        data_orig = torch.from_numpy(data_orig.astype(np.float32))
        data_orig = data_orig.to(local_device)
        data_orig = data_orig.T # (seq_len, x_dim), adapted to torch layers

        
        # model output: (seq_len, data_dim)
        # recon need: (data_dim, seq_len)
        with torch.no_grad():
            data_recon, mean, logvar, z = model(data_orig)
            mean = mean.detach().numpy().T
            data_recon = data_recon.detach().numpy().T
            data_orig = data_orig.detach().numpy().T

        # Re-synthesis
        X_recon = np.sqrt(data_recon) * np.exp(1j * np.angle(X))
        x_recon = librosa.istft(X_recon, hop_length=hop, win_length=wlen, window=win)
        x_orig = x
        
        # Plot spectrogram
        # fig.clf()
        # plt.subplot(311)
        # librosa.display.specshow(librosa.power_to_db(data_orig), y_axis='log', sr=fs, hop_length=hop) 
        # plt.set_cmap('jet')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Original spectrogram')
        # plt.subplot(312)
        # plt.imshow(mean, origin='lower')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('mean value')
        # plt.subplot(313)
        # librosa.display.specshow(librosa.power_to_db(data_recon), y_axis='log', sr=fs, hop_length=hop)
        # plt.set_cmap('jet')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Reconstructed spectrogram')
        # plt.savefig(plot_file)
        
        # Re-mapping audio vector to (-0.9, 0.9) and save wav file
        dir_name, file_name = os.path.split(weight_file)
        dir_name = os.path.join(dir_name, 'test')
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        # scale_norm = 1 / (np.maximum(np.max(np.abs(x_recon)), np.max(np.abs(x_orig)))) * 0.9
        # orig_file = os.path.join(dir_name, '{}_origin-{}.wav'.format(tag, num))
        # recon_file = os.path.join(dir_name, '{}_recon-{}.wav'.format(tag, num))
        # plot_file = os.path.join(dir_name, '{}_resynthesis-{}.png'.format(tag, num))
        # librosa.output.write_wav(orig_file, scale_norm*x_orig, fs)
        # librosa.output.write_wav(recon_file, scale_norm*x_recon, fs)
        # print('=====> {} file {} reconstruction finished'.format(num, file_name))

        # Error estimate
        eval_rmse_td = '{:.6f}'.format(rmse_td(x, x_recon))
        eval_rmse_fd = '{:.6f}'.format(rmse_fd(x, x_recon, wlen_sec, nfft, hop, wlen, win))
        eval_rmse_td_frame, eval_rmse_fd_frame = rmse_frame(x, x_recon, nfft, win)
        eval_rmse_td_frame = '{:.6f}'.format(eval_rmse_td_frame)
        eval_rmse_fd_frame = '{:.6f}'.format(eval_rmse_fd_frame)
        eval_stoi = '{:.6f}'.format(stoi(x, x_recon, fs=fs))
        eval_dic['rmse_td'].append(eval_rmse_td)
        eval_dic['rmse_fd'].append(eval_rmse_fd)
        eval_dic['rmse_td_frame'].append(eval_rmse_td_frame)
        eval_dic['rmse_fd_frame'].append(eval_rmse_fd_frame)
        eval_dic['stoi'].append(eval_stoi)
    return eval_dic, model_class.tag_simple


# test
def test(result_dir):
    dir_list = ['WSJ0_2020-02-26-15h07_FFNN_z_dim=16',
                'WSJ0_2020-02-22-12h32_UniEnc_UniDec_NoRecZ_z_dim=16',
                'WSJ0_2020-02-22-17h12_UniEnc_UniDec_RecZ_z_dim=16',
                'WSJ0_2020-02-22-23h55_BiEnc_BiDec_NoRecZ_z_dim=16',
                'WSJ0_2020-02-23-05h22_BiEnc_BiDec_RecZ_z_dim=16']

    # audio to test
    root_dir = '/Users/xiaoyu/WorkStation/Project_rvae/Data/clean_speech/wsj0_si_et_05'
    male_dir = '440'
    female_dir = '441'
    male_audios = ['440c020a.wav', '440c020b.wav', '440c020c.wav', '440c020d.wav', '440c020e.wav']
    female_audios = ['441c020a.wav', '441c020b.wav', '441c020c.wav', '441c020d.wav', '441c020e.wav']

    # male
    male_list = []
    for file in male_audios:
        audio_file = os.path.join(root_dir, male_dir, file)
        male_list.append(audio_file)
    
    # female
    female_list = []
    for file in female_audios:
        audio_file = os.path.join(root_dir, female_dir, file)
        female_list.append(audio_file)
    
    audio_list = male_list + female_list
    eval_dic_list = []
    tag_list = []

    for f in dir_list:
        train_dir = os.path.join(result_dir, f)
        if os.path.isdir(train_dir):   
            print('=====> Test folder: {}'.format(f))
            cfg_file, weight_file = find_file(train_dir)
            if cfg_file == None or weight_file == None:
                print('cfg file ({}) or weight file ({}) not found'.format(cfg_file, weight_file))
                return
            eval_dic, tag = resynthesis(cfg_file, weight_file, audio_list)
            eval_file = os.path.join(train_dir, 'test', 'eval.txt')
            write_eval(eval_dic, audio_list, eval_file)
            eval_dic_list.append(eval_dic)
            tag_list.append(tag)
    
    write_eval_latex(result_dir, eval_dic_list, tag_list, audio_list)
    


# test all models 
# should indicate the directory where models have been saved
if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        result_dir = sys.argv[1]
        test(result_dir)
        # dir_list = os.listdir(result_dir) 
    else:
        print('Please indicate the directory of results')
        
        
