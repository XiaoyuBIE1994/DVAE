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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
from build_model import build_model


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


# write evaluation in files
def write_eval(eval_dic, audio_list, save_path):
    with open(save_path, 'w') as f:
        eval_list = list(eval_dic.keys())
        headline = ['filename'] + eval_list
        for item in headline:
            f.write('{}\t'.format(item))
        f.write('\n')
        for n, file in enumerate(audio_list):
            _, filename = os.path.split(file)
            f.write(filename)
            for eval_method in eval_list:
                f.write('\t{}'.format(eval_dic[eval_method][n]))
            f.write('\n')

# write result in tex table format
def write_eval_latex(result_dir, eval_dic_list, tag_list, audio_list):
    eval_list = list(eval_dic_list[0].keys())
    for eval_item in eval_list:
        file_name = os.path.join(result_dir, '{}.tex'.format(eval_item))
        with open(file_name, 'w') as f:
            f.write('\\begin{table}[H]\n')             # \begin{table}[H]
            f.write('\\centering\n')                   # \centering
            f.write('\\begin{tabular}{r l l l l l}\n') # \begin{tabular}{rlllll}
            f.write('\\toprule\n')                     # \toprule
            f.write('filename')
            for item in tag_list:
                f.write(' & {}'.format(item))
            f.write(' \\\\\n')
            f.write('\\midrule\n') # \midrule
            for n, audio_file in enumerate(audio_list):
                _, audio_name = os.path.split(audio_file)
                f.write(audio_name)
                for i, model in enumerate(tag_list):
                    f.write(' & {}'.format(eval_dic_list[i][eval_item][n]))
                f.write(' \\\\\n')
            f.write('\\bottomrule\n')     # \bottomrule
            f.write('\\end{tabular}\n') # \end{tabular}
            f.write('\\caption{ }\n')   # \caption{ }
            f.write('\\label{ }\n')     # \label{ }
            f.write('\\end{table}')     # \end{table}


###
# Define different kind of evaluation function
###

# mean square error
def mse(A, B):
    return np.square(A - B).mean()



# resynthesis an audio with pre-trained weight 
def resynthesis(cfg_file, weight_file, audio_file_list):
    
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
    # print('win shape: {}'.format(win.shape))

    # create a dictionary for evaluation results
    eval_dic = {'mse': []}
    fig = plt.figure()
    
    for num, file in enumerate(audio_file_list):
        
        # allocate tag
        root_dir, filename = os.path.split(file)
        if filename[:3] == '440':
            tag = 'male_440'
        elif filename[:3] == '441':
            tag = 'female_441'

        # read audio file and transform via stft
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

        # Re-synthesis
        X_recon = np.sqrt(data_recon) * np.exp(1j * np.angle(X))
        x_recon = librosa.istft(X_recon, hop_length=hop, win_length=wlen, window=win)
        x_origin = x

        scale = 1 / (np.maximum(np.max(np.abs(x_recon)), np.max(np.abs(x_origin)))) * 0.9
        
        dir_name, file_name = os.path.split(weight_file)
        dir_name = os.path.join(dir_name, 'test')

        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        
        orig_file = os.path.join(dir_name, '{}_origin-{}.wav'.format(tag, num))
        recon_file = os.path.join(dir_name, '{}_recon-{}.wav'.format(tag, num))
        plot_file = os.path.join(dir_name, '{}_resynthesis-{}.png'.format(tag, num))

        # librosa.output.write_wav(orig_file, scale*x_origin, fs)
        # librosa.output.write_wav(recon_file, scale*x_recon, fs)
        # plt.savefig(plot_file)
        # print('=====> {} file {} reconstruction finished'.format(num, file_name))

        # Error estimate
        len_x = len(x_recon)
        mse_error = '{:.6f}'.format(mse(x[:len_x], x_recon))
        # mse_error = '{:.3f}'.format(mse(data_orig, data_recon))
        eval_dic['mse'].append(mse_error)
        
    # return eval_dic, x, X, data_orig, x_recon, X_recon, data_recon
    return eval_dic, model_class.tag_simple


###
# Test over 5 males audios and 5 female audios
###
def test(train_dir, audio_list):

    cfg_file, weight_file = find_file(train_dir)
    if cfg_file == None or weight_file == None:
        print('cfg file ({}) or weight file ({}) not found'.format(cfg_file, weight_file))
        return

    eval_dic, tag = resynthesis(cfg_file, weight_file, audio_list)
    eval_file = os.path.join(train_dir, 'test', 'eval.txt')
    write_eval(eval_dic, audio_list, eval_file)
    return eval_dic, tag


# test all models 
# should indicate the directory where models have been saved
if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        result_dir = sys.argv[1]
        # dir_list = os.listdir(result_dir) 
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
                eval_dic, tag = test(train_dir, audio_list)
                eval_dic_list.append(eval_dic)
                tag_list.append(tag)
        
        write_eval_latex(result_dir, eval_dic_list, tag_list, audio_list)
    else:
        print('Please indicate the directory of results')
        
