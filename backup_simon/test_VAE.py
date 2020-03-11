#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 by Inria
Authored by Simon Leglaive (simon.leglaive@inria.fr)
License agreement in LICENSE.txt
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf 
import librosa
from VAEs import VAE
import random
import librosa.display
import os

plt.close('all')

def write_eval(eval_dic, file_list, save_path):
    with open(save_path, 'w') as f:
        eval_list = list(eval_dic.keys())
        headline = ['filename'] + eval_list
        for item in headline:
            f.write('{}\t'.format(item))
        f.write('\n')
        for n, file in enumerate(file_list):
            _, filename = os.path.split(file)
            f.write(filename)
            for eval_method in eval_list:
                f.write('\t{}'.format(eval_dic[eval_method][n]))
            f.write('\n')

def mse(A, B):
    return np.square(A - B).mean()

# network parameters
input_dim = 513
latent_dim = 16   
hidden_dim_encoder = [128]
activation = torch.tanh


# STFT parameters
wlen_sec=64e-3
hop_percent=0.25
fs=16000
zp_percent=0
trim=True
verbose=False

# test parameters

use_gpu = False

# saved_model = '/local_scratch/sileglai/recherche/python/speech_enhancement/SE_FFNN_RNN_VAEs_pytorch/saved_model/VAE_first_test_11_june/final_model_VAE_epoch125.pt'

saved_dir = '/Users/xiaoyu/WorkStation/Project_rvae/saved_model_Simon/WSJ0_2019-07-15-10h21_FFNN_VAE_latent_dim=16'
saved_model = os.path.join(saved_dir, 'final_model_RVAE_epoch65.pt')

device = 'cpu'
# init model
vae = VAE(input_dim=input_dim, latent_dim=latent_dim, 
            hidden_dim_encoder=hidden_dim_encoder,
            activation=activation)

vae.load_state_dict(torch.load(saved_model, map_location='cpu'))

# ! important ! to discard Dropout,BatchNorm layers during test
vae.eval()

if use_gpu:
    vae = vae.cuda()

#plt.close('all')    

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

audiofile_list = []
for file in file_list:
    audio_file = os.path.join(data_dir, file)
    audiofile_list.append(audio_file)

saved_dir = os.path.join(saved_dir, 'test')
if not os.path.isdir(saved_dir):
    os.mkdir(saved_dir)

wlen = int(wlen_sec*fs) # window length of 64 ms
wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2
nfft = wlen
hop = np.int(hop_percent*wlen) # hop size
win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window

for num, wavfile in enumerate(audiofile_list):
    x, fs_x = sf.read(wavfile)    
    x = x/np.max(np.abs(x))
    X = librosa.stft(x, n_fft=nfft, hop_length=hop, 
                                win_length=wlen,
                                window=win) # STFT
    data_orig = np.abs(X)**2

    with torch.no_grad():
        
        data_orig = data_orig.T
        data_orig = torch.from_numpy(data_orig.astype(np.float32))
        data_orig = data_orig.to(device)
        
        data_recon, mean, logvar, z = vae(data_orig)
        mean = mean.detach().numpy().T
        data_recon = data_recon.detach().numpy().T
        
        data_orig = data_orig.detach().numpy().T
        

    plt.figure()
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.power_to_db(data_orig), y_axis='log', sr=fs, hop_length=hop)#, vmin=-50, vmax=20)
    #librosa.display.specshow(librosa.power_to_db(x_train[0:500,:].T), y_axis='log', sr=fs, hop_length=hop)
    plt.set_cmap('jet')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original spectrogram')

    plt.subplot(3, 1, 2)
    plt.imshow(mean, origin='lower')
    plt.set_cmap('jet')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.power_to_db(data_recon), x_axis='time', y_axis='log', sr=fs, hop_length=hop)#, vmin=-50, vmax=20)
    #librosa.display.specshow(librosa.power_to_db(data_decoded.T), x_axis='time', y_axis='log', sr=fs, hop_length=hop)
    plt.set_cmap('jet')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed spectrogram')

    X_recon = np.sqrt(data_recon)*np.exp(1j*np.angle(X))

    x_recon = librosa.istft(X_recon, hop_length=hop, win_length=wlen, window=win)
    x_orig = x

    scale = 1/(np.maximum(np.max(np.abs(x_recon)),np.max(np.abs(x_orig))))*0.9

    

    orig_file = os.path.join(saved_dir, 'man_440_origin-{}.wav'.format(num))
    recon_file = os.path.join(saved_dir, 'man_440_recon-{}.wav'.format(num))
    plot_file = os.path.join(saved_dir, 'man_440_resynthesis-{}.png'.format(num))

    # orig_file = os.path.join(saved_dir, 'women_441_origin-{}.wav'.format(num))
    # recon_file = os.path.join(saved_dir, 'women_441_recon-{}.wav'.format(num))
    # plot_file = os.path.join(saved_dir, 'women_441_resynthesis-{}.png'.format(num))

    # librosa.output.write_wav(orig_file, scale*x_orig, fs)
    # librosa.output.write_wav(recon_file, scale*x_recon, fs)
    # plt.savefig(plot_file)


    mse_error = '{:.4f}'.format(mse(data_orig, data_recon))
    eval_dic['mse'].append(mse_error)
    print('{}-{}: mse error: {}'.format(tag, num, mse_error))
    print('=====> {} file {} reconstruction finished'.format(num, wavfile))

write_eval(eval_dic, audio_list, eval_file)

#%%
#z = np.zeros((latent_dim,500))
#z[:,0] = mean[:,int(np.random.rand()*mean.shape[1])]
#with torch.no_grad():
#    for n in np.arange(1,z.shape[1]):
#        z[:,n] = z[:,n-1] + 0.5*np.random.randn(z.shape[0],)
#    z_torch = torch.from_numpy(z.astype(np.float32))
#    z_torch = z_torch.unsqueeze(0) # add a dimension in axis 0
#    z_torch = z_torch.permute(0,-1,1) # (batch_size, sequence_len, input_dim)
#    data_gen = vae.decode(z_torch)
#    data_gen = data_gen.squeeze().numpy().T
#    
#plt.figure()
#librosa.display.specshow(librosa.power_to_db(data_gen), x_axis='time', y_axis='log', sr=fs, hop_length=hop)#, vmin=-50, vmax=20)
#plt.set_cmap('jet')
#plt.colorbar(format='%+2.0f dB')
#plt.title('generated spectrogram')
