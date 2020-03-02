"""
test your model after training
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf 
import librosa
from VAEs import RVAE
import random
import librosa.display

plt.close('all')
# network parameters

input_dim = 513
latent_dim = 16   
hidden_dim = 128
activation = torch.tanh
num_LSTM = 1
num_dense_enc = 1

# STFT parameters

wlen_sec=64e-3
hop_percent=0.25
fs=16000
zp_percent=0
trim=True
verbose=False

# test parameters

use_gpu = False
# rnn rec z
# saved_dir = '/Users/xiaoyu/WorkStation/saved_model_Simon/WSJ0_2019-07-15-10h14_RVAE_RNNenc_RNNdec_RecZ_latent_dim=16'
# saved_model = os.path.join(saved_dir, 'final_model_RVAE_epoch121.pt')
# rec_over_z = True
# bidir_enc_s = False
# bidir_dec = False

# # rnn no rec z
# saved_dir = '/Users/xiaoyu/WorkStation/saved_model_Simon/WSJ0_2019-08-01-12h45_RVAE_RNNenc_RNNdec_NoRecZ_latent_dim=16'
# saved_model = os.path.join(saved_dir, 'final_model_RVAE_epoch72.pt')
# rec_over_z = False
# bidir_enc_s = False
# bidir_dec = False

# # brnn rec z
# saved_dir = '/Users/xiaoyu/WorkStation/saved_model_Simon/WSJ0_2019-07-15-10h01_RVAE_BRNNenc_BRNNdec_RecZ_latent_dim=16'
# saved_model = os.path.join(saved_dir, 'final_model_RVAE_epoch145.pt')
# rec_over_z = True
# bidir_enc_s = True
# bidir_dec = True

# # brnn no rec z
saved_dir = '/Users/xiaoyu/WorkStation/saved_model_Simon/WSJ0_2019-08-02-08h52_RVAE_BRNNenc_BRNNdec_NoRecZ_latent_dim=16'
saved_model = os.path.join(saved_dir, 'final_model_RVAE_epoch120.pt')
rec_over_z = False
bidir_enc_s = True
bidir_dec = True

device = 'cpu'

# init model
vae = RVAE(input_dim=input_dim, h_dim=hidden_dim, z_dim=latent_dim, 
           num_LSTM=num_LSTM, num_dense_enc=num_dense_enc, 
           bidir_enc_s=bidir_enc_s, bidir_dec=bidir_dec, 
           rec_over_z=rec_over_z, device=device)     


vae.load_state_dict(torch.load(saved_model, map_location='cpu'))

# ! important ! to discard Dropout,BatchNorm layers during test
vae.eval()

if use_gpu:
    vae = vae.cuda()


#plt.close('all')    

# data_dir = '/local_scratch/sileglai/datasets/clean_speech/TIMIT/TEST'
# file_list = librosa.util.find_files(data_dir, ext='wav')
#wavfile = random.choice(file_list)

# data_dir = '/Users/xiaoyu/WorkStation/Data/clean_speech/wsj0_si_et_05/440'
# file_list = ['440c020a.wav', '440c020b.wav', '440c020c.wav', '440c020d.wav', '440c020e.wav']

data_dir = '/Users/xiaoyu/WorkStation/Data/clean_speech/wsj0_si_et_05/441'
file_list = ['441c020a.wav', '441c020b.wav', '441c020c.wav', '441c020d.wav', '441c020e.wav']

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
        
    #    data_orig = torch.from_numpy(data_orig.astype(np.float32))
    #    data_orig = data_orig.unsqueeze(0) # add a dimension in axis 0
    #    data_orig = data_orig.permute(-1,0,1) # (sequence_len, batch_size, input_dim)
    #    data_orig = data_orig.to(device)
        
        data_orig = data_orig.T
        data_orig = torch.from_numpy(data_orig.astype(np.float32))
        data_orig = data_orig.to(device)
        
        data_recon, mean, logvar, z = vae(data_orig)
        mean = mean.detach().numpy().T
        data_recon = data_recon.detach().numpy().T
        data_orig = data_orig.detach().numpy().T
        
    #    mean = mean[:,0,:].detach().numpy().T
    #    data_recon = data_recon[:,0,:].detach().numpy().T
    #    data_orig = data_orig[:,0,:].detach().numpy().T

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

    # orig_file = os.path.join(saved_dir, 'man_440_origin-{}.wav'.format(num))
    # recon_file = os.path.join(saved_dir, 'man_440_recon-{}.wav'.format(num))
    # plot_file = os.path.join(saved_dir, 'man_440_resynthesis-{}.png'.format(num))

    orig_file = os.path.join(saved_dir, 'women_441_origin-{}.wav'.format(num))
    recon_file = os.path.join(saved_dir, 'women_441_recon-{}.wav'.format(num))
    plot_file = os.path.join(saved_dir, 'women_441_resynthesis-{}.png'.format(num))

    librosa.output.write_wav(orig_file, scale*x_orig, fs)
    librosa.output.write_wav(recon_file, scale*x_recon, fs)
    plt.savefig(plot_file)

    print('=====> {} file {} reconstruction finished'.format(num, wavfile))



#%%

#z = np.zeros((latent_dim,150))
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
#
#
## Griffin-Lim
#T = librosa.istft(data_gen, hop_length=hop, win_length=wlen, window=win).shape[0]
#x_reconstruct = np.random.randn(T)
#niter_GL = 100
#for n in np.arange(niter_GL):
#    GL_STFT = librosa.stft(x_reconstruct, n_fft=nfft, 
#                           hop_length=hop, win_length=wlen,
#                           window=win)
#    GL_angle = np.angle(GL_STFT)
#    prop_STFT = data_gen*np.exp(1.0j*GL_angle)
#    prev_x = x_reconstruct
#    x_reconstruct = librosa.istft(prop_STFT, hop_length=hop, win_length=wlen, window=win)
#    diff = np.sqrt(sum((x_reconstruct - prev_x)**2)/x_reconstruct.size)
#    print('Reconstruction iteration: {}/{} RMSE: {} '.format(n, niter_GL, diff))
#
#x_reconstruct = x_reconstruct/np.max(x_reconstruct)
#
#librosa.output.write_wav('./signal_gen.wav', x_reconstruct, fs)
