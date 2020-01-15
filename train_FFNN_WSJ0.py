#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 by Inria
Authored by Simon Leglaive (simon.leglaive@inria.fr)
License agreement in LICENSE.txt
"""

#%% load code

my_seed = 0
import numpy as np
np.random.seed(my_seed)
import torch
torch.manual_seed(my_seed)
from torch.utils import data
from torch import optim
from speech_dataset import SpeechDatasetFrames
import matplotlib
matplotlib.use('Agg')     
import matplotlib.pyplot as plt
import librosa
from VAEs import VAE
import os
import socket
import pickle
import datetime
from prepare_dataset import prepare_dataset_WSJ0


hostname =  socket.gethostname() 
print('HOSTNAME: ' + hostname)
date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")
print(date)

#%% check if dataset is on local machine

dataset_name = 'WSJ0'
# train_data_dir, val_data_dir = prepare_dataset_WSJ0(dataset_name, hostname)
train_data_dir = "/scratch2/octans/sileglai/datasets/clean_speech/wsj0_si_tr_s"
val_data_dir = "/scratch2/octans/sileglai/datasets/clean_speech/wsj0_si_dt_05"

#%% network parameters

input_dim = 513
latent_dim = 16   
hidden_dim_encoder = [128]
activation_str = 'torch.tanh'
activation = eval(activation_str)

#%% create directory for results

save_dir = os.path.join('../saved_model', dataset_name + '_' + date + 
                        '_FFNN_VAE_'  + 'latent_dim=' + str(latent_dim))

if not(os.path.isdir(save_dir)):
    os.makedirs(save_dir)

print(save_dir)

#%% STFT parameters

wlen_sec = 64e-3
hop_percent = 0.25
fs = 16000
zp_percent = 0
trim = True
verbose = False

#%% training parameters

if hostname=='virgo':
    device = 'cpu'
else:
    device = 'cuda'
    
train_file_list = librosa.util.find_files(train_data_dir, ext='wav')
val_file_list = librosa.util.find_files(val_data_dir, ext='wav')

lr = 0.001
epochs = 300
batch_size = 128
num_workers = 1 # if higher, may mess-up everything as the data loading is 
# based on some class attributes rather than on the 'index' variable 
# of __getitem__.
shuffle_file_list = True
shuffle_samples_in_batch = True

save_frequency = 10
early_stopping_patience = 20

# create dataloader 
print('instanciate training data loader')
train_dataset = SpeechDatasetFrames(file_list=train_file_list, 
                                    wlen_sec=wlen_sec, 
                                    hop_percent=hop_percent, fs=fs, 
                                    zp_percent=zp_percent, trim=trim, 
                                    verbose=verbose, batch_size=batch_size, 
                                    shuffle_file_list=shuffle_file_list,
                                    name=dataset_name)

print('instanciate validation data loader')
val_dataset = SpeechDatasetFrames(file_list=val_file_list, 
                                  wlen_sec=wlen_sec, 
                                  hop_percent=hop_percent, fs=fs, 
                                  zp_percent=zp_percent, trim=trim, 
                                  verbose=verbose, batch_size=batch_size, 
                                  shuffle_file_list=shuffle_file_list,
                                  name=dataset_name)

# torch load will call __getitem__ of TIMIT to create Batch by randomly 
# (if shuffle=True) selecting data sample.
print('create training data loader')
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, 
                                   shuffle=shuffle_samples_in_batch, 
                                   num_workers=num_workers)

print('create validation data loader')
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, 
                                   shuffle=shuffle_samples_in_batch, 
                                   num_workers=num_workers)

# init model
print('init VAE')
vae = VAE(input_dim=input_dim, latent_dim=latent_dim, 
            hidden_dim_encoder=hidden_dim_encoder, batch_size=batch_size, 
            activation=activation).to(device)         


num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)

# optimizer
optimizer = optim.Adam(vae.parameters(), lr=lr)

# loss function
def loss_function(recon_x, x, mu, logvar):  
    recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
    KLD = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp())
    return recon + KLD

#%% main loop for training 
    
train_loss = np.zeros((epochs,))
val_loss = np.zeros((epochs,))
best_val_loss = np.inf
cpt_patience = 0

print('training loop')
for epoch in range(epochs):
    
    start_time = datetime.datetime.now()
    vae.train()
    
    for batch_idx, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mean, logvar, z = vae(batch)
        loss = loss_function(recon_batch, batch, mean, logvar)
        loss.backward()
        train_loss[epoch] += loss.item()
        optimizer.step()    
        
    for batch_idx, batch in enumerate(val_dataloader):
        batch = batch.to(device)
        recon_batch, mean, logvar, z = vae(batch)
        loss = loss_function(recon_batch, batch, mean, logvar)
        val_loss[epoch] += loss.item()
        
    if val_loss[epoch] < best_val_loss:
        best_val_loss = val_loss[epoch]
        cpt_patience = 0
        best_state_dict = vae.state_dict()
        cur_best_epoch = epoch
    else:
        cpt_patience += 1
        
    train_loss[epoch] = train_loss[epoch] / train_dataset.num_samples
    val_loss[epoch] = val_loss[epoch] /  val_dataset.num_samples
    
    end_time = datetime.datetime.now()
    interval = (end_time - start_time).seconds / 60
    print('====> Epoch: {} train loss: {:.4f} val loss: {:.4f} traning time: {:.2f}m'.format(
          epoch, train_loss[epoch], val_loss[epoch], interval))
    
    if cpt_patience == early_stopping_patience:
        print('Early stopping patience achieved')
        break
    
    if epoch % save_frequency == 0:
            # save parameters of your model
            save_file = os.path.join(save_dir, 'VAE_epoch' + str(cur_best_epoch) + '.pt')
            torch.save(best_state_dict, save_file)

train_loss = train_loss[:epoch+1]
val_loss = val_loss[:epoch+1]

save_file = os.path.join(save_dir, 'final_model_RVAE_epoch' + str(cur_best_epoch) + '.pt')
torch.save(best_state_dict, save_file)

loss_file = os.path.join(save_dir, 'loss_RVAE.pckl')

with open(loss_file, 'wb') as f:
        pickle.dump([train_loss, val_loss], f)

#%% Save parameters

dic_params = {'input_dim':input_dim,
              'latent_dim':latent_dim,
              'hidden_dim_encoder':hidden_dim_encoder,
              'activation':activation_str,
              'wlen_sec':wlen_sec,
              'hop_percent':hop_percent,
              'fs':fs,
              'zp_percent':zp_percent,
              'trim':trim,
              'epochs':epochs,
              'batch_size':batch_size,
              'num_workers':num_workers,
              'shuffle_file_list':shuffle_file_list,
              'shuffle_samples_in_batch':shuffle_samples_in_batch,
              'save_frequency':save_frequency,
              'early_stopping_patience':early_stopping_patience,
              'my_seed':my_seed,
              'train_data_dir':train_data_dir,
              'val_data_dir':val_data_dir,
              'num_params':num_params,              
              'date':date}

# Write the VAE and training parameters to a text file
params_text_file = os.path.join(save_dir, 'parameters.txt')
with open(params_text_file, 'w') as f:
        for key, value in dic_params.items():
            f.write('%s:%s\n' % (key, value))


# Write the VAE and training parameters to a pickle file
params_pckl_file = os.path.join(save_dir, 'parameters.pckl')
f = open(params_pckl_file, 'wb')
pickle.dump(dic_params, f)
f.close()


# save loss figure   
plt.clf()
plt.plot(train_loss, '--o')
plt.plot(val_loss, '--x')
plt.legend(('train loss', 'val loss'))
plt.xlabel('epochs')

loss_figure_file = os.path.join(save_dir, 'loss.pdf')

plt.savefig(loss_figure_file)