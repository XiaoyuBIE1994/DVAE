#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import os
import numpy as np
import soundfile as sf
import librosa
import random
import torch
from torch.utils import data


class SpeechSequences(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    """
    def __init__(self,file_list, seq_len=50, wlen_sec=64e-3,
                 hop_percent=0.25, fs=16000, zp_percent=0, trim=True,
                 verbose=False, batch_size=32, shuffle_file_list=True,
                 name='WSJ0'):

        super().__init__()

        # STFT parameters
        self.wlen_sec = wlen_sec
        self.hot_percent = hop_percent
        self.fs = fs
        self.zp_percent = zp_percent
        self.wlen = self.wlen_sec * self.fs
        self.wlen = np.int(np.power(2, np.ceil(np.log2(self.wlen)))) # pwoer of 2
        self.hop = np.int(self.hot_percent * self.wlen)
        self.nfft = self.wlen + self.zp_percent * self.wlen
        self.win = np.sin(np.arange(0.5, self.wlen+0.5) / self.wlen * np.pi)

        # data parameters
        self.file_list = file_list
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.name = name
        self.cpt_file = 0
        self.trim = trim
        self.current_frame = 0
        self.tot_num_frame = 0
        self.data = None
        self.verbose = verbose
        self.shuffle_file_list = shuffle_file_list

        self.compute_len()


    def compute_len(self):

        self.num_samples = 0

        for cpt_file, wavfile in enumerate(self.file_list):

            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')
            
            # remove beginning and ending silence
            if self.trim and ('TIMIT' in self.name):
                path, file_name = os.path.split(wavfile)
                path, speaker = os.path.split(path)
                path, dialect = os.path.split(path)
                path, set_type = os.path.split(path)
                with open(os.path.join(path, set_type, dialect, speaker, file_name[:-4] + '.PHN'), 'r') as f:
                    first_line = f.readline() # Read the first line
                    for last_line in f: # Loop through the whole file reading it all
                        pass
                if not('#' in first_line) or not('#' in last_line):
                    raise NameError('The first of last lines of the .phn file should contain #')
                ind_beg = int(first_line.split(' ')[1])
                ind_end = int(last_line.split(' ')[0])
                x = x[ind_beg, ind_end]
            elif self.trim:
                x, _ = librosa.effects.trim(x, top_db=30)
            
            n_seq = (1 + int(len(x) / self.hop) )//self.seq_len

            self.num_samples += n_seq


    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return self.num_samples


    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        """
        
        if (self.tot_num_frame - self.current_frame) < self.seq_len:

            while True:
            
                if self.cpt_file == len(self.file_list):
                    self.cpt_file = 0
                    if self.shuffle_file_list:
                        random.shuffle(self.file_list)
                
                wavfile = self.file_list[self.cpt_file]
                self.cpt_file += 1
                
                x, fs_x = sf.read(wavfile)
                if self.fs != fs_x:
                    raise ValueError('Unexpected sampling rate')        
                x = x/np.max(np.abs(x))
                
                # remove beginning and ending silence
                if self.trim and ('TIMIT' in self.name): 
                    path, file_name = os.path.split(wavfile)
                    path, speaker = os.path.split(path)
                    path, dialect = os.path.split(path)
                    path, set_type = os.path.split(path)
                    with open(os.path.join(path, set_type, dialect, speaker,
                                           file_name[:-4]+'.PHN'), 'r') as f:
                        first_line = f.readline() # Read the first line
                        for last_line in f: # Loop through the whole file reading it all
                            pass
        
                    if not('#' in first_line) or not('#' in last_line):
                        raise NameError('The first or last lines of the .phn file should contain #')
            
                    ind_beg = int(first_line.split(' ')[1])
                    ind_end = int(last_line.split(' ')[0])
                    x = x[ind_beg:ind_end]
                    
                elif self.trim: 
                    x, index = librosa.effects.trim(x, top_db=30)
                
                X = librosa.stft(x, n_fft=self.nfft, hop_length=self.hop, 
                                 win_length=self.wlen,
                                 window=self.win) # STFT
                
                self.data = np.abs(X)**2
                self.current_frame = 0
                self.tot_num_frame = self.data.shape[1]
                
                if self.tot_num_frame >= self.seq_len:
                    break
            
        sample = self.data[:,self.current_frame:self.current_frame+self.seq_len]    
        if sample.shape[1] != self.seq_len:
            print(self.data.shape)
        self.current_frame += self.seq_len
        

        # turn numpy array to torch tensor with torch.from_numpy#
        """
        e.g.
        matrix = torch.from_numpy(matrix.astype(np.float32))
        target = torch.from_numpy(np.load(t_pth).astype(np.int32))
        """
        sample = torch.from_numpy(sample.astype(np.float32))
        return sample

                
class SpeechSequencesQ(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    
    This is a quick speech sequence data loader which allow multiple workers
    """
    def __init__(self,file_list, seq_len=50, wlen_sec=64e-3,
                 hop_percent=0.25, fs=16000, zp_percent=0, trim=True,
                 verbose=False, batch_size=32, shuffle_file_list=True,
                 name='WSJ0'):

        super().__init__()

        # STFT parameters
        self.wlen_sec = wlen_sec
        self.hot_percent = hop_percent
        self.fs = fs
        self.zp_percent = zp_percent
        self.wlen = self.wlen_sec * self.fs
        self.wlen = np.int(np.power(2, np.ceil(np.log2(self.wlen)))) # pwoer of 2
        self.hop = np.int(self.hot_percent * self.wlen)
        self.nfft = self.wlen + self.zp_percent * self.wlen
        self.win = torch.sin(torch.arange(0.5, self.wlen+0.5) / self.wlen * np.pi)

        # data parameters
        self.file_list = file_list
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.name = name
        self.cpt_file = 0
        self.trim = trim
        self.current_frame = 0
        self.tot_num_frame = 0
        self.data = None
        self.verbose = verbose
        self.shuffle_file_list = shuffle_file_list

        self.compute_len()


    def compute_len(self):

        self.valid_file_list = []

        for wavfile in self.file_list:
            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')
            x = x/np.max(np.abs(x))

            # Silence clipping
            if self.trim:
                x, idx = librosa.effects.trim(x, top_db=30)

            # Check valid wav files
            seq_length = (self.seq_len - 1) * self.hop
            if len(x) >= seq_length:
                self.valid_file_list.append(wavfile)

        if self.shuffle_file_list:
            random.shuffle(self.valid_file_list)

        self.num_samples = len(self.valid_file_list)


    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return self.num_samples


    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        """
        
        # Read wav files
        wavfile = self.valid_file_list[index]
        x, fs_x = sf.read(wavfile)
        if self.fs != fs_x:
            raise ValueError('Unexpected sampling rate')
        x = x/np.max(np.abs(x))

        # Silence clipping
        if self.trim:
            x, idx = librosa.effects.trim(x, top_db=30)

        file_length = len(x)
        seq_length = (self.seq_len - 1) * self.hop # sequence length in time domain
        start = np.random.randint(0, file_length - seq_length)
        end = start + seq_length

        audio_spec = torch.stft(torch.from_numpy(x[start:end]), n_fft=self.nfft, hop_length=self.hop, 
                                win_length=self.wlen, window=self.win, 
                                center=True, pad_mode='reflect', normalized=False, onesided=True)

        sample = (audio_spec[:,:,0]**2 + audio_spec[:,:,1]**2).float()

        return sample


        

