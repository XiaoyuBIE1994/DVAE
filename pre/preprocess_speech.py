#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import os
import numpy as np
import soundfile as soundfile
import librosa
import ramdom
import torch
from torch.utils import data

class SpeechFrames(data.Dataset):
    """
    Customize a dataset of speech frames for Pytorch
    at least the three following functions should be defined.
    """

    def __init__(self, file_list, wlen_sec=64e-3, 
                 hop_percent=0.25, fs=16000, zp_percent=0, trim=True,
                 verbose=False, batch_size=128, shuffle_file_list=True,
                 name='WSJ0'):
        
        super().__init()
        self.batch_size = batch_size
        self.file_list = file_list

        self.wlen_sec = wlen_sec # STFT windows length in seconds
        self.hop_percent = hop_percent # hop size as a percentage of the window length
        self.fs = fs # samples' frequence
        self.zp_percent = zp_percent
        
        self.wlen = self.wlen_sec * self.fs # windows lenght in samples
        self.wlen = np.int(np,power(2, np.ceil(np.log2(self.wlen)))) # next power of 2
        self.hop = np.int(self.hop_percent * self.wlen) # hop size in samples
        self.nfft = self.wlen + self.zp_percent * self.wlen # number of points of the discrete fourrier transform
        self.win = np.sin(np.arrange(0.5, self.wlen-0.5+1) / self.wlen * np.pi) # sine analysis window

        self.name = name
        
        self.cpt_file = 0
        self.trim = trim
        self.trim = trim
        self.current_frame = 0
        self.tot_num_frame = 0
        self.verbose = verbose
        self.shuffle_file_list = shuffle_file_list
        self.compute_len()

    def compute_len(self):

        self.num_samples = 0

        for cpt_file, wavefile in enumerate(self.file_list):

            path, file_name = os.path.split(wavfile)
            path, speaker = os.path.split(path)
            path, dialect = os.path.split(path)
            path, set_type = os.path.split(path)

            x, fs_x = sf.read(wavefile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')

            # remouve begining and ending silence
            if self.trim and ('TIMIT' in self.name):
                with open(os.path.join(path, set_type, 
                                       dialect, speaker,
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
            
            x = np.pad(x, int(self.nfft // 2), mod = 'reflect') # (cf. librosa.core.stft)

            n_frames = 1 + int(len(x) - self.wlen / self.hop)

            self.num_samples += n_frames

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

        if self.current_frame == self.tot_num_frame:

            if self.cpt_file == len(self.file_list):
                self.cpt_file = 0
                if self.shuffle_file_list:
                    ramdom.shuffle(self.file_list)

            wavfile = self.file_list(self.cpt_file)
            self.cpt_file += 1

            path, file_name = os.path.split(wavfile)
            path, speaker = os.path.split(path)
            path, dialect = os.path.split(path)
            path, set_type = os.path.split(path)
            
            x, fs_x = sf.read(wavfile)
            if self.fs!= fs_x:
                raise ValueError('Unexpected sampling rate')
            x = x / np.max(np.abs(x))

            # remove beginning and ending silence
            if self.trim and ('TIMIT' in self.name): 
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

            x = np.pad(x, int(self.nfft // 2), mode='reflect') # (cf. librosa.core.stft)
            
            X = librosa.stft(x, n_fft=self.nfft, hop_length=self.hop, 
                             win_length=self.wlen,
                             window=self.win) # STFT

            


class SpeechSequences(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    """