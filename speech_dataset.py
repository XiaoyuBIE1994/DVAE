#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 by Inria
Authored by Simon Leglaive (simon.leglaive@inria.fr)
License agreement in LICENSE.txt
"""

import torch
from torch.utils import data
import os
import numpy as np
import soundfile as sf 
import librosa
import random

class SpeechDatasetFrames(data.Dataset):
    """
    Customize a dataset for PyTorch, in order to be used with torch dataloarder,
    at least the three following functions should be defined.
    """

    def __init__(self, file_list, wlen_sec=64e-3, 
                 hop_percent=0.25, fs=16000, zp_percent=0, trim=True,
                 verbose=False, batch_size=128, shuffle_file_list=True, 
                 name='WSJ0'):

        super(SpeechDatasetFrames).__init__()
        self.batch_size = batch_size
        self.file_list = file_list
        
        self.wlen_sec = wlen_sec # STFT window length in seconds
        self.hop_percent = hop_percent  # hop size as a percentage of the window length
        self.fs = fs
        self.zp_percent = zp_percent
        self.wlen = self.wlen_sec*self.fs # window length in samples
        self.wlen = np.int(np.power(2, np.ceil(np.log2(self.wlen)))) # next power of 2
        self.hop = np.int(self.hop_percent*self.wlen) # hop size in samples
        self.nfft = self.wlen + self.zp_percent*self.wlen # number of points of the discrete Fourier transform
        self.win = np.sin(np.arange(.5,self.wlen-.5+1)/self.wlen*np.pi); # sine analysis window
        
        self.name = name
        
        self.cpt_file = 0
        self.trim = trim
        self.current_frame = 0
        self.tot_num_frame = 0
        self.verbose = verbose
        self.shuffle_file_list = shuffle_file_list
        self.compute_len()
        
    def compute_len(self):
        
        self.num_samples = 0
        
        for cpt_file, wavfile in enumerate(self.file_list):

            path, file_name = os.path.split(wavfile)
            path, speaker = os.path.split(path)
            path, dialect = os.path.split(path)
            path, set_type = os.path.split(path)
            
            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')        
                
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
            
            n_frames = 1 + int((len(x) - self.wlen) / self.hop)
            
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
                    random.shuffle(self.file_list)
            
            wavfile = self.file_list[self.cpt_file]
            self.cpt_file += 1
            
            path, file_name = os.path.split(wavfile)
            path, speaker = os.path.split(path)
            path, dialect = os.path.split(path)
            path, set_type = os.path.split(path)
            
            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')        
            x = x/np.max(np.abs(x))
            
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
            
            self.data = np.abs(X)**2
            self.current_frame = 0
            self.tot_num_frame = self.data.shape[1]
            
        frame = self.data[:,self.current_frame]    
        self.current_frame += 1
        
        # turn numpy array to torch tensor with torch.from_numpy#
        """
        e.g.
        matrix = torch.from_numpy(matrix.astype(np.float32))
        target = torch.from_numpy(np.load(t_pth).astype(np.int32))
        """
        frame = torch.from_numpy(frame.astype(np.float32))
        return frame


class SpeechDatasetSequences(data.Dataset):
    """
    Customize a dataset for PyTorch, in order to be used with torch dataloarder,
    at least the three following functions should be defined.
    """

    def __init__(self, file_list, sequence_len=50, wlen_sec=64e-3, 
                 hop_percent=0.25, fs=16000, zp_percent=0, trim=True,
                 verbose=False, batch_size=128, shuffle_file_list=True,
                 name='WSJ0'):

        super(SpeechDatasetSequences).__init__()
        self.batch_size = batch_size
        self.file_list = file_list
        
        self.wlen_sec = wlen_sec # STFT window length in seconds
        self.hop_percent = hop_percent  # hop size as a percentage of the window length
        self.fs = fs
        self.zp_percent = zp_percent
        self.wlen = self.wlen_sec*self.fs # window length in samples
        self.wlen = np.int(np.power(2, np.ceil(np.log2(self.wlen)))) # next power of 2
        self.hop = np.int(self.hop_percent*self.wlen) # hop size in samples
        self.nfft = self.wlen + self.zp_percent*self.wlen # number of points of the discrete Fourier transform
        self.win = np.sin(np.arange(.5,self.wlen-.5+1)/self.wlen*np.pi); # sine analysis window
        
        self.name = name
        
        self.cpt_file = 0
        self.trim = trim
        self.current_frame = 0
        self.tot_num_frame = 0
        self.data = None
        self.verbose = verbose
        self.shuffle_file_list = shuffle_file_list
        
        self.sequence_len = sequence_len
        
        self.compute_len()
        
    def compute_len(self):
        
        self.num_samples = 0
        
        for cpt_file, wavfile in enumerate(self.file_list):

            path, file_name = os.path.split(wavfile)
            path, speaker = os.path.split(path)
            path, dialect = os.path.split(path)
            path, set_type = os.path.split(path)
            
            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')        
                
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
            
            n_seq = (1 + int((len(x) - self.wlen) / self.hop) )//self.sequence_len
            
            # n_seq can be equal to 0 if some audio files are too short 
            # compared with the expected sequence length
            
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
        
        if  (self.tot_num_frame - self.current_frame) < self.sequence_len:
        
            while True:
            
                if self.cpt_file == len(self.file_list):
                    self.cpt_file = 0
                    if self.shuffle_file_list:
                        random.shuffle(self.file_list)
                
                wavfile = self.file_list[self.cpt_file]
                self.cpt_file += 1
                
                path, file_name = os.path.split(wavfile)
                path, speaker = os.path.split(path)
                path, dialect = os.path.split(path)
                path, set_type = os.path.split(path)
                
                x, fs_x = sf.read(wavfile)
                if self.fs != fs_x:
                    raise ValueError('Unexpected sampling rate')        
                x = x/np.max(np.abs(x))
                
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
                
                self.data = np.abs(X)**2
                self.current_frame = 0
                self.tot_num_frame = self.data.shape[1]
                
                if self.tot_num_frame >= self.sequence_len:
                    break
            
        sample = self.data[:,self.current_frame:self.current_frame+self.sequence_len]    
        if sample.shape[1] != self.sequence_len:
            print(self.data.shape)
        self.current_frame += self.sequence_len
        

        # turn numpy array to torch tensor with torch.from_numpy#
        """
        e.g.
        matrix = torch.from_numpy(matrix.astype(np.float32))
        target = torch.from_numpy(np.load(t_pth).astype(np.int32))
        """
        sample = torch.from_numpy(sample.astype(np.float32))
        return sample

