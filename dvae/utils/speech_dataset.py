#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

Class SpeechSequencesFull():
- generate Pytorch dataloader
- data sequence is clipped from the beginning of each audio signal
- every speech sequence can be divided into multiple data sequences, as long as audio_len >= seq_len
- usually, this method will give larger training sequences

Class SpeechSequencesRandom():
- generate Pytorch dataloader
- data sequence is clipped from a random place in each audio signal
- every speech sequence can only be divided into one single data sequence
- this method will introduce some randomness into training dataset

Both of these two Class use librosa.effects.trim()) to trim leading and trailing silence from an audio signal
"""

import os
import numpy as np
import soundfile as sf
import librosa
import random
import torch
from torch.utils import data



class SpeechSequencesFull(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    """
    def __init__(self, file_list, sequence_len, STFT_dict, shuffle, name='WSJ0'):

        super().__init__()

        # STFT parameters
        self.fs = STFT_dict['fs']
        self.nfft = STFT_dict['nfft']
        self.hop = STFT_dict['hop']
        self.wlen = STFT_dict['wlen']
        self.win = STFT_dict['win']
        self.trim = STFT_dict['trim']
        
        # data parameters
        self.file_list = file_list
        self.sequence_len = sequence_len
        self.name = name
        self.shuffle = shuffle

        self.compute_len()


    def compute_len(self):

        self.valid_seq_list = []

        for wavfile in self.file_list:

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
            elif self.trim:
                _, (ind_beg, ind_end) = librosa.effects.trim(x, top_db=30)


            # Check valid wav files
            seq_length = (self.sequence_len - 1) * self.hop
            file_length = ind_end - ind_beg 
            n_seq = (1 + int(file_length / self.hop)) // self.sequence_len
            for i in range(n_seq):
                seq_start = i * seq_length + ind_beg
                seq_end = (i + 1) * seq_length + ind_beg
                seq_info = (wavfile, seq_start, seq_end)
                self.valid_seq_list.append(seq_info)

        if self.shuffle:
            random.shuffle(self.valid_seq_list)


    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return len(self.valid_seq_list)


    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        """
        
        # Read wav files
        wavfile, seq_start, seq_end = self.valid_seq_list[index]
        x, fs_x = sf.read(wavfile)

        # Sequence tailor
        x = x[seq_start:seq_end]

        # Normalize sequence
        x = x/np.max(np.abs(x))

        # STFT transformation
        audio_spec = torch.stft(torch.from_numpy(x), n_fft=self.nfft, hop_length=self.hop, 
                                win_length=self.wlen, window=self.win, 
                                center=True, pad_mode='reflect', normalized=False, onesided=True)

        # Square of magnitude
        sample = (audio_spec[:,:,0]**2 + audio_spec[:,:,1]**2).float()

        return sample


                
class SpeechSequencesRandom(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    
    This is a quick speech sequence data loader which allow multiple workers
    """
    def __init__(self, file_list, sequence_len, STFT_dict, shuffle, name='WSJ0'):

        super().__init__()

        # STFT parameters
        self.fs = STFT_dict['fs']
        self.nfft = STFT_dict['nfft']
        self.hop = STFT_dict['hop']
        self.wlen = STFT_dict['wlen']
        self.win = STFT_dict['win']
        self.trim = STFT_dict['trim']
        
        # data parameters
        self.file_list = file_list
        self.sequence_len = sequence_len
        self.name = name
        self.shuffle = shuffle

        self.compute_len()


    def compute_len(self):

        self.valid_file_list = []

        for wavfile in self.file_list:

            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')

            # Silence clipping
            if self.trim:
                x, idx = librosa.effects.trim(x, top_db=30)

            # Check valid wav files
            seq_length = (self.sequence_len - 1) * self.hop
            if len(x) >= seq_length:
                self.valid_file_list.append(wavfile)

        if self.shuffle:
            random.shuffle(self.valid_file_list)


    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return len(self.valid_file_list)


    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        """
        
        # Read wav files
        wavfile = self.valid_file_list[index]
        x, fs_x = sf.read(wavfile)

        # Silence clipping
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
            x = x[ind_beg:ind_end]
        elif self.trim:
            x, _ = librosa.effects.trim(x, top_db=30)

        # Sequence tailor
        file_length = len(x)
        seq_length = (self.sequence_len - 1) * self.hop # sequence length in time domain
        start = np.random.randint(0, file_length - seq_length)
        end = start + seq_length
        x = x[start:end]

        # Normalize sequence
        x = x/np.max(np.abs(x))

        # STFT transformation
        audio_spec = torch.stft(torch.from_numpy(x), n_fft=self.nfft, hop_length=self.hop, 
                                win_length=self.wlen, window=self.win, 
                                center=True, pad_mode='reflect', normalized=False, onesided=True)

        # Square of magnitude
        sample = (audio_spec[:,:,0]**2 + audio_spec[:,:,1]**2).float()

        return sample


        

