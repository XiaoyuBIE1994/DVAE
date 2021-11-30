#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt
"""

import os
import random
import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils import data

def build_dataloader(cfg):

    # Load dataset params for WSJ0 subset
    train_data_dir = cfg.get('User', 'train_data_dir')
    val_data_dir = cfg.get('User', 'val_data_dir')
    dataset_name = cfg.get('DataFrame', 'dataset_name')
    batch_size = cfg.getint('DataFrame', 'batch_size')
    shuffle = cfg.getboolean('DataFrame', 'shuffle')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    sequence_len = cfg.getint('DataFrame', 'sequence_len')
    data_suffix = cfg.get('DataFrame', 'suffix')
    use_random_seq = cfg.getboolean('DataFrame', 'use_random_seq')

    # Load STFT parameters
    wlen_sec = cfg.getfloat('STFT', 'wlen_sec')
    hop_percent = cfg.getfloat('STFT', 'hop_percent')
    fs = cfg.getint('STFT', 'fs')
    zp_percent = cfg.getint('STFT', 'zp_percent')
    wlen = wlen_sec * fs
    wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # pwoer of 2
    hop = np.int(hop_percent * wlen)
    nfft = wlen + zp_percent * wlen
    win = torch.sin(torch.arange(0.5, wlen+0.5) / wlen * np.pi)
    trim = cfg.getboolean('STFT', 'trim')

    STFT_dict = {}
    STFT_dict['fs'] = fs
    STFT_dict['wlen'] = wlen
    STFT_dict['hop'] = hop
    STFT_dict['nfft'] = nfft
    STFT_dict['win'] = win
    STFT_dict['trim'] = trim

    # List all available speech audio
    train_file_list = librosa.util.find_files(train_data_dir, ext=data_suffix)
    val_file_list = librosa.util.find_files(val_data_dir, ext=data_suffix)

    # Training dataset
    if use_random_seq:
        train_dataset = SpeechSequencesRandom(file_list=train_file_list, sequence_len=sequence_len,
                                        STFT_dict=STFT_dict, shuffle=shuffle, name=dataset_name)
        val_dataset = SpeechSequencesRandom(file_list=val_file_list, sequence_len=sequence_len,
                                        STFT_dict=STFT_dict, shuffle=shuffle, name=dataset_name)
    else:
        train_dataset = SpeechSequencesFull(file_list=train_file_list, sequence_len=sequence_len,
                                        STFT_dict=STFT_dict, shuffle=shuffle, name=dataset_name)
        val_dataset = SpeechSequencesFull(file_list=val_file_list, sequence_len=sequence_len,
                                        STFT_dict=STFT_dict, shuffle=shuffle, name=dataset_name)
    train_num = train_dataset.__len__()
    val_num = val_dataset.__len__()

    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                shuffle=shuffle, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                shuffle=shuffle, num_workers=num_workers)

    return train_dataloader, val_dataloader, train_num, val_num


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
            else:
                ind_beg = 0
                ind_end = len(x)

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
        else:
                ind_beg = 0
                ind_end = len(x)

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


        

