#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import numpy as np
import librosa

'''
x_ref: (seq_len, )
x_test: (seq_len', )
these two signal may have different length
and a scaling difference
'''

# mse/rmse in time domain
def rmse_td(x_ref, x_test):
    len_x = len(x_test)
    x_ref = x_ref[:len_x]
    x_test_ = np.expand_dims(x_test, axis=1)
    scale = np.linalg.lstsq(x_test_, x_ref, rcond=None)[0][0] # minimize (b-ax)^2
    return np.sqrt(np.square(x_ref - scale*x_test).mean())

# mse/rmse in frequency domain
def rmse_fd(x_ref, x_test, wlen_sec, nfft, hop, wlen, win):
    len_x = len(x_test)
    x_ref = x_ref[:len_x]
    x_test_ = np.expand_dims(x_test, axis=1)
    scale = np.linalg.lstsq(x_test_, x_ref, rcond=None)[0][0]
    x_test_scaled = scale * x_test
    X_ref = librosa.stft(x_ref, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)
    X_test = librosa.stft(x_test_scaled, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)
    E = X_ref - X_test
    return np.sqrt(np.mean(np.square(np.abs(E)))/nfft)

# mse/rmse in time/frequence domain after framing
# rmse in td/fd here will be the same
# but the introduce of a windows function will divide the mse by 2 (rmse by sqrt(2))
# comparing to the mse calculated directly in time sequence
def rmse_frame(x_ref, x_test, nfft, win):
    assert len(win) == nfft
    # align
    htop = int(nfft/2)
    len_x = len(x_test)
    num_frame = len_x // htop
    len_data = num_frame * htop
    x_ref = x_ref[:len_data]
    x_test = x_test[:len_data]
    # re-scale
    x_test_ = np.expand_dims(x_test, axis=1)
    scale = np.linalg.lstsq(x_test_, x_ref, rcond=None)[0][0]
    x_test_scaled = scale * x_test
    # padding
    pad_ref = np.hstack((np.zeros((htop,)), x_ref, np.zeros((htop,))))
    pad_test = np.hstack((np.zeros((htop,)), x_test_scaled, np.zeros((htop,))))
    mat_ref = np.zeros((nfft, num_frame))
    mat_test = np.zeros((nfft, num_frame))
    X_ref = np.zeros((nfft, num_frame), dtype=np.complex128)
    X_test = np.zeros((nfft, num_frame), dtype=np.complex128)
    # framing
    for n in range(num_frame):
        frame_ref = win * pad_ref[n*htop: nfft+n*htop]
        mat_ref[:,n] = frame_ref
        X_ref[:,n] = np.fft.fft(frame_ref)

        frame_test = win * pad_test[n*htop: nfft+n*htop]
        mat_test[:,n] = frame_test
        X_test[:,n] = np.fft.fft(frame_test)
    # calculate rmse
    e_x = mat_ref - mat_test
    e_X = X_ref - X_test
    rmse_td_frame = np.sqrt(np.mean(np.square(e_x)))
    rmse_fd_frame = np.sqrt(np.mean(np.square(np.abs(e_X)))/nfft)
    
    return rmse_td_frame, rmse_fd_frame
    