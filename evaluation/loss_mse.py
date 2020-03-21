#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""
import numpy as np
import librosa

# mean square error in time domain
# x_ref: (seq_len, )
# x_test: (seq_len, )
# these two signal may have different length
# and a scaling difference
def mse_td(x_ref, x_test):
    len_x = len(x_test)
    x_ref = x_ref[:len_x]
    x_test_ = np.expand_dims(x_test, axis=1)
    scale = np.linalg.lstsq(x_test_, x_ref, rcond=None)[0][0] # minimize (b-ax)^2
    return np.square(x_ref - scale*x_test).mean()

def rmse_td(x_ref, x_test):
    return np.sqrt(mse_td(x_ref, x_test))

# mean square error in frequence domain (magnitude from stft)
# X_ref: (x_dim, idx_frame)
# X_rest: (x_dim, idx_frame)
def rmse_fd_mag(X_ref, X_test):
    X_ref = np.abs(X_ref.mean(axis=1))
    X_test = np.abs(X_test.mean(axis=1))
    return np.sqrt(np.square(X_ref - X_test).mean())


def rmse_fd_frame(x_ref, x_test, nfft, hop, wlen, win):
    len_x = len(x_test)
    x_ref = x_ref[:len_x]
    x_test_ = np.expand_dims(x_test, axis=1)
    scale = np.linalg.lstsq(x_test_, x_ref, rcond=None)[0][0]
    x_test_scaled = scale * x_test
    X_ref = librosa.stft(x_ref, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)
    X_test = librosa.stft(x_test_scaled, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)
    E = X_ref - X_test
    return np.sqrt(np.sum(np.square(np.abs(E)))/ (nfft ** 2))

