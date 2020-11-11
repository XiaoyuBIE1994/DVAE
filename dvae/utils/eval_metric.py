#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt
"""

import numpy as np
import soundfile as sf
import soundfile as sf
from pypesq import pesq
from pystoi import stoi

def compute_median(data):
    median = np.median(data, axis=0)
    q75, q25 = np.quantile(data, [.75 ,.25], axis=0)    
    iqr = q75 - q25
    CI = 1.57*iqr/np.sqrt(data.shape[0])
    if np.any(np.isnan(data)):
        raise NameError('nan in data')
    return median, CI


def compute_rmse():

    def get_result(file_est, file_ref):
        x_est, _ = sf.read(file_est)
        x_ref, _ = sf.read(file_ref)
        # align
        len_x = len(x_est)
        x_ref = x_ref[:len_x]
        # scaling
        alpha = np.sum(x_est*x_ref) / np.sum(x_est**2)
        # x_est_ = np.expand_dims(x_est, axis=1)
        # alpha = np.linalg.lstsq(x_est_, x_ref, rcond=None)[0][0]
        x_est_scaled = alpha * x_est
        return np.sqrt(np.square(x_est_scaled - x_ref).mean())

    return get_result


class EvalMetrics():

    def __init__(self, metric='all'):

        self.metric = metric

    def eval(self, audio_est, audio_ref):

        x_est, fs_est = sf.read(audio_est)
        x_ref, fs_ref = sf.read(audio_ref)
        # align
        len_x = np.min([len(x_est), len(x_ref)])
        x_est = x_est[:len_x]
        x_ref = x_ref[:len_x]

        # x_ref = x_ref / np.max(np.abs(x_ref))

        if fs_est != fs_ref:
            raise ValueError('Sampling rate is different amon estimated audio and reference audio')

        if self.metric  == 'rmse':
            return compute_rmse(x_est, x_ref)
        elif self.metric == 'pesq':
            return pesq(x_ref, x_est, fs_est)
        elif self.metric == 'stoi':
            return stoi(x_ref, x_est, fs_est, extended=False)
        elif self.metric == 'estoi':
            return stoi(x_ref, x_est, fs_est, extended=True)
        elif self.metric == 'all':
            score_rmse = compute_rmse(x_est, x_ref)
            score_pesq = pesq(x_ref, x_est, fs_est)
            score_stoi = stoi(x_ref, x_est, fs_est, extended=False)
            return score_rmse, score_pesq, score_stoi
        else:
            raise ValueError('Evaluation only support: rmse, pesq, (e)stoi, all')
