#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pystoi.stoi as st
import soundfile as sf


def stoi(x_ref, x_test, fs=16000):
    # len_x = len(x_test)
    # x_ref = x_ref[:len_x]
    len_x = len(x_ref)
    len_x_t = len(x_test)
    x_test_pad = np.zeros(len_x)
    x_test_pad[:len_x_t] = x_test

    print(x_ref.shape)
    print(x_test_pad.shape)
    
    return st.stoi(x_ref, x_test_pad, fs, extended=True)

if __name__ == '__main__':
    # speech_orig = '/Users/xiaoyu/WorkStation/Project_rvae/saved_model_Xiaoyu/WSJ0_2020-02-23-05h22_BiEnc_BiDec_RecZ_z_dim=16/test/male_440_origin-0.wav'
    # speech_test = '/Users/xiaoyu/WorkStation/Project_rvae/saved_model_Xiaoyu/WSJ0_2020-02-23-05h22_BiEnc_BiDec_RecZ_z_dim=16/test/male_440_recon-0.wav'
    speech_orig = 'audio_orig.wav'
    speech_test = 'audio_recon.wav'
    x_orig, _ = sf.read(speech_orig)
    x_test, _ = sf.read(speech_test)
    res = stoi(x_orig, x_test)
    print(res)