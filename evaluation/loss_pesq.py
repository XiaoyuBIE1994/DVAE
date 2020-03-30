#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""
import numpy as np
import subprocess
import sys
import soundfile as sf

def pesq(x_ref, x_test, TMP_dir='/data/tmp', 
              fs=16000, pesq_path='./pesq'):
    
    speecc_path_ref = os.path.join(tmp_dir, 'x_ref.wav')
    speech_path_test =  os.path.join(tmp_dir, 'x_test.wav')
    sf.write(speecc_path_ref, x_ref, fs, 'PCM_16')
    sf.write(speech_path_test, x_test, fs, 'PCM_16')
    res = subprocess.check_output([pesq_path, speech_ref, 
                                   speech_test, "+16000"]) # it can raise an exception
    os.remove(speecc_path_ref)
    os.remove(speech_path_test)
    try:
        pesq = float(res[-12:-7])
    except ValueError:
        pesq = np.nan
        
    return pesq

if __name__ == '__main__':
    pesq_path = './evaluation/pesq'
    speech_orig = '/Users/xiaoyu/WorkStation/Project_rvae/saved_model_Xiaoyu/WSJ0_2020-02-23-05h22_BiEnc_BiDec_RecZ_z_dim=16/test/male_440_origin-0.wav'
    speech_test = '/Users/xiaoyu/WorkStation/Project_rvae/saved_model_Xiaoyu/WSJ0_2020-02-23-05h22_BiEnc_BiDec_RecZ_z_dim=16/test/male_440_recon-0.wav'
    res = subprocess.check_output([pesq_path, speech_orig, speech_test, "+16000"])
    try:
        pesq = float(res[-12:-7])
    except ValueError:
        pesq = np.nan

    print(pesq)