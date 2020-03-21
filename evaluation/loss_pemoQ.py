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
# from pystoi.stoi import stoi

def pemoQ(x_ref, x_test, tmp_dir='/tmp', fs=16000, pesq_path='./pesq'):
    speecc_path_ref = os.path.join(tmp_dir, 'x_ref.wav')
    speech_path_test =  os.path.join(tmp_dir, 'x_test.wav')
    sf.write(speecc_path_ref, x_ref, fs, )
    return




def compute_pesq(s_hat, s_orig, TMP_dir='/data/tmp', 
                    fs=16000, pesq_path='./PESQ/Software/source/pesq'):
    
    speech_orig_PCM_16 = os.path.join(TMP_dir, next(tempfile._get_candidate_names()) + '.wav')
#        speech_orig_PCM_16 = tempfile.NamedTemporaryFile().name + '.wav'
    
    sf.write(speech_orig_PCM_16, s_orig, fs, 'PCM_16')

    speech_est_PCM_16 = os.path.join(TMP_dir, next(tempfile._get_candidate_names()) + '.wav')
#        speech_est_PCM_16 = tempfile.NamedTemporaryFile().name + '.wav'
    
    sf.write(speech_est_PCM_16, s_hat, fs, 'PCM_16')
    
    res = subprocess.check_output([pesq_path, speech_orig_PCM_16, 
                                    speech_est_PCM_16, 
                                    "+16000"]) # it can raise an exception
    
    os.remove(speech_orig_PCM_16)
    os.remove(speech_est_PCM_16)
    
    try:
        pesq = float(res[-12:-7])
    except ValueError:
        pesq = np.nan
        
    return pesq

# def compute_stoi(s_hat, s_orig, fs=16000):
    
#     return stoi(s_orig, s_hat, fs, extended=True)

if __name__ == '__main__':
    pesq_path = './pesq'
    speech_orig = '/Users/xiaoyu/WorkStation/Project_rvae/saved_model_Xiaoyu/WSJ0_2020-02-23-05h22_BiEnc_BiDec_RecZ_z_dim=16/test/male_440_origin-0.wav'
    speech_test = '/Users/xiaoyu/WorkStation/Project_rvae/saved_model_Xiaoyu/WSJ0_2020-02-23-05h22_BiEnc_BiDec_RecZ_z_dim=16/test/male_440_recon-0.wav'
    res = subprocess.check_output([pesq_path, speech_orig, speech_test, "+16000"])
    try:
        pesq = float(res[-12:-7])
    except ValueError:
        pesq = np.nan

    print(pesq)
