#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""


import sys
from learning_algo import LearningAlgorithm

# cfg_file = '/local_scratch/xbie/Results/2020_DVAE/saved_model_DVAE/WSJ0_2020-08-12-21h00_SRNN_z_dim=16_F/config.ini'
# state_file = '/local_scratch/xbie/Results/2020_DVAE/saved_model_DVAE/WSJ0_2020-08-12-21h00_SRNN_z_dim=16_F/SRNN_final_epoch497.pt'
# 
# learning_algo = LearningAlgorithm(config_file=cfg_file)

# learning_algo.train()

# audio_1 = '/local_scratch/xbie/Data/clean_speech/wsj0_si_et_05/440/440c020a.wav'
# audio_1_recon = '/local_scratch/xbie/Data/recon-440c020a.wav'
# learning_algo.generate(audio_orig=audio_1, audio_recon=audio_1_recon, state_dict_file=state_file)

# score_rmse, score_pesq, score_stoi = learning_algo.eval(audio_ref=audio_1, audio_est=audio_1_recon, metric='all', state_dict_file=state_file)
# score_rmse = learning_algo.eval(audio_ref=audio_1, audio_est=audio_1_recon, metric='rmse', state_dict_file=state_file)


# print(score_rmse)
# print(score_pesq)
# print(score_stoi)


if __name__ == '__main__':

    if len(sys.argv) == 2:
        cfg_file = sys.argv[1]
        learning_algo = LearningAlgorithm(config_file=cfg_file)
        learning_algo.train()
    else:
        print('Error: Please indiquate config file')

    # cfg_file = '/mnt/xbie/Code/dvae-speech/config/cfg_kvae.ini'
    # learning_algo = LearningAlgorithm(config_file=cfg_file)
    # learning_algo.train()

