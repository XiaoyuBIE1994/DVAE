#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dvae.learning_algo import LearningAlgorithm
from dvae.learning_algo_ss import LearningAlgorithm_ss
from dvae.dataset.h36m_dataset import HumanPoseXYZ
from dvae.utils import loss_MPJPE

torch.manual_seed(0)
np.random.seed(0)

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # Basic config file
        self.parser.add_argument('--ss', action='store_true', help='schedule sampling')
        self.parser.add_argument('--cfg', type=str, default=None, help='config path')
        self.parser.add_argument('--saved_dict', type=str, default=None, help='trained model dict')
        # Dataset
        self.parser.add_argument('--test_dir', type=str, default='./data/h3.6m/dataset', help='test dataset')
    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        params = vars(self.opt)
        return params

params = Options().get_params()

if params['ss']:
    learning_algo = LearningAlgorithm_ss(params=params)
    learning_algo.build_model()
    dvae = learning_algo.model
    dvae.out_mean = True
else:
    learning_algo = LearningAlgorithm(params=params)
    learning_algo.build_model()
    dvae = learning_algo.model
dvae.load_state_dict(torch.load(params['saved_dict'], map_location='cpu'))
eval_metrics = EvalMetrics(metric='all')
dvae.eval()
cfg = learning_algo.cfg
print('Total params: %.2fM' % (sum(p.numel() for p in dvae.parameters()) / 1000000.0))


test_dataset = HumanPoseXYZ(path_to_data=params['test_dir'], 
                            seq_len=50, split=1, actions=None, sample_rate=2,
                            skip_rate=2, val_indices=64, h36m13kpts=None)
test_num = test_dataset.__len__()
print('Test samples: {}'.format(test_num))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                            shuffle=True, num_workers=8)

MPJPE = 0
for _, batch_data in tqdm(enumerate(test_dataloader)):

    batch_data = batch_data.to('cuda')
    batch_data = batch_data.permute(1, 0, 2) / 1000 # normalize to meters
    recon_batch_data = dvae(batch_data)
    loss_recon = loss_MPJPE(batch_data*1000, recon_batch_data*1000) # sum over seq_Len and batch size

    seq_len = batch_data.shape[0]
    MPJPE += loss_recon.item() / seq_len

MPJPE = MPJPE / test_num
print('MPJPE: {:.2f}'.format(MPJPE))