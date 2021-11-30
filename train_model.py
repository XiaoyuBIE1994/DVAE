#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""


import sys
import argparse
from dvae.learning_algo import LearningAlgorithm
from dvae.learning_algo_ss import LearningAlgorithm_ss

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # Basic config file
        self.parser.add_argument('--cfg', type=str, default=None, help='config path')
        # Schedule sampling
        self.parser.add_argument('--ss', action='store_true', help='schedule sampling')
        self.parser.add_argument('--use_pretrain', action='store_true', help='if use pretrain')
        self.parser.add_argument('--pretrain_dict', type=str, default=None, help='pretrained model dict')
        # Resume training
        self.parser.add_argument('--reload', action='store_true', help='resume the training')
        self.parser.add_argument('--model_dir', type=str, default=None, help='model directory')

    def get_params(self):
        self._initial()
        self.opt = self.parser.parse_args()
        params = vars(self.opt)
        return params

if __name__ == '__main__':

    params = Options().get_params()
    if not params['ss']:
        learning_algo = LearningAlgorithm(params=params)
        learning_algo.train()
    else:
        learning_algo = LearningAlgorithm_ss(params=params)
        learning_algo.train()


