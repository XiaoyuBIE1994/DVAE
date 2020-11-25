#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt
"""

import sys
from dvae import LearningAlgorithm


if __name__ == '__main__':

    if len(sys.argv) == 2:
        cfg_file = sys.argv[1]
        learning_algo = LearningAlgorithm(config_file=cfg_file)
        learning_algo.train()
    else:
        print('Error: Please indiquate config file')


