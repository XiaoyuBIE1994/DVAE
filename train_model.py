#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import sys
import os
import socket
import datetime
from configparser import ConfigParser

# Re-write configure class, enable to distinguish betwwen upper and lower letters
class myconf(ConfigParser):
    def __init__(self,defaults=None):
        ConfigParser.__init__(self,defaults=None)
    def optionxform(self, optionstr):
        return optionstr


def train_model(config_file):
    # Read config file
    cfg = myconf()
    cfg.read(config_file)
    
    # Get host name and date
    hostname = socket.gethostname()
    print('HOSTNAME: ' + hostname)
    date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")
    print(date)

    # Find data set
    #train_data_dir, val_data_dir = find_dataset()
    train_data_dir = '/mnt/xbie/Data/clean_speech/wsj0_si_tr_s'
    val_data_dir = '/mnt/xbie/Data/clean_speech/wsj0_si_dt_05'

    # Read network parameters
    input_dim = cfg.getint('network', 'input_dim')
    latent_dim = cfg.getint('network','latent_dim')
    hidden_dim_encoder = [int(i) for i in cfg.get('network', 'hidden_dim_encoder').split(',')] # this parameter is a python list
    activation = eval(cfg.get('network', 'activation'))
    
    # Create directory for results and set training device
    local_host = cfg.get('user', 'local_host')
    if hostname == local_host:
        device = 'cpu'
        path_prefix = cfg.get('path', 'path_local')
    else:
        device = 'cuda'
        path_prefix = cfg.get('path', 'path_cluster')
    save_dir = os.path.join(path_prefix, 'saved_model')
    print('Result will be saved in: ' + save_dir)
    if not(os.path.isdir(save_dir)):
        os.makedirs(save_dir)

    # Read STFT parameters
    wlen_sec = cfg.getfloat('STFT', 'wlen_sec')
    hop_percent = cfg.getfloat('STFT', 'hop_percent')
    fs = cfg.getint('STFT', 'fs')

    


if __name__ == '__main__':
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
        train_model(config_file)
    else:
        print("Please indiquate config file")
