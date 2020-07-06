#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""
import os


def perpare_dataset(dataset_name, hostname, save_dir='saved_model'):
    
    if hostname == 'virgo': 
        saved_root = os.path.join('/local_scratch/xbie/Results/2020_DVAE', save_dir)
    elif hostname == 'MacPro-BIE.local':
        saved_root =  os.path.join('/Users/xiaoyu/WorkStation/Project_rvae', save_dir)
    elif 'access' in hostname:
        saved_root = os.path.join('/scratch/virgo/xbie/Results/2020_DVAE', save_dir)
    elif 'gpu' in hostname:
        saved_root = os.path.join('/mnt/xbie/Results/2020_DVAE', save_dir)
    else:
        saved_root = os.path.join('/mnt', save_dir)
    
    if not(os.path.isdir(saved_root)):
        os.makedirs(saved_root)

    
    if dataset_name == 'WSJ0':
        train_data_dir, val_data_dir = find_WSJ0(hostname)
    else:
        train_data_dir = ''
        val_data_dir = ''

    return saved_root, train_data_dir, val_data_dir


def find_WSJ0(hostname):
    if hostname == 'virgo': 
        train_data_dir = '/local_scratch/xbie/Data/clean_speech/wsj0_si_tr_s'
        val_data_dir = '/local_scratch/xbie/Data/clean_speech/wsj0_si_dt_05'
    elif hostname == 'MacPro-BIE.local':
        train_data_dir =  '/Users/xiaoyu/WorkStation/Project_rvae/Data/clean_speech/wsj0_si_dt_05'
        val_data_dir = '/Users/xiaoyu/WorkStation/Project_rvae/Data/clean_speech/wsj0_si_et_05'
    elif 'access' in hostname:
        train_data_dir =  '/scratch/virgo/xbieData/clean_speech/wsj0_si_tr_s'
        val_data_dir = '/scratch/virgo/xbieData/clean_speech/wsj0_si_dt_05'
    elif 'gpu' in hostname:
        train_data_dir = '/mnt/xbie/Data/clean_speech/wsj0_si_tr_s'
        val_data_dir = '/mnt/xbie/Data/clean_speech/wsj0_si_dt_05'
    else:
        train_data_dir =  ''
        val_data_dir =  ''
    return train_data_dir, val_data_dir
        