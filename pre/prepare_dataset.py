#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""
import os


def perpare_dataset(dataset_name, hostname):
    
    if hostname == 'virgo': 
        saved_root = '/local_scratch/xbie/Code/saved_model'
    elif hostname == 'MacPro-BIE.local':
        saved_root =  '/Users/xiaoyu/WorkStation/Project_rvae/saved_model'
    elif 'access' in hostname:
        saved_root = '/scratch/virgo/xbie/Code/saved_model'
    elif 'gpu' in hostname:
        saved_root = '/mnt/xbie/Code/saved_model'
    else:
        saved_root = '/mnt/saved_model'
    
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
        