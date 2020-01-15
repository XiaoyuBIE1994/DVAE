#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 by Inria
Authored by Simon Leglaive (simon.leglaive@inria.fr)
License agreement in LICENSE.txt
"""

import os
import shutil
import tarfile

def prepare_dataset_train_only(dataset_name, hostname):
    
    hostname_test =  (hostname=='gpu5-perception' or 
                      hostname=='gpu6-perception' or 
                      hostname=='gpu1-perception')
    
    if 'TIMIT' in dataset_name:
    
        dataset_root = '/local_scratch/sileglai/datasets/clean_speech/TIMIT'
        if not(os.path.isdir(dataset_root)):
            if hostname_test:
                directory = '/local_scratch/data/perception/sileglai/datasets/clean_speech/'
            else:
                directory = '/local_scratch/sileglai/datasets/clean_speech/'
            if not(os.path.isdir(directory)):
                os.makedirs(directory)
            tar_file = os.path.join(directory, 'TIMIT.tar.gz')
            if not(os.path.isfile(tar_file)):
                print('Copy dataset on local scratch...')
                shutil.copyfile('/services/scratch/perception/sileglai/datasets/clean_speech/TIMIT.tar.gz', tar_file)
            if not(os.path.isdir(dataset_root)):
                tar = tarfile.open(tar_file, "r:gz")
                tar.extractall(path=directory)
                tar.close()
            print('...done')
            
        if hostname_test:
            data_dir = '/local_scratch/data/perception/sileglai/datasets/clean_speech/TIMIT/TRAIN'
        else:
            data_dir = '/local_scratch/sileglai/datasets/clean_speech/TIMIT/TRAIN'

        
    
    elif 'WSJ0' in dataset_name:
    
        dataset_root = '/local_scratch/sileglai/datasets/clean_speech/wsj0_si_tr_s'
        if not(os.path.isdir(dataset_root)):
            if hostname_test:
                directory = '/local_scratch/data/perception/sileglai/datasets/clean_speech/'
            else:
                directory = '/local_scratch/sileglai/datasets/clean_speech/'
            if not(os.path.isdir(directory)):
                os.makedirs(directory)
            tar_file = os.path.join(directory, 'wsj0_si_tr_s.tar.gz')
            if not(os.path.isfile(tar_file)):
                print('Copy dataset on local scratch...')
                shutil.copyfile('/scratch/octans/sileglai/datasets/clean_speech/wsj0_si_tr_s.tar.gz', tar_file)
            if not(os.path.isdir(dataset_root)):
                tar = tarfile.open(tar_file, "r:gz")
                tar.extractall(path=directory)
                tar.close()
            print('...done')
        
        if hostname_test:
            data_dir = '/local_scratch/data/perception/sileglai/datasets/clean_speech/wsj0_si_tr_s'
        else:
            data_dir = '/local_scratch/sileglai/datasets/clean_speech/wsj0_si_tr_s'
    
    
    elif 'chime3' in dataset_name:
    
        dataset_root = '/local_scratch/sileglai/datasets/clean_speech/chime3_tr05_org'
        if not(os.path.isdir(dataset_root)):
            if hostname_test:
                directory = '/local_scratch/data/perception/sileglai/datasets/clean_speech/'
            else:
                directory = '/local_scratch/sileglai/datasets/clean_speech/'
            if not(os.path.isdir(directory)):
                os.makedirs(directory)
            tar_file = os.path.join(directory, 'chime3_tr05_org.tar.gz')
            if not(os.path.isfile(tar_file)):
                print('Copy dataset on local scratch...')
                shutil.copyfile('/scratch/octans/sileglai/datasets/clean_speech/chime3_tr05_org.tar.gz', tar_file)
            if not(os.path.isdir(dataset_root)):
                tar = tarfile.open(tar_file, "r:gz")
                tar.extractall(path=directory)
                tar.close()
            print('...done')
            
            if hostname_test:
                data_dir = '/local_scratch/data/perception/sileglai/datasets/clean_speech/chime3_tr05_org'
            else:
                data_dir = '/local_scratch/sileglai/datasets/clean_speech/chime3_tr05_org'
            
    else:
        
        raise NameError('Unknown dataset')
        
    return data_dir

def prepare_dataset_WSJ0(dataset_name, hostname):
    
    hostname_test =  (hostname=='gpu5-perception' or 
                      hostname=='gpu6-perception' or 
                      hostname=='gpu1-perception')
    

    train_dataset_root = '/local_scratch/sileglai/datasets/clean_speech/wsj0_si_tr_s'
    if not(os.path.isdir(train_dataset_root)):
        if hostname_test:
            directory = '/local_scratch/data/perception/sileglai/datasets/clean_speech/'
        else:
            directory = '/local_scratch/sileglai/datasets/clean_speech/'
        if not(os.path.isdir(directory)):
            os.makedirs(directory)
        tar_file = os.path.join(directory, 'wsj0_si_tr_s.tar.gz')
        if not(os.path.isfile(tar_file)):
            print('Copy dataset on local scratch...')
            shutil.copyfile('/scratch/octans/sileglai/datasets/clean_speech/wsj0_si_tr_s.tar.gz', tar_file)
        if not(os.path.isdir(train_dataset_root)):
            tar = tarfile.open(tar_file, "r:gz")
            tar.extractall(path=directory)
            tar.close()
        print('...done')
    
    val_dataset_root = '/local_scratch/sileglai/datasets/clean_speech/wsj0_si_dt_05'
    if not(os.path.isdir(val_dataset_root)):
        if hostname_test:
            directory = '/local_scratch/data/perception/sileglai/datasets/clean_speech/'
        else:
            directory = '/local_scratch/sileglai/datasets/clean_speech/'
        if not(os.path.isdir(directory)):
            os.makedirs(directory)
        tar_file = os.path.join(directory, 'wsj0_si_dt_05.tar.gz')
        if not(os.path.isfile(tar_file)):
            print('Copy dataset on local scratch...')
            shutil.copyfile('/scratch/octans/sileglai/datasets/clean_speech/wsj0_si_dt_05.tar.gz', tar_file)
        if not(os.path.isdir(val_dataset_root)):
            tar = tarfile.open(tar_file, "r:gz")
            tar.extractall(path=directory)
            tar.close()
        print('...done')
    
    
    if hostname_test:
        train_data_dir = '/local_scratch/data/perception/sileglai/datasets/clean_speech/wsj0_si_tr_s'
        val_data_dir = '/local_scratch/data/perception/sileglai/datasets/clean_speech/wsj0_si_dt_05'
    else:
        train_data_dir = '/local_scratch/sileglai/datasets/clean_speech/wsj0_si_tr_s'
        val_data_dir = '/local_scratch/sileglai/datasets/clean_speech/wsj0_si_dt_05'

    return train_data_dir, val_data_dir