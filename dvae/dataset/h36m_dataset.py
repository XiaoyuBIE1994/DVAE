#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

32 human3.6 joint name:
joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Site", "LeftUpLeg", "LeftLeg","LeftFoot",
            "LeftToeBase", "Site", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm","LeftForeArm",
            "LeftHand", "LeftHandThumb", "Site", "L_Wrist_End", "Site", "RightShoulder", "RightArm","RightForeArm",
            "RightHand", "RightHandThumb", "Site", "R_Wrist_End", "Site"]
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from .utils import data_utils

def build_dataloader(cfg):

    # Load dataset params for H3.6M
    data_dir = cfg.get('User', 'data_dir')
    shuffle = cfg.getboolean('DataFrame', 'shuffle')
    batch_size = cfg.getint('DataFrame', 'batch_size')
    num_workers = cfg.getint('DataFrame', 'num_workers')
    sequence_len = cfg.getint('DataFrame', 'sequence_len')
    actions = None if cfg.get('DataFrame', 'actions') == '' else cfg.get('DataFrame', 'actions')
    sample_rate = cfg.getint('DataFrame', 'sample_rate')
    skip_rate = cfg.getint('DataFrame', 'skip_rate')
    val_indices = cfg.getint('DataFrame', 'val_indices')
    h36m13kpts = cfg.getboolean('DataFrame', 'h36m13kpts')
    use_3D = cfg.getboolean('DataFrame', 'use_3D')

    # Training dataset
    split = 0 # train
    train_dataset = HumanPoseXYZ(path_to_data=data_dir, seq_len=sequence_len,
                            split=split, actions=actions, sample_rate=sample_rate,
                            skip_rate=skip_rate, val_indices=val_indices, 
                            h36m13kpts=h36m13kpts, use_3D=use_3D)
    split = 2 # validation
    val_dataset = HumanPoseXYZ(path_to_data=data_dir, seq_len=sequence_len,
                            split=split, actions=actions, sample_rate=sample_rate,
                            skip_rate=skip_rate, val_indices=val_indices, 
                            h36m13kpts=h36m13kpts, use_3D=use_3D)
    train_num = train_dataset.__len__()
    val_num = val_dataset.__len__()
    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                shuffle=shuffle, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                shuffle=shuffle, num_workers=num_workers)

    return train_dataloader, val_dataloader, train_num, val_num


class HumanPoseXYZ(Dataset):

    def __init__(self, path_to_data="./datasets/h3.6m/", seq_len=50, split=0, actions=None, sample_rate=2, skip_rate=2, val_indices=64, h36m13kpts=False, use_3D=True):
        """
        :param path_to_data:
        :param seq_len: sequence length for data
        :param split: 0 train, 1 test, 2 validation
        :param actions: if use specifical actions
        :param sample_rate: downsampling rate
        :param skip_rate: the skip length to get example, only used for train and test
        :param val_indices: the number of slices used for validation
        :param h36m13kpts: if use 13 key points, common joints as ExPI
        """
        
        self.path_to_data = path_to_data
        self.seq_len = seq_len
        self.split = split
        self.actions = actions
        self.sample_rate = sample_rate
        self.skip_rate = skip_rate
        self.val_indices = val_indices
        self.h36m13kpts = h36m13kpts
        
        self.seq = {}
        self.data_idx = []

        # self.dimensions_to_use = np.array(
        #     [6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42,
        #      43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85, 86])
        # self.dimensions_to_ignore = np.array(
        #     [[0, 1, 2, 3, 4, 5, 10, 11, 16, 17, 18, 19, 20, 25, 26, 31, 32, 33, 34, 35, 48, 49, 50, 58,
        #       59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 82, 83, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
        #       98]])
        # ignore constant joints and joints at same position with other joints
        if self.h36m13kpts:
            joint_to_ignore = np.array([16,20,23,24,28,31,0,11,12,14,15,21,22,29,30,4,5,9,10])
            self.pi2h36m13 = [6,10,7,11,8,12,9,3,0,4,1,5,2] #expi_masked_0 = h36m_masked_6, change h36m to pi order for pretrain
            dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
            self.dimension_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)
        else:
            # joint_to_ignore = np.array([0,1,6,11,16,20,23,24,28,31])
            self.dimension_use = np.arange(96)
            
        self.in_features = len(self.dimension_use)
        subs = [[1, 6, 7, 8, 9], [11], [5]]
        if self.actions is None:
            acts = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]
        else:
            acts = [self.actions]

        subs = subs[split]
        for subj in subs:
            for action in acts:
                for subact in [1, 2]:
                    # read motion data
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, subact)
                    the_sequence = data_utils.readCSVasFloat(filename)
                    
                    # save valid sequences, downsampling if needed
                    n, _ = the_sequence.shape
                    even_list = range(0, n, self.sample_rate)
                    the_sequence = np.array(the_sequence[even_list, :])
                    the_sequence[:, 0:6] = 0 # remove global rotation and translation
                    if use_3D:
                        the_sequence = torch.from_numpy(the_sequence).float().cuda()
                        seq_xyz = data_utils.expmap2xyz_torch(the_sequence).cpu().detach().numpy().reshape(-1, 96) # angle to XYZ, (N, 99) -> (N, 32, 3) -> (N, 96)
                        self.seq[(subj, action, subact)] = seq_xyz[:, self.dimension_use] # (N, 96, 66 or 39)
                    else:
                        self.seq[(subj, action, subact)] = the_sequence[:,6:] # (N, 93)
                        
                    # save valid start frames, based on skip_rate
                    num_frames = len(even_list)
                    if self.split <= 1: # for train and test
                        valid_frames = np.arange(0, num_frames - self.seq_len + 1, self.skip_rate)
                    else: # for validation
                        valid_frames = data_utils.find_indices(num_frames, self.seq_len, self.val_indices)

                    tmp_data_idx_1 = [(subj, action, subact)] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.seq_len)
        return self.seq[key][fs]