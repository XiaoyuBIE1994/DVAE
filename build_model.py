#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

from model import VAE, DKF, KVAE, STORN, VRNN, SRNN, RVAE, DSAE


def build_VAE(cfg, device='cpu'):

    ### Load parameters
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Inference and generation
    dense_x_z = [int(i) for i in cfg.get('Network', 'dense_x_z').split(',')]

    # Build model
    model = VAE(x_dim=x_dim, z_dim=z_dim,
                dense_x_z=dense_x_z, activation=activation,
                dropout_p=dropout_p, device=device).to(device)

    return model


def build_DKF(cfg, device='cpu'):

    ### Load parameters
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Inference
    dense_x_gx = [int(i) for i in cfg.get('Network', 'dense_x_gx').split(',')]
    dim_RNN_gx = cfg.getint('Network', 'dim_RNN_gx')
    num_RNN_gx = cfg.getint('Network', 'num_RNN_gx')
    bidir_gx = cfg.getboolean('Network', 'bidir_gx')
    dense_ztm1_g = [int(i) for i in cfg.get('Network', 'dense_ztm1_g').split(',')]
    dense_g_z = [int(i) for i in cfg.get('Network', 'dense_g_z').split(',')]
    # Generation
    dense_z_x = [int(i) for i in cfg.get('Network', 'dense_z_x').split(',')]

    # Build model
    model = DKF(x_dim=x_dim, z_dim=z_dim, activation=activation,
                dense_x_gx=dense_x_gx, dim_RNN_gx=dim_RNN_gx, 
                num_RNN_gx=num_RNN_gx, bidir_gx=bidir_gx,
                dense_ztm1_g=dense_ztm1_g, dense_g_z=dense_g_z,
                dense_z_x=dense_z_x,
                dropout_p=dropout_p, device=device).to(device)

    return model


def build_KVAE(cfg, device='cpu'):

    ### Load special parameters for KVAE
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    a_dim = cfg.getint('Network', 'a_dim')
    z_dim = cfg.getint('Network', 'z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    scale_reconstruction = cfg.getfloat('Network', 'scale_reconstruction')
    # VAE
    dense_x_a = [int(i) for i in cfg.get('Network', 'dense_x_a').split(',')]
    dense_a_x = [int(i) for i in cfg.get('Network', 'dense_a_x').split(',')]
    # LGSSM
    init_kf_mat = cfg.getfloat('Network', 'init_kf_mat')
    noise_transition = cfg.getfloat('Network', 'noise_transition')
    noise_emission = cfg.getfloat('Network', 'noise_emission')
    init_cov = cfg.getfloat('Network', 'init_cov')
    # Dynamics
    K = cfg.getint('Network', 'K')
    dim_RNN_alpha = cfg.getint('Network', 'dim_RNN_alpha')
    num_RNN_alpha = cfg.getint('Network', 'num_RNN_alpha')
    # Training set
    scheduler_training = cfg.getboolean('Training', 'scheduler_training')
    only_vae_epochs = cfg.getint('Training', 'only_vae_epochs')
    kf_update_epochs = cfg.getint('Training', 'kf_update_epochs')
    
    # Build model
    model = KVAE(x_dim=x_dim, a_dim=a_dim, z_dim=z_dim, activation=activation,
                 dense_x_a=dense_x_a, dense_a_x=dense_a_x,
                 init_kf_mat=init_kf_mat, noise_transition=noise_transition,
                 noise_emission=noise_emission, init_cov=init_cov,
                 K=K, dim_RNN_alpha=dim_RNN_alpha, num_RNN_alpha=num_RNN_alpha,
                 dropout_p=dropout_p, scale_reconstruction = scale_reconstruction,
                 device=device).to(device)

    return model


def build_STORN(cfg, device='cpu'):

    ### Load parameters for STORN
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Inference
    dense_x_g = [int(i) for i in cfg.get('Network', 'dense_x_g').split(',')]
    dim_RNN_g = cfg.getint('Network', 'dim_RNN_g')
    num_RNN_g = cfg.getint('Network', 'num_RNN_g')
    dense_g_z = [int(i) for i in cfg.get('Network', 'dense_g_z').split(',')]
    # Generation
    dense_z_h = [int(i) for i in cfg.get('Network', 'dense_z_h').split(',')]
    dense_xtm1_h = [int(i) for i in cfg.get('Network', 'dense_xtm1_h').split(',')]
    dim_RNN_h = cfg.getint('Network', 'dim_RNN_h')
    num_RNN_h = cfg.getint('Network', 'num_RNN_h')
    dense_h_x = [int(i) for i in cfg.get('Network', 'dense_h_x').split(',')]
    
    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')

    # Build model
    model = STORN(x_dim=x_dim, z_dim=z_dim, activation=activation,
                 dense_x_g=dense_x_g, dense_g_z=dense_g_z,
                 dim_RNN_g=dim_RNN_g, num_RNN_g=num_RNN_g,
                 dense_z_h=dense_z_h, dense_xtm1_h=dense_xtm1_h,
                 dense_h_x=dense_h_x,
                 dim_RNN_h=dim_RNN_h, num_RNN_h=num_RNN_h,
                 dropout_p=dropout_p, beta=beta, device=device).to(device)


def build_VRNN(cfg, device='cpu'):

    ### Load parameters for VRNN
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Feature extractor
    dense_x = [int(i) for i in cfg.get('Network', 'dense_x').split(',')]
    dense_z = [int(i) for i in cfg.get('Network', 'dense_z').split(',')]
    # Dense layers
    dense_hx_z = [int(i) for i in cfg.get('Network', 'dense_hx_z').split(',')]
    dense_hz_x = [int(i) for i in cfg.get('Network', 'dense_hz_x').split(',')]
    dense_h_z = [int(i) for i in cfg.get('Network', 'dense_h_z').split(',')]
    # RNN
    dim_RNN = cfg.getint('Network', 'dim_RNN')
    num_RNN = cfg.getint('Network', 'num_RNN')

    # Build model
    model = VRNN(x_dim=x_dim, z_dim=z_dim, activation=activation,
                 dense_x=dense_x, dense_z=dense_z,
                 dense_hx_z=dense_hx_z, dense_hz_x=dense_hz_x, 
                 dense_h_z=dense_h_z,
                 dim_RNN=dim_RNN, num_RNN=num_RNN,
                 dropout_p= dropout_p, device=device).to(device)

    return model


def build_SRNN(cfg, device='cpu'):

    ### Load parameters for SRNN
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Deterministic
    dense_x_h = [int(i) for i in cfg.get('Network', 'dense_x_h').split(',')]
    dim_RNN_h = cfg.getint('Network', 'dim_RNN_h')
    num_RNN_h = cfg.getint('Network', 'num_RNN_h')
    # Inference
    dense_hx_g = [int(i) for i in cfg.get('Network', 'dense_hx_g').split(',')]
    dim_RNN_g = cfg.getint('Network', 'dim_RNN_g')
    num_RNN_g = cfg.getint('Network', 'num_RNN_g')
    dense_gz_z = [int(i) for i in cfg.get('Network', 'dense_gz_z').split(',')]
    # Prior
    dense_hz_z = [int(i) for i in cfg.get('Network', 'dense_hz_z').split(',')]
    # Generation
    dense_hz_x = [int(i) for i in cfg.get('Network', 'dense_hz_x').split(',')]

    # Build model
    model = SRNN(x_dim=x_dim, z_dim=z_dim, activation=activation,
                 dense_x_h=dense_x_h,
                 dim_RNN_h=dim_RNN_h, num_RNN_h=num_RNN_h,
                 dense_hx_g=dense_hx_g,
                 dim_RNN_g=dim_RNN_g, num_RNN_g=num_RNN_g,
                 dense_gz_z=dense_gz_z,
                 dense_hz_x=dense_hz_x,
                 dense_hz_z=dense_hz_z,
                 dropout_p=dropout_p, device=device).to(device)

    return model


def build_RVAE(cfg, device='cpu'):

    ### Load special paramters for RVAE
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Inference
    dense_x_gx = [int(i) for i in cfg.get('Network', 'dense_x_gx').split(',')]
    dim_RNN_g_x = cfg.getint('Network', 'dim_RNN_g_x')
    num_RNN_g_x = cfg.getint('Network', 'num_RNN_g_x')
    bidir_g_x = cfg.getboolean('Network', 'bidir_g_x')
    dense_z_gz = [int(i) for i in cfg.get('Network', 'dense_z_gz').split(',')]
    dim_RNN_g_z = cfg.getint('Network', 'dim_RNN_g_z')
    num_RNN_g_z = cfg.getint('Network', 'num_RNN_g_z')
    dense_g_z = [int(i) for i in cfg.get('Network', 'dense_g_z').split(',')]
    # Generation
    dense_z_h = [int(i) for i in cfg.get('Network', 'dense_z_h').split(',')]
    dim_RNN_h = cfg.getint('Network', 'dim_RNN_h')
    num_RNN_h = cfg.getint('Network', 'num_RNN_h')
    bidir_h = cfg.getboolean('Network', 'bidir_h')
    dense_h_x = [int(i) for i in cfg.get('Network', 'dense_h_x').split(',')]

    # Build model
    model = RVAE(x_dim=x_dim, z_dim=z_dim, activation=activation,
                 dense_x_gx=dense_x_gx,
                 dim_RNN_g_x=dim_RNN_g_x, num_RNN_g_x=num_RNN_g_x,
                 bidir_g_x=bidir_g_x, 
                 dense_z_gz=dense_z_gz,
                 dim_RNN_g_z=dim_RNN_g_z, num_RNN_g_z=num_RNN_g_z,
                 dense_g_z=dense_g_z,
                 dense_z_h=dense_z_h,
                 dim_RNN_h=dim_RNN_h, num_RNN_h=num_RNN_h,
                 bidir_h=bidir_h,
                 dense_h_x=dense_h_x,
                 dropout_p=dropout_p, device=device).to(device)

    return model


def build_DSAE(cfg, device='cpu'):

    ### Load special parameters for DSAE
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    v_dim = cfg.getint('Network','v_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Inference
    dense_x = [int(i) for i in cfg.get('Network', 'dense_x').split(',')]
    dim_RNN_gv = cfg.getint('Network', 'dim_RNN_gv')
    num_RNN_gv = cfg.getint("Network", 'num_RNN_gv')
    dense_gv_v = [int(i) for i in cfg.get('Network', 'dense_gv_v').split(',')]
    dense_xv_gxv = [int(i) for i in cfg.get('Network', 'dense_xv_gxv').split(',')]
    dim_RNN_gxv = cfg.getint('Network', 'dim_RNN_gxv')
    num_RNN_gxv = cfg.getint('Network', 'num_RNN_gxv')
    dense_gxv_gz = [int(i) for i in cfg.get('Network', 'dense_gxv_gz').split(',')]
    dim_RNN_gz = cfg.getint('Network', 'dim_RNN_gz')
    num_RNN_gz = cfg.getint('Network', 'num_RNN_gz')
    # Prior
    dim_RNN_prior = cfg.getint('Network', 'dim_RNN_prior')
    num_RNN_prior = cfg.getint('Network', 'num_RNN_prior')
    # Generation
    dense_vz_x = [int(i) for i in cfg.get('Network', 'dense_vz_x').split(',')]

    # Build model
    model = DSAE(x_dim=x_dim, z_dim=z_dim, v_dim=v_dim, activation=activation,
                 dense_x=dense_x,
                 dim_RNN_gv=dim_RNN_gv, num_RNN_gv=num_RNN_gv,
                 dense_gv_v=dense_gv_v, dense_xv_gxv=dense_xv_gxv,
                 dim_RNN_gxv=dim_RNN_gxv, num_RNN_gxv=num_RNN_gxv,
                 dense_gxv_gz=dense_gxv_gz,
                 dim_RNN_gz=dim_RNN_gz, num_RNN_gz=num_RNN_gz,
                 dim_RNN_prior=dim_RNN_prior, num_RNN_prior=num_RNN_prior,
                 dense_vz_x=dense_vz_x,
                 dropout_p=dropout_p, device=device).to(device)

    return model