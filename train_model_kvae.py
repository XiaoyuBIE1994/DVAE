#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import sys
import os
import shutil
import socket
import datetime
import pickle
import numpy as np
import torch
from logger import get_logger
from configparser import ConfigParser
from build_model import build_model
from loss import loss_vlb, loss_vlb_beta, loss_vlb_separate

# import plot functions and disabhle the figure display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

my_seed = 0
np.random.seed(my_seed)
torch.manual_seed(my_seed)


def train_model(config_file):
    
    torch.autograd.set_detect_anomaly(True)
    # Build model using config_file
    model_class = build_model(config_file)
    model = model_class.model
    epochs = model_class.epochs
    logger = model_class.logger
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    device = model_class.device

    # Save the model parameters
    save_cfg = os.path.join(model_class.save_dir, 'config.ini')
    shutil.copy(config_file, save_cfg)
    
    # Check if gpu is available on cluster
    if 'gpu' in model_class.hostname and device == 'cpu':
        logger.error('GPU unavailable on cluster, training stop')
        return

    # Create dataloader
    train_dataloader, val_dataloader, train_num, val_num = model_class.build_dataloader()

    # Create python list for loss
    train_loss = np.zeros((epochs,))
    val_loss = np.zeros((epochs,))
    train_vae = np.zeros((epochs,))
    train_lgssm = np.zeros((epochs,))
    val_vae = np.zeros((epochs,))
    val_lgssm = np.zeros((epochs,))
    best_val_loss = np.inf
    cpt_patience = 0
    cur_best_epoch = epochs
    best_state_dict = model.state_dict()
    # Train with mini-batch SGD
    for epoch in range(epochs):
        
       # Scheduler training, beneficial to achieve better convergence not to
       # train alpha from the beginning
        if model_class.scheduler_training:
            if epoch < model_class.only_vae_epochs:
                optimizer = model_class.optimizer_vae
            elif epoch < model_class.only_vae_epochs + model_class.kf_update_epochs:
                optimizer = model_class.optimizer_vae_kf
            else:
                optimizer = model_class.optimizer_all
        else:
            optimizer = model_class.optimizer_old

        start_time = datetime.datetime.now()
        model.train()

        # Batch training
        for batch_idx, batch_data in enumerate(train_dataloader):

            batch_data = batch_data.to(model_class.device)
            optimizer.zero_grad()
            y, loss, loss_vae, loss_lgssm = model(batch_data)
        
            loss.backward()
            train_loss[epoch] += loss.item()
            train_vae[epoch] += loss_vae.item()
            train_lgssm[epoch] += loss_lgssm.item()
            optimizer.step()
            
        # Validation
        for batch_idx, batch_data in enumerate(val_dataloader):

            batch_data = batch_data.to(model_class.device)
            optimizer.zero_grad()
            y, loss, loss_vae, loss_lgssm = model(batch_data)

            val_loss[epoch] += loss.item()
            val_vae[epoch] += loss_vae.item()
            val_lgssm[epoch] += loss_lgssm.item()

        # Early stop patiance
        if val_loss[epoch] < best_val_loss:
            best_val_loss = val_loss[epoch]
            cpt_patience = 0
            best_state_dict = model.state_dict()
            cur_best_epoch = epoch
        else:
            cpt_patience += 1


        train_loss[epoch] = train_loss[epoch]/ train_num
        val_loss[epoch] = val_loss[epoch] / val_num

        train_vae[epoch] = train_vae[epoch] / train_num 
        train_lgssm[epoch] = train_lgssm[epoch]/ train_num
        val_vae[epoch] = val_vae[epoch] / val_num 
        val_lgssm[epoch] = val_lgssm[epoch] / val_num

        end_time = datetime.datetime.now()
        interval = (end_time - start_time).seconds / 60
        log_message = 'Epoch: {} train loss: {:.4f} val loss {:.4f} training time {:.2f}m'.format(epoch, train_loss[epoch], val_loss[epoch], interval)
        logger.info(log_message)

        # Stop traning if early-stop triggers
        if cpt_patience == model_class.early_stop_patience:
            logger.info('Early stop patience achieved')
            break

        # Save model parameters regularly
        if epoch % model_class.save_frequency == 0:
            save_file = os.path.join(model_class.save_dir, 
                                     model_class.model_name + '_epoch' + str(cur_best_epoch) + '.pt')
            torch.save(best_state_dict, save_file)
    
    # Save the final weights of network with the best validation loss
    train_loss = train_loss[:epoch+1]
    val_loss = val_loss[:epoch+1]
    train_vae = train_vae[:epoch+1]
    train_lgssm = train_lgssm[:epoch+1]
    val_vae = val_vae[:epoch+1]
    val_lgssm = val_lgssm[:epoch+1]
    save_file = os.path.join(model_class.save_dir, 
                             model_class.model_name + '_final_epoch' + str(cur_best_epoch) + '.pt')
    torch.save(best_state_dict, save_file)
    
    # Save the training loss and validation loss
    loss_file = os.path.join(model_class.save_dir, 'loss_model.pckl')
    # with open(loss_file, 'wb') as f:
    #     pickle.dump([train_loss, val_loss], f)
    with open(loss_file, 'wb') as f:
        pickle.dump([train_loss, val_loss, train_vae, train_lgssm, val_vae, val_lgssm], f)

    # Save the loss figure
    plt.clf()
    plt.plot(train_loss, '--o')
    plt.plot(val_loss, '--x')
    plt.legend(('train loss', 'val loss'))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(model_class.filename)
    loss_figure_file = os.path.join(model_class.save_dir, 'loss_{}.png'.format(model_class.tag))
    plt.savefig(loss_figure_file) 

    plt.clf()
    plt.plot(train_vae, '--o')
    plt.plot(train_lgssm, '--x')
    plt.legend(('VAE', 'LGSSM'))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(model_class.filename + 'train loss')
    loss_figure_file = os.path.join(model_class.save_dir, 'loss_train_{}.png'.format(model_class.tag))
    plt.savefig(loss_figure_file) 

    plt.clf()
    plt.plot(val_vae, '--o')
    plt.plot(val_lgssm, '--x')
    plt.legend(('VAE', 'LGSSM'))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(model_class.filename + 'validation loss')
    loss_figure_file = os.path.join(model_class.save_dir, 'loss_val_{}.png'.format(model_class.tag))
    plt.savefig(loss_figure_file) 


if __name__ == '__main__':
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
        train_model(config_file)
    else:
        logger.warning("Please indiquate config file")
