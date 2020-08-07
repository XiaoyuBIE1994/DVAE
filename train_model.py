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
    optimizer = model_class.optimizer
    batch_size = model_class.batch_size
    seq_len = model_class.sequence_len
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
    log_message = 'Training samples: {}'.format(train_num)
    logger.info(log_message)
    log_message = 'Validation samples: {}'.format(val_num)
    logger.info(log_message)

    # Create python list for loss
    train_loss = np.zeros((epochs,))
    val_loss = np.zeros((epochs,))
    train_recon = np.zeros((epochs,))
    train_KLD = np.zeros((epochs,))
    val_recon = np.zeros((epochs,))
    val_KLD = np.zeros((epochs,))
    best_val_loss = np.inf
    cpt_patience = 0
    cur_best_epoch = epochs
    best_state_dict = model.state_dict()

    # Train with mini-batch SGD
    for epoch in range(epochs):

        start_time = datetime.datetime.now()
        model.train()

        # Batch training
        for batch_idx, batch_data in enumerate(train_dataloader):
            
            batch_data = batch_data.to(model_class.device)
            recon_batch_data = model(batch_data)

            loss_tot, loss_recon, loss_KLD = model.loss
            train_loss[epoch] += loss_tot.item()
            train_recon[epoch] += loss_recon.item()
            train_KLD[epoch] += loss_KLD.item()

            optimizer.zero_grad()
            loss_tot.backward()
            optimizer.step()
            
        # Validation
        for batch_idx, batch_data in enumerate(val_dataloader):

            batch_data = batch_data.to(model_class.device)
            recon_batch_data = model(batch_data)

            loss_tot, loss_recon, loss_KLD = model.loss
            
            val_loss[epoch] += loss_tot.item()
            val_recon[epoch] += loss_recon.item()
            val_KLD[epoch] += loss_KLD.item()

        # Early stop patiance
        if val_loss[epoch] < best_val_loss:
            best_val_loss = val_loss[epoch]
            cpt_patience = 0
            best_state_dict = model.state_dict()
            cur_best_epoch = epoch
        else:
            cpt_patience += 1


        # Loss normalization
        train_loss[epoch] = train_loss[epoch]/ train_num
        val_loss[epoch] = val_loss[epoch] / val_num
        train_recon[epoch] = train_recon[epoch] / train_num 
        train_KLD[epoch] = train_KLD[epoch]/ train_num
        val_recon[epoch] = val_recon[epoch] / val_num 
        val_KLD[epoch] = val_KLD[epoch] / val_num

        # Training time
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
    train_recon = train_recon[:epoch+1]
    train_KLD = train_KLD[:epoch+1]
    val_recon = val_recon[:epoch+1]
    val_KLD = val_KLD[:epoch+1]
    save_file = os.path.join(model_class.save_dir, 
                             model_class.model_name + '_final_epoch' + str(cur_best_epoch) + '.pt')
    torch.save(best_state_dict, save_file)
    
    # Save the training loss and validation loss
    loss_file = os.path.join(model_class.save_dir, 'loss_model.pckl')
    # with open(loss_file, 'wb') as f:
    #     pickle.dump([train_loss, val_loss], f)
    with open(loss_file, 'wb') as f:
        pickle.dump([train_loss, val_loss, train_recon, train_KLD, val_recon, val_KLD], f)

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
    plt.plot(train_recon, '--o')
    plt.plot(train_KLD, '--x')
    plt.legend(('recon', 'KLD'))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(model_class.filename + 'train loss')
    loss_figure_file = os.path.join(model_class.save_dir, 'loss_train_{}.png'.format(model_class.tag))
    plt.savefig(loss_figure_file) 

    plt.clf()
    plt.plot(val_recon, '--o')
    plt.plot(val_KLD, '--x')
    plt.legend(('recon', 'KLD'))
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
        print("Please indiquate config file")
