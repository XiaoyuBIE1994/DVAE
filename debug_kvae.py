#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil

cfg_file = '/mnt/xbie/Code/dvae-speech/config/cfg_kvae.ini'
stat_file = '/mnt/xbie/Results/2020_DVAE/saved_model/WSJ0_2020-08-25-09h12_KVAE_vae-pretrain_kf20_z_dim=16/KVAE_epoch7.pt'

model_class = build_model(cfg_file)
model = model_class.model
epochs = model_class.model
logger = model_class.logger
device = model_class.device


optim_vae = model_class.optimizer_vae
optim_vae_kf = model_class.optimizer_vae_kf
optim_all = model_class.optimizer_all
optimizer_net = model_class.optimizer_net

save_cfg = os.path.join(model_class.save_dir, 'config.ini')
shutil.copy(config_file, save_cfg)

train_dataloader, val_dataloader, train_num, val_num = model_class.build_dataloader()


model.load_statt_dict(torch.load(stat_file, map_location=device))

train_loss = np.zeros((epochs,))
val_loss = np.zeros((epochs,))
train_vae = np.zeros((epochs,))
train_lgssm = np.zeros((epochs,))
val_vae = np.zeros((epochs,))
val_lgssm = np.zeros((epochs,))

best_val_loss = np.inf
cpt_patience = 0
cur_best_epoch = epochs

logger.info('====> KVAE Training')

for epoch in range(epochs):
    
    # Scheduler training, beneficial to achieve better convergence not to
    # train alpha from the beginning
    if model_class.scheduler_training:
        if epoch < model_class.only_vae_epochs:
            optimizer = model_class.optimizer_vae
            # optimizer = model_class.optimizer_net
        elif epoch < model_class.only_vae_epochs + model_class.kf_update_epochs:
            optimizer = model_class.optimizer_vae_kf
            # optimizer = model_class.optimizer_lgssm
        else:
            optimizer = model_class.optimizer_all
    else:
        optimizer = model_class.optimizer_all

    start_time = datetime.datetime.now()
    model.train()

    # Batch training
    for batch_idx, batch_data in enumerate(train_dataloader):

        batch_data = batch_data.to(model_class.device)
        recon_batch_data = model(batch_data)

        loss, loss_vae, loss_lgssm = model.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss[epoch] += loss.item()
        train_vae[epoch] += loss_vae.item()
        train_lgssm[epoch] += loss_lgssm.item()
        
        
    # Validation
    for batch_idx, batch_data in enumerate(val_dataloader):

        batch_data = batch_data.to(model_class.device)
        recon_batch_data = model(batch_data)

        loss, loss_vae, loss_lgssm = model.loss

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
    # if epoch % model_class.save_frequency == 0:
    if True:
        save_file = os.path.join(model_class.save_dir, 
                                    model_class.model_name + '_epoch' + str(cur_best_epoch) + '.pt')
        torch.save(best_state_dict, save_file)