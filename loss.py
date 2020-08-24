#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt

To be deleted in the futere, already intergrated in build model
"""
import torch

def loss_vlb(recon_x, x, mu, logvar, mu_prior=None, logvar_prior=None, batch_size=32, seq_len=50):
    recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
    KLD = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), logvar_prior.exp()))
    return (recon + KLD) / (batch_size * seq_len)

def loss_vlb_beta(recon_x, x, mu, logvar, mu_prior=None, logvar_prior=None, beta=1, batch_size=32, seq_len=50):
    recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 ) 
    KLD = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), logvar_prior.exp()))
    return (recon + beta * KLD) / (batch_size * seq_len)


def loss_vlb_separate(recon_x, x, mu, logvar, mu_prior=None, logvar_prior=None, batch_size=32, seq_len=50):
    recon = torch.sum(  x/recon_x - torch.log(x/recon_x) - 1 )  / (batch_size * seq_len)
    KLD = -0.5 * torch.sum(logvar - logvar_prior - torch.div((logvar.exp() + (mu - mu_prior).pow(2)), logvar_prior.exp())) / (batch_size * seq_len)
    return recon, KLD


    def get_loss(self, x, y, z_mean, z_logvar, z_mean_p, z_logvar_p, seq_len, batch_size, beta=1):

        loss_recon = torch.sum( x/y - torch.log(x/y) - 1)
        loss_KLD = -0.5 * torch.sum(z_logvar - z_logvar_p 
                - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), 2 * z_logvar_p.exp())) / (batch_size * seq_len)
        
        loss_recon = loss_recon / (batch_size * seq_len)
        loss_KLD = loss_KLD / (batch_size * seq_len)
        loss_tot = beta * loss_recon + loss_KLD

        return loss_tot, loss_recon, loss_KLD

    def get_loss(self, x, y, z_mean, z_logvar, batch_size, seq_len, beta=1):

        loss_recon = torch.sum( x/y - torch.log(x/y) - 1)
        loss_KLD = -0.5 * torch.sum(z_logvar -  z_logvar.exp() - z_mean.pow(2))

        loss_recon = loss_recon / (batch_size * seq_len)
        loss_KLD = loss_KLD / (batch_size * seq_len)
        loss_tot = beta * loss_recon + loss_KLD

        return loss_tot, loss_recon, loss_KLD
