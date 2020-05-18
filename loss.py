#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
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