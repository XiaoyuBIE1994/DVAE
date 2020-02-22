#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is used to test some python grammar
"""

my_seed = 0
import numpy as np
np.random.seed(my_seed)
import torch
torch.manual_seed(my_seed)


from model_vae import VAE as vae_xbie
from backup_simon.VAEs import VAE as vae_simon

def test(x, z_tmp):
    x_dim = 5
    z_dim = 2
    hidden_dim_encoder = [3]
    batch_size = 1
    activation = eval('torch.tanh')
    vae1 = vae_xbie(x_dim, z_dim, hidden_dim_encoder, batch_size, activation)
    vae2 = vae_simon(x_dim, z_dim, hidden_dim_encoder, batch_size, activation)
    # recons_x_1, mu1, logvar1, z1 = vae1.forward(x)
    # recons_x_2, mu2, logvar2, z2 = vae2.forward(x)

    # mean1, logvar1 = vae1.encode(x)
    # mean2, logvar2, _ = vae2.encode(x)

    print(z_tmp)
    recons1 = vae1.decode(z_tmp)
    recons2 = vae2.decode(z_tmp)
    # print(recons1)
    # print(recons2)

    # print('vae from xbie')
    # vae1.print_model()
    # # print('{}\n{}\n{}\n{}'.format(recons_x_1, mu1, logvar1, z1))
    # # print('{}\n{}'.format(mean1, logvar1))
    # print(recons1)
    # print('vae from simon')
    # vae2.print_model()
    # # print('{}\n{}\n{}\n{}'.format(recons_x_2, mu2, logvar2, z2))
    # # print('{}\n{}'.format(mean2, logvar2))
    # print(recons2)
if __name__ == '__main__':
    x = torch.randn(1,5)
    z_tmp = torch.randn(1,2)
    test(x, z_tmp)
    
    