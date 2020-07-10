#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt

The code in this file is based on:
- “A Disentanagled Recognition and Nonlinear Dynamics Model for Unsupervised Learning” NIPS, 2017, Macro Fraccaro et al.

Not include:
- different learning target (alpha first, then KF params, finally total params)
- no imputation
"""

from torch import nn
import torch
from collection import OrderedDict


class KVAE(nn.Module):

    def __init__(self, x_dim, a_dim = 8, z_dim=16, activation='tanh',
                 dense_x_a=[128,128], dense_a_x=[128,128],
                 init_kf_mat=0.05, noise_transition=0.08, noise_emission=0.03, init_cov=20,
                 K=3, dim_alpha=50, dim_RNN_alpha=50, num_RNN_alpha=1,
                 dropout_p=0, device='cpu'):

        super().__init__()
        ## General parameters
        self.x_dim = x_dim
        self.y_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.u_dim = a_dim
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemError('Wrong activation type')
        self.device = device
        # VAE
        self.dense_x_a = dense_x_a
        self.dense_a_x = dense_a_x
        # LGSSM
        self.init_kf_mat = init_kf_mat
        self.noise_transition = noise_transition
        self.noise_emission = noise_emission
        self.init_cov = init_cov
        # Dynamics params (alpha)
        self.K = K
        self.dim_alpha = dim_alpha
        self.dim_RNN_alpha = dim_RNN_alpha
        self.num_RNN_alpha = num_RNN_alpha

        self.build()

    def build():

        #############
        #### VAE ####
        #############
        # 1. Inference of a_t
        dic_layers = OrderedDict()
        if len(self.dense_x_a) == 0:
            dim_x_a = self.x_dim
            dic_layers["Identity"] = nn.Identity()
        else:
            dim_x_a = self.dense_x_a[-1]
            for n in range(len(self.dense_x_a)):
                if n == 0:
                    dic_layers["linear"+str(n)] = nn.Linear(self.x_dim, self.dense_x_a[n])
                else:dim_zn.Dropout(p=self.dropout_p)
        self.mlp_x_a = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_x_a, self.a_dim)
        self.inf_logvar = nn.Linear(dim_x_a, self.a_dim)
        # 2. Generation of x_t
        dic_layers = OrderedDict()
        if len(self.dense_a_x) == 0:
            dim_a_x = self.a_dim
            dic_layers["Identity"] = nn.Identity()
        else:
            dim_a_x = self.dense_a_x[-1]
            for n in range(len(self.dense_x_a)):
                if n == 0:
                    dic_layers["linear"+str(n)] = nn.Linear(self.a_dim, self.dense_a_x[n])
                else:
                    dic_layers["linear"+str(n)] = nn.Linear(self.dense_a_x[n-1], self.dense_a_x[n])
                dic_layers["activation"+str(n)] = self.activation
                dic_layers["dropout"+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_a_x = nn.Sequential(dic_layers)
        self.gen_logvar = nn.Linear(dim_a_x, self.x_dim)

        ###############
        #### LGSSM ####
        ###############
        # Initializers fro LGSSM variables, torch.Tensor() to enforce troch.float32 type
        # A is an identity matrix
        # B and C are randomly sampled from a Gaussian
        # Q and R are isotroipic covariance matrices
        # z = Az + Bu
        # a = Cz
        self.A = torch.Tensor(np.array([np.eye(self.z_dim) for _ in range(self.K)]), requires_grad=True).to(self.device) # (z_dim. z_dim, K)
        self.B = torch.Tensor(np.array([self.init_kf_mat * np.random.randn(self.z_dim, self.u_dim) for _ in range(self.K)]), requires_grad=True).to(self.device) # (z_dim, u_dim, K)
        self.C = torch.Tensor(np.array([self.init_kf_mat * np.random.randn(self.a_dim, self.z_dim) for _ in range(self.K)]), requires_grad=True).to(self.device) # (a_dim, z_dim, K)
        self.Q = self.noise_transition * torch.eye(self.z_dim).to(self.device) # (z_dim, z_dim, K)
        self.R = self.noise_emission * torch.eye(self.a_dim).to(self.device) # (a_dim, a_dim, K)
        self._I = torch.eye(self.z_dim).to(self.device) # (z_dim, z_dim)

        ###############
        #### Alpha ####
        ###############
        self.rnn_alpha = nn.LSTM(self.dim_alpha, self.dim_RNN_alpha, self.num_RNN_alpha, bidirectional=False)
        self.mlp_alpha = nn.Sequential(nn.Linear(self.dim_alpha, self.K),
                                       nn.Softmax(dim=-1))

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)


    def inference(self, x):
        
        x_a = self.mlp_x_a(x)
        a_mean = self.inf_mean(x_a)
        a_logvar = self.inf_logvar(x_a)
        a = self.reparameterization(a_mean, a_logvar)
        
        return a, a_mean, a_logvar


    def generation_x(self, a):
        
        a_x = self.mlp_a_x(a)
        log_y = self.gen_logvar(a_x)
        y = torch.exp(log_y)

        return y

    def kf_smoother(self, a, u, K, A, B, C, R, Q, optimal_gain=False, alpha_sq=1):
        """"
        Kalman Smoother, refer to Murphy's book (MLAPP), section 18.3
        Difference from KVAE: 
            - no imputation
            - only RNN for the calculation of alpha
            - different notations (rather than using same notations as Murphy's book ,we use notation from model KVAE)
            >>>> z_t = A_t * z_tm1 + B_t * u_t
            >>>> a_t = C_t * z_t
        Input:
            - a, (seq_len, bs, a_dim)
            - u, (seq_len, bs, u_dim)
            - alpha, (seq_len, bs, alpha_dim)
            - K, real number
            - A, (K, z_dim, z_dim)
            - B, (K, z_dim, u_dim)
            - C, (K, a_dim, z_dim)
            - R, (z_dim, z_dim)
            - Q , (a_dim, a_dim)
        """
        
        # Initialization
        seq_len = a.shape[0]
        batch_size = a.shape[1]
        z_t_pred = torch.zeros((batch_size, self.z_dim, 1)).to(self.device) # (bs, z_dim, 1), z_0
        z_pred = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device) # (seq_len, bs, z_dim)
        z_filter = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device) # (seq_len, bs, z_dim)
        z_smooth = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device) # (seq_len, bs, z_dim)
        Sigma_t_pred = torch.Tensor(np.array([self.init_cov * np.eye(self.z_dim) for _ in range(batch_size)])).to(self.device) # (bs, z_dim, z_dim), Sigma_0
        Sigma_pred = torch.zeros((seq_len, batch_size, self.z_dim, self.z_dim)).to(self.device) # (seq_len, bs, z_dim, z_dim)
        Sigma_filter = torch.zeros((seq_len, batch_size, self.z_dim, self.z_dim)).to(self.device) # (seq_len, bs, z_dim, z_dim)
        Sigma_smooth = torch.zeros((seq_len, batch_size, self.z_dim, self.z_dim)).to(self.device) # (seq_len, bs, z_dim, z_dim)
        
        # Calculate alpha
        a_0 = torch.zeros((batch_size, self.a_dim), requires_grad=True).to(self.device) # (bs, a_dim)
        a_tm1 = torch.cat([a_0, a[:-1,:,:]], 0) # (seq_len, bs, a_dim)
        alpha = self.get_alpha(a_tm1)

        # Calculate the mixture of A, B and C
        A_flatten = A.view(self.K, self.z_dim*self.z_dim) # (K, z_dim*z_dim) 
        B_flatten = B.view(self.K, self.z_dim*self.u_dim) # (K, z_dim*u_dim) 
        C_flatten = C.view(self.K, self.a_dim*self.z_dim) # (K, a_dim*z_dim) 
        A_mix = torch.matmul(alpha, A_flatten).view(seq_len, batch_size, self.z_dim, self.z_dim)
        B_mix = torch.matmul(alpha, B_flatten).view(seq_len, batch_size, self.z_dim, self.u_dim)
        C_mix = torch.matmul(alpha, C_flatten).view(seq_len, batch_size, self.a_dim. self.z_dim)

        # Forward filter
        for t in range(seq_len):
            
            # Mixture of A, B and C
            A_t = A_mix[t] # (bs, z_dim. z_dim)
            B_t = B_mix[t] # (bs, z_dim, u_dim)
            C_t = C_mix[t] # (bs, a_dim, z_dim)

            if t != 0:
                u_t = u[t,:,:] # (bs, u_dim)
                z_t_pred = torch.bmm(A_t, z_t) + torch.bmm(B_t, u_t.unsqueeze(-1)) # (bs, z_dim, 1)
                Sigma_t_pred = alpha_sq * torch.bmm(torch.bmm(A_t, Sigma_t), A_t.transpose(1,2)) + self.Q # (bs, z_dim, z_dim)
                # alpha_sq is fading memory control, which indicates how much you want to forgert past measurements, see more infos in 'FilterPy' library
            
            # Residual
            a_pred = torch.bmm(C_t, z_t_pred) # (bs, a_dim, z_dim) x (bs, z_dim, 1)
            res_t = a[t, :, :]..unsqueeze(-1) - a_pred # (bs, a_dim, 1)

            # Kalman gain
            S_t = torch.bmm(torch.bmm(C_t, Sigma_t_pred), C_t.transpose(1,2)) + self.R # (bs, a_dim, a_dim)
            S_t_inv = S_t.inverse()
            K_t = torch.bmm(torch.bmm(Sigma_t_pred, C_t.transpose(1,2)), S_t_inv) # (bs, z_dim, a_dim)

            # Update 
            z_t = z_t_pred + torch.bmm(K_t, res_t) # (bs, z_dim, 1)
            I_KC = self._I - torch.bmm(K_t, C_t) # (bs, z_dim, z_dim)
            if optimal_gain:
                Sigma_t = torch.bmm(I_KC, Sigma_t_pred) # (bs, z_dim, z_dim), only valid with optimal Kalman gain
            else:a_mu_pred
                Sigma_t = torch.bmm(torch.bmm(I_KC, Sigma_t_pred), I_KC.transpose(1,2)) + torch.matmul(torch.matmul(K_t, self.R), K_t.transpose(1,2)) # (bs, z_dim, z_dim), general case

            # Save cache
            z_pred[t] = torch.squeeze(z_t_pred)
            z_filter[t] = torch.squeeze(z_t)
            Sigma_pred[t] = Sigma_t_pred
            Sigma_filter[t] = Sigma_t
  
        # Add the final state from filter to the smoother as initialization
        z_smooth[-1] =  z_filter[-1]
        Sigma_smooth[-1] = Sigma_smooth[-1]

        # Backward smooth, reverse loop from pernultimate state
        for t in range(seq_len-2, -1, -1):
            
            # Backward Kalman gain
            J_t = torch.bmm(Sigma_filter[t], A_mix[t+1].view(-1, self.z_dim, self.z_dim))
            J_t = torch.bmm(J_t, Sigma_pred[t+1].inverse()) # (bs, z_dim, z_dim)

            # Backward smoothing
            z_smooth[t] = z_filter[t] + torch.bmm(J_t, z_smooth[t+1] - z_filter[t+1])
            Sigma_smooth[t] = Sigma_filter[t] + torch.bmm(torch.bmm(J_t, Sigma_smooth[t+1] - Sigma_pred[t+1]), J_t.transpose(1,2))

        # Generate a from smoothing z
        a_gen = C_mix.matmul(z_smooth.unsqueeze(-1))).squeeze() # (seq_len, bs, a_dim)
        
        return a_gen, z_smooth, Sigma_smooth


    def get_alpha(self, a_tm1):
        """"
        Dynamics parameter network alpha for mixing transitions in a SSM
        Unlike original code, we only propose RNN here
        """"
        alpha, _ = self.rnn_alpha(a_tm1) # (seq_len, bs, dim_alpha)
        alpha = self.mlp_alpha(alpha) # (seq_len, bs, K), softmax on K dimension
a
        return alpha
    

    def forward(self):

        # train input: (batch_size, x_dim, seq_len)
        # test input:  (seq_len, x_dim) A_mix[t].view(-1, self.z_dim, self.z_dim)
        # need input:  (seq_len, batch_size, x_dim)
        if len(x.shape) == 3:
            x = x.permute(-1, 0, 1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)_dim) for _ in range(self.K)]), requires_grad=True)
        # self.R = torch.tensor(np.array([np.eye(self.z_dim) for _ in range(self.K)]), requires_grad=True)_dim) for _ in range(self.K)]), requires_grad=True)
        # self.R = torch.tensor(np.array([np.eye(self.z_dim) for _ in range(self.K)]), requires_grad=True)

        # main part 
        a, a_mean, a_logvar = self.inference(x)
        batch_size = a.shape[1]
        a_0 = torch.zeros(1, batch_size, self.a_dim)
        u = torch.cat((a_0, a[:-1]), 0)
        a_gen, z_smooth, Sigma_smooth = self.kf_smoother(a, u, self.K, self.A, self.B, self.C, self.R, self.Q)
        y = self.generation_x(a_gen)

        return y


    def get_info(self):
        
        info = []

        return info


if __name__ == '__main__':

    x_dim = 513
    device = 'cpu'

    kvae = KVAE(x_dim).to(device)
    
    x = torch.rand([2,513,3])
    y = kvae.forward(x)

    print(y[0,0,:])