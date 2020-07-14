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
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import numpy as np
from collections import OrderedDict


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

    def build(self):

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
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_a[n-1], self.dense_x_a[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
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
        # Initializers for LGSSM variables, torch.tensor(), enforce torch.float32 type
        # A is an identity matrix
        # B and C are randomly sampled from a Gaussian
        # Q and R are isotroipic covariance matrices
        # z = Az + Bu
        # a = Cz
        self.A = torch.tensor(np.array([np.eye(self.z_dim) for _ in range(self.K)]), dtype=torch.float32, requires_grad=True).to(self.device) # (z_dim. z_dim, K)
        self.B = torch.tensor(np.array([self.init_kf_mat * np.random.randn(self.z_dim, self.u_dim) for _ in range(self.K)]), dtype=torch.float32, requires_grad=True).to(self.device) # (z_dim, u_dim, K)
        self.C = torch.tensor(np.array([self.init_kf_mat * np.random.randn(self.a_dim, self.z_dim) for _ in range(self.K)]), dtype=torch.float32, requires_grad=True).to(self.device) # (a_dim, z_dim, K)
        self.Q = self.noise_transition * torch.eye(self.z_dim).to(self.device) # (z_dim, z_dim)
        self.R = self.noise_emission * torch.eye(self.a_dim).to(self.device) # (a_dim, a_dim)
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
        Difference from KVAE source code: 
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
        self.mu = torch.zeros((batch_size, self.z_dim, 1)).to(self.device) # (bs, z_dim, 1), z_0
        self.Sigma = self.init_cov * torch.eye(self.z_dim).unsqueeze(0).repeat(batch_size, 1, 1) # (bs, z_dim, z_dim), Sigma_0
        z_pred = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device) # (seq_len, bs, z_dim)
        z_filter = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device) # (seq_len, bs, z_dim)
        z_smooth = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device) # (seq_len, bs, z_dim)
        Sigma_pred = torch.zeros((seq_len, batch_size, self.z_dim, self.z_dim)).to(self.device) # (seq_len, bs, z_dim, z_dim)
        Sigma_filter = torch.zeros((seq_len, batch_size, self.z_dim, self.z_dim)).to(self.device) # (seq_len, bs, z_dim, z_dim)
        Sigma_smooth = torch.zeros((seq_len, batch_size, self.z_dim, self.z_dim)).to(self.device) # (seq_len, bs, z_dim, z_dim)
        
        # Calculate alpha, initial observation a_init is assumed to be zero and can be learned
        a_init = torch.zeros((batch_size, self.a_dim), requires_grad=True).to(self.device) # (bs, a_dim)
        a_tm1 = torch.cat([a_init, a[:-1,:,:]], 0) # (seq_len, bs, a_dim)
        alpha = self.get_alpha(a_tm1) # (seq_len, bs, K)

        # Calculate the mixture of A, B and C
        A_flatten = A.view(self.K, self.z_dim*self.z_dim) # (K, z_dim*z_dim) 
        B_flatten = B.view(self.K, self.z_dim*self.u_dim) # (K, z_dim*u_dim) 
        C_flatten = C.view(self.K, self.a_dim*self.z_dim) # (K, a_dim*z_dim) 
        A_mix = alpha.matmul(A_flatten).view(seq_len, batch_size, self.z_dim, self.z_dim)
        B_mix = alpha.matmul(B_flatten).view(seq_len, batch_size, self.z_dim, self.u_dim)
        C_mix = alpha.matmul(C_flatten).view(seq_len, batch_size, self.a_dim. self.z_dim)

        # Forward filter
        for t in range(seq_len):
            
            # Mixture of A, B and C
            A_t = A_mix[t] # (bs, z_dim. z_dim)
            B_t = B_mix[t] # (bs, z_dim, u_dim)
            C_t = C_mix[t] # (bs, a_dim, z_dim)

            if t == 0:
                z_t_pred = self.mu
                Sigma_t_pred = self.Sigma
            else:
                u_t = u[t,:,:] # (bs, u_dim)
                z_t_pred = A_t.bmm(z_t) + B_t.bmm(u_t.unsqueeze(-1)) # (bs, z_dim, 1), z_{t|t-1}
                Sigma_t_pred = alpha_sq * A_t.bmm(Sigma_t).bmm(A_t.transpose(1,2)) + self.Q # (bs, z_dim, z_dim), Sigma_{t|t-1}
                # alpha_sq (>=1) is fading memory control, which indicates how much you want to forgert past measurements, see more infos in 'FilterPy' library
            
            # Residual
            a_pred = C_t.bmm(z_t_pred)  # (bs, a_dim, z_dim) x (bs, z_dim, 1)
            res_t = a[t, :, :].unsqueeze(-1) - a_pred # (bs, a_dim, 1)

            # Kalman gain
            S_t = C_t.bmm(Sigma_t_pred).bmm(C_t.transpose(1,2)) + self.R # (bs, a_dim, a_dim)
            S_t_inv = S_t.inverse()
            K_t = Sigma_t_pred.bmm(C_t.transpose(1,2)).bmm(S_t_inv) # (bs, z_dim, a_dim)

            # Update 
            z_t = z_t_pred + K_t.bmm(res_t) # (bs, z_dim, 1)
            I_KC = self._I - K_t.bmm(C_t) # (bs, z_dim, z_dim)
            if optimal_gain:
                Sigma_t = I_KC.bmm(Sigma_t_pred) # (bs, z_dim, z_dim), only valid with optimal Kalman gain
            else:
                Sigma_t = I_KC.bmm(Sigma_t_pred).bmm(I_KC.transpose(1,2)) + K_t.matmul(self.R).matmul(K_t.transpose(1,2)) # (bs, z_dim, z_dim), general case

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
            J_t = Sigma_filter[t].bmm(A_mix[t+1].transpose(1,2)).bmm(Sigma_pred[t+1].inverse()) # (bs, z_dim, z_dim)

            # Backward smoothing
            z_smooth[t] = z_filter[t] + J_t.bmm(z_smooth[t+1] - z_filter[t+1])
            Sigma_smooth[t] = Sigma_filter[t] + J_t.bmm(Sigma_smooth[t+1] - Sigma_pred[t+1]).bmm(J_t.transpose(1,2))

        # Generate a from smoothing z
        a_gen = C_mix.matmul(z_smooth.unsqueeze(-1)).squeeze() # (seq_len, bs, a_dim)
        
        return a_gen, z_smooth, Sigma_smooth, A_mix, B_mix, C_mix


    def get_alpha(self, a_tm1):
        """
        Dynamics parameter network alpha for mixing transitions in a SSM
        Unlike original code, we only propose RNN here
        """
        alpha, _ = self.rnn_alpha(a_tm1) # (seq_len, bs, dim_alpha)
        alpha = self.mlp_alpha(alpha) # (seq_len, bs, K), softmax on K dimension

        return alpha
    

    def forward(self, x):

        # train input: (batch_size, x_dim, seq_len)
        # test input:  (seq_len, x_dim) A_mix[t].view(-1, self.z_dim, self.z_dim)
        # need input:  (seq_len, batch_size, x_dim)
        if len(x.shape) == 3:
            x = x.permute(-1, 0, 1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)

        # main part 
        a, a_mean, a_logvar = self.inference(x)
        batch_size = a.shape[1]
        u_0 = torch.zeros(1, batch_size, self.u_dim)
        u = torch.cat((u_0, a[:-1]), 0)
        a_gen, z_smooth, Sigma_smooth, A_mix, B_mix, C_mix = self.kf_smoother(a, u, self.K, self.A, self.B, self.C, self.R, self.Q)
        y = self.generation_x(a_gen)

        return y


    def loss(self, x, y, u, a, a_mean, a_logvar, mu_smooth, Sigma_smooth,
             A, B, C, batch_size=32, seq_len=150):

        # log p_{\theta}(x | a_hat), complex Gaussian
        log_px_given_a = - torch.log(y) - x/y

        # log q_{\phi}(a_hat | x), Gaussian
        log_pa_given_x = - 0.5 * a_logvar - torch.square(a - a_mean) / (2 * torch.exp(a_logvar))

        # log p_{\gamma}(a_tilde, z_tilde | u) < in sub-comment, 'tilde' is hidden for simplification >
        # >>> log p(z_t | z_tm1, u_t), transition
        mvn_smooth = MultivariateNormal(mu_smooth, Sigma_smooth.cholesky())
        z_smooth = mvn_smooth.sample() # # (seq_len, bs, z_dim)
        Az_tm1 = A[:-1].matmul(z_smooth[:-1].unsqueeze(-1)).squeeze() # (seq_len, bs, z_dim)
        Bu_t = B[:-1].matmul(u[:-1].unsqueeze(-1)).squeeze() # (seq_len, bs, z_dim)
        mu_t_transition = Az_tm1 +Bu_t
        z_t_transition = z_smooth[1:]
        trans_centered = z_t_transition - mu_t_transition
        mvn_transition = MultivariateNormal(torch.zeros(self.z_dim), self.Q.cholesky())
        log_prob_transition = mvn_transition.log_prob(trans_centered)
        # >>> log p(z_0 | z_init), init state
        z_0 = z_smooth[0]
        mvn_0 = MultivariateNormal(self.mu, self.Sigma)
        log_prob_0 = mvn_0.log_prob(z_0)
        # >>> log p(a_t | z_t), emission
        Cz_t = C.matmul(z_smooth)
        emis_centered = a - Cz_t
        mvn_emission = MultivariateNormal(torch.zeros(self.a_dim), self.R.cholesky())
        log_prob_emission = mvn_emission.log_prob(emis_centered)
        # >>> log p_{\gamma}(a_tilde, z_tilde | u)
        log_paz_given_u = log_prob_transition + log_prob_0 + log_prob_emission

        # log p_{\gamma}(z_tilde | a_tilde, u)
        # >>> log p(z_t | a, u)
        log_pz_given_au = mvn_smooth.log_prob(z_smooth)

        # Normalization
        log_px_given_a = log_px_given_a /  (batch_size * seq_len)
        log_pa_given_x = log_pa_given_x /  (batch_size * seq_len)
        log_paz_given_u = log_paz_given_u /  (batch_size * seq_len)
        log_pz_given_au = log_pz_given_au /  (batch_size * seq_len)


        loss_tot = - log_px_given_a - log_pa_given_x + log_pa_given_x + log_pz_given_au

        return loss_tot, log_px_given_a, log_pa_given_x, log_paz_given_u, log_pz_given_au


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