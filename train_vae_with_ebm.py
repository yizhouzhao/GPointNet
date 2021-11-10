use_tensorboard = False

if use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter() 

import argparse
import json
import logging
import random
from datetime import datetime
from importlib import import_module
from itertools import chain
import os
from os.path import join, exists

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.autograd import Variable

from new.aae.pcutil import plot_3d_point_cloud
#from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging

cudnn.benchmark = True
results_dir = "results/"
device = torch.device("cuda")
dataset_name = "shapenet"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

from new.shapenet import ShapeNetDataset

dataset = ShapeNetDataset(root_dir="shapenet",
                          classes=["chair"])
points_dataloader = DataLoader(dataset, batch_size=24,
                               shuffle=True,
                               num_workers=8,
                               drop_last=True, pin_memory=True)

from new.aae.aae import Generator, Encoder
from new.models import NetE

G =  Generator().to(device)
E = Encoder().to(device)
M = NetE().to(device)

G.apply(weights_init)
E.apply(weights_init)
M.apply(weights_init)

from new.params import *
from new.utils import *

#
# Float Tensors
#
fixed_noise = torch.FloatTensor(16, z_dim)
fixed_noise.normal_(mean=0, std=0.2)
std_assumed = torch.tensor(0.2)

fixed_noise = fixed_noise.to(device)
std_assumed = std_assumed.to(device)


#
# Optimizers
#
optim_params = {
                "lr": 0.0005,
                "weight_decay": 0,
                "betas": [0.9, 0.999],
                "amsgrad": False
            }

optim_params_M = {
                "lr": 1e-5,
                "weight_decay": 0,
                "betas": [0.9, 0.999],
                "amsgrad": False
            }

EG_optim = torch.optim.Adam(chain(E.parameters(), G.parameters()),
                    **optim_params)


optim_M = torch.optim.Adam(M.parameters(),
                        **optim_params_M)

from new.champfer_loss import ChamferLoss
reconstruction_loss = ChamferLoss().to(device)

def prior(z):
    z = z.clone().detach()
    z.requires_grad = True
    for _ in range(e_l_steps):
        en = M(z)
        z_grad = torch.autograd.grad(en.sum(), z)[0]
        
        channel_alpha = - e_alpha * 0.5 * e_l_step_size * e_l_step_size * z_grad
        channel_beta = - e_beta * 0.5 * e_l_step_size * e_l_step_size * (1.0 / (e_prior_sig * e_prior_sig) * z.data)

        # z.data = z.data - 0.5 * e_l_step_size * e_l_step_size * (z_grad + 1.0 / (e_prior_sig * e_prior_sig) * z.data)
        channel_gamma = 0.0
        if e_l_with_noise:
            channel_gamma = e_gamma * e_l_step_size * torch.randn_like(z).data
        
        z.data += channel_alpha + channel_beta + channel_gamma

        if (i % 10 == 0 or i == e_l_steps - 1) and False:
            print('Langevin prior {:3d}/{:3d}: energy={:8.3f} z_norm:{:8.3f} z_grad_norm:{:8.3f}'.format(i+1, e_l_steps, en.sum().item(), 
                torch.mean(torch.linalg.norm(z, dim = 1)).item(), torch.mean(torch.linalg.norm(z_grad, dim = 1)).item()))

    z = z.detach()

    return z

total_step = 0
for epoch in range(400):
    start_epoch_time = datetime.now()
    
    G.train()
    E.train()

    total_loss = 0.0
    for i, point_data in enumerate(points_dataloader, 1):
        # if i > 1:
        #     break
        total_step += 1
        
        X, _ = point_data
        X = X.to(device)
        batch_num = X.shape[0]

        # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
        if X.size(-1) == 3:
            X.transpose_(X.dim() - 2, X.dim() - 1)

    
        codes, mu, logvar = E(X)
        X_rec = G(codes)

        loss_e = torch.mean(
                0.05 *
            reconstruction_loss(X.permute(0, 2, 1) + 0.5,
                                X_rec.permute(0, 2, 1) + 0.5))

        loss_kld = -0.5 * torch.mean(
            1 - 2.0 * torch.log(std_assumed) + logvar -
            (mu.pow(2) + logvar.exp()) / torch.pow(std_assumed, 2))

        # VAE
        loss_eg = loss_e + loss_kld
        EG_optim.zero_grad()
        loss_eg.backward()
        total_loss += loss_eg.item()
        EG_optim.step()

        # langivine
        z = e_init_sig * torch.randn(*[batch_num, z_dim]).to(device)
        z = prior(z)

        loss_M = - torch.mean(M(z) - M(codes).detach())

        optim_M.zero_grad()
        loss_M.backward()
        total_loss += loss_M.item()
        optim_M.step()


        if i % 30 == 0:
            if use_tensorboard:
                    writer.add_scalar('loss/Loss_EG',loss_eg.item(), total_step)
                    writer.add_scalar('loss/REC',loss_e.item(), total_step)
                    writer.add_scalar('loss/KLD',loss_kld.item(), total_step)
                    writer.add_scalar('loss/loss_M',loss_M.item(), total_step)
                    
            else:
                print(f'[{epoch}: ({i})] '
                            f'Loss_EG: {loss_eg.item():.4f} '
                            f'(REC: {loss_e.item(): .4f}'
                            f' KLD: {loss_kld.item(): .4f})'
                            f' loss_M: {loss_M.item(): .4f})'
                            f' Time: {datetime.now() - start_epoch_time}')

    print(
        f'[{epoch}/{400}] '
        f'Loss_G: {total_loss / i:.4f} '
        f'Time: {datetime.now() - start_epoch_time}'
    )
        
    ################################### eval ######################################
    #
    # Save intermediate results
    #

    G.eval()
    E.eval()
    M.eval()

    if not exists(results_dir):     
        os.mkdir(results_dir)

    if not exists(join(results_dir, 'samples')):        
        os.mkdir(join(results_dir, 'samples'))

    
    fake = G(fixed_noise).data.cpu().numpy()
    codes, _, _ = E(X)
    X_rec = G(codes).data.cpu().numpy()
    X = X.data.cpu().numpy()

    # langivine
    z = e_init_sig * torch.randn(*[batch_num, z_dim]).to(device)
    z = prior(z)
    X_sample = G(z).data.cpu().numpy()

    for k in range(5):
        fig = plot_3d_point_cloud(X[k][0], X[k][1], X[k][2],
                                  in_u_sphere=True, show=False)
        fig.savefig(
            join(results_dir, 'samples', f'{epoch}_{k}_real.png'))
        plt.close(fig)

    for k in range(5):
        fig = plot_3d_point_cloud(fake[k][0], fake[k][1], fake[k][2],
                                  in_u_sphere=True, show=False,
                                  title=str(epoch))
        fig.savefig(
            join(results_dir, 'samples', f'{epoch:05}_{k}_fixed.png'))
        plt.close(fig)

    for k in range(5):
        fig = plot_3d_point_cloud(X_rec[k][0],
                                  X_rec[k][1],
                                  X_rec[k][2],
                                  in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'samples',
                         f'{epoch}_{k}_reconstructed.png'))
        plt.close(fig)
    
    for k in range(5):
        fig = plot_3d_point_cloud(X_sample[k][0],
                                  X_sample[k][1],
                                  X_sample[k][2],
                                  in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'samples',
                         f'{epoch}_{k}_prior_sample.png'))
        plt.close(fig)