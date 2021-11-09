use_tensorboard = True

if use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter() 

train_langevine = True
mixed_sampling = True

import argparse
import json
import logging
import random
from datetime import datetime
from importlib import import_module
from itertools import chain
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

def get_grad_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

results_dir = "results/"
device = torch.device("cuda")
dataset_name = "shapenet"

from new.shapenet import ShapeNetDataset
dataset = ShapeNetDataset(root_dir="shapenet",
                          classes=["chair"])
points_dataloader = DataLoader(dataset, batch_size=24,
                               shuffle=True,
                               num_workers=8,
                               drop_last=True, pin_memory=True)


from new.aae.aae import Generator, Encoder
from new.models import LangevinEncoderDecoder, NetE

G =  Generator().to(device)
E = Encoder().to(device)
M = NetE().to(device)

G.apply(weights_init)
E.apply(weights_init)
M.apply(weights_init)


net = LangevinEncoderDecoder(E, G, ebm = M)

from new.params import *
from new.utils import *


hparams_dict = {
"z_dim ": z_dim ,
"point_num ": point_num ,
"e_l_steps  ": e_l_steps ,
"e_l_step_size ": e_l_step_size ,
"e_prior_sig ": e_prior_sig ,
"e_l_with_noise ": e_l_with_noise ,
"e_energy_form ": e_energy_form ,
"e_decay ": e_decay ,
"e_beta1 ": e_beta1 ,
"e_beta2 ": e_beta2 ,
"g_l_steps ": g_l_steps ,
"g_llhd_sigma ": g_llhd_sigma ,
"g_l_step_size ": g_l_step_size ,
"g_l_with_noise ": g_l_with_noise ,
"g_decay ": g_decay ,
"g_beta1 ": g_beta1 ,
"g_beta2 ": g_beta2 ,
"batch_size ": batch_size ,
"n_epochs ": n_epochs ,
"gpu_deterministic ": gpu_deterministic ,
"e_lr ": e_lr ,
"g_lr ": g_lr ,
"e_gamma ": e_gamma ,
"g_gamma ": g_gamma ,
"e_init_sig ": e_init_sig ,
"g_init_sig ": g_init_sig ,
"n_printout ": n_printout ,
"langevin_clip ": langevin_clip ,
"e_alpha ": e_alpha ,
"e_beta ": e_beta ,
"e_gamma ": e_gamma ,
"g_alpha ": g_alpha ,
"g_beta ": g_beta ,
"g_gamma ": g_gamma ,
"g_delta ": g_delta
}

if use_tensorboard:
    writer.add_hparams(hparams_dict, {})



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

optim_G = torch.optim.Adam(G.parameters(),
                        **optim_params)

optim_E = torch.optim.Adam(E.parameters(),
                        **optim_params)

optim_M = torch.optim.Adam(M.parameters(),
                        **optim_params_M)

from new.champfer_loss import ChamferLoss
reconstruction_loss = ChamferLoss().to(device)

verbose = True
from tqdm.auto import tqdm
total_step = 0

for epoch in tqdm(range(100)):
    start_epoch_time = datetime.now()

    net.train()

    total_loss = 0.0
    for i, point_data in enumerate(points_dataloader, 1):
#         if i > 1:
#             break
        total_step += 1

        X, _ = point_data
        X = X.to(device)

        # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
        if X.size(-1) == 3:
            X.transpose_(X.dim() - 2, X.dim() - 1)

        batch_num = X.shape[0]
        
        # encode 
        codes, mu, logvar = net.encoder(X)
        
        # posterior
        codes_real = net.posterior(X, mu, logvar, use_lagivine = True, verbose = False)
        X_rec = G(codes_real)
        loss_g = torch.mean(
                0.05 *reconstruction_loss(X.permute(0, 2, 1) + 0.5, X_rec.permute(0, 2, 1) + 0.5))
        
        loss_e = 0.05 * torch.mean((codes - mu)**2)

        # prior
        codes_fake = net.prior(mu, verbose=False)
        prior_energy = net.ebm(codes_fake).detach() 


        # print("prior_energy", torch.mean(prior_energy))
        loss_energy_diff_1 = torch.sum((prior_energy - net.ebm(codes_real))**2) / batch_num

        loss_reg_M = 0
        for param in M.parameters():
            loss_reg_M += torch.norm(param, 1)

        loss_energy_diff_2 = torch.mean(prior_energy - net.ebm(codes))

        loss_m = loss_energy_diff_1 + 0.05 * loss_energy_diff_2 + 1e-4 *  loss_reg_M

        
        # loss_eg = loss_g + loss_e
        
        # reconstruction loss
        optim_G.zero_grad()
        loss_g.backward()
        optim_G.step()
        
        # energy_loss
        optim_M.zero_grad()
        loss_m.backward()
        optim_M.step()

        if i % 10 == 0:
            if use_tensorboard:
                #writer.add_scalar('loss/loss_eg',loss_eg.item(), total_step)
                writer.add_scalar('loss/loss_g',loss_g.item(), total_step)
                writer.add_scalar('loss/loss_m',loss_m.item(), total_step)
                writer.add_scalar('loss/loss_energy_diff_1',loss_energy_diff_1.item(), total_step)
                writer.add_scalar('loss/loss_energy_diff_2',loss_energy_diff_2.item(), total_step)
                writer.add_scalar('loss/loss_reg_M',loss_reg_M.item(), total_step)
                
                
            else:
                print(f'[{epoch}: ({i})] '
                            #f'Loss_EG: {loss_eg.item():.4f} '
                            f'(M loss: {loss_m.item(): .4f}'
                            f' G loss: {loss_g.item(): .4f})'
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

    # net.eval()
    # if mixed_sampling
    fake = G(fixed_noise).data.cpu().numpy()
    codes, _, _ = net.encoder(X)
    codes = net.prior(codes, verbose=False)
    X_rec = G(codes).data.cpu().numpy()
    X = X.data.cpu().numpy()

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


    torch.save(net, "checkpoint/latest.pth")