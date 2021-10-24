import torch
import  torch.nn.functional as F
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from .params import *

def set_gpu(gpu):
    torch.cuda.set_device('cuda:{}'.format(gpu))
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def energy(score):
    if e_energy_form == 'tanh':
        energy = F.tanh(-score.squeeze())
    elif e_energy_form == 'sigmoid':
        energy = F.sigmoid(score.squeeze())
    elif e_energy_form == 'identity':
        energy = score.squeeze()
    elif e_energy_form == 'softplus':
        energy = F.softplus(score.squeeze())
    return energy

def sample_p_0(n=batch_size, sig=e_init_sig, device = torch.device("cuda")):
    return sig * torch.randn(*[n, z_dim]).to(device)


def show_point_clouds(point_clouds):
    num_cols,num_rows = 4, 4
    idx = None
    if point_clouds.shape[1] < 10: 
        point_clouds = np.swapaxes(point_clouds, 1, 2)
    num_clouds = len(point_clouds)
    # num_rows = min(num_rows, num_clouds // num_cols + 1)

    fig = plt.figure(figsize=(num_cols * 4, num_rows * 4))
    for i, pts in enumerate(point_clouds[:num_cols*num_rows]):
        #print(i)
        if point_clouds.shape[2] == 3: 
            ax = plt.subplot(num_rows, num_cols, i+1, projection='3d')
            plt.subplots_adjust(0,0,1,1,0,0)
            #ax.axis('off')
            if idx is not None:
                ax.set_title(str(idx[i]))
            ax.scatter(pts[:,0], pts[:,2], pts[:,1], marker='.', s=50, c=pts[:,2], cmap=plt.get_cmap('gist_rainbow'))
        else: 
            ax = plt.subplot(num_rows, num_cols, i+1)
            plt.subplots_adjust(0,0,1,1,0,0)
            # ax.axis('off')
            if idx is not None:
                ax.set_title(str(idx[i]))
            ax.scatter(pts[:,1], -pts[:,0], marker='.', s=30)
    
    plt.show()