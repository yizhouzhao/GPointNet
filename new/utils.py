import torch
import  torch.nn.functional as F
import os
import random
import numpy as np

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