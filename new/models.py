import torch
import torch.nn as nn

from .params import *
from metrics.evaluation_metrics import distChamferCUDA, distChamfer

class NetE(nn.Module):
    def __init__(self):
        super().__init__()

        f = nn.GELU()

        self.ebm = nn.Sequential(
            nn.Linear(z_dim, 512),
            f,

            nn.Linear(512, 64),
            f,

            nn.Linear(64, 1)
        )

    def forward(self, z):
        return self.ebm(z.squeeze())


class NetG(nn.Module):
    def __init__(self):
        super().__init__()

        f = nn.GELU()

        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            f,

            nn.Linear(256, 512),
            f,

            nn.Linear(512, 1024),
            f,

            nn.Linear(1024, 3*point_num) # output: batch x (3 Point num)
        )

    def forward(self, z):
        x = self.gen(z).view(-1, 3, point_num)
        return x


class NetWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.netE = NetE()
        self.netG = NetG()

    def loss_fun(self, x:torch.Tensor, y:torch.Tensor, loss_type = "chamfer distance"):
        if x.is_cuda:
            dl, dr = distChamferCUDA(x, y)
        else:
            dl, dr = distChamfer(x, y)

        cd = torch.mean(dl + dr)
        return cd

    def sample_langevin_prior_z(self, z, netE, verbose=False):
        batch_num = z.shape[0]

        z = z.clone().detach()
        z.requires_grad = True
        for i in range(e_l_steps):
            en = netE(z)
            z_grad = torch.autograd.grad(en.sum(), z)[0]

            z.data = z.data - 0.5 * e_l_step_size * e_l_step_size * (z_grad + 1.0 / (e_prior_sig * e_prior_sig) * z.data)
            if e_l_with_noise:
                z.data += e_l_step_size * torch.randn_like(z).data

            if (i % 5 == 0 or i == e_l_steps - 1) and verbose:
                print('Langevin prior {:3d}/{:3d}: energy={:8.3f}'.format(i+1, e_l_steps, en.sum().item()))

            z_grad_norm = z_grad.view(batch_num, -1).norm(dim=1).mean()

        return z.detach(), z_grad_norm

    def sample_langevin_post_z(self, z, x, netG, netE, verbose=False):

        batch_num = z.shape[0]

        z = z.clone().detach()
        z.requires_grad = True
        for i in range(g_l_steps):
            x_hat = netG(z)
            # print("x_hat.shape", x_hat.shape, x.shape)
            g_log_lkhd = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * self.loss_fun(x_hat.transpose(1,2).contiguous(), x.transpose(1,2).contiguous())
            z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

            en = netE(z)
            z_grad_e = torch.autograd.grad(en.sum(), z)[0]

            z.data = z.data - 0.5 * g_l_step_size * g_l_step_size * (z_grad_g + z_grad_e + 1.0 / (e_prior_sig * e_prior_sig) * z.data)
            if g_l_with_noise:
                z.data += g_l_step_size * torch.randn_like(z).data

            if (i % 5 == 0 or i == g_l_steps - 1) and verbose:
                print('Langevin posterior {:3d}/{:3d}: MSE={:8.3f}'.format(i+1, g_l_steps, g_log_lkhd.item()))

            z_grad_g_grad_norm = z_grad_g.view(batch_num, -1).norm(dim=1).mean()
            z_grad_e_grad_norm = z_grad_e.view(batch_num, -1).norm(dim=1).mean()

        return z.detach(), z_grad_g_grad_norm, z_grad_e_grad_norm

    def forward(self, z, x=None, prior=True):
        # print('z', z.shape)
        # if x is not None:
        #    print('x', x.shape)

        if prior:
            return self.sample_langevin_prior_z(z, self.netE)[0]
        else:
            return self.sample_langevin_post_z(z, x, self.netG, self.netE)[0]