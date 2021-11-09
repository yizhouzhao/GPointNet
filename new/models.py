import torch
import torch.nn as nn
from torch.autograd import Variable

from .params import *
from new.champfer_loss import ChamferLoss
# from metrics.evaluation_metrics import distChamferCUDA, distChamfer

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

        self.z_size = z_dim
        self.use_bias = True
        self.gen = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
            nn.ReLU(),

            nn.Linear(in_features=1024, out_features=2048 * 3, bias=self.use_bias),
        )
    def forward(self, z):
        x = self.gen(z).view(-1, 3, point_num)
        return x


class NetWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.netE = NetE()
        self.netG = NetG()

        self.chamfer_loss = ChamferLoss()

    def loss_fun(self, x:torch.Tensor, y:torch.Tensor, loss_type = "chamfer distance"):
        # if x.is_cuda:
        #     dl, dr = distChamferCUDA(x, y)
        # else:
        #     dl, dr = distChamfer(x, y)

        # cd = torch.mean(dl + dr)

        cd = self.chamfer_loss(x, y) / 2048
        return cd

    def sample_langevin_prior_z(self, z, netE, verbose=False):
        batch_num = z.shape[0]

        z = z.clone().detach()
        z.requires_grad = True
        for i in range(e_l_steps):
            en = netE(z)
            z_grad = torch.autograd.grad(en.sum(), z)[0]
            
            channel_alpha = - e_alpha * 0.5 * e_l_step_size * e_l_step_size * z_grad
            channel_beta = - e_beta * 0.5 * e_l_step_size * e_l_step_size * (1.0 / (e_prior_sig * e_prior_sig) * z.data)

            # z.data = z.data - 0.5 * e_l_step_size * e_l_step_size * (z_grad + 1.0 / (e_prior_sig * e_prior_sig) * z.data)
            channel_gamma = 0.0
            if e_l_with_noise:
                channel_gamma = e_gamma * e_l_step_size * torch.randn_like(z).data
            
            z.data += channel_alpha + channel_beta + channel_gamma

            if (i % 10 == 0 or i == e_l_steps - 1) and verbose:
                print('Langevin prior {:3d}/{:3d}: energy={:8.3f} z_norm:{:8.3f} z_grad_norm:{:8.3f}'.format(i+1, e_l_steps, en.sum().item(), 
                    torch.mean(torch.linalg.norm(z, dim = 1)).item(), torch.mean(torch.linalg.norm(z_grad, dim = 1)).item()))

            z_grad_norm = z_grad.view(batch_num, -1).norm(dim=1).mean()

        return z.detach(), z_grad_norm

    def sample_langevin_post_z(self, z, x, netG, netE, verbose=False):

        batch_num = z.shape[0]

        z = z.clone().detach()
        z.requires_grad = True
        for i in range(g_l_steps):
            x_hat = netG(z)
            # print("x_hat.shape", x_hat.shape, x.shape)
            g_log_lkhd = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * self.loss_fun(x_hat.transpose(1,2).contiguous() + 0.5, x.transpose(1,2).contiguous() + 0.5)
            z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

            en = netE(z)
            z_grad_e = torch.autograd.grad(en.sum(), z)[0]

            channel_alpha = - g_alpha * 0.5 * g_l_step_size * g_l_step_size * z_grad_e
            channel_beta = - g_beta * 0.5 * g_l_step_size * g_l_step_size * (1.0 / (e_prior_sig * e_prior_sig) * z.data)

            channel_gamma = 0.0 
            channel_delta = - g_delta * 0.5 * g_l_step_size * g_l_step_size * z_grad_g
    
            # z.data = z.data - 0.5 * g_l_step_size * g_l_step_size * (z_grad_g + z_grad_e + 1.0 / (e_prior_sig * e_prior_sig) * z.data)
            if g_l_with_noise:
                channel_gamma += g_gamma * g_l_step_size * torch.randn_like(z).data

            z.data += channel_alpha + channel_beta + channel_gamma + channel_delta


            if (i % 10 == 0 or i == g_l_steps - 1) and verbose:
                print('Langevin posterior {:3d}/{:3d}: LOSS G={:8.3f} z_norm:{:8.3f}'.format(i+1, g_l_steps, g_log_lkhd.item(),
                    torch.mean(torch.linalg.norm(z, dim = 1)).item()))

            z_grad_g_grad_norm = z_grad_g.view(batch_num, -1).norm(dim=1).mean()
            z_grad_e_grad_norm = z_grad_e.view(batch_num, -1).norm(dim=1).mean()

        return z.detach(), z_grad_g_grad_norm, z_grad_e_grad_norm

    def forward(self, z, x=None, prior=True, verbose = False):
        # print('z', z.shape)
        # if x is not None:
        #    print('x', x.shape)

        if prior:
            return self.sample_langevin_prior_z(z, self.netE, verbose=verbose)[0]
        else:
            return self.sample_langevin_post_z(z, x, self.netG, self.netE, verbose=verbose)[0]

    def sample_x(self, n:int, sig=e_init_sig, device = torch.device("cuda")):
        z_0 = sig * torch.randn(*[n, z_dim]).to(device)
        z_k = self.forward(Variable(z_0), prior=True)
        x_samples = self.netG(z_k).clamp(min= -langevin_clip, max=langevin_clip).detach().cpu()
        return x_samples


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.z_size = z_dim
        self.use_bias = True

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1,
                      bias=self.use_bias),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1,
                      bias=self.use_bias),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True)
        )

        self.mu_layer = nn.Linear(256, self.z_size, bias=True)
        self.std_layer = nn.Linear(256, self.z_size, bias=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # print("input shape", x.shape)
        output = self.conv(x)
        output2 = output.max(dim=2)[0]
        logit = self.fc(output2)
        mu = self.mu_layer(logit)
        logvar = self.std_layer(logit)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class LangevinEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, ebm = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.ebm = ebm
        self.decoder = decoder

        self.chamfer_loss = ChamferLoss()

    def loss_fun(self, x:torch.Tensor, y:torch.Tensor, loss_type = "chamfer distance"):
        # if x.is_cuda:
        #     dl, dr = distChamferCUDA(x, y)
        # else:
        #     dl, dr = distChamfer(x, y)

        # cd = torch.mean(dl + dr)

        cd = self.chamfer_loss(x, y) 
        return torch.mean(cd)

    def prior(self, z, verbose=False):
        z = z.clone().detach()
        z.requires_grad = True
        for i in range(e_l_steps):
            en = self.ebm(z)
            z_grad = torch.autograd.grad(en.sum(), z)[0]
            
            channel_alpha = - e_alpha * 0.5 * e_l_step_size * e_l_step_size * z_grad
            channel_beta = - e_beta * 0.5 * e_l_step_size * e_l_step_size * (1.0 / (e_prior_sig * e_prior_sig) * z.data)

            # z.data = z.data - 0.5 * e_l_step_size * e_l_step_size * (z_grad + 1.0 / (e_prior_sig * e_prior_sig) * z.data)
            channel_gamma = 0.0
            if e_l_with_noise:
                channel_gamma = e_gamma * e_l_step_size * torch.randn_like(z).data
            
            z.data += channel_alpha + channel_beta + channel_gamma

            if (i % 10 == 0 or i == e_l_steps - 1) and verbose:
                print('Langevin prior {:3d}/{:3d}: energy={:8.3f} z_norm:{:8.3f} z_grad_norm:{:8.3f}'.format(i+1, e_l_steps, en.sum().item(), 
                    torch.mean(torch.linalg.norm(z, dim = 1)).item(), torch.mean(torch.linalg.norm(z_grad, dim = 1)).item()))

            # z_grad_norm = z_grad.view(batch_num, -1).norm(dim=1).mean()

        return z.detach()

    def posterior(self, x, mu, logvar, use_lagivine = True, verbose = False):
        if not use_lagivine:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        
        z = mu
        batch_num = z.shape[0]
        z = z.clone().detach()
        z.requires_grad = True

        # print("hidden shape", z.shape)
        for i in range(g_l_steps):
            x_hat = self.decoder(z)
            # print("x_hat.shape", x_hat.shape, x.shape)
            g_log_lkhd = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * self.loss_fun(x_hat.transpose(1,2).contiguous() + 0.5, x.transpose(1,2).contiguous() + 0.5)
            z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

            # en = netE(z)
            # z_grad_e = torch.autograd.grad(en.sum(), z)[0]

            
            channel_beta = - g_beta * 0.5 * g_l_step_size * g_l_step_size * (1.0 / (e_prior_sig * e_prior_sig) * z.data)

            channel_gamma = 0.0 
            channel_delta = - g_delta * 0.5 * g_l_step_size * g_l_step_size * z_grad_g
    
            # z.data = z.data - 0.5 * g_l_step_size * g_l_step_size * (z_grad_g + z_grad_e + 1.0 / (e_prior_sig * e_prior_sig) * z.data)
            if g_l_with_noise:
                channel_gamma += g_gamma * g_l_step_size * torch.randn_like(z).data
    
            z.data +=  channel_beta + channel_gamma + channel_delta


            if (i % 10 == 0 or i == g_l_steps - 1) and verbose:
                print('Langevin posterior {:3d}/{:3d}: LOSS G={:8.3f} z_norm:{:8.3f}'.format(i+1, g_l_steps, g_log_lkhd.item(),
                    torch.mean(torch.linalg.norm(z, dim = 1)).item()))

            # z_grad_g_grad_norm = z_grad_g.view(batch_num, -1).norm(dim=1).mean()
            # z_grad_e_grad_norm = z_grad_e.view(batch_num, -1).norm(dim=1).mean()

        return z.detach() #, z_grad_g_grad_norm, z_grad_e_grad_norm