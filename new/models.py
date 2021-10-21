import torch
import torch.nn as nn

z_dim = 1024
point_num = 2048

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

            nn.Linear(512, 3*point_num) # output: batch x (3 Point num)
        )

    def forward(self, z):
        x = self.gen(z).view(-1, point_num, 3)
