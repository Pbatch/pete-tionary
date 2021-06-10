import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 w0=1.,
                 c=6.,
                 is_first=False,
                 is_last=False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.w0 = w0
        self.c = c
        self.is_first = is_first
        self.is_last = is_last

        self.activation = nn.Identity() if self.is_last else Sine(w0)
        self.weight, self.bias = self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        w_std = (1 / self.dim_in) if self.is_first else (math.sqrt(self.c / self.dim_in) / self.w0)

        weight = torch.zeros(self.dim_out, self.dim_in)
        weight.uniform_(-w_std, w_std)
        weight = nn.Parameter(weight)

        bias = torch.zeros(self.dim_out)
        bias.uniform_(-w_std, w_std)
        bias = nn.Parameter(bias)

        return weight, bias

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class SirenNet(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_hidden,
                 dim_out,
                 n_layers,
                 image_size,
                 w0=1.,
                 w0_initial=30.):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers
        self.image_size = image_size
        self.w0 = w0
        self.w0_initial = w0_initial

        self.layers = self.initialize_layers()
        self.initialize_mesh_grid()

    def initialize_mesh_grid(self):
        tensors = [torch.linspace(-1, 1, steps=self.image_size),
                   torch.linspace(-1, 1, steps=self.image_size)]
        mesh_grid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mesh_grid = rearrange(mesh_grid, 'h w c -> (h w) c')
        self.register_buffer('mesh_grid', mesh_grid)

    def initialize_layers(self):
        layers = nn.ModuleList([])
        layers.append(Siren(dim_in=self.dim_in, dim_out=self.dim_hidden, w0=self.w0_initial, is_first=True))
        for _ in range(self.n_layers - 1):
            layers.append(Siren(dim_in=self.dim_hidden, dim_out=self.dim_hidden, w0=self.w0))
        layers.append(Siren(dim_in=self.dim_hidden, dim_out=self.dim_out, w0=self.w0, is_last=True))

        return layers

    def forward(self):
        x = self.mesh_grid.clone().detach().requires_grad_()
        for layer in self.layers:
            x = layer(x)

        x = rearrange(x, '(h w) c -> () c h w', h=self.image_size, w=self.image_size)
        return x
