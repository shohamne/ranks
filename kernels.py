import torch
from torch import nn

class GaussianKernel(nn.Module):
    def __init__(self, length_scale, sigma_f):
        super().__init__()
        self.length_scale = nn.Parameter(torch.tensor(length_scale))
        self.sigma_f = nn.Parameter(torch.tensor(sigma_f))
    
    def forward(self, x1, x2=None):
        if x2 == None:
            x2 = x1
        D = (torch.cdist(x1, x2)/self.length_scale) ** 2 
        K = self.sigma_f**2 * torch.exp(-D / 2)
        return K

class LinearKernel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2=None):
        if x2 == None:
            x2 = x1
        return x1 @ x2.T