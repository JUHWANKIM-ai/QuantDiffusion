import torch
import torch.nn as nn
from src.baselines.networks.tcn import *

class UserGenerator(nn.Module):
    def __init__(self, config):
        super(UserGenerator, self).__init__()
        
        self.tcn = nn.ModuleList([
            TemporalBlock(config.noise_dim, config.G_hidden_dim, kernel_size=1, stride=1, dilation=1, padding=0, n_steps=config.n_steps),
            *[TemporalBlock(config.G_hidden_dim, config.G_hidden_dim, kernel_size=2, stride=1, dilation=i, padding=i, n_steps=config.n_steps) for i in [1, 2, 4, 8, 16, 32]]
        ])
        self.last = nn.Sequential(            
            nn.Conv1d(config.G_hidden_dim, config.n_vars, kernel_size=1, stride=1),            
            
        )

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return x