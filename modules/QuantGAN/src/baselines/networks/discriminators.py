import torch
from torch import nn
from torch.nn.utils import spectral_norm
from src.baselines.networks.tcn import TemporalBlock

class UserDiscriminator(nn.Module):
    def __init__(self, config):
        super(UserDiscriminator, self).__init__()
        self.tcn = nn.ModuleList([
            TemporalBlock(config.n_vars, config.D_hidden_dim, kernel_size=1, stride=1, dilation=1, padding=0, spec_norm=False, n_steps=config.n_steps),
            *[TemporalBlock(config.D_hidden_dim, config.D_hidden_dim, kernel_size=2, stride=1, dilation=i, padding=i, spec_norm=False, n_steps=config.n_steps) for i in [1, 2, 4, 8, 16, 32]]
        ])
        self.last = spectral_norm(nn.Conv1d(config.D_hidden_dim, 1, kernel_size=1, dilation=1))
        self.to_prob = nn.Sequential(
            nn.Linear(config.n_steps, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return self.to_prob(x).squeeze()

