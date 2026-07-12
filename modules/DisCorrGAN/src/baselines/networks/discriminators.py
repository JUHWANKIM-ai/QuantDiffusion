import torch
from torch import nn
import torch.nn.utils as utils
from src.baselines.networks.tcn import *
from torch.nn.utils import spectral_norm

    
class TCNDiscriminator(nn.Module):
    def __init__(self, config, step_size=16):
        super(TCNDiscriminator, self).__init__()
        self.step_size = step_size
        
        self.tcn = nn.ModuleList([
            TemporalBlock(1, config.D_hidden_dim, kernel_size=1, stride=1, dilation=1, padding=0, dropout=config.D_dropout),            
            *[TemporalBlock(config.D_hidden_dim, config.D_hidden_dim, kernel_size=2, stride=1, dilation=i, padding=i, dropout=config.D_dropout) for i in [1, 2, 4]]
        ])

        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.D_hidden_dim,
            num_heads=4,
            dropout=config.D_dropout,
            batch_first=True
        )

        self.ln_attn_out = nn.LayerNorm(config.D_hidden_dim)
        self.ln_ffn_out = nn.LayerNorm(config.D_hidden_dim)

        self.dropout = nn.Dropout(config.D_dropout)

        self.final_ffn = nn.Sequential(
            nn.Linear(config.D_hidden_dim, config.D_hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.D_hidden_dim * 2, config.D_hidden_dim),
        )
        
        self.last = nn.Linear(config.D_hidden_dim, 1)
        self.output = nn.Sequential(nn.Linear(config.n_steps, 1))

    def forward(self, x):
        for layer in self.tcn:
            x = layer(x)  # (B, hidden_dim, T)        

        B, C, T = x.shape

        x_attn = x.view(B, C, -1, self.step_size)    # (B, C, S_, step_size)
        x_attn = x_attn.permute(0, 3, 2, 1).contiguous()  # (B, step_size, S_, C)

        B_ = B * self.step_size
        S_ = T // self.step_size
        x_attn = x_attn.view(B_, S_, C)  # (B_, S_, C)        


        positions = torch.arange(S_, device=x_attn.device).unsqueeze(1)
        attn_mask = (positions - positions.transpose(0, 1) < 0) | (positions - positions.transpose(0, 1) >= 8)

        attn_output, _ = self.self_attention(
            x_attn, x_attn, x_attn,
            attn_mask=attn_mask
        )
        x_attn = x_attn + self.dropout(attn_output)
        x_attn = self.ln_attn_out(x_attn)

        ffn_out = self.final_ffn(x_attn)
        ffn_out = self.dropout(ffn_out)
        x_attn = x_attn + ffn_out
        x_attn = self.ln_ffn_out(x_attn)

        x_attn = x_attn.view(B, self.step_size, S_, C)      # (B, step_size, S_, C)
        x_attn = x_attn.permute(0, 3, 2, 1).contiguous()    # (B, C, S_, step_size)
        x_attn = x_attn.view(B, C, T)                       # (B, C, T)        
        
        out = x_attn.permute(0, 2, 1)  # (B, T, C)
        out = self.last(out)          # (B, T, 1)
        out = out.permute(0, 2, 1)    # (B, 1, T)

        return self.output(out.squeeze()).reshape(-1, 1)



# class TCNDiscriminator(nn.Module):
#     def __init__(self, config, step_size=16):
#         super(TCNDiscriminator, self).__init__()
#         self.step_size = step_size
        
#         self.tcn = nn.ModuleList([
#             TemporalBlock(1, config.D_hidden_dim, kernel_size=1, stride=1, dilation=1, padding=0, dropout=config.D_dropout, spec_norm=False),
#             *[TemporalBlock(config.D_hidden_dim, config.D_hidden_dim, kernel_size=2, stride=1, dilation=i, padding=i, dropout=config.D_dropout, spec_norm=False) for i in [1, 2, 4, 8, 16, 32]]
#         ])
        
#         self.last = nn.Linear(config.D_hidden_dim, 1)
#         self.output = nn.Sequential(nn.Linear(config.n_steps, 1))

#     def forward(self, x):
#         for layer in self.tcn:
#             x = layer(x)  # (B, hidden_dim, T)        

#         B, C, T = x.shape
        
#         out = x.permute(0, 2, 1)  # (B, T, C)
#         out = self.last(out)          # (B, T, 1)
#         out = out.permute(0, 2, 1)    # (B, 1, T)

#         return self.output(out.squeeze()).reshape(-1, 1)

