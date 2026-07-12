import torch
import torch.nn as nn
from src.baselines.networks.tcn import *
from torch.nn.utils import spectral_norm

class TCNGenerator(nn.Module):
    def __init__(self, config, step_size=16):
        super(TCNGenerator, self).__init__()
        self.step_size = step_size
        self.hidden_dim = config.hidden_dim

        self.tcn = nn.ModuleList([
            TemporalBlock(config.noise_dim, config.hidden_dim, kernel_size=1, stride=1, dilation=1, padding=0, dropout=config.G_dropout),            
            *[TemporalBlock(config.hidden_dim, config.hidden_dim, kernel_size=2, stride=1, dilation=i, padding=i, dropout=config.G_dropout) for i in [1, 2, 4]]
        ])

        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=4,
            dropout=config.G_dropout,
            batch_first=True
        )
   
        self.ln_attn_out = nn.LayerNorm(config.hidden_dim)
        self.ln_ffn_out = nn.LayerNorm(config.hidden_dim)

        self.final_ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        self.dropout = nn.Dropout(config.G_dropout)
        self.last = nn.Linear(config.hidden_dim, 1)        


    def forward(self, z):        
        for layer in self.tcn:
            z = layer(z)  # (B, hidden_dim, T)

        B, C, T = z.shape

        z_attn = z.view(B, C, -1, self.step_size)    # (B, C, S_, step_size)
        z_attn = z_attn.permute(0, 3, 2, 1).contiguous()  # (B, step_size, S_, C)

        B_ = B * self.step_size
        S_ = T // self.step_size
        z_attn = z_attn.view(B_, S_, C)              # (B_, S_, C)

        positions = torch.arange(S_, device=z_attn.device).unsqueeze(1)  
        attn_mask = (positions - positions.transpose(0, 1) < 0) | (positions - positions.transpose(0, 1) >= 8)

        attn_output, _ = self.self_attention(
            z_attn, 
            z_attn, 
            z_attn, 
            attn_mask=attn_mask
        )

        z_attn =  z_attn + self.dropout(attn_output)
        z_attn = self.ln_attn_out(z_attn)

        ffn_out = self.final_ffn(z_attn)      # (B_, S_, C)
        ffn_out = self.dropout(ffn_out)
        z_attn = z_attn + ffn_out           
        z_attn = self.ln_ffn_out(z_attn)    

        z_attn = z_attn.view(B, self.step_size, S_, C)    # (B, step_size, S_, C)
        z_attn = z_attn.permute(0, 3, 2, 1).contiguous()  # (B, C, S_, step_size)
        z_attn = z_attn.view(B, C, T)                     # (B, C, T)        

        out = z_attn.permute(0, 2, 1)  # (B, T, C)
        out = self.last(out)          # (B, T, 1)
        out = out.permute(0, 2, 1)    # (B, 1, T) 

        return out

    
# class TCNGenerator(nn.Module):
#     def __init__(self, config, step_size=16):
#         super(TCNGenerator, self).__init__()
#         self.step_size = step_size
#         self.hidden_dim = config.hidden_dim

#         self.tcn = nn.ModuleList([
#             TemporalBlock(config.noise_dim, config.hidden_dim, kernel_size=1, stride=1, dilation=1, padding=0, dropout=config.G_dropout),
#             *[TemporalBlock(config.hidden_dim, config.hidden_dim, kernel_size=2, stride=1, dilation=i, padding=i, dropout=config.G_dropout) for i in [1, 2, 4, 8, 16, 32]]
#         ])

#         self.last = nn.Linear(config.hidden_dim, 1)        


#     def forward(self, z):        
#         for layer in self.tcn:
#             z = layer(z)  # (B, hidden_dim, T)

#         B, C, T = z.shape    # (B, C, T)        

#         out = z.permute(0, 2, 1)  # (B, T, C)
#         out = self.last(out)          # (B, T, 1)
#         out = out.permute(0, 2, 1)    # (B, 1, T) 

#         return out


