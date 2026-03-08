import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PositionalEmbedding(nn.Module):
    """Timestep positional embedding (same as UNet). Input: (N,), Output: (N, dim)."""

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation,
                     groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv',
                      get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
    result.add_module('bn', get_bn(out_channels))
    return result


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1:, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight[None, None, :] + self.affine_bias[None, None, :]
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias[None, None, :]) / self.affine_weight[None, None, :]
        x = x * self.stdev + self.mean
        return x


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, small_kernel, small_kernel_merged=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups,
                                         bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups)
            if small_kernel is not None:
                self.small_conv = conv_bn(in_channels, out_channels, small_kernel, stride, small_kernel // 2, groups)

    def forward(self, x):
        if hasattr(self, 'lkb_reparam'):
            return self.lkb_reparam(x)
        out = self.lkb_origin(x)
        if hasattr(self, 'small_conv'):
            out += self.small_conv(x)
        return out


class DilatedReparamConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, small_kernel, 
                 small_kernel_merged=False, layer_idx=0):
        super().__init__()
        
        # 1. Dilation 스케줄 고정 (사용자가 바꿀 수 없음)
        # 레이어 인덱스에 따라 1, 2, 4, 8, 16, 32... 순으로 할당
        dilation_schedule = [1, 2, 4, 8, 16, 32, 64]
        self.dilation = dilation_schedule[min(layer_idx, len(dilation_schedule)-1)]
        
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        
        # 2. Padding 자동 계산: (kernel_size - 1) * dilation // 2
        # Dilation이 적용되어도 시퀀스 길이를 유지하기 위함
        padding = (kernel_size - 1) * self.dilation // 2
        
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(
                in_channels, out_channels, kernel_size, stride, 
                padding, dilation=self.dilation, groups=groups, bias=True
            )
        else:
            # lkb_origin에 dilation 적용
            self.lkb_origin = conv_bn(
                in_channels, out_channels, kernel_size, stride, 
                padding, groups, dilation=self.dilation
            )
            
            if small_kernel is not None:
                # small_conv에도 동일한 dilation 적용 (또는 1로 고정 선택 가능)
                small_padding = (small_kernel - 1) * 1 // 2
                self.small_conv = conv_bn(
                    in_channels, out_channels, small_kernel, stride, 
                    small_padding, groups, dilation=1
                )

    def forward(self, x):
        if hasattr(self, 'lkb_reparam'):
            return self.lkb_reparam(x)
        out = self.lkb_origin(x)
        if hasattr(self, 'small_conv'):
            out += self.small_conv(x)
        return out


# class Flatten_Head(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.linear = nn.Linear(d_model, d_model)
#
#     def forward(self, x):         # x: [B, C, T]
#         x = x.permute(0, 2, 1)    # → [B, T, C]
#         x = self.linear(x)        # → [B, T, 1]
#         return x


class ModernTCN(nn.Module):
    """TCN that takes x and timestep t as input and outputs x (same shape). Time conditioning like UNet."""

    def __init__(self, configs):
        super().__init__()
        self.enc_in = configs["enc_in"]
        self.revin = RevIN(self.enc_in, affine=configs.get("affine", True)) if configs.get("revin", False) else None
        self.decomp = series_decomp(configs["kernel_size"]) if configs.get("decomposition", False) else None

        self.use_time_emb = configs.get("use_time_emb", True)
        time_emb_dim = configs.get("time_emb_dim", 64)
        emb_dim = 64  # even dim for PositionalEmbedding
        if self.use_time_emb:
            self.time_mlp = nn.Sequential(
                PositionalEmbedding(emb_dim, scale=1.0),
                nn.Linear(emb_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        else:
            self.time_mlp = None

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.time_projs = nn.ModuleList()

        c_in = configs["enc_in"]
        for i in range(len(configs["dims"])):
            # conv = ReparamLargeKernelConv(
            #     c_in, configs["dims"][i],
            #     kernel_size=configs["large_size"][i],
            #     stride=1,
            #     groups=1,
            #     small_kernel=configs["small_size"][i],
            #     small_kernel_merged=configs["small_kernel_merged"]
            # )

            conv = DilatedReparamConv(
                c_in, configs["dims"][i],
                kernel_size=configs["large_size"][i],
                stride=1,
                groups=1,
                small_kernel=configs["small_size"][i],
                small_kernel_merged=configs["small_kernel_merged"],
                layer_idx=i
            )


            self.conv_layers.append(conv)
            self.norm_layers.append(nn.BatchNorm1d(configs["dims"][i]))
            if self.use_time_emb:
                self.time_projs.append(nn.Linear(time_emb_dim, configs["dims"][i]))
            c_in = configs["dims"][i]

        # 출력 채널을 입력(enc_in)과 동일하게 맞춤 → 입출력 shape 동일 (B, seq, enc_in)
        self.out_proj = nn.Conv1d(configs["dims"][-1], self.enc_in, 1)

    def forward(self, x, t):  # x: [B, seq, channel], t: [B]
        if self.revin:
            x = self.revin(x, 'norm')
        if self.decomp:
            x, _ = self.decomp(x)
        x = x.permute(0, 2, 1)  # [B, C, T]

        if self.use_time_emb:
            time_emb = self.time_mlp(t)  # [B, time_emb_dim]
            for conv, norm, time_proj in zip(self.conv_layers, self.norm_layers, self.time_projs):
                x = conv(x)
                x = norm(x)
                x = x + time_proj(F.silu(time_emb))[:, :, None]
                x = F.relu(x)
        else:
            for conv, norm in zip(self.conv_layers, self.norm_layers):
                x = conv(x)
                x = norm(x)
                x = F.relu(x)

        x = self.out_proj(x)  # [B, enc_in, seq]
        out = x.transpose(1, 2)  # [B, seq, enc_in] — 입력과 동일 shape
        return out

