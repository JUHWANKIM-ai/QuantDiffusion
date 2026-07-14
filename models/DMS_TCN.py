import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def get_conv1d(
    in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
):
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    groups,
    dilation=1,
    bias=False,
):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module(
        "conv",
        get_conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        ),
    )
    result.add_module("bn", get_bn(out_channels))
    return result


class DilatedReparamConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups,
        small_kernel,
        layer_idx=0,
        dropout=0.05,  # Dropout 인자 추가 (기본값 0.1)
    ):
        super().__init__()

        # 1. Dilation 스케줄 고정
        dilation_schedule = [1, 4, 16]
        self.dilation = dilation_schedule[min(layer_idx, len(dilation_schedule) - 1)]

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel

        # 2. Padding 자동 계산
        padding = (kernel_size - 1) * self.dilation // 2

        # lkb_origin에 dilation 적용
        self.lkb_origin = conv_bn(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            dilation=self.dilation,
        )

        if small_kernel is not None:
            # small_conv에는 dilation=1 적용
            small_padding = (small_kernel - 1) * 1 // 2
            self.small_conv = conv_bn(
                in_channels,
                out_channels,
                small_kernel,
                stride,
                small_padding,
                groups,
                dilation=1,
            )

        # Dropout 레이어 추가
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if hasattr(self, "small_conv"):
                out += self.small_conv(x)

        # 최종 출력 전 Dropout 적용
        return self.dropout(out)


class DMSTCN(nn.Module):
    """TCN that takes x and timestep t as input and outputs x (same shape). Time conditioning like UNet."""

    def __init__(self, configs):
        super().__init__()
        self.enc_in = configs["enc_in"]

        self.use_time_emb = configs.get("use_time_emb", True)
        time_emb_dim = configs.get("time_emb_dim", 64)

        if self.use_time_emb:
            self.time_mlp = nn.Sequential(
                PositionalEmbedding(time_emb_dim, scale=1.0),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        else:
            self.time_mlp = None

        num_layers = len(configs["dims"])

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.time_projs = nn.ModuleList()

        c_in = configs["enc_in"]
        for i in range(num_layers):
            conv = DilatedReparamConv(
                c_in,
                configs["dims"][i],
                kernel_size=configs["large_size"][i],
                stride=1,
                groups=1,
                small_kernel=configs["small_size"][i],
                layer_idx=i,
            )

            self.conv_layers.append(conv)
            self.norm_layers.append(nn.BatchNorm1d(configs["dims"][i]))
            if self.use_time_emb:
                self.time_projs.append(nn.Linear(time_emb_dim, configs["dims"][i]))

            c_in = configs["dims"][i]

        # 출력 채널을 입력(enc_in)과 동일하게 맞춤 → 입출력 shape 동일 (B, seq, enc_in)
        self.out_proj = nn.Conv1d(configs["dims"][-1], self.enc_in, 1)

    def forward(
        self, x, t
    ):  # x: [B, seq, channel], t: [B]. 세 번째 인자는 무시 (model(x,t) / model(x,t,None) 호환)
        x = x.permute(0, 2, 1)  # [B, C, T]

        if self.use_time_emb:
            time_emb = self.time_mlp(t)  # [B, time_emb_dim]

            for conv, norm, time_proj in zip(
                self.conv_layers, self.norm_layers, self.time_projs
            ):
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
