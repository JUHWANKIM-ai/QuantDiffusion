"""QuantGAN TCN 구조를 diffusion 모델용으로 차용한 단일 파일.

원본: scratch/QuantGAN/src/baselines/networks/tcn.py, generators.py
 - Chomp1d 기반 causal dilated convolution
 - TemporalBlock: 2개의 dilated conv + BatchNorm + LeakyReLU + residual, skip 출력 반환
 - 여러 TemporalBlock을 dilation을 키워가며 쌓고, skip 출력을 합산 후 1x1 conv

diffusion 적용을 위한 변경점:
 - 입출력 채널을 동일(enc_in)하게 맞춰 (B, seq, C) -> (B, seq, C) 형태로 동작
 - timestep t에 대한 sinusoidal 임베딩을 각 TemporalBlock에 주입 (UNet 방식)
 - 모델 호출 규약: forward(x, t)  (diffusion.py의 self.model(x, t)와 호환)
"""

import math
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class PositionalEmbedding(nn.Module):
    """Timestep positional embedding. Input: (N,), Output: (N, dim)."""

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
        emb = torch.outer(x.float() * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Chomp1d(nn.Module):
    """오른쪽 padding을 잘라 causal convolution을 유지 (QuantGAN 원본과 동일)."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """QuantGAN TemporalBlock + timestep 임베딩 주입.

    forward는 (skip, out)을 반환한다. skip은 generator의 skip-connection 합산에,
    out은 다음 블록 입력으로 전달된다.
    """

    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        time_emb_dim=None,
        spec_norm=False,
        dropout=0.2,
    ):
        super().__init__()

        # 첫 번째 convolution block
        conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.conv1 = spectral_norm(conv1) if spec_norm else conv1
        self.chomp1 = None if padding == 0 else Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        # timestep 임베딩 -> 채널별 bias (UNet 방식)
        self.time_proj = (
            nn.Linear(time_emb_dim, n_outputs) if time_emb_dim is not None else None
        )

        # 두 번째 convolution block
        conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.conv2 = spectral_norm(conv2) if spec_norm else conv2
        self.chomp2 = None if padding == 0 else Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        # residual connection (채널 수가 다르면 1x1 conv로 맞춤)
        self.downsample = None
        if n_inputs != n_outputs:
            self.downsample = nn.Sequential(
                nn.Conv1d(n_inputs, n_outputs, kernel_size=1),
                nn.BatchNorm1d(n_outputs),
            )
        self.relu = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="leaky_relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample[0].weight, nonlinearity="leaky_relu")

    def forward(self, x, time_emb=None):
        # 첫 번째 conv block
        out = self.conv1(x)
        if self.chomp1 is not None:
            out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        # timestep 조건 주입: (B, n_outputs) -> (B, n_outputs, 1) 브로드캐스트
        if self.time_proj is not None and time_emb is not None:
            out = out + self.time_proj(time_emb)[:, :, None]
        out = self.dropout1(out)

        # 두 번째 conv block
        out = self.conv2(out)
        if self.chomp2 is not None:
            out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # residual connection
        res = x if self.downsample is None else self.downsample(x)
        return out, self.relu(out + res)


class QuantTCN(nn.Module):
    """QuantGAN generator 구조를 차용한 diffusion noise/velocity/x0 예측 네트워크.

    configs (dict) 키:
      enc_in         : 입력=출력 채널 수 (필수)
      hidden_dim     : TCN 내부 채널 수 (기본 80)
      dilations      : dilated block들의 dilation 리스트 (기본 [1, 2, 4, 8, 16, 32])
      kernel_size    : dilated block kernel 크기 (기본 2)
      dropout        : dropout 비율 (기본 0.2)
      spec_norm      : spectral norm 사용 여부 (기본 False)
      use_time_emb   : timestep 임베딩 사용 여부 (기본 True)
      time_emb_dim   : timestep 임베딩 차원 (기본 64)

    forward(x, t):
      x: [B, seq, enc_in], t: [B] -> [B, seq, enc_in]
    """

    def __init__(self, configs):
        super().__init__()
        self.enc_in = configs["enc_in"]
        hidden_dim = configs.get("hidden_dim", 80)
        dilations = configs.get("dilations", [1, 2, 4, 8, 16, 32])
        kernel_size = configs.get("kernel_size", 2)
        dropout = configs.get("dropout", 0.2)
        spec_norm = configs.get("spec_norm", False)

        self.use_time_emb = configs.get("use_time_emb", True)
        time_emb_dim = configs.get("time_emb_dim", 64)
        block_time_dim = time_emb_dim if self.use_time_emb else None

        if self.use_time_emb:
            self.time_mlp = nn.Sequential(
                PositionalEmbedding(time_emb_dim, scale=1.0),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        else:
            self.time_mlp = None

        # 첫 블록: enc_in -> hidden, kernel=1 (non-causal point-wise, padding=0)
        blocks = [
            TemporalBlock(
                self.enc_in,
                hidden_dim,
                kernel_size=1,
                stride=1,
                dilation=1,
                padding=0,
                time_emb_dim=block_time_dim,
                spec_norm=spec_norm,
                dropout=dropout,
            )
        ]
        # 이후 블록들: hidden -> hidden, dilation을 키우며 causal conv
        for d in dilations:
            blocks.append(
                TemporalBlock(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=d,
                    padding=(kernel_size - 1) * d,  # causal padding (Chomp로 우측 제거)
                    time_emb_dim=block_time_dim,
                    spec_norm=spec_norm,
                    dropout=dropout,
                )
            )
        self.tcn = nn.ModuleList(blocks)

        # skip 합산 결과를 출력 채널(enc_in)로 투영
        self.last = nn.Conv1d(hidden_dim, self.enc_in, kernel_size=1, stride=1)

    def forward(self, x, t):
        # x: [B, seq, C] -> [B, C, seq]
        x = x.permute(0, 2, 1)

        time_emb = self.time_mlp(t) if self.use_time_emb else None

        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x, time_emb)
            skip_layers.append(skip)

        x = self.last(x + sum(skip_layers))  # [B, C, seq]
        return x.transpose(1, 2)  # [B, seq, C]
