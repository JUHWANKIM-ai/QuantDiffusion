import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightCrossAttention(nn.Module):
    """경량 Cross Attention 모듈 - 파라미터 효율적 설계

    timetransformer의 cross attention 패턴을 참고하되,
    파라미터를 최소화하기 위해 단일 head와 낮은 차원 사용.
    """

    def __init__(self, dim, num_heads=2, dropout=0.05, proj_ratio=0.5):
        super().__init__()
        # 파라미터 절약을 위해 내부 projection 차원 축소 (num_heads로 나누어떨어지게 보정)
        raw_inner = max(int(dim * proj_ratio), 16)
        self.inner_dim = (raw_inner + num_heads - 1) // num_heads * num_heads

        # Query, Key, Value projection (차원 축소)
        self.q_proj = nn.Linear(dim, self.inner_dim)
        self.k_proj = nn.Linear(dim, self.inner_dim)
        self.v_proj = nn.Linear(dim, self.inner_dim)

        # Output projection (원래 차원으로 복원)
        self.out_proj = nn.Linear(self.inner_dim, dim)

        self.num_heads = num_heads
        self.head_dim = self.inner_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        """
        Args:
            query: [B, T, D] - attention을 받을 대상
            key_value: [B, T_kv, D] - attention의 source
        Returns:
            [B, T, D] - query와 동일 shape
        """
        B, T, D = query.shape
        T_kv = key_value.shape[1]

        # Projection
        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(key_value)
            .view(B, T_kv, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(key_value)
            .view(B, T_kv, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Output
        out = (attn @ v).transpose(1, 2).reshape(B, T, self.inner_dim)
        out = self.out_proj(out)

        return out


class SequencePositionalEncoding(nn.Module):
    """시계열(시퀀스) 축 위치 인코딩. Attention이 선후 관계를 구분하도록 함.
    Sinusoidal (고정) 또는 학습 가능하지 않음 — 순수 위치 정보만 부여."""

    def __init__(self, d_model, max_len=5000, dropout=0.05):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: [B, T, D] → x + pos_enc[:, :T, :]"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class DualBranchCrossAttentionBlock(nn.Module):
    """TCN과 Self-Attention 브랜치 간 양방향 Cross Attention

    timetransformer.py의 TimesFormerLayer 구조를 참고하되,
    파라미터를 최소화하기 위해 간소화된 버전.
    use_sequence_pos_encoding=True 시 시퀀스(시간) 축 PE를 더해 선후 관계를 반영.
    """

    def __init__(
        self,
        dim,
        num_heads=2,
        dropout=0.05,
        max_seq_len=512,
        use_sequence_pos_encoding=True,
    ):
        super().__init__()
        self.use_sequence_pos_encoding = use_sequence_pos_encoding
        if use_sequence_pos_encoding:
            self.seq_pos_enc = SequencePositionalEncoding(dim, max_len=max_seq_len)
        else:
            self.seq_pos_enc = None

        # Self-attention for attention branch (경량화)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_self = nn.LayerNorm(dim)

        # Bidirectional Cross Attention (경량 버전)
        self.cross_attn_tcn2attn = LightweightCrossAttention(dim, num_heads, dropout)
        self.cross_attn_attn2tcn = LightweightCrossAttention(dim, num_heads, dropout)

    def forward(self, tcn_feat, attn_feat):
        # 1. Attention 브랜치에만 PE 적용 제안
        if self.seq_pos_enc is not None:
            attn_feat = self.seq_pos_enc(attn_feat)

        # 2. Self-attention (Causal Masking 권장)
        attn_out, _ = self.self_attn(attn_feat, attn_feat, attn_feat)
        attn_out = self.norm_self(attn_out + attn_feat)

        # 3. Cross Attention + Residual Connection
        # Attention 브랜치가 TCN(Local) 정보를 흡수
        attn_cross = self.cross_attn_tcn2attn(attn_out, tcn_feat) + attn_out

        # TCN 브랜치가 Attention(Global) 정보를 흡수
        tcn_cross = self.cross_attn_attn2tcn(tcn_feat, attn_out) + tcn_feat

        return tcn_cross, attn_cross


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


class ModernTCN(nn.Module):
    """TCN that takes x and timestep t as input and outputs x (same shape). Time conditioning like UNet.

    Cross Attention 옵션:
    - use_cross_attention: True로 설정 시 dual branch cross attention 활성화
    - cross_attention_layers: cross attention을 적용할 레이어 인덱스 리스트 (기본: 마지막 레이어만)
    - cross_attention_heads: cross attention head 수 (기본: 2, 파라미터 효율)
    """

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

        # Cross Attention 설정
        self.use_cross_attention = configs.get("use_cross_attention", False)
        num_layers = len(configs["dims"])
        # 기본값: 마지막 레이어에만 cross attention 적용 (파라미터 효율)
        self.cross_attention_layers = configs.get(
            "cross_attention_layers",
            [num_layers - 1] if self.use_cross_attention else [],
        )
        cross_attention_heads = configs.get("cross_attention_heads", 2)
        cross_attention_dropout = configs.get("cross_attention_dropout", 0.05)

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.time_projs = nn.ModuleList()
        self.cross_attention_blocks = nn.ModuleDict()  # 레이어 인덱스별 cross attention

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

            # 지정된 레이어에 Cross Attention 블록 추가
            if i in self.cross_attention_layers:
                self.cross_attention_blocks[str(i)] = DualBranchCrossAttentionBlock(
                    dim=configs["dims"][i],
                    num_heads=cross_attention_heads,
                    dropout=cross_attention_dropout,
                    max_seq_len=configs.get("max_seq_len", 512),
                    use_sequence_pos_encoding=configs.get(
                        "use_sequence_pos_encoding", True
                    ),
                )

            c_in = configs["dims"][i]

        # Attention branch 초기화용 projection (cross attention 사용 시)
        if self.use_cross_attention and len(self.cross_attention_layers) > 0:
            first_cross_layer = min(self.cross_attention_layers)
            if first_cross_layer == 0:
                self.attn_init_proj = nn.Linear(configs["enc_in"], configs["dims"][0])
            else:
                self.attn_init_proj = nn.Linear(
                    configs["dims"][first_cross_layer - 1],
                    configs["dims"][first_cross_layer],
                )
            # 다중 cross-attention 레이어: 이전 cross 레이어 dim → 현재 레이어 dim 투영
            sorted_cross = sorted(self.cross_attention_layers)
            self.attn_branch_projs = nn.ModuleDict()
            for j in range(1, len(sorted_cross)):
                prev_idx, curr_idx = sorted_cross[j - 1], sorted_cross[j]
                self.attn_branch_projs[str(curr_idx)] = nn.Linear(
                    configs["dims"][prev_idx], configs["dims"][curr_idx]
                )
            # 마지막 cross 레이어가 마지막 레이어가 아닐 때 fusion 전 차원 맞춤
            last_cross_idx = max(self.cross_attention_layers)
            if last_cross_idx != num_layers - 1:
                self.attn_to_fusion_proj = nn.Linear(
                    configs["dims"][last_cross_idx], configs["dims"][-1]
                )
            else:
                self.attn_to_fusion_proj = None
        else:
            self.attn_branch_projs = nn.ModuleDict()
            self.attn_to_fusion_proj = None

        # 출력 채널을 입력(enc_in)과 동일하게 맞춤 → 입출력 shape 동일 (B, seq, enc_in)
        # Cross attention 사용 시: TCN + Attention 브랜치 fusion
        if self.use_cross_attention and len(self.cross_attention_layers) > 0:
            # 두 브랜치 결합을 위한 fusion layer
            self.branch_fusion = nn.Conv1d(
                configs["dims"][-1] * 2, configs["dims"][-1], 1
            )

        self.out_proj = nn.Conv1d(configs["dims"][-1], self.enc_in, 1)
        # self.final_norm = nn.LayerNorm(configs["dims"][-1])

        # nn.init.zeros_(self.out_proj.weight)
        # if self.out_proj.bias is not None:
        #     nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, x, t
    ):  # x: [B, seq, channel], t: [B]. 세 번째 인자는 무시 (model(x,t) / model(x,t,None) 호환)
        x_orig = x  # attention branch 초기화용 보존
        x = x.permute(0, 2, 1)  # [B, C, T]

        attn_branch = None  # attention branch feature
        prev_tcn_feat = (
            None  # 이전 레이어 TCN 출력 (first_cross_layer > 0 일 때 attn 초기화용)
        )

        if self.use_time_emb:
            time_emb = self.time_mlp(t)  # [B, time_emb_dim]

            for i, (conv, norm, time_proj) in enumerate(
                zip(self.conv_layers, self.norm_layers, self.time_projs)
            ):
                x = conv(x)
                x = norm(x)
                x = x + time_proj(F.silu(time_emb))[:, :, None]
                x = F.relu(x)

                # Cross Attention 적용
                if str(i) in self.cross_attention_blocks:
                    # TCN feature를 [B, T, C] 형태로 변환
                    tcn_feat = x.transpose(1, 2)  # [B, T, C]

                    # Attention branch 초기화: attn_init_proj는 (이전 레이어 채널 -> 현재 레이어 채널)
                    if attn_branch is None:
                        if i == 0:
                            attn_branch = self.attn_init_proj(x_orig)
                        else:
                            # first_cross_layer > 0 이면 이전 레이어 출력(prev_tcn_feat)으로 초기화
                            attn_branch = self.attn_init_proj(prev_tcn_feat)
                    elif str(i) in self.attn_branch_projs:
                        # 다중 cross 레이어: 이전 cross 출력 차원 → 현재 레이어 차원으로 투영
                        attn_branch = self.attn_branch_projs[str(i)](attn_branch)

                    # Dual branch cross attention
                    tcn_feat, attn_branch = self.cross_attention_blocks[str(i)](
                        tcn_feat, attn_branch
                    )

                    # TCN feature를 다시 [B, C, T]로 변환
                    x = tcn_feat.transpose(1, 2)

                # 다음 레이어에서 cross attention 초기화 시 사용할 수 있도록 저장
                prev_tcn_feat = x.transpose(1, 2)  # [B, T, C]
        else:
            for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
                x = conv(x)
                x = norm(x)
                x = F.relu(x)

                # Cross Attention 적용 (time embedding 없는 경우)
                if str(i) in self.cross_attention_blocks:
                    tcn_feat = x.transpose(1, 2)

                    if attn_branch is None:
                        if i == 0:
                            attn_branch = self.attn_init_proj(x_orig)
                        else:
                            attn_branch = self.attn_init_proj(prev_tcn_feat)
                    elif str(i) in self.attn_branch_projs:
                        attn_branch = self.attn_branch_projs[str(i)](attn_branch)

                    tcn_feat, attn_branch = self.cross_attention_blocks[str(i)](
                        tcn_feat, attn_branch
                    )
                    x = tcn_feat.transpose(1, 2)

                # 다음 레이어에서 cross attention 초기화 시 사용할 수 있도록 저장
                prev_tcn_feat = x.transpose(1, 2)

        # Branch fusion (cross attention 사용 시)
        if self.use_cross_attention and attn_branch is not None:
            if self.attn_to_fusion_proj is not None:
                attn_branch = self.attn_to_fusion_proj(attn_branch)  # [B, T, dims[-1]]
            attn_feat = attn_branch.transpose(1, 2)  # [B, C, T]
            x = torch.cat([x, attn_feat], dim=1)  # [B, C*2, T]
            x = self.branch_fusion(x)  # [B, C, T]

        x = self.out_proj(x)  # [B, enc_in, seq]
        out = x.transpose(1, 2)  # [B, seq, enc_in] — 입력과 동일 shape
        return out
