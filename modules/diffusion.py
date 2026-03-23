import gc
import math
import torch
import numpy as np
from utils.utils import *
from copy import deepcopy
import torch.nn.functional as F
from torch.distributions import StudentT, Chi2
from utils.utils import inverse_scaling
from utils.file_manager import quantDiffusionPath

file_manager = quantDiffusionPath()
gc.collect()


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def sample_joint_student_t_noise(shape, df, device):
    """
    모든 자산(C)이 동일한 변동성 쇼크(V)를 공유하도록 샘플링.
    shape: (B, T, C)
    """
    B, T, C = shape
    # 1. 표준 가우시안 노이즈 생성 (B, T, C)
    z = torch.randn(shape, device=device)

    # 2. 공통 스케일링 인자 V 샘플링 (B, T, 1)
    # 각 시점(T)마다 모든 자산(C)이 하나의 V를 공유함
    # 주의: Chi2(df=길이1 벡터)는 batch_shape=(1,)이라 sample((B,T,1))이 (B,T,1,1)이 되어 z(B,T,C)와 브로드캐스트 불가.
    df_t = torch.as_tensor(df, dtype=torch.float32, device=device).reshape(())
    v = Chi2(df=df_t).sample((B, T, 1))

    # 3. Student-t 노이즈 합성 (분산 1 정규화 포함)
    # scale = sqrt((df-2)/df) * z / sqrt(v/df) => z * sqrt((df-2)/v)
    if df > 2:
        unit_noise = z * torch.sqrt((df - 2) / v)
    else:
        unit_noise = z * torch.sqrt(1.0 / v)  # df <= 2일 때 예외 처리

    return unit_noise


def sample_student_t_noise(shape, df, device, scale=None):
    """
    Student-t 노이즈 샘플링. df > 2일 때 scale=sqrt((df-2)/df)로 분산 1 맞춤.
    shape: (B, T, C) 등, df: 자유도 (권장 3~10), scale=None이면 단위분산.
    """
    if scale is None:
        # StudentT 분산 = df/(df-2), 단위분산이 되려면 scale = sqrt((df-2)/df)
        scale = 1.0 if df <= 2 else ((df - 2) / df) ** 0.5
    d = StudentT(df=float(df), loc=0.0, scale=scale)
    return d.sample(shape).to(device)


def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)


def cosine_schedule(t, start=0, end=1, tau=1, clamp_min=1e-9):
    """t: scalar 또는 tensor 모두 지원"""
    if not isinstance(t, torch.Tensor):
        t = torch.FloatTensor([t])
    v_start = math.cos(start * math.pi / 2) ** (2 * tau)
    v_end = math.cos(end * math.pi / 2) ** (2 * tau)
    gamma = torch.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
    gamma = (v_end - gamma) / (v_end - v_start)
    return gamma.clamp_(min=clamp_min, max=5.0)


def generate_cosine_schedule(T, tau, min_val, max_val):
    t_vals = torch.arange(1, T + 1, dtype=torch.float32) / T
    alphas_cumprod = cosine_schedule(t_vals, tau=tau)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1)
    alphas = alphas_cumprod / alphas_cumprod_prev
    betas = 1 - alphas
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    betas = betas * (max_val - min_val) + min_val
    return betas


class TimeSeriesGaussianDiffusion(torch.nn.Module):
    """시계열 (B, seq_len, C)용 Gaussian DDPM.
    prediction_type: "epsilon" (노이즈 ε) | "velocity" (v) | "x0" (clean x_0 직접 예측).
    """

    def __init__(
        self,
        model,
        seq_len,
        num_channels,
        betas,
        loss_type="l2",
        ema_decay=0.9999,
        ema_start=2000,
        ema_update_rate=1,
        sampling_type="gaussian",
        prediction_type="epsilon",
    ):
        super().__init__()

        self.model = model
        self.ema_model = deepcopy(model)
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.num_timesteps = len(betas)
        self.loss_type = loss_type
        self.sampling_type = sampling_type
        if prediction_type not in ("epsilon", "velocity", "x0"):
            raise ValueError("prediction_type must be 'epsilon', 'velocity', or 'x0'")
        self.prediction_type = prediction_type
        if sampling_type == "gaussian":
            self.perturb_x = self.perturb_x_gaussian
        elif sampling_type == "student_t":
            self.perturb_x = self.perturb_x_student_t
        elif sampling_type == "joint_student_t":
            self.perturb_x = self.perturb_x_joint_student_t
        else:
            raise ValueError(f"Invalid sample type: {sampling_type}")

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        to_torch = lambda x: torch.tensor(x, dtype=torch.float32)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod))
        )
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))
        self.register_buffer(
            "remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod))
        )
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0
        self._ema = lambda old, new: old * ema_decay + (1 - ema_decay) * new

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate != 0:
            return
        if self.step < self.ema_start:
            self.ema_model.load_state_dict(self.model.state_dict())
        else:
            for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data = self._ema(ema_p.data, p.data)

    def perturb_x_gaussian(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def perturb_x_student_t(self, x, t, noise):
        """
        Forward process with Student-t noise (heavy tails).
        x_t = sqrt(ᾱ_t)*x + sqrt(1-ᾱ_t)*noise, noise ~ StudentT(df).
        noise=None이면 내부에서 단위분산 Student-t 샘플링.
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def perturb_x_joint_student_t(self, x, t, noise):
        """
        Forward process with Student-t noise (heavy tails).
        x_t = sqrt(ᾱ_t)*x + sqrt(1-ᾱ_t)*noise, noise ~ StudentT(df).
        noise=None이면 내부에서 단위분산 Student-t 샘플링.
        """
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def get_losses(self, x, t):
        # 연속 t(float)가 넘어오면 버퍼 인덱싱용 정수 인덱스로 변환
        t_idx = (
            t.long().clamp(0, self.num_timesteps - 1) if t.is_floating_point() else t
        )
        if self.sampling_type == "gaussian":
            noise = torch.randn_like(x, device=x.device)
        elif self.sampling_type == "student_t":
            noise = sample_student_t_noise(x.shape, 5.0, x.device)
        elif self.sampling_type == "joint_student_t":
            noise = sample_joint_student_t_noise(x.shape, 5.0, x.device)
        else:
            raise ValueError(f"Invalid sample type: {self.sampling_type}")

        perturbed_x = self.perturb_x(x, t_idx, noise)
        model_out = self.model(perturbed_x, t_idx)

        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "velocity":
            target = (
                extract(self.sqrt_alphas_cumprod, t_idx, x.shape) * noise
                - extract(self.sqrt_one_minus_alphas_cumprod, t_idx, x.shape) * x
            )
        else:  # x0: clean data 직접 예측
            target = x

        if self.loss_type == "l2":
            main_loss = torch.nn.functional.mse_loss(model_out, target)
        else:
            main_loss = torch.nn.functional.smooth_l1_loss(model_out, target)

        alpha_bar = self.alphas_cumprod[t_idx]
        alpha_bar = alpha_bar.clamp(min=1e-5, max=0.99999)

        snr = alpha_bar / (1.0 - alpha_bar)
        min_snr_weights = torch.clamp(snr, max=10.0) / snr
        min_snr_weights = min_snr_weights.view(-1, 1, 1)

        # backward()는 스칼라에만 사용 가능하므로 배치 차원을 평균으로 축소
        weighted_loss = main_loss * min_snr_weights

        return weighted_loss.mean()

    def forward(self, x):
        b = x.shape[0]
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device, dtype=torch.long)
        # rand_like(Long) 미지원 오류 방지: float 랜덤 사용
        t_continuous = t.float() + torch.rand(b, device=device, dtype=torch.float32)
        return self.get_losses(x, t_continuous)

    @torch.no_grad()
    def remove_noise(self, x, t, use_ema=True):
        model = self.ema_model if use_ema else self.model
        out = model(x, t)
        sqrt_ab = extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_1_ab = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        if self.prediction_type == "epsilon":
            pred_noise = out
        elif self.prediction_type == "velocity":

            # 1. v로부터 x0를 먼저 역산
            maybe_x0 = sqrt_ab * x - sqrt_1_ab * out
            # 2. x0를 강제로 [-1, 1] 범위로 가둠 (이게 평균/표준편차 폭주를 막는 핵심)
            maybe_x0 = torch.clamp(maybe_x0, -3.0, 3.0)
            # 3. 정제된 x0로 다시 pred_noise 계산
            pred_noise = (x - sqrt_ab * maybe_x0) / (sqrt_1_ab + 1e-8)

            # ε = sqrt(α̅)*v + sqrt(1-α̅)*x_t
            # pred_noise = sqrt_ab * out + sqrt_1_ab * x
        else:  # x0: pred_x0 → ε = (x_t - sqrt(α̅_t)*x_0) / sqrt(1-α̅_t)
            pred_noise = (x - sqrt_ab * out) / (sqrt_1_ab + 1e-8)
        x_prev = (
            x - extract(self.remove_noise_coeff, t, x.shape) * pred_noise
        ) * extract(self.reciprocal_sqrt_alphas, t, x.shape)
        return x_prev

    @torch.no_grad()
    def sample(self, batch_size, device, use_ema=True, **kwargs):
        """
        sample_type: "gaussian" | "student_t" | "skewed_student_t" | "levy" | "nig"
        student_t kwargs: df=5.0, memory_scale=0.1
        skewed_student_t kwargs: df=5.0, skew_gamma=1.25, memory_scale=0.1
        levy kwargs: lambda_j=0.03, sigma_j=0.3
        nig kwargs: alpha=3.0, beta=2.0 (beta=alpha-1 시 단위분산)
        """
        if self.sampling_type == "gaussian":
            return self._sample_gaussian(batch_size, device, use_ema)
        if self.sampling_type == "student_t":
            return self._sample_student_t(batch_size, device, use_ema, **kwargs)
        if self.sampling_type == "joint_student_t":
            return self._sample_joint_student_t(batch_size, device, use_ema, **kwargs)
        raise ValueError(
            f"sample_type must be 'gaussian', 'student_t', 'skewed_student_t', 'levy', or 'nig', got {self.sampling_type!r}"
        )

    @torch.no_grad()
    def _sample_gaussian(self, batch_size, device, use_ema=True):
        x = torch.randn(batch_size, self.seq_len, self.num_channels, device=device)
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.remove_noise(x, t_batch, use_ema)
            if t > 0:
                x = x + extract(self.sigma, t_batch, x.shape) * torch.randn_like(
                    x, device=device
                )

        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-8)
        return x.cpu()

    @torch.no_grad()
    def _sample_student_t(
        self, batch_size, device, use_ema=True, df=5.0, memory_scale=0.1
    ):
        # 1. 초기 노이즈 분산 정규화
        std_norm = math.sqrt((df - 2) / df) if df > 2 else 1.0
        x = (
            sample_student_t_noise(
                (batch_size, self.seq_len, self.num_channels), df, device
            )
            * std_norm
        )

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_recon = self.remove_noise(x, t_batch, use_ema)

            if t > 0:
                z = sample_student_t_noise(x.shape, df, device) * std_norm

                # vol_adj가 필요하다면 아주 작게 설정하거나, 통계치가 우선이면 1.0으로 고정
                vol_adj = 1.0 + memory_scale * torch.abs(x - x_recon).mean(
                    dim=(1, 2), keepdim=True
                )

                sigma_t = extract(self.sigma, t_batch, x.shape)
                x = x_recon + (sigma_t * vol_adj * z)
            else:
                x = x_recon

        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-8)
        return x.cpu()

    @torch.no_grad()
    def _sample_joint_student_t(
        self, batch_size, device, use_ema=True, df=5.0, memory_scale=0.1
    ):
        """
        역확산 과정 수정 사항:
        1. sample_joint_student_t_noise 사용 (동조화 반영)
        2. vol_adj 가중치 최적화 (VaR 에러 대응)
        3. 마지막 강제 std 정규화 제거 (Tail 보존)
        """
        # 1. 분산 1을 맞추기 위한 정규화 인자
        std_norm = math.sqrt((df - 2) / df) if df > 2 else 1.0

        # 2. 초기 노이즈 (t=T) 생성: 모든 자산이 동일한 V를 공유하도록 생성
        # shape: (B, T, C)
        x = sample_joint_student_t_noise(
            (batch_size, self.seq_len, self.num_channels), df, device
        )

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # 현재 x에서 노이즈 성분을 제거하여 원본 x_0을 추정 (Reconstruction)
            x_recon = self.remove_noise(x, t_batch, use_ema)

            if t > 0:
                # [핵심] 매 스텝마다 모든 자산이 공유하는 '공통 충격' 노이즈 주입
                z = sample_joint_student_t_noise(x.shape, df, device)

                # vol_adj: 모델의 예측 오차가 큰 구간(불확실성)에서 노이즈를 증폭
                # VaR 에러가 높다면 memory_scale을 0.2~0.3까지 시도해 보세요.
                vol_adj = 1.0 + memory_scale * torch.abs(x - x_recon).mean(
                    dim=(1, 2), keepdim=True
                )

                sigma_t = extract(self.sigma, t_batch, x.shape)
                x = x_recon + (sigma_t * vol_adj * z)
            else:
                x = x_recon

        # --- [중요] 마지막 단계 수정 ---
        # x.mean() 제거는 유지하되, x.std()로 나누는 것은 주석 처리합니다.
        # 꼬리 동조화 시나리오는 표준편차가 큰 것이 정상입니다.
        # 이를 강제로 1로 맞추면 VaR 임계값이 가우시안 쪽으로 당겨져 에러가 납니다.

        x = x - x.mean(dim=1, keepdim=True)

        # 만약 전체 스케일이 너무 튄다면, std() 대신 사전에 계산된
        # GARCH의 '장기 평균 변동성' 등으로 스케일링하는 것이 훨씬 안전합니다.
        x = x / (x.std(dim=1, keepdim=True) + 1e-8)

        return x.cpu()
