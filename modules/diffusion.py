import gc
import torch
import numpy as np
from utils.utils import *
from copy import deepcopy
import torch.nn.functional as F
from torch.distributions import StudentT
from utils.file_manager import quantDiffusionPath

file_manager = quantDiffusionPath()
gc.collect()


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


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


def generate_scaled_linear_schedule(T, low, high):
    return (np.linspace(low**0.5, high**0.5, T))**2


def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    
    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
    
    return np.array(betas)



class TimeSeriesGaussianDiffusion(torch.nn.Module):
    """시계열 (B, seq_len, C)용 Gaussian DDPM. model(x, t)가 노이즈 예측."""
    def __init__(self, model, seq_len, num_channels, betas, loss_type="l2", ema_decay=0.9999, ema_start=2000, ema_update_rate=1, sampling_type="gaussian"):
        super().__init__()
        self.model = model
        self.ema_model = deepcopy(model)
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.num_timesteps = len(betas)
        self.loss_type = loss_type
        self.sampling_type = sampling_type
        if sampling_type == "gaussian":
            self.perturb_x = self.perturb_x_gaussian
        elif sampling_type == "student_t":
            self.perturb_x = self.perturb_x_student_t
        elif sampling_type == "levy":
            self.perturb_x = self.perturb_x_levy
        else:
            raise ValueError(f"Invalid sample type: {sampling_type}")

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        to_torch = lambda x: torch.tensor(x, dtype=torch.float32)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))
        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
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

    def perturb_x_student_t(self, x, t, noise=None, df=5.0):
        """
        Forward process with Student-t noise (heavy tails).
        x_t = sqrt(ᾱ_t)*x + sqrt(1-ᾱ_t)*noise, noise ~ StudentT(df).
        noise=None이면 내부에서 단위분산 Student-t 샘플링.
        """
        if noise is None:
            noise = sample_student_t_noise(x.shape, df, x.device)
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def perturb_x_levy(self, x, t, noise=None, lambda_j=0.05, sigma_j=0.5):
        """
        Itô-Lévy Forward Process (Jump-Diffusion).
        lambda_j: 점프 발생 빈도 (Intensity)
        sigma_j: 점프의 크기 (Jump magnitude)
        """
        # 1. 가우시안 확산 부분 (Standard DDPM)
        if noise is None:
            noise = torch.randn_like(x)
        
        x_t_gaussian = (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )
        
        # 2. 레비 점프 부분 (Compound Poisson Process)
        # 각 시점/배치마다 점프 여부를 결정하는 포아송 마스크
        jump_mask = (torch.rand_like(x) < lambda_j).float()
        jump_size = torch.randn_like(x) * sigma_j
        jumps = jump_mask * jump_size
        
        # 시간에 따라 점프 영향력도 누적 확산에 맞춰 조절
        return x_t_gaussian + extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * jumps

    def get_losses(self, x, t):
        noise = torch.randn_like(x, device=x.device)
        perturbed_x = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, t)

        if self.loss_type == "l2":
            main_loss = torch.nn.functional.mse_loss(estimated_noise, noise)
        else:
            main_loss = torch.nn.functional.l1_loss(estimated_noise, noise)
                    
        return main_loss

    def forward(self, x):
        b = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device, dtype=torch.long)
        return self.get_losses(x, t)

    @torch.no_grad()
    def remove_noise(self, x, t, use_ema=True):
        model = self.ema_model if use_ema else self.model
        pred_noise = model(x, t)
        x_prev = (x - extract(self.remove_noise_coeff, t, x.shape) * pred_noise) * extract(self.reciprocal_sqrt_alphas, t, x.shape)
        return x_prev

    @torch.no_grad()
    def sample(self, batch_size, device, use_ema=True, **kwargs):
        """
        sample_type: "gaussian" | "student_t" | "levy"
        student_t kwargs: df=5.0, memory_scale=0.1
        levy kwargs: lambda_j=0.03, sigma_j=0.3
        """
        if self.sampling_type == "gaussian":
            return self._sample_gaussian(batch_size, device, use_ema)
        if self.sampling_type == "student_t":
            return self._sample_student_t(batch_size, device, use_ema, **kwargs)
        if self.sampling_type == "levy":
            return self._sample_levy(batch_size, device, use_ema, **kwargs)
        raise ValueError(f"sample_type must be 'gaussian', 'student_t', or 'levy', got {self.sampling_type!r}")

    @torch.no_grad()
    def _sample_gaussian(self, batch_size, device, use_ema=True):
        x = torch.randn(batch_size, self.seq_len, self.num_channels, device=device)
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.remove_noise(x, t_batch, use_ema)
            if t > 0:
                x = x + extract(self.sigma, t_batch, x.shape) * torch.randn_like(x, device=device)
        return x.cpu()

    @torch.no_grad()
    def _sample_student_t(self, batch_size, device, use_ema=True, df=5.0, memory_scale=0.1):
        x = sample_student_t_noise((batch_size, self.seq_len, self.num_channels), df, device)
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_recon = self.remove_noise(x, t_batch, use_ema)
            if t > 0:
                z = sample_student_t_noise(x.shape, df, device)
                vol_adj = 1.0 + memory_scale * torch.abs(x - x_recon).mean(dim=(1, 2), keepdim=True)
                sigma_t = extract(self.sigma, t_batch, x.shape)
                x = x_recon + (sigma_t * vol_adj * z)
            else:
                x = x_recon
        return x.cpu()

    @torch.no_grad()
    def _sample_levy(self, batch_size, device, use_ema=True, lambda_j=0.03, sigma_j=0.3):
        x = torch.randn(batch_size, self.seq_len, self.num_channels, device=device) * 2.0
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_recon = self.remove_noise(x, t_batch, use_ema)
            if t > 0:
                z_g = torch.randn_like(x)
                jump_mask = (torch.rand_like(x) < lambda_j).float()
                z_j = jump_mask * torch.randn_like(x) * sigma_j
                sigma_t = extract(self.sigma, t_batch, x.shape)
                x = x_recon + sigma_t * (z_g + z_j)
            else:
                x = x_recon
        return x.cpu()
