import pandas as pd
import matplotlib.pyplot as plt
import torch
import math
import torch.nn.functional as F
import numpy as np
from statsmodels.tsa.stattools import acf
import json


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def read_json(PATH):
    ts_list = []
    norm_ts_list = []

    with open(PATH, "r") as file:
        for line in file:
            data = json.loads(line)
            ts = np.array(data["target"])
            ts_list.append(ts)
            norm_ts_list.append(ts / np.mean(np.abs(ts)))

    return ts_list, norm_ts_list


def linear_beta_schedule(timesteps, min_val, max_val):
    return torch.linspace(min_val, max_val, timesteps)


def get_scheduler_utils(T, betas):
    # -----------------------------------------#
    # betas = reversed(betas)
    # betas =torch.ones_like(betas)*0.0001
    # -----------------------------------------#
    sqrt_one_minus_beta = torch.sqrt(1.0 - betas)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return (
        sqrt_one_minus_beta,
        alphas,
        alphas_cumprod,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
        posterior_variance,
    )


def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def return_acf(ts, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, T=100):
    acf_list = []
    for t_idx in range(0, T, 1):
        t = torch.LongTensor([t_idx])
        x_noisy = q_sample(ts, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        acf_values = acf(x_noisy.detach().numpy(), fft=False, nlags=len(x_noisy) - 1)
        acf_list.append(acf_values)
    return np.array(acf_list)


def cosine_schedule(t, start=0, end=1, tau=1, clamp_min=1e-9):
    """t: scalar 또는 tensor 모두 지원"""
    if not isinstance(t, torch.Tensor):
        t = torch.FloatTensor([t])
    v_start = math.cos(start * math.pi / 2) ** (2 * tau)
    v_end = math.cos(end * math.pi / 2) ** (2 * tau)
    gamma = torch.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
    gamma = (v_end - gamma) / (v_end - v_start)
    return gamma.clamp_(min=clamp_min, max=5.0)


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def sigmoid_schedule(t, start=-3, end=3, tau=1, clamp_min=1e-9):
    """t: scalar 또는 tensor 모두 지원"""
    if not isinstance(t, torch.Tensor):
        t = torch.FloatTensor([t])
    v_start = sigmoid(torch.tensor(start / tau))
    v_end = sigmoid(torch.tensor(end / tau))
    A = -sigmoid((t * (end - start) + start) / tau) + v_end
    B = v_end - v_start
    gamma = A / B
    return gamma.clamp_(min=clamp_min, max=5.0)


def cosine_schedule_beta(T, tau, min_val, max_val):
    t_vals = torch.arange(1, T + 1, dtype=torch.float32) / T
    alphas_cumprod = cosine_schedule(t_vals, tau=tau)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1)
    alphas = alphas_cumprod / alphas_cumprod_prev
    betas = 1 - alphas
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    betas = betas * (max_val - min_val) + min_val
    return betas


def sigmoid_schedule_beta(T, tau, min_val, max_val):
    t_vals = torch.arange(1, T + 1, dtype=torch.float32) / T
    alphas_cumprod = sigmoid_schedule(t_vals, tau=tau)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1)
    alphas = alphas_cumprod / alphas_cumprod_prev
    betas = 1 - alphas
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    betas = betas * (max_val - min_val) + min_val
    return betas


def return_betas(scheduler="cosine", tau=2.0, T=75, beta_0=0.0001, beta_T=0.1):
    if scheduler == "linear":
        betas = linear_beta_schedule(T, beta_0, beta_T)
    elif scheduler == "cosine":
        betas = cosine_schedule_beta(T, tau, beta_0, beta_T)
    elif scheduler == "sigmoid":
        betas = sigmoid_schedule_beta(T, tau, beta_0, beta_T)
    return betas


def preprocess_data(x):
    # any functions!
    x = x / np.mean(np.abs(x))
    x = torch.FloatTensor(x)
    return x


def pad_acf_data(TS_acf_variable):
    temp = []
    length_list = []
    for ts_acf in TS_acf_variable:
        ts_acf = np.stack(ts_acf)
        temp.append(ts_acf)
        length_list.append(ts_acf.shape[1])
    med_length = int(np.mean(length_list))

    for idx, ts in enumerate(temp):
        T, length = ts.shape
        if length > med_length:
            temp[idx] = ts[:, :med_length]
        elif length < med_length:
            temp[idx] = np.pad(ts, ((0, 0), (0, med_length - length)), mode="constant")

    TS_acf_variable = np.stack(temp)
    return TS_acf_variable


def get_IAAT_list(TS_acf):
    # TS_acf: (N, T, L)
    acf_mean = np.mean(TS_acf, axis=0)  # (T, L)
    IAAT = 1 + (np.abs(acf_mean).sum(axis=1) * 2) - 2  # (T,)
    scaled_IAAT = (IAAT - IAAT.min()) / (IAAT.max() - IAAT.min())
    return IAAT, scaled_IAAT


def get_mIAAT_list(TS_acf, TS_var):
    """
    TS_acf: (N, T, L)
        N = 변수 개수
        T = 시점 수
        L = lag 수
    TS_var: (N,)
        각 변수의 분산 (원본 데이터 기준)

    return:
        IAAT: (T,)
        scaled_IAAT: (T,)
    """
    TS_acf = np.asarray(TS_acf, dtype=np.float64)
    TS_var = np.asarray(TS_var, dtype=np.float64)

    if TS_acf.ndim != 3:
        raise ValueError(f"TS_acf must have shape (N, T, L), got {TS_acf.shape}")
    if TS_var.ndim != 1:
        raise ValueError(f"TS_var must have shape (N,), got {TS_var.shape}")
    if TS_acf.shape[0] != TS_var.shape[0]:
        raise ValueError(
            f"Mismatch: TS_acf.shape[0]={TS_acf.shape[0]} != TS_var.shape[0]={TS_var.shape[0]}"
        )

    # 분산 기반 가중치 정규화
    weights = TS_var / (TS_var.sum() + 1e-12)  # (N,)

    # 가중 평균 ACF: (T, L)
    weighted_acf = np.average(TS_acf, axis=0, weights=weights)

    # lag 0이 포함된 경우를 가정
    # IAAT = 1 + 2 * sum_{lag=1}^{L-1} |rho(lag)|
    IAAT = 1.0 + 2.0 * np.sum(np.abs(weighted_acf[:, 1:]), axis=1)

    # min-max scaling
    denom = IAAT.max() - IAAT.min()
    if denom < 1e-12:
        scaled_IAAT = np.zeros_like(IAAT)
    else:
        scaled_IAAT = (IAAT - IAAT.min()) / denom

    return IAAT, scaled_IAAT


def calculate_auc(y):
    T = len(y)
    base = np.linspace(1, 0, T)
    diff = np.abs(y - base)
    return np.trapz(diff, dx=1 / T)


def calculate_ANT_score(IAAT, scaled_IAAT):
    T = len(IAAT)
    lambda_linear = calculate_auc(scaled_IAAT)
    lambda_noise = 1 + IAAT[-1] / IAAT[0]
    lambda_steps = 1 + 1 / T
    return lambda_linear * lambda_noise * lambda_steps
