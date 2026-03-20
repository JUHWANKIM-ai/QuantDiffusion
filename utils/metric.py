import gc
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm, kstest  # KS 검정을 위해 kstest 추가
from utils.utils import rolling_window
from statsmodels.tsa.stattools import acf
from scipy.stats import skew, kurtosis, wasserstein_distance, ks_2samp

from utils.file_manager import quantDiffusionPath


file_manager = quantDiffusionPath()


def rolling_window(data, window_size):
    n_windows = data.shape[0] - window_size + 1
    windows = np.zeros((n_windows, window_size, data.shape[1]))
    for idx in range(n_windows):
        windows[idx] = data[idx : idx + window_size]
    return windows


def _scalar_diff_loss(diff, loss_type):
    """Compute scalar loss from diff array: mse, mae, or frobenius."""
    if loss_type == "mse":
        return np.mean(diff**2)
    if loss_type == "mae":
        return np.mean(np.abs(diff))
    if loss_type in ("frobenius", "fro"):
        return np.linalg.norm(diff)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def correlation_mean(data):
    """
    각 샘플의 상관행렬을 계산한 후, 그 평균을 반환.

    Parameters:
        data (np.ndarray): shape = (num_samples, num_time_steps, num_features)

    Returns:
        mean_correlation_matrix (np.ndarray): shape = (num_features, num_features)
    """
    num_samples = data.shape[0]
    correlations = []

    for i in range(num_samples):
        sample = data[i]  # shape = (num_time_steps, num_features)
        corr_mat = np.corrcoef(sample, rowvar=False)
        correlations.append(corr_mat)

    mean_correlation_matrix = np.mean(correlations, axis=0)
    return mean_correlation_matrix


def ccf_mean_vectorized(data, max_lag=10):
    """
    행렬 연산(Vectorization)을 활용하여 매우 빠르고 메모리 효율적인 CCF 평균 계산
    - data: shape = (num_samples, time_steps, num_features)
    """
    B, T, F = data.shape

    # 1. 시간 축(axis=1)을 기준으로 전체 평균과 표준편차 계산
    mu = np.mean(data, axis=1, keepdims=True)  # (B, 1, F)
    # 모집단 표준편차(ddof=0) 사용: statsmodels의 acovf 기본값과 일치
    std = np.std(data, axis=1, keepdims=True)  # (B, 1, F)

    # 2. Z-Score 정규화 (표준편차가 0인 경우 0으로 나누는 것을 방지하기 위해 np.divide 사용)
    Z = np.divide(data - mu, std, out=np.zeros_like(data), where=std != 0)

    # 3. 유효한(valid) 데이터 마스크 생성 (std가 0이 아닌 정상 샘플 필터링)
    valid_mask = std[:, 0, :] != 0  # (B, F)
    # i번째 feature와 j번째 feature 둘 다 유효해야 하므로 (B, F, F) 마스크 생성
    valid_pair_mask = valid_mask[:, :, None] & valid_mask[:, None, :]

    # 4. 결과를 저장할 배열: (F, F, max_lag + 1)
    mean_ccf = np.zeros((F, F, max_lag + 1))

    # 5. 각 시차(lag)별로 행렬곱 연산 수행 (for문은 단 max_lag+1 번만 돎)
    for k in range(max_lag + 1):
        if k == 0:
            Z_0 = Z
            Z_k = Z
        else:
            Z_0 = Z[:, :-k, :]  # 0 ~ (T-k-1)
            Z_k = Z[:, k:, :]  # k ~ (T-1)

        # einsum을 이용해 배치(b) 및 시간(t) 축에 대해 3차원 행렬을 한 번에 내적(Dot product)
        # Z_0: (B, T-k, F), Z_k: (B, T-k, F) -> 결과: (B, F, F)
        # statsmodels.ccf(adjusted=False)의 공식에 맞춰 분모를 전체 길이 T로 나눔
        ccf_k_batch = np.einsum("bti, btj -> bij", Z_0, Z_k) / T

        # 비정상 샘플(표준편차 0이라 연산 불가)의 결과는 제외하기 위해 마스킹 처리
        ccf_k_batch_valid = np.where(valid_pair_mask, ccf_k_batch, 0)

        # 배치 차원(axis=0)에 대해 합산 및 유효 개수 카운트
        sum_ccf = np.sum(ccf_k_batch_valid, axis=0)  # (F, F)
        count_valid = np.sum(valid_pair_mask, axis=0)  # (F, F)

        # 평균 계산 (유효 샘플이 0개인 구간은 NaN 방지를 위해 0으로 채움)
        mean_ccf[:, :, k] = np.divide(
            sum_ccf,
            count_valid,
            out=np.zeros_like(sum_ccf, dtype=float),
            where=count_valid != 0,
        )

    return mean_ccf


def _scaled_ccf_loss(real_mean_ccf, fake_mean_ccf, loss_type="mse"):
    """
    (기존 사용하시던 로직 유지. _scalar_diff_loss 함수가 정의되어 있다고 가정합니다.)
    """
    diff = fake_mean_ccf - real_mean_ccf
    # 사용자의 환경에 맞게 오차 계산 (아래는 예시)
    if loss_type == "mae":
        loss_val = np.mean(np.abs(diff))
    elif loss_type == "mse":
        loss_val = np.mean(diff**2)
    elif loss_type == "frobenius":
        loss_val = np.linalg.norm(diff)
    else:
        # 기존에 _scalar_diff_loss를 호출하셨다면 해당 함수 사용
        loss_val = _scalar_diff_loss(diff, loss_type)

    return round(float(loss_val), 4)


def calculate_ccf(real_data, fake_data, max_lag):
    CCF_dict = {}

    # CCF 계산을 위해 데이터 전치 후, 백터화 함수 직접 호출
    real_mean_ccf = ccf_mean_vectorized(
        np.transpose(real_data, (0, 2, 1)), max_lag=max_lag
    )
    fake_mean_ccf = ccf_mean_vectorized(
        np.transpose(fake_data, (0, 2, 1)), max_lag=max_lag
    )

    # 손실 계산
    loss_mae = _scaled_ccf_loss(real_mean_ccf, fake_mean_ccf, loss_type="mae")
    loss_frob = _scaled_ccf_loss(real_mean_ccf, fake_mean_ccf, loss_type="frobenius")
    loss_mse = _scaled_ccf_loss(real_mean_ccf, fake_mean_ccf, loss_type="mse")

    if max_lag == 0:
        CCF_dict[f"CCF(τ = 0)_MAE"] = loss_mae
        CCF_dict[f"CCF(τ = 0)_FROB"] = loss_frob
        CCF_dict[f"CCF(τ = 0)_MSE"] = loss_mse

    else:
        CCF_dict[f"CCF(τ > 0)_MAE"] = loss_mae
        CCF_dict[f"CCF(τ > 0)_FROB"] = loss_frob
        CCF_dict[f"CCF(τ > 0)_MSE"] = loss_mse

    df_avg_scores = pd.DataFrame(
        list(CCF_dict.values()), index=CCF_dict.keys(), columns=["Average"]
    )
    df_avg_scores.index.name = "Metric"

    return df_avg_scores


def calculate_distribution_scores(real, fake, windows):
    """
    real: (B_real, companys, T)
    fake: (B_fake, companys, T)
    """
    # 1. 배열 차원 확보 (2D가 들어와도 3D로 안전하게 처리)
    real = np.atleast_3d(real)
    fake = np.atleast_3d(fake)

    # B는 다를 수 있으므로 각각 추출, companys(C)와 T는 동일하다고 가정
    B_r, C, T_r = real.shape
    B_f, _, T_f = fake.shape

    results = {}
    metrics = ["EMD", "KS"]

    for w in windows:
        # 1. O(1) 롤링 윈도우 연산 (시간 축인 axis=2 기준 누적합)
        real_cs = np.insert(np.cumsum(real, axis=2), 0, 0, axis=2)
        fake_cs = np.insert(np.cumsum(fake, axis=2), 0, 0, axis=2)

        real_roll = real_cs[:, :, w:] - real_cs[:, :, :-w]  # (B_r, C, T - w + 1)
        fake_roll = fake_cs[:, :, w:] - fake_cs[:, :, :-w]  # (B_f, C, T - w + 1)

        # ==========================================================
        # 3. EMD & KS Statistic 연산 (종목별 풀링 분포 거리 측정)
        # ==========================================================
        # B가 다르므로 1:1 매칭 불가 -> 종목(C)별로 모든 배치의 데이터를 하나로 풀링
        # np.swapaxes를 통해 (C, B, T_w)로 바꾼 뒤 평탄화 -> Shape: (C, B * T_w)
        real_pooled = np.swapaxes(real_roll, 0, 1).reshape(C, -1)
        fake_pooled = np.swapaxes(fake_roll, 0, 1).reshape(C, -1)

        emd_scores = np.zeros(C)
        ks_scores = np.zeros(C)

        # 각 종목(Company) 단위로 완성된 두 분포의 거리를 측정 (for문 C번만 실행됨)
        for c in range(C):
            emd_scores[c] = wasserstein_distance(real_pooled[c], fake_pooled[c])
            ks_scores[c], _ = ks_2samp(real_pooled[c], fake_pooled[c])

        # ==========================================================
        # 4. 결과 저장
        # ==========================================================
        results[f"EMD({w})"] = np.round(np.mean(emd_scores), decimals=4)
        results[f"KS({w})"] = np.round(np.mean(ks_scores), decimals=4)
    # DataFrame 포맷팅
    ordered_keys = [f"{m}({w})" for m in metrics for w in windows]
    df_avg_scores = pd.DataFrame(
        [results[k] for k in ordered_keys], index=ordered_keys, columns=["Average"]
    )
    df_avg_scores.index.name = "Metric"

    return df_avg_scores


def calculate_skew_kurtosis(real_data, fake_data, windows):
    """
    각 샘플(배치)별 Skew/Kurt를 계산하여 평균 차이를 구하고,
    요청하신 단일 컬럼 DataFrame 형식으로 반환합니다.
    - real_data: (B_real, C, T)
    - fake_data: (B_fake, C, T)
    """
    B_r, C, T_r = real_data.shape
    B_f, _, T_f = fake_data.shape

    # 내부에서 자산별 리스트 생성 (제시해주신 입력 방식 반영)
    real_list = [real_data[:, i, :] for i in range(C)]
    fake_list = [fake_data[:, i, :] for i in range(C)]

    results = {}
    metrics = [
        "Skew",
        "Kurt",
    ]  # EMD, KS는 외부에서 합친다고 가정하거나 여기서는 Skew/Kurt만 처리

    for w in windows:
        skew_diffs_assets = []
        kurt_diffs_assets = []

        for i in range(C):
            real_arr = np.asarray(real_list[i])  # (B_r, T)
            fake_arr = np.asarray(fake_list[i])  # (B_f, T)

            # 배치 사이즈가 다를 경우 최소값에 맞춤
            num = min(real_arr.shape[0], fake_arr.shape[0])
            r_arr = real_arr[:num]
            f_arr = fake_arr[:num]

            # 롤링 윈도우 합산 (CumSum 활용 최적화)
            r_cs = np.pad(np.cumsum(r_arr, axis=1), ((0, 0), (1, 0)), mode="constant")
            f_cs = np.pad(np.cumsum(f_arr, axis=1), ((0, 0), (1, 0)), mode="constant")

            r_roll = r_cs[:, w:] - r_cs[:, :-w]
            f_roll = f_cs[:, w:] - f_cs[:, :-w]

            # 배치별 skew / kurtosis 계산 (axis=1)
            r_skews = skew(r_roll, axis=1, bias=True, nan_policy="omit")
            f_skews = skew(f_roll, axis=1, bias=True, nan_policy="omit")

            r_kurts = kurtosis(
                r_roll, axis=1, fisher=False, bias=True, nan_policy="omit"
            )
            f_kurts = kurtosis(
                f_roll, axis=1, fisher=False, bias=True, nan_policy="omit"
            )

            # 샘플 평균값의 차이(Absolute Error of Means)
            skew_diff = np.abs(np.nanmean(r_skews) - np.nanmean(f_skews))
            kurt_diff = np.abs(np.nanmean(r_kurts) - np.nanmean(f_kurts))

            skew_diffs_assets.append(skew_diff)
            kurt_diffs_assets.append(kurt_diff)

        # 모든 자산에 대한 평균값을 결과 딕셔너리에 저장
        results[f"Skew({w})"] = np.round(np.nanmean(skew_diffs_assets), decimals=4)
        results[f"Kurt({w})"] = np.round(np.nanmean(kurt_diffs_assets), decimals=4)

    # DataFrame 포맷팅 (요청하신 형식)
    ordered_keys = [f"{m}({w})" for m in metrics for w in windows]
    df_avg_scores = pd.DataFrame(
        [results[k] for k in ordered_keys], index=ordered_keys, columns=["Average"]
    )
    df_avg_scores.index.name = "Metric"

    return df_avg_scores


def vectorized_batch_acf_3d(data, nlags=30):
    """
    (B, C, T) 형태의 3차원 데이터를 받아 모든 배치와 종목의 ACF를 한 번에 계산
    - data: shape = (B, C, T)
    - return: shape = (B, C, nlags + 1)
    """
    data = np.atleast_3d(data)
    B, C, T = data.shape

    # 시간 축(axis=2)을 기준으로 평균 차감
    mu = np.mean(data, axis=2, keepdims=True)
    x = data - mu

    # c_0 (분산) - ACF의 분모
    c0 = np.sum(x**2, axis=2) / T

    # 결과를 담을 3차원 배열 (lag 0은 항상 1.0)
    acf_res = np.zeros((B, C, nlags + 1))
    acf_res[:, :, 0] = 1.0

    # 시차(lag)별로 행렬 슬라이싱 내적 연산 수행
    for lag in range(1, nlags + 1):
        # 시간 축(axis=2)을 lag만큼 이동시킨 두 배열 간의 내적 (c_k)
        ck = np.sum(x[:, :, :-lag] * x[:, :, lag:], axis=2) / T

        # 0으로 나누는 오류 방지
        acf_res[:, :, lag] = np.divide(ck, c0, out=np.zeros((B, C)), where=c0 != 0)

    return acf_res


def calculate_acf_score(real, fake, lags=30, loss_type="mse"):
    """
    real: (B_r, C, T)
    fake: (B_f, C, T)
    """
    real = np.atleast_3d(real)
    fake = np.atleast_3d(fake)

    B_r, C, T_r = real.shape
    B_f, _, T_f = fake.shape

    titles = ["ACF(x)", "ACF(|x|)", "ACF(x^2)"]
    data_transforms = [lambda x: x, np.abs, np.square]

    # 결과를 저장할 딕셔너리 초기화
    acf_scores = {f"Group {i+1}": {} for i in range(C)}
    avg_scores_per_transform = {title: [] for title in titles}

    for transform, title in zip(data_transforms, titles):
        # (1) 3차원 데이터 전체 변환 (Identity, Absolute, Squared)
        t_real = transform(real)
        t_fake = transform(fake)

        # (2) 초고속 3D ACF 병렬 계산 -> shape: (B, C, lags + 1)
        acf_real = vectorized_batch_acf_3d(t_real, nlags=lags)
        acf_fake = vectorized_batch_acf_3d(t_fake, nlags=lags)

        # (3) 배치 축(axis=0)에 대한 평균 ACF 도출 -> shape: (C, lags + 1)
        # 배치(B) 사이즈가 서로 달라도 이 단계에서 대푯값으로 압축되므로 문제없이 비교 가능합니다.
        mean_acf_real = np.mean(acf_real, axis=0)
        mean_acf_fake = np.mean(acf_fake, axis=0)

        # (4) Real 평균 ACF vs Fake 평균 ACF 오차 계산 -> shape: (C, lags + 1)
        diff = mean_acf_fake - mean_acf_real

        # (5) 종목(C)별로 Loss 계산
        for i in range(C):
            # _scalar_diff_loss가 1D 배열(lags + 1 길이)을 받는다고 가정
            val = round(float(_scalar_diff_loss(diff[i], loss_type)), 4)
            acf_scores[f"Group {i+1}"][title] = val
            avg_scores_per_transform[title].append(val)

    # (6) 전체 "평균의 평균(Mean of Means)" 계산
    overall_average_scores = {
        title: round(np.mean(vals), 4)
        for title, vals in avg_scores_per_transform.items()
    }

    # (7) DataFrame 포맷팅
    df_overall = pd.DataFrame(
        list(overall_average_scores.values()),
        index=overall_average_scores.keys(),
        columns=["Average"],
    )
    df_overall.index.name = "Metric"

    return acf_scores, df_overall


def vectorized_batch_leverage_effect_3d(data, lags=30):
    """
    (B, C, T) 형태의 3차원 데이터를 받아 모든 배치와 종목의 레버리지 효과를 한 번에 계산
    - data: shape = (B, C, T)
    - return: shape = (B, C, lags)
    """
    data = np.atleast_3d(data)
    B, C, T = data.shape

    # 결과를 담을 3차원 배열
    leverage_res = np.zeros((B, C, lags))

    # tau(시차)별로 슬라이싱하여 모든 샘플, 모든 종목의 상관계수를 행렬 내적으로 한 번에 계산
    for tau in range(1, lags + 1):
        # 시간 축(axis=2)을 기준으로 슬라이싱
        # X: r_t (앞부분), Y: r_{t+tau}^2 (뒷부분의 제곱)
        X = data[:, :, :-tau]
        Y = data[:, :, tau:] ** 2

        # 시간 축(axis=2) 기준 평균 차감 (Z-score 분자 부분)
        X_c = X - np.mean(X, axis=2, keepdims=True)
        Y_c = Y - np.mean(Y, axis=2, keepdims=True)

        # 상관계수 분자 (공분산 계산 시 분모인 N-1은 약분되므로 생략 가능)
        cov_num = np.sum(X_c * Y_c, axis=2)

        # 상관계수 분모 (분산 계산 시 분모인 N-1은 약분됨)
        var_X = np.sum(X_c**2, axis=2)
        var_Y = np.sum(Y_c**2, axis=2)
        denom = np.sqrt(var_X * var_Y)

        # 0으로 나누는 오류 방지
        leverage_res[:, :, tau - 1] = np.divide(
            cov_num, denom, out=np.zeros((B, C)), where=denom != 0
        )

    return leverage_res


def calculate_leverage_effect(real, fake, lags=30, loss_type="mse"):
    """
    real: (B_r, C, T)
    fake: (B_f, C, T)
    """
    real = np.atleast_3d(real)
    fake = np.atleast_3d(fake)

    B_r, C, T_r = real.shape
    B_f, _, T_f = fake.shape

    leverage_scores = {}
    all_group_losses = []

    # (1) 3D 벡터화를 통한 초고속 병렬 계산 -> shape: (B_r, C, lags) & (B_f, C, lags)
    real_leverage = vectorized_batch_leverage_effect_3d(real, lags=lags)
    fake_leverage = vectorized_batch_leverage_effect_3d(fake, lags=lags)

    # (2) 배치 축(axis=0)에 대한 평균 도출 -> shape: (C, lags)
    # 배치 크기(B_r, B_f)가 달라도 여기서 종목별 대푯값으로 압축되므로 1:1 비교가 가능해집니다.
    mean_real_leverage = np.mean(real_leverage, axis=0)
    mean_fake_leverage = np.mean(fake_leverage, axis=0)

    # (3) Real과 Fake 평균 간의 차이 계산 -> shape: (C, lags)
    diff = mean_fake_leverage - mean_real_leverage

    # (4) 종목(C)별로 Loss 계산
    for i in range(C):
        # _scalar_diff_loss가 1D 배열(lags 길이)을 받는다고 가정
        loss_val = round(float(_scalar_diff_loss(diff[i], loss_type)), 4)
        leverage_scores[f"Group {i+1}"] = loss_val
        all_group_losses.append(loss_val)

    # (5) 전체 평균 손실 계산
    overall_mean = round(float(np.mean(all_group_losses)), 4)

    # (6) DataFrame 래핑
    df_overall = pd.DataFrame(
        [overall_mean], index=["Leverage Effect"], columns=["Average"]
    )
    df_overall.index.name = "Metric"

    return leverage_scores, df_overall


# ---------------------------------------------------------
# [내부 헬퍼 함수] 시각화 함수 내부에서 쓸 초고속 연산기
# ---------------------------------------------------------
def _fast_acf_2d(data, nlags):
    """시각화용 초고속 2D ACF 계산기"""
    data = np.atleast_2d(data)
    N, T = data.shape
    mu = np.mean(data, axis=1, keepdims=True)
    x = data - mu
    c0 = np.sum(x**2, axis=1) / T

    res = np.zeros((N, nlags + 1))
    res[:, 0] = 1.0
    for lag in range(1, nlags + 1):
        ck = np.sum(x[:, :-lag] * x[:, lag:], axis=1) / T
        res[:, lag] = np.divide(ck, c0, out=np.zeros(N), where=c0 != 0)
    return res


# ---------------------------------------------------------
# 1. ACF 비교 시각화 (초고속 연산 + 고해상도 투명 배경)
# ---------------------------------------------------------
def plot_acf_comparison(real_list, fake_list, n_vars, lags=30, sampling_type=None):
    data_transforms = [lambda x: x, np.abs, np.square]
    titles = ["Identity log returns", "Absolute log returns", "Squared log returns"]

    for i in range(n_vars):
        # dpi=400 적용
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 4), dpi=400)

        # 피규어 전체 투명 배경 설정
        fig.patch.set_alpha(0.0)

        for ax, transform, title in zip(axs, data_transforms, titles):
            # 플롯 내부(Axes) 투명 배경 설정
            ax.set_facecolor("none")

            transformed_real = transform(np.asarray(real_list[i]))
            transformed_fake = transform(np.asarray(fake_list[i]))

            # [병목 제거] for문 + statsmodels.acf 대신 100배 빠른 벡터화 함수 사용
            acf_real = _fast_acf_2d(transformed_real, nlags=lags)
            acf_fake = _fast_acf_2d(transformed_fake, nlags=lags)

            mean_real = np.mean(acf_real, axis=0)
            std_real = np.std(acf_real, axis=0)
            mean_fake = np.mean(acf_fake, axis=0)
            std_fake = np.std(acf_fake, axis=0)

            ax.plot(mean_real, label="Real (Mean)", color="tab:blue")
            ax.fill_between(
                range(lags + 1),
                mean_real - 0.5 * std_real,
                mean_real + 0.5 * std_real,
                color="tab:blue",
                alpha=0.2,
                label="Real ± 1/2 Std",
            )

            ax.plot(mean_fake, label="Synthetic (Mean)", color="tab:orange")
            ax.fill_between(
                range(lags + 1),
                mean_fake - 0.5 * std_fake,
                mean_fake + 0.5 * std_fake,
                color="tab:orange",
                alpha=0.2,
                label="Synthetic ± 1/2 Std",
            )

            ax.set_ylim(-0.15, 0.3)
            ax.set_title(title)
            ax.grid(True, alpha=0.5)  # 그리드도 살짝 투명하게 조절
            ax.axhline(y=0, color="k")
            ax.axvline(x=0, color="k")
            ax.set_xlabel("Lag", fontsize=16)

        axs[0].legend()
        plt.tight_layout()

        # 파일로 저장할 경우 투명 배경(transparent=True) 유지
        if sampling_type:
            os.makedirs(
                f"{file_manager.get_image_path()}/{sampling_type}", exist_ok=True
            )
            plt.savefig(
                f"{file_manager.get_image_path()}/{sampling_type}/acf_comparison_asset_{i}.png",
                dpi=400,
                transparent=True,
                bbox_inches="tight",
            )

        plt.close()
        gc.collect()


# ---------------------------------------------------------
# 2. 상관계수 히트맵 (고해상도 투명 배경)
# ---------------------------------------------------------
def plot_correlation_heatmap(
    correlation_matrix, title, feature_names, sampling_type=None
):
    # dpi=400 및 피규어 투명화
    fig, ax = plt.subplots(figsize=(5, 4), dpi=400)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    sns.heatmap(
        correlation_matrix,
        annot=False,
        cmap="coolwarm",
        linewidths=0,
        ax=ax,
        center=0.45,
        vmin=-0.1,
        vmax=1.0,
        cbar_kws={"ticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
    )

    ax.set_xticklabels(feature_names, fontsize=12, rotation=45)
    ax.set_yticklabels(feature_names, fontsize=12, rotation=0)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)

    plt.tight_layout()

    if sampling_type:
        os.makedirs(f"{file_manager.get_image_path()}/{sampling_type}", exist_ok=True)
        plt.savefig(
            f"{file_manager.get_image_path()}/{sampling_type}/correlation_heatmap_{title}.png",
            dpi=400,
            transparent=True,
            bbox_inches="tight",
        )

    plt.close()
    gc.collect()


# ---------------------------------------------------------
# 3. 누적 수익률 분포 플롯 (초고속 연산 + 고해상도 투명 배경)
# ---------------------------------------------------------
def plot_cumulative_return_dist(
    real_list,
    fake_list,
    windows=[1, 20, 100],
    n_assets=6,
    bins_n=81,
    sampling_type=None,
):
    for j in range(n_assets):
        # dpi=400 및 투명도 설정
        fig, axs = plt.subplots(
            nrows=1, ncols=len(windows), figsize=(6 * len(windows), 4), dpi=400
        )
        fig.patch.set_alpha(0.0)

        if len(windows) == 1:
            axs = [axs]

        # 배열화
        real_arr = np.atleast_2d(real_list[j])
        fake_arr = np.atleast_2d(fake_list[j])

        for i, window_size in enumerate(windows):
            axs[i].set_facecolor("none")  # Axes 투명화

            # [병목 제거] 무거운 rolling_window 대신 O(1) CumSum 활용
            real_cs = np.insert(np.cumsum(real_arr, axis=1), 0, 0, axis=1)
            fake_cs = np.insert(np.cumsum(fake_arr, axis=1), 0, 0, axis=1)

            real_dist = (real_cs[:, window_size:] - real_cs[:, :-window_size]).ravel()
            fake_dist = (fake_cs[:, window_size:] - fake_cs[:, :-window_size]).ravel()

            min_val, max_val = real_dist.min(), real_dist.max()
            bins = np.linspace(min_val, max_val, bins_n)

            sns.histplot(
                real_dist,
                bins=bins,
                kde=False,
                ax=axs[i],
                color="tab:blue",
                alpha=0.5,
                stat="density",
                label="Historical",
            )
            sns.histplot(
                fake_dist,
                bins=bins,
                kde=False,
                ax=axs[i],
                color="tab:orange",
                alpha=0.5,
                stat="density",
                label="Synthetic",
            )

            xlim_low, xlim_high = np.quantile(real_dist, [0.001, 0.999])
            axs[i].set_xlim(xlim_low, xlim_high)
            axs[i].yaxis.grid(True, alpha=0.3)
            axs[i].set_xlabel(f"Cumulative log return ({window_size} Day)", fontsize=12)
            axs[i].set_ylabel("Density", fontsize=12)

            if i == 0:
                axs[i].legend()

        plt.tight_layout()

        if sampling_type:
            os.makedirs(
                f"{file_manager.get_image_path()}/{sampling_type}", exist_ok=True
            )
            plt.savefig(
                f"{file_manager.get_image_path()}/{sampling_type}/cumulative_return_dist_asset_{j}.png",
                dpi=400,
                transparent=True,
                bbox_inches="tight",
            )

        plt.close()
        gc.collect()


def plot_transformations_comparison(
    original, scaled, inversed, feature_idx=0, num_batches_to_show=10, MODWT=False
):
    """
    학술적 고도화가 적용된 시각화 함수
    original, scaled, inversed: (B, T, C) 형태의 numpy array
    feature_idx: 시각화할 채널 인덱스
    """
    # 데이터 슬라이싱 및 Flatten (수천 개 배치를 합치므로 flatten()을 미리 수행)
    orig_sub_flat = original[:num_batches_to_show, :, feature_idx].flatten()
    scal_sub_flat = scaled[:num_batches_to_show, :, feature_idx].flatten()
    inv_sub_flat = inversed[:num_batches_to_show, :, feature_idx].flatten()

    # ---------------------------------------------------------
    # KS Test 수행: 표준 정규분포('norm')와 비교
    # ---------------------------------------------------------
    ks_stat, p_val = kstest(scal_sub_flat, "norm")

    # 학술적 표기법: p-value가 매우 작을 경우 < 0.001로 표기
    p_text = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.4f}"
    ks_results_text = f"KS Statistic ($D$): {ks_stat:.4f}\n{p_text}"

    # Figure 설정 (흑백 인쇄를 고려한 스타일 적용 가능)
    # plt.style.use('grayscale') # 필요시 활성화
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. 시계열 복원력 확인 (Time Series Recovery Fidelity)
    ax_ts = axes[0, 0]
    ax_ts.plot(
        original[0, :, feature_idx], label="Original", alpha=0.8, color="blue", lw=1.5
    )
    ax_ts.plot(
        inversed[0, :, feature_idx],
        label="Inversed",
        linestyle="--",
        alpha=0.7,
        color="orange",
        lw=2,
    )
    ax_ts.set_title(
        f"Channel {feature_idx}: Time Series Recovery", fontsize=14, fontweight="bold"
    )
    ax_ts.set_ylabel("Log Returns", fontsize=12)
    ax_ts.grid(axis="y", linestyle="--", alpha=0.5)
    ax_ts.legend(fontsize=11)

    # 2. 가우시안화된 라텐트 흐름 (Scaled Latent Space Flow)
    ax_latent = axes[0, 1]
    ax_latent.plot(
        scaled[0, :, feature_idx], color="red", lw=1, label="Normalized $z$-score"
    )
    ax_latent.set_title(
        f"Normalized $z$-score Latent Flow", fontsize=14, fontweight="bold"
    )
    ax_latent.set_ylabel("$z$-score", fontsize=12)
    ax_latent.set_ylim(-4, 4)  # z-score 일반 범위 고정
    ax_latent.yaxis.set_major_locator(
        ticker.MultipleLocator(1)
    )  # y축 눈금 간격 1로 설정
    ax_latent.grid(axis="y", linestyle="--", alpha=0.5)
    ax_latent.legend(fontsize=11)

    # 3. 분포 복원력 비교 (PDF Comparison: Original vs Inversed)
    ax_dist_rec = axes[1, 0]

    # 원본 분포: 파란색 채움
    sns.kdeplot(
        orig_sub_flat,
        ax=ax_dist_rec,
        label="Original PDF",
        fill=True,
        color="blue",
        alpha=0.5,
    )

    # 역변환 분포: fill=True로 설정하여 면적을 만들고 hatch 적용
    sns.kdeplot(
        inv_sub_flat,
        ax=ax_dist_rec,
        label="Inversed PDF",
        fill=True,
        color="orange",
        lw=2,
        linestyle="--",
        hatch="///",
        alpha=0.4,
        edgecolor="orange",
    )  # edgecolor로 선 색상 강조

    ax_dist_rec.set_title(
        f"PDF Comparison: Distribution Recovery (N={num_batches_to_show} Batches)",
        fontsize=14,
        fontweight="bold",
    )
    ax_dist_rec.set_ylabel("Density", fontsize=12)
    ax_dist_rec.legend(fontsize=11)

    # 4. 정규성 수렴 확인 (Gaussian Target Convergence Check)
    # 학술적 고도화: 히스토그램 + 변환 데이터 KDE + 가우시안 타겟 3중 비교
    ax_norm = axes[1, 1]
    # 변환된 데이터 히스토그램 (매우 투명하게)
    sns.histplot(
        scal_sub_flat, ax=ax_norm, stat="density", color="green", alpha=0.1, bins=50
    )
    # 변환된 데이터 자체의 KDE (초록색 실선)
    sns.kdeplot(
        scal_sub_flat, ax=ax_norm, color="green", lw=2.5, label="Transformed Data PDF"
    )

    x = np.linspace(-5, 5, 200)
    ax_norm.plot(x, norm.pdf(x), "r--", lw=3, label="Target: $N(0,1)$ PDF")

    # KS 검정 결과 박스 추가 (오른쪽 상단 배치)
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray")
    ax_norm.text(
        0.95,
        0.95,
        ks_results_text,
        transform=ax_norm.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
        fontweight="bold",
    )

    ax_norm.set_title(f"Gaussian Convergence & KS Test", fontsize=14, fontweight="bold")
    ax_norm.set_ylabel("Density", fontsize=12)
    ax_norm.legend(
        loc="upper left", fontsize=11
    )  # 텍스트 박스와 겹치지 않게 범례 위치 조정

    title_text = f"Transformations Comparison: Channel {feature_idx} "
    title_text += "(MODWT Enabled)" if MODWT else "(SA and LW Only)"
    fig.suptitle(title_text, fontsize=16, fontweight="bold", y=0.98)

    # 파일 저장 (투명 배경 적용 가능 여부 재확인: transparent=True)

    folder_path = os.path.join(file_manager.get_image_path(), "preprocessing")
    os.makedirs(folder_path, exist_ok=True)
    if MODWT:
        save_path = f"{folder_path}/transformations_advanced_ch{feature_idx}_MODWT.png"
    else:
        save_path = (
            f"{folder_path}/transformations_advanced_ch{feature_idx}_SA_after_LW.png"
        )

    plt.savefig(save_path, dpi=400, transparent=True)  # 배경 투명하게 고도화

    plt.close(fig)
    gc.collect()


def check_attention_position_sensitivity(
    model, x, t=None, device="cpu", num_timesteps=10
):
    """Cross-Attention이 시계열 선후 관계(위치)를 반영하는지 진단.

    Positional Encoding이 없으면 입력을 시간축으로 뒤집어도 출력이 대칭적으로 뒤집힘(permutation equivariant).
    PE가 있으면 뒤집은 입력의 출력이 단순 뒤집기와 다르게 나와야 함.

    Args:
        model: 노이즈 예측 모델 (forward(x, t, x_self_cond=None) 호출 가능)
        x: [B, T, C] 입력
        t: [B] diffusion timestep (None이면 랜덤)
        device: torch device
        num_timesteps: t 샘플 개수로 평균 낼 때 사용 (선택)

    Returns:
        dict: flip_diff (뒤집기 차이, 클수록 위치 민감), position_sensitive (bool)
    """
    import torch

    model = model.to(device).eval()
    x = x.to(device)
    B, T, C = x.shape
    if t is None:
        t = torch.randint(0, num_timesteps, (B,), device=device, dtype=torch.long)
    else:
        t = t.to(device)

    with torch.no_grad():
        out = model(x, t)
        out_rev = model(x.flip(1), t)

    # 위치 인코딩이 있으면: 입력을 뒤집었을 때 출력은 "내용이 뒤바뀐 것"이지, "같은 출력을 뒤집은 것"과는 다름
    flip_diff = (out_rev - out.flip(1)).abs().mean().item()
    # 기준: 차이가 충분히 크면 위치 민감하다고 판단 (스케일에 따라 0.01~0.1 이상)
    position_sensitive = flip_diff > 1e-5

    return {
        "flip_diff": flip_diff,
        "position_sensitive": position_sensitive,
        "out_shape": tuple(out.shape),
    }
