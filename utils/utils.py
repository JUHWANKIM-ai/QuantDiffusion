import gc
import torch
import psutil  # 시스템 RAM 확인용
import numpy as np
from utils.gaussianize import *
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler


class SinhArcsinhTransformer:
    def __init__(self):
        self.epsilon = 0.0  # 왜도 파라미터
        self.delta = 1.0  # 첨도 파라미터

    def _log_likelihood(self, params, x):
        epsilon, delta = params
        if delta <= 0:
            return 1e10  # delta는 양수여야 함

        # SA 변환: z = sinh(delta * asinh(x) - epsilon)
        # Jacobian(미분값)이 포함된 로그 우도 함수
        C = np.cosh(delta * np.arcsinh(x) - epsilon)
        D = delta / np.sqrt(1 + x**2)
        z = np.sinh(delta * np.arcsinh(x) - epsilon)

        # log-likelihood = log(pdf(z)) + log(Jacobian)
        ll = norm.logpdf(z) + np.log(C) + np.log(D)
        return -np.sum(ll)  # minimize를 위해 음수화

    def fit(self, x):
        x = x.flatten()
        # 초기값: epsilon=0, delta=1 (정규분포 가정)
        res = minimize(
            self._log_likelihood,
            [0.0, 1.0],
            args=(x,),
            bounds=[(None, None), (1e-5, None)],
        )
        self.epsilon, self.delta = res.x
        return self

    def transform(self, x):
        return np.sinh(self.delta * np.arcsinh(x) - self.epsilon)

    def inverse_transform(self, z):
        return np.sinh((np.arcsinh(z) + self.epsilon) / self.delta)


def clean_memory(device):
    """CPU와 GPU 메모리를 강제로 정리합니다."""
    gc.collect()  # Python 레벨의 가비지 컬렉션 수행
    if device.type == "cuda":
        torch.cuda.empty_cache()  # GPU 예약 메모리 해제
        torch.cuda.reset_peak_memory_stats(device)  # 피크 메모리 통계 초기화


def print_memory_stats(device, prefix=""):
    """현재 리소스 상태를 출력합니다."""
    vm = psutil.virtual_memory()
    stats = f"{prefix} [RAM: {vm.percent}%]"
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        stats += f" [VRAM Alloc: {allocated:.1f}MB / Resrv: {reserved:.1f}MB]"
    print(stats)


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


def rolling_window(data, window_size):
    n_windows = data.shape[0] - window_size + 1
    windows = np.zeros((n_windows, window_size, data.shape[1]))
    for idx in range(n_windows):
        windows[idx] = data[idx : idx + window_size]
    return windows


def set_seed(seed: int):
    """Sets the seed to a specified value. Needed for reproducibility of experiments."""
    torch.manual_seed(seed)
    np.random.seed(seed)


""" Scale the log returns using Sinh-Arcsinh + Gaussianize + StandardScaler """
def scaling(data):
    scalers = []
    scaled_data = []

    for i in range(data.shape[1]):
        feature_data = data[:, i].reshape(-1, 1)

        ss1, ss2 = StandardScaler(), StandardScaler()
        # sa_transformer = SinhArcsinhTransformer()
        gaussianize = Gaussianize()

        # 1단계: SS1 -> Sinh-Arcsinh
        step1 = ss1.fit_transform(feature_data)
        # sa_transformer.fit(step1)
        # step1_sa = sa_transformer.transform(step1)

        # 2단계: SS2 -> Lambert W (Gaussianize)
        # step2 = ss2.fit_transform(step1_sa)
        step2_g = gaussianize.fit_transform(step1)

        # 3단계: SS3 (최종 정규화)
        feature_scaled = ss2.fit_transform(step2_g)

        scalers.append((ss1, ss2, gaussianize))
        scaled_data.append(feature_scaled.flatten())

    scaled_data = np.array(scaled_data).T
    return scaled_data, scalers


""" Inverse the scaling process for all features """


def inverse_scaling(y, scalers):
    y = y.cpu().detach().numpy() if hasattr(y, "cpu") else y
    y_original = np.zeros_like(y)

    for idx in range(y.shape[1]):
        ss1, ss2, gaussianize = scalers[idx]

        y_feature = y[:, idx, :]  # (batch, seq_len)

        # Diffusion 출력값 수치 안정화
        EPS = 1e-8
        y_feature = (y_feature - y_feature.mean(axis=0, keepdims=True)) / (
            y_feature.std(axis=0, keepdims=True) + EPS
        )

        orig_shape = y_feature.shape
        y_flat = y_feature.reshape(-1, 1)

        # 정방향의 역순: SS3 -> Gaussianize -> SS2 -> SinhArcsinh -> SS1
        y_flat = ss2.inverse_transform(y_flat)
        y_flat = np.array(
            [gaussianize.inverse_transform(s.reshape(-1, 1)) for s in y_flat]
        ).reshape(-1, 1)
        y_flat = ss1.inverse_transform(y_flat)

        y_original[:, idx, :] = y_flat.reshape(orig_shape)

    return y_original



# def scaling(data, p_lower=1, p_upper=99):
#     """
#     p_lower: 하위 백분위수 (예: 1은 하위 1%)
#     p_upper: 상위 백분위수 (예: 99는 상위 99%)
#     """
#     scalers = []
#     scaled_data = []

#     for i in range(data.shape[1]):
#         ss1, ss2, ss3 = StandardScaler(), StandardScaler(), StandardScaler()
#         sa_transformer = SinhArcsinhTransformer()
#         gaussianize = Gaussianize()

#         raw_feature = data[:, i].reshape(-1, 1)

#         # --- 1. Percentile 찾기 및 Winsorize ---
#         # 실제 데이터의 분포에서 경계값을 계산합니다.
#         lower_bound = np.percentile(raw_feature, p_lower)
#         upper_bound = np.percentile(raw_feature, p_upper)

#         # Fit용 데이터: 경계값을 벗어나는 점프(Jump)들을 경계값으로 고정
#         wins_feature = np.clip(raw_feature, lower_bound, upper_bound)

#         # --- 2. Fit(Winsorized) & Transform(Raw) 전략 ---

#         # 1단계: SS1 -> Sinh-Arcsinh
#         ss1.fit(wins_feature)
#         wins_step1 = ss1.transform(wins_feature)
#         raw_step1 = ss1.transform(raw_feature)

#         sa_transformer.fit(wins_step1)
#         wins_step1_sa = sa_transformer.transform(wins_step1)
#         raw_step1_sa = sa_transformer.transform(raw_step1)

#         # 2단계: SS2 -> Lambert W
#         ss2.fit(wins_step1_sa)
#         wins_step2 = ss2.transform(wins_step1_sa)
#         raw_step2 = ss2.transform(raw_step1_sa)

#         gaussianize.fit(wins_step2)
#         wins_step2_g = gaussianize.transform(wins_step2)
#         raw_step2_g = gaussianize.transform(raw_step2)

#         # 3단계: SS3 (최종 정규화)
#         ss3.fit(wins_step2_g)
#         feature_scaled = ss3.transform(raw_step2_g)

#         scalers.append((ss1, ss2, ss3, sa_transformer, gaussianize))
#         scaled_data.append(feature_scaled.flatten())

#     return np.array(scaled_data).T, scalers

# def inverse_scaling(y, scalers):
#     """
#     y shape: (batch_size, num_features, seq_len)
#     """
#     y = y.cpu().detach().numpy() if hasattr(y, 'cpu') else y
#     y_original = np.zeros_like(y)

#     for idx in range(y.shape[1]):
#         # 저장된 스케일러들 꺼내기 (순서 주의: sa_transformer, gaussianize)
#         ss1, ss2, ss3, sa_transformer, gaussianize = scalers[idx]

#         y_feature = y[:, idx, :]

#         # Diffusion 출력값의 수치적 안정을 위한 보정 (선택 사항)
#         EPS = 1e-8
#         y_feature = (y_feature - y_feature.mean(axis=0, keepdims=True)) / (y_feature.std(axis=0, keepdims=True) + EPS)

#         orig_shape = y_feature.shape
#         y_flat = y_feature.reshape(-1, 1)

#         # --- 역변환 단계 (정방향의 역순: SS3 -> LW -> SS2 -> SA -> SS1) ---

#         # 1. SS3 역변환
#         y_flat = ss3.inverse_transform(y_flat)

#         # 2. Lambert W (Gaussianize) 역변환 (LW가 마지막에 적용됐으므로 먼저 역변환)
#         y_flat = np.array([
#             gaussianize.inverse_transform(sample.reshape(-1, 1))
#             for sample in y_flat
#         ]).reshape(-1, 1)

#         # 3. SS2 역변환
#         y_flat = ss2.inverse_transform(y_flat)

#         # 4. Sinh-Arcsinh 역변환
#         y_flat = sa_transformer.inverse_transform(y_flat)

#         # 5. SS1 역변환
#         y_flat = ss1.inverse_transform(y_flat)

#         y_original[:, idx, :] = y_flat.reshape(orig_shape)

#     return y_original


# def scaling(data, p_lower=0.5, p_upper=99.5):
#     """
#     Sinh-Arcsinh를 제외한 버전: Winsorize -> SS1 -> Gaussianize -> SS2
#     p_lower/p_upper: Winsorize 분위수
#     """
#     scalers = []
#     scaled_data = []

#     for i in range(data.shape[1]):
#         ss1, ss2 = StandardScaler(), StandardScaler()
#         gaussianize = Gaussianize()

#         raw_feature = data[:, i].reshape(-1, 1)

#         # --- 1. Winsorize ---
#         lower_bound = np.percentile(raw_feature, p_lower)
#         upper_bound = np.percentile(raw_feature, p_upper)
#         wins_feature = np.clip(raw_feature, lower_bound, upper_bound)

#         # --- 2. Fit(Winsorized) & Transform(Raw) 전략 ---

#         # 1단계: SS1
#         ss1.fit(wins_feature)
#         wins_step1 = ss1.transform(wins_feature)
#         raw_step1 = ss1.transform(raw_feature)

#         # 2단계: Gaussianize (Lambert W)
#         gaussianize.fit(wins_step1)
#         wins_step1_g = gaussianize.transform(wins_step1)
#         raw_step1_g = gaussianize.transform(raw_step1)

#         # 3단계: SS2 (최종 정규화)
#         ss2.fit(wins_step1_g)
#         feature_scaled = ss2.transform(raw_step1_g)

#         scalers.append((ss1, ss2, gaussianize))
#         scaled_data.append(feature_scaled.flatten())

#     return np.array(scaled_data).T, scalers


# def inverse_scaling(y, scalers):
#     """
#     y shape: (batch_size, num_features, seq_len)
#     scalers: (ss1, ss2, gaussianize) 튜플 리스트
#     """
#     y = y.cpu().detach().numpy() if hasattr(y, "cpu") else y
#     y_original = np.zeros_like(y)

#     for idx in range(y.shape[1]):
#         ss1, ss2, gaussianize = scalers[idx]

#         y_feature = y[:, idx, :]

#         # 수치 안정화
#         EPS = 1e-8
#         y_feature = (y_feature - y_feature.mean(axis=0, keepdims=True)) / (
#             y_feature.std(axis=0, keepdims=True) + EPS
#         )

#         orig_shape = y_feature.shape
#         y_flat = y_feature.reshape(-1, 1)

#         # 정방향의 역순: SS2 -> Gaussianize -> SS1
#         y_flat = ss2.inverse_transform(y_flat)
#         y_flat = np.array(
#             [gaussianize.inverse_transform(s.reshape(-1, 1)) for s in y_flat]
#         ).reshape(-1, 1)
#         y_flat = ss1.inverse_transform(y_flat)

#         y_original[:, idx, :] = y_flat.reshape(orig_shape)

#     return y_original


def count_parameters(model):
    # 학습 가능한(Trainable) 파라미터 수
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 전체(Total) 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())

    print(f"✅ 전체 파라미터 수: {total_params:,}")
    print(f"✅ 학습 가능한 파라미터 수: {trainable_params:,}")
    return total_params


def compute_avg_emd(real_data, fake_data, window):
    """
    Compute the Earth Mover's Distance (EMD) for real and fake data over a rolling window.
    """
    emd = 0
    for i in range(real_data.shape[1]):  # Iterate over features
        real_dist = rolling_window(real_data[:, i, :].T, window).sum(axis=1).ravel()
        fake_dist = rolling_window(fake_data[:, i, :].T, window).sum(axis=1).ravel()
        emd += wasserstein_distance(real_dist, fake_dist)
    return emd
