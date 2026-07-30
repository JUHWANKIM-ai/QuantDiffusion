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


def count_parameters(model):
    # 학습 가능한(Trainable) 파라미터 수
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 전체(Total) 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())

    print(f"전체 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터 수: {trainable_params:,}")
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
