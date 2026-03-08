import torch
import numpy as np
from utils.gaussianize import *
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler


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
        windows[idx] = data[idx:idx + window_size]
    return windows


def set_seed(seed: int):
    """ Sets the seed to a specified value. Needed for reproducibility of experiments. """
    torch.manual_seed(seed)
    np.random.seed(seed)

""" Scale the log returns using Gaussianize and StandardScaler """
def scaling(data):
    scalers = [] 
    scaled_data = []

    for i in range(data.shape[1]):  # Iterate over features
        standardScaler1 = StandardScaler()
        standardScaler2 = StandardScaler()
        gaussianize = Gaussianize()

        # Scale data for the current feature
        feature_data = data[:, i].reshape(-1, 1)  # Reshape to (samples, 1)
        feature_scaled = standardScaler2.fit_transform(
            gaussianize.fit_transform(
                standardScaler1.fit_transform(feature_data)
            )
        )

        # Append scalers and scaled data
        scalers.append((standardScaler1, standardScaler2, gaussianize))
        scaled_data.append(feature_scaled.flatten())  # Flatten back to (samples,)

    # Combine scaled features back into a single array
    scaled_data = np.array(scaled_data).T  # Transpose to shape (samples, features)
    return scaled_data, scalers


""" Inverse the scaling process for all features """
def inverse_scaling(y, scalers):
    y = y.cpu().detach().numpy()  # Convert to NumPy for compatibility with scalers    
    y_original = np.zeros_like(y)  # Placeholder for original data

    for idx in range(y.shape[1]):
        
        standardScaler1, standardScaler2, gaussianize = scalers[idx]
        
        y_feature = y[:, idx, :]  # Shape: (batch_size, seq_len)

        # Normalize by batch mean and std for the current feature
        EPS = 1e-8
        y_feature = (y_feature - y_feature.mean(axis=0, keepdims=True)) / (y_feature.std(axis=0, keepdims=True) + EPS)

        # Perform inverse scaling step-by-step
        y_feature = standardScaler2.inverse_transform(y_feature)
        y_feature = np.array([
            gaussianize.inverse_transform(np.expand_dims(sample, 1)) for sample in y_feature
        ]).squeeze()
        y_feature = standardScaler1.inverse_transform(y_feature)

        # Assign back the feature's inverse transformed data
        y_original[:, idx, :] = y_feature

    return y_original


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