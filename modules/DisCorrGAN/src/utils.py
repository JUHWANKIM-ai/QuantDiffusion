from torch.nn.functional import one_hot
import torch, math
import torch.nn as nn
import numpy as np
import pickle
import os
from torch.utils.data import DataLoader, TensorDataset
from src.preprocess.gaussianize import *
from sklearn.preprocessing import StandardScaler

import ml_collections
import yaml
import random


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

def loader_to_tensor(dl):
    tensor = []
    for x in dl:
        tensor.append(x[0])
    return torch.cat(tensor)


def load_config(file_dir: str):
    with open(file_dir, encoding="utf-8") as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    return config
        

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

def inverse_scaling(y, scalers, idx):
    y = y.cpu().detach().numpy()  # Convert to NumPy for compatibility with scalers    
    y_original = np.zeros_like(y)  # Placeholder for original data
        
    standardScaler1, standardScaler2, gaussianize = scalers[idx]
    
    y_feature = y[:, 0, :]  # Shape: (batch_size, seq_len)

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
    y_original[:, 0, :] = y_feature

    return y_original