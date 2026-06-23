from torch.nn.functional import one_hot
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from torch.utils.data import DataLoader, TensorDataset
from src.preprocess.gaussianize import *
from sklearn.preprocessing import StandardScaler

# import ml_collections
# import yaml
import random

def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


def count_parameters(model: torch.nn.Module) -> int:
    """

    Args:
        model (torch.nn.Module): input models
    Returns:
        int: number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_time_vector(size: int, length: int) -> torch.Tensor:
    return torch.linspace(1/length, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)


"""
class BaseAugmentation:
    pass

    def apply(self, *args: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError('Needs to be implemented by child.')


@dataclass
class AddTime(BaseAugmentation):

    def apply(self, x: torch.Tensor):
        t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
        return torch.cat([t, x], dim=-1)
"""


def AddTime(x):
    t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    return torch.cat([t, x], dim=-1)


def sample_indices(dataset_size, batch_size):
    indices = torch.from_numpy(np.random.choice(
        dataset_size, size=batch_size, replace=False)).cuda()
    # functions torch.-multinomial and torch.-choice are extremely slow -> back to numpy
    return indices.long()


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


def set_seed(seed: int):
    """ Sets the seed to a specified value. Needed for reproducibility of experiments. """
    torch.manual_seed(seed)
    np.random.seed(seed)


def save_obj(obj: object, filepath: str):
    """ Generic function to save an object with different methods. """
    if filepath.endswith('pkl'):
        saver = pickle.dump
    elif filepath.endswith('pt'):
        saver = torch.save
    else:
        raise NotImplementedError()
    with open(filepath, 'wb') as f:
        saver(obj, f)
    return 0


def load_obj(filepath):
    """ Generic function to load an object. """
    if filepath.endswith('pkl'):
        loader = pickle.load
    elif filepath.endswith('pt'):
        loader = torch.load
    elif filepath.endswith('json'):
        import json
        loader = json.load
    else:
        raise NotImplementedError()
    with open(filepath, 'rb') as f:
        return loader(f)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(
            m.weight.data, gain=nn.init.calculate_gain('relu'))
        try:
            # m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
        except:
            pass


def get_experiment_dir(config):
    if config.model_type == 'VAE':
        exp_dir = './numerical_results/{dataset}/algo_{gan}_Model_{model}_n_lag_{n_lags}_{seed}'.format(
            dataset=config.dataset, gan=config.algo, model=config.model, n_lags=config.n_lags, seed=config.seed)
    else:
        exp_dir = './numerical_results/{dataset}/algo_{gan}_G_{generator}_D_{discriminator}_includeD_{include_D}_n_lag_{n_lags}_{seed}'.format(
            dataset=config.dataset, gan=config.algo, generator=config.generator,
            discriminator=config.discriminator, include_D=config.include_D, n_lags=config.n_lags, seed=config.seed)
    os.makedirs(exp_dir, exist_ok=True)
    if config.train and os.path.exists(exp_dir):
        print("WARNING! The model exists in directory and will be overwritten")
    config.exp_dir = exp_dir


def loader_to_tensor(dl):
    tensor = []
    for x in dl:
        tensor.append(x[0])
    return torch.cat(tensor)


def loader_to_cond_tensor(dl, config):
    tensor = []
    for _, y in dl:
        tensor.append(y)

    return one_hot(torch.cat(tensor), config.num_classes).unsqueeze(1).repeat(1, config.n_lags, 1)

def combine_dls(dls):
    return torch.cat([loader_to_tensor(dl) for dl in dls])


def fake_loader(generator, x_past, n_lags, batch_size, **kwargs):
    """
    Helper function that transforms the generated data into dataloader, adapted from different generative models
    Parameters
    ----------
    generator: nn.module, trained generative model
    x_past: torch.tensor, real past path
    num_samples: int,  number of paths to be generated
    n_lags: int, the length of path to be generated
    batch_size: int, batch size for dataloader
    kwargs

    Returns
    Dataload of generated data
    -------

    """
    with torch.no_grad():
        fake_data_future = generator(n_lags, x_past)
        fake_data = torch.cat([x_past, fake_data_future], dim=1)
    return DataLoader(TensorDataset(fake_data), batch_size=batch_size)

# def load_config(file_dir: str):
#     with open(file_dir) as file:
#         config = ml_collections.ConfigDict(yaml.safe_load(file))
#     return config

# cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor
def deterministic_NeuralSort(s, tau):
    """
    s: input elements to be sorted. Shape: batch_size x n x 1
    tau: temperature for relaxation. Scalar. default: 0.01
    """
    n = s.size()[1]
    one = torch.ones((n, 1)).type(Tensor).to("cuda")
    A_s = torch.abs(s - s.permute(0, 2, 1)).to("cuda")
    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1).to("cuda")))
    scaling = (n + 1 - 2 * (torch.arange(n) + 1)).type(Tensor)
    C = torch.matmul(s, scaling.unsqueeze(0))
    P_max = (C - B).permute(0, 2, 1)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat
        

""" Compute the gradient penalty for WGAN-GP for conditional GANs with time series data """
def compute_gradient_penalty(discriminator, real_samples, fake_samples, z, i):
    alpha = torch.rand(real_samples.size(0), 1, 1, device=real_samples.device).expand_as(real_samples)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    critic_interpolates = discriminator(interpolates, z, i)
        
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size(), device=real_samples.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty

# def scaling(data):
#     """
#     Perform scaling for each feature and return the scaled data and the scalers.
    
#     Args:
#         data (numpy array): The input data in the format (n_samples, seq_len, n_features).        
    
#     Returns:
#         scaled_data (numpy array): The scaled log-return sequence data in the same shape as input.
#         scalers (list): A list of scalers for each feature (StandardScaler1, Gaussianize, StandardScaler2).
#     """
#     scaled_columns = []
#     scalers = []
    
#     for i in range(data.shape[2]):        
#         # Create scalers for each feature
#         standardScaler1 = StandardScaler()
#         standardScaler2 = StandardScaler()
#         gaussianize = Gaussianize()
        
#         # Reshape the feature data to 2D (n_samples * seq_len, 1)
#         log_returns = data[:, :, i].cpu().numpy().reshape(-1, 1)
        
#         # Apply scaling transformations: StandardScaler -> Gaussianize -> StandardScaler
#         log_returns_scaled = standardScaler2.fit_transform(
#             gaussianize.fit_transform(
#                 standardScaler1.fit_transform(log_returns)
#             )
#         )
        
#         # Store the scaled data (reshape back to (n_samples, seq_len))
#         scaled_columns.append(log_returns_scaled.reshape(data.shape[0], data.shape[1]))
        
#         # Store the scalers for inverse transformation later
#         scalers.append((standardScaler1, gaussianize, standardScaler2))
        
#     # Stack the scaled columns back to the original shape (n_samples, seq_len, n_features)
#     scaled_data = np.stack(scaled_columns, axis=2)
#     scaled_data = torch.tensor(scaled_data).float().cuda()
    
#     return scaled_data, scalers

# def inverse_scaling(scaled_data, scalers):
#     """
#     Perform inverse scaling to recover the original log returns.
    
#     Args:
#         scaled_data (numpy array): The scaled data in the format (n_samples, seq_len, n_features).
#         scalers (list): The list of scalers corresponding to each feature (StandardScaler1, Gaussianize, StandardScaler2).        
    
#     Returns:
#         original_data (numpy array): The data restored to its original scale.
#     """
#     org_columns = []
    
#     for i in range(scaled_data.shape[2]):
#         # Retrieve the saved scalers for each feature
#         standardScaler1, gaussianize, standardScaler2 = scalers[i]
        
#         # Reshape the scaled data back to 2D for inverse transformation
#         log_returns_scaled = scaled_data[:, :, i].reshape(-1, 1)
        
#         # Normalize y by removing its mean and dividing by its standard deviation
#         y = (log_returns_scaled - log_returns_scaled.mean(axis=0)) / log_returns_scaled.std(axis=0)
        
#         # Apply inverse transformations: StandardScaler2 -> Gaussianize -> StandardScaler1
#         y = standardScaler2.inverse_transform(y)
#         y = np.array([gaussianize.inverse_transform(np.expand_dims(x, 1)) for x in y]).squeeze()
#         log_returns_original = standardScaler1.inverse_transform(y.reshape(-1, 1))
        
#         # Reshape back to the original sequence format (n_samples, seq_len)
#         org_columns.append(log_returns_original.reshape(scaled_data.shape[0], scaled_data.shape[1]))
    
#     # Stack the original columns back to the original shape (n_samples, seq_len, n_features)
#     original_data = np.stack(org_columns, axis=2)
    
#     return original_data


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

def compute_gradient_penalty(discriminator, real_samples, fake_samples, i):
    alpha = torch.rand(real_samples.size(0), 1, 1, device=real_samples.device).expand_as(real_samples)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    critic_interpolates = discriminator(interpolates, i)
        
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients_norm = gradients.norm(2, dim=(1, 2))
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty



