import numpy as np
import torch
from torch.nn import functional as F


def reconstruction_loss_with_l1(X, X_, S_, l1_weight=0.01):
    recon_loss = F.mse_loss(X_, X)
    l1_loss = l1_weight * torch.mean(torch.abs(S_))
    return recon_loss + l1_loss

def reconstruction_loss(X, X_):
    return F.mse_loss(X_, X)

def numpy_to_list(obj):
    # Function to convert NumPy arrays to lists recursively
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    return obj

def generate_data(N, M, K, num_data, seed, alpha=1.0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create Zipf-like weights for each dimension
    weights = 1 / np.arange(1, N + 1) ** alpha  # Zipf distribution
    weights = weights / weights.sum()  # Normalize to probability distribution
    
    S = []
    for _ in range(num_data):
        s = np.abs(np.random.normal(0, 1, N))
        
        # Instead of uniform random choice, use weighted sampling
        mask = np.zeros(N)
        mask[np.random.choice(N, K, replace=False, p=weights)] = 1
        
        # Scale the values by the weights to introduce frequency bias
        # in magnitude as well as occurrence
        s *= mask * (weights + 0.1)  # Adding small constant to avoid zero values
        S.append(s)
    S = np.array(S)
    
    D = torch.randn(N, M, dtype=torch.float32)
    D /= torch.linalg.norm(D, dim=1, keepdim=True)
    
    X = torch.tensor(S, dtype=torch.float32) @ D
    
    return torch.tensor(S, dtype=torch.float32), X, D