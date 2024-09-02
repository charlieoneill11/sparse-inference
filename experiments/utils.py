import numpy as np
import torch
from torch.nn import functional as F


def reconstruction_loss_with_l1(X, X_, S_, l1_weight=0.01):
    recon_loss = F.mse_loss(X_, X)
    l1_loss = l1_weight * torch.mean(torch.abs(S_))
    return recon_loss + l1_loss

def numpy_to_list(obj):
    # Function to convert NumPy arrays to lists recursively
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    return obj

def generate_data(N, M, K, num_data, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    S = []
    for _ in range(num_data):
        s = np.abs(np.random.normal(0, 1, N))
        mask = np.zeros(N)
        mask[np.random.choice(N, K, replace=False)] = 1
        s *= mask
        S.append(s)
    S = np.array(S)
    
    D = torch.randn(N, M, dtype=torch.float32)
    D /= torch.linalg.norm(D, dim=1, keepdim=True)
    
    X = torch.tensor(S, dtype=torch.float32) @ D
    
    return torch.tensor(S, dtype=torch.float32), X, D