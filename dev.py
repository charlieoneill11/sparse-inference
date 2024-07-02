import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as corr
import yaml

from models import SparseCoding

# Parameters
N = 16  # number of sparse sources
K = 3  # number of active components
M = 8  # number of measurements
seed = 20240625
num_data = 1024
num_step = 20000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sample_signal():
    s = np.random.normal(0, 1, N)
    s = np.abs(s)
    ind = np.random.choice(N, K, replace=False)
    mask = np.zeros(N)
    mask[ind] = 1
    s *= mask
    return s

def criterion(S_, X, X_, l1_weight):
    loss = torch.sum((X - X_) ** 2) + l1_weight * torch.sum(torch.abs(S_))
    return loss

def train(model, X, lr, l1_weight):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(num_step):
        S_, X_ = model.forward(X)
        loss = criterion(S_, X, X_, l1_weight=l1_weight)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i > 0 and not i % 1000:
            print('step', i, 'loss', loss.item())
    return model

def main():
    # Load configuration
    with open('train_configs.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    lr = float(config['SparseCoding']['lr'])
    l1_weight = float(config['SparseCoding']['l1_weight'])

    # Generate sample data
    np.random.seed(seed)
    S = np.array([sample_signal() for _ in range(num_data)])
    
    torch.manual_seed(seed)
    D = torch.randn(N, M, dtype=torch.float32).to(device)
    D /= torch.linalg.norm(D, dim=1, keepdim=True)
    X = torch.tensor(S, dtype=torch.float32).to(device) @ D

    # Initialize and train model
    model = SparseCoding(S, D, learn_D=False, seed=seed).to(device)
    trained_model = train(model, X, lr, l1_weight)

    # Generate new data for inference
    new_S = np.array([sample_signal() for _ in range(100)])
    new_X = torch.tensor(new_S, dtype=torch.float32).to(device) @ D

    # Perform inference
    inferred_S = trained_model.infer(new_X)
    
    # Calculate correlation
    correlations = [corr(new_S[i], inferred_S[i].cpu().numpy())[0] for i in range(100)]
    print(f"Average correlation: {np.mean(correlations):.4f}")

if __name__ == "__main__":
    main()