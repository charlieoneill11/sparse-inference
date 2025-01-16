"""
known_Z.py
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm

from metrics import mcc, corr
from utils import numpy_to_list, generate_data, reconstruction_loss_with_l1
from calculate_flops import (calculate_sae_training_flops, calculate_sae_inference_flops, calculate_mlp_training_flops, 
                            calculate_mlp_inference_flops, calculate_sparse_coding_training_flops, calculate_sparse_coding_inference_flops)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def pairwise_corr(z, z_):
    return np.mean([corr(a, b)[0] for a, b in zip(z.T, z_.T)])

def cossim(z, z_):
    return -F.cosine_similarity(z.T, z_.T).mean()

def train(model, X_train, S_train, X_test, S_test, lr=1e-3, num_step=10000, log_step=100, verbose=0):
    criterion = cossim
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log = {'step': [], 'mcc_train': [], 'loss_train': [], 'mcc_test': [], 'loss_test': [], 'flops': []}
    
    # Calculate initial FLOPs
    if isinstance(model, nn.Sequential) and len(model) == 2:  # SAE
        total_flops = calculate_sae_training_flops(M, N, num_data, 0)  # 0 steps initially
    else:  # MLP
        h = model[0].out_features
        total_flops = calculate_mlp_training_flops(M, h, N, num_data, 0)  # 0 steps initially
    
    for i in tqdm(range(num_step), disable=not verbose):
        S_ = model(X_train)
        loss = criterion(S_train, S_)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if i > 0 and not i % log_step:
            log['step'].append(i)
            log['loss_train'].append(loss.item())
            log['mcc_train'].append(pairwise_corr(S_train.detach().cpu().numpy(), S_.detach().cpu().numpy()))
            with torch.no_grad():
                S_ = model(X_test)
                loss = criterion(S_test, S_)
            log['loss_test'].append(loss.item())
            log['mcc_test'].append(pairwise_corr(S_test.detach().cpu().numpy(), S_.detach().cpu().numpy()))
            
            # Calculate and log total FLOPs up to this point
            if isinstance(model, nn.Sequential) and len(model) == 2:  # SAE
                total_flops = calculate_sae_training_flops(M, N, num_data, i+1)
            else:  # MLP
                h = model[0].out_features
                total_flops = calculate_mlp_training_flops(M, h, N, num_data, i+1)
            log['flops'].append(total_flops)

            print(f"Step {i+1}, Loss: {loss.item()}, MCC: {log['mcc_train'][-1]}, FLOPs: {total_flops}")    
    
    return log

def run_experiment(model, X_train, S_train, X_test, S_test, num_step=20000, log_step=100, seed=20240625):
    torch.manual_seed(seed)
    log = train(model, X_train, S_train, X_test, S_test, num_step=num_step, log_step=log_step)
    return log

def average_logs(logs):
    avg_log = {
        'step': logs[0]['step'],
        'mcc_test': np.mean([log['mcc_test'] for log in logs], axis=0),
        'mcc_test_std': np.std([log['mcc_test'] for log in logs], axis=0),
        'flops': logs[0]['flops']  # FLOPs are deterministic, so we can just take the first run
    }
    return avg_log

# Parameters
N = 1000  # number of sparse sources
M = 200   # number of measurements
K = 20   # number of active components
hidden_layers = [32, 256]  # list of hidden layer widths
num_runs = 5
num_data = 500000
num_step = 20000
log_step = 100
seed = 20240625

# Generate data
S, X, D = generate_data(N, M, K, num_data * 2, seed=seed)
S_train = S[:num_data].to(device)
X_train = X[:num_data].to(device)
S_test = S[num_data:].to(device)
X_test = X[num_data:].to(device)

# Run experiments
logs_sae = []
logs_mlps = {h: [] for h in hidden_layers}

for i in tqdm(range(num_runs), desc="Running experiments"):
    run_seed = seed + i
    
    SAE = nn.Sequential(nn.Linear(M, N), nn.ReLU()).to(device)
    logs_sae.append(run_experiment(SAE, X_train, S_train, X_test, S_test, seed=run_seed))
    
    for h in hidden_layers:
        MLP = nn.Sequential(nn.Linear(M, h), nn.ReLU(), nn.Linear(h, N), nn.ReLU()).to(device)
        logs_mlps[h].append(run_experiment(MLP, X_train, S_train, X_test, S_test, seed=run_seed))

# Average logs
avg_sae = average_logs(logs_sae)
avg_mlps = {h: average_logs(logs) for h, logs in logs_mlps.items()}

# Save results as JSON
results = {
    "SAE": avg_sae,
    "MLPs": avg_mlps
}

with open('results/fixed_Z_flops.json', 'w') as f:
    json.dump(numpy_to_list(results), f)

print("Experiment completed. Results saved to 'results/fixed_Z_flops.json'.")