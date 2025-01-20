"""
known_Z.py
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from metrics import mcc, corr
from utils import numpy_to_list, generate_data, reconstruction_loss_with_l1
from calculate_flops import (calculate_sae_training_flops, calculate_sae_inference_flops, calculate_mlp_training_flops, 
                            calculate_mlp_inference_flops, calculate_sparse_coding_training_flops, calculate_sparse_coding_inference_flops)
from time import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def normalize(x, threshold=1e-10):
    x -= torch.mean(x, axis=0, keepdims=True)
    norms = torch.linalg.norm(x, axis=0, keepdims=True)
    norms[norms < threshold] = 1
    return x / norms

def pairwise_corr(z, z_):
    "vectorized and on GPU, much faster than old."
    with torch.no_grad():
        corrs = torch.sum(normalize(z) * normalize(z_), axis=0)
        mcc = torch.mean(corrs)
    return mcc

def cossim(z, z_):
    return -F.cosine_similarity(z.T, z_.T).mean()

def create_data_loaders(X_train, S_train, X_test, S_test, batch_size):
    train_dataset = TensorDataset(X_train, S_train)
    test_dataset = TensorDataset(X_test, S_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train(model, X_train, S_train, X_test, S_test, lr=1e-3, num_epochs=100):
    criterion = cossim
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log = {'step': [], 'mcc_train': [], 'loss_train': [], 'mcc_test': [], 'loss_test': [], 'flops': []}

    # Calculate initial FLOPs
    if isinstance(model, nn.Sequential) and len(model) == 2:  # SAE
        total_flops = calculate_sae_training_flops(M, N, num_data, 0)  # 0 steps initially
    else:  # MLP
        h = model[0].out_features
        total_flops = calculate_mlp_training_flops(M, h, N, num_data, 0)  # 0 steps initially

    train_loader, test_loader = create_data_loaders(X_train, S_train, X_test, S_test, batch_size=1024)


    model.train()
    train_loss = 0
    train_mcc = 0
    for epoch in range(num_epochs):
        t0 = time()
        for X_batch, S_batch in train_loader:
            S_pred = model(X_batch)
            loss = criterion(S_batch, S_pred)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item()

            mcc = pairwise_corr(S_batch.detach(), S_pred.detach())
            train_mcc += mcc.item()

        train_mcc /= len(train_loader)
        train_loss /= len(train_loader)

        # if not (epoch + 1) % log_epoch:
        log['step'].append(i)
        log['loss_train'].append(train_loss)
        log['mcc_train'].append(train_mcc)
        model.eval()
        with torch.no_grad():
            test_loss = 0
            test_mcc = 0
            for X_batch, S_batch in test_loader:
                S_pred = model(X_batch)
                loss = criterion(S_batch, S_pred)
                mcc = pairwise_corr(S_batch.detach(), S_pred.detach())
                test_loss += loss.item()
                test_mcc += mcc.item()
        
        test_loss /= len(test_loader)
        test_mcc /= len(test_loader)
        log['loss_test'].append(test_loss)
        log['mcc_test'].append(test_mcc)
        
        # Calculate and log total FLOPs up to this point
        if isinstance(model, nn.Sequential) and len(model) == 2:  # SAE
            total_flops = calculate_sae_training_flops(M, N, num_data, epoch+1)
        else:  # MLP
            h = model[0].out_features
            total_flops = calculate_mlp_training_flops(M, h, N, num_data, epoch+1)
        log['flops'].append(total_flops)

        print(f"Epoch {epoch+1}, Loss: {loss.item()}, MCC: {log['mcc_train'][-1]}, FLOPs: {total_flops}, took=%.2fs" % (time() - t0))   
    
    return log

def run_experiment(model, X_train, S_train, X_test, S_test, num_epochs=100, seed=20240625):
    torch.manual_seed(seed)
    log = train(model, X_train, S_train, X_test, S_test, num_epochs=num_epochs)
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