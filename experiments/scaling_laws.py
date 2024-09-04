import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import SparseAutoEncoder, MLP, SparseCoding
from metrics import mcc, greedy_mcc
from utils import numpy_to_list, generate_data, reconstruction_loss_with_l1
from calculate_flops import (calculate_sae_training_flops, calculate_sae_inference_flops, 
                             calculate_mlp_training_flops, calculate_mlp_inference_flops, 
                             calculate_sparse_coding_training_flops, calculate_sparse_coding_inference_flops)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_experiment(model, X_train, S_train, X_test, S_test, D_true, num_step=30000, log_step=100):
    if isinstance(model, SparseCoding):
        return train_sparse_coding(model, X_train, S_train, X_test, S_test, D_true, num_step=num_step, log_step=log_step)
    else:
        return train(model, X_train, S_train, X_test, S_test, D_true, num_step=num_step, log_step=log_step)

def train_sparse_coding(model, X_train, S_train, X_test, S_test, D_true, lr=1e-3, num_step=30000, log_step=100):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_mcc = 0
    for i in range(num_step):
        S_, X_ = model(X_train)
        loss = reconstruction_loss_with_l1(X_train, X_, S_)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if i % log_step == 0:
            with torch.no_grad():
                S_test_opt = model.optimize_codes(X_test, num_iterations=1000)
            mcc_test = mcc(S_test.cpu().numpy(), S_test_opt.cpu().numpy())
            if mcc_test > best_mcc:
                best_mcc = mcc_test
    return best_mcc

def train(model, X_train, S_train, X_test, S_test, D_true, lr=1e-3, num_step=30000, log_step=100):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_mcc = 0
    for i in range(num_step):
        S_, X_ = model(X_train)
        loss = reconstruction_loss_with_l1(X_train, X_, S_)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if i % log_step == 0:
            with torch.no_grad():
                S_, X_ = model(X_test)
            mcc_test = mcc(S_test.detach().cpu().numpy(), S_.detach().cpu().numpy())
            if mcc_test > best_mcc:
                best_mcc = mcc_test
    return best_mcc

def scaling_laws_experiment(N_values, M_values, K, num_data=1024, num_step=30000, log_step=100, seed=20240927):
    results = {method: {M: [] for M in M_values} for method in ['SAE', 'MLP', 'SparseCoding']}
    
    for M in M_values:
        for N in N_values:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Generate data
            S, X, D = generate_data(N, M, K, num_data * 2, seed=seed)
            S_train, X_train = S[:num_data].to(device), X[:num_data].to(device)
            S_test, X_test = S[num_data:].to(device), X[num_data:].to(device)
            D_true = D.to(device)
            
            # SAE
            sae = SparseAutoEncoder(M, N, D.to(device), learn_D=True).to(device)
            sae_mcc = run_experiment(sae, X_train, S_train, X_test, S_test, D_true, num_step=num_step, log_step=log_step)
            results['SAE'][M].append(sae_mcc)
            
            # MLP (using hidden layer size of 256)
            mlp = MLP(M, N, 256, D.to(device), learn_D=True).to(device)
            mlp_mcc = run_experiment(mlp, X_train, S_train, X_test, S_test, D_true, num_step=num_step, log_step=log_step)
            results['MLP'][M].append(mlp_mcc)
            
            # Sparse Coding
            sc = SparseCoding(X_test, D.to(device), learn_D=True).to(device)
            sc_mcc = run_experiment(sc, X_train, S_train, X_test, S_test, D_true, num_step=num_step, log_step=log_step)
            results['SparseCoding'][M].append(sc_mcc)
            
            print(f"Completed N={N}, M={M}")
    
    return results

# Parameters
N_values = [16, 32, 64, 128, 256]
M_values = [8, 16, 32]
K = 3
num_data = 1024
num_step = 30000
log_step = 100
seed = 20240927

# Run experiment
results = scaling_laws_experiment(N_values, M_values, K, num_data, num_step, log_step, seed)

# Plot results
plt.figure(figsize=(12, 8))
colors = {'SAE': 'blue', 'MLP': 'green', 'SparseCoding': 'red'}
markers = {8: 'o', 16: 's', 32: '^'}

for method in results:
    for M in M_values:
        plt.plot(N_values, results[method][M], label=f'{method} (M={M})', 
                 color=colors[method], marker=markers[M], linestyle='--')

plt.xscale('log')
plt.xlabel('N (number of sparse sources)')
plt.ylabel('MCC')
plt.title(f'Scaling Laws in Sparse Coding (K={K})')
plt.legend()
plt.grid(True)
plt.savefig('scaling_laws_plot.png')
plt.close()

# Save results
with open('scaling_laws_results.json', 'w') as f:
    json.dump(results, f)

print("Experiment completed. Results saved to 'scaling_laws_results.json' and plot saved as 'scaling_laws_plot.png'.")