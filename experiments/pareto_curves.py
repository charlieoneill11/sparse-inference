import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm

from models import SparseAutoEncoder, SparseCoding, MLP
from metrics import mcc
from utils import numpy_to_list, generate_data, reconstruction_loss_with_l1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_losses(model, X, S_true, l1_weight):
    if isinstance(model, SparseCoding):
        S = model.optimize_codes(X, num_iterations=10_000, l1_weight=l1_weight)
        #S, _ = model(X)
    else:
        S, _ = model(X)
    
    if isinstance(model, SparseAutoEncoder) or isinstance(model, MLP):
        X_recon = S @ model.decoder.weight.T
    else:  # SparseCoding
        X_recon = S @ model.D.T
    
    mse_loss = F.mse_loss(X_recon, X).item()
    l1_loss = torch.mean(torch.abs(S)).item()
    l0_loss_1e3 = torch.mean((S.abs() > 1e-3).float()).item()
    l0_loss_1e4 = torch.mean((S.abs() > 1e-4).float()).item()
    l0_loss_1e5 = torch.mean((S.abs() > 1e-5).float()).item()
    l0_loss_0 = torch.mean((S.abs() > 0).float()).item()
    mcc_val = mcc(S_true.cpu().numpy(), S.detach().cpu().numpy())

    print(f"Model: {type(model).__name__}, MSE Loss: {mse_loss:.6f}, L1 Loss: {l1_loss:.6f}")
    print(f"L0 Loss (>1e-3): {l0_loss_1e3:.6f}, L0 Loss (>1e-4): {l0_loss_1e4:.6f}")
    print(f"L0 Loss (>1e-5): {l0_loss_1e5:.6f}, L0 Loss (>0): {l0_loss_0:.6f}")
    print(f"MCC: {mcc_val:.6f}")
    
    return mse_loss, l1_loss, l0_loss_1e3, l0_loss_1e4, l0_loss_1e5, l0_loss_0, mcc_val

def train_model(model, X_train, S_train, X_test, S_test, l1_weight, num_step=100_000):
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    
    for _ in tqdm(range(num_step), desc=f"Training {type(model).__name__}", leave=False):
        S_, X_ = model(X_train)
        
        loss = reconstruction_loss_with_l1(X_train, X_, S_, l1_weight=l1_weight)
        optim.zero_grad()
        loss.backward()
        optim.step()

    print(f"Final loss: {loss.item():.4f}")
    
    return calculate_losses(model, X_test, S_test, l1_weight)

def run_experiment(N, M, K, num_data, l1_weights, num_runs=3, seed=20240926):
    results = {
        "SAE": {w: [] for w in l1_weights},
        "MLP": {w: [] for w in l1_weights},
        "SparseCoding": {w: [] for w in l1_weights}
    }
    
    for run in range(num_runs):
        run_seed = seed + run
        S, X, D = generate_data(N, M, K, num_data * 2, seed=run_seed)
        D = D.T
        S_train = S[:num_data].to(device)
        X_train = X[:num_data].to(device)
        S_test = S[num_data:].to(device)
        X_test = X[num_data:].to(device)
        D_true = D.to(device)
        
        for l1_weight in tqdm(l1_weights, desc=f"Run {run + 1}/{num_runs}"):
            # SparseCoding
            print(f"\nTraining SparseCoding with L1 weight: {l1_weight}")
            sc = SparseCoding(X_train, D_true, learn_D=True, seed=run_seed).to(device)
            sc_losses = train_model(sc, X_train, S_train, X_test, S_test, l1_weight)
            results["SparseCoding"][l1_weight].append(sc_losses)

            # SAE
            print(f"\nTraining SAE with L1 weight: {l1_weight}")
            sae = SparseAutoEncoder(M, N, D_true, learn_D=True, seed=run_seed).to(device)
            sae_losses = train_model(sae, X_train, S_train, X_test, S_test, l1_weight)
            results["SAE"][l1_weight].append(sae_losses)
            
            # MLP
            print(f"\nTraining MLP with L1 weight: {l1_weight}")
            mlp = MLP(M, N, 32, D_true, learn_D=True, seed=run_seed).to(device)  # Using hidden size of 32
            mlp_losses = train_model(mlp, X_train, S_train, X_test, S_test, l1_weight)
            results["MLP"][l1_weight].append(mlp_losses)
        
    
    # Average results
    avg_results = {}
    for method in results:
        avg_results[method] = {}
        for l1_weight in l1_weights:
            avg_results[method][l1_weight] = np.mean(results[method][l1_weight], axis=0).tolist()
    
    return avg_results

if __name__ == "__main__":
    N = 16  # number of sparse sources
    M = 8   # number of measurements
    K = 3   # number of active components
    num_data = 1024
    l1_weights = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    num_runs = 3
    seed = 20240926

    results = run_experiment(N, M, K, num_data, l1_weights, num_runs, seed)

    # Save results as JSON
    with open('results/l1_penalty_experiment.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Experiment completed. Results saved to 'results/l1_penalty_experiment.json'.")