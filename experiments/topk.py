# import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F
# import json
# from tqdm import tqdm
# from typing import Callable

# from metrics import mcc
# from utils import numpy_to_list, generate_data, reconstruction_loss, reconstruction_loss_with_l1
# from models import SparseCoding

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def apply_topk(S, k):
#     topk = torch.topk(S.abs(), k=k, dim=-1)
#     result = torch.zeros_like(S)
#     result.scatter_(-1, topk.indices, S.gather(-1, topk.indices))
#     return result

# def optimize_codes_topk(model, X, k, num_iterations=10_000, lr=3e-3):
#     log_S = nn.Parameter(data=-10 * torch.ones(X.shape[0], model.D.shape[1]), requires_grad=True)
#     opt = torch.optim.Adam([log_S], lr=lr)

#     for _ in range(num_iterations):
#         S = apply_topk(torch.exp(log_S), k)
#         X_ = S @ model.D.T
#         if model.use_bias:
#             X_ += model.bias
#         loss = reconstruction_loss(X, X_)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#         #print(f"Loss: {loss.item():.4f}")

#     return apply_topk(torch.exp(log_S.detach()), k)

# def calculate_losses(model, X, S_true, l1_weight=0.0, k=None, method='original'):
#     if method == 'original':
#         S = model.optimize_codes(X, num_iterations=10_000, l1_weight=l1_weight)
#     elif method == 'inference_topk':
#         S = model.optimize_codes(X, num_iterations=10_000, l1_weight=l1_weight)
#         S = apply_topk(S, k)
#     elif method == 'optimize_topk':
#         S = optimize_codes_topk(model, X, k, num_iterations=10_000)
    
#     X_recon = S @ model.D.T
#     if model.use_bias:
#         X_recon += model.bias
    
#     mse_loss = F.mse_loss(X_recon, X).item()
#     l0_loss = torch.mean((S.abs() > 0).float()).item()
#     mcc_val = mcc(S_true.cpu().numpy(), S.detach().cpu().numpy())

#     print(f"Method: {method}, MSE Loss: {mse_loss:.6f}, L0 Loss: {l0_loss:.6f}")
#     print(f"MCC: {mcc_val:.6f}")
    
#     return mse_loss, l0_loss, mcc_val

# def train_model(model, X_train, S_train, X_test, S_test, l1_weight, num_step=100_000):
#     optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    
#     for _ in tqdm(range(num_step), desc=f"Training SparseCoding", leave=False):
#         S, X_ = model(X_train)
#         loss = reconstruction_loss_with_l1(X_train, X_, S, l1_weight)
#         optim.zero_grad()
#         loss.backward()
#         optim.step()

#     print(f"Final loss: {loss.item():.4f}")
    
#     return calculate_losses(model, X_test, S_test, l1_weight, method='original')

# def run_experiment(N, M, K, num_data, l1_weights, k_values, num_runs=3, seed=20240926):
#     results = {
#         "SparseCoding_L1": {w: [] for w in l1_weights},
#         "SparseCoding_InferenceTopK": {w: {k: [] for k in k_values} for w in l1_weights},
#         "SparseCoding_OptimizeTopK": {w: {k: [] for k in k_values} for w in l1_weights}
#     }
    
#     for run in range(num_runs):
#         run_seed = seed + run
#         S, X, D = generate_data(N, M, K, num_data * 2, seed=run_seed)
#         D = D.T
#         S_train = S[:num_data].to(device)
#         X_train = X[:num_data].to(device)
#         S_test = S[num_data:].to(device)
#         X_test = X[num_data:].to(device)
#         D_true = D.to(device)

#         for l1_weight in tqdm(l1_weights, desc=f"Run {run + 1}/{num_runs} - L1 weights"):
#             print(f"\nTraining SparseCoding with L1 weight: {l1_weight}")
#             sc = SparseCoding(X_train, D_true, learn_D=True, seed=run_seed).to(device)
#             sc_losses = train_model(sc, X_train, S_train, X_test, S_test, l1_weight)
#             results["SparseCoding_L1"][l1_weight].append(sc_losses)
            
#             for k in k_values:
#                 print(f"\nEvaluating InferenceTopK with L1 weight: {l1_weight} and k: {k}")
#                 sc_inftopk_losses = calculate_losses(sc, X_test, S_test, l1_weight, k, method='inference_topk')
#                 results["SparseCoding_InferenceTopK"][l1_weight][k].append(sc_inftopk_losses)

#                 print(f"\nEvaluating OptimizeTopK with L1 weight: {l1_weight} and k: {k}")
#                 sc_opttopk_losses = calculate_losses(sc, X_test, S_test, l1_weight, k, method='optimize_topk')
#                 results["SparseCoding_OptimizeTopK"][l1_weight][k].append(sc_opttopk_losses)
    
#     # Average results
#     avg_results = {
#         "SparseCoding_L1": {},
#         "SparseCoding_InferenceTopK": {},
#         "SparseCoding_OptimizeTopK": {}
#     }
#     for l1_weight in l1_weights:
#         avg_results["SparseCoding_L1"][l1_weight] = np.mean(results["SparseCoding_L1"][l1_weight], axis=0).tolist()
#         avg_results["SparseCoding_InferenceTopK"][l1_weight] = {}
#         avg_results["SparseCoding_OptimizeTopK"][l1_weight] = {}
#         for k in k_values:
#             avg_results["SparseCoding_InferenceTopK"][l1_weight][k] = np.mean(results["SparseCoding_InferenceTopK"][l1_weight][k], axis=0).tolist()
#             avg_results["SparseCoding_OptimizeTopK"][l1_weight][k] = np.mean(results["SparseCoding_OptimizeTopK"][l1_weight][k], axis=0).tolist()
    
#     return avg_results

# if __name__ == "__main__":
#     N = 16  # number of sparse sources
#     M = 8   # number of measurements
#     K = 3   # number of active components
#     num_data = 1024
#     l1_weights = [0.005, 0.05, 0.5, 5.0]
#     k_values = [1, 3, 5, 10]
#     num_runs = 3
#     seed = 20240926

#     results = run_experiment(N, M, K, num_data, l1_weights, k_values, num_runs, seed)

#     # Save results as JSON
#     with open('results/sparse_coding_l1_vs_inferencetopk_vs_optimizetopk_experiment.json', 'w') as f:
#         json.dump(results, f, indent=2)

#     print("Experiment completed. Results saved to 'results/sparse_coding_l1_vs_inferencetopk_vs_optimizetopk_experiment.json'.")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm
from typing import Callable

from metrics import mcc
from utils import numpy_to_list, generate_data, reconstruction_loss, reconstruction_loss_with_l1
from models import SparseCoding, SparseAutoEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def apply_topk(S, k):
#     topk = torch.topk(S.abs(), k=k, dim=-1)
#     result = torch.zeros_like(S)
#     result.scatter_(-1, topk.indices, S.gather(-1, topk.indices))
#     return result

# def optimize_codes_topk(model, X, k, num_iterations=10_000, lr=1e-3):
#     log_S = nn.Parameter(data=-10 * torch.ones(X.shape[0], model.D.shape[1]), requires_grad=True)
#     opt = torch.optim.Adam([log_S], lr=lr)

#     for _ in range(num_iterations):
#         S = apply_topk(torch.exp(log_S), k)
#         X_ = S @ model.D.T
#         if model.use_bias:
#             X_ += model.bias
#         loss = reconstruction_loss(X, X_)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()

#     return apply_topk(torch.exp(log_S.detach()), k)

def optimize_codes_topk(model, X, k, num_iterations=10_000, lr=9e-1):
    S = nn.Parameter(data=torch.zeros(X.shape[0], model.D.shape[1]), requires_grad=True)
    opt = torch.optim.Adam([S], lr=lr)

    for _ in range(num_iterations):
        S_topk = apply_topk(S, k)
        X_ = S_topk @ model.D.T
        if model.use_bias:
            X_ += model.bias
        loss = reconstruction_loss(X, X_)
        opt.zero_grad()
        loss.backward()
        print(f"Loss: {loss.item():.4f}")
        opt.step()

    return apply_topk(S.detach(), k)

def apply_topk(S, k):
    topk = torch.topk(S.abs(), k=k, dim=-1)
    result = torch.zeros_like(S)
    result.scatter_(-1, topk.indices, S.gather(-1, topk.indices))
    return result

def calculate_losses(model, X, S_true, l1_weight=0.0, k=None, method='original'):
    if isinstance(model, SparseCoding):
        if method == 'original':
            S = model.optimize_codes(X, num_iterations=10_000, l1_weight=l1_weight)
        elif method == 'inference_topk':
            S = model.optimize_codes(X, num_iterations=10_000, l1_weight=l1_weight)
            S = apply_topk(S, k)
        elif method == 'optimize_topk':
            S = optimize_codes_topk(model, X, k, num_iterations=1_000)
        
        X_recon = S @ model.D.T
        if model.use_bias:
            X_recon += model.bias
    elif isinstance(model, SparseAutoEncoder):
        S, X_recon = model(X)
    
    mse_loss = F.mse_loss(X_recon, X).item()
    l0_loss = torch.mean((S.abs() > 0).float()).item()
    mcc_val = mcc(S_true.cpu().numpy(), S.detach().cpu().numpy())

    print(f"Method: {method}, MSE Loss: {mse_loss:.6f}, L0 Loss: {l0_loss:.6f}")
    print(f"MCC: {mcc_val:.6f}")
    
    return mse_loss, l0_loss, mcc_val

def train_model(model, X_train, S_train, X_test, S_test, l1_weight, num_step=100_000):
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    
    for _ in tqdm(range(num_step), desc=f"Training {type(model).__name__}", leave=False):
        if isinstance(model, SparseCoding):
            S, X_ = model(X_train)
            loss = reconstruction_loss_with_l1(X_train, X_, S, l1_weight)
        elif isinstance(model, SparseAutoEncoder):
            S, X_ = model(X_train)
            loss = reconstruction_loss_with_l1(X_train, X_, S, l1_weight)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

    print(f"Final loss: {loss.item():.4f}")
    
    return calculate_losses(model, X_test, S_test, l1_weight, method='original')

def run_experiment(N, M, K, num_data, l1_weights, k_values, num_runs=3, seed=20240926):
    results = {
        "SparseCoding_L1": {w: [] for w in l1_weights},
        "SparseCoding_InferenceTopK": {w: {k: [] for k in k_values} for w in l1_weights},
        "SparseCoding_OptimizeTopK": {w: {k: [] for k in k_values} for w in l1_weights},
        "SparseAutoEncoder_L1": {w: [] for w in l1_weights}
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

        for l1_weight in tqdm(l1_weights, desc=f"Run {run + 1}/{num_runs} - L1 weights"):
            print(f"\nTraining SparseAutoEncoder with L1 weight: {l1_weight}")
            sae = SparseAutoEncoder(M, N, D_true, learn_D=True, seed=run_seed).to(device)
            sae_losses = train_model(sae, X_train, S_train, X_test, S_test, l1_weight)
            results["SparseAutoEncoder_L1"][l1_weight].append(sae_losses)
    
            print(f"\nTraining SparseCoding with L1 weight: {l1_weight}")
            sc = SparseCoding(X_train, D_true, learn_D=True, seed=run_seed).to(device)
            sc_losses = train_model(sc, X_train, S_train, X_test, S_test, l1_weight)
            results["SparseCoding_L1"][l1_weight].append(sc_losses)
            
            for k in k_values:
                print(f"\nEvaluating InferenceTopK with L1 weight: {l1_weight} and k: {k}")
                sc_inftopk_losses = calculate_losses(sc, X_test, S_test, l1_weight, k, method='inference_topk')
                results["SparseCoding_InferenceTopK"][l1_weight][k].append(sc_inftopk_losses)

                print(f"\nEvaluating OptimizeTopK with L1 weight: {l1_weight} and k: {k}")
                sc_opttopk_losses = calculate_losses(sc, X_test, S_test, l1_weight, k, method='optimize_topk')
                results["SparseCoding_OptimizeTopK"][l1_weight][k].append(sc_opttopk_losses)

            
    # Average results
    avg_results = {
        "SparseCoding_L1": {},
        "SparseCoding_InferenceTopK": {},
        "SparseCoding_OptimizeTopK": {},
        "SparseAutoEncoder_L1": {}
    }
    for l1_weight in l1_weights:
        avg_results["SparseCoding_L1"][l1_weight] = np.mean(results["SparseCoding_L1"][l1_weight], axis=0).tolist()
        avg_results["SparseAutoEncoder_L1"][l1_weight] = np.mean(results["SparseAutoEncoder_L1"][l1_weight], axis=0).tolist()
        avg_results["SparseCoding_InferenceTopK"][l1_weight] = {}
        avg_results["SparseCoding_OptimizeTopK"][l1_weight] = {}
        for k in k_values:
            avg_results["SparseCoding_InferenceTopK"][l1_weight][k] = np.mean(results["SparseCoding_InferenceTopK"][l1_weight][k], axis=0).tolist()
            avg_results["SparseCoding_OptimizeTopK"][l1_weight][k] = np.mean(results["SparseCoding_OptimizeTopK"][l1_weight][k], axis=0).tolist()
    
    return avg_results

if __name__ == "__main__":
    N = 16  # number of sparse sources
    M = 8   # number of measurements
    K = 3   # number of active components
    num_data = 1024
    l1_weights = [0.005, 0.05, 0.5, 5.0]
    k_values = [1, 3, 5, 10]
    num_runs = 2
    seed = 20240926

    results = run_experiment(N, M, K, num_data, l1_weights, k_values, num_runs, seed)

    # Save results as JSON
    with open('results/sparse_coding_autoencoder_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Experiment completed. Results saved to 'results/sparse_coding_autoencoder_comparison.json'.")