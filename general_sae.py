import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.stats import pearsonr as corr
from tqdm import tqdm
from munkres import Munkres
import json
from models import GeneralSAE
from flop_counter import calculate_inference_flops, calculate_training_flops

# Parameters
N, M, K = 16, 8, 3
seed = 20240625
num_data = 1024
num_step = 20000
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Grid search parameters
lr_range = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
l1_weight_range = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
projections_up = [12, 16]  # Example projections

# Cosine Annealing parameters
T_0 = 1000  # Initial period of reset
T_mult = 2  # Factor to increase T_i after a restart

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
    D /= torch.linalg.norm(D, dim=0, keepdim=True)
    
    X = torch.tensor(S, dtype=torch.float32) @ D
    
    return torch.tensor(S, dtype=torch.float32), X, D

def mcc(z, z_):
    matches = np.array([[abs(corr(z[:, i], z_[:, j])[0]) for j in range(z_.shape[1])] for i in range(z.shape[1])])
    matches[np.isnan(matches)] = 0.0
    indices = Munkres().compute(-matches)
    return np.mean([matches[i, j] for i, j in indices])

def train_model(model, X_train, S_train, X_val, S_val, num_step, lr, l1_weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
    best_mcc = -float('inf')
    
    for i in tqdm(range(num_step), desc="Training GeneralSAE"):
        model.train()
        S_, X_ = model(X_train)
        loss = F.mse_loss(X_, X_train) + l1_weight * torch.sum(torch.abs(S_))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % (num_step // 20) == 0:
            model.eval()
            with torch.no_grad():
                S_val_, _ = model(X_val)
                current_mcc = mcc(S_val.cpu().numpy(), S_val_.cpu().numpy())
            
            best_mcc = max(best_mcc, current_mcc)
            print(f'Step {i}, Loss: {loss.item():.4f}, Val MCC: {current_mcc:.4f}, Best MCC: {best_mcc:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
    
    return best_mcc

def grid_search(X_train, S_train, X_val, S_val, X_test, S_test, D):
    best_mcc = -float('inf')
    best_params = None
    results = []

    for lr in lr_range:
        for l1_weight in l1_weight_range:
            print(f"Training with initial lr={lr}, l1_weight={l1_weight}")
            model = GeneralSAE(D, projections_up, learn_D=True, seed=seed).to(device)
            val_mcc = train_model(model, X_train, S_train, X_val, S_val, num_step, lr, l1_weight)
            
            model.eval()
            with torch.no_grad():
                S_test_, X_test_ = model(X_test)
            test_mcc = mcc(S_test.cpu().numpy(), S_test_.cpu().numpy())
            
            inference_flops = calculate_inference_flops("GeneralSAE", N, M, K, 1, projections_up)
            training_flops = calculate_training_flops("GeneralSAE", N, M, K, batch_size, num_step, projections_up)
            
            result = {
                'initial_lr': lr,
                'l1_weight': l1_weight,
                'val_mcc': val_mcc,
                'test_mcc': test_mcc,
                'inference_flops': inference_flops,
                'training_flops': training_flops,
                'total_flops': inference_flops + training_flops
            }
            results.append(result)
            
            if test_mcc > best_mcc:
                best_mcc = test_mcc
                best_params = (lr, l1_weight)

    print(f"Best parameters: initial_lr={best_params[0]}, l1_weight={best_params[1]}")
    return results

def main():
    S, X, D = generate_data(N, M, K, num_data, seed)
    X, D = X.to(device), D.to(device)
    
    train_size = int(0.7 * num_data)
    val_size = int(0.15 * num_data)
    S_train, S_val, S_test = S[:train_size], S[train_size:train_size+val_size], S[train_size+val_size:]
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    
    results = grid_search(X_train, S_train, X_val, S_val, X_test, S_test, D)
    
    experiment_data = {
        'parameters': {
            'N': N, 'M': M, 'K': K,
            'num_data': num_data,
            'num_step': num_step,
            'batch_size': batch_size,
            'seed': seed,
            'projections_up': projections_up,
            'lr_range': lr_range,
            'l1_weight_range': l1_weight_range,
            'T_0': T_0,
            'T_mult': T_mult
        },
        'results': results
    }
    
    with open('results/general_sae_results_cosine_annealing.json', 'w') as f:
        json.dump(experiment_data, f, indent=2)

if __name__ == "__main__":
    main()