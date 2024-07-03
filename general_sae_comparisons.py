import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import pearsonr as corr
from tqdm import tqdm
import yaml
from munkres import Munkres
import json
from models import SparseCoding, SparseAutoEncoder, GeneralSAE
from flop_counter import calculate_inference_flops, calculate_training_flops

# Parameters
N = 16  # number of sparse sources (true dimension)
K = 3  # number of active components
M = 8  # number of measurements
seed = 20240625
num_data = 1024
num_step = 10000
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# GeneralSAE encoder depths to test
encoder_depths = [2, 3, 4, 5]

# Define grid search parameters
lr_range = [1e-4, 5e-3, 1e-3, 5e-2, 1e-2]
l1_weight_range = [1e-5, 5e-4, 1e-4, 5e-3, 1e-3]

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

def pad_to_size(array, target_size):
    current_size = array.shape[1]
    if (current_size < target_size):
        padding = np.zeros((array.shape[0], target_size - current_size))
        return np.concatenate([array, padding], axis=1)
    return array

def mcc(z, z_):
    target_size = max(z.shape[1], z_.shape[1])
    if not z.shape[1] == z_.shape[1] == target_size:
        z = pad_to_size(z, target_size)
        z_ = pad_to_size(z_, target_size)
    matches = np.zeros((z.shape[1], z_.shape[1]))
    for i in range(z.shape[1]):
        for j in range(z_.shape[1]):
            matches[i, j] = abs(corr(z[:, i], z_[:, j])[0])
    matches[np.isnan(matches)] = 0.0

    munk = Munkres()
    indexes = munk.compute(-matches)

    corrs = []
    for i in indexes:
        corrs.append(matches[i[0], i[1]])

    return np.mean(corrs)

def train_model(model, X_train, S_train, num_step, lr, l1_weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mcc = -float('inf')
    
    for i in tqdm(range(num_step), desc=f"Training {model.__class__.__name__}"):
        S_, X_ = model.forward(X_train)
        loss = F.mse_loss(X_, X_train) + l1_weight * torch.sum(torch.abs(S_))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i > 0 and not i % (num_step // 10)) or i == 0:
            if not torch.all(S_.var(0) > 0):
                print('dead latents')
            current_mcc = mcc(S_train.cpu().numpy(), S_.detach().cpu().numpy())
            best_mcc = max(best_mcc, current_mcc)
            print('step', i, 'loss', loss.item(), 'MCC', current_mcc, 'Best MCC', best_mcc)
    
    return best_mcc

def train_sparse_coding(model, X_train, S_train, num_step, lr, l1_weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mcc = -float('inf')
    
    for i in tqdm(range(num_step), desc="Training SparseCoding"):
        # S_ = model.infer(X_train, num_iterations=10000, lr=lr, l1_weight=l1_weight)
        # X_ = S_ @ model.D_
        S_, X_ = model.forward(X_train)
        #loss = F.mse_loss(X_, X_train) + l1_weight * torch.sum(torch.abs(S_))
        loss = torch.sum((X_train - X_) ** 2) + l1_weight * torch.sum(torch.abs(S_))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if model.learn_D:
            model.D_.data /= torch.linalg.norm(model.D_, dim=1, keepdim=True)

        if (i > 0 and not i % (num_step // 10)) or i == 0:
            if not torch.all(S_.var(0) > 0):
                print('dead latents')
            current_mcc = mcc(S_train.cpu().numpy(), S_.detach().cpu().numpy())
            best_mcc = max(best_mcc, current_mcc)
            print('step', i, 'loss', loss.item(), 'MCC', current_mcc, 'Best MCC', best_mcc)
    
    return best_mcc

def run_experiment(model_class, encoder_depth, X_train, S_train, D, X_test, S_test, num_step, lr, l1_weight):
    if model_class == SparseAutoEncoder:
        model = model_class(D, learn_D=True, seed=seed).to(device)
        best_mcc = train_model(model, X_train, S_train, num_step, lr, l1_weight)
        S_test_, X_test_ = model.forward(X_test)
    elif model_class == GeneralSAE:
        projections_up = generate_projections(M, N, encoder_depth)
        model = model_class(D, projections_up, learn_D=True, seed=seed).to(device)
        best_mcc = train_model(model, X_train, S_train, num_step, lr, l1_weight)
        S_test_, X_test_ = model.forward(X_test)
    elif model_class == SparseCoding:
        S_init = torch.zeros(S_train.shape[0], N).to(device)
        model = model_class(S_init, D, learn_D=True, seed=seed).to(device)
        best_mcc = train_sparse_coding(model, X_train, S_train, num_step, lr, l1_weight)
        
        # Inference-time optimization for SparseCoding
        num_iterations = 10000
        S_test_ = model.infer(X_test, num_iterations=num_iterations, lr=lr, l1_weight=l1_weight)
    else:
        raise ValueError("Unsupported model class")
    
    test_mcc = mcc(S_test.cpu().numpy(), S_test_.detach().cpu().numpy())
    
    if model_class == GeneralSAE:
        projections_up = generate_projections(M, N, encoder_depth)
        inference_flops = calculate_inference_flops(model_class.__name__, N, M, K, 1, projections_up)
        training_flops = calculate_training_flops(model_class.__name__, N, M, K, batch_size, num_step, projections_up)
    else:
        inference_flops = calculate_inference_flops(model_class.__name__, N, M, K, num_iterations if model_class == SparseCoding else 1)
        training_flops = calculate_training_flops(model_class.__name__, N, M, K, batch_size, num_step)
    total_flops = inference_flops + training_flops
    
    result = {
        'model': model_class.__name__,
        'encoder_depth': encoder_depth if model_class == GeneralSAE else None,
        'lr': lr,
        'l1_weight': l1_weight,
        'inference_flops': inference_flops,
        'training_flops': training_flops,
        'total_flops': total_flops,
        'mcc': test_mcc
    }

    return result

def generate_projections(M, N, num_layers):
    if num_layers == 1:
        return [N]
    
    step = (N - M) / (num_layers - 1)
    projections = [round(M + i * step) for i in range(num_layers)]
    projections[-1] = N  # Ensure the last projection is exactly N
    return projections

def grid_search(model_class, encoder_depth, X_train, S_train, D, X_test, S_test, num_step, lr_range, l1_weight_range):
    best_mcc = -float('inf')
    best_params = None

    for lr in lr_range:
        for l1_weight in l1_weight_range:
            print(f"Grid search for {model_class.__name__} with encoder_depth={encoder_depth}, lr={lr}, l1_weight={l1_weight}")
            current_result = run_experiment(model_class, encoder_depth, X_train, S_train, D, X_test, S_test, num_step, lr, l1_weight)
            if current_result['mcc'] > best_mcc:
                best_mcc = current_result['mcc']
                best_params = (lr, l1_weight)

    return best_params

def main():
    S, X, D = generate_data(N, M, K, num_data, seed)
    X, D = X.to(device), D.to(device)
    
    train_size = int(0.8 * num_data)
    S_train, S_test = S[:train_size], S[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]
    
    results = []
    
    # Experiment with SparseCoding
    print("Running grid search for SparseCoding")
    best_lr, best_l1_weight = grid_search(SparseCoding, None, X_train, S_train, D, X_test, S_test, num_step, lr_range, l1_weight_range)
    print(f"Best parameters for SparseCoding: lr={best_lr}, l1_weight={best_l1_weight}")
    result = run_experiment(SparseCoding, None, X_train, S_train, D, X_test, S_test, num_step, best_lr, best_l1_weight)
    results.append(result)
    
    # Experiment with vanilla SparseAutoEncoder
    print("Running grid search for SparseAutoEncoder")
    best_lr, best_l1_weight = grid_search(SparseAutoEncoder, 1, X_train, S_train, D, X_test, S_test, num_step, lr_range, l1_weight_range)
    print(f"Best parameters for SparseAutoEncoder: lr={best_lr}, l1_weight={best_l1_weight}")
    result = run_experiment(SparseAutoEncoder, 1, X_train, S_train, D, X_test, S_test, num_step, best_lr, best_l1_weight)
    results.append(result)
    
    # Experiments with GeneralSAE for different encoder depths
    for depth in encoder_depths:
        print(f"Running grid search for GeneralSAE with encoder_depth={depth}")
        best_lr, best_l1_weight = grid_search(GeneralSAE, depth, X_train, S_train, D, X_test, S_test, num_step, lr_range, l1_weight_range)
        print(f"Best parameters for GeneralSAE with encoder_depth={depth}: lr={best_lr}, l1_weight={best_l1_weight}")
        result = run_experiment(GeneralSAE, depth, X_train, S_train, D, X_test, S_test, num_step, best_lr, best_l1_weight)
        results.append(result)
    
    # Save results
    experiment_data = {
        'parameters': {
            'N': N,
            'M': M,
            'K': K,
            'num_data': num_data,
            'num_step': num_step,
            'batch_size': batch_size,
            'seed': seed,
            'encoder_depths': encoder_depths,
            'lr_range': lr_range,
            'l1_weight_range': l1_weight_range
        },
        'results': results
    }
    
    with open('results/sae_comparison_results.json', 'w') as f:
        json.dump(experiment_data, f, indent=2)

if __name__ == "__main__":
    main()