import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import pearsonr as corr
from tqdm import tqdm
import yaml
from munkres import Munkres
import json
from models import SparseCoding, TopKSAE, MLP, SparseAutoEncoder
from flop_counter import calculate_inference_flops, calculate_training_flops
from metrics import corr

# Parameters
N = 16  # number of sparse sources (true dimension)
K = 3  # number of active components
M = 8  # number of measurements
seed = 20240625
num_data = 1024 * 2
num_step = 10000
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 100

# Define grid search parameters
lr_range = [1e-3] #[1e-4, 5e-3, 1e-3]

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

def pad_to_size(array, target_size):
    current_size = array.shape[1]
    if current_size < target_size:
        padding = np.zeros((array.shape[0], target_size - current_size))
        return np.concatenate([array, padding], axis=1)
    return array

def mcc(z, z_):
    return np.mean([corr(a, b)[0] for a, b in zip(z.T, z_.T)])

def train_model(model, X_train, S_train, X_test, S_test, num_step, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mcc = -float('inf')
    performance_log = []
    best_train_mccs = []
    best_test_mccs = []
    
    for i in tqdm(range(num_step), desc=f"Training {model.__class__.__name__}"):
        S_, _ = model(X_train)
        loss = 1 - F.cosine_similarity(S_, S_train).mean() 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % eval_interval == 0 or i == 0:
            with torch.no_grad():
                S_train_eval, _ = model(X_train)
                S_test_eval, _ = model(X_test)
                
                train_mcc = mcc(S_train.cpu().numpy(), S_train_eval.cpu().numpy())
                test_mcc = mcc(S_test.cpu().numpy(), S_test_eval.cpu().numpy())
                
                # Calculate FLOPs
                if isinstance(model, MLP):
                    inference_flops = calculate_inference_flops(model.__class__.__name__, N, M, K, 1, model.projections_up)
                    training_flops = calculate_training_flops(model.__class__.__name__, N, M, K, batch_size, i+1, model.projections_up)
                else:
                    inference_flops = calculate_inference_flops(model.__class__.__name__, N, M, K, 1)
                    training_flops = calculate_training_flops(model.__class__.__name__, N, M, K, batch_size, i+1)
                total_flops = inference_flops + training_flops
                
                performance_log.append({
                    'step': i + 1,
                    'loss': loss.item(),
                    'train_mcc': train_mcc,
                    'test_mcc': test_mcc,
                    'inference_flops': inference_flops,
                    'training_flops': training_flops,
                    'total_flops': total_flops
                })
                print(f'Step {i+1}, Loss: {loss.item():.4f}, Train MCC: {train_mcc:.4f}, Test MCC: {test_mcc:.4f}, Total FLOPs: {total_flops}')
                
                if test_mcc > best_mcc:
                    best_mcc = test_mcc
                    best_train_mccs = [log['train_mcc'] for log in performance_log]
                    best_test_mccs = [log['test_mcc'] for log in performance_log]

    return best_mcc, performance_log, best_train_mccs, best_test_mccs

def run_experiment(model_class, X_train, S_train, D, X_test, S_test, num_step, lr, seed):
    if model_class == MLP:
        projections_up = [M * 16, N]
        model = model_class(D, projections_up, learn_D=True, seed=seed).to(device)
    elif model_class == SparseAutoEncoder:
        model = model_class(D, learn_D=True, seed=seed).to(device)
    else:  # TopKSAE
        model = model_class(D, learn_D=True, k=K, seed=seed).to(device)
    
    best_mcc, performance_log, best_train_mccs, best_test_mccs = train_model(model, X_train, S_train, X_test, S_test, num_step, lr)

    with torch.no_grad():
        S_test_, _ = model(X_test)
    
    test_mcc = mcc(S_test.cpu().numpy(), S_test_.detach().cpu().numpy())
    
    # Final FLOPs calculation
    if isinstance(model, MLP):
        inference_flops = calculate_inference_flops(model_class.__name__, N, M, K, 1, projections_up)
        training_flops = calculate_training_flops(model_class.__name__, N, M, K, batch_size, num_step, projections_up)
    else:
        inference_flops = calculate_inference_flops(model_class.__name__, N, M, K, 1)
        training_flops = calculate_training_flops(model_class.__name__, N, M, K, batch_size, num_step)
    total_flops = inference_flops + training_flops
    
    result = {
        'model': model_class.__name__,
        'lr': lr,
        'seed': seed,
        'inference_flops': inference_flops,
        'training_flops': training_flops,
        'total_flops': total_flops,
        'mcc': best_mcc,
        'final_test_mcc': test_mcc,
        'performance_log': performance_log,
        'best_train_mccs': best_train_mccs,
        'best_test_mccs': best_test_mccs
    }

    return result

def grid_search(model_class, X_train, S_train, D, X_test, S_test, num_step, lr_range):
    best_mcc = -float('inf')
    best_params = None
    best_result = None

    for lr in lr_range:
        print(f"Grid search for {model_class.__name__} with lr={lr}")
        current_result = run_experiment(model_class, X_train, S_train, D, X_test, S_test, num_step, lr, seed)
        if current_result['mcc'] > best_mcc:
            best_mcc = current_result['mcc']
            best_params = lr
            best_result = current_result

    return best_params, best_result

def main():
    S, X, D = generate_data(N, M, K, num_data, seed)
    X, D = X.to(device), D.to(device)
    
    train_size = int(0.5 * num_data)
    S_train, S_test = S[:train_size], S[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]
    
    with open('train_configs.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    results = []
    
    for model_class in [MLP, SparseAutoEncoder]:
        print(f"Running grid search for {model_class.__name__}")
        #best_lr, best_result = grid_search(model_class, X_train, S_train, D, X_test, S_test, num_step, lr_range)
        best_lr = 1e-3
        print(f"Best parameters for {model_class.__name__}: lr={best_lr}")
        
        model_results = []
        for run in range(5):
            run_seed = seed + run  # Different seed for each run
            num_step_final = int(num_step * 10) if model_class == SparseAutoEncoder else int(num_step / 4)
            print(f"Running {model_class.__name__} with best parameters, run {run + 1}, num steps = {num_step_final}")
            result = run_experiment(model_class, X_train, S_train, D, X_test, S_test, num_step_final, best_lr, run_seed)
            model_results.append(result)
        results.append({
            'model': model_class.__name__,
            'best_lr': best_lr,
            'runs': model_results
        })
    
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
            'lr_range': lr_range,
            'eval_interval': eval_interval
        },
        'results': results
    }
    
    with open('results/latent_regression_results.json', 'w') as f:
        json.dump(experiment_data, f, indent=2)

if __name__ == "__main__":
    main()
