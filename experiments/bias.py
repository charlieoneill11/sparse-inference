import numpy as np
import torch
from torch import nn
import json
from tqdm import tqdm

from models import SparseAutoEncoder, MLP, SparseCoding
from metrics import mcc, greedy_mcc
from utils import numpy_to_list, generate_data, reconstruction_loss_with_l1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_dict_mcc(D_true, D_learned):
    return greedy_mcc(D_true.cpu().numpy(), D_learned.cpu().numpy())

def calculate_l0(S):
    return (S.abs() > 1e-4).float().mean().item()

def train_model(model, X_train, S_train, X_test, S_test, D_true, lr, num_epochs=100_000):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    for _ in tqdm(range(num_epochs)):#, disable=True):
        S_, X_ = model(X_train)
        loss = reconstruction_loss_with_l1(X_train, X_, S_)
        optim.zero_grad()
        loss.backward()
        optim.step()
    
    if isinstance(model, MLP) or isinstance(model, SparseAutoEncoder):
        S_test_, X_test_ = model(X_test)
    else:
        S_test_ = model.optimize_codes(X_test, num_iterations=10_000)
        X_test_ = S_test_ @ model.D.T
    S_test_ = S_test_.detach()
    X_test_ = X_test_.detach()
    loss_test = reconstruction_loss_with_l1(X_test, X_test_, S_test_).item()
    mcc_test = mcc(S_test.cpu().numpy(), S_test_.cpu().numpy())
    l0_test = calculate_l0(S_test_)
    
    if isinstance(model, SparseAutoEncoder) or isinstance(model, MLP):
        D_learned = model.decoder.weight.data
    elif isinstance(model, SparseCoding):
        D_learned = model.D.data
    dict_mcc = calculate_dict_mcc(D_true, D_learned)

    print(f"Loss: {loss_test:.4f}, MCC: {mcc_test:.4f}, Dict MCC: {dict_mcc:.4f}, L0: {l0_test:.4f}")
    
    return {
        'loss_test': loss_test,
        'mcc_test': mcc_test,
        'dict_mcc': dict_mcc,
        'l0_test': l0_test
    }

def run_experiment(model_class, X_train, S_train, X_test, S_test, D_true, use_bias, **model_kwargs):
    if model_class == MLP:
        lr = 1e-4
    else:
        lr = 1e-3
    
    model = model_class(**model_kwargs, use_bias=use_bias).to(device)
    return train_model(model, X_train, S_train, X_test, S_test, D_true, lr)

# Parameters
N = 16  # number of sparse sources
M = 8   # number of measurements
K = 3   # number of active components
hidden_layers = [64]  # list of hidden layer widths
num_runs = 3
num_data = 1024
seed = 20240926

# Generate data
S, X, D = generate_data(N, M, K, num_data * 2, seed=seed)
D = D.T
S_train = S[:num_data].to(device)
X_train = X[:num_data].to(device)
S_test = S[num_data:].to(device)
X_test = X[num_data:].to(device)
D_true = D.to(device)

# Run experiments
results = {
    'SAE': {'with_bias': [], 'without_bias': []},
    'SparseCoding': {'with_bias': [], 'without_bias': []},
    'MLP': {h: {'with_bias': [], 'without_bias': []} for h in hidden_layers}
}

for i in range(num_runs):
    run_seed = seed + i
    torch.manual_seed(run_seed)
    print(f"Doing run {i} out of {num_runs}")
    
    for use_bias in [True, False]:
        bias_key = 'with_bias' if use_bias else 'without_bias'
        print(f"Using bias: {bias_key}")
        
        # SAE
        print("Running SAE")
        sae_result = run_experiment(SparseAutoEncoder, X_train, S_train, X_test, S_test, D_true, 
                                    use_bias, M=M, N=N, D=D.to(device), learn_D=True, seed=run_seed)
        results['SAE'][bias_key].append(sae_result)
        
        # Sparse Coding
        print("Running Sparse Coding")
        sc_result = run_experiment(SparseCoding, X_train, S_train, X_test, S_test, D_true, 
                                   use_bias, X=X_test, D=D.to(device), learn_D=True, seed=run_seed)
        results['SparseCoding'][bias_key].append(sc_result)
        
        # MLPs
        for h in hidden_layers:
            print(f"Running MLP with hidden layer width {h}")
            mlp_result = run_experiment(MLP, X_train, S_train, X_test, S_test, D_true, 
                                        use_bias, M=M, N=N, h=h, D=D.to(device), learn_D=True, seed=run_seed)
            results['MLP'][h][bias_key].append(mlp_result)

# Calculate averages and standard deviations
for model_type in results:
    if model_type == 'MLP':
        for h in results[model_type]:
            for bias_type in results[model_type][h]:
                avg_results = {metric: np.mean([run[metric] for run in results[model_type][h][bias_type]]) 
                               for metric in results[model_type][h][bias_type][0]}
                std_results = {f"{metric}_std": np.std([run[metric] for run in results[model_type][h][bias_type]]) 
                               for metric in results[model_type][h][bias_type][0]}
                results[model_type][h][bias_type] = {**avg_results, **std_results}
    else:
        for bias_type in results[model_type]:
            avg_results = {metric: np.mean([run[metric] for run in results[model_type][bias_type]]) 
                           for metric in results[model_type][bias_type][0]}
            std_results = {f"{metric}_std": np.std([run[metric] for run in results[model_type][bias_type]]) 
                           for metric in results[model_type][bias_type][0]}
            results[model_type][bias_type] = {**avg_results, **std_results}

# Save results as JSON
with open('results/bias_comparison_results.json', 'w') as f:
    json.dump(numpy_to_list(results), f)

print("Experiment completed. Results saved to 'results/bias_comparison_results.json'.")