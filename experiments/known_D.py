# """
# known_D.py
# """

# import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F
# import json
# from tqdm import tqdm

# from models import SparseAutoEncoder, MLP, SparseCoding
# from metrics import mcc, greedy_mcc
# from utils import numpy_to_list, generate_data, reconstruction_loss_with_l1
# from calculate_flops import (calculate_sae_training_flops, calculate_sae_inference_flops, calculate_mlp_training_flops, calculate_optimize_codes_flops,
#                             calculate_mlp_inference_flops, calculate_sparse_coding_training_flops, calculate_sparse_coding_inference_flops)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def reconstruction_loss_with_l1(X, X_, S_, l1_weight=0.05):
#     recon_loss = F.mse_loss(X_, X)
#     l1_loss = l1_weight * torch.mean(torch.abs(S_))
#     return recon_loss + l1_loss

# def train(model, X_train, S_train, X_test, S_test, lr=1e-3, num_step=10000, log_step=100, verbose=0):
#     optim = torch.optim.Adam(model.parameters(), lr=lr)
#     log = {'step': [], 'mcc_train': [], 'loss_train': [], 'mcc_test': [], 'loss_test': [], 'flops': []}
    
#     # Calculate initial FLOPs
#     if isinstance(model, SparseAutoEncoder):
#         total_flops = calculate_sae_training_flops(M, N, num_data, 0, learn_D=model.learn_D)
#     elif isinstance(model, MLP):
#         h = model.encoder[0].out_features
#         total_flops = calculate_mlp_training_flops(M, h, N, num_data, 0, learn_D=model.learn_D)
    
#     for i in tqdm(range(num_step), disable=not verbose):
#         S_, X_ = model(X_train)
#         loss = reconstruction_loss_with_l1(X_train, X_, S_)
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
        
#         if i > 0 and not i % log_step:
#             log['step'].append(i)
#             log['loss_train'].append(loss.item())
#             log['mcc_train'].append(mcc(S_train.detach().cpu().numpy(), S_.detach().cpu().numpy()))
#             with torch.no_grad():
#                 S_, X_ = model(X_test)
#                 loss = reconstruction_loss_with_l1(X_test, X_, S_)
#             loss_test = loss.item()
#             mcc_test = greedy_mcc(S_test.detach().cpu().numpy(), S_.detach().cpu().numpy())
#             log['loss_test'].append(loss_test)
#             log['mcc_test'].append(mcc_test)
#             # Print every 1000 steps
#             if i % 1000 == 0:
#                 print(f"Step {i}: Loss Test = {loss_test:.4f}, MCC Test = {mcc_test:.4f}")
            
#             # Calculate and log total FLOPs up to this point
#             if isinstance(model, SparseAutoEncoder):
#                 training_flops = calculate_sae_training_flops(M, N, num_data, i+1, learn_D=model.learn_D)
#                 inference_flops = calculate_sae_inference_flops(M, N, num_data)
#                 total_flops = training_flops + inference_flops
#             elif isinstance(model, MLP):
#                 h = model.encoder[0].out_features
#                 training_flops = calculate_mlp_training_flops(M, h, N, num_data, i+1, learn_D=model.learn_D)
#                 inference_flops = calculate_mlp_inference_flops(M, h, N, num_data)
#                 total_flops = training_flops + inference_flops
#             log['flops'].append(total_flops)

#     print(f"Final MCC: {log['mcc_test'][-1]:.4f}") 
    
#     return log

# def run_sae_ito(model, X_test, S_test, lr=1e-3, num_step=10000, log_step=100, verbose=0):
#     optim = torch.optim.Adam(model.parameters(), lr=lr)
#     log = {'step': [], 'mcc_test': [], 'loss_test': [], 'flops': []}
    
#     for i in tqdm(range(num_step), disable=not verbose):
#         S_, X_ = model(X_test)
#         loss = reconstruction_loss_with_l1(X_test, X_, S_)
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
        
#         if i > 0 and not i % log_step:
#             log['step'].append(i)
#             loss_test = loss.item()
#             mcc_test = mcc(S_test.detach().cpu().numpy(), S_.detach().cpu().numpy())
#             log['loss_test'].append(loss_test)
#             log['mcc_test'].append(mcc_test)
#             print(f"Step {i}: Loss Test = {loss_test:.4f}, MCC Test = {mcc_test:.4f}")
            
#             # Calculate and log total FLOPs up to this point
#             #total_flops = calculate_sparse_coding_training_flops(M, N, X_test.shape[0], i+1, learn_D=model.learn_D)
#             total_flops = calculate_optimize_codes_flops(M, N, X_test.shape[0], num_step)
#             log['flops'].append(total_flops)

#     print(f"Final MCC: {log['mcc_test'][-1]:.4f}") 
    
#     return log

# def run_experiment(model, X_train, S_train, X_test, S_test, num_step=20000, log_step=100, seed=20240625):
#     torch.manual_seed(seed)
#     if isinstance(model, SparseCoding):
#         log = run_sae_ito(model, X_test, S_test, num_step=num_step, log_step=log_step)
#     else:
#         log = train(model, X_train, S_train, X_test, S_test, num_step=num_step, log_step=log_step)
#     return log

# def average_logs(logs):
#     avg_log = {
#         'step': logs[0]['step'],
#         'mcc_test': np.mean([log['mcc_test'] for log in logs], axis=0),
#         'mcc_test_std': np.std([log['mcc_test'] for log in logs], axis=0),
#         'flops': logs[0]['flops']  # FLOPs are deterministic, so we can just take the first run
#     }
#     return avg_log

# # Parameters
# N = 16  # number of sparse sources
# M = 8   # number of measurements
# K = 3   # number of active components
# hidden_layers = [32, 256]  # list of hidden layer widths
# num_runs = 5
# num_data = 1024
# num_step = 20000
# log_step = 100
# seed = 20240625

# # Generate data
# S, X, D = generate_data(N, M, K, num_data * 2, seed=seed)
# D = D.T
# S_train = S[:num_data].to(device)
# X_train = X[:num_data].to(device)
# S_test = S[num_data:].to(device)
# X_test = X[num_data:].to(device)

# # Run experiments
# logs_sae = []
# logs_mlps = {h: [] for h in hidden_layers}
# logs_sae_ito = []

# for i in tqdm(range(num_runs), desc="Running experiments"):
#     run_seed = seed + i

#     SAE = SparseAutoEncoder(M, N, D.to(device), learn_D=False, seed=run_seed)
#     print(f"Running experiment {i+1}/{num_runs} with SAE")
#     logs_sae.append(run_experiment(SAE, X_train, S_train, X_test, S_test, seed=run_seed))

#     SAE_ITO = SparseCoding(X_test, D.to(device), learn_D=False, seed=run_seed)
#     print(f"Running experiment {i+1}/{num_runs} with SAE_ITO")
#     logs_sae_ito.append(run_experiment(SAE_ITO, X_train, S_train, X_test, S_test, seed=run_seed))
    
#     for h in hidden_layers:
#         print(f"Running experiment {i+1}/{num_runs} with MLP (H={h})")
#         MLP_model = MLP(M, N, h, D.to(device), learn_D=False, seed=run_seed).to(device)
#         logs_mlps[h].append(run_experiment(MLP_model, X_train, S_train, X_test, S_test, seed=run_seed))

# # Average logs
# avg_sae = average_logs(logs_sae)
# avg_sae_ito = average_logs(logs_sae_ito)
# avg_mlps = {h: average_logs(logs) for h, logs in logs_mlps.items()}

# # Save results as JSON
# results = {
#     "SAE": avg_sae,
#     "SAE_ITO": avg_sae_ito,
#     "MLPs": avg_mlps
# }

# with open('results/fixed_Z_flops_reconstruction.json', 'w') as f:
#     json.dump(numpy_to_list(results), f)

# print("Experiment completed. Results saved to 'results/fixed_Z_flops_reconstruction.json'.")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm

from models import SparseAutoEncoder, MLP
from metrics import mcc, greedy_mcc
from utils import numpy_to_list, generate_data, reconstruction_loss_with_l1
from calculate_flops import (calculate_sae_training_flops, calculate_sae_inference_flops, calculate_mlp_training_flops, calculate_optimize_codes_flops,
                            calculate_mlp_inference_flops, calculate_sparse_coding_training_flops, calculate_sparse_coding_inference_flops)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SparseCoding(nn.Module):
    def __init__(self, X, D, learn_D, seed=20240625, relu_activation=False, use_bias=False, initial_S=None):
        super().__init__()
        self.learn_D = learn_D
        self.relu_activation = relu_activation
        torch.manual_seed(seed + 42)
        if learn_D:
            self.D = nn.Parameter(data=torch.randn(D.shape), requires_grad=True)
        else:
            self.D = nn.Parameter(data=D, requires_grad=False)
        
        if initial_S is not None:
            if not self.relu_activation:
                self.log_S = nn.Parameter(data=torch.log(initial_S + 1e-10), requires_grad=True)
            else:
                self.log_S = nn.Parameter(data=initial_S, requires_grad=True)
        else:
            if not self.relu_activation:
                self.log_S = nn.Parameter(data=-10 * torch.ones(X.shape[0], D.shape[1]), requires_grad=True)
            else:
                self.log_S = nn.Parameter(data=torch.randn(X.shape[0], D.shape[1]), requires_grad=True)
        
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(D.shape[0]), requires_grad=True)

    def forward(self, X, norm_D = True):
        if self.learn_D and norm_D:
            self.D.data /= torch.linalg.norm(self.D.data, dim=0, keepdim=True)
        if self.relu_activation:
            S = F.relu(self.log_S)
        else:
            S = torch.exp(self.log_S)
        X_ = S @ self.D.T
        if self.use_bias:
            X_ += self.bias
        return S, X_

def train(model, X_train, S_train, X_test, S_test, lr=1e-3, num_step=10000, log_step=100, verbose=0):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log = {'step': [], 'mcc_train': [], 'loss_train': [], 'mcc_test': [], 'loss_test': [], 'flops': []}
    
    # Calculate initial FLOPs
    if isinstance(model, SparseAutoEncoder):
        total_flops = calculate_sae_training_flops(M, N, num_data, 0, learn_D=model.learn_D)
    elif isinstance(model, MLP):
        h = model.encoder[0].out_features
        total_flops = calculate_mlp_training_flops(M, h, N, num_data, 0, learn_D=model.learn_D)
    
    for i in tqdm(range(num_step), disable=not verbose):
        S_, X_ = model(X_train)
        loss = reconstruction_loss_with_l1(X_train, X_, S_)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if i > 0 and not i % log_step:
            log['step'].append(i)
            log['loss_train'].append(loss.item())
            log['mcc_train'].append(mcc(S_train.detach().cpu().numpy(), S_.detach().cpu().numpy()))
            with torch.no_grad():
                S_, X_ = model(X_test)
                loss = reconstruction_loss_with_l1(X_test, X_, S_)
            loss_test = loss.item()
            mcc_test = mcc(S_test.detach().cpu().numpy(), S_.detach().cpu().numpy())
            log['loss_test'].append(loss_test)
            log['mcc_test'].append(mcc_test)
            # Print every 1000 steps
            if i % 1000 == 0:
                print(f"Step {i}: Loss Test = {loss_test:.4f}, MCC Test = {mcc_test:.4f}")
            
            # Calculate and log total FLOPs up to this point
            if isinstance(model, SparseAutoEncoder):
                training_flops = calculate_sae_training_flops(M, N, num_data, i+1, learn_D=model.learn_D)
                inference_flops = calculate_sae_inference_flops(M, N, num_data)
                total_flops = training_flops + inference_flops
            elif isinstance(model, MLP):
                h = model.encoder[0].out_features
                training_flops = calculate_mlp_training_flops(M, h, N, num_data, i+1, learn_D=model.learn_D)
                inference_flops = calculate_mlp_inference_flops(M, h, N, num_data)
                total_flops = training_flops + inference_flops
            log['flops'].append(total_flops)

    print(f"Final MCC: {log['mcc_test'][-1]:.4f}") 
    
    return log

# def run_sae_ito(model, X_test, S_test, lr=1e-3, num_step=10000, log_step=100, verbose=0):
#     optim = torch.optim.Adam(model.parameters(), lr=lr)
#     log = {'step': [], 'mcc_test': [], 'loss_test': [], 'flops': []}
    
#     for i in tqdm(range(num_step), disable=not verbose):
#         S_, X_ = model(X_test)
#         loss = reconstruction_loss_with_l1(X_test, X_, S_)
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
        
#         if i > 0 and not i % log_step:
#             log['step'].append(i)
#             loss_test = loss.item()
#             mcc_test = mcc(S_test.detach().cpu().numpy(), S_.detach().cpu().numpy())
#             log['loss_test'].append(loss_test)
#             log['mcc_test'].append(mcc_test)
#             print(f"Step {i}: Loss Test = {loss_test:.4f}, MCC Test = {mcc_test:.4f}")
            
#             # Calculate and log total FLOPs up to this point
#             total_flops = calculate_optimize_codes_flops(M, N, X_test.shape[0], i+1)
#             log['flops'].append(total_flops)

#     print(f"Final MCC: {log['mcc_test'][-1]:.4f}") 
    
#     return log

def run_sae_ito(model, X_test, S_test, lr=1e-3, num_step=10000, log_step=100, verbose=0):
    log_S_ = nn.Parameter(data=-10 * torch.ones(X_test.shape[0], model.D.shape[1]), requires_grad=True)
    opt = torch.optim.Adam([log_S_], lr=lr)
    log = {'step': [], 'mcc_test': [], 'loss_test': [], 'flops': []}
    
    for i in tqdm(range(num_step), disable=not verbose):
        S = torch.exp(log_S_).to(device) if not model.relu_activation else F.relu(log_S_).to(device)
        X_ = S @ model.D.T
        if model.use_bias:
            X_ += model.bias
        loss = reconstruction_loss_with_l1(X_test, X_, S)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if i > 0 and not i % log_step:
            log['step'].append(i)
            loss_test = loss.item()
            mcc_test = mcc(S_test.detach().cpu().numpy(), S.detach().cpu().numpy())
            log['loss_test'].append(loss_test)
            log['mcc_test'].append(mcc_test)
            if i % 1000 == 0:
                print(f"Step {i}: Loss Test = {loss_test:.4f}, MCC Test = {mcc_test:.4f}")
            
            # Calculate and log total FLOPs up to this point
            total_flops = calculate_optimize_codes_flops(M, N, X_test.shape[0], i+1)
            log['flops'].append(total_flops)

    print(f"Final MCC: {log['mcc_test'][-1]:.4f}") 
    
    return log

def run_experiment(model, X_train, S_train, X_test, S_test, num_step=20000, log_step=100, seed=20240625):
    torch.manual_seed(seed)
    if isinstance(model, SparseAutoEncoder):
        log = train(model, X_train, S_train, X_test, S_test, num_step=num_step, log_step=log_step)
        # Get the predicted latents for SAE_ITO initialization
        with torch.no_grad():
            initial_S, _ = model(X_test)
        return log, initial_S
    elif isinstance(model, SparseCoding):
        log = run_sae_ito(model, X_test, S_test, num_step=num_step, log_step=log_step)
        return log, None
    else:
        log = train(model, X_train, S_train, X_test, S_test, num_step=num_step, log_step=log_step)
        return log, None

def average_logs(logs):
    avg_log = {
        'step': logs[0]['step'],
        'mcc_test': np.mean([log['mcc_test'] for log in logs], axis=0),
        'mcc_test_std': np.std([log['mcc_test'] for log in logs], axis=0),
        'flops': logs[0]['flops']  # FLOPs are deterministic, so we can just take the first run
    }
    return avg_log

# Parameters
N = 16  # number of sparse sources
M = 8   # number of measurements
K = 3   # number of active components
hidden_layers = [32, 256]  # list of hidden layer widths
num_runs = 5
num_data = 1024
num_step = 20000
log_step = 100
seed = 20240625

# Generate data
S, X, D = generate_data(N, M, K, num_data * 2, seed=seed)
D = D.T
S_train = S[:num_data].to(device)
X_train = X[:num_data].to(device)
S_test = S[num_data:].to(device)
X_test = X[num_data:].to(device)

# Run experiments
logs_sae = []
logs_mlps = {h: [] for h in hidden_layers}
logs_sae_ito_init = []
logs_sae_ito_random = []

for i in tqdm(range(num_runs), desc="Running experiments"):
    run_seed = seed + i

    SAE = SparseAutoEncoder(M, N, D.to(device), learn_D=False, seed=run_seed).to(device)
    print(f"Running experiment {i+1}/{num_runs} with SAE")
    log_sae, initial_S = run_experiment(SAE, X_train, S_train, X_test, S_test, num_step=num_step, seed=run_seed)
    logs_sae.append(log_sae)

    SAE_ITO_init = SparseCoding(X_test, D.to(device), learn_D=False, seed=run_seed, initial_S=initial_S).to(device)
    print(f"Running experiment {i+1}/{num_runs} with SAE_ITO (initialized)")
    log_sae_ito_init, _ = run_experiment(SAE_ITO_init, X_train, S_train, X_test, S_test, num_step=num_step, seed=run_seed)
    logs_sae_ito_init.append(log_sae_ito_init)

    SAE_ITO_random = SparseCoding(X_test, D.to(device), learn_D=False, seed=run_seed).to(device)
    print(f"Running experiment {i+1}/{num_runs} with SAE_ITO (random)")
    log_sae_ito_random, _ = run_experiment(SAE_ITO_random, X_train, S_train, X_test, S_test, num_step=num_step, seed=run_seed)
    logs_sae_ito_random.append(log_sae_ito_random)
    
    for h in hidden_layers:
        print(f"Running experiment {i+1}/{num_runs} with MLP (H={h})")
        MLP_model = MLP(M, N, h, D.to(device), learn_D=False, seed=run_seed).to(device)
        log_mlp, _ = run_experiment(MLP_model, X_train, S_train, X_test, S_test, seed=run_seed)
        logs_mlps[h].append(log_mlp)

# Average logs
avg_sae = average_logs(logs_sae)
avg_sae_ito_init = average_logs(logs_sae_ito_init)
avg_sae_ito_random = average_logs(logs_sae_ito_random)
avg_mlps = {h: average_logs(logs) for h, logs in logs_mlps.items()}

# Save results as JSON
results = {
    "SAE": avg_sae,
    "SAE_ITO_init": avg_sae_ito_init,
    "SAE_ITO_random": avg_sae_ito_random,
    "MLPs": avg_mlps
}

with open('results/fixed_D_flops_reconstruction.json', 'w') as f:
    json.dump(numpy_to_list(results), f)

print("Experiment completed. Results saved to 'results/fixed_D_flops_reconstruction.json'.")