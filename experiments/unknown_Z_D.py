"""
unknown_Z_D.py
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm

from models import SparseAutoEncoder, MLP, SparseCoding
from metrics import mcc, greedy_mcc
from utils import numpy_to_list, generate_data, reconstruction_loss_with_l1
from calculate_flops import (calculate_sae_training_flops, calculate_sae_inference_flops, calculate_mlp_training_flops, calculate_optimize_codes_flops,
                            calculate_mlp_inference_flops, calculate_sparse_coding_training_flops, calculate_sparse_coding_inference_flops)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_dict_mcc(D_true, D_learned):
    #print(f"D_true shape = {D_true.shape}, D_learned shape = {D_learned.shape}")
    return greedy_mcc(D_true.cpu().numpy(), D_learned.cpu().numpy())
    #return mcc(D_true.cpu().numpy(), D_learned.cpu().numpy())

def train_sparse_coding(model, X_train, S_train, X_test, S_test, D_true, lr=1e-3, num_step=30000, log_step=100, verbose=0):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log = {'step': [], 'mcc_train': [], 'loss_train': [], 'mcc_test': [], 'loss_test': [], 'train_flops': [], 'test_flops': [], 'dict_mcc': []}
    
    for i in tqdm(range(num_step), disable=not verbose):
        S_, X_ = model(X_train)
        loss = reconstruction_loss_with_l1(X_train, X_, S_)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Assert that the 
        
        if i > 0 and not i % log_step or i == 0:
            log['step'].append(i)
            log['loss_train'].append(loss.item())
            log['mcc_train'].append(mcc(S_train.detach().cpu().numpy(), S_.detach().cpu().numpy()))
            
            # Optimize codes for test set
            #with torch.no_grad():
            S_test_opt = model.optimize_codes(X_test, num_iterations=10_000)
            X_test_ = S_test_opt @ model.D.T
            loss_test = reconstruction_loss_with_l1(X_test, X_test_, S_test_opt)
            
            mcc_test = mcc(S_test.cpu().numpy(), S_test_opt.cpu().numpy())
            log['loss_test'].append(loss_test.item())
            log['mcc_test'].append(mcc_test)
            
            # Calculate dictionary MCC
            dict_mcc = calculate_dict_mcc(D_true, model.D.data)
            log['dict_mcc'].append(dict_mcc)
            
            # Calculate and log FLOPs
            train_flops = calculate_sparse_coding_training_flops(M, N, X_train.shape[0], i+1, learn_D=model.learn_D)
            #test_flops = calculate_sparse_coding_inference_flops(M, N, X_test.shape[0], learn_D=False)  # We're not learning D during testing
            test_flops = calculate_optimize_codes_flops(M, N, X_test.shape[0], 10_000)
            log['train_flops'].append(train_flops)
            log['test_flops'].append(test_flops)

            if i % 1000 == 0 or i == 0:
                print(f"Step {i}: Loss Test = {loss_test:.4f}, MCC Test = {mcc_test:.4f}, Dict MCC = {dict_mcc:.4f}")

    print(f"Final MCC: {log['mcc_test'][-1]:.4f}, Final Dict MCC: {log['dict_mcc'][-1]:.4f}, Final Test Loss: {log['loss_test'][-1]:.4f}") 
    return log, model.D.data

def train(model, X_train, S_train, X_test, S_test, D_true, lr=1e-3, num_step=30000, log_step=100, verbose=0):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log = {'step': [], 'mcc_train': [], 'loss_train': [], 'mcc_test': [], 'loss_test': [], 'train_flops': [], 'test_flops': [], 'dict_mcc': []}
    
    for i in tqdm(range(num_step), disable=not verbose):
        S_, X_ = model(X_train)
        loss = reconstruction_loss_with_l1(X_train, X_, S_)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if i > 0 and not i % log_step or i == 0:
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
            
            # Calculate dictionary MCC
            if isinstance(model, SparseAutoEncoder) or isinstance(model, MLP):
                D_learned = model.decoder.weight.data
            elif isinstance(model, SparseCoding):
                D_learned = model.D_.data
            dict_mcc = calculate_dict_mcc(D_true, D_learned)
            log['dict_mcc'].append(dict_mcc)
            
            # Calculate and log FLOPs
            if isinstance(model, SparseAutoEncoder):
                train_flops = calculate_sae_training_flops(M, N, num_data, i+1, learn_D=model.learn_D)
                test_flops = calculate_sae_inference_flops(M, N, num_data)
            elif isinstance(model, MLP):
                h = model.encoder[0].out_features
                train_flops = calculate_mlp_training_flops(M, h, N, num_data, i+1, learn_D=model.learn_D)
                test_flops = calculate_mlp_inference_flops(M, h, N, num_data)
            elif isinstance(model, SparseCoding):
                train_flops = calculate_sparse_coding_training_flops(M, N, num_data, i+1, learn_D=model.learn_D)
                #test_flops = calculate_sparse_coding_inference_flops(M, N, num_data, learn_D=model.learn_D)
                test_flops = calculate_optimize_codes_flops(M, N, X_test.shape[0], 10_000)
            log['train_flops'].append(train_flops)
            log['test_flops'].append(test_flops)

            if i % 1000 == 0 or i == 0:
                print(f"Step {i}: Loss Test = {loss_test:.4f}, MCC Test = {mcc_test:.4f}, Dict MCC = {dict_mcc:.4f} (Train: {train_flops/1e9:.2f}B, Test: {test_flops/1e9:.2f}B)")

    print(f"Final MCC: {log['mcc_test'][-1]:.4f}, Final Dict MCC: {log['dict_mcc'][-1]:.4f}, Final Test Loss: {log['loss_test'][-1]:.4f}") 
    return log, model.decoder.weight.data.T if hasattr(model, 'decoder') else model.D_.data

def train_sae_with_ito(model, X_train, S_train, X_test, S_test, D_true, lr=1e-3, num_step=30000, log_step=100, verbose=0):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log_sae = {'step': [], 'mcc_train': [], 'loss_train': [], 'mcc_test': [], 'loss_test': [], 'train_flops': [], 'test_flops': [], 'dict_mcc': []}
    log_ito = {'step': [], 'mcc_test': [], 'loss_test': [], 'test_flops': [], 'dict_mcc': []}
    
    for i in tqdm(range(num_step), disable=not verbose):
        S_, X_ = model(X_train)
        loss = reconstruction_loss_with_l1(X_train, X_, S_)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if i > 0 and not i % log_step or i == 0:
            log_sae['step'].append(i)
            log_sae['loss_train'].append(loss.item())
            log_sae['mcc_train'].append(mcc(S_train.detach().cpu().numpy(), S_.detach().cpu().numpy()))
            
            with torch.no_grad():
                S_, X_ = model(X_test)
                loss = reconstruction_loss_with_l1(X_test, X_, S_)
            loss_test = loss.item()
            mcc_test = mcc(S_test.detach().cpu().numpy(), S_.detach().cpu().numpy())
            log_sae['loss_test'].append(loss_test)
            log_sae['mcc_test'].append(mcc_test)
            
            # Calculate dictionary MCC for SAE
            D_learned = model.decoder.weight.data
            dict_mcc = calculate_dict_mcc(D_true, D_learned)
            log_sae['dict_mcc'].append(dict_mcc)
            
            # Calculate and log FLOPs for SAE
            train_flops = calculate_sae_training_flops(M, N, X_train.shape[0], i+1, learn_D=model.learn_D)
            test_flops = calculate_sae_inference_flops(M, N, X_test.shape[0])
            log_sae['train_flops'].append(train_flops)
            log_sae['test_flops'].append(test_flops)

            # Run SAE_ITO
            ito_model = SparseCoding(X_test, D_learned, learn_D=False).to(device)
            S_test_opt = ito_model.optimize_codes(X_test, num_iterations=10_000)
            X_test_ = S_test_opt @ ito_model.D.T
            loss_test_ito = reconstruction_loss_with_l1(X_test, X_test_, S_test_opt)
            mcc_test_ito = mcc(S_test.cpu().numpy(), S_test_opt.cpu().numpy())
            
            log_ito['step'].append(i)
            log_ito['loss_test'].append(loss_test_ito.item())
            log_ito['mcc_test'].append(mcc_test_ito)
            log_ito['dict_mcc'].append(dict_mcc)  # Same as SAE
            #log_ito['test_flops'].append(calculate_sparse_coding_inference_flops(M, N, X_test.shape[0], learn_D=False))
            log_ito['test_flops'].append(calculate_optimize_codes_flops(M, N, X_test.shape[0], 10_000))

            if i % 1000 == 0 or i == 0:
                print(f"Step {i}: SAE Loss Test = {loss_test:.4f}, SAE MCC Test = {mcc_test:.4f}, SAE Dict MCC = {dict_mcc:.4f}")
                print(f"         ITO Loss Test = {loss_test_ito:.4f}, ITO MCC Test = {mcc_test_ito:.4f}")

    print(f"Final SAE MCC: {log_sae['mcc_test'][-1]:.4f}, Final SAE Dict MCC: {log_sae['dict_mcc'][-1]:.4f}, Final SAE Test Loss: {log_sae['loss_test'][-1]:.4f}")
    print(f"Final ITO MCC: {log_ito['mcc_test'][-1]:.4f}")
    return log_sae, log_ito, D_learned

def run_experiment(model, X_train, S_train, X_test, S_test, D_true, num_step=30000, log_step=10, seed=20240625):
    torch.manual_seed(seed)
    if isinstance(model, SparseCoding):
        if model.learn_D:
            log, learned_D = train_sparse_coding(model, X_train, S_train, X_test, S_test, D_true, num_step=num_step, log_step=log_step)
            return log, None, learned_D
        else:
            # This is the standalone SAE_ITO case (not used in the main loop anymore)
            S_test_opt = model.optimize_codes(X_test, num_iterations=10_000)
            X_test_ = S_test_opt @ model.D.T
            loss_test = reconstruction_loss_with_l1(X_test, X_test_, S_test_opt)
            mcc_test = mcc(S_test.cpu().numpy(), S_test_opt.cpu().numpy())
            dict_mcc = calculate_dict_mcc(D_true, model.D.data)
            test_flops = calculate_sparse_coding_inference_flops(M, N, X_test.shape[0], learn_D=False)
            log = {
                'step': [num_step],
                'mcc_test': [mcc_test],
                'loss_test': [loss_test.item()],
                'train_flops': [0],  # No training for SAE_ITO
                'test_flops': [test_flops],
                'dict_mcc': [dict_mcc]
            }
            learned_D = model.D.data
            print(f"Final MCC: {log['mcc_test'][-1]:.4f}, Final Dict MCC: {log['dict_mcc'][-1]:.4f}, Final Test Loss: {log['loss_test'][-1]:.4f}") 
            return log, None, learned_D
    elif isinstance(model, SparseAutoEncoder):
        return train_sae_with_ito(model, X_train, S_train, X_test, S_test, D_true, num_step=num_step, log_step=log_step)
    else:  # MLP
        log, learned_D = train(model, X_train, S_train, X_test, S_test, D_true, num_step=num_step, log_step=log_step)
        return log, None, learned_D

def average_logs(logs):
    avg_log = {
        'step': logs[0]['step'],
        'mcc_test': np.mean([log['mcc_test'] for log in logs], axis=0),
        'mcc_test_std': np.std([log['mcc_test'] for log in logs], axis=0),
        'dict_mcc': np.mean([log['dict_mcc'] for log in logs], axis=0),
        'dict_mcc_std': np.std([log['dict_mcc'] for log in logs], axis=0),
        'train_flops': logs[0]['train_flops'],
        'test_flops': logs[0]['test_flops']
    }
    return avg_log

# Parameters
N = 16  # number of sparse sources
M = 8   # number of measurements
K = 3   # number of active components
hidden_layers = [32, 256]  # list of hidden layer widths
num_runs = 5
num_data = 1024
num_step = 100_000
log_step = 500 #2500
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
logs_sae = []
logs_sae_ito = []
logs_mlps = {h: [] for h in hidden_layers}
logs_sparse_coding = []

for i in tqdm(range(num_runs), desc="Running experiments"):
    run_seed = seed + i

    # Sparse Coding
    SC = SparseCoding(X_test, D.to(device), learn_D=True, seed=run_seed).to(device)
    print(f"Running experiment {i+1}/{num_runs} with Sparse Coding")
    sc_log, _, _ = run_experiment(SC, X_train, S_train, X_test, S_test, D_true, seed=run_seed, num_step=num_step, log_step=log_step)
    logs_sparse_coding.append(sc_log)

    # SAE and SAE_ITO
    SAE = SparseAutoEncoder(M, N, D.to(device), learn_D=True, seed=run_seed).to(device)
    print(f"Running experiment {i+1}/{num_runs} with SAE and SAE_ITO")
    sae_log, sae_ito_log, _ = run_experiment(SAE, X_train, S_train, X_test, S_test, D_true, seed=run_seed, num_step=num_step, log_step=log_step)
    logs_sae.append(sae_log)
    logs_sae_ito.append(sae_ito_log)
    
    for h in hidden_layers:
        print(f"Running experiment {i+1}/{num_runs} with MLP (H={h})")
        MLP_model = MLP(M, N, h, D.to(device), learn_D=True, seed=run_seed).to(device)
        mlp_log, _, _ = run_experiment(MLP_model, X_train, S_train, X_test, S_test, D_true, seed=run_seed, num_step=num_step, log_step=log_step)
        logs_mlps[h].append(mlp_log)

def average_logs(logs, ito=False, sae_avg_logs=None):
    if ito:
        assert sae_avg_logs is not None, "SAE average logs must be provided if ito=True"
    avg_log = {
        'step': logs[0]['step'],
        'mcc_test': np.mean([log['mcc_test'] for log in logs], axis=0),
        'mcc_test_std': np.std([log['mcc_test'] for log in logs], axis=0),
        'loss_test': np.mean([log['loss_test'] for log in logs], axis=0),  # Added this line
        'loss_test_std': np.std([log['loss_test'] for log in logs], axis=0),  # Added this line
        'dict_mcc': np.mean([log['dict_mcc'] for log in logs], axis=0),
        'dict_mcc_std': np.std([log['dict_mcc'] for log in logs], axis=0),
        'train_flops': logs[0]['train_flops'] if not ito else sae_avg_logs['train_flops'],
        'test_flops': logs[0]['test_flops']
    }
    return avg_log


# Average logs
avg_sae = average_logs(logs_sae)
avg_sae_ito = average_logs(logs_sae_ito, ito=True, sae_avg_logs=avg_sae)
avg_sparse_coding = average_logs(logs_sparse_coding)
avg_mlps = {h: average_logs(logs) for h, logs in logs_mlps.items()}

# Save results as JSON
results = {
    "SAE": avg_sae,
    "SAE_ITO": avg_sae_ito,
    "SparseCoding": avg_sparse_coding,
    "MLPs": avg_mlps
}

with open('results/unknown_Z_D_flops_reconstruction.json', 'w') as f:
    json.dump(numpy_to_list(results), f)

print("Experiment completed. Results saved to 'results/unknown_Z_D_flops_reconstruction.json'.")