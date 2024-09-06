import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm
import itertools

from models import MLP
from metrics import mcc
from utils import numpy_to_list, generate_data, reconstruction_loss_with_l1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_dict_mcc(D_true, D_learned):
    return mcc(D_true.cpu().numpy(), D_learned.cpu().numpy())

def train(model, X_train, S_train, X_test, S_test, D_true, lr=3e-4, num_step=100_000, log_step=10_000):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    final_log = {}
    
    for i in tqdm(range(num_step), leave=False):
        S_, X_ = model(X_train)
        loss = reconstruction_loss_with_l1(X_train, X_, S_)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if i == num_step - 1:
            with torch.no_grad():
                S_, X_ = model(X_test)
                loss_test = reconstruction_loss_with_l1(X_test, X_, S_)
            
            mcc_test = mcc(S_test.detach().cpu().numpy(), S_.detach().cpu().numpy())
            dict_mcc = calculate_dict_mcc(D_true, model.decoder.weight.data)
            l0_test = torch.mean((S_.abs() > 0).float()) #torch.mean((S_.abs() > 1e-10).float().sum(dim=1))
            
            final_log = {
                'loss_test': loss_test.item(),
                'mcc_test': mcc_test,
                'dict_mcc': dict_mcc,
                'l0_test': l0_test.item()
            }

            print(f"Loss: {loss_test.item():.4f}, Mcc: {mcc_test:.4f}, Dict Mcc: {dict_mcc:.4f}, L0: {l0_test.item():.4f}")

    return final_log

def run_experiment(N, M, K, hidden_widths, num_runs, num_data, seed):
    results = {h: [] for h in hidden_widths}
    
    # Generate data
    S, X, D = generate_data(N, M, K, num_data * 2, seed=seed)
    D = D.T
    S_train = S[:num_data].to(device)
    X_train = X[:num_data].to(device)
    S_test = S[num_data:].to(device)
    X_test = X[num_data:].to(device)
    D_true = D.to(device)
    
    for h in tqdm(hidden_widths, desc=f"Hidden widths for N={N}, M={M}, K={K}"):
        for run in range(num_runs):
            print(f"Running MLP experiment for N={N}, M={M}, K={K}, hidden width={h}, run={run}")
            run_seed = seed + run
            torch.manual_seed(run_seed)
            
            model = MLP(M, N, h, D.to(device), learn_D=True, seed=run_seed).to(device)
            log = train(model, X_train, S_train, X_test, S_test, D_true)
            results[h].append(log)
    
    return results

def main():
    # Parameters
    configs = [(16, 8, 3), (32, 16, 6), (64, 16, 6)]
    num_runs = 3
    num_data = 1024
    seed = 20240926
    
    all_results = {}
    
    for N, M, K in configs:
        hidden_widths = [M * 2**i for i in range(int(np.log2(32)) + 1)]
        print(f"Running MLP experiment for N={N}, M={M}, K={K}")
        results = run_experiment(N, M, K, hidden_widths, num_runs, num_data, seed)
        all_results[f"N{N}_M{M}_K{K}"] = results
    
    # Save results as JSON
    with open('results/mlp_hidden_width_experiment.json', 'w') as f:
        json.dump(numpy_to_list(all_results), f)

    print("Experiment completed. Results saved to 'results/mlp_hidden_width_experiment.json'.")

if __name__ == "__main__":
    main()