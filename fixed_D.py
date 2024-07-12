# import numpy as np
# import torch
# from torch import nn
# import torch.nn.functional as F
# from scipy.stats import pearsonr as corr
# from tqdm import tqdm
# import json
# from munkres import Munkres

# # Import your existing models and functions
# from models import MLP, SparseAutoEncoder
# from flop_counter import calculate_inference_flops, calculate_training_flops

# # Parameters
# N = 16  # number of sparse sources
# K = 3  # number of active components
# M = 8  # number of measurements
# seed = 20240625
# num_data = 1024
# lr = 1e-3
# num_step = 20000
# weight = 1e-2
# batch_size = 32
# eval_interval = 100
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using {device}")

# LEARN_D = True

# # Helper functions
# def sample_signal():
#     s = np.random.normal(0, 1, N)
#     s = np.abs(s)
#     ind = np.random.choice(N, K, replace=False)
#     mask = np.zeros(N)
#     mask[ind] = 1
#     s *= mask
#     return s

# def criterion(S_, X, X_, weight=weight):
#     loss = torch.sum((X - X_) ** 2) + weight * torch.sum(torch.abs(S_))
#     return loss


# def mcc(z, z_):
#     def match_latents(z, z_):
#         munk = Munkres()
#         matches = np.zeros((z.shape[1], z_.shape[1]))
#         for i in range(z.shape[1]):
#             for j in range(z_.shape[1]):
#                 matches[i, j] = abs(corr(z[:, i], z_[:, j])[0])
#         matches[np.isnan(matches)] = 0
#         indexes = munk.compute(-matches)
#         return matches, indexes
    
#     matches, indexes = match_latents(z, z_)
#     corrs = [matches[i[0], i[1]] for i in indexes]
#     return np.mean(corrs)

# def train_and_evaluate(model, X, S, num_step, lr, weight):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     results = []
#     total_flops = 0
    
#     for i in tqdm(range(num_step)):
#         S_, X_ = model(X)
#         loss = criterion(S_, X, X_, weight=weight)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if i % eval_interval == 0 or i == num_step - 1:
#             with torch.no_grad():
#                 S_np = S_.cpu().numpy()
#                 current_mcc = mcc(S, S_np)
                
#                 if isinstance(model, MLP):
#                     inference_flops = calculate_inference_flops(model.__class__.__name__, N, M, K, 1, model.projections_up)
#                     training_flops = calculate_training_flops(model.__class__.__name__, N, M, K, batch_size, i+1, model.projections_up)
#                 else:
#                     inference_flops = calculate_inference_flops(model.__class__.__name__, N, M, K, 1)
#                     training_flops = calculate_training_flops(model.__class__.__name__, N, M, K, batch_size, i+1)
                
#                 total_flops = inference_flops + training_flops
                
#                 results.append({
#                     'step': i,
#                     'loss': loss.item(),
#                     'mcc': current_mcc,
#                     'inference_flops': inference_flops,
#                     'training_flops': training_flops,
#                     'total_flops': total_flops
#                 })
                
#                 print(f'Step {i}, Loss: {loss.item():.4f}, MCC: {current_mcc:.4f}, Total FLOPs: {total_flops}')
    
#     return results

# # Generate data
# np.random.seed(seed)
# S = np.array([sample_signal() for _ in range(num_data)])
# torch.manual_seed(seed)
# D = torch.randn(N, M, dtype=torch.float32).to(device)
# D = F.normalize(D, dim=1)
# X = torch.tensor(S, dtype=torch.float32).to(device) @ D
# print(S.shape, X.shape, D.shape)

# # Train models
# models = {
#     'MLP': MLP(D, projections_up=[M*4, N], learn_D=LEARN_D, seed=seed),
#     'SparseAutoEncoder': SparseAutoEncoder(D, learn_D=LEARN_D, seed=seed)
# }

# results = {}

# for model_name, model in models.items():
#     print(f"\nTraining {model_name}")
#     model = model.to(device)
#     if model_name == 'SparseAutoEncoder':
#         num_steps_final = num_step * 4
#     else:
#         num_steps_final = num_step
#     model_results = train_and_evaluate(model, X, S, num_steps_final, lr, weight)
#     results[model_name] = model_results

# # Save results to JSON
# experiment_data = {
#     'parameters': {
#         'N': N,
#         'M': M,
#         'K': K,
#         'num_data': num_data,
#         'num_step': num_step,
#         'batch_size': batch_size,
#         'seed': seed,
#         'lr': lr,
#         'weight': weight,
#         'eval_interval': eval_interval
#     },
#     'results': results
# }
# filename = 'fixed_D' if not LEARN_D else 'learned_D'
# with open(f'results/{filename}_results.json', 'w') as f:
#     json.dump(experiment_data, f, indent=2)

# print(f"File saved to {filename}_results.json")

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import pearsonr as corr
from tqdm import tqdm
import json
from munkres import Munkres

# Import your existing models and functions
from models import MLP, SparseAutoEncoder
from flop_counter import calculate_inference_flops, calculate_training_flops

# Parameters
N = 16  # number of sparse sources
K = 3  # number of active components
M = 8  # number of measurements
base_seed = 20240625
num_data = 1024
lr = 1e-3
num_step = 20000
weight = 1e-2
batch_size = 32
eval_interval = 100
num_runs = 5  # Number of runs for each model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

LEARN_D = True

# Helper functions
def sample_signal():
    s = np.random.normal(0, 1, N)
    s = np.abs(s)
    ind = np.random.choice(N, K, replace=False)
    mask = np.zeros(N)
    mask[ind] = 1
    s *= mask
    return s

def criterion(S_, X, X_, weight=weight):
    loss = torch.sum((X - X_) ** 2) + weight * torch.sum(torch.abs(S_))
    return loss

def mcc(z, z_):
    def match_latents(z, z_):
        munk = Munkres()
        matches = np.zeros((z.shape[1], z_.shape[1]))
        for i in range(z.shape[1]):
            for j in range(z_.shape[1]):
                matches[i, j] = abs(corr(z[:, i], z_[:, j])[0])
        matches[np.isnan(matches)] = 0
        indexes = munk.compute(-matches)
        return matches, indexes
    
    matches, indexes = match_latents(z, z_)
    corrs = [matches[i[0], i[1]] for i in indexes]
    return np.mean(corrs)

def train_and_evaluate(model, X, S, num_step, lr, weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    results = []
    total_flops = 0
    
    for i in tqdm(range(num_step)):
        S_, X_ = model(X)
        loss = criterion(S_, X, X_, weight=weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % eval_interval == 0 or i == num_step - 1:
            with torch.no_grad():
                S_np = S_.cpu().numpy()
                current_mcc = mcc(S, S_np)
                
                if isinstance(model, MLP):
                    inference_flops = calculate_inference_flops(model.__class__.__name__, N, M, K, 1, model.projections_up)
                    training_flops = calculate_training_flops(model.__class__.__name__, N, M, K, batch_size, i+1, model.projections_up)
                else:
                    inference_flops = calculate_inference_flops(model.__class__.__name__, N, M, K, 1)
                    training_flops = calculate_training_flops(model.__class__.__name__, N, M, K, batch_size, i+1)
                
                total_flops = inference_flops + training_flops
                
                results.append({
                    'step': i,
                    'loss': loss.item(),
                    'mcc': current_mcc,
                    'inference_flops': inference_flops,
                    'training_flops': training_flops,
                    'total_flops': total_flops
                })
                
                print(f'Step {i}, Loss: {loss.item():.4f}, MCC: {current_mcc:.4f}, Total FLOPs: {total_flops}')
    
    return results

def generate_data(seed):
    np.random.seed(seed)
    S = np.array([sample_signal() for _ in range(num_data)])
    torch.manual_seed(seed)
    D = torch.randn(N, M, dtype=torch.float32).to(device)
    D = F.normalize(D, dim=1)
    X = torch.tensor(S, dtype=torch.float32).to(device) @ D
    return S, X, D

# Train models
models = {
    'MLP': lambda D, seed: MLP(D, projections_up=[M*2, N], learn_D=LEARN_D, seed=seed),
    'SparseAutoEncoder': lambda D, seed: SparseAutoEncoder(D, learn_D=LEARN_D, seed=seed)
}

results = {}

for model_name, model_fn in models.items():
    model_results = []
    for run in range(num_runs):
        run_seed = base_seed + run
        print(f"\nTraining {model_name}, Run {run + 1}/{num_runs}, Seed: {run_seed}")
        
        # Generate new data for each run
        S, X, D = generate_data(run_seed)
        print(f"Data shapes - S: {S.shape}, X: {X.shape}, D: {D.shape}")
        
        model = model_fn(D, run_seed).to(device)
        if model_name == 'SparseAutoEncoder':
            num_steps_final = int(num_step * 2)
        else:
            num_steps_final = int(num_step)# / 5)
        run_results = train_and_evaluate(model, X, S, num_steps_final, lr, weight)
        model_results.append(run_results)
    results[model_name] = model_results

# Save results to JSON
experiment_data = {
    'parameters': {
        'N': N,
        'M': M,
        'K': K,
        'num_data': num_data,
        'num_step': num_step,
        'batch_size': batch_size,
        'base_seed': base_seed,
        'lr': lr,
        'weight': weight,
        'eval_interval': eval_interval,
        'num_runs': num_runs
    },
    'results': results
}
filename = 'fixed_D' if not LEARN_D else 'learned_D'
with open(f'results/{filename}_results.json', 'w') as f:
    json.dump(experiment_data, f, indent=2)

print(f"File saved to results/{filename}_results.json")