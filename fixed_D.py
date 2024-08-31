import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import pearsonr as corr
from tqdm import tqdm
import json
from munkres import Munkres
from metrics import mcc

# Import your existing models and functions
from models import MLP, SparseAutoEncoder, SparseCoding
from flop_counter import calculate_inference_flops, calculate_training_flops

# Parameters
N = 16  # number of sparse sources
K = 3  # number of active components
M = 8  # number of measurements
base_seed = 20240625
num_data = 1024
lr = 1e-3
num_step = 10000
weight = 1e-2
batch_size = 32
eval_interval = 250
num_runs = 5  # Number of runs for each model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

LEARN_D = True

# MLP expansion factors
mlp_expansions = [4, 8, 16]

# SparseCoding parameters
sc_inference_iterations = 50000
sc_inference_lr = 3e-3

# ITO parameters
ito_num_steps = 10000
ito_lr = 1e-2

class ITO(nn.Module):
    def __init__(self, D_, X, initial_S):
        super().__init__()
        self.log_S_ = nn.Parameter(data=initial_S.detach().clone(), requires_grad=True)
        self.D_ = nn.Parameter(D_, requires_grad=False)

    def forward(self):
        return torch.exp(self.log_S_), torch.exp(self.log_S_) @ self.D_
    
    def optimise(self, X, lr=3e-3, num_steps=1000):
        optimizer = torch.optim.Adam([self.log_S_], lr=lr)
        for i in range(num_steps):
            optimizer.zero_grad()
            S_, X_ = self.forward()
            loss = torch.sum((X - X_) ** 2) + 1e-1 * torch.sum(torch.abs(S_))
            loss.backward()
            if i == 0:
                print(f"ITO Loss: {loss.item()}")
            optimizer.step()
        print(f"ITO Loss: {loss.item()}")
        return torch.exp(self.log_S_)
    

def calculate_ito_flops(N, M, num_steps):
    flops_per_step = 2 * N * M + N + M  # forward pass
    flops_per_step += N  # loss calculation
    flops_per_step += 4 * N  # backward pass and parameter update
    return flops_per_step * num_steps

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

def train_and_evaluate(model, X, S, num_step, lr, weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    results = []
    ito_results = []
    total_flops = 0
    
    for i in tqdm(range(num_step)):
        S_, X_ = model(X)
        loss = criterion(S_, X, X_, weight=weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % eval_interval == 0 or i == num_step - 1:
            if isinstance(model, SparseCoding):
                S_ = model.infer(X, num_iterations=sc_inference_iterations, lr=sc_inference_lr, l1_weight=weight)
            else:
                with torch.no_grad():
                    S_, _ = model(X)
            S_np = S_.cpu().numpy()
            current_mcc = mcc(S, S_np)
            
            if isinstance(model, MLP):
                inference_flops = calculate_inference_flops(model.__class__.__name__, N, M, K, 1, model.projections_up)
                training_flops = calculate_training_flops(model.__class__.__name__, N, M, K, batch_size, i+1, model.projections_up)
            elif isinstance(model, SparseCoding):
                inference_flops = calculate_inference_flops(model.__class__.__name__, N, M, K, sc_inference_iterations)
                training_flops = calculate_training_flops(model.__class__.__name__, N, M, K, batch_size, i+1)
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
            
            # Run ITO for SparseAutoEncoder
            if isinstance(model, SparseAutoEncoder):
                initial_S, _ = model(X)
                ito = ITO(model.D_, X, initial_S).to(device)
                ito_S_ = ito.optimise(X, lr=ito_lr, num_steps=ito_num_steps)
                ito_S_np = ito_S_.detach().cpu().numpy()
                print(f"S: {S.shape}, ITO S: {ito_S_np.shape}, X shape: {X.shape}")
                # Sanity check
                X_test = ito_S_ @ model.D_
                # print(f"ITO Loss sanity check: {torch.sum((X - X_test) ** 2).item()}")
                # print(f"ITO S: {ito_S_np}, True S: {S}")
                ito_mcc = mcc(S, ito_S_np)
                ito_flops = calculate_ito_flops(N, M, ito_num_steps)
                
                ito_results.append({
                    'step': i,
                    'mcc': ito_mcc,
                    'inference_flops': inference_flops + ito_flops,
                    'training_flops': training_flops,
                    'total_flops': total_flops + ito_flops
                })
                
                print(f'ITO Step {i}, MCC: {ito_mcc:.4f}, Total FLOPs: {total_flops + ito_flops}')
    
    return results, ito_results

def generate_data(seed):
    S = []
    np.random.seed(seed)
    for i in range(num_data):
        S.append(sample_signal())
    S = np.array(S)
    torch.manual_seed(seed)
    D = torch.randn(N, M, dtype=torch.float32).to(device)
    D /= torch.linalg.norm(D, dim=1, keepdim=True)
    X = torch.tensor(S, dtype=torch.float32).to(device) @ D
    print(S.shape, X.shape, D.shape)
    return S, X, D

# Train models
models = {
    'SparseAutoEncoder': lambda D, seed: SparseAutoEncoder(D, learn_D=LEARN_D, seed=seed),
    #'SparseCoding': lambda D, seed: SparseCoding(torch.zeros(num_data, N), D, learn_D=LEARN_D, seed=seed)
}

# Add MLP models with different expansions
for k in mlp_expansions:
    models[f'MLP_x{k}'] = lambda D, seed, k=k: MLP(D, projections_up=[M*k, N], learn_D=LEARN_D, seed=seed)

results = {}

for model_name, model_fn in models.items():
    model_results = []
    ito_model_results = []
    for run in range(num_runs):
        run_seed = base_seed + run
        print(f"\nTraining {model_name}, Run {run + 1}/{num_runs}, Seed: {run_seed}")
        
        # Generate new data for each run
        S, X, D = generate_data(run_seed)
        print(f"Data shapes - S: {S.shape}, X: {X.shape}, D: {D.shape}")
        
        model = model_fn(D, run_seed).to(device)
        if model_name == 'SparseAutoEncoder':
            num_steps_final = int(num_step * 2)
        elif model_name == 'SparseCoding':
            num_steps_final = int(num_step / 1)
        else:
            num_steps_final = int(num_step / 2)
        run_results, ito_results = train_and_evaluate(model, X, S, num_steps_final, lr, weight)
        model_results.append(run_results)
        if ito_results:
            ito_model_results.append(ito_results)
    results[model_name] = model_results
    if ito_model_results:
        results[f'{model_name}_ITO'] = ito_model_results

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
        'num_runs': num_runs,
        'mlp_expansions': mlp_expansions,
        'sc_inference_iterations': sc_inference_iterations,
        'sc_inference_lr': sc_inference_lr,
        'ito_num_steps': ito_num_steps,
        'ito_lr': ito_lr
    },
    'results': results
}
filename = 'fixed_D' if not LEARN_D else 'learned_D'
with open(f'results/{filename}_results.json', 'w') as f:
    json.dump(experiment_data, f, indent=2)

print(f"File saved to results/{filename}_results.json")