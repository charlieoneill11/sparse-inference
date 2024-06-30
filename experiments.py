import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import pearsonr as corr
from tqdm import tqdm
import yaml
from munkres import Munkres
import json
from models import SparseCoding, SparseAutoEncoder, GatedSAE, TopKSAE
from flop_counter import calculate_flops

# Parameters
N = 16  # number of sparse sources (true dimension)
K = 3  # number of active components
M = 8  # number of measurements
seed = 20240625
num_data = 1024
num_step = 20000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Autoencoder hidden dimensions to test
hidden_dims = [4, 8, 10, 12, 16]

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
    target_size = max(z.shape[1], z_.shape[1])
    print('target size', target_size)
    print('z shape', z.shape)
    print('z_ shape', z_.shape)
    if not z.shape[1] == z_.shape[1] == target_size:
        z = pad_to_size(z, target_size)
        z_ = pad_to_size(z_, target_size)
    matches = np.zeros((z.shape[1], z_.shape[1]))
    for i in range(z.shape[1]):
        for j in range(z_.shape[1]):
            matches[i, j] = abs(corr(z[:, i], z_[:, j])[0])
    matches[np.isnan(matches)] = 0.0
    #return np.mean([matches[i, j] for i, j in enumerate(np.argmax(matches, axis=1))])

    munk = Munkres()
    indexes = munk.compute(-matches)

    corrs = []
    for i in indexes:
        corrs.append(matches[i[0], i[1]])

    return np.mean(corrs[0])
    


def train_model(model, X, S, num_step, lr, l1_weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_flops = 0
    
    for i in tqdm(range(num_step), desc=f"Training {model.__class__.__name__}"):
        if isinstance(model, GatedSAE):
            S_, X_, loss = model.loss_forward(X, l1_weight=l1_weight)
        else:
            S_, X_ = model.forward(X)
            loss = torch.sum((X - X_) ** 2) + l1_weight * torch.sum(torch.abs(S_))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_flops += calculate_flops(model.__class__.__name__, model.D_.shape[0], M, K, 1)  # 1 iteration

        if (i > 0 and not i % (num_step // 10)) or i == 0:
            if not torch.all(S_.var(0) > 0):
                print('dead latents')
            print('step', i, 'loss', loss.item(),
                  'MCC', mcc(S.cpu().numpy(), S_.detach().cpu().numpy()))
    
    return S_.detach().cpu().numpy(), train_flops

def run_experiment(model_class, hidden_dim, X, S, D, num_step, lr, l1_weight):
    if hidden_dim <= N:
        print(f"Reducing - Hidden dim = {hidden_dim}, N = {N}, D shape = {D.shape}")
        # Randomly project D to hidden_dim dimensions
        D_hidden = torch.randn(hidden_dim, D.shape[0]) @ D
        D_hidden /= torch.linalg.norm(D_hidden, dim=1, keepdim=True)

        # Create D_hidden with dimensions hidden_dim x M for all models
        D_hidden_test = torch.randn(hidden_dim, M, dtype=torch.float32).to(device)
        D_hidden_test /= torch.linalg.norm(D_hidden_test, dim=1, keepdim=True)

        # Assert these have the same size
        assert D_hidden.shape == D_hidden_test.shape

    else:
        print(f"Staying the same - Hidden dim = {hidden_dim}, N = {N}, D shape = {D.shape}")
        D_hidden = D
    
    if model_class == SparseCoding:
        if hidden_dim <= N:
            S_hidden = torch.zeros(S.shape[0], hidden_dim).to(device)
        else:
            S_hidden = S
        model = model_class(S_hidden, D_hidden, learn_D=True).to(device)
    elif model_class == TopKSAE:
        model = model_class(D_hidden, learn_D=True, k=K, seed=seed).to(device)
    else:
        model = model_class(D_hidden, learn_D=True, seed=seed).to(device)
    
    S_, train_flops = train_model(model, X, S, num_step, lr, l1_weight)
    
    inference_flops = calculate_flops(model_class.__name__, hidden_dim, M, K, 1)
    final_mcc = mcc(S.numpy(), S_)
    
    return {
        'model': model_class.__name__,
        'hidden_dim': hidden_dim,
        'train_flops': train_flops,
        'inference_flops': inference_flops,
        'mcc': final_mcc
    }

def main():
    S, X, D = generate_data(N, M, K, num_data, seed)
    X, D = X.to(device), D.to(device)
    
    with open('train_configs.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    results = []
    
    for model_class in [SparseCoding, SparseAutoEncoder, GatedSAE, TopKSAE]:
        model_config = config[model_class.__name__]
        lr = float(model_config['lr'])
        l1_weight = float(model_config['l1_weight'])
        print(f"Running experiment for {model_class.__name__} with lr={lr}, l1_weight={l1_weight}")
        
        for hidden_dim in hidden_dims:
            print(f"Running experiment for {model_class.__name__} with hidden_dim={hidden_dim}")
            result = run_experiment(model_class, hidden_dim, X, S, D, num_step, lr, l1_weight)
            results.append(result)
    
    # Save results
    experiment_data = {
        'parameters': {
            'N': N,
            'M': M,
            'K': K,
            'num_data': num_data,
            'num_step': num_step,
            'seed': seed,
            'hidden_dims': hidden_dims
        },
        'results': results
    }
    
    with open('results/experiment_results.json', 'w') as f:
        json.dump(experiment_data, f, indent=2)

if __name__ == "__main__":
    main()