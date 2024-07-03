# import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F
# from scipy.stats import pearsonr as corr
# from tqdm import tqdm
# import yaml
# from munkres import Munkres
# import json
# from models import SparseCoding, SparseAutoEncoder, GatedSAE, TopKSAE
# from flop_counter import calculate_inference_flops, calculate_training_flops

# # Parameters
# N = 16  # number of sparse sources (true dimension)
# K = 3  # number of active components
# M = 8  # number of measurements
# seed = 20240625
# num_data = 1024
# num_step = 20000
# batch_size = 32
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Autoencoder hidden dimensions to test
# hidden_dims = [4, 8, 10, 12, 16]

# def generate_data(N, M, K, num_data, seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
    
#     S = []
#     for _ in range(num_data):
#         s = np.abs(np.random.normal(0, 1, N))
#         mask = np.zeros(N)
#         mask[np.random.choice(N, K, replace=False)] = 1
#         s *= mask
#         S.append(s)
#     S = np.array(S)
    
#     D = torch.randn(N, M, dtype=torch.float32)
#     D /= torch.linalg.norm(D, dim=1, keepdim=True)
    
#     X = torch.tensor(S, dtype=torch.float32) @ D
    
#     return torch.tensor(S, dtype=torch.float32), X, D

# def pad_to_size(array, target_size):
#     current_size = array.shape[1]
#     if current_size < target_size:
#         padding = np.zeros((array.shape[0], target_size - current_size))
#         return np.concatenate([array, padding], axis=1)
#     return array

# def mcc(z, z_):
#     target_size = max(z.shape[1], z_.shape[1])
#     if not z.shape[1] == z_.shape[1] == target_size:
#         z = pad_to_size(z, target_size)
#         z_ = pad_to_size(z_, target_size)
#     matches = np.zeros((z.shape[1], z_.shape[1]))
#     for i in range(z.shape[1]):
#         for j in range(z_.shape[1]):
#             matches[i, j] = abs(corr(z[:, i], z_[:, j])[0])
#     matches[np.isnan(matches)] = 0.0

#     munk = Munkres()
#     indexes = munk.compute(-matches)

#     corrs = []
#     for i in indexes:
#         corrs.append(matches[i[0], i[1]])

#     return np.mean(corrs)

# def train_model(model, X, S, num_step, lr, l1_weight):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     best_mcc = -float('inf')
    
#     for i in tqdm(range(num_step), desc=f"Training {model.__class__.__name__}"):
#         if isinstance(model, GatedSAE):
#             S_, X_, loss = model.loss_forward(X, l1_weight=l1_weight)
#         else:
#             S_, X_ = model.forward(X)
#             loss = torch.sum((X - X_) ** 2) + l1_weight * torch.sum(torch.abs(S_))
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i > 0 and not i % (num_step // 10)) or i == 0:
#             if not torch.all(S_.var(0) > 0):
#                 print('dead latents')
#             current_mcc = mcc(S.cpu().numpy(), S_.detach().cpu().numpy())
#             best_mcc = max(best_mcc, current_mcc)
#             print('step', i, 'loss', loss.item(), 'MCC', current_mcc, 'Best MCC', best_mcc)
    
#     return best_mcc

# def run_experiment(model_class, hidden_dim, X, S, D, num_step, lr, l1_weight):
#     if hidden_dim <= N:
#         D_hidden = torch.randn(hidden_dim, D.shape[0]) @ D
#         D_hidden /= torch.linalg.norm(D_hidden, dim=1, keepdim=True)
#     else:
#         D_hidden = D
    
#     if model_class == SparseCoding:
#         S_hidden = torch.zeros(S.shape[0], hidden_dim).to(device) if hidden_dim <= N else S
#         model = model_class(S_hidden, D_hidden, learn_D=True).to(device)
#     elif model_class == TopKSAE:
#         model = model_class(D_hidden, learn_D=True, k=K, seed=seed).to(device)
#     else:
#         model = model_class(D_hidden, learn_D=True, seed=seed).to(device)
    
#     best_mcc = train_model(model, X, S, num_step, lr, l1_weight)
    
#     num_iterations = 100  # or whatever value you want to use
#     inference_flops = calculate_inference_flops(model_class.__name__, hidden_dim, M, K, num_iterations)
#     training_flops = calculate_training_flops(model_class.__name__, hidden_dim, M, K, batch_size, num_step)
#     total_flops = inference_flops + training_flops
    
#     return {
#         'model': model_class.__name__,
#         'hidden_dim': hidden_dim,
#         'inference_flops': inference_flops,
#         'training_flops': training_flops,
#         'total_flops': total_flops,
#         'mcc': best_mcc
#     }

# def main():
#     S, X, D = generate_data(N, M, K, num_data, seed)
#     X, D = X.to(device), D.to(device)
    
#     with open('train_configs.yaml', 'r') as f:
#         config = yaml.safe_load(f)
    
#     results = []
    
#     for model_class in [SparseCoding, SparseAutoEncoder, GatedSAE, TopKSAE]:
#         model_config = config[model_class.__name__]
#         lr = float(model_config['lr'])
#         l1_weight = float(model_config['l1_weight'])
#         print(f"Running experiment for {model_class.__name__} with lr={lr}, l1_weight={l1_weight}")
        
#         for hidden_dim in hidden_dims:
#             print(f"Running experiment for {model_class.__name__} with hidden_dim={hidden_dim}")
#             result = run_experiment(model_class, hidden_dim, X, S, D, num_step, lr, l1_weight)
#             results.append(result)
    
#     # Save results
#     experiment_data = {
#         'parameters': {
#             'N': N,
#             'M': M,
#             'K': K,
#             'num_data': num_data,
#             'num_step': num_step,
#             'batch_size': batch_size,
#             'seed': seed,
#             'hidden_dims': hidden_dims
#         },
#         'results': results
#     }
    
#     with open('results/experiment_results.json', 'w') as f:
#         json.dump(experiment_data, f, indent=2)

# if __name__ == "__main__":
#     main()


# import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F
# from scipy.stats import pearsonr as corr
# from tqdm import tqdm
# import yaml
# from munkres import Munkres
# import json
# from models import SparseCoding, SparseAutoEncoder, GatedSAE, TopKSAE
# from flop_counter import calculate_inference_flops, calculate_training_flops

# # Parameters
# N = 16  # number of sparse sources (true dimension)
# K = 3  # number of active components
# M = 8  # number of measurements
# seed = 20240625
# num_data = 1024
# num_step = 20000
# batch_size = 32
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Autoencoder hidden dimensions to test
# hidden_dims = [4, 8, 10, 12, 16]

# def generate_data(N, M, K, num_data, seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
    
#     S = []
#     for _ in range(num_data):
#         s = np.abs(np.random.normal(0, 1, N))
#         mask = np.zeros(N)
#         mask[np.random.choice(N, K, replace=False)] = 1
#         s *= mask
#         S.append(s)
#     S = np.array(S)
    
#     D = torch.randn(N, M, dtype=torch.float32)
#     D /= torch.linalg.norm(D, dim=1, keepdim=True)
    
#     X = torch.tensor(S, dtype=torch.float32) @ D
    
#     return torch.tensor(S, dtype=torch.float32), X, D

# def pad_to_size(array, target_size):
#     current_size = array.shape[1]
#     if current_size < target_size:
#         padding = np.zeros((array.shape[0], target_size - current_size))
#         return np.concatenate([array, padding], axis=1)
#     return array

# def mcc(z, z_):
#     target_size = max(z.shape[1], z_.shape[1])
#     if not z.shape[1] == z_.shape[1] == target_size:
#         z = pad_to_size(z, target_size)
#         z_ = pad_to_size(z_, target_size)
#     matches = np.zeros((z.shape[1], z_.shape[1]))
#     for i in range(z.shape[1]):
#         for j in range(z_.shape[1]):
#             matches[i, j] = abs(corr(z[:, i], z_[:, j])[0])
#     matches[np.isnan(matches)] = 0.0

#     munk = Munkres()
#     indexes = munk.compute(-matches)

#     corrs = []
#     for i in indexes:
#         corrs.append(matches[i[0], i[1]])

#     return np.mean(corrs)

# def train_model(model, X_train, S_train, num_step, lr, l1_weight):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     best_mcc = -float('inf')
    
#     for i in tqdm(range(num_step), desc=f"Training {model.__class__.__name__}"):
#         if isinstance(model, GatedSAE):
#             S_, X_, loss = model.loss_forward(X_train, l1_weight=l1_weight)
#         else:
#             S_, X_ = model.forward(X_train)
#             loss = torch.sum((X_train - X_) ** 2) + l1_weight * torch.sum(torch.abs(S_))
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i > 0 and not i % (num_step // 10)) or i == 0:
#             if not torch.all(S_.var(0) > 0):
#                 print('dead latents')
#             current_mcc = mcc(S_train.cpu().numpy(), S_.detach().cpu().numpy())
#             best_mcc = max(best_mcc, current_mcc)
#             print('step', i, 'loss', loss.item(), 'MCC', current_mcc, 'Best MCC', best_mcc)
    
#     return best_mcc

# def run_experiment(model_class, hidden_dim, X_train, S_train, D, X_test, S_test, num_step, lr, l1_weight):
#     if hidden_dim <= N:
#         D_hidden = torch.randn(hidden_dim, D.shape[0]) @ D
#         D_hidden /= torch.linalg.norm(D_hidden, dim=1, keepdim=True)
#     else:
#         D_hidden = D
    
#     if model_class == SparseCoding:
#         S_hidden = torch.zeros(S_train.shape[0], hidden_dim).to(device) if hidden_dim <= N else S_train
#         model = model_class(S_hidden, D_hidden, learn_D=True).to(device)
#     elif model_class == TopKSAE:
#         model = model_class(D_hidden, learn_D=True, k=K, seed=seed).to(device)
#     else:
#         model = model_class(D_hidden, learn_D=True, seed=seed).to(device)
    
#     best_mcc = train_model(model, X_train, S_train, num_step, lr, l1_weight)
    
#     if model_class == SparseCoding:
#         S_test_ = model.infer(X_test, num_iterations=10000, lr=lr, l1_weight=l1_weight)
#     elif model_class == GatedSAE:
#         S_test_, _, _ = model.loss_forward(X_test, l1_weight=l1_weight)
#     else:
#         S_test_, _ = model.forward(X_test)
    
#     test_mcc = mcc(S_test.cpu().numpy(), S_test_.detach().cpu().numpy())
    
#     num_iterations = 100  # or whatever value you want to use
#     inference_flops = calculate_inference_flops(model_class.__name__, hidden_dim, M, K, num_iterations)
#     training_flops = calculate_training_flops(model_class.__name__, hidden_dim, M, K, batch_size, num_step)
#     total_flops = inference_flops + training_flops
    
#     return {
#         'model': model_class.__name__,
#         'hidden_dim': hidden_dim,
#         'inference_flops': inference_flops,
#         'training_flops': training_flops,
#         'total_flops': total_flops,
#         'mcc': test_mcc
#     }

# def main():
#     S, X, D = generate_data(N, M, K, num_data, seed)
#     X, D = X.to(device), D.to(device)
    
#     train_size = int(0.8 * num_data)
#     S_train, S_test = S[:train_size], S[train_size:]
#     X_train, X_test = X[:train_size], X[train_size:]
    
#     with open('train_configs.yaml', 'r') as f:
#         config = yaml.safe_load(f)
    
#     results = []
    
#     for model_class in [SparseCoding, SparseAutoEncoder, GatedSAE, TopKSAE]:
#         model_config = config[model_class.__name__]
#         lr = float(model_config['lr'])
#         l1_weight = float(model_config['l1_weight'])
#         print(f"Running experiment for {model_class.__name__} with lr={lr}, l1_weight={l1_weight}")
        
#         for hidden_dim in hidden_dims:
#             print(f"Running experiment for {model_class.__name__} with hidden_dim={hidden_dim}")
#             result = run_experiment(model_class, hidden_dim, X_train, S_train, D, X_test, S_test, num_step, lr, l1_weight)
#             results.append(result)
    
#     # Save results
#     experiment_data = {
#         'parameters': {
#             'N': N,
#             'M': M,
#             'K': K,
#             'num_data': num_data,
#             'num_step': num_step,
#             'batch_size': batch_size,
#             'seed': seed,
#             'hidden_dims': hidden_dims
#         },
#         'results': results
#     }
    
#     with open('results/experiment_results.json', 'w') as f:
#         json.dump(experiment_data, f, indent=2)

# if __name__ == "__main__":
#     main()

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
from flop_counter import calculate_inference_flops, calculate_training_flops

# Parameters
N = 16  # number of sparse sources (true dimension)
K = 3  # number of active components
M = 8  # number of measurements
seed = 20240625
num_data = 1024
num_step = 20000
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Autoencoder hidden dimensions to test
hidden_dims = [4, 8, 10, 12, 16]

# Define grid search parameters
lr_range = [1e-4, 1e-3, 1e-2]
l1_weight_range = [1e-5, 1e-4, 1e-3]

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
        if isinstance(model, GatedSAE):
            S_, X_, loss = model.loss_forward(X_train, l1_weight=l1_weight)
        elif isinstance(model, TopKSAE): # we don't want to apply L1 penalty to TopK
            S_, X_ = model.forward(X_train)
            loss = torch.sum((X_train - X_) ** 2)
        else:
            S_, X_ = model.forward(X_train)
            loss = torch.sum((X_train - X_) ** 2) + l1_weight * torch.sum(torch.abs(S_))
        
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

def run_experiment(model_class, hidden_dim, X_train, S_train, D, X_test, S_test, num_step, lr, l1_weight):
    if hidden_dim <= N:
        D_hidden = torch.randn(hidden_dim, D.shape[0]) @ D
        D_hidden /= torch.linalg.norm(D_hidden, dim=1, keepdim=True)
    else:
        D_hidden = D
    
    if model_class == SparseCoding:
        S_hidden = torch.zeros(S_train.shape[0], hidden_dim).to(device) if hidden_dim <= N else S_train
        model = model_class(S_hidden, D_hidden, learn_D=True).to(device)
    elif model_class == TopKSAE:
        model = model_class(D_hidden, learn_D=True, k=K, seed=seed).to(device)
    else:
        model = model_class(D_hidden, learn_D=True, seed=seed).to(device)
    
    best_mcc = train_model(model, X_train, S_train, num_step, lr, l1_weight)

    num_iterations = 10000
    
    if model_class == SparseCoding:
        S_test_ = model.infer(X_test, num_iterations=num_iterations, lr=lr, l1_weight=l1_weight)
    elif model_class == GatedSAE:
        S_test_, _, _ = model.loss_forward(X_test, l1_weight=l1_weight)
        num_iterations = None
    else:
        S_test_, _ = model.forward(X_test)
        num_iterations = None
    
    test_mcc = mcc(S_test.cpu().numpy(), S_test_.detach().cpu().numpy())
    
    num_iterations_flops = 100
    inference_flops = calculate_inference_flops(model_class.__name__, hidden_dim, M, K, num_iterations_flops)
    training_flops = calculate_training_flops(model_class.__name__, hidden_dim, M, K, batch_size, num_step)
    total_flops = inference_flops + training_flops
    
    result = {
        'model': model_class.__name__,
        'hidden_dim': hidden_dim,
        'lr': lr,
        'l1_weight': l1_weight,
        'inference_flops': inference_flops,
        'training_flops': training_flops,
        'total_flops': total_flops,
        'mcc': test_mcc
    }
    if num_iterations is not None:
        result['num_iterations'] = num_iterations

    return result

def grid_search(model_class, hidden_dim, X_train, S_train, D, X_test, S_test, num_step, lr_range, l1_weight_range):
    best_mcc = -float('inf')
    best_params = None

    for lr in lr_range:
        for l1_weight in l1_weight_range:
            print(f"Grid search for {model_class.__name__} with lr={lr}, l1_weight={l1_weight}")
            current_mcc = run_experiment(model_class, hidden_dim, X_train, S_train, D, X_test, S_test, num_step, lr, l1_weight)['mcc']
            if current_mcc > best_mcc:
                best_mcc = current_mcc
                best_params = (lr, l1_weight)

    return best_params

def main():
    S, X, D = generate_data(N, M, K, num_data, seed)
    X, D = X.to(device), D.to(device)
    
    train_size = int(0.8 * num_data)
    S_train, S_test = S[:train_size], S[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]
    
    with open('train_configs.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    results = []
    
    for model_class in [SparseCoding, SparseAutoEncoder, GatedSAE, TopKSAE]:
        print(f"Running grid search for {model_class.__name__}")
        for hidden_dim in hidden_dims:
            print(f"Running grid search for {model_class.__name__} with hidden_dim={hidden_dim}")
            best_lr, best_l1_weight = grid_search(model_class, hidden_dim, X_train, S_train, D, X_test, S_test, num_step, lr_range, l1_weight_range)
            print(f"Best parameters for {model_class.__name__} with hidden_dim={hidden_dim}: lr={best_lr}, l1_weight={best_l1_weight}")
            result = run_experiment(model_class, hidden_dim, X_train, S_train, D, X_test, S_test, num_step, best_lr, best_l1_weight)
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
            'hidden_dims': hidden_dims,
            'lr_range': lr_range,
            'l1_weight_range': l1_weight_range
        },
        'results': results
    }
    
    with open('results/experiment_results.json', 'w') as f:
        json.dump(experiment_data, f, indent=2)

if __name__ == "__main__":
    main()