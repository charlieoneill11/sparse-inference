import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import pearsonr as corr
from tqdm import tqdm
import yaml
import json
from munkres import Munkres
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
hidden_dims = [10, 16] #[8, 10, 12, 16]

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

def train_model(model, X, S, num_step, lr, l1_weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_flops = 0
    
    for i in tqdm(range(num_step), desc=f"Training {model.__class__.__name__}"):
        if isinstance(model, GatedSAE):
            S_, X_, loss = model.loss_forward(X, l1_weight=l1_weight)
        else:
            S_, X_ = model.forward(X)
            loss = F.mse_loss(X_, X) + l1_weight * torch.sum(torch.abs(S_))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_flops += calculate_flops(model.__class__.__name__, model.D_.shape[0], M, K, 1)  # 1 iteration

        if i > 0 and not i % 1000:
            if not torch.all(S_.var(0) > 0):
                print('dead latents')
            print('step', i, 'loss', loss.item())
    
    return S_.detach().cpu().numpy(), train_flops

hidden_dim = 14
S, X, D = generate_data(N, M, K, num_data, seed)
print(S.shape, X.shape, D.shape)

S_hidden = S @ torch.randn(N, hidden_dim, dtype=torch.float32)
D_hidden = torch.randn(hidden_dim, N, dtype=torch.float32) @ D
print('S_hidden shape: ', S_hidden.shape)
print('D_hidden shape: ', D_hidden.shape)

model = SparseCoding(S_hidden, D_hidden, learn_D=False, seed=seed).to(device)

S_, train_flops = train_model(model, X, S, num_step, 1e-4, 3e-4)
print('train_flops', train_flops)
print('S_ shape: ', S_.shape)

# Pad the second dimension of S_ with zeros to be the same size as S
pad_amount = N - hidden_dim
S_padded = np.pad(S_, ((0, 0), (0, pad_amount)), mode='constant', constant_values=0)
S = np.array(S)
S_padded = np.array(S_padded)
print("Original S_ shape:", S_.shape)
print("Padded S_ shape:", S_padded.shape)
print("S shape:", S.shape)

print('S_:', S_)
print('S:', S)
print('S_padded:', S_padded)

# Calculate MCC
# Evaluation metric
munk = Munkres()
def match_latents(z, z_):
    matches = np.zeros((z.shape[1], z_.shape[1]))
    for i in range(z.shape[1]):
        for j in range(z_.shape[1]):
            print(z[:, i])
            print(z_[:, j])
            corr_val = abs(corr(z[:, i], z_[:, j])[0])
            matches[i, j] = corr_val if not np.isnan(corr_val) else 1e-2
            print(matches[i, j])
            print()
    matches[np.isnan(matches)] = 0.0
    indexes = munk.compute(-matches)
    print('matches:', matches)
    print('indexes:', indexes)
    return matches, indexes


def eval_nd(z, z_):
    matches, indexes = match_latents(z, z_)
    corrs = []
    for i in indexes:
        corrs.append(matches[i[0], i[1]])
    return corrs


def mcc(z, z_):
    return np.mean(eval_nd(z, z_)[0])

mcc_value = mcc(S, S_padded)
print('MCC:', mcc_value)