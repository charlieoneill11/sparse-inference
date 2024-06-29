import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as corr
import pickle
from tqdm import tqdm
import sys
from munkres import Munkres
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import gennorm
from typing import Union
from scipy.fft import dct
import einops

from models import SparseCoding, SparseAutoEncoder, GatedSAE, TopKSAE

# parameters
N = 16  # number of sparse sources
K = 3  # number of active components
M = 8  # number of measurements
seed = 20240625
num_data = 1024
lr = 1e-4 #3e-3
num_step = 20000
l1_weight = 3e-4 #1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def criterion(S_, X, X_, l1_weight=l1_weight):
    loss = torch.sum((X - X_) ** 2) + l1_weight * torch.sum(torch.abs(S_))
    return loss

def train(model):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(num_step):
        if isinstance(model, GatedSAE):
            S_, X_, loss = model.loss_forward(X, l1_weight=l1_weight)
        else:
            S_, X_ = model.forward(X)
            loss = criterion(S_, X, X_, l1_weight=l1_weight)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i > 0 and not i % 1000:
            if not torch.all(S_.var(0) > 0):
                print('dead latents')
            print('step', i, 'loss', loss.item(),
                  'MCC', mcc(S, S_.detach().cpu().numpy()))
    S_ = S_.detach().cpu().numpy()
    print('final MCC', mcc(S, S_))
    return S_

def bound(N, K):
    return K * np.log10(N / K)

print('minimum M to solve problem', bound(N, K))

# helper
def sample_codes(S, S_):
    np.random.seed(seed)
    ind = np.random.choice(num_data, 9, replace=False)
    for j, i in enumerate(ind):
        plt.subplot(3, 3, j + 1)
        plt.plot(S[i], '.-')
        plt.plot(S_[i], '.-')
        plt.legend(['true', 'pred'])
    plt.tight_layout()
    plt.show()


def show_correlations(S, S_):
    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.scatter(S[:, i], S_[:, i], s=3)
        plt.grid()
        plt.title('c=%.4f' % corr(S[:, i], S_[:, i])[0])
        plt.xlabel('true')
        plt.ylabel('learned')
    plt.tight_layout()
    plt.show()


def find_permutation(S, S_):
    matches, indexes = match_latents(S, S_)
    S__ = np.zeros_like(S_)
    for i in range(N):
        S__[:, i] = S_[:, indexes[i][1]].copy()
    return S__


def sample_signal():
    s = np.random.normal(0, 1, N)
    s = np.abs(s)
    ind = np.random.choice(N, K, replace=False)
    mask = np.zeros(N)
    mask[ind] = 1
    s *= mask
    return s


def analyze(S, S_):
    # permute
    plt.imshow(match_latents(S, S_)[0])
    plt.colorbar()
    plt.xlabel('true')
    plt.ylabel('learned')
    plt.title('As Learned')
    plt.show()
    S_ = find_permutation(S, S_)
    plt.imshow(match_latents(S, S_)[0])
    plt.colorbar()
    plt.xlabel('true')
    plt.ylabel('learned')
    plt.title('Optimal Permutation')
    plt.show()
    # analyze
    sample_codes(S, S_)
    show_correlations(S, S_)




# Evaluation metric
munk = Munkres()
def match_latents(z, z_):
    matches = np.zeros((z.shape[1], z_.shape[1]))
    for i in range(z.shape[1]):
        for j in range(z_.shape[1]):
            matches[i, j] = abs(corr(z[:, i], z_[:, j])[0])
    matches[np.isnan(matches)] = 0
    indexes = munk.compute(-matches)
    return matches, indexes


def eval_nd(z, z_):
    matches, indexes = match_latents(z, z_)
    corrs = []
    for i in indexes:
        corrs.append(matches[i[0], i[1]])
    return corrs


def mcc(z, z_):
    return np.mean(eval_nd(z, z_)[0])

# generate sample and move to torch
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


#model = SparseCoding(S, D, learn_D=False, seed=seed).to(device)
#model = SparseAutoEncoder(D, learn_D=False, seed=seed, relu=False).to(device)
#model = GatedSAE(D, learn_D=False, seed=seed).to(device)
model = TopKSAE(D, learn_D=False, seed=seed, k=K).to(device)

# Read in lr and l1_weight from config.yaml
import yaml

with open('train_configs.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# If model is SparseCoding
if isinstance(model, SparseCoding):
    lr = float(config['SparseCoding']['lr'])
    l1_weight = float(config['SparseCoding']['l1_weight'])

elif isinstance(model, SparseAutoEncoder):
    lr = float(config['SparseAutoencoder']['lr'])
    l1_weight = float(config['SparseAutoencoder']['l1_weight'])

elif isinstance(model, GatedSAE):
    lr = float(config['GatedSAE']['lr'])
    l1_weight = float(config['GatedSAE']['l1_weight'])

elif isinstance(model, TopKSAE):
    lr = float(config['TopKSAE']['lr'])
    l1_weight = float(config['TopKSAE']['l1_weight'])

else:
    raise ValueError("Model not recognised")

# Print out the lr and l1_weight
print("lr: ", lr)
print("l1_weight: ", l1_weight)



S_ = train(model)
# analyze(S, S_)

# Example 2: Gated SAE
# gated_sae = GatedAutoEncoder(D, learn_D=False, seed=seed).to(device)
# S_ = train(gated_sae)
# analyze(S, S_)