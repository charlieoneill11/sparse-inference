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

# parameters
N = 16  # number of sparse sources
K = 3  # number of active components
M = 8  # number of measurements
seed = 20240625
num_data = 1024
lr = 3e-3
num_step = 20000
weight = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Need to write a superclass of which each model is a subclass
# Needs to contain forward and loss

class SparseCoding(nn.Module):
    def __init__(self, S, D, learn_D, seed: int = 42):
        super().__init__()
        self.learn_D = learn_D
        torch.manual_seed(seed + 42)
        self.log_S_ = nn.Parameter(data=-10 * torch.ones(S.shape), requires_grad=True)
        if learn_D:
            self.D_ = nn.Parameter(data=torch.randn(D.shape), requires_grad=True)
        else:
            self.D_ = nn.Parameter(data=D, requires_grad=False)

    def forward(self, X = None):
        if self.learn_D:
            self.D_.data /= torch.linalg.norm(self.D_, dim=1, keepdim=True)
        S_ = torch.exp(self.log_S_)
        X_ = S_ @ self.D_
        return S_, X_
    
    def loss_forward(self, X, weight=weight):
        S_, X_ = self.forward(X)
        loss = torch.sum((X - X_) ** 2) + weight * torch.sum(torch.abs(S_))
        return S_, X_, loss
    
class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return torch.exp(X)
    
class SparseAutoEncoder(nn.Module):
    def __init__(self, D, learn_D, seed: int = 42, relu=True):
        super().__init__()
        self.learn_D = learn_D
        torch.manual_seed(seed + 42)
        if relu:
            self.encoder = nn.Sequential(nn.Linear(M, N), nn.ReLU())
        else:
            self.encoder = nn.Sequential(nn.Linear(M, N), Exp())
        if learn_D:
            self.D_ = nn.Parameter(data=torch.randn(D.shape), requires_grad=True)
        else:
            self.D_ = nn.Parameter(data=D, requires_grad=False)

    def forward(self, X):
        if self.learn_D:
            self.D_.data /= torch.linalg.norm(self.D_, dim=1, keepdim=True)
        S_ = self.encoder(X)
        X_ = S_ @ self.D_
        return S_, X_
    
    def loss_forward(self, X, weight=weight):
        S_, X_ = self.forward(X)
        loss = torch.sum((X - X_) ** 2) + weight * torch.sum(torch.abs(S_))
        return S_, X_, loss
    
    
class GatedAutoEncoder(nn.Module):
    def __init__(self, D, learn_D, seed: int = 42):
        super().__init__()
        self.learn_D = learn_D
        torch.manual_seed(seed + 42)

        self.W_gate = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(M, N)
            ), requires_grad=True
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(N, M)
            ), requires_grad=True
        )

        self.b_dec = nn.Parameter(torch.zeros(M))

        self.b_enc_gate = nn.Parameter(torch.zeros(N))
        self.b_dec_gate = nn.Parameter(torch.zeros(M))

        self.r_mag = nn.Parameter(torch.zeros(N))
        self.b_mag = nn.Parameter(torch.zeros(N))

        self.b_gate = nn.Parameter(torch.zeros(N))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = N

    def forward(self, X):
        preactivations_hidden = einops.einsum(X - self.b_dec, self.W_gate, "... input_dim, input_dim hidden_dim -> ... hidden_dim")

        #print("Preactivations hidden: ", preactivations_hidden)

        pre_mag_hidden = preactivations_hidden * torch.exp(self.r_mag) + self.b_mag
        post_mag_hidden = torch.relu(pre_mag_hidden)
        #print("Post mag hidden: ", post_mag_hidden)

        pre_gate_hidden = preactivations_hidden + self.b_gate
        post_gate_hidden = (pre_gate_hidden > 0).float()
        #print("Post gate hidden: ", post_gate_hidden)

        S_ = post_mag_hidden * post_gate_hidden
        #print("S_: ", S_)
        # Print L0 norm of S_
        #print("L0 norm of S_: ", torch.sum(S_ != 0))

        X_ =  einops.einsum(S_, self.W_dec, "... hidden_dim, hidden_dim output_dim -> ... output_dim") + self.b_dec
        #print("X_: ", X_)

        #return S_, X_, pre_gate_hidden
        return pre_gate_hidden, X_, pre_gate_hidden

    
    def loss_forward(self, X, weight=weight):
        S_, X_, pre_gate_hidden = self.forward(X)

        # Reconstruction Loss
        gated_sae_loss = F.mse_loss(X_, X, reduction='mean')

        # L1 loss
        gate_magnitude = F.relu(pre_gate_hidden)
        gated_sae_loss += weight * gate_magnitude.sum()

        # Auxiliary loss
        gate_reconstruction = einops.einsum(gate_magnitude, self.W_dec.detach(), "... hidden_dim, hidden_dim output_dim -> ... output_dim") + self.b_dec.detach()
        auxiliary_loss = F.mse_loss(gate_reconstruction, X, reduction='mean')

        gated_sae_loss += auxiliary_loss

        return S_, X_, gated_sae_loss

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj


def criterion(S_, X, X_, weight=weight):
    loss = torch.sum((X - X_) ** 2) + weight * torch.sum(torch.abs(S_))
    return loss

def train(model):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(num_step):
        if isinstance(model, GatedAutoEncoder):
            S_, X_, loss = model.loss_forward(X, weight=weight)
        else:
            S_, X_ = model.forward(X)
            loss = criterion(S_, X, X_, weight=weight)
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

# def train(model):
#     optim = torch.optim.Adam(model.parameters(), lr=lr)
#     losses, mccs = [], []
#     for i in tqdm(range(num_step)):
#         # S_, X_, _ = model.forward(X)
#         # loss = criterion(S_, X, X_, weight=weight)
#         S_, X_, loss = model.loss_forward(X, weight=weight)
#         optim.zero_grad()
#         loss.backward()
#         # If model is of type GatedSAE, we need to make sure the decoder weights are unit norm
#         if isinstance(model, GatedAutoEncoder):
#             model.make_decoder_weights_and_grad_unit_norm()
#         optim.step()
#         # New loss - keeps them equal (for gated SAE, which has a different loss above)
#         loss = criterion(S_, X, X_, weight=weight).detach()
#         losses.append(loss.item())
#         mcc_val = mcc(S, S_.detach().cpu().numpy())
#         mccs.append(mcc_val)
#         if i > 0 and not i % 100:
#             if not torch.all(S_.var(0) > 0):
#                 print('dead latents')
#             print('step', i, 'loss', loss.item(),
#                   'MCC', mcc_val)
            
#     S_ = S_.detach().cpu().numpy()
#     print('final MCC', mcc(S, S_))
#     return S_
    

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

# Example 1: Sparse Coding  
# sc0 = SparseCoding(S, D, learn_D=False, seed=seed).to(device)
# S_ = train(sc0)
# analyze(S, S_)

# Example 2: Gated SAE
gated_sae = GatedAutoEncoder(D, learn_D=False, seed=seed).to(device)
S_ = train(gated_sae)
# analyze(S, S_)
