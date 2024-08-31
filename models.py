import torch
from torch import nn
from torch.nn import functional as F
import einops
from typing import Callable, Any
from tqdm import tqdm

class SparseCoding(nn.Module):
    def __init__(self, S, D, learn_D, seed: int = 42):
        super().__init__()
        self.learn_D = learn_D
        torch.manual_seed(seed + 42)
        self.log_S_ = nn.Parameter(data=-10 * torch.ones(S.shape[0], D.shape[0]), requires_grad=True)
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

    def infer(self, X, num_iterations=10000, lr=3e-3, l1_weight=1e-3):
        # initialise S_ randomly
        self.log_S_infer = nn.Parameter(data=-10 * torch.ones(X.shape[0], self.D_.shape[0]), requires_grad=True)
        #print(f"Initial log_S_infer: {self.log_S_infer}")

        optimizer = torch.optim.Adam([self.log_S_infer], lr=lr)

        for _ in tqdm(range(num_iterations), desc='Infer'):
            optimizer.zero_grad()
            X_ = torch.exp(self.log_S_infer) @ self.D_
            loss = torch.sum((X - X_) ** 2) + l1_weight * torch.sum(torch.abs(torch.exp(self.log_S_infer)))
            # Print log_S_infer gradient
            #print(f"Log_S_infer grad: {self.log_S_infer.grad}")
            loss.backward()
            optimizer.step()
            #print(f"Infer loss: {loss.item()}")
            # print(f"Log_S_infer: {self.log_S_infer}")  
            # 

        # Set S_ to the inferred value, as a copy (with a gradient) 
        self.log_S_.data = self.log_S_infer.detach()
        self.log_S_.requires_grad = True

        return self.log_S_infer.detach()
    
    def loss_forward(self, X, weight):
        S_, X_ = self.forward(X)
        loss = torch.sum((X - X_) ** 2) + weight * torch.sum(torch.abs(S_))
        return S_, X_, loss

class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return torch.exp(X)

        
    

# class SparseAutoEncoder(nn.Module):
#     def __init__(self, D, learn_D, seed=20240625, relu=True):
#         super().__init__()
#         self.learn_D = learn_D
#         torch.manual_seed(seed + 42)
#         N, M = D.shape
#         if relu:
#             self.encoder = nn.Sequential(nn.Linear(M, N), nn.ReLU())
#         else:
#             self.encoder = nn.Sequential(nn.Linear(M, N), Exp())
#         if learn_D:
#             self.D_ = nn.Parameter(data=torch.randn(D.shape), requires_grad=True)
#             print("D_: ", self.D_[:5, :5])
#         else:
#             self.D_ = nn.Parameter(data=D, requires_grad=False)
#             print("D_: ", self.D_[:5, :5])

#     def forward(self, X):
#         if self.learn_D:
#             #print(f"Before normalization: {torch.norm(self.D_)}")
#             self.D_.data /= torch.linalg.norm(self.D_, dim=1, keepdim=True)
#             #print(f"After normalization: {torch.norm(self.D_)}")
#         S_ = self.encoder(X)
#         X_ = S_ @ self.D_
#         return S_, X_

class SparseAutoEncoder(nn.Module):
    def __init__(self, D, learn_D, seed=20240625, relu=True):
        super().__init__()
        N, M = D.shape
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
    
    def loss_forward(self, X, weight):
        S_, X_ = self.forward(X)
        loss = torch.sum((X - X_) ** 2) + weight * torch.sum(torch.abs(S_))
        return S_, X_, loss

class GatedSAE(nn.Module):
    def __init__(self, D, learn_D, seed: int = 20240625):
        super().__init__()
        self.learn_D = learn_D
        torch.manual_seed(seed + 42)

        N, M = D.shape

        if self.learn_D:
            self.D_ = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(N, M)), requires_grad=True)
        else:
            self.D_ = nn.Parameter(D, requires_grad=False)

        self.W_gate = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(M, N)), requires_grad=True)
        self.b_dec = nn.Parameter(torch.zeros(M))
        self.b_enc_gate = nn.Parameter(torch.zeros(N))
        self.b_dec_gate = nn.Parameter(torch.zeros(M))
        self.r_mag = nn.Parameter(torch.zeros(N))
        self.b_mag = nn.Parameter(torch.zeros(N))
        self.b_gate = nn.Parameter(torch.zeros(N))

        self.D_.data /= torch.linalg.norm(self.D_, dim=1, keepdim=True)

        self.d_hidden = N

    # def forward(self, X):
    #     preactivations_hidden = einops.einsum(X - self.b_dec, self.W_gate, "... input_dim, input_dim hidden_dim -> ... hidden_dim")
    #     pre_mag_hidden = preactivations_hidden * torch.exp(self.r_mag) + self.b_mag
    #     post_mag_hidden = torch.relu(pre_mag_hidden)
    #     pre_gate_hidden = preactivations_hidden + self.b_gate
    #     post_gate_hidden = (torch.sign(pre_gate_hidden) + 1) / 2
    #     S_ = post_mag_hidden * post_gate_hidden
    #     X_ =  einops.einsum(S_, self.W_dec, "... hidden_dim, hidden_dim output_dim -> ... output_dim") + self.b_dec
    #     return S_, X_, pre_gate_hidden

    def forward(self, X):
        if self.learn_D:
            self.D_.data /= torch.linalg.norm(self.D_, dim=1, keepdim=True)

        preactivations_hidden = torch.matmul(X - self.b_dec, self.W_gate)
        pre_mag_hidden = preactivations_hidden * torch.exp(self.r_mag) + self.b_mag
        post_mag_hidden = torch.relu(pre_mag_hidden)
        pre_gate_hidden = preactivations_hidden + self.b_gate
        post_gate_hidden = (torch.sign(pre_gate_hidden) + 1) / 2
        S_ = post_mag_hidden * post_gate_hidden
        X_ = torch.matmul(S_, self.D_) + self.b_dec
        return S_, X_, pre_gate_hidden

    def loss_forward(self, X, l1_weight):
        S_, X_, pre_gate_hidden = self.forward(X)
        gated_sae_loss = F.mse_loss(X_, X, reduction='mean') #(X_ - X).pow(2).mean()
        gate_magnitude = F.relu(pre_gate_hidden)
        gated_sae_loss += l1_weight * gate_magnitude.sum()
        gate_reconstruction = torch.matmul(gate_magnitude, self.D_.detach()) + self.b_dec.detach()
        auxiliary_loss = F.mse_loss(gate_reconstruction, X, reduction='mean')
        gated_sae_loss += auxiliary_loss
        return S_, X_, gated_sae_loss

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        D_normed = self.D_ / self.D_.norm(dim=-1, keepdim=True)
        D_grad_proj = (self.D_.grad * D_normed).sum(-1, keepdim=True) * D_normed
        self.D_.grad -= D_grad_proj

class TopK(nn.Module):
    def __init__(self, k: int, postact_fn: nn.Module = nn.ReLU()):
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

class TopKSAE(nn.Module):
    def __init__(self, D, learn_D, k, seed=20240625, postact_fn=nn.ReLU()):
        super().__init__()
        self.learn_D = learn_D
        torch.manual_seed(seed + 42)
        N, M = D.shape

        if learn_D:
            self.D_ = nn.Parameter(data=torch.randn(D.shape), requires_grad=True)
            self.D_.data /= torch.linalg.norm(self.D_, dim=1, keepdim=True)
        else:
            self.D_ = nn.Parameter(data=D, requires_grad=False)
            self.D_.data /= torch.linalg.norm(self.D_, dim=1, keepdim=True)

        # initialise encoder with transposed D
        self.encoder = nn.Linear(M, N, bias=False)
        with torch.no_grad():
            self.encoder.weight.data = self.D_.clone()

        self.latent_bias = nn.Parameter(torch.zeros(N))
        self.activation = TopK(k=k, postact_fn=postact_fn)
        self.pre_bias = nn.Parameter(torch.zeros(M))

    def forward(self, X):
        if self.learn_D:
            self.D_.data /= torch.linalg.norm(self.D_, dim=1, keepdim=True)

        X_centered = X - self.pre_bias
        S_pre_act = self.encoder(X_centered) + self.latent_bias
        S_ = self.activation(S_pre_act)
        X_ = S_ @ self.D_ + self.pre_bias

        return S_, X_

    def loss_forward(self, X, weight):
        S_, X_ = self.forward(X)
        loss = F.mse_loss(X_, X, reduction='sum')
        return S_, X_, loss

    @property
    def k(self):
        return self.activation.k
    

class TopK(nn.Module):
    def __init__(self, k: int, postact_fn: nn.Module = nn.ReLU()):
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

class GeneralSAETopK(nn.Module):
    def __init__(self, initial_D, projections_up, k, learn_D=True, seed: int = 20240625, postact_fn=nn.ReLU()):
        super().__init__()
        torch.manual_seed(seed + 42)
        N, M = initial_D.shape
        
        self.projections_up = projections_up
        self.learn_D = learn_D
        
        # Create the encoder
        layers = []
        input_dim = M
        for output_dim in projections_up[:-1]:  # All layers except the last one
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        
        # Last layer of the encoder (before top-k)
        layers.append(nn.Linear(input_dim, projections_up[-1]))
        
        self.encoder = nn.Sequential(*layers)
        
        # Top-K activation
        self.topk_activation = TopK(k=k, postact_fn=postact_fn)
        
        # Initialize D
        if learn_D:
            self.D = nn.Parameter(torch.randn(projections_up[-1], M), requires_grad=True)
        else:
            self.D = nn.Parameter(initial_D.T, requires_grad=False)
        
        # Bias terms
        self.latent_bias = nn.Parameter(torch.zeros(projections_up[-1]))
        self.pre_bias = nn.Parameter(torch.zeros(M))
    
    def forward(self, X):
        if self.learn_D:
            self.D.data /= torch.linalg.norm(self.D, dim=0, keepdim=True)
        
        # Encoder
        X_centered = X - self.pre_bias
        S_pre_act = self.encoder(X_centered) + self.latent_bias
        
        # Apply top-k activation
        S = self.topk_activation(S_pre_act)
        
        # Reconstruction
        X_recon = S @ self.D + self.pre_bias
        
        return S, X_recon

    def loss_forward(self, X, l1_weight):
        S, X_recon = self.forward(X)
        reconstruction_loss = F.mse_loss(X_recon, X, reduction='mean')
        sparsity_loss = l1_weight * torch.sum(torch.abs(S))
        total_loss = reconstruction_loss + sparsity_loss
        return S, X_recon, total_loss

    @property
    def k(self):
        return self.topk_activation.k
    

### MLP

class MLP(nn.Module):
    def __init__(self, initial_D, projections_up, resnet=True, learn_D=True, seed: int = 20240625):
        super().__init__()
        torch.manual_seed(seed + 42)
        N, M = initial_D.shape
        print(f"MLP init - N: {N}, M: {M}")
        print(f"Projections: {projections_up}")
        
        self.projections_up = projections_up
        self.resnet = resnet
        self.learn_D = learn_D
        
        # Create the encoder
        layers = []
        input_dim = M
        for output_dim in projections_up:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            input_dim = output_dim
        
        self.encoder = nn.Sequential(*layers)

        if learn_D: 
            self.D_ = nn.Parameter(torch.randn(projections_up[-1], M), requires_grad=True)
        else: 
            self.D_ = nn.Parameter(initial_D.clone(), requires_grad=False)

        print(f"D norm after setting: {torch.norm(self.D_)}")
        print(f"D requires_grad: {self.D_.requires_grad}")
    
    def forward(self, X):
        if self.learn_D:
            #print(f"Before normalization: {torch.norm(self.D_)}")
            self.D_.data /= torch.linalg.norm(self.D_, dim=1, keepdim=True)
            # print(f"After normalization: {torch.norm(self.D_)}")
        
        S = self.encoder(X)
        X_recon = S @ self.D_
        
        return S, X_recon


class ITO(nn.Module):

    def __init__(self, D_):
        super().__init__()
        self.S_ = nn.Parameter(torch.randn(D_.shape[0]), requires_grad=True)
        self.D_ = nn.Parameter(D_, requires_grad=False)

    def forward(self, X):
        return self.S_, X @ self.D_
    
    def optimise(self, X, lr=3e-3, num_steps=10_000):
        optimizer = torch.optim.Adam([self.S_], lr=lr)
        for _ in range(num_steps):
            optimizer.zero_grad()
            S_, X_ = self.forward(X)
            loss = torch.sum((X - X_) ** 2)
            loss.backward()
            optimizer.step()
        return self.S_