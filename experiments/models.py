import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import reconstruction_loss_with_l1

# class SparseAutoEncoder(nn.Module):
#     def __init__(self, M, N, D, learn_D=False, seed=20240625):
#         super().__init__()
#         self.learn_D = learn_D
#         if self.learn_D:
#             # Assert that D is not None
#             assert D is not None, "D must be provided if learn_D is True"
#         torch.manual_seed(seed + 42)
#         self.encoder = nn.Sequential(nn.Linear(M, N), nn.ReLU())
#         self.decoder = nn.Linear(N, M, bias=False)
#         if learn_D:
#             self.decoder.weight = nn.Parameter(torch.randn(M, N), requires_grad=True)
#         else:
#             self.decoder.weight = nn.Parameter(D.T, requires_grad=False)

#     def forward(self, X):
#         if self.learn_D:
#             self.decoder.weight.data /= torch.linalg.norm(self.decoder.weight.data, dim=1, keepdim=True)
#         S_ = self.encoder(X)
#         X_ = self.decoder(S_)
#         return S_, X_


# class MLP(nn.Module):
#     def __init__(self, M, N, h, D, learn_D=True, seed=20240625):
#         super().__init__()
#         self.learn_D = learn_D
#         if self.learn_D:
#             # Assert that D is not None
#             assert D is not None, "D must be provided if learn_D is True"
#         torch.manual_seed(seed + 42)
#         self.encoder = nn.Sequential(nn.Linear(M, h), nn.ReLU(), nn.Linear(h, N), nn.ReLU())
#         self.decoder = nn.Linear(N, M)
#         if learn_D:
#             self.decoder.weight = nn.Parameter(torch.randn(M, N), requires_grad=True)
#         else:
#             self.decoder.weight = nn.Parameter(D.T.clone(), requires_grad=False)

#     def forward(self, X):
#         if self.learn_D:
#             self.decoder.weight.data /= torch.linalg.norm(self.decoder.weight.data, dim=1, keepdim=True)
#         S_ = self.encoder(X)
#         X_ = self.decoder(S_)
#         return S_, X_

class SparseAutoEncoder(nn.Module):
    def __init__(self, M, N, D, learn_D=False, seed=20240625):
        super().__init__()
        self.learn_D = learn_D
        if self.learn_D:
            assert D is not None, "D must be provided if learn_D is True"
        torch.manual_seed(seed + 42)
        self.encoder = nn.Sequential(nn.Linear(M, N), nn.ReLU())
        self.decoder = nn.Linear(N, M)
        #print(f"SAE decoder shape = {self.decoder.weight.shape}")   
        if learn_D:
            self.decoder.weight = nn.Parameter(torch.randn(M, N), requires_grad=True)
            #print(f"SAE decoder shape = {self.decoder.weight.shape}")
        else:
            self.decoder.weight = nn.Parameter(D, requires_grad=False)

    def forward(self, X, norm_D = True):
        if self.learn_D and norm_D:
            self.decoder.weight.data /= torch.linalg.norm(self.decoder.weight.data, dim=0, keepdim=True)
            # Print norms of the columns and rows of D
            # print(f"D norms for columns = {torch.linalg.norm(self.decoder.weight.data, dim=0)}")
            # print(f"D norms for rows = {torch.linalg.norm(self.decoder.weight.data, dim=1)}")
        S_ = self.encoder(X)
        X_ = S_ @ self.decoder.weight.T
        return S_, X_

class MLP(nn.Module):
    def __init__(self, M, N, h, D, learn_D=True, seed=20240625):
        super().__init__()
        self.learn_D = learn_D
        if self.learn_D:
            assert D is not None, "D must be provided if learn_D is True"
        torch.manual_seed(seed + 42)
        self.encoder = nn.Sequential(nn.Linear(M, h), nn.ReLU(), nn.Linear(h, N), nn.ReLU())
        self.decoder = nn.Linear(N, M)
        #print(f"MLP decoder shape = {self.decoder.weight.shape}")
        if learn_D:
            self.decoder.weight = nn.Parameter(torch.randn(M, N), requires_grad=True)
            #print(f"MLP decoder shape = {self.decoder.weight.shape}")
        else:
            self.decoder.weight = nn.Parameter(D.clone(), requires_grad=False)

    def forward(self, X, norm_D = True):
        if self.learn_D and norm_D:
            self.decoder.weight.data /= torch.linalg.norm(self.decoder.weight.data, dim=0, keepdim=True)
            # Print norms of the columns and rows of D
            # print(f"D norms for columns = {torch.linalg.norm(self.decoder.weight.data, dim=0)}")
            # print(f"D norms for rows = {torch.linalg.norm(self.decoder.weight.data, dim=1)}")
        S_ = self.encoder(X)
        X_ = S_ @ self.decoder.weight.T
        return S_, X_


class SparseCoding(nn.Module):
    def __init__(self, X, D, learn_D, seed=20240625, relu_activation = False):
        super().__init__()
        self.learn_D = learn_D
        self.relu_activation = relu_activation
        torch.manual_seed(seed + 42)
        if learn_D:
            self.D = nn.Parameter(data=torch.randn(D.shape), requires_grad=True)
        else:
            self.D = nn.Parameter(data=D, requires_grad=False)
        if not self.relu_activation:
            self.log_S = nn.Parameter(data=-10 * torch.ones(X.shape[0], D.shape[1]), requires_grad=True)
        else:
            self.log_S = nn.Parameter(data=torch.randn(X.shape[0], D.shape[1]), requires_grad=True)

    def forward(self, X, norm_D = True):
        if self.learn_D and norm_D:
            self.D.data /= torch.linalg.norm(self.D.data, dim=0, keepdim=True)
        if self.relu_activation:
            S = F.relu(self.log_S)
        else:
            S = torch.exp(self.log_S)
        X_ = S @ self.D.T
        return S, X_

    def optimize_codes(self, X, num_iterations=1000, lr=3e-3, l1_weight=0.01):
        log_S_ = nn.Parameter(data=-10 * torch.ones(X.shape[0], self.D.shape[1]), requires_grad=True)
        opt = torch.optim.Adam([log_S_], lr=lr)

        for j in range(num_iterations):
            S = torch.exp(log_S_) if not self.relu_activation else F.relu(log_S_)
            X_ = S @ self.D.T
            loss = reconstruction_loss_with_l1(X, X_, S, l1_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()

        return torch.exp(log_S_.detach())