import torch
import torch.nn as nn
from utils import reconstruction_loss_with_l1

class SparseAutoEncoder(nn.Module):
    def __init__(self, M, N, D, learn_D=False, seed=20240625):
        super().__init__()
        self.learn_D = learn_D
        if self.learn_D:
            # Assert that D is not None
            assert D is not None, "D must be provided if learn_D is True"
        torch.manual_seed(seed + 42)
        self.encoder = nn.Sequential(nn.Linear(M, N), nn.ReLU())
        self.decoder = nn.Linear(N, M, bias=False)
        if learn_D:
            self.decoder.weight = nn.Parameter(torch.randn(M, N), requires_grad=True)
        else:
            self.decoder.weight = nn.Parameter(D.T, requires_grad=False)

    def forward(self, X):
        if self.learn_D:
            self.decoder.weight.data /= torch.linalg.norm(self.decoder.weight.data, dim=1, keepdim=True)
        S_ = self.encoder(X)
        X_ = self.decoder(S_)
        return S_, X_


class MLP(nn.Module):
    def __init__(self, M, N, h, D, learn_D=True, seed=20240625):
        super().__init__()
        self.learn_D = learn_D
        if self.learn_D:
            # Assert that D is not None
            assert D is not None, "D must be provided if learn_D is True"
        torch.manual_seed(seed + 42)
        self.encoder = nn.Sequential(nn.Linear(M, h), nn.ReLU(), nn.Linear(h, N), nn.ReLU())
        self.decoder = nn.Linear(N, M)
        if learn_D:
            self.decoder.weight = nn.Parameter(torch.randn(M, N), requires_grad=True)
        else:
            self.decoder.weight = nn.Parameter(D.T.clone(), requires_grad=False)

    def forward(self, X):
        if self.learn_D:
            self.decoder.weight.data /= torch.linalg.norm(self.decoder.weight.data, dim=1, keepdim=True)
        S_ = self.encoder(X)
        X_ = self.decoder(S_)
        return S_, X_


class SparseCoding(nn.Module):
    def __init__(self, X, D, learn_D, seed=20240625):
        super().__init__()
        self.learn_D = learn_D
        torch.manual_seed(seed + 42)
        if learn_D:
            self.D = nn.Parameter(data=torch.randn(D.shape), requires_grad=True)
        else:
            self.D = nn.Parameter(data=D, requires_grad=False)
        self.log_S = nn.Parameter(data=-10 * torch.ones(X.shape[0], D.shape[0]), requires_grad=True)

    def forward(self, X):
        if self.learn_D:
            self.D.data /= torch.linalg.norm(self.D.data, dim=0, keepdim=True)
        S = torch.exp(self.log_S)
        X_ = S @ self.D
        return S, X_

    def optimize_codes(self, X, num_iterations=1000, lr=1e-3):
        log_S_ = nn.Parameter(data=-10 * torch.ones(X.shape[0], self.D.shape[0]), requires_grad=True)
        opt = torch.optim.Adam([log_S_], lr=lr)

        for j in range(num_iterations):
            S = torch.exp(log_S_)
            X_ = S @ self.D
            loss = reconstruction_loss_with_l1(X, X_, S)
            opt.zero_grad()
            loss.backward()
            opt.step()

        return torch.exp(log_S_.detach())