import torch
import numpy as np
from torchprofile import profile_macs
import sys
import os
import warnings
warnings.filterwarnings("ignore")

from models import SparseCoding, SparseAutoEncoder, GatedSAE, TopKSAE

def generate_sample_data(N, M, K, num_samples=1):
    S = np.random.normal(0, 1, (num_samples, N))
    S = np.abs(S)
    for i in range(num_samples):
        ind = np.random.choice(N, K, replace=False)
        mask = np.zeros(N)
        mask[ind] = 1
        S[i] *= mask
    D = np.random.randn(N, M)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    X = S @ D
    return torch.tensor(S, dtype=torch.float32), torch.tensor(X, dtype=torch.float32), torch.tensor(D, dtype=torch.float32)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, weight):
        super().__init__()
        self.model = model
        self.weight = weight

    def forward(self, X):
        return self.model.loss_forward(X, self.weight)
        #return self.model.forward(X)

def measure_flops(model_wrapper, input_data):
    macs = profile_macs(model_wrapper, input_data)
    return macs * 2  # Approximate FLOPs as 2 * MACs

if __name__ == "__main__":
    # Set parameters
    N, M, K = 16, 8, 3
    num_samples = 1
    seed = 20240625
    weight = 3e-4  # Sparsity weight for loss calculation

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate sample data
    S, X, D = generate_sample_data(N, M, K, num_samples)

    # Initialize models
    models = [
        SparseCoding(S, D, learn_D=False),
        SparseAutoEncoder(D, learn_D=False),
        GatedSAE(D, learn_D=False),
        TopKSAE(D, learn_D=False, k=K)
    ]

    print(f"Measuring FLOPs for {num_samples} sample(s) with dimensions: N={N}, M={M}, K={K} (input shape: {X.shape})..., D shape: {D.shape}")
    
    # Measure FLOPs for each model
    for model in models:
        model.eval()  # Set model to evaluation mode
        model_wrapper = ModelWrapper(model, weight)
        with torch.no_grad():
            flops = measure_flops(model_wrapper, X)
        print(f"{model.__class__.__name__}: {flops:,} FLOPs")

print("FLOP counting completed.")

# import torch
# import numpy as np
# from torchprofile import profile_macs
# import sys
# import os
# import warnings
# warnings.filterwarnings("ignore")

# # Add the parent directory to the Python path to import our custom modules
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

# from models import SparseCoding, SparseAutoEncoder, GatedSAE, TopKSAE

# def generate_sample_data(N, M, K, num_samples=1):
#     S = np.random.normal(0, 1, (num_samples, N))
#     S = np.abs(S)
#     for i in range(num_samples):
#         ind = np.random.choice(N, K, replace=False)
#         mask = np.zeros(N)
#         mask[ind] = 1
#         S[i] *= mask
#     D = np.random.randn(N, M)
#     D /= np.linalg.norm(D, axis=1, keepdims=True)
#     X = S @ D
#     return torch.tensor(S, dtype=torch.float32), torch.tensor(X, dtype=torch.float32), torch.tensor(D, dtype=torch.float32)

# class ModelWrapper(torch.nn.Module):
#     def __init__(self, model, weight):
#         super().__init__()
#         self.model = model
#         self.weight = weight

#     def forward(self, X):
#         return self.model.loss_forward(X, self.weight)

# class SparseCodingWrapper(torch.nn.Module):
#     def __init__(self, model, l1_weight):
#         super().__init__()
#         self.model = model
#         self.l1_weight = l1_weight

#     def forward(self, X):
#         S_ = torch.randn_like(self.model.D_.t())
#         X_ = S_ @ self.model.D_
#         loss = torch.sum((X - X_) ** 2) + self.l1_weight * torch.sum(torch.abs(S_))
#         return loss

# def measure_flops(model_wrapper, input_data):
#     macs = profile_macs(model_wrapper, input_data)
#     return macs * 2  # Approximate FLOPs as 2 * MACs

# def calculate_flops(model_type, N, M, K, l1_weight=3e-4, num_iterations=100, seed=20240625):
#     # Set random seed for reproducibility
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     # Generate sample data
#     _, X, D = generate_sample_data(N, M, K, num_samples=1)

#     # Initialize model
#     if model_type == "SparseCoding":
#         S_ = torch.randn_like(D.t())
#         model = SparseCoding(S_, D, learn_D=False)
#         model_wrapper = ModelWrapper(model, l1_weight) #SparseCodingWrapper(model, l1_weight)
#     elif model_type == "SparseAutoEncoder":
#         model = SparseAutoEncoder(D, learn_D=False)
#         model_wrapper = ModelWrapper(model, l1_weight)
#     elif model_type == "GatedSAE":
#         model = GatedSAE(D, learn_D=False)
#         model_wrapper = ModelWrapper(model, l1_weight)
#     elif model_type == "TopKSAE":
#         model = TopKSAE(D, learn_D=False, k=K)
#         model_wrapper = ModelWrapper(model, l1_weight)
#     else:
#         raise ValueError("Invalid model type")

#     model_wrapper.eval()  # Set model to evaluation mode

#     with torch.no_grad():
#         flops = measure_flops(model_wrapper, X)

#     # For SparseCoding, multiply by the number of iterations
#     if model_type == "SparseCoding":
#         flops *= num_iterations

#     return flops

# if __name__ == "__main__":
#     # Example usage
#     N, M, K = 16, 8, 3
#     l1_weight = 3e-4
#     num_iterations = 1

#     model_types = ["SparseCoding", "SparseAutoEncoder", "GatedSAE", "TopKSAE"]

#     for model_type in model_types:
#         flops = calculate_flops(model_type, N, M, K, l1_weight, num_iterations)
#         print(f"{model_type}: {flops:,} FLOPs")

#     print("FLOP counting completed.")