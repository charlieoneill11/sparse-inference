# import torch
# import numpy as np
# from torchprofile import profile_macs
# import sys
# import os
# import warnings
# warnings.filterwarnings("ignore")

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
#         #return self.model.forward(X)

# def measure_flops(model_wrapper, input_data):
#     macs = profile_macs(model_wrapper, input_data)
#     return macs * 2  # Approximate FLOPs as 2 * MACs

# if __name__ == "__main__":
#     # Set parameters
#     N, M, K = 16, 8, 3
#     num_samples = 1
#     seed = 20240625
#     weight = 3e-4  # Sparsity weight for loss calculation

#     # Set random seed for reproducibility
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     # Generate sample data
#     S, X, D = generate_sample_data(N, M, K, num_samples)

#     # Initialize models
#     models = [
#         SparseCoding(S, D, learn_D=False),
#         SparseAutoEncoder(D, learn_D=False),
#         GatedSAE(D, learn_D=False),
#         TopKSAE(D, learn_D=False, k=K)
#     ]

#     print(f"Measuring FLOPs for {num_samples} sample(s) with dimensions: N={N}, M={M}, K={K} (input shape: {X.shape})..., D shape: {D.shape}")
    
#     # Measure FLOPs for each model
#     for model in models:
#         model.eval()  # Set model to evaluation mode
#         model_wrapper = ModelWrapper(model, weight)
#         with torch.no_grad():
#             flops = measure_flops(model_wrapper, X)
#         print(f"{model.__class__.__name__}: {flops:,} FLOPs")

# print("FLOP counting completed.")

import yaml

def calculate_flops(model_name, N, M, K, num_iterations=1):
    flops = 0

    if model_name == 'SparseCoding':
        sparse_coding_base = 2*N*M + 3*M + 2*N + M - 1
        sparse_coding_iteration = 2 * sparse_coding_base  # Accounting for forward and backward pass
        flops = sparse_coding_iteration * num_iterations
    elif model_name == 'SparseAutoEncoder':
        flops = M + 2*N*M + N + 2*N*M + M
    elif model_name == 'GatedSAE':
        flops = (M + 2*N*M + 2*N + N + N + N + 3*N + N + 2*N*M + M)
    elif model_name == 'TopKSAE':
        topk_flops = estimate_topk_flops(N, K)
        flops = M + 2*N*M + N + topk_flops + 2*N*M + M
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return flops

def estimate_topk_flops(N, K):
    # Estimate FLOPs for partial selection algorithm
    comparison_flops = min(2 * N, N + K * (2 + K))  # Caps at N + K*(2+K) for very large K
    
    # ReLU on top K values
    relu_flops = K
    
    # Zeroing out non-top-k
    zero_flops = N - K
    
    return comparison_flops + relu_flops + zero_flops


def generate_yaml(N, M, K, num_iterations):
    flops = calculate_flops(N, M, K, num_iterations)

    yaml_data = {
        'flop_counts': {
            'SparseCoding': {
                'total_flops': flops['SparseCoding'],
                'note': f"Total FLOPs for {num_iterations} optimization iterations"
            },
            'SparseAutoEncoder': {
                'total_flops': flops['SparseAutoEncoder']
            },
            'GatedSAE': {
                'total_flops': flops['GatedSAE']
            },
            'TopKSAE': {
                'total_flops': flops['TopKSAE']
            }
        },
        'model_parameters': {
            'N': N,
            'M': M,
            'K': K,
            'num_iterations': num_iterations
        },
        'notes': [
            "FLOPs for SparseCoding represent multiple optimization iterations.",
            "FLOPs for other models represent a single forward pass during inference."
        ]
    }

    return yaml_data

if __name__ == "__main__":
    # Example usage
    N, M, K = 16, 8, 3
    num_iterations = 100

    yaml_data = generate_yaml(N, M, K, num_iterations)

    print("FLOP counts:")
    for model, flops in yaml_data['flop_counts'].items():
        print(f"{model}: {flops['total_flops']:,} FLOPs")
    
    # Print TopK FLOPs separately for verification
    print(f"\nTopK operation FLOPs: {estimate_topk_flops(N, K)}")

# Next, we need to implement it for training as well
