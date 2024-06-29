import torch
import numpy as np
from torchprofile import profile_macs
import sys
import os

# Add the parent directory to the Python path to import our custom modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

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