import torch
from torch.nn import functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import einops
import argparse
import os
from transformer_lens import HookedTransformer
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader  
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from models import SparseAutoencoder

def get_device():
    """Determine the available device (GPU if available, else CPU)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_activation_store(config):
    """
    Initialize the activation store using the session loader.

    Args:
        config (dict): Configuration dictionary for the session loader.

    Returns:
        Tuple: (transformer_model, activation_store)
    """
    # Load the transformer model and activation store
    hook_point = "blocks.8.hook_resid_pre"  # Placeholder hook point
    saes, _ = get_gpt2_res_jb_saes(hook_point)
    sparse_autoencoder = saes[hook_point]
    sparse_autoencoder.to(config['device'])
    sparse_autoencoder.cfg.device = config['device']
    sparse_autoencoder.cfg.hook_point = f"blocks.{config['layer']}.attn.hook_z"
    sparse_autoencoder.cfg.store_batch_size = config['batch_size']

    loader = LMSparseAutoencoderSessionloader(sparse_autoencoder.cfg)
    transformer_model, _, activation_store = loader.load_sae_training_group_session()
    return transformer_model.to(config['device']), activation_store

def mse(output, target):
    """
    Compute the Mean Squared Error between output and target.

    Args:
        output (torch.Tensor): Reconstructed activations.
        target (torch.Tensor): Original activations.

    Returns:
        torch.Tensor: MSE value.
    """
    return F.mse_loss(output, target)

def normalized_mse(recon, xs):
    """
    Compute the Normalized Mean Squared Error.

    Normalizes the reconstruction loss by the MSE of the target's mean.

    Args:
        recon (torch.Tensor): Reconstructed activations.
        xs (torch.Tensor): Original activations.

    Returns:
        torch.Tensor: Normalized MSE value.
    """
    # Compute MSE between recon and xs
    mse_recon = mse(recon, xs)
    
    # Compute MSE between the mean of xs and xs
    mean_xs = xs.mean(dim=0, keepdim=True).expand_as(xs)
    mse_mean = mse(mean_xs, xs)
    
    # To avoid division by zero, add a small epsilon to mse_mean
    epsilon = 1e-8
    return mse_recon / (mse_mean + epsilon)

def loss_fn(X, X_, S_, l1_weight=0.01):
    """
    Compute the combined normalized reconstruction and sparsity loss.

    Args:
        X (torch.Tensor): Original activations.
        X_ (torch.Tensor): Reconstructed activations.
        S_ (torch.Tensor): Sparse activations.
        l1_weight (float): Weight for the L1 loss component.

    Returns:
        torch.Tensor: Combined loss value.
    """
    # Use Normalised MSE for reconstruction loss
    recon_loss = normalized_mse(X_, X)
    
    # Compute L1 and L0 sparsity losses
    l1_loss = S_.norm(p=1, dim=-1).mean()
    l0_loss = S_.norm(p=0, dim=-1).mean()
    
    print(f"Reconstruction loss: {recon_loss.item():.6f}, "
          f"L1 loss: {l1_loss.item():.6f}, L0 loss: {l0_loss.item():.6f}")
    
    # Combine losses
    return recon_loss + l1_weight * l1_loss

def train(model, transformer, activation_store, optimizer, device, n_batches, l1_weight, layer):
    """
    Train the sparse autoencoder using activations from the activation store.

    Args:
        model (nn.Module): The sparse autoencoder model.
        transformer (HookedTransformer): The transformer model to obtain activations.
        activation_store: Activation store object to fetch activation batches.
        optimizer (optim.Optimizer): Optimizer.
        device (str): Device to train on.
        n_batches (int): Number of training batches.
        l1_weight (float): Weight for the L1 loss component.
        layer (int): Layer number to extract activations from.

    Returns:
        None
    """
    model.train()
    for batch_num in range(1, n_batches + 1):
        # Fetch a batch of tokens from the activation store
        batch_tokens = activation_store.get_batch_tokens().to(device)
        
        # Obtain activations from the transformer model
        with torch.no_grad():
            _, cache = transformer.run_with_cache(batch_tokens)
            X = cache["resid_pre", layer]  # Shape: (batch, pos, d_model)
            X = einops.rearrange(X, "batch pos d_model -> (batch pos) d_model")
            del cache

        optimizer.zero_grad()
        S_, X_ = model(X)
        loss = loss_fn(X, X_, S_, l1_weight=l1_weight)
        loss.backward()
        optimizer.step()
        
        print(f"Batch [{batch_num}/{n_batches}], Loss: {loss.item():.6f}")
    
    print("Training complete.")

def main(args):
    """
    Main function to set up and train the sparse autoencoder.

    Args:
        args: Parsed command-line arguments.

    Returns:
        None
    """
    device = get_device()
    print(f"Using device: {device}")
    
    # Configuration for the activation store
    config = {
        'device': device,
        'hook_point': f"blocks.{args.layer}.attn.hook_z",
        'batch_size': args.batch_size,
        'layer': args.layer
    }
    
    # Initialize the transformer model and activation store
    transformer, activation_store = initialize_activation_store(config)
    
    # Initialize the Sparse Autoencoder
    model = SparseAutoencoder(
        input_dim=args.input_dim,
        hidden_dim=args.projection_up * args.input_dim,
    ).to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train the model
    train(
        model=model,
        transformer=transformer,
        activation_store=activation_store,
        optimizer=optimizer,
        device=device,
        n_batches=args.n_batches,
        l1_weight=args.l1_weight,
        layer=args.layer
    )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Save the trained model
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Sparse Autoencoder on Transformer Activations with Normalized MSE"
    )
    parser.add_argument(
        '--layer',
        type=int,
        required=True,
        help='Layer number to extract activations from'
    )
    parser.add_argument(
        '--input_dim',
        type=int,
        required=True,
        help='Dimensionality of the input activations'
    )
    parser.add_argument(
        '--projection_up',
        type=int,
        default=22,
        help='Dimensionality of the hidden layer'
    )
    parser.add_argument(
        '--l1_weight',
        type=float,
        default=0.01,
        help='L1 coefficient for the loss function'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--n_batches',
        type=int,
        default=5000,
        help='Number of training batches'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for training'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='models/sparse_autoencoder.pth',
        help='Path to save the trained model'
    )
    
    args = parser.parse_args()
    main(args)