import torch
from torch.nn import functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import einops
import argparse
import os
import math
from transformer_lens import HookedTransformer
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes

# We keep the import of SparseAutoencoder (though not used) to minimize code changes
from models import SparseAutoencoder
from huggingface_hub import HfApi
import sys
import time
import yaml

import warnings
warnings.filterwarnings("ignore")


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
    """
    return F.mse_loss(output, target)


def normalized_mse(recon, xs):
    """
    Compute the Normalized Mean Squared Error.

    Normalizes the reconstruction loss by the MSE of the target's mean.
    """
    mse_recon = mse(recon, xs)
    mean_xs = xs.mean(dim=0, keepdim=True).expand_as(xs)
    mse_mean = mse(mean_xs, xs)
    epsilon = 1e-8
    return mse_recon / (mse_mean + epsilon)


def loss_fn(X, X_, S_, l1_weight=0.01):
    """
    Compute the combined normalized reconstruction and sparsity loss.
    (Retained from the sparse autoencoder version to minimize changes.)
    """
    recon_loss = normalized_mse(X_, X)
    l1_loss = S_.norm(p=1, dim=-1).mean()
    l0_loss = S_.norm(p=0, dim=-1).mean()
    
    total_loss = recon_loss + l1_weight * l1_loss
    return recon_loss, l1_loss, l0_loss, total_loss


def upload_to_huggingface(model_path, repo_name, hf_token, commit_message):
    """
    Upload the model to Hugging Face Hub.
    """
    api = HfApi()
    
    # Check if the repository exists; if not, create it
    try:
        api.repo_info(repo_id=repo_name)
        print(f"Repository '{repo_name}' already exists on Hugging Face.")
    except Exception as e:
        print(f"Creating repository '{repo_name}' on Hugging Face.")
        try:
            api.create_repo(repo_id=repo_name, private=False)
            print(f"Repository '{repo_name}' created successfully.")
        except Exception as e:
            print(f"Failed to create repository '{repo_name}'. Error: {e}")
            sys.exit(1)
    
    # Define the path within the repository
    filename = os.path.basename(model_path)
    path_in_repo = filename
    
    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=path_in_repo,
            repo_id=repo_name,
            token=hf_token,
            commit_message=commit_message
        )
        print(f"Model '{filename}' uploaded to Hugging Face "
              f"repo '{repo_name}' with commit message '{commit_message}'.")
    except Exception as e:
        print(f"Failed to upload file to Hugging Face. Error: {e}")
        sys.exit(1)


def resample_dead_neurons(
    model, 
    optimizer, 
    dead_neurons: torch.Tensor
):
    """
    Resample dead neurons by reinitializing their encoder and decoder rows/columns.
    Kept as-is from the sparse autoencoder script for minimal changes,
    though it might not match the new MLP structure.
    """
    if not dead_neurons.any():
        print("No dead neurons to resample.")
        return

    with torch.no_grad():
        # These lines expect a single encoder/decoder pair
        # from the old SparseAutoencoder. We keep them anyway:
        encoder_weight = model.encoder[0].weight.data  # [hidden_dim, input_dim]
        decoder_weight = model.decoder.weight.data     # [input_dim, hidden_dim]
        hidden_dim, input_dim = encoder_weight.shape

        num_resampled = 0
        for idx in torch.where(dead_neurons)[0]:
            # Initialize a random vector for the decoder column
            new_dec_col = torch.randn(input_dim, device=decoder_weight.device)
            new_dec_col /= (new_dec_col.norm() + 1e-12)
            decoder_weight[:, idx] = new_dec_col

            # For the encoder row, pick a smaller norm for minimal interference
            new_enc_row = new_dec_col.clone() * 0.2
            encoder_weight[idx] = new_enc_row

            # Reset encoder bias
            model.encoder[0].bias.data[idx] = 0.0

            # Reset optimizer states for these parameters
            params_to_reset = [
                model.encoder[0].weight,  
                model.decoder.weight,     
                model.encoder[0].bias     
            ]
            for param in params_to_reset:
                if param in optimizer.state:
                    optimizer.state[param] = {}

            num_resampled += 1

    print(f"Resampled {num_resampled} dead neurons.")


def train(
    model, 
    transformer, 
    activation_store, 
    optimizer, 
    device, 
    n_batches, 
    l1_weight, 
    layer, 
    save_path, 
    repo_name, 
    hf_token
):
    """
    Train the MLP model using activations from the activation store.
    Function body unchanged except for references to the newly passed-in model.
    """
    model.train()
    
    # For logging
    recon_loss_acc = 0.0
    l1_loss_acc = 0.0
    l0_loss_acc = 0.0
    total_loss_acc = 0.0

    log_interval = 100
    upload_interval = 2000
    
    hidden_dim = model.encoder[0].weight.shape[0]

    # We'll track neuron firing over some steps
    max_track_steps = 1000
    fired_buffer = torch.zeros(
        (max_track_steps, hidden_dim), dtype=torch.bool, device='cpu'
    )
    step_idx = 0
    
    # Define at which steps to resample
    resample_steps = [25000, 50000, 75000, 100000]

    for batch_num in range(1, n_batches + 1):
        # Fetch tokens
        batch_tokens = activation_store.get_batch_tokens().to(device)
        
        # Get activations
        with torch.no_grad():
            _, cache = transformer.run_with_cache(batch_tokens)
            X = cache["resid_pre", layer]  # (batch, pos, d_model)
            X = einops.rearrange(X, "batch pos d_model -> (batch pos) d_model")
            del cache

        optimizer.zero_grad()
        S_, X_ = model(X)
        recon_loss, l1_loss, l0_loss, total_loss = loss_fn(X, X_, S_, l1_weight=l1_weight)
        total_loss.backward()
        optimizer.step()
        
        recon_loss_acc += recon_loss.item()
        l1_loss_acc += l1_loss.item()
        l0_loss_acc += l0_loss.item()
        total_loss_acc += total_loss.item()

        # Track neuron firing across the batch
        fired = (S_ > 0).any(dim=0).cpu()
        fired_buffer[step_idx] = fired
        step_idx = (step_idx + 1) % max_track_steps

        # Logging
        if batch_num % log_interval == 0:
            avg_recon = recon_loss_acc / log_interval
            avg_l1 = l1_loss_acc / log_interval
            avg_l0 = l0_loss_acc / log_interval
            avg_tot = total_loss_acc / log_interval

            print(
                f"Batch [{batch_num}/{n_batches}], "
                f"AvgRecon: {avg_recon:.6f}, "
                f"AvgL1: {avg_l1:.6f}, "
                f"AvgL0: {avg_l0:.6f}, "
                f"AvgTotal: {avg_tot:.6f}"
            )

            recon_loss_acc = 0.0
            l1_loss_acc = 0.0
            l0_loss_acc = 0.0
            total_loss_acc = 0.0
        
        # Upload model periodically
        if batch_num % upload_interval == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved to {save_path}")
            commit_message = f"Checkpoint at batch {batch_num}"
            upload_to_huggingface(save_path, repo_name, hf_token, commit_message)
        
        # Perform neuron resampling at the specified intervals
        # if batch_num in resample_steps:
        if batch_num % 1000 == 0 and batch_num > 0:
            print(f"*** Resampling dead neurons at step {batch_num} ***")
            sum_of_fires = fired_buffer.sum(dim=0)  # shape [hidden_dim]
            dead_neurons = (sum_of_fires == 0)
            print(f"Number of dead neurons: {dead_neurons.sum().item()}")

            resample_dead_neurons(model, optimizer, dead_neurons)

            # Reset buffer after resampling
            fired_buffer.zero_()
            step_idx = 0

    print("Training complete.")


def upload_final_model(model_path, repo_name, hf_token):
    """
    Upload the final trained model to Hugging Face Hub.
    """
    commit_message = "Final model upload after training completion."
    upload_to_huggingface(model_path, repo_name, hf_token, commit_message)


###############################################################################
# Below is our MLP definition (identical to the snippet you provided),
# kept in the same file for minimal changes.
###############################################################################
class MLP(nn.Module):
    def __init__(self, M, N, h, seed=20240625, use_bias=False):
        super().__init__()
        torch.manual_seed(seed + 42)
        self.encoder = nn.Sequential(
            nn.Linear(M, h, bias=use_bias),
            nn.ReLU(),
            nn.Linear(h, N, bias=use_bias),
            nn.ReLU()
        )
        self.decoder = nn.Linear(N, M, bias=use_bias)
        # Initialize the decoder's weight somewhat randomly
        with torch.no_grad():
            self.decoder.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, X, norm_D=True):
        if norm_D:
            # L2-normalize columns of decoder weight
            w = self.decoder.weight.data
            self.decoder.weight.data = w / (w.norm(dim=0, keepdim=True) + 1e-12)

        S_ = self.encoder(X)
        # Reconstructed input
        X_ = self.decoder(S_)

        return S_, X_


def main(args):
    """
    Main function to set up and train an MLP (instead of the SparseAutoencoder).
    """
    device = get_device()
    print(f"Using device: {device}")

    seq_len = 128
    total_tokens = seq_len * args.batch_size * args.n_batches
    print(f"Training on {total_tokens/1e6:.2f}M total tokens.")

    # Configuration for the activation store
    config = {
        'device': device,
        'hook_point': f"blocks.{args.layer}.attn.hook_z",
        'batch_size': args.batch_size,
        'layer': args.layer
    }
    
    # Initialize the transformer and activation store
    transformer, activation_store = initialize_activation_store(config)
    
    # Instantiate the MLP instead of the SparseAutoencoder
    # We use the same notion that "N = projection_up * input_dim" is the dimension
    # of the final hidden representation, and we'll just pick h = N for simplicity.
    N = args.projection_up * args.input_dim
    h = N  # second hidden dimension (feel free to adjust)
    model = MLP(M=args.input_dim, N=N, h=h).to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Hugging Face token
    config_yml = yaml.safe_load(open("config.yaml"))
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: Hugging Face token not found.")
        sys.exit(1)
    
    repo_name = "charlieoneill/sparse-coding"

    # Train
    train(
        model=model,
        transformer=transformer,
        activation_store=activation_store,
        optimizer=optimizer,
        device=device,
        n_batches=args.n_batches,
        l1_weight=args.l1_weight,
        layer=args.layer,
        save_path=args.save_path,
        repo_name=repo_name,
        hf_token=hf_token
    )
    
    # Save final
    final_model_path = args.save_path
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Upload final to HF
    upload_final_model(final_model_path, repo_name, hf_token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an MLP on Transformer Activations with Normalized MSE"
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
        help='Expansion factor for hidden dimension = projection_up*input_dim'
    )
    parser.add_argument(
        '--l1_weight',
        type=float,
        default=1e-4,
        help='L1 coefficient'
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
        default=50000,
        help='Number of training batches'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )
    # Use mlp_model.pth as the new default save path
    parser.add_argument(
        '--save_path',
        type=str,
        default='models/mlp_model.pth',
        help='Path to save the trained MLP model'
    )
    
    args = parser.parse_args()
    main(args)