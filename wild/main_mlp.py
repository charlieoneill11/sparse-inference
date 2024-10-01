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

import wandb
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
    return F.mse_loss(output, target)

def normalized_mse(recon, xs):
    mse_recon = mse(recon, xs)
    mean_xs = xs.mean(dim=0, keepdim=True).expand_as(xs)
    mse_mean = mse(mean_xs, xs)
    epsilon = 1e-8
    return mse_recon / (mse_mean + epsilon)

def loss_fn(X, X_):
    recon_loss = normalized_mse(X_, X)
    total_loss = recon_loss
    return recon_loss, total_loss

def upload_to_huggingface(model_path, repo_name, hf_token, commit_message):
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
    
    # Define the path within the repository where the model will be uploaded
    filename = os.path.basename(model_path)
    path_in_repo = filename  # You can change this to a subdirectory if desired
    
    # Upload the file
    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=path_in_repo,
            repo_id=repo_name,
            token=hf_token,
            commit_message=commit_message
        )
        print(f"Model '{filename}' uploaded to Hugging Face repository '{repo_name}' with commit message '{commit_message}'.")
    except Exception as e:
        print(f"Failed to upload file to Hugging Face. Error: {e}")
        sys.exit(1)

def train(model, transformer, activation_store, optimizer, device, n_batches, layer, save_path, repo_name, hf_token):
    model.train()

    recon_loss_acc = 0.0
    log_interval = 1
    upload_interval = 1000

    for batch_num in range(1, n_batches + 1):
        batch_tokens = activation_store.get_batch_tokens().to(device)

        with torch.no_grad():
            _, cache = transformer.run_with_cache(batch_tokens)
            X = cache["resid_pre", layer]
            X = einops.rearrange(X, "batch pos d_model -> (batch pos) d_model")
            del cache

        optimizer.zero_grad()
        S_, X_ = model(X)
        recon_loss, total_loss = loss_fn(X, X_)
        total_loss.backward()
        optimizer.step()

        recon_loss_acc += recon_loss.item()

        if batch_num % log_interval == 0:
            avg_recon_loss = recon_loss_acc / log_interval

            print(f"Batch [{batch_num}/{n_batches}], "
                  f"Avg Reconstruction Loss: {avg_recon_loss:.6f}")

            recon_loss_acc = 0.0

        if batch_num % upload_interval == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved to {save_path}")

            commit_message = f"Checkpoint at batch {batch_num}"

            upload_to_huggingface(save_path, repo_name, hf_token, commit_message)

    print("Training complete.")

def upload_final_model(model_path, repo_name, hf_token):
    commit_message = "Final model upload after training completion."
    upload_to_huggingface(model_path, repo_name, hf_token, commit_message)

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
        self.decoder.weight = nn.Parameter(torch.randn(M, N), requires_grad=True)

    def forward(self, X, norm_D=True):
        if norm_D:
            self.decoder.weight.data /= torch.linalg.norm(self.decoder.weight.data, dim=0, keepdim=True)
        S_ = self.encoder(X)
        X_ = torch.matmul(S_, self.decoder.weight.T)
        if self.decoder.bias is not None:
            X_ += self.decoder.bias
        return S_, X_

def main(args):
    device = get_device()
    print(f"Using device: {device}")

    seq_len = 128
    total_tokens = seq_len * args.batch_size * args.n_batches
    print(f"Training on {total_tokens / 1e6}M total tokens.")

    config = {
        'device': device,
        'batch_size': args.batch_size,
        'layer': args.layer,
        'seq_len': seq_len
    }

    transformer, activation_store = initialize_activation_store(config)

    model = MLP(
        M=args.input_dim,
        N=args.output_dim,
        h=args.hidden_dim,
        seed=20240625,
        use_bias=args.use_bias
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: Hugging Face token not found. Please set the 'HF_TOKEN' environment variable.")
        sys.exit(1)

    repo_name = "charlieoneill/sparse-coding"

    train(
        model=model,
        transformer=transformer,
        activation_store=activation_store,
        optimizer=optimizer,
        device=device,
        n_batches=args.n_batches,
        layer=args.layer,
        save_path=args.save_path,
        repo_name=repo_name,
        hf_token=hf_token
    )

    final_model_path = args.save_path
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    upload_final_model(final_model_path, repo_name, hf_token)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP on Transformer Activations with Normalized MSE")
    parser.add_argument('--layer', type=int, required=True, help='Layer number to extract activations from')
    parser.add_argument('--input_dim', type=int, required=True, help='Dimensionality of the input activations')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='Number of neurons in the hidden layer')
    parser.add_argument('--output_dim', type=int, default=16896, help='Number of neurons in the output layer (before decoder)')
    parser.add_argument('--use_bias', action='store_true', help='Whether to use bias in the layers')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--n_batches', type=int, default=50000, help='Number of training batches')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--save_path', type=str, default='models/mlp_model.pth', help='Path to save the trained model')

    args = parser.parse_args()
    main(args)