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
    """
    hook_point = "blocks.8.hook_resid_pre"
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
    """
    Minimal version: just reconstruction + normalized MSE
    """
    recon_loss = normalized_mse(X_, X)
    total_loss = recon_loss
    return recon_loss, total_loss


def upload_to_huggingface(model_path, repo_name, hf_token, commit_message):
    """
    Uploads the model to a Hugging Face repository.
    """
    api = HfApi()
    
    try:
        api.repo_info(repo_id=repo_name)
        print(f"Repository '{repo_name}' already exists on Hugging Face.")
    except Exception:
        print(f"Creating repository '{repo_name}' on Hugging Face.")
        try:
            api.create_repo(repo_id=repo_name, private=False)
            print(f"Repository '{repo_name}' created successfully.")
        except Exception as e:
            print(f"Failed to create repository '{repo_name}'. Error: {e}")
            sys.exit(1)
    
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
        print(f"Model '{filename}' uploaded to Hugging Face repository '{repo_name}' "
              f"with commit message '{commit_message}'.")
    except Exception as e:
        print(f"Failed to upload file to Hugging Face. Error: {e}")
        sys.exit(1)


def resample_dead_neurons(model, optimizer, dead_neurons):
    """
    Resample "dead" neurons by reinitializing relevant rows/cols of encoder/decoder.
    Minimal logic: picks random directions for the decoder side, 
    sets encoder row accordingly.
    """
    if not dead_neurons.any():
        print("No dead neurons to resample.")
        return

    with torch.no_grad():
        # We'll assume the model has: model.encoder(...) and model.decoder(...) 
        # That means something like an MLP or autoencoder with separate 
        # parameters. Adjust to your actual code.
        encoder_weight = model.encoder[0].weight.data
        decoder_weight = model.decoder.weight.data

        hidden_dim, input_dim = encoder_weight.shape
        num_resampled = 0

        for idx in torch.where(dead_neurons)[0]:
            # Random new direction for decoder column
            new_dec = torch.randn(input_dim, device=decoder_weight.device)
            new_dec /= (new_dec.norm() + 1e-12)
            decoder_weight[:, idx] = new_dec

            # For encoder row, make it smaller norm
            new_enc = new_dec.clone() * 0.2
            encoder_weight[idx] = new_enc

            # Reset bias if it exists
            if hasattr(model.encoder[0], 'bias') and model.encoder[0].bias is not None:
                model.encoder[0].bias.data[idx] = 0.0

            # Reset optimizer states
            params_to_reset = [
                model.encoder[0].weight,
                model.decoder.weight
            ]
            # If bias param is in the optimizer, we reset that as well
            if hasattr(model.encoder[0], 'bias') and model.encoder[0].bias is not None:
                params_to_reset.append(model.encoder[0].bias)

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
    layer,
    save_path,
    repo_name,
    hf_token,
    skip_upload=False
):
    model.train()

    # Logging accumulators
    recon_loss_acc = 0.0
    log_interval = 100
    upload_interval = 1000

    # We define max_track_steps=5000
    max_track_steps = 5000

    # We'll keep track of neuron firing: (max_track_steps, hidden_dim)
    # Mark a neuron "fired" if its activation is ever >0 in the current batch.
    hidden_dim = model.encoder[0].weight.shape[0]
    fired_buffer = torch.zeros((max_track_steps, hidden_dim), dtype=torch.bool, device='cpu')
    step_idx = 0

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

        # Mark which neurons fired
        # If any activation > 0 for that neuron in the batch
        fired = (S_ > 0).any(dim=0).cpu()
        fired_buffer[step_idx] = fired
        step_idx = (step_idx + 1) % max_track_steps

        # Logging
        if batch_num % log_interval == 0:
            avg_recon_loss = recon_loss_acc / log_interval
            print(f"Batch [{batch_num}/{n_batches}], "
                  f"Avg Reconstruction Loss: {avg_recon_loss:.6f}")
            recon_loss_acc = 0.0

        # Model checkpoint and upload
        if batch_num % upload_interval == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved to {save_path}")
            if not skip_upload and hf_token is not None:
                commit_message = f"Checkpoint at batch {batch_num}"
                upload_to_huggingface(save_path, repo_name, hf_token, commit_message)

        # Resample every 5000 batches
        if batch_num % 5000 == 0:
            print(f"--- Resampling at batch {batch_num} ---")
            sum_of_fires = fired_buffer.sum(dim=0)  # [hidden_dim]
            dead_neurons = (sum_of_fires == 0)
            print(f"Dead neurons: {dead_neurons.sum().item()}")
            resample_dead_neurons(model, optimizer, dead_neurons)

            # Reset buffer
            fired_buffer.zero_()
            step_idx = 0

    print("Training complete.")


def upload_final_model(model_path, repo_name, hf_token, skip_upload=False):
    if not skip_upload and hf_token is not None:
        commit_message = "Final model upload after training completion."
        upload_to_huggingface(model_path, repo_name, hf_token, commit_message)


# Example MLP or Autoencoder model
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
    device = get_device()
    print(f"Using device: {device}")

    seq_len = 128
    total_tokens = seq_len * args.batch_size * args.n_batches
    print(f"Training on {total_tokens / 1e6:.2f}M total tokens.")

    config = {
        'device': device,
        'batch_size': args.batch_size,
        'layer': args.layer,
        'seq_len': seq_len
    }

    # Load the model that yields the activations + store
    transformer, activation_store = initialize_activation_store(config)

    # Create our MLP model
    model = MLP(
        M=args.input_dim,
        N=args.output_dim,
        h=args.hidden_dim,
        seed=20240625,
        use_bias=args.use_bias
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Get HF token if not skipping upload
    if not args.skip_upload:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("Error: Hugging Face token not found. Please set HF_TOKEN.")
            sys.exit(1)
    else:
        hf_token = None

    repo_name = "charlieoneill/sparse-coding"

    # Train
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
        hf_token=hf_token,
        skip_upload=args.skip_upload
    )

    final_model_path = args.save_path
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    upload_final_model(final_model_path, repo_name, hf_token, args.skip_upload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP with Dead-Neuron Resampling")
    parser.add_argument('--layer', type=int, required=True, help='Layer to extract activations from')
    parser.add_argument('--input_dim', type=int, default=768, help='Input dimension (d_model of the LLM layer)')
    parser.add_argument('--hidden_dim', type=int, default=4224, help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=16896, help='Number of neurons in the second encoder layer')
    parser.add_argument('--use_bias', action='store_true', help='Whether to use bias in MLP layers')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--n_batches', type=int, default=50000, help='Number of training batches')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--save_path', type=str, default='models/mlp_model.pth', help='Where to save the model')
    parser.add_argument('--skip_upload', action='store_true', help='Skip uploading to Hugging Face')
    args = parser.parse_args()

    main(args)