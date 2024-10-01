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
    
    # Combine losses
    total_loss = recon_loss + l1_weight * l1_loss
    
    return recon_loss, l1_loss, l0_loss, total_loss


def upload_to_huggingface(model_path, repo_name, hf_token, commit_message):
    """
    Upload the model to Hugging Face Hub.

    Args:
        model_path (str): Local path to the saved model.
        repo_name (str): Name of the Hugging Face repository (e.g., "username/sparse-coding").
        hf_token (str): Hugging Face authentication token.
        commit_message (str): Commit message for the upload.

    Returns:
        None
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


def train(model, transformer, activation_store, optimizer, device, n_batches, l1_weight, layer, save_path, repo_name, hf_token):
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
        save_path (str): Path to save the trained model.
        repo_name (str): Hugging Face repository name.
        hf_token (str): Hugging Face authentication token.

    Returns:
        None
    """
    model.train()
    
    # Initialize accumulators for logging
    recon_loss_acc = 0.0
    l1_loss_acc = 0.0
    l0_loss_acc = 0.0
    total_loss_acc = 0.0
    log_interval = 1  # Log every batch
    upload_interval = 1000  # Upload every 100 batches

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
        recon_loss, l1_loss, l0_loss, total_loss = loss_fn(X, X_, S_, l1_weight=l1_weight)
        total_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        recon_loss_acc += recon_loss.item()
        l1_loss_acc += l1_loss.item()
        l0_loss_acc += l0_loss.item()
        total_loss_acc += total_loss.item()
        
        # Log every log_interval batches
        if batch_num % log_interval == 0:
            avg_recon_loss = recon_loss_acc / log_interval
            avg_l1_loss = l1_loss_acc / log_interval
            avg_l0_loss = l0_loss_acc / log_interval
            avg_total_loss = total_loss_acc / log_interval
            
            print(f"Batch [{batch_num}/{n_batches}], "
                  f"Avg Reconstruction Loss: {avg_recon_loss:.6f}, "
                  f"Avg L1 Loss: {avg_l1_loss:.6f}, "
                  f"Avg L0 Loss: {avg_l0_loss:.6f}, "
                  f"Avg Total Loss: {avg_total_loss:.6f}")
            
            # Log to wandb
            wandb.log({
                "reconstruction_loss": avg_recon_loss,
                "l1_loss": avg_l1_loss,
                "l0_loss": avg_l0_loss,
                "total_loss": avg_total_loss,
                "batch": batch_num
            })
            
            # Reset accumulators
            recon_loss_acc = 0.0
            l1_loss_acc = 0.0
            l0_loss_acc = 0.0
            total_loss_acc = 0.0
        
        # Upload every upload_interval batches
        if batch_num % upload_interval == 0:
            # Save the model
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved to {save_path}")
            
            # Commit message
            commit_message = f"Checkpoint at batch {batch_num}"
            
            # Upload to Hugging Face
            upload_to_huggingface(save_path, repo_name, hf_token, commit_message)
    
    print("Training complete.")


def upload_final_model(model_path, repo_name, hf_token):
    """
    Upload the final trained model to Hugging Face Hub.

    Args:
        model_path (str): Path to the saved model.
        repo_name (str): Hugging Face repository name.
        hf_token (str): Hugging Face authentication token.

    Returns:
        None
    """
    commit_message = "Final model upload after training completion."
    upload_to_huggingface(model_path, repo_name, hf_token, commit_message)


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

    # Initialize wandb
    wandb.init(
        project="sparse-coding",
        config={
            "layer": args.layer,
            "input_dim": args.input_dim,
            "projection_up": args.projection_up,
            "l1_weight": args.l1_weight,
            "lr": args.lr,
            "n_batches": args.n_batches,
            "batch_size": args.batch_size,
            "save_path": args.save_path
        },
        name=f"layer_{args.layer}_hidden_{args.projection_up*args.input_dim}_l1_{args.l1_weight}"
    )
    
    # Print how many tokens we're training on
    seq_len = 128
    total_tokens = seq_len * args.batch_size * args.n_batches
    print(f"Training on {total_tokens / 1e6}M total tokens.")

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
    
    # Get Hugging Face token
    config = yaml.safe_load(open("config.yaml"))
    hf_token = os.getenv("HF_TOKEN")  # Alternatively, use config['HF_TOKEN']
    print(f"Using Hugging Face token: {hf_token}")
    if not hf_token:
        print("Error: Hugging Face token not found. Please set the 'HF_TOKEN' environment variable.")
        sys.exit(1)
    
    repo_name = "charlieoneill/sparse-coding"

    # Train the model
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
    
    # Save the final model
    final_model_path = args.save_path
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Upload the final model to Hugging Face
    upload_final_model(final_model_path, repo_name, hf_token)
    
    # Finish wandb run
    wandb.finish()


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
        default=1e-4,
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
        default=50000,
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

# training_script.py

# import torch
# from torch.nn import functional as F
# from torch import nn, optim
# from torch.utils.data import DataLoader, TensorDataset
# import einops
# import argparse
# import os
# from transformer_lens import HookedTransformer
# from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader  
# from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
# from vanilla_sae import SparseAutoencoder

# import wandb
# from huggingface_hub import HfApi
# import sys
# import time
# import yaml

# import warnings
# warnings.filterwarnings("ignore")


# def get_device():
#     """Determine the available device (GPU if available, else CPU)."""
#     return 'cuda' if torch.cuda.is_available() else 'cpu'


# def initialize_activation_store(config):
#     """
#     Initialize the activation store using the session loader.

#     Args:
#         config (dict): Configuration dictionary for the session loader.

#     Returns:
#         Tuple: (transformer_model, activation_store)
#     """
#     # Load the transformer model and activation store
#     hook_point = "blocks.8.hook_resid_pre"  # Placeholder hook point
#     saes, _ = get_gpt2_res_jb_saes(hook_point)
#     sparse_autoencoder = saes[hook_point]
#     sparse_autoencoder.to(config['device'])
#     sparse_autoencoder.cfg.device = config['device']
#     sparse_autoencoder.cfg.hook_point = f"blocks.{config['layer']}.attn.hook_z"
#     sparse_autoencoder.cfg.store_batch_size = config['batch_size']

#     loader = LMSparseAutoencoderSessionloader(sparse_autoencoder.cfg)
#     transformer_model, _, activation_store = loader.load_sae_training_group_session()
#     return transformer_model.to(config['device']), activation_store


# def mse(output, target):
#     """
#     Compute the Mean Squared Error between output and target.

#     Args:
#         output (torch.Tensor): Reconstructed activations.
#         target (torch.Tensor): Original activations.

#     Returns:
#         torch.Tensor: MSE value.
#     """
#     return F.mse_loss(output, target)


# def normalized_mse(recon, xs):
#     """
#     Compute the Normalized Mean Squared Error.

#     Normalizes the reconstruction loss by the MSE of the target's mean.

#     Args:
#         recon (torch.Tensor): Reconstructed activations.
#         xs (torch.Tensor): Original activations.

#     Returns:
#         torch.Tensor: Normalized MSE value.
#     """
#     # Compute MSE between recon and xs
#     mse_recon = mse(recon, xs)
    
#     # Compute MSE between the mean of xs and xs
#     mean_xs = xs.mean(dim=0, keepdim=True).expand_as(xs)
#     mse_mean = mse(mean_xs, xs)
    
#     # To avoid division by zero, add a small epsilon to mse_mean
#     epsilon = 1e-8
#     return mse_recon / (mse_mean + epsilon)


# def loss_fn(X, X_, S_, l1_weight=0.01):
#     """
#     Compute the combined normalized reconstruction and sparsity loss.

#     Args:
#         X (torch.Tensor): Original activations.
#         X_ (torch.Tensor): Reconstructed activations.
#         S_ (torch.Tensor): Sparse activations.
#         l1_weight (float): Weight for the L1 loss component.

#     Returns:
#         torch.Tensor: Combined loss value.
#     """
#     # Use Normalised MSE for reconstruction loss
#     recon_loss = normalized_mse(X_, X)
    
#     # Compute L1 and L0 sparsity losses
#     l1_loss = S_.norm(p=1, dim=-1).mean()
#     l0_loss = S_.norm(p=0, dim=-1).mean()
    
#     # Combine losses
#     total_loss = recon_loss + l1_weight * l1_loss
    
#     return recon_loss, l1_loss, l0_loss, total_loss


# def upload_to_huggingface(model_path, repo_name, hf_token, commit_message):
#     """
#     Upload the model to Hugging Face Hub.

#     Args:
#         model_path (str): Local path to the saved model.
#         repo_name (str): Name of the Hugging Face repository (e.g., "username/sparse-coding").
#         hf_token (str): Hugging Face authentication token.
#         commit_message (str): Commit message for the upload.

#     Returns:
#         None
#     """
#     api = HfApi()
    
#     # Check if the repository exists; if not, create it
#     try:
#         api.repo_info(repo_id=repo_name)
#         print(f"Repository '{repo_name}' already exists on Hugging Face.")
#     except Exception as e:
#         print(f"Creating repository '{repo_name}' on Hugging Face.")
#         try:
#             api.create_repo(repo_id=repo_name, private=False)
#             print(f"Repository '{repo_name}' created successfully.")
#         except Exception as e:
#             print(f"Failed to create repository '{repo_name}'. Error: {e}")
#             sys.exit(1)
    
#     # Define the path within the repository where the model will be uploaded
#     filename = os.path.basename(model_path)
#     path_in_repo = filename  # You can change this to a subdirectory if desired
    
#     # Upload the file
#     try:
#         api.upload_file(
#             path_or_fileobj=model_path,
#             path_in_repo=path_in_repo,
#             repo_id=repo_name,
#             token=hf_token,
#             commit_message=commit_message
#         )
#         print(f"Model '{filename}' uploaded to Hugging Face repository '{repo_name}' with commit message '{commit_message}'.")
#     except Exception as e:
#         print(f"Failed to upload file to Hugging Face. Error: {e}")
#         sys.exit(1)


# # --- New Functions for Neuron Tracking and Resampling ---

# def track_neuron_activity(S_, neuron_activity, step, log_interval=25000):
#     """
#     Track the activity of each neuron.

#     Args:
#         S_ (torch.Tensor): Sparse activations from the model. Shape: (batch_size, n_features)
#         neuron_activity (torch.Tensor): Tensor to accumulate neuron activity.
#         step (int): Current training step.
#         log_interval (int): Interval for tracking steps.

#     Returns:
#         None
#     """
#     with torch.no_grad():
#         fired_neurons = (S_ != 0).float().sum(dim=0)  # Shape: (n_features,)
#         neuron_activity[step % log_interval] = fired_neurons


# def resample_neurons(model, neuron_activity, optimizer, resampling_threshold=1e-7, log_interval=25000):
#     """
#     Resample (reinitialize) dead neurons based on their firing rates.

#     Args:
#         model (nn.Module): The sparse autoencoder model.
#         neuron_activity (torch.Tensor): Tensor tracking neuron activity.
#         optimizer (torch.optim.Optimizer): Optimizer for the model.
#         resampling_threshold (float): Threshold below which neurons are considered dead.
#         log_interval (int): Interval used for tracking neuron activity.

#     Returns:
#         int: Number of neurons resampled.
#     """
#     with torch.no_grad():
#         total_steps = neuron_activity.shape[0]
#         fired_counts = neuron_activity.sum(dim=0)  # Total fires per neuron
#         firing_rates = fired_counts / total_steps  # Firing rate per neuron
#         dead_neurons = firing_rates < resampling_threshold  # Boolean mask

#         num_dead = dead_neurons.sum().item()
#         if num_dead == 0:
#             print("No dead neurons detected. No resampling needed.")
#             return 0

#         print(f"Resampling {num_dead} dead neurons out of {dead_neurons.shape[0]} total neurons.")

#         # Reinitialize encoder weights for dead neurons
#         if hasattr(model, 'W_enc'):
#             nn.init.kaiming_uniform_(model.W_enc[:, dead_neurons], a=math.sqrt(5))
        
#         # Reinitialize decoder weights for dead neurons
#         if hasattr(model, 'W_dec'):
#             nn.init.kaiming_uniform_(model.W_dec.weight[dead_neurons, :], a=math.sqrt(5))
        
#         # Reset biases for dead neurons
#         if hasattr(model, 'b_mag'):
#             model.b_mag[dead_neurons] = 0.0
#         if hasattr(model, 'b_gate'):
#             model.b_gate[dead_neurons] = 0.0
#         if hasattr(model.W_dec, 'bias'):
#             model.W_dec.bias[dead_neurons] = 0.0

#         # Reset optimizer state for dead neurons
#         for group in optimizer.param_groups:
#             for p in group['params']:
#                 if p.grad is not None:
#                     state = optimizer.state[p]
#                     if 'exp_avg' in state:
#                         state['exp_avg'][dead_neurons] = 0
#                     if 'exp_avg_sq' in state:
#                         state['exp_avg_sq'][dead_neurons] = 0

#         return num_dead


# import math  # Required for resampling


# def train(model, transformer, activation_store, optimizer, device, n_batches, l1_weight, layer, save_path, repo_name, hf_token, resample_steps=None, resample_threshold=1e-7, log_interval=1, upload_interval=1000, neuron_log_interval=25000):
#     """
#     Train the sparse autoencoder using activations from the activation store.

#     Args:
#         model (nn.Module): The sparse autoencoder model.
#         transformer (HookedTransformer): The transformer model to obtain activations.
#         activation_store: Activation store object to fetch activation batches.
#         optimizer (optim.Optimizer): Optimizer.
#         device (str): Device to train on.
#         n_batches (int): Number of training batches.
#         l1_weight (float): Weight for the L1 loss component.
#         layer (int): Layer number to extract activations from.
#         save_path (str): Path to save the trained model.
#         repo_name (str): Hugging Face repository name.
#         hf_token (str): Hugging Face authentication token.
#         resample_steps (set, optional): Set of steps at which to resample neurons.
#         resample_threshold (float): Threshold for neuron firing rates to consider dead.
#         log_interval (int): Interval for logging.
#         upload_interval (int): Interval for uploading model checkpoints.
#         neuron_log_interval (int): Number of steps to track neuron activity.

#     Returns:
#         None
#     """
#     model.train()
    
#     # Initialize accumulators for logging
#     recon_loss_acc = 0.0
#     l1_loss_acc = 0.0
#     l0_loss_acc = 0.0
#     total_loss_acc = 0.0
    
#     # Initialize neuron activity tracker
#     neuron_activity = torch.zeros((neuron_log_interval, model.n_learned_features), device=device)
#     total_steps = 0

#     for batch_num in range(1, n_batches + 1):
#         # Fetch a batch of tokens from the activation store
#         batch_tokens = activation_store.get_batch_tokens().to(device)
        
#         # Obtain activations from the transformer model
#         with torch.no_grad():
#             _, cache = transformer.run_with_cache(batch_tokens)
#             X = cache["resid_pre", layer]  # Shape: (batch, pos, d_model)
#             X = einops.rearrange(X, "batch pos d_model -> (batch pos) d_model")
#             del cache

#         optimizer.zero_grad()
#         S_, X_ = model(X)
#         recon_loss, l1_loss, l0_loss, total_loss = loss_fn(X, X_, S_, l1_weight=l1_weight)
#         total_loss.backward()
#         optimizer.step()
        
#         # Accumulate losses
#         recon_loss_acc += recon_loss.item()
#         l1_loss_acc += l1_loss.item()
#         l0_loss_acc += l0_loss.item()
#         total_loss_acc += total_loss.item()
        
#         # Track neuron activity
#         track_neuron_activity(S_, neuron_activity, total_steps, log_interval=neuron_log_interval)
#         total_steps += 1  # Increment step count

#         # Log every log_interval batches
#         if batch_num % log_interval == 0:
#             avg_recon_loss = recon_loss_acc / log_interval
#             avg_l1_loss = l1_loss_acc / log_interval
#             avg_l0_loss = l0_loss_acc / log_interval
#             avg_total_loss = total_loss_acc / log_interval
            
#             print(f"Batch [{batch_num}/{n_batches}], "
#                   f"Avg Reconstruction Loss: {avg_recon_loss:.6f}, "
#                   f"Avg L1 Loss: {avg_l1_loss:.6f}, "
#                   f"Avg L0 Loss: {avg_l0_loss:.6f}, "
#                   f"Avg Total Loss: {avg_total_loss:.6f}")
            
#             # Log to wandb
#             wandb.log({
#                 "reconstruction_loss": avg_recon_loss,
#                 "l1_loss": avg_l1_loss,
#                 "l0_loss": avg_l0_loss,
#                 "total_loss": avg_total_loss,
#                 "batch": batch_num
#             })
            
#             # Reset accumulators
#             recon_loss_acc = 0.0
#             l1_loss_acc = 0.0
#             l0_loss_acc = 0.0
#             total_loss_acc = 0.0
        
#         # Resample neurons at specified steps
#         if resample_steps and total_steps in resample_steps:
#             num_resampled = resample_neurons(model, neuron_activity, optimizer, resample_threshold, log_interval=neuron_log_interval)
#             wandb.log({"resampled_neurons": num_resampled})
#             print(f"Resampled {num_resampled} neurons at step {total_steps}.")
        
#         # Upload every upload_interval batches
#         if batch_num % upload_interval == 0:
#             # Save the model
#             torch.save(model.state_dict(), save_path)
#             print(f"Model checkpoint saved to {save_path}")
            
#             # Commit message
#             commit_message = f"Checkpoint at batch {batch_num}"
            
#             # Upload to Hugging Face
#             # Uncomment the following line to enable uploading
#             # upload_to_huggingface(save_path, repo_name, hf_token, commit_message)
    
#     print("Training complete.")


# def upload_final_model(model_path, repo_name, hf_token):
#     """
#     Upload the final trained model to Hugging Face Hub.

#     Args:
#         model_path (str): Path to the saved model.
#         repo_name (str): Hugging Face repository name.
#         hf_token (str): Hugging Face authentication token.

#     Returns:
#         None
#     """
#     commit_message = "Final model upload after training completion."
#     upload_to_huggingface(model_path, repo_name, hf_token, commit_message)


# def main(args):
#     """
#     Main function to set up and train the sparse autoencoder.

#     Args:
#         args: Parsed command-line arguments.

#     Returns:
#         None
#     """
#     device = get_device()
#     print(f"Using device: {device}")

#     # Initialize wandb
#     wandb.init(
#         project="sparse-coding",
#         config={
#             "layer": args.layer,
#             "input_dim": args.input_dim,
#             "projection_up": args.projection_up,
#             "l1_weight": args.l1_weight,
#             "lr": args.lr,
#             "n_batches": args.n_batches,
#             "batch_size": args.batch_size,
#             "save_path": args.save_path
#         },
#         name=f"layer_{args.layer}_hidden_{args.projection_up * args.input_dim}_l1_{args.l1_weight}"
#     )
    
#     # Print how many tokens we're training on
#     seq_len = 128
#     total_tokens = seq_len * args.batch_size * args.n_batches
#     print(f"Training on {total_tokens / 1e6}M total tokens.")

#     # Configuration for the activation store
#     config = {
#         'device': device,
#         'hook_point': f"blocks.{args.layer}.attn.hook_z",
#         'batch_size': args.batch_size,
#         'layer': args.layer
#     }
    
#     # Initialize the transformer model and activation store
#     transformer, activation_store = initialize_activation_store(config)
    
#     # Initialize the Sparse Autoencoder (Updated Initialization)
#     model = SparseAutoencoder(
#         n_input_features=args.input_dim,
#         n_learned_features=args.projection_up * args.input_dim
#     ).to(device)
    
#     # Define optimizer
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
#     # Get Hugging Face token
#     # Assuming 'config.yaml' contains necessary configurations
#     # You may need to adjust this based on your actual config structure
#     config_file = "config.yaml"
#     if os.path.exists(config_file):
#         config_data = yaml.safe_load(open(config_file))
#         hf_token = os.getenv("HF_TOKEN", config_data.get('HF_TOKEN'))
#     else:
#         hf_token = os.getenv("HF_TOKEN")  # Alternatively, use config['HF_TOKEN']
#     print(f"Using Hugging Face token: {'***' if hf_token else 'None'}")
#     if not hf_token:
#         print("Error: Hugging Face token not found. Please set the 'HF_TOKEN' environment variable.")
#         sys.exit(1)
    
#     repo_name = "charlieoneill/sparse-coding"

#     # Define resampling steps (e.g., every 50,000 steps)
#     resample_steps = set(range(50000, args.n_batches + 1, 50000)) if args.resample else None

#     # Train the model
#     train(
#         model=model,
#         transformer=transformer,
#         activation_store=activation_store,
#         optimizer=optimizer,
#         device=device,
#         n_batches=args.n_batches,
#         l1_weight=args.l1_weight,
#         layer=args.layer,
#         save_path=args.save_path,
#         repo_name=repo_name,
#         hf_token=hf_token,
#         resample_steps=resample_steps,
#         resample_threshold=args.resample_threshold,
#         log_interval=args.log_interval,
#         upload_interval=args.upload_interval,
#         neuron_log_interval=args.neuron_log_interval
#     )
    
#     # Save the final model
#     final_model_path = args.save_path
#     torch.save(model.state_dict(), final_model_path)
#     print(f"Final model saved to {final_model_path}")
    
#     # Upload the final model to Hugging Face
#     upload_final_model(final_model_path, repo_name, hf_token)
    
#     # Finish wandb run
#     wandb.finish()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Train a Sparse Autoencoder on Transformer Activations with Normalized MSE and Neuron Resampling"
#     )
#     parser.add_argument(
#         '--layer',
#         type=int,
#         required=True,
#         help='Layer number to extract activations from'
#     )
#     parser.add_argument(
#         '--input_dim',
#         type=int,
#         required=True,
#         help='Dimensionality of the input activations'
#     )
#     parser.add_argument(
#         '--projection_up',
#         type=int,
#         default=22,
#         help='Dimensionality of the hidden layer'
#     )
#     parser.add_argument(
#         '--l1_weight',
#         type=float,
#         default=1e-4,
#         help='L1 coefficient for the loss function'
#     )
#     parser.add_argument(
#         '--lr',
#         type=float,
#         default=3e-3,
#         help='Learning rate'
#     )
#     parser.add_argument(
#         '--n_batches',
#         type=int,
#         default=50000,
#         help='Number of training batches'
#     )
#     parser.add_argument(
#         '--batch_size',
#         type=int,
#         default=64,
#         help='Batch size for training'
#     )
#     parser.add_argument(
#         '--save_path',
#         type=str,
#         default='models/sparse_autoencoder.pth',
#         help='Path to save the trained model'
#     )
#     # --- New Arguments for Resampling ---
#     parser.add_argument(
#         '--resample',
#         action='store_true',
#         help='Enable neuron resampling during training'
#     )
#     parser.add_argument(
#         '--resample_threshold',
#         type=float,
#         default=1e-7,
#         help='Threshold below which neurons are considered dead for resampling'
#     )
#     parser.add_argument(
#         '--log_interval',
#         type=int,
#         default=1,
#         help='Number of batches between each logging step'
#     )
#     parser.add_argument(
#         '--upload_interval',
#         type=int,
#         default=1000,
#         help='Number of batches between each model upload'
#     )
#     parser.add_argument(
#         '--neuron_log_interval',
#         type=int,
#         default=25000,
#         help='Number of steps to track neuron activity before resampling'
#     )
    
#     args = parser.parse_args()
#     main(args)
