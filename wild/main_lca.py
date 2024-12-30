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
import sys
import time
import warnings
from huggingface_hub import HfApi

warnings.filterwarnings("ignore")

def get_device():
    """Determine the available device (GPU if available, else CPU)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_activation_store(config):
    """Initialize the activation store using the session loader."""
    hook_point = "blocks.8.hook_resid_pre"
    saes, _ = get_gpt2_res_jb_saes(hook_point)
    sparse_autoencoder = saes[hook_point]
    sparse_autoencoder.to(config['device'])
    sparse_autoencoder.cfg.device = config['device']
    sparse_autoencoder.cfg.hook_point = f"blocks.{config['layer']}.attn.hook_z"
    sparse_autoencoder.cfg.batch_size = config['batch_size']
    sparse_autoencoder.cfg.store_batch_size = config['batch_size']
    sparse_autoencoder.cfg.seq_length = config['seq_len']

    loader = LMSparseAutoencoderSessionloader(sparse_autoencoder.cfg)
    transformer_model, _, activation_store = loader.load_sae_training_group_session()
    return transformer_model.to(config['device']), activation_store

def upload_to_huggingface(model_path, repo_name, hf_token, commit_message):
    """Upload model to HuggingFace Hub."""
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
    path_in_repo = filename
    
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

def upload_final_model(model_path, repo_name, hf_token, skip_upload=False):
    """Upload the final model to HuggingFace."""
    if not skip_upload:
        commit_message = "Final model upload after training completion."
        upload_to_huggingface(model_path, repo_name, hf_token, commit_message)

class LCA(nn.Module):
    def __init__(self, input_dim, dict_size, lambd=0.1, lr=None, max_iter=300, 
                 fac=0.5, tol=1e-6, device='cuda', verbose=False):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.D = nn.Parameter(torch.randn(input_dim, dict_size, device=device))
        self.normalize_dictionary()
        self.lambd = lambd
        self.lr = lr if lr is not None else 1.0 / dict_size
        self.max_iter = max_iter
        self.facs = [fac, 1/fac]
        self.tol = tol
        self.device = device
        self.verbose = verbose

    def normalize_dictionary(self):
        with torch.no_grad():
            self.D.data = F.normalize(self.D.data, dim=0)

    def inference(self, x):
        batch_size = x.shape[0]
        u = torch.zeros(batch_size, self.dict_size, device=self.device)
        a = torch.relu(u)
        
        if isinstance(self.lr, float):
            lr = torch.full((batch_size,), self.lr, device=self.device)
        else:
            lr = self.lr.clone()
        
        best_loss = torch.full((batch_size,), float('inf'), device=self.device)
        
        for iter_idx in range(self.max_iter):
            rec = torch.matmul(a, self.D.T)
            
            recon_error = torch.mean((x - rec) ** 2, dim=1)
            l1_penalty = self.lambd * torch.mean(torch.abs(a), dim=1)
            loss = recon_error + l1_penalty
            l0_per_item = (a > 0).float().sum(dim=1).mean().item()
            
            if self.verbose and iter_idx % 10 == 0:
                avg_loss = loss.mean().item()
                avg_recon = recon_error.mean().item()
                avg_l1 = l1_penalty.mean().item()
                sparsity = (a > 0).float().mean().item() * 100
                print(f"LCA Iteration {iter_idx}: "
                      f"Loss = {avg_loss:.6f}, "
                      f"Recon = {avg_recon:.6f}, "
                      f"L1 = {avg_l1:.6f}, "
                      f"L0 (avg features) = {l0_per_item:.1f}, "
                      f"Sparsity = {sparsity:.1f}%")
            
            if torch.max(best_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iter_idx}")
                break
            
            best_loss = loss
            du = torch.matmul((rec - x), self.D) + self.lambd
            
            losses = []
            u_candidates = []
            a_candidates = []
            
            for fac in self.facs:
                lr_expanded = lr.view(-1, 1)
                u_new = u - du * (lr_expanded * fac)
                a_new = torch.relu(u_new)
                rec_new = torch.matmul(a_new, self.D.T)
                
                recon_error_new = torch.mean((x - rec_new) ** 2, dim=1)
                l1_penalty_new = self.lambd * torch.mean(torch.abs(a_new), dim=1)
                loss_new = recon_error_new + l1_penalty_new
                
                losses.append(loss_new)
                u_candidates.append(u_new)
                a_candidates.append(a_new)
            
            losses = torch.stack(losses, dim=0)
            best_idx = torch.argmin(losses, dim=0)
            
            u = torch.stack([
                u_candidates[best_idx[i]][i] for i in range(batch_size)
            ])
            
            a = torch.relu(u)
            lr = lr * torch.tensor([self.facs[idx.item()] for idx in best_idx], 
                                 device=self.device)
        
        return a

    def forward(self, x):
        self.normalize_dictionary()
        a = self.inference(x)
        x_recon = torch.matmul(a, self.D.T)
        return a, x_recon

def normalized_mse(recon, xs):
    mse = F.mse_loss(recon, xs)
    mean_xs = xs.mean(dim=0, keepdim=True).expand_as(xs)
    mse_mean = F.mse_loss(mean_xs, xs)
    epsilon = 1e-8
    return mse / (mse_mean + epsilon)

def loss_fn(X, X_, S_):
    recon_loss = normalized_mse(X_, X)
    total_loss = recon_loss
    return recon_loss, total_loss

def train(model, transformer, activation_store, optimizer, device, n_batches, 
          layer, save_path, repo_name, hf_token, skip_upload=False):
    model.train()
    recon_loss_acc = 0.0
    log_interval = 1
    upload_interval = 100

    print("\nStarting training...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Dictionary shape: {model.D.shape}")
    
    # Track dictionary changes
    old_dict = model.D.data.clone()
    
    for batch_num in range(1, n_batches + 1):
        batch_tokens = activation_store.get_batch_tokens().to(device)
        print(f"Batch tokens shape: {batch_tokens.shape}")
        
        with torch.no_grad():
            _, cache = transformer.run_with_cache(batch_tokens)
            X = cache["resid_pre", layer]
            X = einops.rearrange(X, "batch pos d_model -> (batch pos) d_model")
            X = X[:128]
            print(f"\nBatch {batch_num}, Input shape: {X.shape}")
            del cache

        optimizer.zero_grad()
        S_, X_ = model(X)
        recon_loss, total_loss = loss_fn(X, X_, S_)
        total_loss.backward()
        optimizer.step()
        
        # Explicitly normalize dictionary after optimization step
        model.normalize_dictionary()
        
        # Calculate dictionary update magnitude
        with torch.no_grad():
            dict_diff = torch.norm(model.D.data - old_dict).item()
            dict_grad_norm = torch.norm(model.D.grad).item() if model.D.grad is not None else 0.0
            old_dict = model.D.data.clone()

        recon_loss_acc += recon_loss.item()

        if batch_num % log_interval == 0:
            avg_recon_loss = recon_loss_acc / log_interval
            l0_per_item = (S_ > 0).float().sum(dim=1).mean().item()
            print(f"Batch [{batch_num}/{n_batches}]")
            print(f"  Reconstruction Loss: {avg_recon_loss:.6f}")
            print(f"  L0 (avg features): {l0_per_item:.1f}")
            print(f"  Dictionary update: {dict_diff:.6f}")
            print(f"  Dictionary grad norm: {dict_grad_norm:.6f}")
            print(f"  Dictionary norm: {torch.norm(model.D.data).item():.6f}")
            recon_loss_acc = 0.0

        if batch_num % upload_interval == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved to {save_path}")

            if not skip_upload:
                commit_message = f"Checkpoint at batch {batch_num}"
                upload_to_huggingface(save_path, repo_name, hf_token, commit_message)

    print("Training complete.")

def main(args):
    device = get_device()
    print(f"Using device: {device}")

    seq_len = 128
    total_tokens = seq_len * args.batch_size * args.n_batches
    print(f"Training on {total_tokens / 1e6}M total tokens.")

    config = {
        'device': device,
        'batch_size': 128,  # Fixed batch size
        'layer': args.layer,
        'seq_len': seq_len
    }

    transformer, activation_store = initialize_activation_store(config)

    model = LCA(
        input_dim=args.input_dim,
        dict_size=args.dict_size,
        lambd=args.lambd,
        lr=args.lca_lr,
        max_iter=args.max_iter,
        fac=args.fac,
        tol=args.tol,
        device=device,
        verbose=True
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if not args.skip_upload:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("Error: Hugging Face token not found. Please set the 'HF_TOKEN' environment variable.")
            sys.exit(1)
    else:
        hf_token = None

    repo_name = "charlieoneill/sparse-coding-lca"

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
    parser = argparse.ArgumentParser(description="Train LCA on Transformer Activations")
    parser.add_argument('--layer', type=int, required=True, 
                       help='Layer number to extract activations from')
    parser.add_argument('--input_dim', type=int, default=768, 
                       help='Dimensionality of the input activations')
    parser.add_argument('--dict_size', type=int, default=16896, 
                       help='Number of dictionary elements')
    parser.add_argument('--lambd', type=float, default=0.1, 
                       help='Sparsity coefficient')
    parser.add_argument('--lr', type=float, default=3e-3, 
                       help='Learning rate for dictionary learning')
    parser.add_argument('--lca_lr', type=float, default=None, 
                       help='Initial learning rate for LCA inference')
    parser.add_argument('--max_iter', type=int, default=300, 
                       help='Maximum iterations for LCA inference')
    parser.add_argument('--fac', type=float, default=0.5, 
                       help='Learning rate adaptation factor')
    parser.add_argument('--tol', type=float, default=1e-6, 
                       help='Convergence tolerance')
    parser.add_argument('--n_batches', type=int, default=50000, 
                       help='Number of training batches')
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='Batch size for training')
    parser.add_argument('--save_path', type=str, default='models/lca_model.pth', 
                       help='Path to save the trained model')
    parser.add_argument('--skip_upload', action='store_true', 
                       help='Skip uploading to HuggingFace')

    args = parser.parse_args()
    main(args)