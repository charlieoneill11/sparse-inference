"""
Sparse Autoencoder Analysis Script

This script loads a pre-trained sparse autoencoder model and analyzes activations
on GPT-2 using the OpenWebText dataset. It computes and analyzes feature scores
across text samples.
"""

import numpy as np
import torch
import einops
from tqdm.auto import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer
from transformer_lens.utils import to_numpy

# Import custom modules
from models import SparseAutoencoder
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes

def load_sparse_autoencoder(repo_name, model_filename, input_dim, hidden_dim):
    """Load the sparse autoencoder model from HuggingFace Hub."""
    print("Loading SAE...")
    model_path = hf_hub_download(repo_id=repo_name, filename=model_filename)
    
    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    sae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    sae.eval()
    
    return sae

def tokenize_and_concatenate(dataset, tokenizer, streaming=False, max_length=1024, 
                           column_name="text", add_bos_token=True):
    """Tokenize and process the dataset."""
    # Remove unnecessary columns
    for key in dataset.features:
        if key != column_name:
            dataset = dataset.remove_columns(key)
            
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    
    seq_len = max_length - 1 if add_bos_token else max_length

    def tokenize_function(examples):
        text = examples[column_name]
        full_text = tokenizer.eos_token.join(text)
        
        # Split text into chunks
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [full_text[i * chunk_length : (i + 1) * chunk_length]
                 for i in range(num_chunks)]
        
        # Tokenize chunks
        tokens = tokenizer(chunks, return_tensors="np", padding=True)["input_ids"].flatten()
        tokens = tokens[tokens != tokenizer.pad_token_id]
        
        # Reshape tokens into batches
        num_tokens = len(tokens)
        num_batches = num_tokens // seq_len
        tokens = tokens[: seq_len * num_batches]
        tokens = einops.rearrange(tokens, "(batch seq) -> batch seq", 
                                batch=num_batches, seq=seq_len)
        
        # Add BOS token if needed
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
            
        return {"tokens": tokens}

    return dataset.map(tokenize_function, batched=True, remove_columns=[column_name])

def compute_activations(model, tokens, sae, layer, batch_size, feature_indices):
    """Compute activations for the given tokens."""
    print("Computing activations...")
    scores = []
    
    for i in tqdm(range(0, tokens.shape[0], batch_size)):
        batch_tokens = tokens[i:i + batch_size]
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch_tokens,
                stop_at_layer=layer + 1,
                names_filter=None
            )
            X = cache["resid_pre", layer].cpu()
            X = einops.rearrange(X, "batch pos d_model -> (batch pos) d_model")
            del cache
            
            cur_scores = sae.encoder(X)[:, feature_indices]
            cur_scores_reshaped = to_numpy(
                einops.rearrange(cur_scores, "(b pos) n -> b n pos", 
                               pos=batch_tokens.shape[1])
            ).astype(np.float16)
            
            scores.append(cur_scores_reshaped)
    
    return np.concatenate(scores, axis=0)

def main():
    # Configuration
    config = {
        'repo_name': "charlieoneill/sparse-coding",
        'model_filename': "sparse_autoencoder.pth",
        'input_dim': 768,
        'hidden_dim': 22 * 768,  # Projection up parameter * input_dim
        'device': 'cpu',
        'layer': 9,
        'l1_weight': 1e-4,
        'batch_size': 64,
        'hook_point': f"blocks.8.hook_resid_pre"
    }
    
    # Load models
    sae = load_sparse_autoencoder(
        config['repo_name'], 
        config['model_filename'],
        config['input_dim'], 
        config['hidden_dim']
    )
    
    # Load transformer and activation store
    saes, _ = get_gpt2_res_jb_saes(config['hook_point'])
    sparse_autoencoder = saes[config['hook_point']]
    sparse_autoencoder.to(config['device'])
    sparse_autoencoder.cfg.device = config['device']
    sparse_autoencoder.cfg.hook_point = f"blocks.{config['layer']}.attn.hook_z"
    sparse_autoencoder.cfg.store_batch_size = config['batch_size']
    
    loader = LMSparseAutoencoderSessionloader(sparse_autoencoder.cfg)
    transformer_model, _, activation_store = loader.load_sae_training_group_session()
    
    # Load GPT-2 model
    print("Loading GPT-2 model...")
    model = HookedTransformer.from_pretrained('gpt2-small')
    
    # Load and process dataset
    print("Loading OpenWebText dataset...")
    dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    tokenized_owt = tokenize_and_concatenate(dataset, model.tokenizer, max_length=128, 
                                           streaming=True)
    tokenized_owt = tokenized_owt.shuffle(42)
    tokenized_owt = tokenized_owt.take(12800 * 2)
    
    # Prepare tokens
    owt_tokens = np.stack([x['tokens'] for x in tokenized_owt])
    owt_tokens_torch = torch.tensor(owt_tokens)
    
    # Compute activations
    feature_indices = list(range(2500))
    scores = compute_activations(model, owt_tokens_torch, sae, config['layer'], 
                               config['batch_size'], feature_indices)
    
    # Calculate and print results
    scores_sum = scores.sum(axis=0).sum(axis=1)
    print(f"Number of non-zero activations: {np.count_nonzero(scores_sum)}")
    non_zero_indices = np.nonzero(scores_sum)
    print(f"Indices of non-zero activations: {non_zero_indices}")

if __name__ == "__main__":
    main()