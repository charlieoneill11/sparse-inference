import torch
import numpy as np
import einops
from tqdm.auto import tqdm
import yaml

from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import to_numpy
from huggingface_hub import hf_hub_download
from models import SparseAutoencoder

# ------------------------------------------------------------------
# 2. Tokenization helper
# ------------------------------------------------------------------
def tokenize_and_concatenate(
    dataset,
    tokenizer,
    streaming=False,
    max_length=1024,
    column_name="text",
    add_bos_token=True,
):
    """
    Given a Hugging Face dataset and a tokenizer, chunk the data into sequences
    of length `max_length`, optionally adding a BOS token at the start of each.
    """
    for key in dataset.features:
        if key != column_name:
            dataset = dataset.remove_columns(key)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    seq_len = max_length - 1 if add_bos_token else max_length

    def tokenize_function(examples):
        text = examples[column_name]
        # Join text examples with eos_token
        full_text = tokenizer.eos_token.join(text)
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [
            full_text[i * chunk_length : (i + 1) * chunk_length]
            for i in range(num_chunks)
        ]
        tokens = tokenizer(chunks, return_tensors="np", padding=True)["input_ids"].flatten()
        tokens = tokens[tokens != tokenizer.pad_token_id]

        # Break into multiple sequences
        num_tokens = len(tokens)
        num_batches = num_tokens // seq_len
        tokens = tokens[: seq_len * num_batches]
        tokens = einops.rearrange(tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len)
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=[column_name]
    )
    return tokenized_dataset


# ------------------------------------------------------------------
# 3. Main logic: 
#    - First pass to find non-zero feature indices among first 10k features
#    - Second pass to compute and store activation scores using those features
# ------------------------------------------------------------------
def main():
    # ------------------------------
    # Configurable parameters
    # ------------------------------
    device = "cpu"
    gpt2_name = "gpt2-small"
    layer = 8  # e.g. hooking 'resid_pre' at layer 8 => blocks.8.hook_resid_pre
    first_pass_samples = 12800 * 2
    second_pass_samples = 12800 * 8
    max_features_to_consider = 10_000  # We only consider indices [0..9999] in the first pass
    max_nonzero_features = 200        # We'll keep up to 200 from the non-zero set

    # Where to save final .npy
    output_numpy_path = "scores.npy"

    # Example: your saved SAE (on disk). Could also be an hf_hub_download path
    sae_model_path = "sparse_autoencoder.pth"

    # ---------------
    # 3.1 Load GPT-2
    # ---------------
    print("Loading GPT-2 model via TransformerLens...")
    model = HookedTransformer.from_pretrained(gpt2_name)
    model.to(device)

    # ---------------
    # 3.2 Load SAE
    # ---------------
    print("Loading SparseAutoencoder...")
    repo_name = "charlieoneill/sparse-coding"  # Adjust this with your repo name
    model_filename = "sparse_autoencoder.pth"  # Name of the model file you uploaded
    input_dim = 768  # Example input dim, adjust based on your model
    hidden_dim = 22 * input_dim  # Projection up parameter * input_dim
    # Download the model from Hugging Face Hub
    model_path = hf_hub_download(repo_id=repo_name, filename=model_filename)
    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    sae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    sae.eval()  # Set the model to evaluation model
    sae.to(device)

    # ---------------
    # 3.3 Dataset
    # ---------------
    print("Loading & tokenizing dataset for first pass...")
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)

    # We'll do smaller max_length here, can set as needed
    tokenized_ds = tokenize_and_concatenate(
        dataset, model.tokenizer, max_length=128, streaming=True
    )
    tokenized_ds = tokenized_ds.shuffle(42)

    # Take a limited number for first pass
    first_pass_data = tokenized_ds.take(first_pass_samples)
    # Stack into one big array
    first_pass_tokens = np.stack([x["tokens"] for x in first_pass_data], axis=0)
    first_pass_tokens_torch = torch.tensor(first_pass_tokens, device=device)

    # ---------------
    # 3.4 First pass
    #     Compute scores for the first max_features_to_consider = 10k features
    #     Then pick up to max_nonzero_features from among them
    # ---------------
    print(f"First pass: scanning up to {max_features_to_consider} features for non-zero sums...")
    batch_size = 64
    # We'll store sums for the first 10k features
    feature_sums = np.zeros((max_features_to_consider,), dtype=np.float32)

    for i in tqdm(range(0, first_pass_tokens_torch.shape[0], batch_size)):
        batch_tokens_torch = first_pass_tokens_torch[i : i + batch_size]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch_tokens_torch, 
                stop_at_layer=layer + 1,  # we want "resid_pre" at this layer
            )
            X = cache["resid_pre", layer]  # shape (batch, pos, d_model)
            X = einops.rearrange(X, "b pos d_model -> (b pos) d_model")
            del cache
            # We only slice sae.encoder outputs from 0..max_features_to_consider-1
            cur_scores = sae.encoder(X)[:, :max_features_to_consider]
        
        # Reshape to sum across all tokens
        cur_scores_np = to_numpy(cur_scores)  # shape: (b*pos, #features)
        feature_sums += cur_scores_np.sum(axis=0)

    # Identify which features are non-zero
    nonzero_mask = (feature_sums != 0)
    nonzero_indices = np.where(nonzero_mask)[0]

    # Keep only up to 200 of them
    if len(nonzero_indices) > max_nonzero_features:
        nonzero_indices = nonzero_indices[:max_nonzero_features]

    print(f"Found {len(nonzero_indices)} non-zero feature indices (among first 10k).")
    print("Indices:", nonzero_indices)

    # ---------------
    # 3.5 Second pass
    #     Now we want to run on a bigger chunk of data, 
    #     but only compute/keep the scores for the chosen features
    # ---------------
    print("Loading & tokenizing dataset for second pass...")
    # Re-load or re-shuffle the dataset
    dataset_2 = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    dataset_2 = dataset_2.shuffle(seed=42, buffer_size=10_000)
    tokenized_ds_2 = tokenize_and_concatenate(
        dataset_2, model.tokenizer, max_length=128, streaming=True
    )
    tokenized_ds_2 = tokenized_ds_2.shuffle(42)

    # Take a bigger chunk
    second_pass_data = tokenized_ds_2.take(second_pass_samples)
    # Stack
    second_pass_tokens = np.stack([x["tokens"] for x in second_pass_data], axis=0)
    second_pass_tokens_torch = torch.tensor(second_pass_tokens, device=device)

    print("Computing final scores for selected features...")
    final_scores_list = []
    for i in tqdm(range(0, second_pass_tokens_torch.shape[0], batch_size)):
        batch_tokens_torch = second_pass_tokens_torch[i : i + batch_size]
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch_tokens_torch,
                stop_at_layer=layer + 1
            )
            X = cache["resid_pre", layer]
            X = einops.rearrange(X, "b pos d_model -> (b pos) d_model")
            del cache
            # Only grab the subset of features of interest
            cur_scores = sae.encoder(X)[:, nonzero_indices]

        # Reshape: (b*pos, len(nonzero_indices)) -> (b, len(nonzero_indices), pos)
        cur_scores_reshaped = einops.rearrange(
            cur_scores, 
            "(b pos) f -> b f pos", 
            b=batch_tokens_torch.shape[0], 
            pos=batch_tokens_torch.shape[1]
        )
        final_scores_list.append(to_numpy(cur_scores_reshaped))

    final_scores = np.concatenate(final_scores_list, axis=0)  
    print("Final scores shape:", final_scores.shape)

    # ---------------
    # 3.6 Summation & saving
    # ---------------
    # Example: sum along (batch, seq), leaving just feature-dimension
    # shape (batch, f, seq) -> sum over batch & seq => (f,)
    sums_per_feature = final_scores.sum(axis=(0, 2))  
    print("Sums per selected feature:", sums_per_feature)

    np.save(output_numpy_path, final_scores.astype(np.float16))
    print(f"Done! Saved final scores array to {output_numpy_path}.")


if __name__ == "__main__":
    main()
