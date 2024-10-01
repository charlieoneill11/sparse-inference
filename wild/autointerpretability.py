import numpy as np
import einops

def tokenize_and_concatenate(
    dataset,
    tokenizer,
    streaming=False,
    max_length=1024,
    column_name="text",
    add_bos_token=True,
):
    for key in dataset.features:
        if key != column_name:
            dataset = dataset.remove_columns(key)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    seq_len = max_length - 1 if add_bos_token else max_length

    def tokenize_function(examples):
        text = examples[column_name]
        full_text = tokenizer.eos_token.join(text)
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [
            full_text[i * chunk_length : (i + 1) * chunk_length]
            for i in range(num_chunks)
        ]
        tokens = tokenizer(chunks, return_tensors="np", padding=True)[
            "input_ids"
        ].flatten()
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)
        num_batches = num_tokens // seq_len
        tokens = tokens[: seq_len * num_batches]
        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=[column_name]
    )
    return tokenized_dataset