#!/usr/bin/env python3
"""
Script to analyze sparse autoencoder features by retrieving top-k activating examples,
obtaining top-k and bottom-k boosted logits, formatting prompts, and getting responses
from an AI interpreter.

Requirements:
- torch
- huggingface_hub
- einops
- numpy
- yaml
- transformer_lens
- datasets
- tqdm
- openai
- IPython
"""

import torch
from models import SparseAutoencoder  # Ensure this matches your model definition
from huggingface_hub import hf_hub_download
import einops
import numpy as np
import yaml
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from datasets import load_dataset
from openai import OpenAI, AzureOpenAI
import re
import html
from IPython.display import HTML, display

# ----------------------------- Configuration -----------------------------

# Parameters
REPO_NAME = "charlieoneill/sparse-coding"  # Hugging Face repo name
MODEL_FILENAME = "sparse_autoencoder.pth"  # Model file name in the repo
INPUT_DIM = 768  # Example input dimension
HIDDEN_DIM = 22 * INPUT_DIM  # Projection up parameter * input_dim
SCORES_PATH = "scores.npy"  # Path to the saved scores
FEATURE_INDICES = [  0,   8,  15,  25,  32,  37,  39,  43,  50,  51,  55,  61,  63,
        65,  70,  74,  81,  93,  94,  97, 112, 117, 118, 119, 120, 122,
       128, 130, 139, 150, 152, 153, 154, 156, 163, 172, 173, 177, 183,
       191, 202, 206, 207, 215, 216, 218, 224, 229, 243, 260, 264, 273,
       275, 284, 287, 293, 297, 298, 299, 301, 302, 306, 308, 309, 325,
       330, 332, 335, 337, 338, 345, 347, 354, 356, 358, 370, 371, 373,
       381, 384, 389, 402, 413, 417, 420, 421, 426, 436, 438, 449, 451,
       453, 456, 462, 463, 464, 465, 467, 472, 481, 485, 488, 492, 509,
       513, 517, 524, 527, 538, 541, 548, 552, 560, 563, 566, 568, 573,
       575, 577, 579, 581, 582, 585, 587, 590, 594, 596, 603, 605, 621,
       624, 626, 628, 634, 636, 638, 639, 650, 656, 657, 669, 676, 677,
       687, 688, 693, 699, 709, 712, 715, 718, 728, 730, 738, 742, 743,
       751, 766, 772, 774, 776, 778, 781, 782, 786, 787, 789, 794, 796,
       797, 798, 802, 805, 808, 810, 821, 822, 823, 828, 829, 845, 851,
       864, 873, 874, 878, 879, 881, 898, 907, 908, 917, 924, 926, 930,
       931, 933, 936, 938, 953, 954, 975, 981, 983, 987, 988, 996]
feature_indices = FEATURE_INDICES
TOP_K = 10  # Number of top activating examples
BOTTOM_K = 10  # Number of bottom boosted logits

# OpenAI Configuration
CONFIG_PATH = "config.yaml"  # Path to your config file containing API keys

# ----------------------------- Helper Functions -----------------------------

def load_sparse_autoencoder(repo_name: str, model_filename: str, input_dim: int, hidden_dim: int) -> SparseAutoencoder:
    model_path = hf_hub_download(repo_id=repo_name, filename=model_filename)
    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    sae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    sae.eval()
    return sae

def load_transformer_model(model_name: str = 'gpt2-small') -> HookedTransformer:
    model = HookedTransformer.from_pretrained(model_name)
    return model.cpu()

def load_scores(scores_path: str) -> np.ndarray:
    scores = np.load(scores_path)
    return scores

def load_tokenized_data(max_length: int = 128, batch_size: int = 64, take_size: int = 102400) -> torch.Tensor:
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

    transformer_model = load_transformer_model()
    dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    tokenized_owt = tokenize_and_concatenate(dataset, transformer_model.tokenizer, max_length=max_length, streaming=True)
    tokenized_owt = tokenized_owt.shuffle(42)
    tokenized_owt = tokenized_owt.take(take_size)
    owt_tokens = np.stack([x['tokens'] for x in tokenized_owt])
    owt_tokens_torch = torch.tensor(owt_tokens)
    return owt_tokens_torch

def compute_scores(sae: SparseAutoencoder, transformer_model: HookedTransformer, owt_tokens_torch: torch.Tensor, layer: int, feature_indices: list, device: str = 'cpu') -> np.ndarray:
    sae.eval()

    # Compute scores
    scores = []
    batch_size = 64
    for i in tqdm(range(0, owt_tokens_torch.shape[0], batch_size), desc="Computing scores"):
        with torch.no_grad():
            _, cache = transformer_model.run_with_cache(
                owt_tokens_torch[i : i + batch_size],
                stop_at_layer=layer + 1,
                names_filter=None,
            )
            X = cache["resid_pre", layer].cpu()  # Shape: (batch, pos, d_model)
            X = einops.rearrange(X, "batch pos d_model -> (batch pos) d_model")
            del cache
            cur_scores = sae.encoder(X)[:, feature_indices]
            cur_scores_reshaped = einops.rearrange(cur_scores, "(b pos) n -> b n pos", pos=owt_tokens_torch.shape[1]).cpu().numpy().astype(np.float16)
            scores.append(cur_scores_reshaped)

    scores = np.concatenate(scores, axis=0)
    np.save(SCORES_PATH, scores)
    return scores

def get_top_k_indices(scores: np.ndarray, feature_index: int, k: int = TOP_K) -> np.ndarray:
    """ 
    Get the indices of the examples where the feature activates the most
    scores is shape (batch, feature, pos), so we index with feature_index
    """
    feature_scores = scores[:, feature_index, :]
    top_k_indices = feature_scores.argsort()[-k:][::-1]
    return top_k_indices

def get_topk_bottomk_logits(feature_index: int, sae: SparseAutoencoder, transformer_model: HookedTransformer, k: int = TOP_K) -> tuple:
    feature_vector = sae.decoder.weight.data[:, feature_index]
    W_U = transformer_model.W_U  # (d_model, vocab)
    logits = einops.einsum(W_U, feature_vector, "d_model vocab, d_model -> vocab")
    top_k_logits = logits.topk(k).indices
    bottom_k_logits = logits.topk(k, largest=False).indices
    top_k_tokens = [transformer_model.to_string(x.item()) for x in top_k_logits]
    bottom_k_tokens = [transformer_model.to_string(x.item()) for x in bottom_k_logits]
    return top_k_tokens, bottom_k_tokens

def highlight_scores_in_html(token_strs: list, scores: list, seq_idx: int, max_color: str = "#ff8c00", zero_color: str = "#ffffff", show_score: bool = True) -> tuple:
    if len(token_strs) != len(scores):
        print(f"Length mismatch between tokens and scores (len(tokens)={len(token_strs)}, len(scores)={len(scores)})") 
        return "", ""
    scores_min = min(scores)
    scores_max = max(scores)
    if scores_max - scores_min == 0:
        scores_normalized = np.zeros_like(scores)
    else:
        scores_normalized = (np.array(scores) - scores_min) / (scores_max - scores_min)
    max_color_vec = np.array(
        [int(max_color[1:3], 16), int(max_color[3:5], 16), int(max_color[5:7], 16)]
    )
    zero_color_vec = np.array(
        [int(zero_color[1:3], 16), int(zero_color[3:5], 16), int(zero_color[5:7], 16)]
    )
    color_vecs = np.einsum("i, j -> ij", scores_normalized, max_color_vec) + np.einsum(
        "i, j -> ij", 1 - scores_normalized, zero_color_vec
    )
    color_strs = [f"#{int(x[0]):02x}{int(x[1]):02x}{int(x[2]):02x}" for x in color_vecs]
    if show_score:
        tokens_html = "".join(
            [
                f"""<span class='token' style='background-color: {color_strs[i]}'>{html.escape(token_str)}<span class='feature_val'> ({scores[i]:.2f})</span></span>"""
                for i, token_str in enumerate(token_strs)
            ]
        )
        clean_text = " | ".join(
            [f"{token_str} ({scores[i]:.2f})" for i, token_str in enumerate(token_strs)]
        )
    else:
        tokens_html = "".join(
            [
                f"""<span class='token' style='background-color: {color_strs[i]}'>{html.escape(token_str)}</span>"""
                for i, token_str in enumerate(token_strs)
            ]
        )
        clean_text = " | ".join(token_strs)
    head = """
    <style>
        span.token {
            font-family: monospace;
            border-style: solid;
            border-width: 1px;
            border-color: #dddddd;
        }
        span.feature_val {
            font-size: smaller;
            color: #555555;
        }
    </style>
    """
    return head + tokens_html, convert_clean_text(clean_text)

def convert_clean_text(clean_text: str, k: int = 1, tokens_left: int = 30, tokens_right: int = 5) -> str:
    # Split the clean text on the "|" separator
    token_score_pairs = clean_text.split(" | ")

    # Remove the first token if present
    if token_score_pairs:
        token_score_pairs = token_score_pairs[1:]

    # Initialize a list to hold tuples of (token, score)
    tokens_with_scores = []

    # Define regex to capture tokens with scores
    token_score_pattern = re.compile(r"^(.+?) \((\d+\.\d+)\)$")

    for token_score in token_score_pairs:
        match = token_score_pattern.match(token_score.strip())
        if match:
            token = match.group(1)
            score = float(match.group(2))
            tokens_with_scores.append((token, score))
        else:
            # Handle cases where score is zero or absent
            token = token_score.split(' (')[0].strip()
            tokens_with_scores.append((token, 0.0))

    # Sort tokens by score in descending order
    sorted_tokens = sorted(tokens_with_scores, key=lambda x: x[1], reverse=True)

    # Select top k tokens with non-zero scores
    top_k_tokens = [token for token, score in sorted_tokens if score > 0][:k]

    # Find all indices of top k tokens
    top_k_indices = [i for i, (token, score) in enumerate(tokens_with_scores) if token in top_k_tokens and score >0]

    # Define windows around each top token
    windows = []
    for idx in top_k_indices:
        start = max(0, idx - tokens_left)
        end = min(len(tokens_with_scores) - 1, idx + tokens_right)
        windows.append((start, end))

    # Merge overlapping windows
    merged_windows = []
    for window in sorted(windows, key=lambda x: x[0]):
        if not merged_windows:
            merged_windows.append(window)
        else:
            last_start, last_end = merged_windows[-1]
            current_start, current_end = window
            if current_start <= last_end + 1:
                # Overlapping or adjacent windows, merge them
                merged_windows[-1] = (last_start, max(last_end, current_end))
            else:
                merged_windows.append(window)

    # Collect all unique indices within the merged windows
    selected_indices = set()
    for start, end in merged_windows:
        selected_indices.update(range(start, end + 1))

    # Create the converted tokens list with wrapping
    converted_tokens = []
    for i, (token, score) in enumerate(tokens_with_scores):
        if i in selected_indices:
            if token in top_k_tokens and score > 0:
                token = f"<<{token}>>"
            converted_tokens.append(token)
        # Else, skip tokens outside the selected windows

    # Join the converted tokens into a single string
    converted_text = " ".join(converted_tokens)
    return converted_text

def format_interpreter_prompt(clean_text: str, top_k_tokens: list, bottom_k_tokens: list) -> str:
    system_prompt = """### SYSTEM PROMPT ###

You are a meticulous AI researcher conducting an important investigation into a certain neuron in a language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior.
Guidelines:

You will be given a list of text examples on which the neuron activates. The specific tokens which cause the neuron to activate will appear between delimiters like <<this>>. If a sequence of consecutive tokens all cause the neuron to activate, the entire sequence of tokens will be contained between delimiters <<just like this>>. The activation value of the example is listed after each example in parentheses.

- Try to produce a concise final description. Simply describe the text features that activate the neuron, and what its role might be based on the tokens it predicts.
- If either the text features or the predicted tokens are completely uninformative, you don't need to mention them.
- The last line of your response must be the formatted explanation."""

    cot_prompt = """
(Part 1) Tokens that the neuron activates highly on in text

Step 1: List a couple activating and contextual tokens you find interesting. Search for patterns in these tokens, if there are any. Don't list more than 5 tokens.
Step 2: Write down general shared features of the text examples.
"""

    activations_section = """
(Part 1) Tokens that the neuron activates highly on in text

Step 1: List a couple activating and contextual tokens you find interesting. Search for patterns in these tokens, if there are any.
Step 2: Write down several general shared features of the text examples.
Step 3: Take note of the activation values to understand which examples are most representative of the neuron.
"""

    logits_section = """
(Part 2) Tokens that the neuron boosts in the next token prediction

You will also be shown a list called Top_logits. The logits promoted by the neuron shed light on how the neuron's activation influences the model's predictions or outputs. Look at this list of Top_logits and refine your hypotheses from part 1. It is possible that this list is more informative than the examples from part 1.

Pay close attention to the words in this list and write down what they have in common. Then look at what they have in common, as well as patterns in the tokens you found in Part 1, to produce a single explanation for what features of text cause the neuron to activate. Propose your explanation in the following format:
[EXPLANATION]: <your explanation>
"""

    # Define the Examples and Their Responses (can be customized or loaded from a file)
    examples = """
### EXAMPLE STEP-BY-STEP WALKTHROUGH 1 ###
Example 1:  and he was <<over the moon>> to find
Example 2:  we'll be laughing <<till the cows come home>>! Pro
Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd

Top_logits: ["elated", "joyful", "story", "thrilled", "spider"]

(Part 1)
ACTIVATING TOKENS: "over the moon", "than meets the eye".
PREVIOUS TOKENS: No interesting patterns.

Step 1.
The activating tokens are all parts of common idioms.
The previous tokens have nothing in common.

Step 2.
- The examples contain common idioms.
- In some examples, the activating tokens are followed by an exclamation mark.

Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities?
- Yes, I missed one: The text examples all convey positive sentiment.

(Part 2)
SIMILAR TOKENS: "elated", "joyful", "thrilled".
- The top logits list contains words that are strongly associated with positive emotions.

[EXPLANATION]: Common idioms in text conveying positive sentiment.

### EXAMPLE STEP-BY-STEP WALKTHROUGH 2 ###

Example 1:  a river is wide but the ocean is wid<<er>>. The ocean
Example 2:  every year you get tall<<er>>," she
Example 3:  the hole was small<<er>> but deep<<er>> than the

Top_logits: ["apple", "running", "book", "wider", "quickly"]

(Part 1)
ACTIVATING TOKENS: "er", "er", "er".
PREVIOUS TOKENS: "wid", "tall", "small", "deep".

Step 1.
- The activating tokens are mostly "er".
- The previous tokens are mostly adjectives, or parts of adjectives, describing size.
- The neuron seems to activate on, or near, the tokens in comparative adjectives describing size.

Step 2.
- In each example, the activating token appeared at the end of a comparative adjective.
- The comparative adjectives ("wider", "tallish", "smaller", "deeper") all describe size.

Let me look again for patterns in the examples. Are there any links or hidden linguistic commonalities that I missed?
- I can't see any.

(Part 2)
SIMILAR TOKENS: None
- The top logits list contains unrelated nouns and adverbs.

[EXPLANATION]: The token "er" at the end of a comparative adjective describing size.

### EXAMPLE STEP-BY-STEP WALKTHROUGH 3 ###

Example 1:  something happening inside my <<house>>", he
Example 2:  presumably was always contained in <<a box>>", according
Example 3:  people were coming into the <<smoking area>>".

However he
Example 4:  Patrick: "why are you getting in the << way?>>" Later,

Top_logits: ["room", "end", "container", "space", "plane"]

(Part 1)
ACTIVATING TOKENS: "house", "a box", "smoking area", "way?".
PREVIOUS TOKENS: No interesting patterns.

Step 1.
- The activating tokens are all things that one can be in.
- The previous tokens have nothing in common.

Step 2.
- The examples involve being inside something, sometimes figuratively.
- The activating token is a thing which something else is inside of.

Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities?
- Yes, I missed one: The activating token is followed by a quotation mark, suggesting it occurs within speech.

(Part 2)
SIMILAR TOKENS: "room", "container", "space".
- The top logits list suggests a focus on nouns representing physical or metaphorical spaces.

[EXPLANATION]: Nouns preceding a quotation mark, representing distinct objects that contain something.
"""

    # Combine all predefined sections
    predefined_prompt = f"""{system_prompt}

{cot_prompt}

{logits_section}

We will now provide three step-by-step examples of how you should approach this.

{examples}
"""

    # Prepare the current data sections
    # Format the top_k_tokens and bottom_k_tokens as JSON-like lists
    top_logits_str = "[" + ", ".join(f'"{token}"' for token in top_k_tokens) + "]"
    bottom_logits_str = "[" + ", ".join(f'"{token}"' for token in bottom_k_tokens) + "]"

    current_activations = f"""
(Part 1) Tokens that the neuron activates highly on in text

{clean_text}
"""

    current_logits = f"""
(Part 2) Tokens that the neuron boosts in the next token prediction

Top_logits: {top_logits_str}
Bottom_logits: {bottom_logits_str}
"""

    # Combine all parts into the final prompt
    full_prompt = f"""{predefined_prompt}

### OUR NEURON WE NEED TO INTERPRET STEP-BY-STEP ###

{current_activations}

{current_logits}

Walk through the steps to interpret this neuron.
"""
    return full_prompt

def get_ai_response(formatted_prompt: str, config: dict) -> str:
    azure_client = AzureOpenAI(
        azure_endpoint=config["base_url"],
        api_key=config["azure_api_key"],
        api_version=config["api_version"],
    )
    
    # If prompt is a str
    if type(formatted_prompt) == str:
        messages = [{"role": "user", "content": formatted_prompt}]
    elif type(formatted_prompt) == list:
        messages = formatted_prompt
    response = azure_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    return response.choices[0].message.content

# ----------------------------- Main Processing Function -----------------------------

def analyze_feature(feature_index: int, sae: SparseAutoencoder, transformer_model: HookedTransformer, 
                   owt_tokens_torch: torch.Tensor, scores: np.ndarray, top_k_indices, top_k_batch_indices, top_k_tokens, top_k_tokens_str, top_k_scores_per_seq,
                   config: dict, 
                   top_k: int = TOP_K, bottom_k: int = BOTTOM_K) -> None:
    """ 
    Returns formatted_prompt, analysis, interp_text, scoring_text, false_text
    """

    
    # Get top-k and bottom-k boosted logits
    top_logits, bottom_logits = get_topk_bottomk_logits(feature_indices[feature_index], sae, transformer_model, k=top_k)
    
    # Highlight scores in HTML and prepare clean text
    examples_html = []
    examples_clean_text = []
    examples_false_text = []

    # False feature index
    false_feature_idx = 0 if feature_index != 0 else 15

    # Single-feature slice: (batch, seq_len)
    single_feature_scores_false = scores[:, false_feature_idx, :]  

    # Flatten, pick top k
    flat_scores_false = single_feature_scores_false.flatten()
    top_k_indices_false = flat_scores_false.argsort()[-top_k:][::-1]
    top_k_batch_indices_false, top_k_seq_indices_false = np.unravel_index(
        top_k_indices_false, single_feature_scores_false.shape
    )
    top_k_tokens_false = [owt_tokens_torch[b].tolist() for b in top_k_batch_indices_false]
    top_k_tokens_str_false = [
        [transformer_model.to_string(tok_id) for tok_id in seq]
        for seq in top_k_tokens_false
    ]
    top_k_scores_per_seq_false = [scores[b] for b in top_k_batch_indices_false]

    for i in range(top_k):
        try:
            example_html, clean_text = highlight_scores_in_html(
                top_k_tokens_str[i],
                top_k_scores_per_seq[i][feature_index],
                seq_idx=i,
                show_score=True
            )
            _, false_text = highlight_scores_in_html(
                top_k_tokens_str_false[i],
                top_k_scores_per_seq_false[i][false_feature_idx],
                seq_idx=i,
                show_score=True
            )
            examples_html.append(example_html)
            if len(clean_text) > 10:
                #print(f'Length of clean text: {len(clean_text)}')
                examples_clean_text.append(clean_text)
                examples_false_text.append(false_text)
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue


    # If length of examples_clean_text is less than top_k, raise an exception
    if len(examples_clean_text) < 4:
        raise Exception(f"Not enough activating examples (number of examples: {len(examples_clean_text)})")

    
    # Combine clean texts
    length_examples = len(examples_clean_text) // 2
    examples_to_insert = [f"Example {i+1}: {example}" for i, example in enumerate(examples_clean_text)][:length_examples]
    combined_clean_text = "\n\n".join(examples_to_insert).strip()

    if len(combined_clean_text) < 100:
        raise Exception("No activating examples for this one")
    
    # Format the prompt
    formatted_prompt = format_interpreter_prompt(combined_clean_text, top_logits, bottom_logits)
    
    # Get the AI response
    analysis = get_ai_response(formatted_prompt, config)

    interp_text = examples_clean_text[:length_examples]
    scoring_text = examples_clean_text[length_examples:length_examples*2]
    false_text = examples_false_text[:length_examples]

    return formatted_prompt, analysis, interp_text, scoring_text, false_text