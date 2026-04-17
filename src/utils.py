import torch
import sys
import os
from tqdm import tqdm
from prepare_data import prepare_data
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
import numpy as np
import datasets

def match_state_dict(
        state_dict_a,
        state_dict_b
):
    """
    Filter `state_dict_b` to keep only keys present in `state_dict_a` with matching shapes.

    Matching happens according to two criteria:
        - Is the key present in state_dict_a?
        - Does the state with the same key in state_dict_a have the same shape?

    Returns
        (matched_state_dict, unmatched_state_dict)

        States in matched_state_dict contains states from state_dict_b that are also
        in state_dict_a and unmatched_state_dict contains states that have no
        corresponding state in state_dict_a.

        In addition: state_dict_b = matched_state_dict U unmatched_state_dict.
    """
    matched_state_dict = {
        key: state
        for (key, state) in state_dict_b.items()
        if key in state_dict_a and state.shape == state_dict_a[key].shape
    }
    unmatched_state_dict = {
        key: state
        for (key, state) in state_dict_b.items()
        if key not in matched_state_dict
    }
    return matched_state_dict, unmatched_state_dict


def compute_num_basis(nx, nf, compression_ratio, group, strategy="default", similarity=None, minimum_basis_ratio=0.3):
    """
    Compute the number of basis vectors per group under a compression target.

    This function converts a percentage `compression_ratio` into a 0–1 ratio and dispatches to a specific allocation strategy. 
    - 'similarity' (or None): allocate more basis to groups with lower intra-group similarity

    Parameters:
        nx: Input feature size of the linear weight being factorized.
        nf: Output feature size per layer (columns per layer block).
        compression_ratio: Percentage compression (0–100). Higher means
            fewer parameters retained.
        group: Layer index groups sharing a basis.
        strategy: Allocation strategy. 
        similarity: Precomputed similarity matrix between layers; required for 'similarity' strategy.
        minimum_basis_ratio: Lower bound of the basis per group before allocation by strategy.

    Returns:
        num_basis: Number of basis allocated per group.
    """
    compression_ratio = 1 - compression_ratio / 100
    if strategy == 'similarity':
        # Default and current behavior: similarity-weighted allocation
        num_basis = similarity_based_num_basis(
            nx, nf, compression_ratio, group, similarity, minimum_basis_ratio=minimum_basis_ratio
        )
    elif strategy == 'default':
        group_size = len(group[0])
        total = nx * nf * group_size
        num_basis = (total * compression_ratio) // (nx + nf * group_size)
        return int(num_basis)
    return num_basis

def similarity_based_num_basis(nx, nf, compression_ratio, group, similarity, minimum_basis_ratio=0.3):
    """
    Allocate basis counts across groups weighted by inverse similarity.

    Parameters:
        nx: Input feature size.
        nf: Output feature size.
        compression_ratio: Ratio in [0, 1].
        group: information of grouped layers.
        similarity: Similarity matrix between layers.
        minimum_basis_ratio: Minimum ratio of original basis.

    Returns:
        total_basis: Number of basis(rank) per group after allocation.
    """
    print("nx, nf, compression_ratio", nx, nf, compression_ratio)
    print("group", group)
    g_av_similarity = []
    for g in group:
        g_similarities = []
        for i in range(0,len(g)):
            if len(g) > 1:
                for j in range(i+1,len(g)):
                    g_similarities.append(similarity[g[i], g[j]].detach().cpu().item())
            elif len(g) == 1:
                g_similarities = -1
        g_av_similarity.append(float(np.array(g_similarities).mean()))
    print(f"g_av_similarity: {g_av_similarity}")
    total_layers = sum(len(g) for g in group)
    total_original_params = nx * nf * total_layers

    original_basis_arr = np.array([])
    for g in group:
        group_size = len(g)
        original_basis = int((nx * nf * group_size * compression_ratio) // (nx + nf * group_size))
        original_basis_arr = np.append(original_basis_arr, original_basis)
    min_basis_arr = minimum_basis_ratio * original_basis_arr
    print(f"min_basis: {min_basis_arr}")
    
    # Calculate total minimum parameters
    total_min_params = sum(min_basis_arr[i] * (nx + nf * len(group[i])) for i in range(len(group)))
    print(f"total_min_params: {total_min_params}")

    # Calculate remaining parameters to distribute
    target_total_params = int(total_original_params * compression_ratio)
    remaining_params = target_total_params - total_min_params

    if remaining_params <= 0:
        print("remaining_params <= 0, return original basis")
        return original_basis_arr.tolist()
        
    # Calculate inverse similarity weights (lower similarity = higher weight): Add small epsilon to avoid division by zero
    epsilon = 1e-8
    inverse_similarity = [1.0 / (sim + epsilon) if sim >= 0 else -1 for sim in g_av_similarity]
    
    # Normalize weights to sum to 1
    total_weight = sum(inverse_similarity) + inverse_similarity.count(-1)
    weights = [w / total_weight if w >= 0 else -1 for w in inverse_similarity]
    
    # Calculate additional basis for each group based on remaining parameters
    additional_basis_arr = np.array([])
    for i, g in enumerate(group):
        # Additional parameters for this group
        if weights[i] < 0:
            additional_basis_arr = np.append(additional_basis_arr, 0) 
        else:
            additional_params = int(remaining_params * weights[i])
            group_layers = len(g)
            additional_basis = additional_params // (nx + nf * group_layers)
            additional_basis_arr = np.append(additional_basis_arr, additional_basis)
    print(f"additional_basis: {additional_basis_arr}")
    total_basis = min_basis_arr + additional_basis_arr
    total_basis = total_basis.astype(int)
    print(f"total_basis: {total_basis}")
    
    return total_basis.tolist()
    
def compute_ppl(max_length, stride, data, model, device):
    """
    Compute perplexity

    Parameters:
        max_length: sequence length for evaluation.
        stride: sliding window stride.
        data: Tokenized dataset with input_ids tensor.
        model: Model to evaluate.
        device: Device for computation.

    Returns:
        ppl: Perplexity value.
    """
    model.to(device)
    model = model.eval()
    seq_len = data.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = data.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            output = model(input_ids, labels=target_ids)

            neg_log_likelihood = output.loss
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    try:
        return ppl.item()
    except:
        return ppl


def _hflm_from_objects(model, tokenizer, **kwargs):
    """
    Wrap a HF model/tokenizer pair into an lm-eval HFLM instance, saving to disk if needed.

    Parameters:
        model: HF model instance.
        tokenizer: HF tokenizer instance.
        **kwargs: Extra arguments forwarded to HFLM.

    Returns:
        HFLM: Language model wrapper for evaluation tasks.
    """
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        if getattr(model, "config", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    try:
        return HFLM(pretrained=model, tokenizer=tokenizer, **kwargs)
    except (AssertionError, TypeError):
        pass

    import tempfile
    tmp = tempfile.mkdtemp(prefix="lmeval_")
    model.save_pretrained(tmp, safe_serialization=True)
    tokenizer.save_pretrained(tmp)
    return HFLM(pretrained=tmp, tokenizer=tmp, **kwargs)


def run_lm_eval(model, tokenizer, device, tasks='arc_easy,piqa,arc_challenge,openbookqa,wsc,copa,winogrande,lambada', limit=None, **kwargs):
    """
    Run lm-eval harness over specified tasks and return accuracy-like metrics.

    Parameters:
        model: Model or path passed to HFLM.
        tokenizer: Tokenizer used by the model.
        device: Target device for evaluation.
        tasks: Comma-separated task list.
        limit: Optional sample limit.
        **kwargs: Extra args for HFLM construction.

    Returns:
        result: Mapping task name to score (acc or acc_norm).
    """
    tasks = tasks.split(',')
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    lm = _hflm_from_objects(
        model,
        tokenizer,
        batch_size=32,
        max_batch_size=128,
        device=device,
        trust_remote_code=True,
    )
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        res = simple_evaluate(model=lm, tasks=tasks, batch_size=32)
    finally:
        sys.stdout = saved_out
        sys.stderr = saved_err
    res = res['results'] if isinstance(res, dict) and 'results' in res else res
    result = {}
    for task_name, task_data in res.items():
        result[task_name] = task_data['acc,norm'] if ('acc,norm' in task_data) else task_data['acc,none']
    return result


def eval_model(tokenizer, max_length, stride, model, device, dataset_cache_dir, ppl_only=False):
    """
    Evaluate perplexity on Wikitext and C4 and optionally run lm-eval tasks.

    Parameters:
        tokenizer: HF tokenizer instance.
        max_length: sequence length used for perplexity.
        stride: sliding window stride.
        model: Model to evaluate.
        device: Device for inference.
        dataset_cache_dir: HF datasets cache directory override.
        ppl_only: If True, skip lm-eval and only compute perplexities.

    Returns:
        result: Task scores if `ppl_only` is False; otherwise None.
    """
    ppl = {'wikitext': 0, 'c4': 0}
    for data in ppl:
        _, _, test_dataset, _ = prepare_data(data, tokenizer, max_length, dataset_cache_dir)
        ppl[data] = compute_ppl(max_length, stride, test_dataset, model, device)
        print(ppl)

    if ppl_only:
        return

    result = run_lm_eval(model, tokenizer, device)
    print(result)
    return