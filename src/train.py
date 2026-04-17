import math
from transformers import AutoTokenizer
from transformers import set_seed
from model_factory import create_model
from config import ShareConfig, add_args
import torch
import torch.nn.functional as F
from utils import eval_model
import gc
import psutil
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def print_memory_usage():
    """
    Print current GPU and system memory usage for quick diagnostics.

    Parameters:
        None

    Returns:
        None
    """
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
    print(f"RAM usage: {psutil.virtual_memory().percent:.1f}%")

def clear_memory():
    """
    Clear Python and CUDA caches to reduce peak memory usage.

    Parameters:
        None

    Returns:
        None
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_train_loader(tokenizer, config): 
    """
    Construct a training DataLoader from a subset of the SlimPajama-6B dataset.

    Parameters:
        tokenizer: Tokenizer used for encoding text.
        config: Configuration containing dataset and dataloader settings.

    Returns:
        DataLoader that yields training batches.
    """
    train_ds = load_dataset("DKYoon/SlimPajama-6B", split="train[:5000]")
    texts = train_ds['text']
    input_ids_list = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]
        input_ids_list.append(enc)
    input_ids = torch.cat(input_ids_list, dim=1)  
    total_length = input_ids.size(1)
    trunc_length = total_length - (total_length % config.context_length)
    input_ids = input_ids[:, :trunc_length]
    input_ids = input_ids.view(-1, config.context_length)
    dataset_tensor = TensorDataset(input_ids)
    train_loader = DataLoader(
        dataset_tensor, 
        batch_size=config.train_batch_size, 
        shuffle=False,
        num_workers=config.train_num_workers, 
        pin_memory=bool(config.pin_memory) and torch.cuda.is_available(),  # Disabled to save memory
        persistent_workers=config.train_num_workers > 0
    )
    del input_ids_list, input_ids, total_length, trunc_length
    clear_memory()
    print(f"Dataset created with {len(dataset_tensor)} samples")
    print_memory_usage()
    return train_loader


def set_train_param_and_optimizer(model, lr):
    """
    Freeze all model parameters except two vector parameters (sigma and identity_vector) and build an optimizer.

    Parameters:
        model: SharVeT model.
        lr: Learning rate for trainable parameters.

    Returns:
        (trainable_params, optimizer): The parameters to optimize and an AdamW optimizer.
    """
    vera_keywords = ['sigma', 'identity_vector']
    for name, param in model.named_parameters():
        if any(k in name for k in vera_keywords):
            param.requires_grad = True
        else:
            param.requires_grad = False
            param.grad = None
    vera_params = [p for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable tensors after filtering: {len(vera_params)}")
    optimizer = torch.optim.AdamW(
        vera_params,
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    return vera_params, optimizer


def compute_distillation_loss(std_model, model, inputs, s_outputs):
    """
    Compute distillation losses between teacher and student.

    Parameters:
        std_model: Teacher model.
        model: Student model.
        inputs: Input ids.
        s_outputs: Forward outputs from student, including logits and hidden states.

    Returns:
        (kld_loss, feat_loss): KL divergence loss and feature matching loss.
    """
    device = next(model.parameters()).device
    input_ids = inputs
    s_logits = s_outputs.logits
    s_hiddens = s_outputs.hidden_states
    # Teacher (std_model) forward with autocast
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            t_outputs = std_model(input_ids=input_ids,
                            use_cache=False,
                            output_hidden_states=True,  
                            return_dict=True)
            t_logits = t_outputs.logits
            t_hiddens = t_outputs.hidden_states
    # 1. KL Divergence Loss (logit distillation)
    T = float(config.distill_temperature)
    s_log_prob = F.log_softmax(s_logits.float() / T, dim=-1)
    t_prob = F.softmax(t_logits.float() / T, dim=-1)
    kld_loss = (T * T) * F.kl_div(s_log_prob, t_prob, reduction='batchmean')
    # 2. Feature Loss (hidden states matching)
    feat_loss = torch.tensor(0.0, device=device)
    s_hiddens_tensor = torch.stack(s_hiddens)
    t_hiddens_tensor = torch.stack(t_hiddens)
    feat_loss = F.mse_loss(s_hiddens_tensor.float(), t_hiddens_tensor.float())
    return kld_loss, feat_loss 


def train(config):
    """
    Train the SharVeT model with optional distillation and regularization.

    Parameters:
        config: Configuration detailing model, data, training, and optimization settings.

    Returns:
        None
    """
    set_seed(2024)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if config.model_type == "llama":
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.pad_token = "[PAD]"

    model, std_model = create_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.float32)
    std_model.to(device, dtype=torch.float32)
    scaler = torch.amp.GradScaler('cuda') if config.fp16 else None
    train_loader = load_train_loader(tokenizer, config)
    vera_params, optimizer = set_train_param_and_optimizer(model, getattr(config, 'learning_rate', 0.0002))
    total_steps = len(train_loader) * config.train_epoch // config.gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0, 
        end_factor=0.8,  
        total_iters=total_steps
    )
    for epoch in range(config.train_epoch):
        print(f"Epoch {epoch+1}/{config.train_epoch}")
        progress_bar = tqdm(train_loader)
        optimizer.zero_grad()
        model.train()
        clear_memory()
        print_memory_usage()

        for batch_idx, batch in enumerate(progress_bar):
            batch_input_ids = batch[0].to(device)
            with torch.amp.autocast('cuda', enabled=bool(config.fp16)):
                outputs = model(batch_input_ids, labels=batch_input_ids, output_hidden_states=True)
                lm_loss = outputs.loss 
                kld_loss, feat_loss = compute_distillation_loss(std_model, model, batch_input_ids, outputs)
                kld_loss = config.distill_weight * kld_loss
                feat_loss = config.feature_weight * feat_loss
                l2_loss = 0.0
                for layer in model.model.layers:
                    if hasattr(layer, 'self_attn'):
                        for name, param in layer.self_attn.named_parameters():
                            if 'sigma' in name:
                                l2_loss += torch.norm(param, p=2)
                            if 'identity_vector' in name:
                                l2_loss += torch.norm(param, p=2)
                l2_loss = config.l2_weight * l2_loss
                total_loss = lm_loss + kld_loss + feat_loss + l2_loss
                
                loss = total_loss / config.gradient_accumulation_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(vera_params, config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(vera_params, config.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            current_lr = optimizer.param_groups[0]['lr']
            l2_val = float(l2_loss) if isinstance(l2_loss, (int, float)) else float(l2_loss.item())
            progress_bar.set_postfix({
                'total': f"{total_loss.item():.4f}",
                'lm': f"{lm_loss.item():.4f}",
                'kld': f"{kld_loss.item():.4f}",
                'feat': f"{feat_loss.item():.4f}",
                'l2': f"{l2_val:.4f}",
                'lr': f"{current_lr:.2e}",
                'ppl': f"{math.exp(outputs.loss.item()):.2f}",
            })
            if (batch_idx+1) % 5 == 0:
                print(
                    f"[iter {batch_idx+1}] total={total_loss.item():.4f}, "
                    f"lm={lm_loss.item():.4f}, kld={kld_loss.item():.4f}, "
                    f"feat={feat_loss.item():.4f}, l2={l2_val:.4f}, "
                    f"lr={current_lr:.2e}, ppl={math.exp(outputs.loss.item()):.2f}"
                )
            
        clear_memory()
        print_memory_usage()
        if (epoch + 1) % 2 == 0:
            print(f"Evaluating model at end of epoch {epoch+1}")
            model.eval()
            eval_model(tokenizer, config.context_length, config.stride, model, "cuda", config.dataset_cache_dir)
            model.train()
        
    print(f"Evaluating model at the end of training")
    eval_model(tokenizer, config.context_length, config.stride, model, "cuda", config.dataset_cache_dir)

if __name__ == '__main__':
    cmd_args = add_args()
    config = ShareConfig(cmd_args)
    print("================================================config.yaml_config_file: ", config.yaml_config_file,"================================================")
    train(config)