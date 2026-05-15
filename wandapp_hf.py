#!/usr/bin/env python3
"""
Wanda++-style pruning for Hugging Face causal LM decoder models.

Target models:
    - mistralai/Mistral-7B-Instruct-v0.3
    - mistralai/Mistral-7B-v0.1
    - meta-llama/Llama-*
    - Qwen/Qwen2*
    - similar AutoModelForCausalLM decoder-only models

Mathematics implemented from Wanda++:

    1. Regional Gradient Score:

        L_RGS^l(X) = || f_l(X) ||_2

        G_ij = sqrt( mean_n ( d L_RGS^l(X_n) / d W_ij )^2 )

        S_ij = (alpha * G_ij + ||X_j||_2) * |W_ij|

    2. Regional Optimization:

        L_RO = || f_l(X_ro) - f_hat_l(X_ro) ||_2^2

       where f_l is the dense decoder block and f_hat_l is the pruned block.

Important:
    - This is pruning only.
    - It does not perform GPTQ quantization.
    - It saves compact sparse layer data:
          mask packed as bits
          surviving values as flattened tensor
    - It can also save a dense masked state_dict for easier evaluation.
    - For real storage compression, use:
          --save_sparse_values
          and leave --keep_dense_state_dict OFF.

Recommended first Mistral test:
    2:4 pruning:
        --pattern 2:4
        --sparsity 0.5

    unstructured 50%:
        --pattern unstructured
        --sparsity 0.5

Notes:
    - For exact per-sample regional gradient aggregation from the paper,
      use --rgs_batch_size 1.
    - RO is expensive. Start with --ro_iters 1 or 2.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# ============================================================
# Basic helpers
# ============================================================

def now() -> float:
    return time.time()


def format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "?"
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float32", "fp32"):
        return torch.float32
    if name in ("float64", "fp64", "double"):
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def cuda_memory_string() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB"


def get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_module_by_name(root: nn.Module, full_name: str) -> nn.Module:
    obj = root
    for part in full_name.split("."):
        obj = getattr(obj, part)
    return obj


def set_module_by_name(root: nn.Module, full_name: str, new_module: nn.Module) -> None:
    parts = full_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def strip_prefix(name: str, prefix: str) -> str:
    if name == prefix:
        return ""
    if name.startswith(prefix + "."):
        return name[len(prefix) + 1:]
    return name


# ============================================================
# Calibration loading
# ============================================================

def load_calibration_tokens(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and "tokens" in obj:
        tokens = obj["tokens"]
    elif torch.is_tensor(obj):
        tokens = obj
    else:
        raise ValueError("Calibration file must be a tensor or dict with key 'tokens'.")

    if tokens.ndim != 2:
        raise ValueError(f"Expected calibration tokens [N,T], got {tuple(tokens.shape)}")

    return tokens.long()


# ============================================================
# Mask packing
# ============================================================

def pack_bool_mask_rows(mask: torch.Tensor) -> torch.Tensor:
    """
    Pack bool mask [rows, cols] into uint8 [rows, ceil(cols / 8)].

    True  = keep
    False = prune
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    rows, cols = mask.shape
    packed_cols = (cols + 7) // 8
    padded_cols = packed_cols * 8

    if padded_cols != cols:
        pad = torch.zeros(
            (rows, padded_cols - cols),
            dtype=torch.bool,
            device=mask.device,
        )
        mask = torch.cat([mask, pad], dim=1)

    mask_u8 = mask.to(torch.uint8).view(rows, packed_cols, 8)
    shifts = torch.tensor(
        [1, 2, 4, 8, 16, 32, 64, 128],
        dtype=torch.uint8,
        device=mask.device,
    )

    return (mask_u8 * shifts.view(1, 1, 8)).sum(dim=2).to(torch.uint8)


def unpack_bool_mask_rows(packed: torch.Tensor, original_cols: int) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        packed = packed.to(torch.uint8)

    rows, packed_cols = packed.shape
    shifts = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.uint8,
        device=packed.device,
    )

    bits = ((packed.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 1).bool()
    mask = bits.view(rows, packed_cols * 8)
    return mask[:, :original_cols]


# ============================================================
# Sparsity pattern helpers
# ============================================================

def parse_nm_pattern(pattern: str) -> Optional[Tuple[int, int]]:
    pattern = pattern.strip().lower()
    if pattern in ("", "none", "unstructured"):
        return None

    if ":" not in pattern:
        raise ValueError("Pattern must be 'unstructured' or N:M, e.g. '2:4', '4:8'.")

    n_str, m_str = pattern.split(":")
    n = int(n_str)
    m = int(m_str)

    if n < 0 or m <= 0 or n > m:
        raise ValueError(f"Invalid N:M pattern: {pattern}")

    return n, m


@torch.no_grad()
def select_unstructured_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Select global unstructured mask for one matrix.

    Larger score = more important = keep.
    """
    if sparsity <= 0:
        return torch.ones_like(score, dtype=torch.bool)

    if sparsity >= 1:
        return torch.zeros_like(score, dtype=torch.bool)

    rows, cols = score.shape
    total = rows * cols
    n_prune = int(round(sparsity * total))
    n_keep = total - n_prune

    if n_prune <= 0:
        return torch.ones_like(score, dtype=torch.bool)

    if n_keep <= 0:
        return torch.zeros_like(score, dtype=torch.bool)

    flat = score.reshape(-1)
    keep_idx = torch.topk(flat, k=n_keep, largest=True, sorted=False).indices

    mask_flat = torch.zeros(total, dtype=torch.bool, device=score.device)
    mask_flat[keep_idx] = True

    return mask_flat.view(rows, cols)


@torch.no_grad()
def select_nm_mask(score: torch.Tensor, n_zero: int, m: int) -> torch.Tensor:
    """
    N:M pruning.

    For pattern 2:4:
        n_zero=2, m=4
        prune 2 smallest scores in every group of 4 columns per row.
    """
    rows, cols = score.shape
    mask = torch.ones_like(score, dtype=torch.bool)

    for g0 in range(0, cols, m):
        g1 = min(g0 + m, cols)
        group_cols = g1 - g0

        if group_cols == m:
            prune_count = n_zero
        else:
            prune_count = int(round((n_zero / float(m)) * group_cols))

        prune_count = max(0, min(prune_count, group_cols))

        if prune_count == 0:
            continue

        if prune_count == group_cols:
            mask[:, g0:g1] = False
            continue

        local_score = score[:, g0:g1]
        prune_idx = torch.topk(
            local_score,
            k=prune_count,
            largest=False,
            dim=1,
            sorted=False,
        ).indices

        row_idx = torch.arange(rows, device=score.device).view(-1, 1).expand_as(prune_idx)
        local_mask = mask[:, g0:g1]
        local_mask[row_idx, prune_idx] = False
        mask[:, g0:g1] = local_mask

    return mask


# ============================================================
# HF decoder block helpers
# ============================================================

def get_decoder_and_layers(model: nn.Module) -> Tuple[nn.Module, nn.ModuleList, str]:
    """
    Supports common HF decoder-only architectures.

    Mistral/LLaMA/Qwen:
        model.model.layers
    GPT-like fallback:
        model.transformer.h
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model, model.model.layers, "model.layers"

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer, model.transformer.h, "transformer.h"

    raise RuntimeError("Could not find decoder layers. Expected model.model.layers or model.transformer.h.")


def get_embed_tokens(decoder: nn.Module, model: nn.Module) -> nn.Module:
    if hasattr(decoder, "embed_tokens"):
        return decoder.embed_tokens
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb
    if hasattr(decoder, "wte"):
        return decoder.wte
    raise RuntimeError("Could not find token embedding module.")


@torch.no_grad()
def compute_initial_hidden_cache(
    model: nn.Module,
    calib_tokens: torch.Tensor,
    batch_size: int,
    device: torch.device,
    hidden_cache_dtype: torch.dtype,
) -> torch.Tensor:
    decoder, _, _ = get_decoder_and_layers(model)
    embed_tokens = get_embed_tokens(decoder, model)

    n = calib_tokens.size(0)
    outs: List[torch.Tensor] = []

    iterator = range(0, n, batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, total=math.ceil(n / batch_size), desc="embedding cache", unit="batch")

    for i in iterator:
        input_ids = calib_tokens[i:i + batch_size].to(device)
        hidden = embed_tokens(input_ids)
        outs.append(hidden.detach().to("cpu", dtype=hidden_cache_dtype))

    return torch.cat(outs, dim=0)


def make_position_ids(batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, device=device, dtype=torch.long).view(1, -1).expand(batch, -1)


def make_cache_position(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, device=device, dtype=torch.long)


def make_causal_mask(
    batch: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Manual 4D causal mask:
        shape [batch, 1, seq_len, seq_len]

    0 for allowed positions, very negative for masked future positions.
    """
    min_val = torch.finfo(dtype).min if dtype.is_floating_point else -1e9

    mask = torch.full(
        (seq_len, seq_len),
        fill_value=min_val,
        device=device,
        dtype=dtype,
    )
    mask = torch.triu(mask, diagonal=1)
    mask = mask.view(1, 1, seq_len, seq_len).expand(batch, 1, seq_len, seq_len)
    return mask


def get_position_embeddings(
    decoder: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not hasattr(decoder, "rotary_emb"):
        return None

    rotary = decoder.rotary_emb

    try:
        return rotary(hidden_states, position_ids)
    except TypeError:
        try:
            return rotary(hidden_states, position_ids=position_ids)
        except TypeError:
            return None


def block_main_dtype(block: nn.Module) -> torch.dtype:
    for p in block.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float16


@torch.no_grad()
def forward_decoder_block_no_grad(
    model: nn.Module,
    block: nn.Module,
    hidden: torch.Tensor,
    use_causal_mask: bool = True,
) -> torch.Tensor:
    return forward_decoder_block(
        model=model,
        block=block,
        hidden=hidden,
        use_causal_mask=use_causal_mask,
        grad_enabled=False,
    )


def forward_decoder_block(
    model: nn.Module,
    block: nn.Module,
    hidden: torch.Tensor,
    use_causal_mask: bool = True,
    grad_enabled: bool = False,
) -> torch.Tensor:
    """
    Robust block forward for recent Mistral/LLaMA/Qwen Transformers versions.

    Critical for recent Mistral:
        block.forward expects position_embeddings=(cos,sin).
    """
    decoder, _, _ = get_decoder_and_layers(model)

    device = hidden.device
    bsz, seq_len, _ = hidden.shape
    dtype = block_main_dtype(block)

    hidden = hidden.to(device=device, dtype=dtype)

    position_ids = make_position_ids(bsz, seq_len, device)
    cache_position = make_cache_position(seq_len, device)
    position_embeddings = get_position_embeddings(decoder, hidden, position_ids)

    attention_mask = None
    if use_causal_mask:
        attention_mask = make_causal_mask(
            batch=bsz,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )

    sig = inspect.signature(block.forward)
    kwargs: Dict[str, Any] = {}

    if "hidden_states" in sig.parameters:
        kwargs["hidden_states"] = hidden

    if "attention_mask" in sig.parameters:
        kwargs["attention_mask"] = attention_mask

    if "position_ids" in sig.parameters:
        kwargs["position_ids"] = position_ids

    if "cache_position" in sig.parameters:
        kwargs["cache_position"] = cache_position

    if "position_embeddings" in sig.parameters and position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings

    if "output_attentions" in sig.parameters:
        kwargs["output_attentions"] = False

    if "use_cache" in sig.parameters:
        kwargs["use_cache"] = False

    if "past_key_value" in sig.parameters:
        kwargs["past_key_value"] = None

    if grad_enabled:
        out = block(**kwargs)
    else:
        with torch.no_grad():
            out = block(**kwargs)

    if isinstance(out, tuple):
        return out[0]

    if hasattr(out, "hidden_states"):
        return out.hidden_states

    return out


# ============================================================
# Linear layer selection
# ============================================================

def default_suffixes() -> Tuple[str, ...]:
    return (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def parse_suffixes(raw: str) -> Tuple[str, ...]:
    raw = raw.strip()
    if not raw:
        return default_suffixes()
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def should_compress_name(
    full_name: str,
    include: str,
    exclude: str,
    suffixes: Tuple[str, ...],
    compress_lm_head: bool,
    skip_attn_out: bool,
    skip_mlp_out: bool,
) -> bool:
    if include and include not in full_name:
        return False

    if exclude and exclude in full_name:
        return False

    if full_name == "lm_head":
        return compress_lm_head

    if skip_attn_out and full_name.endswith("o_proj"):
        return False

    if skip_mlp_out and full_name.endswith("down_proj"):
        return False

    return full_name.endswith(suffixes)


def find_block_linear_layers(
    block: nn.Module,
    block_prefix: str,
    include: str,
    exclude: str,
    suffixes: Tuple[str, ...],
    skip_attn_out: bool,
    skip_mlp_out: bool,
) -> List[Tuple[str, nn.Linear]]:
    out: List[Tuple[str, nn.Linear]] = []

    for local_name, mod in block.named_modules():
        if not isinstance(mod, nn.Linear):
            continue

        full_name = f"{block_prefix}.{local_name}" if local_name else block_prefix

        if should_compress_name(
            full_name=full_name,
            include=include,
            exclude=exclude,
            suffixes=suffixes,
            compress_lm_head=False,
            skip_attn_out=skip_attn_out,
            skip_mlp_out=skip_mlp_out,
        ):
            out.append((full_name, mod))

    return out


# ============================================================
# RGS collection
# ============================================================

@dataclass
class RGSStats:
    grad_mag: Dict[str, torch.Tensor]
    act_norm: Dict[str, torch.Tensor]
    grad_samples: int
    act_samples: int


class ActivationNormCollector:
    def __init__(self, layer: nn.Linear, device: torch.device):
        self.layer = layer
        self.device = device
        self.sum_sq = torch.zeros(layer.in_features, device=device, dtype=torch.float64)
        self.samples = 0
        self.handle = None

    def hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        x = inputs[0].detach()
        x = x.reshape(-1, x.shape[-1]).to(device=self.device, dtype=torch.float64)
        self.sum_sq += (x * x).sum(dim=0)
        self.samples += x.size(0)

    def register(self) -> None:
        self.handle = self.layer.register_forward_pre_hook(self.hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def norm(self) -> torch.Tensor:
        return torch.sqrt(self.sum_sq.clamp(min=0.0)).to(torch.float32)


@torch.no_grad()
def collect_activation_norms_only(
    model: nn.Module,
    block: nn.Module,
    block_layers: List[Tuple[str, nn.Linear]],
    hidden_cache: torch.Tensor,
    batch_size: int,
    main_device: torch.device,
    accum_device: torch.device,
    use_causal_mask: bool,
) -> Dict[str, torch.Tensor]:
    collectors: Dict[str, ActivationNormCollector] = {}

    for name, layer in block_layers:
        c = ActivationNormCollector(layer=layer, device=accum_device)
        c.register()
        collectors[name] = c

    n = hidden_cache.size(0)

    for i in range(0, n, batch_size):
        hidden = hidden_cache[i:i + batch_size].to(main_device)
        _ = forward_decoder_block_no_grad(
            model=model,
            block=block,
            hidden=hidden,
            use_causal_mask=use_causal_mask,
        )

    for c in collectors.values():
        c.remove()

    return {name: c.norm().cpu() for name, c in collectors.items()}


def collect_rgs_stats(
    model: nn.Module,
    block: nn.Module,
    block_layers: List[Tuple[str, nn.Linear]],
    hidden_cache: torch.Tensor,
    rgs_batch_size: int,
    main_device: torch.device,
    accum_device: torch.device,
    alpha: float,
    use_causal_mask: bool,
) -> RGSStats:
    """
    CRUCIAL:
        This computes Wanda++ regional gradients:

            L_RGS = || block(hidden) ||_2

        Then accumulates squared gradients per selected Linear weight.

    For exact paper-like stochastic gradient aggregation, use rgs_batch_size=1.
    """
    del alpha

    block.zero_grad(set_to_none=True)

    act_collectors: Dict[str, ActivationNormCollector] = {}
    for name, layer in block_layers:
        c = ActivationNormCollector(layer=layer, device=accum_device)
        c.register()
        act_collectors[name] = c

    grad_sq: Dict[str, torch.Tensor] = {}
    for name, layer in block_layers:
        grad_sq[name] = torch.zeros(
            tuple(layer.weight.shape),
            device=accum_device,
            dtype=torch.float32,
        )

    n = hidden_cache.size(0)
    grad_steps = 0

    iterator = range(0, n, rgs_batch_size)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=math.ceil(n / rgs_batch_size),
            desc="RGS backward",
            unit="batch",
            dynamic_ncols=True,
        )

    was_training = block.training
    block.eval()

    for i in iterator:
        hidden = hidden_cache[i:i + rgs_batch_size].to(main_device)

        block.zero_grad(set_to_none=True)

        with torch.enable_grad():
            out = forward_decoder_block(
                model=model,
                block=block,
                hidden=hidden,
                use_causal_mask=use_causal_mask,
                grad_enabled=True,
            )

            # CRUCIAL:
            # Paper defines regional loss as L2 norm of decoder-block output.
            loss = out.float().norm(p=2)

            loss.backward()

        for name, layer in block_layers:
            if layer.weight.grad is None:
                continue

            g = layer.weight.grad.detach()
            g = torch.nan_to_num(g.float(), nan=0.0, posinf=0.0, neginf=0.0)
            grad_sq[name] += (g * g).to(accum_device)

        grad_steps += 1

    for c in act_collectors.values():
        c.remove()

    block.zero_grad(set_to_none=True)

    if was_training:
        block.train()
    else:
        block.eval()

    grad_mag: Dict[str, torch.Tensor] = {}
    act_norm: Dict[str, torch.Tensor] = {}

    for name in grad_sq:
        grad_mag[name] = torch.sqrt(grad_sq[name] / max(grad_steps, 1)).cpu()

    for name, c in act_collectors.items():
        act_norm[name] = c.norm().cpu()

    return RGSStats(
        grad_mag=grad_mag,
        act_norm=act_norm,
        grad_samples=grad_steps,
        act_samples=sum(c.samples for c in act_collectors.values()),
    )


# ============================================================
# Pruning
# ============================================================

@torch.no_grad()
def compute_wandapp_score(
    weight: torch.Tensor,
    grad_mag: torch.Tensor,
    act_norm: torch.Tensor,
    alpha: float,
    score_dtype: torch.dtype,
) -> torch.Tensor:
    """
    CRUCIAL Wanda++ RGS score:

        S_ij = (alpha * G_ij + ||X_j||_2) * |W_ij|
    """
    device = weight.device

    Wabs = weight.detach().to(device=device, dtype=score_dtype).abs()
    G = grad_mag.to(device=device, dtype=score_dtype)
    X = act_norm.to(device=device, dtype=score_dtype).view(1, -1)

    score = Wabs * (float(alpha) * G + X)
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    return score


@torch.no_grad()
def apply_wandapp_pruning_to_block(
    block_layers: List[Tuple[str, nn.Linear]],
    rgs_stats: RGSStats,
    alpha: float,
    sparsity: float,
    pattern: str,
    score_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """
    Computes masks from Wanda++ score and applies them in-place.

    Returns:
        masks[name] on CPU, bool [out,in]
    """
    nm = parse_nm_pattern(pattern)
    masks: Dict[str, torch.Tensor] = {}

    for name, layer in block_layers:
        score = compute_wandapp_score(
            weight=layer.weight,
            grad_mag=rgs_stats.grad_mag[name],
            act_norm=rgs_stats.act_norm[name],
            alpha=alpha,
            score_dtype=score_dtype,
        )

        if nm is None:
            mask = select_unstructured_mask(score, sparsity=sparsity)
        else:
            n_zero, m = nm
            mask = select_nm_mask(score, n_zero=n_zero, m=m)

        layer.weight.data.mul_(mask.to(device=layer.weight.device, dtype=layer.weight.dtype))
        masks[name] = mask.detach().cpu()

        total = mask.numel()
        kept = int(mask.sum().item())
        actual_sparsity = 1.0 - kept / float(total)

        print(
            f"      pruned {name}: "
            f"shape={tuple(layer.weight.shape)} "
            f"sparsity={100.0 * actual_sparsity:.2f}%"
        )

        del score

    return masks


@torch.no_grad()
def enforce_masks(block_layers: List[Tuple[str, nn.Linear]], masks: Dict[str, torch.Tensor]) -> None:
    for name, layer in block_layers:
        if name not in masks:
            continue
        mask = masks[name].to(device=layer.weight.device, dtype=layer.weight.dtype)
        layer.weight.data.mul_(mask)


# ============================================================
# Regional optimization
# ============================================================

@torch.no_grad()
def compute_block_outputs_for_indices(
    model: nn.Module,
    block: nn.Module,
    hidden_cache: torch.Tensor,
    indices: List[int],
    batch_size: int,
    device: torch.device,
    target_dtype: torch.dtype,
    use_causal_mask: bool,
) -> torch.Tensor:
    outs: List[torch.Tensor] = []

    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        hidden = hidden_cache[idx].to(device)

        out = forward_decoder_block_no_grad(
            model=model,
            block=block,
            hidden=hidden,
            use_causal_mask=use_causal_mask,
        )

        outs.append(out.detach().to("cpu", dtype=target_dtype))

    return torch.cat(outs, dim=0)


def register_mask_gradient_hooks(
    block_layers: List[Tuple[str, nn.Linear]],
    masks: Dict[str, torch.Tensor],
) -> List[Any]:
    """
    During RO, pruned weights must remain pruned.

    This hook zeros gradients for pruned positions.
    """
    handles: List[Any] = []

    for name, layer in block_layers:
        if name not in masks:
            continue

        mask = masks[name].to(device=layer.weight.device, dtype=layer.weight.dtype)

        def hook_fn(grad: torch.Tensor, m: torch.Tensor = mask) -> torch.Tensor:
            return grad * m

        handles.append(layer.weight.register_hook(hook_fn))

    return handles


def regional_optimization(
    model: nn.Module,
    block: nn.Module,
    block_layers: List[Tuple[str, nn.Linear]],
    hidden_cache: torch.Tensor,
    dense_targets: torch.Tensor,
    ro_indices: List[int],
    masks: Dict[str, torch.Tensor],
    ro_batch_size: int,
    ro_lr: float,
    ro_weight_decay: float,
    ro_optimizer: str,
    device: torch.device,
    use_causal_mask: bool,
) -> None:
    """
    CRUCIAL Wanda++ RO:

        minimize || dense_block_output - pruned_block_output ||_2^2

    The zero mask is enforced:
        - gradient hook zeros pruned-weight gradients
        - after every optimizer step, weight *= mask
    """
    block.eval()

    for p in block.parameters():
        p.requires_grad_(True)

    params = [p for p in block.parameters() if p.requires_grad]

    if ro_optimizer.lower() == "rmsprop":
        opt = torch.optim.RMSprop(
            params,
            lr=ro_lr,
            weight_decay=ro_weight_decay,
            momentum=0.0,
            centered=False,
        )
    elif ro_optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(
            params,
            lr=ro_lr,
            weight_decay=ro_weight_decay,
        )
    else:
        raise ValueError("--ro_optimizer must be rmsprop or adamw")

    handles = register_mask_gradient_hooks(block_layers, masks)

    try:
        iterator = range(0, len(ro_indices), ro_batch_size)
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=math.ceil(len(ro_indices) / ro_batch_size),
                desc="RO optimize",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )

        for start in iterator:
            local_idx = list(range(start, min(start + ro_batch_size, len(ro_indices))))
            sample_idx = [ro_indices[j] for j in local_idx]

            hidden = hidden_cache[sample_idx].to(device)
            target = dense_targets[local_idx].to(device)

            opt.zero_grad(set_to_none=True)

            with torch.enable_grad():
                out = forward_decoder_block(
                    model=model,
                    block=block,
                    hidden=hidden,
                    use_causal_mask=use_causal_mask,
                    grad_enabled=True,
                )

                loss = F.mse_loss(out.float(), target.float(), reduction="mean")
                loss.backward()

            opt.step()

            # CRUCIAL:
            # Restore exact sparsity after update.
            enforce_masks(block_layers, masks)

    finally:
        for h in handles:
            h.remove()

        opt.zero_grad(set_to_none=True)
        del opt


# ============================================================
# Sparse checkpoint storage
# ============================================================

def build_sparse_layer_state(
    layer: nn.Linear,
    mask_cpu: torch.Tensor,
    save_values_dtype: torch.dtype,
    pattern: str,
) -> Dict[str, Any]:
    W = layer.weight.detach().cpu()
    mask = mask_cpu.bool()

    if W.shape != mask.shape:
        raise ValueError(f"Weight/mask shape mismatch: {tuple(W.shape)} vs {tuple(mask.shape)}")

    values = W[mask].to(save_values_dtype).contiguous()
    packed_mask = pack_bool_mask_rows(mask)

    total = mask.numel()
    kept = int(mask.sum().item())
    pruned = total - kept

    return {
        "shape": list(W.shape),
        "mask_packing": "packedbits",
        "mask": packed_mask,
        "values": values,
        "values_dtype": str(save_values_dtype).replace("torch.", ""),
        "pattern": pattern,
        "kept_count": kept,
        "pruned_count": pruned,
        "total_count": total,
        "sparsity": pruned / float(total),
    }


class WandaSparseLinear(nn.Module):
    """
    Runtime wrapper for saved sparse values + packed mask.

    This reconstructs dense W before F.linear.
    It is mathematically correct, not kernel-optimized.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
        values: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        cache_dequantized: bool = False,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.cache_dequantized = bool(cache_dequantized)

        self.register_buffer("mask", mask.contiguous().to(torch.uint8))
        self.register_buffer("values", values.contiguous())

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
        else:
            self.bias = None

        self._cached_weight: Optional[torch.Tensor] = None

    @torch.no_grad()
    def dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        mask = unpack_bool_mask_rows(self.mask, original_cols=self.in_features)
        w = torch.zeros(
            (self.out_features, self.in_features),
            device=self.values.device,
            dtype=dtype,
        )
        w[mask] = self.values.to(dtype=dtype)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache_dequantized:
            if self._cached_weight is None or self._cached_weight.device != x.device:
                self._cached_weight = self.dequantize_weight(dtype=x.dtype).to(x.device)
            w = self._cached_weight.to(dtype=x.dtype)
        else:
            w = self.dequantize_weight(dtype=x.dtype).to(x.device)

        bias = self.bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)

        return F.linear(x, w, bias)


def build_partial_state_dict_excluding_pruned_weights(
    model: nn.Module,
    pruned_layers: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    skip = {f"{name}.weight" for name in pruned_layers.keys()}
    out: Dict[str, torch.Tensor] = {}

    for k, v in model.state_dict().items():
        if k in skip:
            continue
        out[k] = v.detach().cpu()

    return out


def save_wandapp_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    out_path: str,
    model_id: str,
    meta: Dict[str, Any],
    sparse_layers: Dict[str, Any],
    keep_dense_state_dict: bool,
) -> None:
    if keep_dense_state_dict:
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    else:
        model_state = build_partial_state_dict_excluding_pruned_weights(
            model=model,
            pruned_layers=sparse_layers,
        )

    ckpt = {
        "format": "hf_wandapp_pruned",
        "model_id": model_id,
        "config": model.config.to_dict() if hasattr(model, "config") else None,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", model_id),
        "model": model_state,
        "wandapp_meta": meta,
        "wandapp_layers": sparse_layers,
    }

    torch.save(ckpt, out_path)

    meta_path = out_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved checkpoint: {out_path}")
    print(f"Saved meta JSON : {meta_path}")


# ============================================================
# Main compression loop
# ============================================================

@torch.no_grad()
def run_compressed_block_to_next_hidden(
    model: nn.Module,
    block: nn.Module,
    hidden_cache: torch.Tensor,
    batch_size: int,
    device: torch.device,
    hidden_cache_dtype: torch.dtype,
    use_causal_mask: bool,
) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    n = hidden_cache.size(0)

    iterator = range(0, n, batch_size)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=math.ceil(n / batch_size),
            desc="compressed block forward",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )

    for i in iterator:
        hidden = hidden_cache[i:i + batch_size].to(device)
        out = forward_decoder_block_no_grad(
            model=model,
            block=block,
            hidden=hidden,
            use_causal_mask=use_causal_mask,
        )
        outs.append(out.detach().to("cpu", dtype=hidden_cache_dtype))

    return torch.cat(outs, dim=0)


def compress_model_wandapp_blockwise(
    model: nn.Module,
    calib_tokens: torch.Tensor,
    alpha: float,
    sparsity: float,
    pattern: str,
    rgs_batch_size: int,
    forward_batch_size: int,
    ro_batch_size: int,
    ro_samples: int,
    ro_iters: int,
    ro_lr: float,
    ro_weight_decay: float,
    ro_optimizer: str,
    main_device: torch.device,
    accum_device: torch.device,
    hidden_cache_dtype: torch.dtype,
    score_dtype: torch.dtype,
    include: str,
    exclude: str,
    suffixes: Tuple[str, ...],
    skip_attn_out: bool,
    skip_mlp_out: bool,
    final_rgs_refresh: bool,
    use_causal_mask: bool,
    save_values_dtype: torch.dtype,
    seed: int,
) -> Dict[str, Any]:
    decoder, layers, layers_prefix = get_decoder_and_layers(model)

    print(f"Decoder layers found: {layers_prefix}")
    print(f"Number of decoder blocks: {len(layers)}")

    print("\nComputing initial embedding hidden cache...")
    hidden_cache = compute_initial_hidden_cache(
        model=model,
        calib_tokens=calib_tokens,
        batch_size=forward_batch_size,
        device=main_device,
        hidden_cache_dtype=hidden_cache_dtype,
    )

    print(f"Initial hidden cache: {tuple(hidden_cache.shape)}, dtype={hidden_cache.dtype}")
    print(f"CUDA memory: {cuda_memory_string()}")

    sparse_layers: Dict[str, Any] = {}
    rng = random.Random(seed)

    total_start = now()

    for bi, block in enumerate(layers):
        block_prefix = f"{layers_prefix}.{bi}"

        block_layers = find_block_linear_layers(
            block=block,
            block_prefix=block_prefix,
            include=include,
            exclude=exclude,
            suffixes=suffixes,
            skip_attn_out=skip_attn_out,
            skip_mlp_out=skip_mlp_out,
        )

        print("\n" + "=" * 100)
        print(f"BLOCK {bi}/{len(layers) - 1}")
        print(f"Selected Linear layers: {len(block_layers)}")
        print(f"Hidden cache entering block: {tuple(hidden_cache.shape)}, dtype={hidden_cache.dtype}")
        print(f"CUDA memory: {cuda_memory_string()}")

        for name, layer in block_layers:
            print(f"  - {name}: {tuple(layer.weight.shape)}, dtype={layer.weight.dtype}")

        if not block_layers:
            print("No selected layers in this block. Running block forward only.")
            hidden_cache = run_compressed_block_to_next_hidden(
                model=model,
                block=block,
                hidden_cache=hidden_cache,
                batch_size=forward_batch_size,
                device=main_device,
                hidden_cache_dtype=hidden_cache_dtype,
                use_causal_mask=use_causal_mask,
            )
            continue

        block_t0 = now()

        # ------------------------------------------------------------
        # Dense RO target is computed before pruning.
        # ------------------------------------------------------------
        ro_count = min(int(ro_samples), hidden_cache.size(0))
        ro_indices = list(range(hidden_cache.size(0)))
        rng.shuffle(ro_indices)
        ro_indices = ro_indices[:ro_count]

        print(f"RO samples selected: {len(ro_indices)}")

        dense_targets = None
        if ro_iters > 0 and len(ro_indices) > 0:
            print("Computing dense block outputs for RO targets...")
            dense_targets = compute_block_outputs_for_indices(
                model=model,
                block=block,
                hidden_cache=hidden_cache,
                indices=ro_indices,
                batch_size=ro_batch_size,
                device=main_device,
                target_dtype=hidden_cache_dtype,
                use_causal_mask=use_causal_mask,
            )

        # ------------------------------------------------------------
        # Initial RGS.
        # ------------------------------------------------------------
        print("\nComputing Wanda++ regional gradients and activation norms...")
        rgs_stats = collect_rgs_stats(
            model=model,
            block=block,
            block_layers=block_layers,
            hidden_cache=hidden_cache,
            rgs_batch_size=rgs_batch_size,
            main_device=main_device,
            accum_device=accum_device,
            alpha=alpha,
            use_causal_mask=use_causal_mask,
        )

        print(f"RGS backward steps: {rgs_stats.grad_samples}")
        print(f"Activation samples: {rgs_stats.act_samples}")

        masks: Dict[str, torch.Tensor] = {}

        # ------------------------------------------------------------
        # Wanda++ loop:
        #   prune -> RO -> prune -> RO ...
        # ------------------------------------------------------------
        for k in range(ro_iters):
            print("\n" + "-" * 100)
            print(f"Wanda++ iteration {k + 1}/{ro_iters}: RGS pruning + RO")

            # The paper reuses regional gradients during RO iterations
            # and blends them with fresh layer input norms.
            if k > 0:
                print("Refreshing activation norms after previous RO step...")
                act_norm = collect_activation_norms_only(
                    model=model,
                    block=block,
                    block_layers=block_layers,
                    hidden_cache=hidden_cache,
                    batch_size=forward_batch_size,
                    main_device=main_device,
                    accum_device=accum_device,
                    use_causal_mask=use_causal_mask,
                )
                rgs_stats.act_norm = act_norm

            masks = apply_wandapp_pruning_to_block(
                block_layers=block_layers,
                rgs_stats=rgs_stats,
                alpha=alpha,
                sparsity=sparsity,
                pattern=pattern,
                score_dtype=score_dtype,
            )

            if dense_targets is not None and len(ro_indices) > 0:
                print("Running Regional Optimization...")
                regional_optimization(
                    model=model,
                    block=block,
                    block_layers=block_layers,
                    hidden_cache=hidden_cache,
                    dense_targets=dense_targets,
                    ro_indices=ro_indices,
                    masks=masks,
                    ro_batch_size=ro_batch_size,
                    ro_lr=ro_lr,
                    ro_weight_decay=ro_weight_decay,
                    ro_optimizer=ro_optimizer,
                    device=main_device,
                    use_causal_mask=use_causal_mask,
                )

                enforce_masks(block_layers, masks)

        # ------------------------------------------------------------
        # Final RGS pruning.
        # Paper performs another RGS backward and final pruning.
        # ------------------------------------------------------------
        print("\n" + "-" * 100)
        print("Final Wanda++ pruning pass")

        if final_rgs_refresh:
            print("Refreshing regional gradients for final pruning...")
            rgs_stats = collect_rgs_stats(
                model=model,
                block=block,
                block_layers=block_layers,
                hidden_cache=hidden_cache,
                rgs_batch_size=rgs_batch_size,
                main_device=main_device,
                accum_device=accum_device,
                alpha=alpha,
                use_causal_mask=use_causal_mask,
            )
        else:
            print("Refreshing only activation norms for final pruning...")
            act_norm = collect_activation_norms_only(
                model=model,
                block=block,
                block_layers=block_layers,
                hidden_cache=hidden_cache,
                batch_size=forward_batch_size,
                main_device=main_device,
                accum_device=accum_device,
                use_causal_mask=use_causal_mask,
            )
            rgs_stats.act_norm = act_norm

        masks = apply_wandapp_pruning_to_block(
            block_layers=block_layers,
            rgs_stats=rgs_stats,
            alpha=alpha,
            sparsity=sparsity,
            pattern=pattern,
            score_dtype=score_dtype,
        )

        enforce_masks(block_layers, masks)

        # ------------------------------------------------------------
        # Save sparse layer states.
        # ------------------------------------------------------------
        for name, layer in block_layers:
            sparse_layers[name] = build_sparse_layer_state(
                layer=layer,
                mask_cpu=masks[name],
                save_values_dtype=save_values_dtype,
                pattern=pattern,
            )

        # ------------------------------------------------------------
        # Sequential blockwise compression:
        # run compressed block to make next block inputs.
        # ------------------------------------------------------------
        print("\nRunning pruned block to create next hidden cache...")
        hidden_cache = run_compressed_block_to_next_hidden(
            model=model,
            block=block,
            hidden_cache=hidden_cache,
            batch_size=forward_batch_size,
            device=main_device,
            hidden_cache_dtype=hidden_cache_dtype,
            use_causal_mask=use_causal_mask,
        )

        del rgs_stats
        del masks
        del dense_targets

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Finished block {bi}. Block time: {format_seconds(now() - block_t0)}")
        print(f"Total elapsed: {format_seconds(now() - total_start)}")
        print(f"CUDA memory: {cuda_memory_string()}")

    return sparse_layers


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hidden_cache_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--score_dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--save_values_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--pattern", type=str, default="2:4", help="'unstructured', '2:4', or '4:8'")
    parser.add_argument("--alpha", type=float, default=100.0)

    parser.add_argument("--rgs_batch_size", type=int, default=1)
    parser.add_argument("--forward_batch_size", type=int, default=1)
    parser.add_argument("--ro_batch_size", type=int, default=1)

    parser.add_argument("--ro_iters", type=int, default=2)
    parser.add_argument("--ro_samples", type=int, default=32)
    parser.add_argument("--ro_lr", type=float, default=3e-7)
    parser.add_argument("--ro_weight_decay", type=float, default=0.0)
    parser.add_argument("--ro_optimizer", type=str, default="rmsprop", choices=["rmsprop", "adamw"])

    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")
    parser.add_argument("--suffixes", type=str, default="")
    parser.add_argument("--skip_attn_out", action="store_true")
    parser.add_argument("--skip_mlp_out", action="store_true")

    parser.add_argument("--final_rgs_refresh", action="store_true")
    parser.add_argument("--no_causal_mask", action="store_true")

    parser.add_argument("--keep_dense_state_dict", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="eager")

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--accum_device", type=str, default="cpu", choices=["cpu", "cuda"])

    args = parser.parse_args()

    script_t0 = now()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable.")

    if args.rgs_batch_size != 1:
        print(
            "[warn] For paper-like per-sample regional gradient aggregation, "
            "use --rgs_batch_size 1."
        )

    if args.pattern != "unstructured":
        nm = parse_nm_pattern(args.pattern)
        if nm is not None:
            n_zero, m = nm
            implied = n_zero / float(m)
            print(f"[info] N:M pattern {args.pattern} implies sparsity={implied:.4f}")
            if abs(implied - args.sparsity) > 1e-6:
                print(
                    f"[warn] You passed --sparsity {args.sparsity}, but pattern {args.pattern} "
                    f"implies {implied}. For N:M, the pattern controls actual sparsity."
                )

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    main_device = torch.device(args.device)
    model_dtype = parse_dtype(args.model_dtype)
    hidden_cache_dtype = parse_dtype(args.hidden_cache_dtype)
    score_dtype = parse_dtype(args.score_dtype)
    save_values_dtype = parse_dtype(args.save_values_dtype)
    accum_device = torch.device(args.accum_device)

    if accum_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--accum_device cuda requested but CUDA unavailable.")

    suffixes = parse_suffixes(args.suffixes)

    print("=" * 100)
    print("Wanda++ HF/Mistral pruning")
    print("=" * 100)
    print(f"model_id              : {args.model_id}")
    print(f"calib                 : {args.calib}")
    print(f"out                   : {args.out}")
    print(f"device                : {main_device}")
    print(f"model_dtype           : {model_dtype}")
    print(f"hidden_cache_dtype    : {hidden_cache_dtype}")
    print(f"score_dtype           : {score_dtype}")
    print(f"save_values_dtype     : {save_values_dtype}")
    print(f"sparsity              : {args.sparsity}")
    print(f"pattern               : {args.pattern}")
    print(f"alpha                 : {args.alpha}")
    print(f"rgs_batch_size        : {args.rgs_batch_size}")
    print(f"forward_batch_size    : {args.forward_batch_size}")
    print(f"ro_iters              : {args.ro_iters}")
    print(f"ro_samples            : {args.ro_samples}")
    print(f"ro_lr                 : {args.ro_lr}")
    print(f"final_rgs_refresh     : {args.final_rgs_refresh}")
    print(f"accum_device          : {accum_device}")
    print(f"suffixes              : {suffixes}")

    print("\nLoading tokenizer/model...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=bool(args.trust_remote_code),
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": bool(args.low_cpu_mem_usage),
        "trust_remote_code": bool(args.trust_remote_code),
    }

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    if hasattr(model, "config"):
        model.config.use_cache = False

    model.eval()
    model.to(main_device)

    print(f"Model loaded. CUDA memory: {cuda_memory_string()}")

    calib_tokens = load_calibration_tokens(args.calib)
    print(f"Loaded calibration tokens: {tuple(calib_tokens.shape)}")

    if args.max_seq_len > 0 and calib_tokens.size(1) > args.max_seq_len:
        calib_tokens = calib_tokens[:, :args.max_seq_len].contiguous()
        print(f"Trimmed calibration sequence length to {args.max_seq_len}")

    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
        max_pos = int(model.config.max_position_embeddings)
        if calib_tokens.size(1) > max_pos:
            calib_tokens = calib_tokens[:, :max_pos].contiguous()
            print(f"Trimmed calibration sequence length to model max_position_embeddings={max_pos}")

    sparse_layers = compress_model_wandapp_blockwise(
        model=model,
        calib_tokens=calib_tokens,
        alpha=float(args.alpha),
        sparsity=float(args.sparsity),
        pattern=str(args.pattern),
        rgs_batch_size=int(args.rgs_batch_size),
        forward_batch_size=int(args.forward_batch_size),
        ro_batch_size=int(args.ro_batch_size),
        ro_samples=int(args.ro_samples),
        ro_iters=int(args.ro_iters),
        ro_lr=float(args.ro_lr),
        ro_weight_decay=float(args.ro_weight_decay),
        ro_optimizer=str(args.ro_optimizer),
        main_device=main_device,
        accum_device=accum_device,
        hidden_cache_dtype=hidden_cache_dtype,
        score_dtype=score_dtype,
        include=str(args.include),
        exclude=str(args.exclude),
        suffixes=suffixes,
        skip_attn_out=bool(args.skip_attn_out),
        skip_mlp_out=bool(args.skip_mlp_out),
        final_rgs_refresh=bool(args.final_rgs_refresh),
        use_causal_mask=not bool(args.no_causal_mask),
        save_values_dtype=save_values_dtype,
        seed=int(args.seed),
    )

    total_pruned = sum(int(v["pruned_count"]) for v in sparse_layers.values())
    total_weights = sum(int(v["total_count"]) for v in sparse_layers.values())
    total_kept = sum(int(v["kept_count"]) for v in sparse_layers.values())

    mask_bytes = sum(int(v["mask"].numel()) for v in sparse_layers.values())
    values_bytes = sum(int(v["values"].numel() * v["values"].element_size()) for v in sparse_layers.values())
    stored_bytes = mask_bytes + values_bytes

    dense_bf16_bytes = total_weights * 2
    dense_fp16_bytes = total_weights * 2

    actual_sparsity = total_pruned / float(total_weights) if total_weights > 0 else 0.0
    compression_vs_fp16 = dense_fp16_bytes / float(stored_bytes) if stored_bytes > 0 else 0.0

    meta = {
        "method": "wanda_plus_plus_hf_regional_gradients_regional_optimization",
        "model_id": args.model_id,
        "calibration_source": args.calib,
        "alpha": float(args.alpha),
        "sparsity": float(args.sparsity),
        "pattern": str(args.pattern),
        "rgs_loss": "||decoder_block_output||_2",
        "rgs_score": "(alpha * regional_gradient_magnitude + input_channel_l2_norm) * abs(weight)",
        "ro_loss": "MSE(dense_decoder_block_output, pruned_decoder_block_output)",
        "ro_iters": int(args.ro_iters),
        "ro_samples": int(args.ro_samples),
        "ro_lr": float(args.ro_lr),
        "ro_optimizer": str(args.ro_optimizer),
        "final_rgs_refresh": bool(args.final_rgs_refresh),
        "model_dtype": args.model_dtype,
        "hidden_cache_dtype": args.hidden_cache_dtype,
        "score_dtype": args.score_dtype,
        "save_values_dtype": args.save_values_dtype,
        "suffixes": list(suffixes),
        "skip_attn_out": bool(args.skip_attn_out),
        "skip_mlp_out": bool(args.skip_mlp_out),
        "total_layers": int(len(sparse_layers)),
        "total_weights": int(total_weights),
        "total_kept": int(total_kept),
        "total_pruned": int(total_pruned),
        "actual_sparsity": float(actual_sparsity),
        "mask_bytes": int(mask_bytes),
        "values_bytes": int(values_bytes),
        "stored_bytes": int(stored_bytes),
        "dense_fp16_bytes": int(dense_fp16_bytes),
        "compression_vs_fp16_estimate": float(compression_vs_fp16),
        "keep_dense_state_dict": bool(args.keep_dense_state_dict),
        "script_seconds": float(now() - script_t0),
        "note": (
            "This checkpoint stores Wanda++-pruned layers as packed masks plus surviving values. "
            "Runtime reconstruction is dense and not kernel-optimized. "
            "This is pruning only, not GPTQ quantization."
        ),
    }

    print("\nSaving checkpoint...")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    save_wandapp_checkpoint(
        model=model,
        tokenizer=tokenizer,
        out_path=args.out,
        model_id=args.model_id,
        meta=meta,
        sparse_layers=sparse_layers,
        keep_dense_state_dict=bool(args.keep_dense_state_dict),
    )

    print("\nDone.")
    print(f"Pruned layers                 : {len(sparse_layers)}")
    print(f"Total selected weights        : {total_weights:,}")
    print(f"Kept weights                  : {total_kept:,}")
    print(f"Pruned weights                : {total_pruned:,}")
    print(f"Actual sparsity               : {100.0 * actual_sparsity:.2f}%")
    print(f"Packed mask bytes             : {mask_bytes:,}")
    print(f"Surviving value bytes         : {values_bytes:,}")
    print(f"Raw sparse stored bytes       : {stored_bytes:,}")
    print(f"Dense FP16/BF16 bytes         : {dense_fp16_bytes:,}")
    print(f"Compression vs FP16 estimate  : {compression_vs_fp16:.2f}x")
    print(f"Total script time             : {format_seconds(now() - script_t0)}")
    print(f"CUDA memory                   : {cuda_memory_string()}")

    if args.keep_dense_state_dict:
        print("Checkpoint includes full dense masked state_dict.")
    else:
        print("Checkpoint includes only non-compressed parameters plus sparse layer storage.")


if __name__ == "__main__":
    main()