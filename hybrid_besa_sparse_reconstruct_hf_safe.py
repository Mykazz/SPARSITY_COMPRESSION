#!/usr/bin/env python3
"""
Hybrid BESA / SparseGPT / Wanda++ sparse-only compression for Hugging Face decoder-only LMs.

This is a safer full rewrite for your failing 70% sparsity run.

Main fixes for the CUDA assert you saw:

1. All mask/value packing is done on CPU.
   The previous failure happened after reconstruction when a CUDA indexing kernel asserted.
   Once a CUDA device-side assert happens, the CUDA context is poisoned and even saving can fail.

2. dense_weight_from_sparse_state now reconstructs on CPU first, then moves the dense tensor to CUDA.
   This avoids CUDA boolean-index assignment/indexing in checkpoint conversion paths.

3. extract_states_from_trainable_sparse now moves weight/mask to CPU before boolean indexing.

4. replace_trainable_sparse_with_dense_linear never performs w[mask] = values on CUDA.

5. Emergency partial checkpoint saving no longer tries to copy the full CUDA model state after a CUDA failure.
   It saves only the sparse layer dictionary and metadata, which is enough for eval_universal.py because
   evaluation loads the base HF model from --model_id and applies compressed layers on top.

6. Partial checkpoints are saved after every completed block by default.
   If the run dies at block 21, you should have a valid .partial.pt through block 20.

Expected checkpoint keys:
    ckpt["hybrid_sparse_layers"][layer_name]

Each sparse layer stores:
    mask          : packedbits uint8 mask
    values        : 1D kept values in row-major mask order
    values_format : "kept_1d"
    shape         : [out_features, in_features]

Run example:

/venv/main/bin/python hybrid_besa_sparse_reconstruct_hf_safe.py \
  --model_id mistralai/Mistral-7B-Instruct-v0.3 \
  --calib data/calib_wikitext103_train_128x2048_mistral.pt \
  --out compressed/mistral_hybrid_sparse70_safe_recon100_bf16_1024.pt \
  --target_sparsity 0.70 \
  --batch_size 1 \
  --model_dtype bfloat16 \
  --hidden_cache_dtype float16 \
  --hessian_dtype float32 \
  --value_dtype bfloat16 \
  --max_seq_len 1024 \
  --percdamp 0.1 \
  --blocksize 128 \
  --mask_blocksize 128 \
  --max_candidates 5 \
  --recon_steps 100 \
  --recon_lr 2e-5 \
  --recon_grad_clip 1.0 \
  --large_layer_cpu_threshold 8192 \
  --attn_implementation eager \
  --resume_from_partial \
  2>&1 | tee -a logs/mistral_hybrid_sparse70_safe_recon100_bf16_1024.log
"""

from __future__ import annotations

import argparse
import copy
import gc
import inspect
import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Basic utilities
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
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


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


def dtype_to_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float64:
        return "float64"
    return str(dtype)


def cuda_memory_string() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB"


def print_cuda_memory(prefix: str = "CUDA memory") -> None:
    if torch.cuda.is_available():
        print(f"{prefix}: {cuda_memory_string()}")


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return fallback


def get_module_dtype(module: nn.Module, fallback: torch.dtype = torch.float16) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        try:
            return next(module.buffers()).dtype
        except StopIteration:
            return fallback


def sanitize_tensor(x: torch.Tensor, clamp_abs: Optional[float] = None) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if clamp_abs is not None and clamp_abs > 0:
        x = x.clamp(min=-clamp_abs, max=clamp_abs)
    return x


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
# Module helpers
# ============================================================

def get_module_by_name(root: nn.Module, full_name: str) -> nn.Module:
    obj = root
    if full_name == "":
        return obj
    for part in full_name.split("."):
        obj = getattr(obj, part)
    return obj


def set_module_by_name(root: nn.Module, full_name: str, new_module: nn.Module) -> None:
    if full_name == "":
        raise ValueError("Cannot replace root module with set_module_by_name.")
    parts = full_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def get_base_decoder_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    raise RuntimeError("Could not find decoder body. Expected model.model or model.transformer.")


def get_decoder_layers(base: nn.Module) -> Tuple[str, nn.ModuleList]:
    for name in ("layers", "h", "blocks"):
        if hasattr(base, name):
            obj = getattr(base, name)
            if isinstance(obj, (nn.ModuleList, list)):
                return name, obj
    raise RuntimeError("Could not find decoder layers. Expected base.layers / base.h / base.blocks.")


def default_suffixes() -> Tuple[str, ...]:
    return ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_suffixes(raw: str) -> Tuple[str, ...]:
    raw = raw.strip()
    if not raw:
        return default_suffixes()
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def layer_suffix(name: str) -> str:
    return name.split(".")[-1]


def should_compress_name(name: str, suffixes: Tuple[str, ...], include: str, exclude: str) -> bool:
    if include and include not in name:
        return False
    if exclude and exclude in name:
        return False
    return name.endswith(suffixes)


def find_block_linears(
    block: nn.Module,
    block_prefix: str,
    suffixes: Tuple[str, ...],
    include: str,
    exclude: str,
) -> List[Tuple[str, str, nn.Linear]]:
    out: List[Tuple[str, str, nn.Linear]] = []
    for local_name, mod in block.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        full_name = f"{block_prefix}.{local_name}" if local_name else block_prefix
        if should_compress_name(full_name, suffixes=suffixes, include=include, exclude=exclude):
            out.append((full_name, local_name, mod))
    return out


# ============================================================
# Mask packing
# ============================================================

def pack_bool_mask_rows(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.detach().cpu().bool()
    rows, cols = mask.shape
    packed_cols = (cols + 7) // 8
    padded_cols = packed_cols * 8
    if padded_cols != cols:
        pad = torch.zeros((rows, padded_cols - cols), dtype=torch.bool)
        mask = torch.cat([mask, pad], dim=1)
    mask_u8 = mask.to(torch.uint8).view(rows, packed_cols, 8)
    shifts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
    return (mask_u8 * shifts.view(1, 1, 8)).sum(dim=2).to(torch.uint8).cpu()


def unpack_bool_mask_rows_cpu(packed: torch.Tensor, original_cols: int) -> torch.Tensor:
    packed = packed.detach().cpu().to(torch.uint8)
    rows, packed_cols = packed.shape
    shifts = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8)
    bits = ((packed.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 1).bool()
    return bits.view(rows, packed_cols * 8)[:, :original_cols].contiguous()


# ============================================================
# SparseGPT pruning containers and helpers
# ============================================================

@dataclass
class SparseLayerResult:
    mask: torch.Tensor
    dense_sparse_weight: torch.Tensor
    original_shape: Tuple[int, int]
    sparsity: float
    target_sparsity: float
    pattern: str
    pruned_count: int
    total_count: int


def parse_nm_pattern(pattern: str) -> Optional[Tuple[int, int]]:
    pattern = pattern.strip().lower()
    if pattern in ("", "none", "unstructured"):
        return None
    if ":" not in pattern:
        raise ValueError("Pattern must be 'unstructured' or N:M, e.g. '2:4'.")
    a, b = pattern.split(":")
    n = int(a)
    m = int(b)
    if n < 0 or m <= 0 or n > m:
        raise ValueError(f"Invalid N:M pattern: {pattern}")
    return n, m


@torch.no_grad()
def select_unstructured_mask_block_from_score(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    rows, cols = score.shape
    total = rows * cols
    n_prune = int(round(float(sparsity) * total))
    if n_prune <= 0:
        return torch.ones_like(score, dtype=torch.bool)
    if n_prune >= total:
        return torch.zeros_like(score, dtype=torch.bool)
    n_keep = total - n_prune
    score = sanitize_tensor(score)
    flat = score.reshape(-1)
    keep_idx = torch.topk(flat, k=n_keep, largest=True, sorted=False).indices
    mask = torch.zeros(total, dtype=torch.bool, device=score.device)
    mask[keep_idx] = True
    return mask.view(rows, cols)


@torch.no_grad()
def select_nm_mask_block_from_score(score: torch.Tensor, n_zero: int, m: int) -> torch.Tensor:
    rows, cols = score.shape
    score = sanitize_tensor(score)
    mask = torch.ones_like(score, dtype=torch.bool)
    for g0 in range(0, cols, m):
        g1 = min(g0 + m, cols)
        group_cols = g1 - g0
        prune_count = n_zero if group_cols == m else int(round((n_zero / float(m)) * group_cols))
        prune_count = max(0, min(prune_count, group_cols))
        if prune_count == 0:
            continue
        if prune_count == group_cols:
            mask[:, g0:g1] = False
            continue
        local_score = score[:, g0:g1]
        prune_idx = torch.topk(local_score, k=prune_count, largest=False, dim=1, sorted=False).indices
        row_idx = torch.arange(rows, device=score.device).view(-1, 1).expand_as(prune_idx)
        local_mask = mask[:, g0:g1]
        local_mask[row_idx, prune_idx] = False
        mask[:, g0:g1] = local_mask
    return mask


@torch.no_grad()
def stable_inverse_cholesky(H: torch.Tensor, percdamp: float, max_tries: int = 12) -> Tuple[torch.Tensor, float]:
    if H.ndim != 2 or H.size(0) != H.size(1):
        raise ValueError(f"H must be square, got {tuple(H.shape)}")
    device = H.device
    n = H.size(0)
    H64 = H.to(torch.float64)
    H64 = sanitize_tensor(H64)
    H64 = 0.5 * (H64 + H64.T)

    diag = torch.diag(H64)
    diag_abs = diag.abs()
    diag_mean = max(float(diag_abs.mean().item()), 1e-12)
    diag_max = max(float(diag_abs.max().item()), diag_mean, 1e-12)
    ar = torch.arange(n, device=device)

    multipliers = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    for mult in multipliers[:max_tries]:
        used_damp = percdamp * mult * diag_mean
        H_try = H64.clone()
        H_try[ar, ar] += used_damp
        try:
            chol = torch.linalg.cholesky(H_try)
            Hinv = torch.cholesky_inverse(chol)
            Hinv = 0.5 * (Hinv + Hinv.T)
            Hinv_chol_upper = torch.linalg.cholesky(Hinv, upper=True)
            return Hinv_chol_upper.to(torch.float32), used_damp
        except RuntimeError:
            pass

    used_damp = percdamp * 5000.0 * diag_max
    H_try = H64.clone()
    H_try[ar, ar] += used_damp
    evals, evecs = torch.linalg.eigh(H_try)
    floor = max(float(evals.abs().max().item()) * 1e-8, 1e-8)
    evals = evals.clamp(min=floor)
    Hinv = (evecs * (1.0 / evals).view(1, -1)) @ evecs.T
    Hinv = 0.5 * (Hinv + Hinv.T)
    try:
        Hinv_chol_upper = torch.linalg.cholesky(Hinv, upper=True)
    except RuntimeError:
        Hinv = Hinv + torch.eye(n, device=device, dtype=Hinv.dtype) * floor
        Hinv_chol_upper = torch.linalg.cholesky(Hinv, upper=True)
    return Hinv_chol_upper.to(torch.float32), used_damp


@torch.no_grad()
def sparsegpt_prune_linear(
    layer: nn.Linear,
    H: torch.Tensor,
    sparsity: float,
    pattern: str,
    percdamp: float,
    blocksize: int,
    mask_blocksize: int,
    compress_device: torch.device,
    act_order: bool,
    verbose: bool = True,
) -> SparseLayerResult:
    original_device = layer.weight.device
    original_dtype = layer.weight.dtype

    W_orig = layer.weight.detach().to(device=compress_device, dtype=torch.float32).clone()
    W_orig = sanitize_tensor(W_orig)
    rows, cols = W_orig.shape

    if H.shape != (cols, cols):
        raise ValueError(f"H shape mismatch. Expected {(cols, cols)}, got {tuple(H.shape)}")

    H = sanitize_tensor(H.to(compress_device))

    nm = parse_nm_pattern(pattern)
    if nm is not None:
        n_zero, m = nm
        mask_blocksize_eff = m
        effective_sparsity = n_zero / float(m)
    else:
        n_zero, m = None, None
        mask_blocksize_eff = mask_blocksize
        effective_sparsity = float(sparsity)

    if verbose:
        print(f"      target sparsity   : {effective_sparsity:.4f}")
        print(f"      pattern           : {pattern}")
        print(f"      blocksize         : {blocksize}")
        print(f"      mask blocksize    : {mask_blocksize_eff}")
        print(f"      compression device: {compress_device}")

    t_inv = now()
    Hinv_chol_upper, used_damp = stable_inverse_cholesky(H, percdamp=percdamp)
    if verbose:
        print(f"      used damping      : {used_damp:.6e}")
        print(f"      inverse time      : {format_seconds(now() - t_inv)}")

    Hinv = Hinv_chol_upper.T @ Hinv_chol_upper
    Hinv = sanitize_tensor(Hinv.to(device=compress_device, dtype=torch.float32))

    if act_order:
        perm = torch.argsort(torch.diag(H).abs(), descending=True)
        invperm = torch.argsort(perm)
        W = W_orig[:, perm].contiguous()
        Hinv = Hinv[perm][:, perm].contiguous()
    else:
        invperm = None
        W = W_orig.clone()

    Q = torch.zeros_like(W)
    M = torch.ones((rows, cols), dtype=torch.bool, device=compress_device)
    Hinv_diag = torch.diag(Hinv).abs().clamp(min=1e-12)
    selected_mask_until = -1

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1
        W1 = sanitize_tensor(W[:, i1:i2].clone())
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = sanitize_tensor(Hinv[i1:i2, i1:i2].contiguous())

        for local_i in range(count):
            global_col = i1 + local_i

            if global_col >= selected_mask_until:
                mb0 = global_col
                mb1 = min(mb0 + mask_blocksize_eff, cols)
                diag = Hinv_diag[mb0:mb1].view(1, -1).clamp(min=1e-12)
                score = (sanitize_tensor(W[:, mb0:mb1]).float() ** 2) / diag.float()
                score = sanitize_tensor(score)

                if nm is None:
                    M_block = select_unstructured_mask_block_from_score(score, float(sparsity))
                else:
                    assert n_zero is not None and m is not None
                    M_block = select_nm_mask_block_from_score(score, n_zero=n_zero, m=m)

                M[:, mb0:mb1] = M_block
                selected_mask_until = mb1

            d = Hinv1[local_i, local_i].abs().clamp(min=1e-12)
            w = sanitize_tensor(W1[:, local_i])
            keep = M[:, global_col]
            q = torch.where(keep, w, torch.zeros_like(w))

            Q1[:, local_i] = q
            Q[:, global_col] = q

            err = sanitize_tensor((w - q) / d)
            Err1[:, local_i] = err

            if local_i + 1 < count:
                W1[:, local_i + 1:count] -= err.unsqueeze(1) @ Hinv1[local_i, local_i + 1:count].unsqueeze(0)
                W1[:, local_i + 1:count] = sanitize_tensor(W1[:, local_i + 1:count])

        W[:, i1:i2] = Q1
        if i2 < cols:
            W[:, i2:cols] -= Err1 @ Hinv[i1:i2, i2:cols]
            W[:, i2:cols] = sanitize_tensor(W[:, i2:cols])

    if act_order:
        assert invperm is not None
        Q = Q[:, invperm].contiguous()
        M = M[:, invperm].contiguous()

    Q = sanitize_tensor(Q * M.to(Q.dtype))
    layer.weight.data.copy_(Q.to(device=original_device, dtype=original_dtype))

    total_count = rows * cols
    kept_count = int(M.sum().item())
    pruned_count = total_count - kept_count
    actual_sparsity = pruned_count / float(total_count)

    return SparseLayerResult(
        mask=M.detach().cpu(),
        dense_sparse_weight=Q.detach().cpu(),
        original_shape=(rows, cols),
        sparsity=actual_sparsity,
        target_sparsity=effective_sparsity,
        pattern=pattern,
        pruned_count=pruned_count,
        total_count=total_count,
    )


# ============================================================
# Block forward helpers
# ============================================================

def build_position_inputs(base: nn.Module, hidden: torch.Tensor) -> Dict[str, Any]:
    batch, seq_len, _ = hidden.shape
    device = hidden.device
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    cache_position = torch.arange(seq_len, device=device, dtype=torch.long)
    out: Dict[str, Any] = {
        "position_ids": position_ids,
        "cache_position": cache_position,
    }
    if hasattr(base, "rotary_emb"):
        try:
            out["position_embeddings"] = base.rotary_emb(hidden, position_ids)
        except Exception:
            pass
    return out


def forward_decoder_block_impl(
    base: nn.Module,
    block: nn.Module,
    hidden: torch.Tensor,
    model_dtype: torch.dtype,
    use_autocast: bool,
) -> torch.Tensor:
    block_dtype = get_module_dtype(block, fallback=model_dtype)
    hidden = hidden.to(device=get_module_device(block, hidden.device), dtype=block_dtype)
    pos = build_position_inputs(base, hidden)

    sig = inspect.signature(block.forward)
    supported = set(sig.parameters.keys())

    candidates: Dict[str, Any] = {
        "hidden_states": hidden,
        "attention_mask": None,
        "position_ids": pos.get("position_ids"),
        "past_key_value": None,
        "output_attentions": False,
        "use_cache": False,
        "cache_position": pos.get("cache_position"),
        "position_embeddings": pos.get("position_embeddings"),
    }

    kwargs: Dict[str, Any] = {}
    for k, v in candidates.items():
        if k in supported:
            kwargs[k] = v

    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=model_dtype)
        if use_autocast and hidden.device.type == "cuda" and model_dtype in (torch.float16, torch.bfloat16)
        else torch.enable_grad()
    )

    with ctx:
        try:
            out = block(**kwargs)
        except TypeError:
            out = block(hidden)

    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if torch.is_tensor(out):
        return out
    raise RuntimeError(f"Unsupported block output type: {type(out)}")


@torch.no_grad()
def forward_decoder_block_no_grad(
    base: nn.Module,
    block: nn.Module,
    hidden: torch.Tensor,
    model_dtype: torch.dtype,
    use_autocast: bool,
) -> torch.Tensor:
    return forward_decoder_block_impl(base, block, hidden, model_dtype, use_autocast).detach()


def forward_decoder_block_grad(
    base: nn.Module,
    block: nn.Module,
    hidden: torch.Tensor,
    model_dtype: torch.dtype,
    use_autocast: bool,
) -> torch.Tensor:
    return forward_decoder_block_impl(base, block, hidden, model_dtype, use_autocast)


@torch.no_grad()
def compute_initial_hidden(
    model: nn.Module,
    calib_tokens: torch.Tensor,
    batch_size: int,
    device: torch.device,
    hidden_cache_dtype: torch.dtype,
) -> torch.Tensor:
    base = get_base_decoder_model(model)
    if not hasattr(base, "embed_tokens"):
        raise RuntimeError("This script expects base.embed_tokens, as in Mistral/LLaMA.")

    hidden_chunks: List[torch.Tensor] = []
    n = calib_tokens.size(0)
    print("\nComputing initial embedding hidden cache...")
    t0 = now()
    for i in range(0, n, batch_size):
        input_ids = calib_tokens[i:i + batch_size].to(device)
        h = base.embed_tokens(input_ids)
        hidden_chunks.append(h.detach().to("cpu", dtype=hidden_cache_dtype))
        done = min(i + batch_size, n)
        print(
            f"\r  embeddings: {done}/{n} ({100.0 * done / n:.1f}%) "
            f"elapsed={format_seconds(now() - t0)}",
            end="",
            flush=True,
        )
    print()
    hidden = torch.cat(hidden_chunks, dim=0).contiguous()
    print(f"Initial hidden cache shape: {tuple(hidden.shape)}, dtype={hidden.dtype}")
    return hidden


@torch.no_grad()
def run_block_on_hidden_cache(
    base: nn.Module,
    block: nn.Module,
    hidden_cpu: torch.Tensor,
    batch_size: int,
    main_device: torch.device,
    model_dtype: torch.dtype,
    output_dtype: torch.dtype,
    desc: str,
) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    n = hidden_cpu.size(0)
    t0 = now()
    for i in range(0, n, batch_size):
        hidden = hidden_cpu[i:i + batch_size].to(main_device, dtype=model_dtype)
        out = forward_decoder_block_no_grad(base, block, hidden, model_dtype, use_autocast=True)
        outs.append(out.detach().to("cpu", dtype=output_dtype))
        done = min(i + batch_size, n)
        print(
            f"\r  {desc}: {done}/{n} ({100.0 * done / n:.1f}%) "
            f"elapsed={format_seconds(now() - t0)}",
            end="",
            flush=True,
        )
    print()
    return torch.cat(outs, dim=0).contiguous()


# ============================================================
# Hessian collection
# ============================================================

class BlockHessianCollector:
    def __init__(
        self,
        layer_infos: List[Tuple[str, str, nn.Linear]],
        hessian_dtype: torch.dtype,
        main_device: torch.device,
        large_layer_cpu_threshold: int,
    ):
        self.layer_infos = layer_infos
        self.hessian_dtype = hessian_dtype
        self.main_device = main_device
        self.large_layer_cpu_threshold = int(large_layer_cpu_threshold)
        self.Hs: Dict[str, torch.Tensor] = {}
        self.nsamples: Dict[str, int] = {}
        self.devices: Dict[str, torch.device] = {}
        self.handles: List[Any] = []

        for full_name, _local_name, layer in layer_infos:
            in_features = int(layer.in_features)
            if self.large_layer_cpu_threshold <= 0:
                hdev = torch.device("cpu")
            elif in_features > self.large_layer_cpu_threshold:
                hdev = torch.device("cpu")
            else:
                hdev = main_device
            self.devices[full_name] = hdev
            self.nsamples[full_name] = 0
            self.Hs[full_name] = torch.zeros((in_features, in_features), device=hdev, dtype=hessian_dtype)

    def _make_hook(self, full_name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            x = x.detach().reshape(-1, x.size(-1))
            x = sanitize_tensor(x)
            hdev = self.devices[full_name]
            x = x.to(device=hdev, dtype=self.hessian_dtype)
            local = 2.0 * x.T.matmul(x)
            local = sanitize_tensor(local)
            self.Hs[full_name] += local
            self.nsamples[full_name] += x.size(0)
        return hook

    def register(self) -> None:
        for full_name, _local_name, layer in self.layer_infos:
            self.handles.append(layer.register_forward_pre_hook(self._make_hook(full_name)))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


@torch.no_grad()
def collect_block_hessians(
    base: nn.Module,
    block: nn.Module,
    layer_infos: List[Tuple[str, str, nn.Linear]],
    hidden_cpu: torch.Tensor,
    batch_size: int,
    main_device: torch.device,
    model_dtype: torch.dtype,
    hessian_dtype: torch.dtype,
    large_layer_cpu_threshold: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], Dict[str, torch.device]]:
    collector = BlockHessianCollector(layer_infos, hessian_dtype, main_device, large_layer_cpu_threshold)
    collector.register()
    n = hidden_cpu.size(0)
    t0 = now()
    print("  Collecting Hessians for all selected Linear layers in this block...")
    try:
        for i in range(0, n, batch_size):
            hidden = hidden_cpu[i:i + batch_size].to(main_device, dtype=model_dtype)
            _ = forward_decoder_block_no_grad(base, block, hidden, model_dtype, use_autocast=True)
            done = min(i + batch_size, n)
            print(
                f"\r  block Hessian pass: {done}/{n} ({100.0 * done / n:.1f}%) "
                f"elapsed={format_seconds(now() - t0)}",
                end="",
                flush=True,
            )
        print()
    finally:
        collector.remove()
    return collector.Hs, collector.nsamples, collector.devices


# ============================================================
# Save/restore block weights
# ============================================================

@torch.no_grad()
def save_original_block_weights(layer_infos: List[Tuple[str, str, nn.Linear]]) -> Dict[str, torch.Tensor]:
    return {full_name: layer.weight.detach().cpu().clone() for full_name, _local_name, layer in layer_infos}


@torch.no_grad()
def restore_block_weights(layer_infos: List[Tuple[str, str, nn.Linear]], saved: Dict[str, torch.Tensor]) -> None:
    for full_name, _local_name, layer in layer_infos:
        layer.weight.data.copy_(saved[full_name].to(device=layer.weight.device, dtype=layer.weight.dtype))


# ============================================================
# BESA-style allocation search
# ============================================================

def default_caps_70() -> Dict[str, Tuple[float, float]]:
    return {
        "q_proj": (0.35, 0.60),
        "k_proj": (0.35, 0.60),
        "v_proj": (0.25, 0.55),
        "o_proj": (0.15, 0.45),
        "gate_proj": (0.70, 0.92),
        "up_proj": (0.70, 0.92),
        "down_proj": (0.35, 0.68),
    }


def default_caps_general() -> Dict[str, Tuple[float, float]]:
    return {
        "q_proj": (0.20, 0.65),
        "k_proj": (0.20, 0.65),
        "v_proj": (0.20, 0.60),
        "o_proj": (0.10, 0.50),
        "gate_proj": (0.35, 0.92),
        "up_proj": (0.35, 0.92),
        "down_proj": (0.15, 0.70),
    }


def parse_caps(raw: str, use_70_defaults: bool = True) -> Dict[str, Tuple[float, float]]:
    caps = default_caps_70() if use_70_defaults else default_caps_general()
    raw = raw.strip()
    if not raw:
        return caps
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, val = item.split(":")
        lo_s, hi_s = val.split("-")
        lo = float(lo_s)
        hi = float(hi_s)
        if not (0.0 <= lo <= hi < 1.0):
            raise ValueError(f"Bad cap {item}. Need 0 <= lo <= hi < 1.")
        caps[key.strip()] = (lo, hi)
    return caps


def weighted_average_sparsity(rates: Dict[str, float], layer_infos: List[Tuple[str, str, nn.Linear]]) -> float:
    total = 0
    pruned = 0.0
    for full_name, _local_name, layer in layer_infos:
        n = layer.weight.numel()
        total += n
        pruned += float(rates[full_name]) * n
    return pruned / float(total)


def project_rates_to_target(
    raw_rates: Dict[str, float],
    layer_infos: List[Tuple[str, str, nn.Linear]],
    caps: Dict[str, Tuple[float, float]],
    target: float,
) -> Dict[str, float]:
    def clipped(offset: float) -> Dict[str, float]:
        out = {}
        for full_name, _local_name, _layer in layer_infos:
            suf = layer_suffix(full_name)
            lo, hi = caps.get(suf, (0.0, 0.95))
            out[full_name] = min(hi, max(lo, raw_rates[full_name] + offset))
        return out

    lo_off, hi_off = -1.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo_off + hi_off)
        r = clipped(mid)
        avg = weighted_average_sparsity(r, layer_infos)
        if avg < target:
            lo_off = mid
        else:
            hi_off = mid
    return clipped(0.5 * (lo_off + hi_off))


def make_allocation_candidates(
    layer_infos: List[Tuple[str, str, nn.Linear]],
    target_sparsity: float,
    caps: Dict[str, Tuple[float, float]],
    max_candidates: int,
) -> List[Dict[str, float]]:
    templates: List[Dict[str, float]] = []

    def template(values: Dict[str, float]) -> Dict[str, float]:
        return {full_name: values.get(layer_suffix(full_name), target_sparsity) for full_name, _local_name, _layer in layer_infos}

    templates.append({full_name: target_sparsity for full_name, _local_name, _layer in layer_infos})
    templates.append(template({"q_proj": target_sparsity - 0.12, "k_proj": target_sparsity - 0.12, "v_proj": target_sparsity - 0.20, "o_proj": target_sparsity - 0.35, "gate_proj": target_sparsity + 0.18, "up_proj": target_sparsity + 0.18, "down_proj": target_sparsity - 0.18}))
    templates.append(template({"q_proj": target_sparsity - 0.18, "k_proj": target_sparsity - 0.18, "v_proj": target_sparsity - 0.25, "o_proj": target_sparsity - 0.38, "gate_proj": target_sparsity + 0.23, "up_proj": target_sparsity + 0.23, "down_proj": target_sparsity - 0.12}))
    templates.append(template({"q_proj": target_sparsity - 0.05, "k_proj": target_sparsity - 0.05, "v_proj": target_sparsity - 0.20, "o_proj": target_sparsity - 0.30, "gate_proj": target_sparsity + 0.17, "up_proj": target_sparsity + 0.17, "down_proj": target_sparsity - 0.30}))
    templates.append(template({"q_proj": target_sparsity - 0.02, "k_proj": target_sparsity - 0.02, "v_proj": target_sparsity - 0.28, "o_proj": target_sparsity - 0.35, "gate_proj": target_sparsity + 0.16, "up_proj": target_sparsity + 0.16, "down_proj": target_sparsity - 0.25}))
    templates.append(template({"q_proj": target_sparsity - 0.08, "k_proj": target_sparsity - 0.08, "v_proj": target_sparsity - 0.12, "o_proj": target_sparsity - 0.22, "gate_proj": target_sparsity + 0.12, "up_proj": target_sparsity + 0.12, "down_proj": target_sparsity - 0.10}))

    projected = [project_rates_to_target(t, layer_infos, caps, target_sparsity) for t in templates]

    unique: List[Dict[str, float]] = []
    seen = set()
    for r in projected:
        key = tuple(round(r[full_name], 4) for full_name, _local_name, _layer in layer_infos)
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique[:max_candidates]


def print_allocation(title: str, rates: Dict[str, float], layer_infos: List[Tuple[str, str, nn.Linear]]) -> None:
    avg = weighted_average_sparsity(rates, layer_infos)
    print(f"  {title}: weighted avg sparsity={100.0 * avg:.2f}%")
    for full_name, _local_name, layer in layer_infos:
        print(f"    {full_name:<55s} {100.0 * rates[full_name]:6.2f}% shape={tuple(layer.weight.shape)}")


# ============================================================
# Sparse state and reconstruction layers
# ============================================================

def result_to_layer_state(result: SparseLayerResult, value_dtype: torch.dtype) -> Dict[str, Any]:
    mask_bool = result.mask.detach().cpu().bool()
    mask_packed = pack_bool_mask_rows(mask_bool)
    dense = result.dense_sparse_weight.detach().cpu()
    if dense.shape != mask_bool.shape:
        raise ValueError(f"dense/mask shape mismatch: dense={tuple(dense.shape)}, mask={tuple(mask_bool.shape)}")
    kept_values = dense[mask_bool].to(dtype=value_dtype).contiguous()
    return {
        "shape": list(result.original_shape),
        "mask": mask_packed,
        "mask_packing": "packedbits",
        "values": kept_values,
        "values_format": "kept_1d",
        "value_dtype": dtype_to_name(value_dtype),
        "sparsity": float(result.sparsity),
        "target_sparsity": float(result.target_sparsity),
        "pattern": str(result.pattern),
        "pruned_count": int(result.pruned_count),
        "total_count": int(result.total_count),
    }


def dense_weight_from_sparse_state_cpu(state: Dict[str, Any], dtype: torch.dtype) -> torch.Tensor:
    rows, cols = tuple(state["shape"])
    values = state["values"].detach().cpu().to(dtype=dtype)
    mask = unpack_bool_mask_rows_cpu(state["mask"], original_cols=cols)

    if values.ndim == 2:
        if tuple(values.shape) != (rows, cols):
            raise ValueError(f"Bad dense values shape {tuple(values.shape)} expected {(rows, cols)}")
        return sanitize_tensor(values * mask.to(values.dtype))

    n_kept = int(mask.sum().item())
    if values.numel() != n_kept:
        raise ValueError(f"Kept values count mismatch: values={values.numel():,}, mask_kept={n_kept:,}, shape={(rows, cols)}")

    w = torch.zeros((rows, cols), dtype=dtype)
    w[mask] = values
    return sanitize_tensor(w)


@torch.no_grad()
def dense_weight_from_sparse_state(state: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Critical safety: do all boolean indexing on CPU, then move dense tensor to CUDA.
    return dense_weight_from_sparse_state_cpu(state, dtype=dtype).to(device=device, dtype=dtype)


class TrainableSparseLinear(nn.Module):
    def __init__(self, original: nn.Linear, mask: torch.Tensor, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.in_features = int(original.in_features)
        self.out_features = int(original.out_features)
        self.register_buffer("mask", mask.detach().to(device=device, dtype=torch.bool).contiguous())
        w0 = original.weight.detach().to(device=device, dtype=dtype) * self.mask.to(dtype)
        self.weight = nn.Parameter(w0.contiguous(), requires_grad=True)
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.detach().to(device=device, dtype=dtype), requires_grad=False)
        else:
            self.bias = None

    def clamp_mask(self) -> None:
        with torch.no_grad():
            self.weight.data.mul_(self.mask.to(self.weight.dtype))
            self.weight.data = sanitize_tensor(self.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.mask.to(self.weight.dtype)
        bias = self.bias
        if bias is not None:
            bias = bias.to(dtype=x.dtype, device=x.device)
        return F.linear(x, w.to(dtype=x.dtype, device=x.device), bias)


@torch.no_grad()
def apply_dense_weights_from_states(layer_infos: List[Tuple[str, str, nn.Linear]], states: Dict[str, Dict[str, Any]], device: torch.device) -> None:
    for full_name, _local_name, layer in layer_infos:
        w = dense_weight_from_sparse_state(states[full_name], device=device, dtype=layer.weight.dtype)
        layer.weight.data.copy_(w.to(layer.weight.device, dtype=layer.weight.dtype))


# ============================================================
# Candidate compression and scoring
# ============================================================

@torch.no_grad()
def compress_block_with_rates(
    layer_infos: List[Tuple[str, str, nn.Linear]],
    Hs: Dict[str, torch.Tensor],
    hessian_devices: Dict[str, torch.device],
    rates: Dict[str, float],
    pattern: str,
    percdamp: float,
    blocksize: int,
    mask_blocksize: int,
    act_order: bool,
    value_dtype: torch.dtype,
    verbose: bool,
) -> Dict[str, Dict[str, Any]]:
    states: Dict[str, Dict[str, Any]] = {}
    for li, (full_name, _local_name, layer) in enumerate(layer_infos, start=1):
        print(f"\n    [{li}/{len(layer_infos)}] Pruning {full_name}")
        print(f"      shape: {tuple(layer.weight.shape)}")
        print(f"      chosen sparsity: {100.0 * rates[full_name]:.2f}%")
        hdev = hessian_devices[full_name]
        result = sparsegpt_prune_linear(
            layer=layer,
            H=Hs[full_name],
            sparsity=float(rates[full_name]),
            pattern=pattern,
            percdamp=percdamp,
            blocksize=blocksize,
            mask_blocksize=mask_blocksize,
            compress_device=hdev,
            act_order=act_order,
            verbose=verbose,
        )
        states[full_name] = result_to_layer_state(result, value_dtype=value_dtype)
        print(
            f"      actual sparsity : {100.0 * result.sparsity:.2f}%\n"
            f"      values stored   : {tuple(states[full_name]['values'].shape)} kept values\n"
            f"      mask stored     : {tuple(states[full_name]['mask'].shape)}"
        )
        del result
        cleanup()
    return states


@torch.no_grad()
def block_relative_mse(y_dense: torch.Tensor, y_comp: torch.Tensor) -> float:
    yd = sanitize_tensor(y_dense.float())
    yc = sanitize_tensor(y_comp.float())
    num = torch.mean((yd - yc) ** 2).item()
    den = torch.mean(yd ** 2).item() + 1e-12
    return num / den


# ============================================================
# Regional reconstruction over surviving weights
# ============================================================

def install_trainable_sparse_layers(
    block: nn.Module,
    layer_infos: List[Tuple[str, str, nn.Linear]],
    states: Dict[str, Dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    for full_name, local_name, old_layer in layer_infos:
        st = states[full_name]
        rows, cols = tuple(st["shape"])
        mask_cpu = unpack_bool_mask_rows_cpu(st["mask"], original_cols=cols)

        tmp = nn.Linear(cols, rows, bias=old_layer.bias is not None, device=device, dtype=dtype)
        w = dense_weight_from_sparse_state(st, device=device, dtype=dtype)
        tmp.weight.data.copy_(w)
        if old_layer.bias is not None:
            tmp.bias.data.copy_(old_layer.bias.detach().to(device=device, dtype=dtype))

        mod = TrainableSparseLinear(tmp, mask=mask_cpu, device=device, dtype=dtype)
        set_module_by_name(block, local_name, mod)


@torch.no_grad()
def extract_states_from_trainable_sparse(
    block: nn.Module,
    layer_infos: List[Tuple[str, str, nn.Linear]],
    states: Dict[str, Dict[str, Any]],
    value_dtype: torch.dtype,
) -> None:
    # Critical safety: move to CPU before boolean indexing.
    for full_name, local_name, _old_layer in layer_infos:
        mod = get_module_by_name(block, local_name)
        if not isinstance(mod, TrainableSparseLinear):
            raise TypeError(f"Expected TrainableSparseLinear at {local_name}, got {type(mod)}")
        mod.clamp_mask()
        mask_cpu = mod.mask.detach().cpu().bool()
        dense_cpu = mod.weight.detach().cpu()
        states[full_name]["values"] = dense_cpu[mask_cpu].to(dtype=value_dtype).contiguous()
        states[full_name]["values_format"] = "kept_1d"


@torch.no_grad()
def replace_trainable_sparse_with_dense_linear(
    block: nn.Module,
    layer_infos_original: List[Tuple[str, str, nn.Linear]],
    states: Dict[str, Dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    for full_name, local_name, old_layer in layer_infos_original:
        st = states[full_name]
        rows, cols = tuple(st["shape"])
        dense = nn.Linear(cols, rows, bias=old_layer.bias is not None, device=device, dtype=dtype)
        w_cpu = dense_weight_from_sparse_state_cpu(st, dtype=dtype)
        dense.weight.data.copy_(w_cpu.to(device=device, dtype=dtype))
        if old_layer.bias is not None:
            dense.bias.data.copy_(old_layer.bias.detach().to(device=device, dtype=dtype))
        set_module_by_name(block, local_name, dense)


def regional_reconstruct_surviving_weights(
    base: nn.Module,
    block: nn.Module,
    layer_infos: List[Tuple[str, str, nn.Linear]],
    states: Dict[str, Dict[str, Any]],
    hidden_cpu: torch.Tensor,
    y_dense_cpu: torch.Tensor,
    batch_size: int,
    main_device: torch.device,
    model_dtype: torch.dtype,
    steps: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    value_dtype: torch.dtype,
) -> None:
    if steps <= 0:
        return

    print(f"\n  Regional reconstruction: optimizing surviving weights for {steps} steps")
    print(f"  Reconstruction LR          : {lr}")
    print(f"  Reconstruction weight decay: {weight_decay}")
    print(f"  Reconstruction grad clip   : {grad_clip}")

    install_trainable_sparse_layers(block, layer_infos, states, device=main_device, dtype=model_dtype)

    params: List[nn.Parameter] = []
    trainable_modules: List[TrainableSparseLinear] = []
    for _full_name, local_name, _old_layer in layer_infos:
        mod = get_module_by_name(block, local_name)
        if isinstance(mod, TrainableSparseLinear):
            params.append(mod.weight)
            trainable_modules.append(mod)

    if not params:
        print("  [warn] No trainable sparse weights found; skipping reconstruction.")
        return

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    n = hidden_cpu.size(0)
    step_t0 = now()
    block.eval()

    for step in range(1, steps + 1):
        total_loss = 0.0
        total_batches = 0

        for i in range(0, n, batch_size):
            x = hidden_cpu[i:i + batch_size].to(main_device, dtype=model_dtype)
            y = y_dense_cpu[i:i + batch_size].to(main_device, dtype=model_dtype)

            optimizer.zero_grad(set_to_none=True)
            pred = forward_decoder_block_grad(base, block, x, model_dtype, use_autocast=True)
            loss = F.mse_loss(sanitize_tensor(pred.float()), sanitize_tensor(y.float()))

            if not torch.isfinite(loss):
                print("    [warn] Non-finite reconstruction loss; skipping this mini-batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

            optimizer.step()

            for mod in trainable_modules:
                mod.clamp_mask()

            total_loss += float(loss.item())
            total_batches += 1

        if step == 1 or step % 10 == 0 or step == steps:
            avg = total_loss / max(total_batches, 1)
            print(f"    recon step {step:4d}/{steps} loss={avg:.6e} elapsed={format_seconds(now() - step_t0)}")

    extract_states_from_trainable_sparse(block, layer_infos, states, value_dtype=value_dtype)
    replace_trainable_sparse_with_dense_linear(block, layer_infos, states, device=main_device, dtype=model_dtype)
    block.eval()


# ============================================================
# Checkpoint saving helpers
# ============================================================

def build_partial_noncompressed_state_dict(model: nn.Module, compressed_layer_names: Iterable[str]) -> Dict[str, torch.Tensor]:
    weight_keys = {name + ".weight" for name in compressed_layer_names}
    prefixes = [name + "." for name in compressed_layer_names]
    out: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if k in weight_keys:
            continue
        skip = False
        for p in prefixes:
            if k.startswith(p) and any(s in k for s in (".mask", ".values", ".weight")):
                skip = True
                break
        if skip:
            continue
        out[k] = v.detach().cpu()
    return out


def build_meta(
    args: argparse.Namespace,
    sparse_layers: Dict[str, Dict[str, Any]],
    suffixes: Tuple[str, ...],
    caps: Dict[str, Tuple[float, float]],
    script_t0: float,
) -> Dict[str, Any]:
    total_pruned = sum(int(st["pruned_count"]) for st in sparse_layers.values())
    total_weights = sum(int(st["total_count"]) for st in sparse_layers.values())
    actual_sparsity = total_pruned / float(total_weights) if total_weights else 0.0

    mask_bytes = sum(int(st["mask"].numel() * st["mask"].element_size()) for st in sparse_layers.values())
    values_bytes = sum(int(st["values"].numel() * st["values"].element_size()) for st in sparse_layers.values())
    raw_stored_bytes = mask_bytes + values_bytes
    dense_bf16_bytes = total_weights * 2

    return {
        "method": "hybrid_besa_sparsegpt_wandapp_sparse_only_surviving_weight_reconstruction_cuda_safe",
        "model_id": str(args.model_id),
        "target_sparsity": float(args.target_sparsity),
        "actual_total_sparsity": float(actual_sparsity),
        "pattern": str(args.pattern),
        "percdamp": float(args.percdamp),
        "blocksize": int(args.blocksize),
        "mask_blocksize": int(args.mask_blocksize),
        "act_order": bool(args.act_order),
        "max_candidates": int(args.max_candidates),
        "recon_steps": int(args.recon_steps),
        "recon_lr": float(args.recon_lr),
        "recon_weight_decay": float(args.recon_weight_decay),
        "recon_grad_clip": float(args.recon_grad_clip),
        "model_dtype": str(args.model_dtype),
        "hidden_cache_dtype": str(args.hidden_cache_dtype),
        "hessian_dtype": str(args.hessian_dtype),
        "value_dtype": str(args.value_dtype),
        "values_format": "kept_1d",
        "calibration_source": str(args.calib),
        "max_seq_len": int(args.max_seq_len),
        "large_layer_cpu_threshold": int(args.large_layer_cpu_threshold),
        "suffixes": list(suffixes),
        "caps": {k: [float(v[0]), float(v[1])] for k, v in caps.items()},
        "compressed_layers": int(len(sparse_layers)),
        "total_pruned_weights": int(total_pruned),
        "total_compressed_weights": int(total_weights),
        "mask_bytes": int(mask_bytes),
        "values_bytes": int(values_bytes),
        "raw_stored_bytes": int(raw_stored_bytes),
        "dense_bf16_bytes": int(dense_bf16_bytes),
        "raw_compression_vs_bf16": float(dense_bf16_bytes / raw_stored_bytes) if raw_stored_bytes else 0.0,
        "total_script_seconds": float(now() - script_t0),
        "note": "CPU-safe sparse-only high-sparsity checkpoint storing only 1D kept values in row-major mask order.",
    }


def save_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    out_path: str,
    model_id: str,
    meta: Dict[str, Any],
    sparse_layers: Dict[str, Dict[str, Any]],
    keep_dequantized_state_dict: bool,
    sparse_only_state: bool = False,
) -> None:
    if sparse_only_state:
        model_state = {}
    elif keep_dequantized_state_dict:
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    else:
        model_state = build_partial_noncompressed_state_dict(model, sparse_layers.keys())

    ckpt = {
        "format": "hf_hybrid_sparse_reconstruct",
        "model_id": model_id,
        "config": model.config.to_dict() if hasattr(model, "config") else None,
        "model": model_state,
        "hybrid_sparse_meta": meta,
        "hybrid_sparse_layers": sparse_layers,
        "compression_meta": meta,
    }
    try:
        ckpt["tokenizer_name_or_path"] = tokenizer.name_or_path
    except Exception:
        ckpt["tokenizer_name_or_path"] = model_id

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = now()
    torch.save(ckpt, out)
    print(f"Saved checkpoint in {format_seconds(now() - t0)}: {out}")

    meta_path = Path(str(out) + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta JSON: {meta_path}")


def save_partial_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    sparse_layers: Dict[str, Dict[str, Any]],
    suffixes: Tuple[str, ...],
    caps: Dict[str, Tuple[float, float]],
    script_t0: float,
    completed_block: int,
    sparse_only_state: bool = False,
) -> None:
    partial_path = str(args.out) + ".partial.pt"
    meta = build_meta(args, sparse_layers, suffixes, caps, script_t0)
    meta["partial"] = True
    meta["completed_block"] = int(completed_block)
    meta["sparse_only_state"] = bool(sparse_only_state)
    meta["resume_note"] = "Partial checkpoint saved after completing this decoder block."
    print(f"\n  Saving partial checkpoint after block {completed_block}: {partial_path}")
    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        out_path=partial_path,
        model_id=str(args.model_id),
        meta=meta,
        sparse_layers=sparse_layers,
        keep_dequantized_state_dict=bool(args.keep_dequantized_state_dict),
        sparse_only_state=sparse_only_state,
    )


# ============================================================
# Main blockwise compressor
# ============================================================

@torch.no_grad()
def compress_model_blockwise(
    model: nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    calib_tokens: torch.Tensor,
    target_sparsity: float,
    pattern: str,
    batch_size: int,
    main_device: torch.device,
    model_dtype: torch.dtype,
    hidden_cache_dtype: torch.dtype,
    hessian_dtype: torch.dtype,
    percdamp: float,
    blocksize: int,
    mask_blocksize: int,
    act_order: bool,
    large_layer_cpu_threshold: int,
    suffixes: Tuple[str, ...],
    include: str,
    exclude: str,
    caps: Dict[str, Tuple[float, float]],
    max_candidates: int,
    recon_steps: int,
    recon_lr: float,
    recon_weight_decay: float,
    recon_grad_clip: float,
    value_dtype: torch.dtype,
    store_debug_dense_weight: bool,
    script_t0: float,
) -> Dict[str, Dict[str, Any]]:
    base = get_base_decoder_model(model)
    layers_name, decoder_layers = get_decoder_layers(base)
    print(f"\nDecoder layers found: {layers_name}")
    print(f"Number of decoder blocks: {len(decoder_layers)}")

    existing_sparse_layers: Dict[str, Dict[str, Any]] = {}
    start_block = 0

    if args.resume_from_partial:
        partial_path = str(args.out) + ".partial.pt"
        if os.path.exists(partial_path):
            print(f"\n[resume] Loading partial checkpoint: {partial_path}")
            partial = torch.load(partial_path, map_location="cpu")
            existing_sparse_layers = dict(partial.get("hybrid_sparse_layers", {}))
            partial_meta = partial.get("hybrid_sparse_meta", partial.get("compression_meta", {}))
            start_block = int(partial_meta.get("completed_block", -1)) + 1
            print(f"[resume] Found {len(existing_sparse_layers)} compressed layers. Starting at block {start_block}.")
        else:
            print(f"[resume] Requested but no partial checkpoint found: {partial_path}")

    hidden_cpu = compute_initial_hidden(model, calib_tokens, batch_size, main_device, hidden_cache_dtype)

    root_prefix = "model" if hasattr(model, "model") else "transformer"
    sparse_layers: Dict[str, Dict[str, Any]] = dict(existing_sparse_layers)

    if start_block > 0:
        print(f"\n[resume] Replaying blocks 0..{start_block - 1} to rebuild hidden cache...")
        for bi in range(start_block):
            block = decoder_layers[bi]
            block_prefix = f"{root_prefix}.{layers_name}.{bi}"
            layer_infos = find_block_linears(block, block_prefix, suffixes, include, exclude)
            block_states = {full_name: sparse_layers[full_name] for full_name, _local_name, _layer in layer_infos if full_name in sparse_layers}
            if len(block_states) != len(layer_infos):
                raise RuntimeError(f"Cannot resume: block {bi} expected {len(layer_infos)} layer states but found {len(block_states)}.")
            apply_dense_weights_from_states(layer_infos, block_states, main_device)
            hidden_cpu = run_block_on_hidden_cache(base, block, hidden_cpu, batch_size, main_device, model_dtype, hidden_cache_dtype, desc=f"resume block {bi} sparse forward")
            cleanup()

    for bi, block in enumerate(decoder_layers):
        if bi < start_block:
            continue

        block_prefix = f"{root_prefix}.{layers_name}.{bi}"
        layer_infos = find_block_linears(block, block_prefix, suffixes, include, exclude)

        print("\n" + "=" * 100)
        print(f"BLOCK {bi}/{len(decoder_layers) - 1}")
        print(f"Selected linear layers in block: {len(layer_infos)}")
        print(f"Hidden cache entering block: {tuple(hidden_cpu.shape)}, dtype={hidden_cpu.dtype}")
        print_cuda_memory("CUDA memory")
        for full_name, _local_name, layer in layer_infos:
            print(f"  - {full_name}: {tuple(layer.weight.shape)} dtype={layer.weight.dtype}")

        try:
            if not layer_infos:
                print("  No selected layers in this block. Forwarding dense block.")
                hidden_cpu = run_block_on_hidden_cache(base, block, hidden_cpu, batch_size, main_device, model_dtype, hidden_cache_dtype, f"block {bi} dense forward")
                continue

            print("\n  Computing dense block output target Y_dense...")
            y_dense_cpu = run_block_on_hidden_cache(base, block, hidden_cpu, batch_size, main_device, model_dtype, hidden_cache_dtype, f"block {bi} dense target")

            Hs, nsamples, hessian_devices = collect_block_hessians(
                base=base,
                block=block,
                layer_infos=layer_infos,
                hidden_cpu=hidden_cpu,
                batch_size=batch_size,
                main_device=main_device,
                model_dtype=model_dtype,
                hessian_dtype=hessian_dtype,
                large_layer_cpu_threshold=large_layer_cpu_threshold,
            )
            for name in Hs:
                print(f"  H[{name}] shape={tuple(Hs[name].shape)} samples={nsamples[name]} device={Hs[name].device}")

            original_weights = save_original_block_weights(layer_infos)
            candidates = make_allocation_candidates(layer_infos, target_sparsity, caps, max_candidates)
            print(f"\n  Generated {len(candidates)} BESA-style allocation candidates.")

            best_idx = -1
            best_loss = float("inf")
            best_rates: Optional[Dict[str, float]] = None

            for ci, rates in enumerate(candidates, start=1):
                print("\n" + "-" * 100)
                print(f"  Candidate {ci}/{len(candidates)}")
                print_allocation("allocation", rates, layer_infos)

                restore_block_weights(layer_infos, original_weights)
                cand_t0 = now()

                states = compress_block_with_rates(
                    layer_infos=layer_infos,
                    Hs=Hs,
                    hessian_devices=hessian_devices,
                    rates=rates,
                    pattern=pattern,
                    percdamp=percdamp,
                    blocksize=blocksize,
                    mask_blocksize=mask_blocksize,
                    act_order=act_order,
                    value_dtype=value_dtype,
                    verbose=False,
                )

                y_comp_cpu = run_block_on_hidden_cache(base, block, hidden_cpu, batch_size, main_device, model_dtype, hidden_cache_dtype, f"candidate {ci} sparse block")
                rel = block_relative_mse(y_dense_cpu, y_comp_cpu)
                avg_sp = weighted_average_sparsity(rates, layer_infos)
                print(f"  Candidate {ci} result:\n    block relative MSE : {rel:.8e}\n    weighted sparsity  : {100.0 * avg_sp:.2f}%\n    time               : {format_seconds(now() - cand_t0)}")

                if rel < best_loss:
                    best_loss = rel
                    best_idx = ci
                    best_rates = copy.deepcopy(rates)

                del y_comp_cpu
                del states
                cleanup()

            assert best_rates is not None
            print("\n" + "-" * 100)
            print(f"  Best candidate for block {bi}: {best_idx}")
            print(f"  Best block relative MSE: {best_loss:.8e}")
            print_allocation("best allocation", best_rates, layer_infos)

            restore_block_weights(layer_infos, original_weights)
            print("\n  Recompressing block with best allocation permanently...")
            final_states = compress_block_with_rates(
                layer_infos=layer_infos,
                Hs=Hs,
                hessian_devices=hessian_devices,
                rates=best_rates,
                pattern=pattern,
                percdamp=percdamp,
                blocksize=blocksize,
                mask_blocksize=mask_blocksize,
                act_order=act_order,
                value_dtype=value_dtype,
                verbose=True,
            )

            if recon_steps > 0:
                with torch.enable_grad():
                    regional_reconstruct_surviving_weights(
                        base=base,
                        block=block,
                        layer_infos=layer_infos,
                        states=final_states,
                        hidden_cpu=hidden_cpu,
                        y_dense_cpu=y_dense_cpu,
                        batch_size=batch_size,
                        main_device=main_device,
                        model_dtype=model_dtype,
                        steps=recon_steps,
                        lr=recon_lr,
                        weight_decay=recon_weight_decay,
                        grad_clip=recon_grad_clip,
                        value_dtype=value_dtype,
                    )
                apply_dense_weights_from_states(layer_infos, final_states, main_device)

            for full_name, _local_name, _layer in layer_infos:
                st = final_states[full_name]
                if store_debug_dense_weight:
                    st["dense_debug_weight"] = dense_weight_from_sparse_state_cpu(st, dtype=torch.float16)
                sparse_layers[full_name] = st

            print("\n  Running final sparse block to create next hidden cache...")
            hidden_cpu = run_block_on_hidden_cache(base, block, hidden_cpu, batch_size, main_device, model_dtype, hidden_cache_dtype, f"block {bi} final sparse forward")

            del Hs, nsamples, hessian_devices, y_dense_cpu, original_weights
            cleanup()

            if args.save_partial_every_block:
                save_partial_checkpoint(model, tokenizer, args, sparse_layers, suffixes, caps, script_t0, completed_block=bi, sparse_only_state=bool(args.partial_sparse_only_state))

            print(f"Finished block {bi}. Total elapsed: {format_seconds(now() - script_t0)}")
            print_cuda_memory("CUDA memory")

        except Exception as exc:
            print("\n[error] Exception during block compression.")
            print(f"[error] block={bi}")
            print(f"[error] {type(exc).__name__}: {exc}")
            traceback.print_exc()
            if sparse_layers:
                print("[error] Saving emergency sparse-only partial checkpoint before re-raising...")
                try:
                    save_partial_checkpoint(model, tokenizer, args, sparse_layers, suffixes, caps, script_t0, completed_block=bi - 1, sparse_only_state=True)
                except Exception as save_exc:
                    print(f"[error] Emergency save also failed: {type(save_exc).__name__}: {save_exc}")
                    print("[error] If this was a CUDA device-side assert, restart the Python process and use the latest .partial.pt saved before this block.")
            raise

    return sparse_layers


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--target_sparsity", type=float, default=0.70)
    parser.add_argument("--pattern", type=str, default="unstructured")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--model_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hidden_cache_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hessian_dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--value_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--percdamp", type=float, default=0.1)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--mask_blocksize", type=int, default=128)
    parser.add_argument("--act_order", action="store_true")

    parser.add_argument("--suffixes", type=str, default="")
    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")
    parser.add_argument("--caps", type=str, default="")
    parser.add_argument("--max_candidates", type=int, default=5)

    parser.add_argument("--recon_steps", type=int, default=100)
    parser.add_argument("--recon_lr", type=float, default=2e-5)
    parser.add_argument("--recon_weight_decay", type=float, default=0.0)
    parser.add_argument("--recon_grad_clip", type=float, default=1.0)

    parser.add_argument("--large_layer_cpu_threshold", type=int, default=8192)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--attn_implementation", type=str, default="eager")

    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--keep_dequantized_state_dict", action="store_true")
    parser.add_argument("--store_debug_dense_weight", action="store_true")

    parser.add_argument("--save_partial_every_block", action="store_true", default=True)
    parser.add_argument("--no_save_partial_every_block", dest="save_partial_every_block", action="store_false")
    parser.add_argument("--resume_from_partial", action="store_true")
    parser.add_argument("--partial_sparse_only_state", action="store_true", default=True)
    parser.add_argument("--no_partial_sparse_only_state", dest="partial_sparse_only_state", action="store_false")

    args = parser.parse_args()
    script_t0 = now()

    if not (0.0 <= args.target_sparsity < 1.0):
        raise ValueError("--target_sparsity must be in [0,1).")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    main_device = torch.device(args.device)
    model_dtype = parse_dtype(args.model_dtype)
    hidden_cache_dtype = parse_dtype(args.hidden_cache_dtype)
    hessian_dtype = parse_dtype(args.hessian_dtype)
    value_dtype = parse_dtype(args.value_dtype)
    suffixes = parse_suffixes(args.suffixes)
    caps = parse_caps(args.caps, use_70_defaults=True)

    print("=" * 100)
    print("Hybrid BESA / SparseGPT / Wanda++ sparse-only compression CUDA-SAFE")
    print("=" * 100)
    print(f"model_id                  : {args.model_id}")
    print(f"calib                     : {args.calib}")
    print(f"out                       : {args.out}")
    print(f"device                    : {main_device}")
    print(f"model_dtype               : {model_dtype}")
    print(f"hidden_cache_dtype        : {hidden_cache_dtype}")
    print(f"hessian_dtype             : {hessian_dtype}")
    print(f"value_dtype               : {value_dtype}")
    print(f"target_sparsity           : {args.target_sparsity}")
    print(f"pattern                   : {args.pattern}")
    print(f"blocksize                 : {args.blocksize}")
    print(f"mask_blocksize            : {args.mask_blocksize}")
    print(f"percdamp                  : {args.percdamp}")
    print(f"max_candidates            : {args.max_candidates}")
    print(f"recon_steps               : {args.recon_steps}")
    print(f"recon_lr                  : {args.recon_lr}")
    print(f"recon_weight_decay        : {args.recon_weight_decay}")
    print(f"recon_grad_clip           : {args.recon_grad_clip}")
    print(f"large_layer_cpu_threshold : {args.large_layer_cpu_threshold}")
    print(f"save_partial_every_block  : {args.save_partial_every_block}")
    print(f"resume_from_partial       : {args.resume_from_partial}")
    print(f"partial_sparse_only_state : {args.partial_sparse_only_state}")
    print(f"suffixes                  : {suffixes}")
    print(f"caps                      : {caps}")

    print("\nLoading tokenizer/model...")
    t0 = now()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=bool(args.trust_remote_code))
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": model_dtype,
        "trust_remote_code": bool(args.trust_remote_code),
        "low_cpu_mem_usage": bool(args.low_cpu_mem_usage),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()
    model.to(main_device)
    if hasattr(model, "config"):
        model.config.use_cache = False
    print(f"Loaded model in {format_seconds(now() - t0)}")
    print_cuda_memory("CUDA memory after model load")

    calib_tokens = load_calibration_tokens(args.calib)
    print(f"Loaded calibration tokens: {tuple(calib_tokens.shape)}")
    if args.max_seq_len > 0 and calib_tokens.size(1) > args.max_seq_len:
        calib_tokens = calib_tokens[:, :args.max_seq_len]
        print(f"Trimmed calibration sequence length to {args.max_seq_len}")

    sparse_layers = compress_model_blockwise(
        model=model,
        tokenizer=tokenizer,
        args=args,
        calib_tokens=calib_tokens,
        target_sparsity=float(args.target_sparsity),
        pattern=str(args.pattern),
        batch_size=int(args.batch_size),
        main_device=main_device,
        model_dtype=model_dtype,
        hidden_cache_dtype=hidden_cache_dtype,
        hessian_dtype=hessian_dtype,
        percdamp=float(args.percdamp),
        blocksize=int(args.blocksize),
        mask_blocksize=int(args.mask_blocksize),
        act_order=bool(args.act_order),
        large_layer_cpu_threshold=int(args.large_layer_cpu_threshold),
        suffixes=suffixes,
        include=str(args.include),
        exclude=str(args.exclude),
        caps=caps,
        max_candidates=int(args.max_candidates),
        recon_steps=int(args.recon_steps),
        recon_lr=float(args.recon_lr),
        recon_weight_decay=float(args.recon_weight_decay),
        recon_grad_clip=float(args.recon_grad_clip),
        value_dtype=value_dtype,
        store_debug_dense_weight=bool(args.store_debug_dense_weight),
        script_t0=script_t0,
    )

    meta = build_meta(args, sparse_layers, suffixes, caps, script_t0)

    print("\nSaving checkpoint...")
    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        out_path=args.out,
        model_id=args.model_id,
        meta=meta,
        sparse_layers=sparse_layers,
        keep_dequantized_state_dict=bool(args.keep_dequantized_state_dict),
        sparse_only_state=bool(args.partial_sparse_only_state),
    )

    print("\nDone.")
    print(f"Compressed layers              : {len(sparse_layers)}")
    print(f"Total selected weights         : {meta['total_compressed_weights']:,}")
    print(f"Pruned weights                 : {meta['total_pruned_weights']:,}")
    print(f"Actual sparsity                : {100.0 * meta['actual_total_sparsity']:.2f}%")
    print(f"mask bytes                     : {meta['mask_bytes']:,}")
    print(f"values bytes                   : {meta['values_bytes']:,}")
    print(f"raw stored bytes               : {meta['raw_stored_bytes']:,}")
    print(f"dense BF16 bytes               : {meta['dense_bf16_bytes']:,}")
    print(f"raw compression vs BF16        : {meta['raw_compression_vs_bf16']:.2f}x")
    print(f"Total script time              : {format_seconds(now() - script_t0)}")
    print_cuda_memory("CUDA memory")


if __name__ == "__main__":
    main()
