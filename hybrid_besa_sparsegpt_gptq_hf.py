#!/usr/bin/env python3
"""
Hybrid BESA/Wanda++/SparseGPT/GPTQ compression for Hugging Face decoder-only LMs.

This is a complete fixed rewrite.

Fixes versus your failing version:
    - Regional reconstruction no longer calls get_module_by_name(base, "model.layers...").
      It resolves module names relative to the current block.
    - Reconstruction forward pass is differentiable. The normal calibration/eval forward path
      still uses torch.no_grad(), but reconstruction uses gradients for scales/zero_points.
    - Trainable wrappers are installed inside the current decoder block using local names like
      self_attn.q_proj, mlp.down_proj.
    - After reconstruction, wrappers are replaced by normal dense nn.Linear layers reconstructed
      from qweight/mask/scales/zero_points.

Checkpoint format:
    ckpt["joint_sparsegpt_gptq_layers"][layer_name]

Runtime reconstruction:
    W = mask * ((q - zero_point) * scale)
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import time
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
# Module name helpers
# ============================================================

def get_module_by_name(root: nn.Module, full_name: str) -> nn.Module:
    obj = root
    if full_name == "":
        return obj
    for part in full_name.split("."):
        obj = getattr(obj, part)
    return obj


def set_module_by_name(root: nn.Module, full_name: str, new_module: nn.Module) -> None:
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
    """
    Returns:
        (full_name_for_checkpoint, local_name_inside_block, module)

    For Mistral full name example:
        model.layers.0.self_attn.q_proj
    Local name example:
        self_attn.q_proj
    """
    out: List[Tuple[str, str, nn.Linear]] = []
    for local_name, mod in block.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        full_name = f"{block_prefix}.{local_name}" if local_name else block_prefix
        if should_compress_name(full_name, suffixes=suffixes, include=include, exclude=exclude):
            out.append((full_name, local_name, mod))
    return out


# ============================================================
# Quantization containers and helpers
# ============================================================

@dataclass
class QuantParams:
    scale: torch.Tensor
    zero_point: torch.Tensor
    qmin: int
    qmax: int
    symmetric: bool


@dataclass
class JointLayerResult:
    qweight_uint8: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor
    mask: torch.Tensor
    dequant_masked_weight: torch.Tensor
    bits: int
    groupsize: int
    packing: str
    mask_packing: str
    original_shape: Tuple[int, int]
    symmetric: bool
    sparsity: float
    target_sparsity: float
    pattern: str
    pruned_count: int
    total_count: int


def get_group_bounds(col_idx: int, cols: int, groupsize: int) -> Tuple[int, int]:
    if groupsize == -1 or groupsize >= cols:
        return 0, cols
    g0 = (col_idx // groupsize) * groupsize
    return g0, min(g0 + groupsize, cols)


def get_num_groups(cols: int, groupsize: int) -> int:
    if groupsize == -1 or groupsize >= cols:
        return 1
    return math.ceil(cols / groupsize)


def get_group_index(col_idx: int, cols: int, groupsize: int) -> int:
    if groupsize == -1 or groupsize >= cols:
        return 0
    return col_idx // groupsize


def is_group_start(col_idx: int, groupsize: int) -> bool:
    if groupsize == -1:
        return col_idx == 0
    return (col_idx % groupsize) == 0


def pack_4bit_rows(qweight_uint8: torch.Tensor) -> torch.Tensor:
    """
    Pack two 4-bit symbols per byte.

    Robustness note:
        In exact math qweight is already in [0, 15]. In real CUDA/BF16 runs,
        a rare NaN/Inf in an upstream Hessian update can convert to uint8=255.
        We clamp here as a final storage guard, but the quantization path below
        also sanitizes q before uint8 conversion.
    """
    if qweight_uint8.dtype != torch.uint8:
        qweight_uint8 = qweight_uint8.to(torch.uint8)

    bad = torch.count_nonzero(qweight_uint8 > 15).item()
    if bad:
        print(f"[warn] pack_4bit_rows: clamping {bad:,} qweight values > 15 to 15")
        qweight_uint8 = qweight_uint8.clamp_(0, 15)

    rows, cols = qweight_uint8.shape
    packed_cols = (cols + 1) // 2
    packed = torch.zeros((rows, packed_cols), dtype=torch.uint8, device=qweight_uint8.device)
    even = qweight_uint8[:, 0::2]
    odd = qweight_uint8[:, 1::2]
    packed[:, :even.size(1)] |= even
    if odd.numel() > 0:
        packed[:, :odd.size(1)] |= odd << 4
    return packed


def unpack_4bit_rows(packed: torch.Tensor, original_cols: int) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        packed = packed.to(torch.uint8)
    rows, packed_cols = packed.shape
    out = torch.zeros((rows, packed_cols * 2), dtype=torch.uint8, device=packed.device)
    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = (packed >> 4) & 0x0F
    return out[:, :original_cols]


def pack_bool_mask_rows(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    rows, cols = mask.shape
    packed_cols = (cols + 7) // 8
    padded_cols = packed_cols * 8
    if padded_cols != cols:
        pad = torch.zeros((rows, padded_cols - cols), dtype=torch.bool, device=mask.device)
        mask = torch.cat([mask, pad], dim=1)
    mask_u8 = mask.to(torch.uint8).view(rows, packed_cols, 8)
    shifts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=mask.device)
    return (mask_u8 * shifts.view(1, 1, 8)).sum(dim=2).to(torch.uint8)


def unpack_bool_mask_rows(packed: torch.Tensor, original_cols: int) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        packed = packed.to(torch.uint8)
    rows, packed_cols = packed.shape
    shifts = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8, device=packed.device)
    bits = ((packed.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 1).bool()
    return bits.view(rows, packed_cols * 8)[:, :original_cols]


def maybe_pack_qweight(qweight_uint8: torch.Tensor, bits: int, packing: str) -> torch.Tensor:
    if packing == "uint8":
        return qweight_uint8.cpu()
    if packing == "packed4":
        if bits != 4:
            raise ValueError("packed4 only works with bits=4.")
        return pack_4bit_rows(qweight_uint8).cpu()
    raise ValueError(f"Unsupported packing: {packing}")


def maybe_pack_mask(mask: torch.Tensor, mask_packing: str) -> torch.Tensor:
    if mask_packing == "bool":
        return mask.cpu().bool()
    if mask_packing == "packedbits":
        return pack_bool_mask_rows(mask).cpu()
    raise ValueError(f"Unsupported mask packing: {mask_packing}")


def make_quant_params_for_slice(
    W_slice: torch.Tensor,
    bits: int,
    symmetric: bool,
    mask_slice: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> QuantParams:
    if bits < 2 or bits > 8:
        raise ValueError("bits must be in [2,8].")
    qmin = 0
    qmax = (1 << bits) - 1

    if mask_slice is not None:
        if mask_slice.shape != W_slice.shape:
            raise ValueError("mask_slice shape must match W_slice.")
        mask_bool = mask_slice.bool()
        has_kept = mask_bool.any(dim=1, keepdim=True)
    else:
        mask_bool = None
        has_kept = None

    if symmetric:
        if mask_slice is None:
            max_abs = W_slice.abs().amax(dim=1, keepdim=True).clamp(min=eps)
        else:
            masked_abs = torch.where(mask_bool, W_slice.abs(), torch.zeros_like(W_slice))
            max_abs_kept = masked_abs.amax(dim=1, keepdim=True)
            max_abs_all = W_slice.abs().amax(dim=1, keepdim=True)
            max_abs = torch.where(has_kept, max_abs_kept, max_abs_all).clamp(min=eps)
        mid = (qmin + qmax) / 2.0
        scale = max_abs / max(mid, 1.0)
        zero = torch.full_like(scale, mid)
        return QuantParams(scale, zero, qmin, qmax, True)

    if mask_slice is None:
        w_min = W_slice.amin(dim=1, keepdim=True)
        w_max = W_slice.amax(dim=1, keepdim=True)
    else:
        inf = torch.tensor(float("inf"), device=W_slice.device, dtype=W_slice.dtype)
        ninf = torch.tensor(float("-inf"), device=W_slice.device, dtype=W_slice.dtype)
        w_min_kept = torch.where(mask_bool, W_slice, inf).amin(dim=1, keepdim=True)
        w_max_kept = torch.where(mask_bool, W_slice, ninf).amax(dim=1, keepdim=True)
        w_min_all = W_slice.amin(dim=1, keepdim=True)
        w_max_all = W_slice.amax(dim=1, keepdim=True)
        w_min = torch.where(has_kept, w_min_kept, w_min_all)
        w_max = torch.where(has_kept, w_max_kept, w_max_all)

    same = (w_max - w_min).abs() < eps
    w_max = torch.where(same, w_min + eps, w_max)
    scale = ((w_max - w_min) / float(qmax - qmin)).clamp(min=eps)
    zero = torch.round(qmin - w_min / scale).clamp(qmin, qmax)
    return QuantParams(scale, zero, qmin, qmax, False)


def quantize_with_given_params(w_col: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    qmin = 0
    qmax = (1 << bits) - 1

    # CRUCIAL numerical guard:
    # NaN/Inf in scale/zero/w would otherwise become invalid uint8 symbols,
    # often 255, which breaks packed4 storage.
    w_safe = torch.nan_to_num(w_col, nan=0.0, posinf=0.0, neginf=0.0)
    scale_safe = torch.nan_to_num(scale, nan=1e-8, posinf=1e-8, neginf=1e-8).abs().clamp(min=1e-8)
    zero_safe = torch.nan_to_num(zero, nan=0.0, posinf=float(qmax), neginf=0.0).clamp(qmin, qmax)

    q = torch.round(w_safe / scale_safe + zero_safe)
    q = torch.nan_to_num(q, nan=0.0, posinf=float(qmax), neginf=0.0).clamp(qmin, qmax)

    q_int = q.to(torch.uint8)
    q_deq = (q.to(w_col.dtype) - zero_safe.to(w_col.dtype)) * scale_safe.to(w_col.dtype)
    q_deq = torch.nan_to_num(q_deq, nan=0.0, posinf=0.0, neginf=0.0)
    return q_int, q_deq


# ============================================================
# Mask selection
# ============================================================

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
    flat = score.reshape(-1)
    keep_idx = torch.topk(flat, k=n_keep, largest=True, sorted=False).indices
    mask = torch.zeros(total, dtype=torch.bool, device=score.device)
    mask[keep_idx] = True
    return mask.view(rows, cols)


@torch.no_grad()
def select_nm_mask_block_from_score(score: torch.Tensor, n_zero: int, m: int) -> torch.Tensor:
    rows, cols = score.shape
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
def compute_joint_mask_score_block(
    W_block: torch.Tensor,
    Hinv_diag_block: torch.Tensor,
    bits: int,
    symmetric: bool,
    quant_aware: bool,
    eps: float = 1e-12,
) -> torch.Tensor:
    diag = Hinv_diag_block.to(device=W_block.device, dtype=W_block.dtype).abs().clamp(min=eps)
    if not quant_aware:
        return (W_block.float() ** 2) / diag.float().view(1, -1)

    qp = make_quant_params_for_slice(W_block, bits=bits, symmetric=symmetric, mask_slice=None)
    scale = qp.scale.to(W_block.dtype)
    zero = qp.zero_point.to(W_block.dtype)
    q = torch.round(W_block / scale + zero).clamp(qp.qmin, qp.qmax)
    q_deq = (q.to(W_block.dtype) - zero) * scale
    prune_err2 = W_block.float() ** 2
    quant_err2 = (W_block.float() - q_deq.float()) ** 2
    return (prune_err2 - quant_err2) / diag.float().view(1, -1)


# ============================================================
# Hessian inverse
# ============================================================

@torch.no_grad()
def stable_inverse_cholesky(H: torch.Tensor, percdamp: float, max_tries: int = 12) -> Tuple[torch.Tensor, float]:
    if H.ndim != 2 or H.size(0) != H.size(1):
        raise ValueError(f"H must be square, got {tuple(H.shape)}")
    device = H.device
    n = H.size(0)
    H64 = H.to(torch.float64)
    H64 = torch.nan_to_num(H64, nan=0.0, posinf=0.0, neginf=0.0)
    H64 = 0.5 * (H64 + H64.T)
    diag = torch.diag(H64)
    diag_abs = diag.abs()
    diag_mean = max(float(diag_abs.mean().item()), 1e-12)
    diag_max = max(float(diag_abs.max().item()), diag_mean, 1e-12)
    ar = torch.arange(n, device=device)

    multipliers = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0][:max_tries]
    for mult in multipliers:
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


# ============================================================
# SparseGPT + GPTQ single layer
# ============================================================

@torch.no_grad()
def joint_sparsegpt_gptq_linear(
    layer: nn.Linear,
    H: torch.Tensor,
    bits: int,
    sparsity: float,
    pattern: str,
    percdamp: float,
    blocksize: int,
    mask_blocksize: int,
    groupsize: int,
    packing: str,
    mask_packing: str,
    symmetric: bool,
    act_order: bool,
    quant_aware_mask: bool,
    compress_device: torch.device,
    verbose: bool = True,
) -> JointLayerResult:
    if packing == "packed4" and bits != 4:
        raise ValueError("packed4 only supports bits=4.")

    original_device = layer.weight.device
    original_dtype = layer.weight.dtype
    W_orig = layer.weight.detach().to(device=compress_device, dtype=torch.float32).clone()
    W_orig = torch.nan_to_num(W_orig, nan=0.0, posinf=0.0, neginf=0.0)
    rows, cols = W_orig.shape

    if H.shape != (cols, cols):
        raise ValueError(f"H shape mismatch. Expected {(cols, cols)}, got {tuple(H.shape)}")

    H = torch.nan_to_num(H.to(compress_device), nan=0.0, posinf=0.0, neginf=0.0)

    nm = parse_nm_pattern(pattern)
    if nm is not None:
        n_zero, m = nm
        mask_blocksize_eff = m
        effective_sparsity = n_zero / float(m)
    else:
        n_zero, m = None, None
        mask_blocksize_eff = mask_blocksize
        effective_sparsity = sparsity

    if verbose:
        print(f"      bits              : {bits}")
        print(f"      target sparsity   : {effective_sparsity:.4f}")
        print(f"      groupsize         : {groupsize}")
        print(f"      blocksize         : {blocksize}")
        print(f"      mask blocksize    : {mask_blocksize_eff}")
        print(f"      quant-aware mask  : {quant_aware_mask}")
        print(f"      compression device: {compress_device}")

    t_inv = now()
    Hinv_chol_upper, used_damp = stable_inverse_cholesky(H, percdamp=percdamp)
    if verbose:
        print(f"      used damping      : {used_damp:.6e}")
        print(f"      inverse time      : {format_seconds(now() - t_inv)}")

    Hinv = Hinv_chol_upper.T @ Hinv_chol_upper
    Hinv = Hinv.to(device=compress_device, dtype=torch.float32)
    Hinv = torch.nan_to_num(Hinv, nan=0.0, posinf=0.0, neginf=0.0)

    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        invperm = torch.argsort(perm)
        W = W_orig[:, perm].contiguous()
        Hinv = Hinv[perm][:, perm].contiguous()
    else:
        invperm = None
        W = W_orig.clone()

    Q_deq_masked = torch.zeros_like(W)
    qweight_uint8 = torch.zeros((rows, cols), dtype=torch.uint8, device=compress_device)
    M_global = torch.ones((rows, cols), dtype=torch.bool, device=compress_device)

    ngroups = get_num_groups(cols, groupsize)
    scales = torch.zeros((rows, ngroups), dtype=torch.float32, device=compress_device)
    zero_points = torch.zeros((rows, ngroups), dtype=torch.float32, device=compress_device)

    selected_mask_until = -1
    Hinv_diag = torch.diag(Hinv).abs().clamp(min=1e-12)

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        W1 = torch.nan_to_num(W1, nan=0.0, posinf=0.0, neginf=0.0)
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2].contiguous()
        Hinv1 = torch.nan_to_num(Hinv1, nan=0.0, posinf=0.0, neginf=0.0)

        for local_i in range(count):
            global_col = i1 + local_i

            if global_col >= selected_mask_until:
                mb0 = global_col
                mb1 = min(mb0 + mask_blocksize_eff, cols)
                score = compute_joint_mask_score_block(
                    W_block=torch.nan_to_num(W[:, mb0:mb1], nan=0.0, posinf=0.0, neginf=0.0),
                    Hinv_diag_block=Hinv_diag[mb0:mb1],
                    bits=bits,
                    symmetric=symmetric,
                    quant_aware=quant_aware_mask,
                )
                if nm is None:
                    M_block = select_unstructured_mask_block_from_score(score, sparsity)
                else:
                    M_block = select_nm_mask_block_from_score(score, n_zero=n_zero, m=m)
                M_global[:, mb0:mb1] = M_block
                selected_mask_until = mb1

            group_idx = get_group_index(global_col, cols, groupsize)

            if is_group_start(global_col, groupsize):
                g0, g1 = get_group_bounds(global_col, cols, groupsize)
                while selected_mask_until < g1:
                    mb0 = selected_mask_until
                    mb1 = min(mb0 + mask_blocksize_eff, cols)
                    score = compute_joint_mask_score_block(
                        W_block=W[:, mb0:mb1],
                        Hinv_diag_block=Hinv_diag[mb0:mb1],
                        bits=bits,
                        symmetric=symmetric,
                        quant_aware=quant_aware_mask,
                    )
                    if nm is None:
                        M_block = select_unstructured_mask_block_from_score(score, sparsity)
                    else:
                        M_block = select_nm_mask_block_from_score(score, n_zero=n_zero, m=m)
                    M_global[:, mb0:mb1] = M_block
                    selected_mask_until = mb1

                qp = make_quant_params_for_slice(
                    W_slice=W[:, g0:g1],
                    bits=bits,
                    symmetric=symmetric,
                    mask_slice=M_global[:, g0:g1],
                )
                scales[:, group_idx] = qp.scale.squeeze(1).to(torch.float32)
                zero_points[:, group_idx] = qp.zero_point.squeeze(1).to(torch.float32)

            d = Hinv1[local_i, local_i].abs().clamp(min=1e-12)
            scale = scales[:, group_idx].to(W.dtype)
            zero = zero_points[:, group_idx].to(W.dtype)

            w = W1[:, local_i]
            q_int, q_deq = quantize_with_given_params(w, scale, zero, bits)
            keep_mask_col = M_global[:, global_col]
            compressed = torch.where(keep_mask_col, q_deq, torch.zeros_like(q_deq))

            Q1[:, local_i] = compressed
            Q_deq_masked[:, global_col] = compressed
            qweight_uint8[:, global_col] = torch.where(keep_mask_col, q_int, torch.zeros_like(q_int))

            err = (torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0) - compressed) / d
            err = torch.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
            Err1[:, local_i] = err
            if local_i + 1 < count:
                W1[:, local_i + 1:count] -= err.unsqueeze(1) @ Hinv1[local_i, local_i + 1:count].unsqueeze(0)

        W[:, i1:i2] = Q1
        if i2 < cols:
            W[:, i2:cols] -= Err1 @ Hinv[i1:i2, i2:cols]
            W[:, i2:cols] = torch.nan_to_num(W[:, i2:cols], nan=0.0, posinf=0.0, neginf=0.0)

    if act_order:
        assert invperm is not None
        Q_deq_masked = Q_deq_masked[:, invperm].contiguous()
        qweight_uint8 = qweight_uint8[:, invperm].contiguous()
        M_global = M_global[:, invperm].contiguous()

        ngroups_orig = get_num_groups(cols, groupsize)
        scales_re = torch.zeros((rows, ngroups_orig), dtype=torch.float32, device=compress_device)
        zero_re = torch.zeros((rows, ngroups_orig), dtype=torch.float32, device=compress_device)
        q_re = torch.zeros_like(qweight_uint8)
        Q_re = torch.zeros_like(Q_deq_masked)

        for g in range(ngroups_orig):
            g0 = 0 if groupsize == -1 else g * groupsize
            g1 = cols if groupsize == -1 else min((g + 1) * groupsize, cols)
            qp = make_quant_params_for_slice(Q_deq_masked[:, g0:g1], bits=bits, symmetric=symmetric, mask_slice=M_global[:, g0:g1])
            scales_re[:, g] = qp.scale.squeeze(1).to(torch.float32)
            zero_re[:, g] = qp.zero_point.squeeze(1).to(torch.float32)
            scale = scales_re[:, g].to(Q_deq_masked.dtype)
            zero = zero_re[:, g].to(Q_deq_masked.dtype)
            for c in range(g0, g1):
                q_int, q_deq = quantize_with_given_params(Q_deq_masked[:, c], scale, zero, bits)
                q_re[:, c] = torch.where(M_global[:, c], q_int, torch.zeros_like(q_int))
                Q_re[:, c] = torch.where(M_global[:, c], q_deq, torch.zeros_like(q_deq))

        scales = scales_re
        zero_points = zero_re
        qweight_uint8 = q_re
        Q_deq_masked = Q_re

    Q_deq_masked = Q_deq_masked * M_global.to(Q_deq_masked.dtype)
    Q_deq_masked = torch.nan_to_num(Q_deq_masked, nan=0.0, posinf=0.0, neginf=0.0)
    qweight_uint8 = qweight_uint8.clamp(0, (1 << bits) - 1)
    layer.weight.data.copy_(Q_deq_masked.to(device=original_device, dtype=original_dtype))

    total_count = rows * cols
    kept_count = int(M_global.sum().item())
    pruned_count = total_count - kept_count
    actual_sparsity = pruned_count / float(total_count)

    return JointLayerResult(
        qweight_uint8=qweight_uint8.detach().cpu(),
        scales=scales.detach().cpu(),
        zero_points=zero_points.detach().cpu(),
        mask=M_global.detach().cpu(),
        dequant_masked_weight=Q_deq_masked.detach().cpu(),
        bits=bits,
        groupsize=groupsize,
        packing=packing,
        mask_packing=mask_packing,
        original_shape=(rows, cols),
        symmetric=symmetric,
        sparsity=actual_sparsity,
        target_sparsity=float(effective_sparsity),
        pattern=pattern,
        pruned_count=pruned_count,
        total_count=total_count,
    )


# ============================================================
# Runtime trainable quant layer for regional reconstruction
# ============================================================

class TrainableSparseQuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        qweight_uint8: torch.Tensor,
        mask_bool: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        groupsize: int,
        bias: Optional[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bits = int(bits)
        self.groupsize = int(groupsize)
        self.register_buffer("qweight", qweight_uint8.to(device=device, dtype=torch.uint8).contiguous())
        self.register_buffer("mask", mask_bool.to(device=device, dtype=torch.bool).contiguous())
        self.scales = nn.Parameter(scales.to(device=device, dtype=torch.float32).contiguous())
        self.zero_points = nn.Parameter(zero_points.to(device=device, dtype=torch.float32).contiguous())
        col_group_idx = torch.tensor(
            [get_group_index(c, self.in_features, self.groupsize) for c in range(self.in_features)],
            dtype=torch.long,
            device=device,
        )
        self.register_buffer("col_group_idx", col_group_idx)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().to(device=device, dtype=dtype), requires_grad=False)
        else:
            self.bias = None

    def dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        scale_expanded = self.scales[:, self.col_group_idx]
        zero_expanded = self.zero_points[:, self.col_group_idx]
        w = (self.qweight.to(torch.float32) - zero_expanded) * scale_expanded
        w = w * self.mask.to(w.dtype)
        return w.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize_weight(dtype=x.dtype).to(x.device)
        bias = self.bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, w, bias)


def unpack_state_for_reconstruction(state: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    rows, cols = tuple(state["shape"])
    if state["packing"] == "packed4":
        q = unpack_4bit_rows(state["qweight"].to(device), cols)
    elif state["packing"] == "uint8":
        q = state["qweight"].to(device).to(torch.uint8)
    else:
        raise ValueError(f"Unsupported packing: {state['packing']}")

    if state["mask_packing"] == "packedbits":
        mask = unpack_bool_mask_rows(state["mask"].to(device), cols)
    elif state["mask_packing"] == "bool":
        mask = state["mask"].to(device).bool()
    else:
        raise ValueError(f"Unsupported mask packing: {state['mask_packing']}")

    return q, mask


@torch.no_grad()
def dense_weight_from_state(state: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rows, cols = tuple(state["shape"])
    q, mask = unpack_state_for_reconstruction(state, device)
    groupsize = int(state["groupsize"])
    col_group_idx = torch.tensor([get_group_index(c, cols, groupsize) for c in range(cols)], device=device, dtype=torch.long)
    scales = state["scales"].to(device=device, dtype=torch.float32)
    zeros = state["zero_points"].to(device=device, dtype=torch.float32)
    w = (q.to(torch.float32) - zeros[:, col_group_idx]) * scales[:, col_group_idx]
    w = w * mask.to(w.dtype)
    return w.to(dtype=dtype)


# ============================================================
# Block forward helpers
# ============================================================

def build_position_inputs(base: nn.Module, hidden: torch.Tensor) -> Dict[str, Any]:
    batch, seq_len, _ = hidden.shape
    device = hidden.device
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    cache_position = torch.arange(seq_len, device=device, dtype=torch.long)
    out: Dict[str, Any] = {"position_ids": position_ids, "cache_position": cache_position}
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
        raise RuntimeError("This script expects base.embed_tokens as in Mistral/LLaMA.")
    hidden_chunks: List[torch.Tensor] = []
    n = calib_tokens.size(0)
    print("\nComputing initial embedding hidden cache...")
    t0 = now()
    for i in range(0, n, batch_size):
        input_ids = calib_tokens[i:i + batch_size].to(device)
        h = base.embed_tokens(input_ids)
        hidden_chunks.append(h.detach().to("cpu", dtype=hidden_cache_dtype))
        done = min(i + batch_size, n)
        print(f"\r  embeddings: {done}/{n} ({100.0 * done / n:.1f}%) elapsed={format_seconds(now() - t0)}", end="", flush=True)
    print()
    hidden = torch.cat(hidden_chunks, dim=0).contiguous()
    print(f"Initial hidden cache shape: {tuple(hidden.shape)}, dtype={hidden.dtype}")
    return hidden


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
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            hdev = self.devices[full_name]
            x = x.to(device=hdev, dtype=self.hessian_dtype)
            local = 2.0 * x.T.matmul(x)
            local = torch.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0)
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
    for i in range(0, n, batch_size):
        hidden = hidden_cpu[i:i + batch_size].to(main_device, dtype=model_dtype)
        _ = forward_decoder_block_no_grad(base, block, hidden, model_dtype, use_autocast=True)
        done = min(i + batch_size, n)
        print(f"\r  block Hessian pass: {done}/{n} ({100.0 * done / n:.1f}%) elapsed={format_seconds(now() - t0)}", end="", flush=True)
    print()
    collector.remove()
    return collector.Hs, collector.nsamples, collector.devices


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
        print(f"\r  {desc}: {done}/{n} ({100.0 * done / n:.1f}%) elapsed={format_seconds(now() - t0)}", end="", flush=True)
    print()
    return torch.cat(outs, dim=0).contiguous()


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

def default_caps() -> Dict[str, Tuple[float, float]]:
    return {
        "q_proj": (0.20, 0.45),
        "k_proj": (0.20, 0.45),
        "v_proj": (0.20, 0.45),
        "o_proj": (0.10, 0.35),
        "gate_proj": (0.35, 0.65),
        "up_proj": (0.35, 0.65),
        "down_proj": (0.15, 0.45),
    }


def parse_caps(raw: str) -> Dict[str, Tuple[float, float]]:
    caps = default_caps()
    raw = raw.strip()
    if not raw:
        return caps
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, val = item.split(":")
        lo_s, hi_s = val.split("-")
        caps[key.strip()] = (float(lo_s), float(hi_s))
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
    for _ in range(60):
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

    def template_from_suffix_values(values: Dict[str, float]) -> Dict[str, float]:
        return {full_name: values.get(layer_suffix(full_name), target_sparsity) for full_name, _local_name, _layer in layer_infos}

    templates.append({full_name: target_sparsity for full_name, _local_name, _layer in layer_infos})
    templates.append(template_from_suffix_values({
        "q_proj": target_sparsity - 0.05,
        "k_proj": target_sparsity - 0.05,
        "v_proj": target_sparsity - 0.05,
        "o_proj": target_sparsity - 0.15,
        "gate_proj": target_sparsity + 0.12,
        "up_proj": target_sparsity + 0.12,
        "down_proj": target_sparsity - 0.10,
    }))
    templates.append(template_from_suffix_values({
        "q_proj": target_sparsity - 0.12,
        "k_proj": target_sparsity - 0.12,
        "v_proj": target_sparsity - 0.12,
        "o_proj": target_sparsity - 0.18,
        "gate_proj": target_sparsity + 0.18,
        "up_proj": target_sparsity + 0.18,
        "down_proj": target_sparsity - 0.05,
    }))
    templates.append(template_from_suffix_values({
        "q_proj": target_sparsity + 0.05,
        "k_proj": target_sparsity + 0.05,
        "v_proj": target_sparsity - 0.12,
        "o_proj": target_sparsity - 0.17,
        "gate_proj": target_sparsity + 0.10,
        "up_proj": target_sparsity + 0.10,
        "down_proj": target_sparsity - 0.15,
    }))
    templates.append(template_from_suffix_values({
        "q_proj": target_sparsity,
        "k_proj": target_sparsity,
        "v_proj": target_sparsity,
        "o_proj": target_sparsity - 0.10,
        "gate_proj": target_sparsity + 0.08,
        "up_proj": target_sparsity + 0.08,
        "down_proj": target_sparsity - 0.18,
    }))

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
# State construction
# ============================================================

def result_to_layer_state(result: JointLayerResult) -> Dict[str, Any]:
    qweight_stored = maybe_pack_qweight(result.qweight_uint8.cpu(), bits=result.bits, packing=result.packing)
    mask_stored = maybe_pack_mask(result.mask.cpu(), mask_packing=result.mask_packing)
    return {
        "bits": int(result.bits),
        "groupsize": int(result.groupsize),
        "packing": str(result.packing),
        "mask_packing": str(result.mask_packing),
        "shape": list(result.original_shape),
        "qweight": qweight_stored,
        "scales": result.scales.cpu(),
        "zero_points": result.zero_points.cpu(),
        "mask": mask_stored,
        "symmetric": bool(result.symmetric),
        "sparsity": float(result.sparsity),
        "target_sparsity": float(result.target_sparsity),
        "pattern": str(result.pattern),
        "pruned_count": int(result.pruned_count),
        "total_count": int(result.total_count),
    }


@torch.no_grad()
def apply_dense_weights_from_states(layer_infos: List[Tuple[str, str, nn.Linear]], states: Dict[str, Dict[str, Any]], device: torch.device) -> None:
    for full_name, _local_name, layer in layer_infos:
        st = states[full_name]
        w = dense_weight_from_state(st, device=device, dtype=layer.weight.dtype)
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
    bits: int,
    pattern: str,
    percdamp: float,
    blocksize: int,
    mask_blocksize: int,
    groupsize: int,
    packing: str,
    mask_packing: str,
    symmetric: bool,
    act_order: bool,
    quant_aware_mask: bool,
    verbose: bool,
) -> Dict[str, Dict[str, Any]]:
    states: Dict[str, Dict[str, Any]] = {}
    for li, (full_name, _local_name, layer) in enumerate(layer_infos, start=1):
        print(f"\n    [{li}/{len(layer_infos)}] Compressing {full_name}")
        print(f"      shape: {tuple(layer.weight.shape)}")
        print(f"      chosen sparsity: {100.0 * rates[full_name]:.2f}%")
        hdev = hessian_devices[full_name]
        result = joint_sparsegpt_gptq_linear(
            layer=layer,
            H=Hs[full_name],
            bits=bits,
            sparsity=float(rates[full_name]),
            pattern=pattern,
            percdamp=percdamp,
            blocksize=blocksize,
            mask_blocksize=mask_blocksize,
            groupsize=groupsize,
            packing=packing,
            mask_packing=mask_packing,
            symmetric=symmetric,
            act_order=act_order,
            quant_aware_mask=quant_aware_mask,
            compress_device=hdev,
            verbose=verbose,
        )
        states[full_name] = result_to_layer_state(result)
        print(
            f"      actual sparsity : {100.0 * result.sparsity:.2f}%\n"
            f"      qweight stored  : {tuple(states[full_name]['qweight'].shape)}\n"
            f"      mask stored     : {tuple(states[full_name]['mask'].shape)}"
        )
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return states


@torch.no_grad()
def block_relative_mse(y_dense: torch.Tensor, y_comp: torch.Tensor) -> float:
    yd = y_dense.float()
    yc = y_comp.float()
    num = torch.mean((yd - yc) ** 2).item()
    den = torch.mean(yd ** 2).item() + 1e-12
    return num / den


# ============================================================
# Regional reconstruction
# ============================================================

def replace_block_layers_with_trainable_quant(
    block: nn.Module,
    layer_infos: List[Tuple[str, str, nn.Linear]],
    states: Dict[str, Dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    for full_name, local_name, old_layer in layer_infos:
        st = states[full_name]
        q, mask = unpack_state_for_reconstruction(st, device=device)
        bias = old_layer.bias.detach().clone() if old_layer.bias is not None else None
        qlayer = TrainableSparseQuantLinear(
            in_features=int(st["shape"][1]),
            out_features=int(st["shape"][0]),
            bits=int(st["bits"]),
            qweight_uint8=q,
            mask_bool=mask,
            scales=st["scales"],
            zero_points=st["zero_points"],
            groupsize=int(st["groupsize"]),
            bias=bias,
            dtype=dtype,
            device=device,
        )
        set_module_by_name(block, local_name, qlayer)


@torch.no_grad()
def extract_states_from_trainable_quant(block: nn.Module, layer_infos: List[Tuple[str, str, nn.Linear]], states: Dict[str, Dict[str, Any]]) -> None:
    for full_name, local_name, _old_layer in layer_infos:
        mod = get_module_by_name(block, local_name)
        if not isinstance(mod, TrainableSparseQuantLinear):
            raise TypeError(f"Expected TrainableSparseQuantLinear at {local_name}, got {type(mod)}")
        states[full_name]["scales"] = mod.scales.detach().cpu()
        states[full_name]["zero_points"] = mod.zero_points.detach().cpu()


@torch.no_grad()
def replace_trainable_quant_with_dense_linear(
    block: nn.Module,
    layer_infos_original: List[Tuple[str, str, nn.Linear]],
    states: Dict[str, Dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    for full_name, local_name, old_layer in layer_infos_original:
        st = states[full_name]
        dense = nn.Linear(
            int(st["shape"][1]),
            int(st["shape"][0]),
            bias=old_layer.bias is not None,
            device=device,
            dtype=dtype,
        )
        dense.weight.data.copy_(dense_weight_from_state(st, device=device, dtype=dtype))
        if old_layer.bias is not None:
            dense.bias.data.copy_(old_layer.bias.detach().to(device=device, dtype=dtype))
        set_module_by_name(block, local_name, dense)


def regional_reconstruct_scales(
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
) -> None:
    if steps <= 0:
        return

    print(f"\n  Regional reconstruction: optimizing group scales/zeros for {steps} steps")
    print(f"  Reconstruction LR: {lr}")

    replace_block_layers_with_trainable_quant(
        block=block,
        layer_infos=layer_infos,
        states=states,
        device=main_device,
        dtype=model_dtype,
    )

    params: List[nn.Parameter] = []
    for _full_name, local_name, _old_layer in layer_infos:
        mod = get_module_by_name(block, local_name)
        if isinstance(mod, TrainableSparseQuantLinear):
            params.append(mod.scales)
            params.append(mod.zero_points)

    if not params:
        print("  [warn] No trainable scale/zero parameters found; skipping reconstruction.")
        return

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
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
            loss = F.mse_loss(pred.float(), y.float())
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for p in params[0::2]:
                    p.clamp_(min=1e-8)

            total_loss += float(loss.item())
            total_batches += 1

        if step == 1 or step % 10 == 0 or step == steps:
            avg = total_loss / max(total_batches, 1)
            print(f"    recon step {step:4d}/{steps} loss={avg:.6e} elapsed={format_seconds(now() - step_t0)}")

    extract_states_from_trainable_quant(block=block, layer_infos=layer_infos, states=states)
    replace_trainable_quant_with_dense_linear(
        block=block,
        layer_infos_original=layer_infos,
        states=states,
        device=main_device,
        dtype=model_dtype,
    )
    block.eval()


# ============================================================
# Main blockwise compressor
# ============================================================

@torch.no_grad()
def compress_model_hybrid_blockwise(
    model: nn.Module,
    calib_tokens: torch.Tensor,
    target_sparsity: float,
    bits: int,
    pattern: str,
    batch_size: int,
    main_device: torch.device,
    model_dtype: torch.dtype,
    hidden_cache_dtype: torch.dtype,
    hessian_dtype: torch.dtype,
    percdamp: float,
    blocksize: int,
    mask_blocksize: int,
    groupsize: int,
    packing: str,
    mask_packing: str,
    symmetric: bool,
    act_order: bool,
    quant_aware_mask: bool,
    large_layer_cpu_threshold: int,
    suffixes: Tuple[str, ...],
    include: str,
    exclude: str,
    caps: Dict[str, Tuple[float, float]],
    max_candidates: int,
    recon_steps: int,
    recon_lr: float,
    store_debug_dequant: bool,
) -> Dict[str, Dict[str, Any]]:
    base = get_base_decoder_model(model)
    layers_name, decoder_layers = get_decoder_layers(base)
    print(f"\nDecoder layers found: {layers_name}")
    print(f"Number of decoder blocks: {len(decoder_layers)}")

    hidden_cpu = compute_initial_hidden(model, calib_tokens, batch_size, main_device, hidden_cache_dtype)
    joint_layers: Dict[str, Dict[str, Any]] = {}
    script_t0 = now()

    root_prefix = "model" if hasattr(model, "model") else "transformer"

    for bi, block in enumerate(decoder_layers):
        block_prefix = f"{root_prefix}.{layers_name}.{bi}"
        layer_infos = find_block_linears(block, block_prefix, suffixes, include, exclude)

        print("\n" + "=" * 100)
        print(f"BLOCK {bi}/{len(decoder_layers) - 1}")
        print(f"Selected linear layers in block: {len(layer_infos)}")
        print(f"Hidden cache entering block: {tuple(hidden_cpu.shape)}, dtype={hidden_cpu.dtype}")
        print_cuda_memory("CUDA memory")
        for full_name, _local_name, layer in layer_infos:
            print(f"  - {full_name}: {tuple(layer.weight.shape)} dtype={layer.weight.dtype}")

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
                bits=bits,
                pattern=pattern,
                percdamp=percdamp,
                blocksize=blocksize,
                mask_blocksize=mask_blocksize,
                groupsize=groupsize,
                packing=packing,
                mask_packing=mask_packing,
                symmetric=symmetric,
                act_order=act_order,
                quant_aware_mask=quant_aware_mask,
                verbose=False,
            )

            y_comp_cpu = run_block_on_hidden_cache(base, block, hidden_cpu, batch_size, main_device, model_dtype, hidden_cache_dtype, f"candidate {ci} compressed block")
            rel = block_relative_mse(y_dense_cpu, y_comp_cpu)
            avg_sp = weighted_average_sparsity(rates, layer_infos)
            print(
                f"  Candidate {ci} result:\n"
                f"    block relative MSE : {rel:.8e}\n"
                f"    weighted sparsity  : {100.0 * avg_sp:.2f}%\n"
                f"    time               : {format_seconds(now() - cand_t0)}"
            )

            if rel < best_loss:
                best_loss = rel
                best_idx = ci
                best_rates = copy.deepcopy(rates)

            del y_comp_cpu
            del states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            bits=bits,
            pattern=pattern,
            percdamp=percdamp,
            blocksize=blocksize,
            mask_blocksize=mask_blocksize,
            groupsize=groupsize,
            packing=packing,
            mask_packing=mask_packing,
            symmetric=symmetric,
            act_order=act_order,
            quant_aware_mask=quant_aware_mask,
            verbose=True,
        )

        if recon_steps > 0:
            # Need gradients here, so temporarily leave no_grad context.
            with torch.enable_grad():
                regional_reconstruct_scales(
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
                )
            apply_dense_weights_from_states(layer_infos, final_states, main_device)

        for full_name, _local_name, _layer in layer_infos:
            st = final_states[full_name]
            if store_debug_dequant:
                st["dequant_masked_weight"] = dense_weight_from_state(st, device=main_device, dtype=torch.float16).cpu()
            joint_layers[full_name] = st

        print("\n  Running final compressed block to create next hidden cache...")
        hidden_cpu = run_block_on_hidden_cache(base, block, hidden_cpu, batch_size, main_device, model_dtype, hidden_cache_dtype, f"block {bi} final forward")

        del Hs, nsamples, hessian_devices, y_dense_cpu, original_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Finished block {bi}. Total elapsed: {format_seconds(now() - script_t0)}")
        print_cuda_memory("CUDA memory")

    return joint_layers


# ============================================================
# Checkpoint save
# ============================================================

def build_partial_noncompressed_state_dict(model: nn.Module, compressed_layer_names: Iterable[str]) -> Dict[str, torch.Tensor]:
    prefixes = [name + "." for name in compressed_layer_names]
    weight_keys = {name + ".weight" for name in compressed_layer_names}
    out: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if k in weight_keys:
            continue
        skip = False
        for p in prefixes:
            if k.startswith(p) and any(s in k for s in (".qweight", ".mask", ".scales", ".zero_points", ".col_group_idx")):
                skip = True
                break
        if skip:
            continue
        out[k] = v.detach().cpu()
    return out


def save_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    out_path: str,
    model_id: str,
    meta: Dict[str, Any],
    joint_layers: Dict[str, Dict[str, Any]],
    keep_dequantized_state_dict: bool,
) -> None:
    if keep_dequantized_state_dict:
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    else:
        model_state = build_partial_noncompressed_state_dict(model, joint_layers.keys())

    ckpt = {
        "format": "hf_joint_sparsegpt_gptq",
        "model_id": model_id,
        "config": model.config.to_dict() if hasattr(model, "config") else None,
        "model": model_state,
        "joint_sparsegpt_gptq_meta": meta,
        "joint_sparsegpt_gptq_layers": joint_layers,
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


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--target_sparsity", type=float, default=0.50)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--pattern", type=str, default="unstructured")
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hidden_cache_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hessian_dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--percdamp", type=float, default=0.1)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--mask_blocksize", type=int, default=128)
    parser.add_argument("--packing", type=str, default="packed4", choices=["uint8", "packed4"])
    parser.add_argument("--mask_packing", type=str, default="packedbits", choices=["bool", "packedbits"])
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--act_order", action="store_true")
    parser.add_argument("--no_quant_aware_mask", action="store_true")
    parser.add_argument("--suffixes", type=str, default="")
    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")
    parser.add_argument("--caps", type=str, default="")
    parser.add_argument("--max_candidates", type=int, default=4)
    parser.add_argument("--recon_steps", type=int, default=0)
    parser.add_argument("--recon_lr", type=float, default=1e-3)
    parser.add_argument("--large_layer_cpu_threshold", type=int, default=8192)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--keep_dequantized_state_dict", action="store_true")
    parser.add_argument("--store_debug_dequant", action="store_true")
    args = parser.parse_args()

    script_t0 = now()

    if args.packing == "packed4" and args.bits != 4:
        raise ValueError("--packing packed4 requires --bits 4.")
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
    suffixes = parse_suffixes(args.suffixes)
    caps = parse_caps(args.caps)

    print("=" * 100)
    print("Hybrid BESA/Wanda++/SparseGPT/GPTQ compression")
    print("=" * 100)
    print(f"model_id                  : {args.model_id}")
    print(f"calib                     : {args.calib}")
    print(f"out                       : {args.out}")
    print(f"device                    : {main_device}")
    print(f"model_dtype               : {model_dtype}")
    print(f"hidden_cache_dtype        : {hidden_cache_dtype}")
    print(f"hessian_dtype             : {hessian_dtype}")
    print(f"target_sparsity           : {args.target_sparsity}")
    print(f"bits                      : {args.bits}")
    print(f"groupsize                 : {args.groupsize}")
    print(f"blocksize                 : {args.blocksize}")
    print(f"mask_blocksize            : {args.mask_blocksize}")
    print(f"percdamp                  : {args.percdamp}")
    print(f"packing                   : {args.packing}")
    print(f"mask_packing              : {args.mask_packing}")
    print(f"max_candidates            : {args.max_candidates}")
    print(f"recon_steps               : {args.recon_steps}")
    print(f"large_layer_cpu_threshold : {args.large_layer_cpu_threshold}")
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

    joint_layers = compress_model_hybrid_blockwise(
        model=model,
        calib_tokens=calib_tokens,
        target_sparsity=float(args.target_sparsity),
        bits=int(args.bits),
        pattern=str(args.pattern),
        batch_size=int(args.batch_size),
        main_device=main_device,
        model_dtype=model_dtype,
        hidden_cache_dtype=hidden_cache_dtype,
        hessian_dtype=hessian_dtype,
        percdamp=float(args.percdamp),
        blocksize=int(args.blocksize),
        mask_blocksize=int(args.mask_blocksize),
        groupsize=int(args.groupsize),
        packing=str(args.packing),
        mask_packing=str(args.mask_packing),
        symmetric=bool(args.symmetric),
        act_order=bool(args.act_order),
        quant_aware_mask=not bool(args.no_quant_aware_mask),
        large_layer_cpu_threshold=int(args.large_layer_cpu_threshold),
        suffixes=suffixes,
        include=str(args.include),
        exclude=str(args.exclude),
        caps=caps,
        max_candidates=int(args.max_candidates),
        recon_steps=int(args.recon_steps),
        recon_lr=float(args.recon_lr),
        store_debug_dequant=bool(args.store_debug_dequant),
    )

    total_pruned = sum(int(st["pruned_count"]) for st in joint_layers.values())
    total_weights = sum(int(st["total_count"]) for st in joint_layers.values())
    actual_sparsity = total_pruned / float(total_weights) if total_weights else 0.0
    qweight_bytes = sum(int(st["qweight"].numel() * st["qweight"].element_size()) for st in joint_layers.values())
    mask_bytes = sum(int(st["mask"].numel() * st["mask"].element_size()) for st in joint_layers.values())
    scales_bytes = sum(int(st["scales"].numel() * st["scales"].element_size()) for st in joint_layers.values())
    zeros_bytes = sum(int(st["zero_points"].numel() * st["zero_points"].element_size()) for st in joint_layers.values())
    raw_stored_bytes = qweight_bytes + mask_bytes + scales_bytes + zeros_bytes
    dense_bf16_bytes = total_weights * 2

    meta = {
        "method": "hybrid_besa_wandapp_sparsegpt_gptq_blockwise_fixed",
        "model_id": str(args.model_id),
        "target_sparsity": float(args.target_sparsity),
        "actual_total_sparsity": float(actual_sparsity),
        "bits": int(args.bits),
        "pattern": str(args.pattern),
        "groupsize": int(args.groupsize),
        "percdamp": float(args.percdamp),
        "blocksize": int(args.blocksize),
        "mask_blocksize": int(args.mask_blocksize),
        "packing": str(args.packing),
        "mask_packing": str(args.mask_packing),
        "symmetric": bool(args.symmetric),
        "act_order": bool(args.act_order),
        "quant_aware_mask": not bool(args.no_quant_aware_mask),
        "max_candidates": int(args.max_candidates),
        "recon_steps": int(args.recon_steps),
        "recon_lr": float(args.recon_lr),
        "model_dtype": str(args.model_dtype),
        "hidden_cache_dtype": str(args.hidden_cache_dtype),
        "hessian_dtype": str(args.hessian_dtype),
        "calibration_source": str(args.calib),
        "max_seq_len": int(args.max_seq_len),
        "large_layer_cpu_threshold": int(args.large_layer_cpu_threshold),
        "suffixes": list(suffixes),
        "caps": {k: [float(v[0]), float(v[1])] for k, v in caps.items()},
        "compressed_layers": int(len(joint_layers)),
        "total_pruned_weights": int(total_pruned),
        "total_compressed_weights": int(total_weights),
        "qweight_bytes": int(qweight_bytes),
        "mask_bytes": int(mask_bytes),
        "scales_bytes": int(scales_bytes),
        "zero_points_bytes": int(zeros_bytes),
        "raw_stored_bytes": int(raw_stored_bytes),
        "dense_bf16_bytes": int(dense_bf16_bytes),
        "raw_compression_vs_bf16": float(dense_bf16_bytes / raw_stored_bytes) if raw_stored_bytes else 0.0,
        "total_script_seconds": float(now() - script_t0),
        "note": "BESA-style block allocation search + SparseGPT/GPTQ + optional regional reconstruction over scales/zero_points.",
    }

    print("\nSaving checkpoint...")
    save_checkpoint(model, tokenizer, args.out, args.model_id, meta, joint_layers, bool(args.keep_dequantized_state_dict))

    print("\nDone.")
    print(f"Compressed layers              : {len(joint_layers)}")
    print(f"Total selected weights         : {total_weights:,}")
    print(f"Pruned weights                 : {total_pruned:,}")
    print(f"Actual sparsity                : {100.0 * actual_sparsity:.2f}%")
    print(f"qweight bytes                  : {qweight_bytes:,}")
    print(f"mask bytes                     : {mask_bytes:,}")
    print(f"scale bytes                    : {scales_bytes:,}")
    print(f"zero-point bytes               : {zeros_bytes:,}")
    print(f"raw stored bytes               : {raw_stored_bytes:,}")
    print(f"dense BF16 bytes               : {dense_bf16_bytes:,}")
    print(f"raw compression vs BF16        : {dense_bf16_bytes / raw_stored_bytes:.2f}x")
    print(f"Total script time              : {format_seconds(now() - script_t0)}")
    print_cuda_memory("CUDA memory")


if __name__ == "__main__":
    main()
