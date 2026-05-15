#!/usr/bin/env python3
"""
Blockwise Hugging Face Joint SparseGPT + GPTQ-style post-training compression.

For decoder-only HF models such as:
    - mistralai/Mistral-7B-Instruct-v0.3
    - meta-llama/Meta-Llama-3-8B-Instruct
    - Qwen/Qwen2.5-7B-Instruct

Main workflow:
    For each transformer block:
        1. Cache block input hidden states once.
        2. Register hooks for all selected Linear layers in that block.
        3. Run the block once over calibration hidden states.
        4. Collect H = 2 X^T X for q,k,v,o,gate,up,down together.
        5. Compress selected layers in that block.
        6. Run the compressed block to produce next block inputs.

Important changes:
    - If --sparsity 0.0 and --pattern unstructured, sparse score computation is skipped.
    - Non-finite activation values are sanitized before Hessian construction.
    - If local Hessian becomes non-finite, that local contribution is skipped instead of zeroing individual entries.
    - Cholesky has stronger damping and eigenvalue fallback.
"""

from __future__ import annotations

import argparse
import inspect
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Utilities
# ============================================================

def fmt_time(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "?"
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def cuda_mem() -> str:
    if not torch.cuda.is_available():
        return "cuda unavailable"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_alloc = torch.cuda.max_memory_allocated() / 1024**3
    return f"alloc={allocated:.2f}GB reserved={reserved:.2f}GB max={max_alloc:.2f}GB"


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


# ============================================================
# Calibration
# ============================================================

def load_calibration_tokens(calib_path: str) -> torch.Tensor:
    obj = torch.load(calib_path, map_location="cpu")

    if isinstance(obj, dict) and "tokens" in obj:
        tokens = obj["tokens"]
    elif torch.is_tensor(obj):
        tokens = obj
    else:
        raise ValueError("Calibration file must contain dict['tokens'] or be a tensor.")

    if tokens.ndim != 2:
        raise ValueError(f"Expected calibration tokens [N, T], got {tuple(tokens.shape)}")

    return tokens.long()


# ============================================================
# Data containers
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


# ============================================================
# Group helpers
# ============================================================

def get_group_bounds(col_idx: int, cols: int, groupsize: int) -> Tuple[int, int]:
    if groupsize == -1 or groupsize >= cols:
        return 0, cols
    g0 = (col_idx // groupsize) * groupsize
    g1 = min(g0 + groupsize, cols)
    return g0, g1


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


# ============================================================
# Quantization helpers
# ============================================================

def make_quant_params_for_slice(
    W_slice: torch.Tensor,
    bits: int,
    symmetric: bool,
    mask_slice: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> QuantParams:
    if bits < 2 or bits > 8:
        raise ValueError("bits must be in [2, 8].")

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
        zero_point = torch.full_like(scale, fill_value=mid)
        return QuantParams(scale, zero_point, qmin, qmax, True)

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
    zero_point = torch.round(qmin - w_min / scale).clamp(qmin, qmax)

    return QuantParams(scale, zero_point, qmin, qmax, False)


def quantize_with_given_params(
    w_col: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    qmin = 0
    qmax = (1 << bits) - 1
    q = torch.round(w_col / scale + zero).clamp(qmin, qmax)
    q_int = q.to(torch.uint8)
    q_deq = (q.to(w_col.dtype) - zero) * scale
    return q_int, q_deq


# ============================================================
# Packing helpers
# ============================================================

def pack_4bit_rows(qweight_uint8: torch.Tensor) -> torch.Tensor:
    if qweight_uint8.dtype != torch.uint8:
        raise ValueError("qweight_uint8 must be uint8.")
    if torch.any(qweight_uint8 > 15):
        raise ValueError("4-bit packing requires values <= 15.")

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
        raise ValueError("packed must be uint8.")

    rows, packed_cols = packed.shape
    out = torch.zeros((rows, packed_cols * 2), dtype=torch.uint8, device=packed.device)
    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = (packed >> 4) & 0x0F
    return out[:, :original_cols]


def maybe_pack_qweight(qweight_uint8: torch.Tensor, bits: int, packing: str) -> torch.Tensor:
    if packing == "uint8":
        return qweight_uint8.cpu()

    if packing == "packed4":
        if bits != 4:
            raise ValueError("--packing packed4 only works with --bits 4.")
        return pack_4bit_rows(qweight_uint8).cpu()

    raise ValueError(f"Unsupported packing: {packing}")


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


def maybe_pack_mask(mask: torch.Tensor, mask_packing: str) -> torch.Tensor:
    if mask_packing == "bool":
        return mask.cpu().bool()

    if mask_packing == "packedbits":
        return pack_bool_mask_rows(mask).cpu()

    raise ValueError(f"Unsupported mask_packing: {mask_packing}")


# ============================================================
# Sparse mask helpers
# ============================================================

def parse_nm_pattern(pattern: str) -> Optional[Tuple[int, int]]:
    pattern = pattern.strip().lower()

    if pattern in ("", "unstructured", "none"):
        return None

    if ":" not in pattern:
        raise ValueError("Pattern must be 'unstructured' or N:M, e.g. '2:4'.")

    a, b = pattern.split(":")
    n = int(a)
    m = int(b)

    if n < 0 or m <= 0 or n > m:
        raise ValueError(f"Invalid N:M pattern {pattern}. Need 0 <= N <= M.")

    return n, m


def select_unstructured_mask_block_from_score(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    rows, block_cols = score.shape
    total = rows * block_cols
    n_prune_total = int(round(sparsity * total))

    if n_prune_total <= 0:
        return torch.ones_like(score, dtype=torch.bool)
    if n_prune_total >= total:
        return torch.zeros_like(score, dtype=torch.bool)

    flat_score = score.reshape(-1)
    n_keep_total = total - n_prune_total

    keep_idx = torch.topk(flat_score, k=n_keep_total, largest=True, sorted=False).indices

    flat_mask = torch.zeros(total, dtype=torch.bool, device=score.device)
    flat_mask[keep_idx] = True
    return flat_mask.view(rows, block_cols)


def select_nm_mask_block_from_score(score: torch.Tensor, n_zero: int, m: int) -> torch.Tensor:
    rows, block_cols = score.shape
    mask = torch.ones_like(score, dtype=torch.bool)

    for g0 in range(0, block_cols, m):
        g1 = min(g0 + m, block_cols)
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
        prune_idx = torch.topk(local_score, k=prune_count, largest=False, dim=1, sorted=False).indices

        row_idx = torch.arange(rows, device=score.device).view(-1, 1).expand_as(prune_idx)
        local_mask = mask[:, g0:g1]
        local_mask[row_idx, prune_idx] = False
        mask[:, g0:g1] = local_mask

    return mask


def compute_joint_mask_score_block(
    W_block: torch.Tensor,
    Hinv_diag_block: torch.Tensor,
    bits: int,
    symmetric: bool,
    quant_aware: bool,
    eps: float = 1e-12,
) -> torch.Tensor:
    diag = Hinv_diag_block.to(W_block.device, dtype=W_block.dtype).abs().clamp(min=eps)

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
# Robust Cholesky / inverse
# ============================================================

def stable_cholesky_inverse_info(
    H: torch.Tensor,
    percdamp: float,
    max_tries: int = 12,
    sanitize_nonfinite: bool = True,
    eig_fallback: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if H.ndim != 2 or H.size(0) != H.size(1):
        raise ValueError(f"H must be square, got {tuple(H.shape)}")

    if not torch.isfinite(H).all():
        bad = torch.isfinite(H).logical_not().sum().item()
        total = H.numel()
        print(
            f"[warn] Hessian contains non-finite values before Cholesky: "
            f"{bad:,}/{total:,} values are NaN/Inf."
        )

        if not sanitize_nonfinite:
            raise RuntimeError("Hessian contains non-finite values.")

        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

    H64 = H.to(torch.float64)
    H64 = 0.5 * (H64 + H64.T)

    n = H64.size(0)
    ar = torch.arange(n, device=H64.device)

    diag = torch.diag(H64)

    if not torch.isfinite(diag).all():
        bad = torch.isfinite(diag).logical_not().sum().item()
        print(f"[warn] Hessian diagonal has {bad:,} non-finite values.")

        if not sanitize_nonfinite:
            raise RuntimeError("Hessian diagonal contains non-finite values.")

        diag_clean = torch.nan_to_num(diag, nan=0.0, posinf=0.0, neginf=0.0)
        H64[ar, ar] = diag_clean
        diag = torch.diag(H64)

    diag_abs = diag.abs()
    diag_mean = diag_abs.mean().item()
    diag_max = diag_abs.max().item()
    base = max(diag_mean, diag_max * 1.0e-6, 1.0e-12)

    last_exc: Optional[Exception] = None

    multipliers = [
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
        300.0,
        1000.0,
        3000.0,
        10000.0,
        30000.0,
        100000.0,
        300000.0,
    ][:max_tries]

    for mult in multipliers:
        used_damp = percdamp * mult * base

        H_try = H64.clone()
        H_try[ar, ar] += used_damp

        try:
            chol = torch.linalg.cholesky(H_try)
            Hinv = torch.cholesky_inverse(chol)
            Hinv = 0.5 * (Hinv + Hinv.T)
            Hinv_chol_upper = torch.linalg.cholesky(Hinv, upper=True)
            return H_try, Hinv_chol_upper, used_damp
        except RuntimeError as exc:
            last_exc = exc

    if eig_fallback:
        print("[warn] Cholesky damping retries failed. Trying eigenvalue-based diagonal shift...")

        try:
            evals = torch.linalg.eigvalsh(H64)
            eig_on_cpu = False
        except RuntimeError:
            H_cpu = H64.cpu()
            evals = torch.linalg.eigvalsh(H_cpu)
            eig_on_cpu = True

        min_eval = float(evals.min().item())
        max_eval = float(evals.abs().max().item())
        needed_shift = max(0.0, -min_eval) + max(percdamp * base, 1.0e-6 * max(max_eval, 1.0))

        print(
            f"[warn] eig fallback: min_eigenvalue={min_eval:.6e}, "
            f"shift={needed_shift:.6e}"
        )

        if eig_on_cpu and H64.device.type != "cpu":
            H_try_cpu = H64.cpu()
            ar_cpu = torch.arange(n, device="cpu")
            H_try_cpu[ar_cpu, ar_cpu] += needed_shift

            chol = torch.linalg.cholesky(H_try_cpu)
            Hinv = torch.cholesky_inverse(chol)
            Hinv = 0.5 * (Hinv + Hinv.T)
            Hinv_chol_upper = torch.linalg.cholesky(Hinv, upper=True)

            return H_try_cpu.to(H64.device), Hinv_chol_upper.to(H64.device), float(needed_shift)

        H_try = H64.clone()
        H_try[ar, ar] += needed_shift

        chol = torch.linalg.cholesky(H_try)
        Hinv = torch.cholesky_inverse(chol)
        Hinv = 0.5 * (Hinv + Hinv.T)
        Hinv_chol_upper = torch.linalg.cholesky(Hinv, upper=True)
        return H_try, Hinv_chol_upper, float(needed_shift)

    raise RuntimeError(
        "Cholesky failed after adaptive damping retries. "
        f"Initial percdamp={percdamp}. Last error: {last_exc}"
    )


# ============================================================
# Joint SparseGPT + GPTQ for one Linear layer
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
) -> JointLayerResult:
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(layer)}")

    if packing == "packed4" and bits != 4:
        raise ValueError("--packing packed4 only valid with --bits 4.")

    original_weight_device = layer.weight.device
    original_weight_dtype = layer.weight.dtype

    W_orig = layer.weight.detach().to(device=compress_device, dtype=torch.float32).clone()
    rows, cols = W_orig.shape

    if H.shape != (cols, cols):
        raise ValueError(f"H shape mismatch. Expected {(cols, cols)}, got {tuple(H.shape)}")

    H = H.to(compress_device)

    nm = parse_nm_pattern(pattern)

    if nm is not None:
        n_zero, m = nm
        mask_blocksize_eff = m
        effective_sparsity = n_zero / float(m)
    else:
        n_zero, m = None, None
        mask_blocksize_eff = mask_blocksize
        effective_sparsity = sparsity

    no_sparsity_shortcut = (nm is None and effective_sparsity <= 0.0)

    print(f"      bits: {bits}")
    print(f"      target sparsity: {effective_sparsity:.4f}")
    print(f"      pattern: {pattern}")
    print(f"      groupsize: {groupsize}")
    print(f"      qweight packing: {packing}")
    print(f"      mask packing: {mask_packing}")
    print(f"      lazy update blocksize B: {blocksize}")
    print(f"      mask selection blocksize Bs: {mask_blocksize_eff}")
    print(f"      quant-aware mask: {quant_aware_mask}")
    print(f"      no-sparsity shortcut: {no_sparsity_shortcut}")
    print(f"      compression device: {compress_device}")

    t0 = time.time()
    H_damped, Hinv_chol_upper, used_damp = stable_cholesky_inverse_info(H, percdamp=percdamp)
    print(f"      used damping: {used_damp:.6e}")
    print(f"      H inverse/cholesky time: {fmt_time(time.time() - t0)}")

    Hinv = Hinv_chol_upper.T @ Hinv_chol_upper
    Hinv = Hinv.to(W_orig.dtype)

    if act_order:
        perm = torch.argsort(torch.diag(H_damped), descending=True)
        invperm = torch.argsort(perm)
        W = W_orig[:, perm].contiguous()
        Hinv = Hinv[perm][:, perm].contiguous()
    else:
        invperm = None
        W = W_orig.clone()

    Q_deq_masked = torch.zeros_like(W)
    qweight_uint8 = torch.zeros((rows, cols), dtype=torch.uint8, device=W.device)
    M_global = torch.ones((rows, cols), dtype=torch.bool, device=W.device)

    ngroups = get_num_groups(cols, groupsize)
    scales = torch.zeros((rows, ngroups), dtype=torch.float32, device=W.device)
    zero_points = torch.zeros((rows, ngroups), dtype=torch.float32, device=W.device)

    selected_mask_until = cols if no_sparsity_shortcut else -1
    Hinv_diag = torch.diag(Hinv)

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2].contiguous()

        for local_i in range(count):
            global_col = i1 + local_i

            if not no_sparsity_shortcut and global_col >= selected_mask_until:
                mb0 = global_col
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

            group_idx = get_group_index(global_col, cols, groupsize)

            if is_group_start(global_col, groupsize):
                g0, g1 = get_group_bounds(global_col, cols, groupsize)

                if not no_sparsity_shortcut:
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

            d = Hinv1[local_i, local_i]

            if d.abs().item() < 1e-12:
                raise RuntimeError(f"Near-zero Hinv diagonal at column {global_col}: {d.item()}")

            scale = scales[:, group_idx].to(W.dtype)
            zero = zero_points[:, group_idx].to(W.dtype)

            w = W1[:, local_i]
            q_int, q_deq = quantize_with_given_params(w, scale, zero, bits)

            if no_sparsity_shortcut:
                compressed = q_deq
                qweight_uint8[:, global_col] = q_int
            else:
                keep_mask_col = M_global[:, global_col]
                compressed = torch.where(keep_mask_col, q_deq, torch.zeros_like(q_deq))
                qweight_uint8[:, global_col] = torch.where(
                    keep_mask_col,
                    q_int,
                    torch.zeros_like(q_int),
                )

            Q1[:, local_i] = compressed
            Q_deq_masked[:, global_col] = compressed

            err = (w - compressed) / d
            Err1[:, local_i] = err

            if local_i + 1 < count:
                W1[:, local_i + 1:count] -= (
                    err.unsqueeze(1)
                    @ Hinv1[local_i, local_i + 1:count].unsqueeze(0)
                )

        W[:, i1:i2] = Q1

        if i2 < cols:
            W[:, i2:cols] -= Err1 @ Hinv[i1:i2, i2:cols]

    if act_order:
        assert invperm is not None

        Q_deq_masked = Q_deq_masked[:, invperm].contiguous()
        qweight_uint8 = qweight_uint8[:, invperm].contiguous()
        M_global = M_global[:, invperm].contiguous()

        ngroups_orig = get_num_groups(cols, groupsize)
        scales_re = torch.zeros((rows, ngroups_orig), dtype=torch.float32, device=W_orig.device)
        zero_re = torch.zeros((rows, ngroups_orig), dtype=torch.float32, device=W_orig.device)
        q_re = torch.zeros_like(qweight_uint8)
        Q_re = torch.zeros_like(Q_deq_masked)

        for g in range(ngroups_orig):
            g0 = 0 if groupsize == -1 else g * groupsize
            g1 = cols if groupsize == -1 else min((g + 1) * groupsize, cols)

            qp = make_quant_params_for_slice(
                W_slice=Q_deq_masked[:, g0:g1],
                bits=bits,
                symmetric=symmetric,
                mask_slice=M_global[:, g0:g1],
            )

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

    layer.weight.data.copy_(
        Q_deq_masked.to(device=original_weight_device, dtype=original_weight_dtype)
    )

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
        target_sparsity=effective_sparsity,
        pattern=pattern,
        pruned_count=pruned_count,
        total_count=total_count,
    )


# ============================================================
# HF module helpers
# ============================================================

def get_module_by_name(root: nn.Module, full_name: str) -> nn.Module:
    obj = root
    for part in full_name.split("."):
        obj = getattr(obj, part)
    return obj


def model_has_tied_lm_head_hf(model: nn.Module) -> bool:
    if not hasattr(model, "lm_head"):
        return False
    try:
        emb = model.get_input_embeddings()
        if emb is None:
            return False
        return model.lm_head.weight.data_ptr() == emb.weight.data_ptr()
    except Exception:
        return False


def default_hf_linear_suffixes() -> Tuple[str, ...]:
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
        return default_hf_linear_suffixes()
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def should_compress_hf_layer_name(
    name: str,
    include: str,
    exclude: str,
    suffixes: Tuple[str, ...],
    compress_lm_head: bool,
    skip_attn_out: bool,
    skip_mlp_out: bool,
) -> bool:
    if include and include not in name:
        return False
    if exclude and exclude in name:
        return False
    if name == "lm_head":
        return compress_lm_head
    if skip_attn_out and name.endswith("o_proj"):
        return False
    if skip_mlp_out and name.endswith("down_proj"):
        return False
    return name.endswith(suffixes)


def find_decoder_layers(model: nn.Module) -> Tuple[str, nn.ModuleList]:
    candidates = [
        "model.layers",
        "model.decoder.layers",
        "transformer.h",
        "gpt_neox.layers",
    ]

    for name in candidates:
        try:
            obj = get_module_by_name(model, name)
            if isinstance(obj, nn.ModuleList):
                return name, obj
        except Exception:
            pass

    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            first_name = mod[0].__class__.__name__.lower()
            if "decoder" in first_name or "layer" in first_name or "block" in first_name:
                return name, mod

    raise RuntimeError(
        "Could not find decoder block ModuleList. "
        "For Llama/Mistral/Qwen this is usually model.layers."
    )


def find_selected_linear_names(
    model: nn.Module,
    include: str,
    exclude: str,
    suffixes: Tuple[str, ...],
    compress_lm_head: bool,
    skip_tied_lm_head: bool,
    skip_attn_out: bool,
    skip_mlp_out: bool,
) -> List[str]:
    tied = model_has_tied_lm_head_hf(model)
    out: List[str] = []

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue

        if skip_tied_lm_head and tied and name == "lm_head":
            print("[info] skipping tied lm_head")
            continue

        if should_compress_hf_layer_name(
            name=name,
            include=include,
            exclude=exclude,
            suffixes=suffixes,
            compress_lm_head=compress_lm_head,
            skip_attn_out=skip_attn_out,
            skip_mlp_out=skip_mlp_out,
        ):
            out.append(name)

    return out


def choose_layer_math_device(
    layer: nn.Linear,
    main_device: torch.device,
    large_layer_cpu_threshold: int,
) -> torch.device:
    if large_layer_cpu_threshold <= 0:
        return torch.device("cpu")
    if int(layer.in_features) > large_layer_cpu_threshold:
        return torch.device("cpu")
    return main_device


def get_joint_weight_keys(joint_layers: Dict[str, Any]) -> Set[str]:
    return {f"{layer_name}.weight" for layer_name in joint_layers.keys()}


def build_partial_noncompressed_state_dict(
    model: nn.Module,
    joint_layers: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    full_sd = model.state_dict()
    compressed_weight_keys = get_joint_weight_keys(joint_layers)

    partial_sd: Dict[str, torch.Tensor] = {}

    for k, v in full_sd.items():
        if k in compressed_weight_keys:
            continue
        partial_sd[k] = v.detach().cpu()

    return partial_sd


def save_hf_joint_checkpoint(
    model: nn.Module,
    tokenizer: Optional[Any],
    out_path: str,
    model_id: str,
    joint_meta: Dict[str, Any],
    joint_layers: Dict[str, Any],
    keep_dequantized_state_dict: bool,
) -> None:
    if keep_dequantized_state_dict:
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    else:
        model_state = build_partial_noncompressed_state_dict(model=model, joint_layers=joint_layers)

    ckpt = {
        "format": "hf_joint_sparsegpt_gptq",
        "model_id": model_id,
        "config": model.config.to_dict() if hasattr(model, "config") else None,
        "model": model_state,
        "joint_sparsegpt_gptq_meta": joint_meta,
        "joint_sparsegpt_gptq_layers": joint_layers,
        "compression_meta": joint_meta,
    }

    if tokenizer is not None:
        ckpt["tokenizer_name_or_path"] = getattr(tokenizer, "name_or_path", model_id)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    print(f"Saved HF joint SparseGPT+GPTQ checkpoint to: {out}")


# ============================================================
# HF block forward helpers
# ============================================================

def get_backbone(model: nn.Module) -> nn.Module:
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    return model


def get_embedding_layer(model: nn.Module) -> nn.Module:
    emb = model.get_input_embeddings()
    if emb is None:
        raise RuntimeError("model.get_input_embeddings() returned None.")
    return emb


def make_position_ids(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)


def make_4d_causal_attention_mask(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if dtype in (torch.float16, torch.bfloat16):
        mask_value = torch.finfo(dtype).min
    else:
        mask_value = -1.0e30

    mask = torch.full(
        (seq_len, seq_len),
        fill_value=mask_value,
        dtype=dtype,
        device=device,
    )

    mask = torch.triu(mask, diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.expand(batch_size, 1, seq_len, seq_len)

    return mask


def maybe_make_position_embeddings(
    backbone: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
):
    if not hasattr(backbone, "rotary_emb"):
        return None

    try:
        return backbone.rotary_emb(hidden_states, position_ids)
    except TypeError:
        try:
            return backbone.rotary_emb(hidden_states, seq_len=hidden_states.shape[1])
        except Exception:
            return None
    except Exception:
        return None


def call_decoder_block(
    block: nn.Module,
    hidden_states: torch.Tensor,
    backbone: nn.Module,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    position_ids = make_position_ids(batch_size=batch_size, seq_len=seq_len, device=device)
    cache_position = torch.arange(seq_len, device=device, dtype=torch.long)

    position_embeddings = maybe_make_position_embeddings(
        backbone=backbone,
        hidden_states=hidden_states,
        position_ids=position_ids,
    )

    causal_mask = make_4d_causal_attention_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        dtype=dtype,
        device=device,
    )

    sig = inspect.signature(block.forward)
    accepted = set(sig.parameters.keys())

    kwargs: Dict[str, Any] = {}

    if "attention_mask" in accepted:
        kwargs["attention_mask"] = causal_mask
    if "position_ids" in accepted:
        kwargs["position_ids"] = position_ids
    if "past_key_value" in accepted:
        kwargs["past_key_value"] = None
    if "output_attentions" in accepted:
        kwargs["output_attentions"] = False
    if "use_cache" in accepted:
        kwargs["use_cache"] = False
    if "cache_position" in accepted:
        kwargs["cache_position"] = cache_position
    if "position_embeddings" in accepted and position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings

    out = block(hidden_states, **kwargs)

    if isinstance(out, tuple):
        return out[0]

    return out


@torch.no_grad()
def compute_initial_hidden_cache(
    model: nn.Module,
    calib_tokens: torch.Tensor,
    batch_size: int,
    main_device: torch.device,
    storage_dtype: torch.dtype,
) -> torch.Tensor:
    embedding = get_embedding_layer(model)
    embedding.eval()

    outs: List[torch.Tensor] = []
    n = calib_tokens.size(0)

    print("\nComputing initial embedding hidden cache...")
    t0 = time.time()

    for i in range(0, n, batch_size):
        input_ids = calib_tokens[i:i + batch_size].to(main_device)
        h = embedding(input_ids)
        outs.append(h.detach().to("cpu", dtype=storage_dtype))

        done = min(i + batch_size, n)
        print(
            f"\r  embeddings: {done}/{n} "
            f"({100.0 * done / n:.1f}%) elapsed={fmt_time(time.time() - t0)}",
            end="",
            flush=True,
        )

    print()

    hidden = torch.cat(outs, dim=0).contiguous()
    print(f"Initial hidden cache shape: {tuple(hidden.shape)}, dtype={hidden.dtype}")

    return hidden


@torch.no_grad()
def run_block_to_cache(
    block: nn.Module,
    backbone: nn.Module,
    hidden_cache: torch.Tensor,
    batch_size: int,
    main_device: torch.device,
    amp_dtype: torch.dtype,
    storage_dtype: torch.dtype,
    desc: str,
) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    n = hidden_cache.size(0)
    autocast_enabled = main_device.type == "cuda"

    t0 = time.time()

    for i in range(0, n, batch_size):
        h = hidden_cache[i:i + batch_size].to(main_device)

        with torch.autocast(device_type=main_device.type, dtype=amp_dtype, enabled=autocast_enabled):
            out = call_decoder_block(block, h, backbone)

        if not torch.isfinite(out).all():
            bad = torch.isfinite(out).logical_not().sum().item()
            total = out.numel()
            print(f"\n[warn] Non-finite output in {desc}: {bad:,}/{total:,}. Sanitizing.")
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0e4, neginf=-1.0e4).clamp(-1.0e4, 1.0e4)

        outs.append(out.detach().to("cpu", dtype=storage_dtype))

        done = min(i + batch_size, n)
        print(
            f"\r  {desc}: {done}/{n} "
            f"({100.0 * done / n:.1f}%) elapsed={fmt_time(time.time() - t0)}",
            end="",
            flush=True,
        )

    print()

    return torch.cat(outs, dim=0).contiguous()


# ============================================================
# Hessian collector
# ============================================================

class HessianCollector:
    def __init__(
        self,
        layer: nn.Linear,
        hessian_device: torch.device,
        dtype: torch.dtype,
        use_factor_2: bool = True,
        name: str = "",
        sanitize_nonfinite: bool = True,
        activation_clamp: float = 1.0e4,
    ):
        self.layer = layer
        self.hessian_device = hessian_device
        self.dtype = dtype
        self.use_factor_2 = use_factor_2
        self.name = name
        self.sanitize_nonfinite = bool(sanitize_nonfinite)
        self.activation_clamp = float(activation_clamp)

        self.in_features = int(layer.in_features)

        self.H = torch.zeros(
            (self.in_features, self.in_features),
            device=hessian_device,
            dtype=dtype,
        )

        self.nsamples = 0
        self.handle = None
        self.warned_nonfinite_input = False
        self.warned_nonfinite_hessian = False

    def _hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        x = inputs[0]

        if not torch.is_tensor(x):
            return

        x = x.detach().reshape(-1, x.size(-1))

        if not torch.isfinite(x).all():
            if not self.warned_nonfinite_input:
                bad = torch.isfinite(x).logical_not().sum().item()
                total = x.numel()
                print(
                    f"\n[warn] Non-finite activation entering {self.name}: "
                    f"{bad:,}/{total:,} values are NaN/Inf."
                )
                self.warned_nonfinite_input = True

            if not self.sanitize_nonfinite:
                raise RuntimeError(f"Non-finite activation entering {self.name}")

            x = torch.nan_to_num(
                x,
                nan=0.0,
                posinf=self.activation_clamp,
                neginf=-self.activation_clamp,
            )

        x = x.clamp(min=-self.activation_clamp, max=self.activation_clamp)

        x = x.to(device=self.hessian_device, dtype=self.dtype)

        scale = 2.0 if self.use_factor_2 else 1.0
        local_H = scale * x.t().matmul(x)

        if not torch.isfinite(local_H).all():
            if not self.warned_nonfinite_hessian:
                bad = torch.isfinite(local_H).logical_not().sum().item()
                total = local_H.numel()
                print(
                    f"\n[warn] Non-finite local Hessian for {self.name}: "
                    f"{bad:,}/{total:,} values are NaN/Inf. "
                    f"Skipping this local contribution instead of zeroing entries."
                )
                self.warned_nonfinite_hessian = True

            if not self.sanitize_nonfinite:
                raise RuntimeError(f"Non-finite local Hessian for {self.name}")

            # Important:
            # Do not zero individual entries; that can destroy PSD structure.
            # Skip the whole corrupted local contribution.
            return

        self.H += local_H
        self.nsamples += x.size(0)

    def register(self) -> None:
        self.handle = self.layer.register_forward_pre_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def collect_block_hessians_once(
    block: nn.Module,
    backbone: nn.Module,
    block_linear_items: List[Tuple[str, nn.Linear]],
    hidden_cache: torch.Tensor,
    batch_size: int,
    main_device: torch.device,
    hessian_dtype: torch.dtype,
    amp_dtype: torch.dtype,
    large_layer_cpu_threshold: int,
) -> Dict[str, Tuple[torch.Tensor, int, torch.device]]:
    collectors: Dict[str, HessianCollector] = {}

    for name, layer in block_linear_items:
        math_device = choose_layer_math_device(
            layer=layer,
            main_device=main_device,
            large_layer_cpu_threshold=large_layer_cpu_threshold,
        )

        collectors[name] = HessianCollector(
            layer=layer,
            hessian_device=math_device,
            dtype=hessian_dtype,
            use_factor_2=True,
            name=name,
            sanitize_nonfinite=True,
            activation_clamp=1.0e4,
        )

    for collector in collectors.values():
        collector.register()

    n = hidden_cache.size(0)
    autocast_enabled = main_device.type == "cuda"

    print(f"  Collecting Hessians for {len(block_linear_items)} linear layers in this block...")
    t0 = time.time()

    for i in range(0, n, batch_size):
        h = hidden_cache[i:i + batch_size].to(main_device)

        with torch.autocast(device_type=main_device.type, dtype=amp_dtype, enabled=autocast_enabled):
            out = call_decoder_block(block, h, backbone)

        if not torch.isfinite(out).all():
            bad = torch.isfinite(out).logical_not().sum().item()
            total = out.numel()
            print(f"\n[warn] Non-finite block output during Hessian pass: {bad:,}/{total:,}.")

        done = min(i + batch_size, n)
        print(
            f"\r  block Hessian pass: {done}/{n} "
            f"({100.0 * done / n:.1f}%) elapsed={fmt_time(time.time() - t0)}",
            end="",
            flush=True,
        )

    print()

    for collector in collectors.values():
        collector.remove()

    out: Dict[str, Tuple[torch.Tensor, int, torch.device]] = {}

    for name, collector in collectors.items():
        out[name] = (collector.H, collector.nsamples, collector.hessian_device)

    return out


# ============================================================
# Blockwise compression
# ============================================================

@torch.no_grad()
def compress_hf_model_blockwise(
    model: nn.Module,
    tokenizer: Any,
    calib_tokens: torch.Tensor,
    selected_layer_names: List[str],
    bits: int,
    sparsity: float,
    pattern: str,
    batch_size: int,
    main_device: torch.device,
    model_dtype: torch.dtype,
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
    store_debug_dequant: bool,
) -> Dict[str, Any]:
    selected_set = set(selected_layer_names)

    layers_prefix, decoder_layers = find_decoder_layers(model)
    backbone = model.model if hasattr(model, "model") else model

    print(f"\nDecoder layers found: {layers_prefix}")
    print(f"Number of decoder blocks: {len(decoder_layers)}")

    storage_dtype = torch.float16 if model_dtype in (torch.float16, torch.bfloat16) else torch.float32

    hidden_cache = compute_initial_hidden_cache(
        model=model,
        calib_tokens=calib_tokens,
        batch_size=batch_size,
        main_device=main_device,
        storage_dtype=storage_dtype,
    )

    joint_layers_out: Dict[str, Any] = {}
    total_selected_in_blocks = 0
    script_block_t0 = time.time()

    for block_idx, block in enumerate(decoder_layers):
        block_prefix = f"{layers_prefix}.{block_idx}"

        block_linear_items: List[Tuple[str, nn.Linear]] = []

        for subname, mod in block.named_modules():
            if not isinstance(mod, nn.Linear):
                continue

            full_name = f"{block_prefix}.{subname}" if subname else block_prefix

            if full_name in selected_set:
                block_linear_items.append((full_name, mod))

        print("\n" + "=" * 100)
        print(f"BLOCK {block_idx}/{len(decoder_layers) - 1}")
        print(f"Selected linear layers in block: {len(block_linear_items)}")
        print(f"Hidden cache shape entering block: {tuple(hidden_cache.shape)}")
        print(f"CUDA memory: {cuda_mem()}")

        for name, layer in block_linear_items:
            print(f"  - {name}: {tuple(layer.weight.shape)}")

        if block_linear_items:
            total_selected_in_blocks += len(block_linear_items)

            hessians = collect_block_hessians_once(
                block=block,
                backbone=backbone,
                block_linear_items=block_linear_items,
                hidden_cache=hidden_cache,
                batch_size=batch_size,
                main_device=main_device,
                hessian_dtype=hessian_dtype,
                amp_dtype=model_dtype,
                large_layer_cpu_threshold=large_layer_cpu_threshold,
            )

            for local_idx, (layer_name, layer) in enumerate(block_linear_items, start=1):
                print("\n" + "-" * 100)
                print(f"  [{local_idx}/{len(block_linear_items)}] Compressing {layer_name}")
                print(f"      weight shape: {tuple(layer.weight.shape)}")
                print(f"      in_features : {layer.in_features}")
                print(f"      out_features: {layer.out_features}")

                H, nsamples, math_device = hessians[layer_name]

                print(f"      H shape       : {tuple(H.shape)}")
                print(f"      H samples     : {nsamples}")
                print(f"      math device   : {math_device}")

                if nsamples == 0:
                    raise RuntimeError(
                        f"No valid Hessian samples collected for {layer_name}. "
                        f"Try --model_dtype bfloat16, --max_seq_len 1024, or smaller calibration."
                    )

                t_layer = time.time()

                result = joint_sparsegpt_gptq_linear(
                    layer=layer,
                    H=H,
                    bits=bits,
                    sparsity=sparsity,
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
                    compress_device=math_device,
                )

                qweight_stored = maybe_pack_qweight(
                    result.qweight_uint8.cpu(),
                    bits=result.bits,
                    packing=result.packing,
                )

                mask_stored = maybe_pack_mask(
                    result.mask.cpu(),
                    mask_packing=result.mask_packing,
                )

                layer_state: Dict[str, Any] = {
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

                if store_debug_dequant:
                    layer_state["dequant_masked_weight"] = result.dequant_masked_weight.cpu()

                joint_layers_out[layer_name] = layer_state

                print(
                    f"      saved tensors:\n"
                    f"        qweight stored : {tuple(qweight_stored.shape)}\n"
                    f"        scales         : {tuple(result.scales.shape)}\n"
                    f"        zero_points    : {tuple(result.zero_points.shape)}\n"
                    f"        mask stored    : {tuple(mask_stored.shape)}\n"
                    f"        sparsity       : {100.0 * result.sparsity:.2f}%\n"
                    f"        layer time     : {fmt_time(time.time() - t_layer)}"
                )

                del H
                del hessians[layer_name]
                del result

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print("\n  Running compressed block to create next hidden cache...")
        hidden_cache = run_block_to_cache(
            block=block,
            backbone=backbone,
            hidden_cache=hidden_cache,
            batch_size=batch_size,
            main_device=main_device,
            amp_dtype=model_dtype,
            storage_dtype=storage_dtype,
            desc=f"block {block_idx} forward",
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed_total = time.time() - script_block_t0
        print(f"Finished block {block_idx}. Elapsed blockwise compression: {fmt_time(elapsed_total)}")

    nonblock_selected = [name for name in selected_layer_names if not name.startswith(layers_prefix + ".")]

    if nonblock_selected:
        print("\n" + "=" * 100)
        print("WARNING: selected non-block linears are not compressed in blockwise path:")
        for name in nonblock_selected:
            print(f"  - {name}")

    print(f"\nTotal selected block linears compressed: {len(joint_layers_out)}")
    print(f"Total selected linears in blocks seen  : {total_selected_in_blocks}")

    return joint_layers_out


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--pattern", type=str, default="unstructured")
    parser.add_argument("--groupsize", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--model_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hessian_dtype", type=str, default="float32", choices=["float64", "float32"])

    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--mask_blocksize", type=int, default=128)

    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")

    parser.add_argument(
        "--suffixes",
        type=str,
        default="",
        help="Comma-separated Linear layer suffixes. Default: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--packing", type=str, default="uint8", choices=["uint8", "packed4"])
    parser.add_argument("--mask_packing", type=str, default="packedbits", choices=["bool", "packedbits"])

    parser.add_argument("--keep_dequantized_state_dict", action="store_true")
    parser.add_argument("--store_debug_dequant", action="store_true")

    parser.add_argument("--act_order", action="store_true")
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--no_quant_aware_mask", action="store_true")

    parser.add_argument("--compress_lm_head", action="store_true")
    parser.add_argument("--skip_tied_lm_head", action="store_true")
    parser.add_argument("--skip_attn_out", action="store_true")
    parser.add_argument("--skip_mlp_out", action="store_true")

    parser.add_argument(
        "--large_layer_cpu_threshold",
        type=int,
        default=8192,
        help="If layer.in_features exceeds this value, Hessian/compression math uses CPU. Use 0 to force all CPU.",
    )

    parser.add_argument("--max_seq_len", type=int, default=0)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Use eager for safest manual blockwise compression.",
    )

    args = parser.parse_args()
    script_t0 = time.time()

    if args.bits < 2 or args.bits > 8:
        raise ValueError("--bits must be in [2, 8].")

    if args.packing == "packed4" and args.bits != 4:
        raise ValueError("--packing packed4 is only valid with --bits 4.")

    if not (0.0 <= args.sparsity < 1.0):
        raise ValueError("--sparsity must be in [0, 1).")

    nm = parse_nm_pattern(args.pattern)

    if nm is not None:
        n_zero, m = nm
        print(f"[info] using semi-structured pattern {args.pattern}: {n_zero} zeros per {m} weights")
    else:
        print(f"[info] using unstructured sparsity={args.sparsity}")

    main_device = torch.device(args.device)
    model_dtype = parse_dtype(args.model_dtype)
    hessian_dtype = parse_dtype(args.hessian_dtype)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("=" * 100)
    print("HF Blockwise Joint SparseGPT + GPTQ Compression")
    print("=" * 100)
    print(f"Model id      : {args.model_id}")
    print(f"Main device   : {main_device}")
    print(f"Model dtype   : {model_dtype}")
    print(f"Hessian dtype : {hessian_dtype}")
    print(f"Attention impl: {args.attn_implementation}")
    print(f"CUDA memory   : {cuda_mem()}")

    print("\nLoading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )

    if hasattr(model, "config"):
        model.config.use_cache = False

    model.eval()
    model.to(main_device)

    print(f"Model loaded. CUDA memory: {cuda_mem()}")

    calib_tokens = load_calibration_tokens(args.calib)
    print(f"Loaded calibration tokens: {tuple(calib_tokens.shape)}")

    if args.max_seq_len > 0 and calib_tokens.size(1) > args.max_seq_len:
        calib_tokens = calib_tokens[:, :args.max_seq_len]
        print(f"Trimmed calibration sequence length to --max_seq_len={args.max_seq_len}")

    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
        max_pos = int(model.config.max_position_embeddings)
        if calib_tokens.size(1) > max_pos:
            calib_tokens = calib_tokens[:, :max_pos]
            print(f"Trimmed calibration sequence length to model max_position_embeddings={max_pos}")

    tied = model_has_tied_lm_head_hf(model)
    if tied:
        print("[info] model appears to have tied input embeddings and lm_head weights")

    suffixes = parse_suffixes(args.suffixes)
    print(f"Layer suffixes selected for compression: {suffixes}")

    selected_layer_names = find_selected_linear_names(
        model=model,
        include=args.include,
        exclude=args.exclude,
        suffixes=suffixes,
        compress_lm_head=bool(args.compress_lm_head),
        skip_tied_lm_head=bool(args.skip_tied_lm_head),
        skip_attn_out=bool(args.skip_attn_out),
        skip_mlp_out=bool(args.skip_mlp_out),
    )

    if not selected_layer_names:
        raise RuntimeError("No HF nn.Linear layers selected for compression.")

    print(f"Selected linear layers: {len(selected_layer_names)}")
    for name in selected_layer_names:
        mod = get_module_by_name(model, name)
        print(f"  - {name}: {tuple(mod.weight.shape)}")

    joint_layers = compress_hf_model_blockwise(
        model=model,
        tokenizer=tokenizer,
        calib_tokens=calib_tokens,
        selected_layer_names=selected_layer_names,
        bits=args.bits,
        sparsity=args.sparsity,
        pattern=args.pattern,
        batch_size=args.batch_size,
        main_device=main_device,
        model_dtype=model_dtype,
        hessian_dtype=hessian_dtype,
        percdamp=args.percdamp,
        blocksize=args.blocksize,
        mask_blocksize=args.mask_blocksize,
        groupsize=args.groupsize,
        packing=args.packing,
        mask_packing=args.mask_packing,
        symmetric=bool(args.symmetric),
        act_order=bool(args.act_order),
        quant_aware_mask=not bool(args.no_quant_aware_mask),
        large_layer_cpu_threshold=int(args.large_layer_cpu_threshold),
        store_debug_dequant=bool(args.store_debug_dequant),
    )

    total_pruned = sum(int(v["pruned_count"]) for v in joint_layers.values())
    total_weights = sum(int(v["total_count"]) for v in joint_layers.values())
    actual_total_sparsity = total_pruned / float(total_weights) if total_weights > 0 else 0.0

    joint_meta = {
        "method": "hf_blockwise_joint_sparsegpt_gptq",
        "model_id": str(args.model_id),
        "bits": int(args.bits),
        "sparsity": float(args.sparsity),
        "pattern": str(args.pattern),
        "groupsize": int(args.groupsize),
        "percdamp": float(args.percdamp),
        "blocksize": int(args.blocksize),
        "mask_blocksize": int(args.mask_blocksize),
        "packing": str(args.packing),
        "mask_packing": str(args.mask_packing),
        "act_order": bool(args.act_order),
        "symmetric": bool(args.symmetric),
        "quant_aware_mask": not bool(args.no_quant_aware_mask),
        "attn_implementation": str(args.attn_implementation),
        "hessian_form": "2 * X^T X + adaptive_damp * I",
        "hessian_collection": "blockwise hooks for all selected Linear layers in a decoder block",
        "no_sparsity_shortcut": bool(nm is None and args.sparsity <= 0.0),
        "calibration_source": args.calib,
        "model_dtype": str(args.model_dtype),
        "hessian_dtype": str(args.hessian_dtype),
        "large_layer_cpu_threshold": int(args.large_layer_cpu_threshold),
        "keep_dequantized_state_dict": bool(args.keep_dequantized_state_dict),
        "store_debug_dequant": bool(args.store_debug_dequant),
        "model_field_contents": (
            "full_dense_dequantized_masked_state_dict"
            if args.keep_dequantized_state_dict
            else "non_compressed_parameters_only"
        ),
        "compress_lm_head": bool(args.compress_lm_head),
        "skip_tied_lm_head": bool(args.skip_tied_lm_head),
        "skip_attn_out": bool(args.skip_attn_out),
        "skip_mlp_out": bool(args.skip_mlp_out),
        "suffixes": list(suffixes),
        "total_pruned_weights": int(total_pruned),
        "total_compressed_weights": int(total_weights),
        "actual_total_sparsity": float(actual_total_sparsity),
        "script_seconds": float(time.time() - script_t0),
        "note": (
            "Compressed linear layers store qweight + scales + zero_points + packed/unpacked mask. "
            "Runtime reconstruction is W = mask * ((q - zero_point) * scale). "
            "For --sparsity 0.0 with unstructured pattern, sparse score computation is skipped."
        ),
    }

    save_hf_joint_checkpoint(
        model=model,
        tokenizer=tokenizer,
        out_path=args.out,
        model_id=args.model_id,
        joint_meta=joint_meta,
        joint_layers=joint_layers,
        keep_dequantized_state_dict=bool(args.keep_dequantized_state_dict),
    )

    print("\nDone.")
    print("Checkpoint now contains:")
    print("  - ckpt['joint_sparsegpt_gptq_meta']")
    print("  - ckpt['joint_sparsegpt_gptq_layers'][layer_name]['qweight'/'scales'/'zero_points'/'mask']")
    print(f"  - qweight packing: {args.packing}")
    print(f"  - mask packing: {args.mask_packing}")
    print(f"  - total joint-compressed layers: {len(joint_layers)}")
    print(f"  - total pruned weights: {total_pruned:,} / {total_weights:,}")
    print(f"  - actual sparsity over selected layers: {100.0 * actual_total_sparsity:.2f}%")
    print(f"  - total script time: {fmt_time(time.time() - script_t0)}")
    print(f"  - CUDA memory: {cuda_mem()}")

    if args.keep_dequantized_state_dict:
        print("  - ckpt['model'] with FULL dense dequantized masked weights")
    else:
        print("  - ckpt['model'] with ONLY NON-COMPRESSED parameters")
        print("    compressed linear weights are omitted from ckpt['model']")


if __name__ == "__main__":
    main()