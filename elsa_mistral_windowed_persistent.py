#!/usr/bin/env python3
"""
Windowed ELSA / ELSA-L style ADMM sparsification for Hugging Face decoder-only LLMs.

Stable v6 dynamic changes: includes v5 FP32 active-window master weights and gradient accumulation, plus optional dynamic ELSA/Fisher sparsity allocation. Dynamic allocation keeps the requested global sparsity but protects tensors whose Fisher-weighted pruning damage is high.

Target use-case:
    - Mistral/LLaMA/Qwen decoder-only models on a single GPU such as RTX 3090.
    - High unstructured sparsity experiments, e.g. 70%, 75%, 80%.
    - Numerical robustness: non-finite checks, gradient clipping, dual clipping,
      projection on CPU, low-precision ADMM state storage.

What this script implements:
    ELSA-style objective:
        min_x f(x) subject to ||x||_0 <= k

    ADMM splitting:
        x update: minimize next-token loss + lambda/2 * ||x - z + u||^2
        z update: sparse projection of x + u
        u update: u <- u + x - z

    Objective-aware projection:
        z = argmin_{z in S} sum_i fisher_i * (z_i - (x_i + u_i))^2

    Single-GPU safeguard:
        Exact full-model ELSA for a 7B model requires far more memory than a 3090.
        This script therefore supports a windowed variant: only selected linears in
        one or a few decoder blocks are trainable at once, while the true LM loss is
        still computed through the full model. ADMM states are stored on disk in
        fp16/bf16/fp32/int8 form and streamed per active window.

        Updated close-to-full-ELSA mode:
            - persists x, z, u for every selected layer across windows/passes;
            - optionally persists Adam m/v moments and step counters per layer;
            - reloads those states when a block is revisited, making the method
              a block-coordinate/windowed approximation of full ELSA instead of
              independent per-window sparse fine-tuning.

Important defaults for RTX 3090:
    --model_dtype float16
    --active_block_window 1
    --batch_size 1
    --max_seq_len 128 or 256
    --state_format int8
    --projection_device cpu
    --gradient_checkpointing

This is not a drop-in reproduction of the paper's multi-GPU FSDP ELSA training.
It is an ELSA/ELSA-L-faithful single-GPU implementation with windowed training,
low-precision state storage, and stability guards.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Basic utilities
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
    name = str(name).lower().strip()
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float32", "fp32"):
        return torch.float32
    if name in ("float64", "fp64", "double"):
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float32:
        return "float32"
    if dtype is torch.float64:
        return "float64"
    return str(dtype)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def finite_or_zero_(t: torch.Tensor, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> torch.Tensor:
    if torch.isfinite(t).all():
        return t
    return torch.nan_to_num(t, nan=nan, posinf=posinf, neginf=neginf)


def safe_tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        x = t.detach()
        finite = torch.isfinite(x)
        finite_frac = float(finite.float().mean().item()) if x.numel() else 1.0
        if finite.any():
            xf = x[finite].float()
            return {
                "finite_frac": finite_frac,
                "mean": float(xf.mean().item()),
                "std": float(xf.std(unbiased=False).item()) if xf.numel() > 1 else 0.0,
                "absmax": float(xf.abs().max().item()),
            }
        return {"finite_frac": finite_frac, "mean": 0.0, "std": 0.0, "absmax": 0.0}


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
    return tokens.long().contiguous()


# ============================================================
# HF model helpers
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
        "Could not find decoder block ModuleList. For Mistral/LLaMA/Qwen this is usually model.layers."
    )


def should_compress_hf_layer_name(
    name: str,
    include: str,
    exclude: str,
    suffixes: Tuple[str, ...],
    compress_lm_head: bool,
    skip_tied_lm_head: bool,
    tied_lm_head: bool,
    skip_attn_out: bool,
    skip_mlp_out: bool,
) -> bool:
    if include and include not in name:
        return False
    if exclude and exclude in name:
        return False
    if name == "lm_head":
        if skip_tied_lm_head and tied_lm_head:
            return False
        return compress_lm_head
    if skip_attn_out and name.endswith("o_proj"):
        return False
    if skip_mlp_out and name.endswith("down_proj"):
        return False
    return name.endswith(suffixes)


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
        if should_compress_hf_layer_name(
            name=name,
            include=include,
            exclude=exclude,
            suffixes=suffixes,
            compress_lm_head=compress_lm_head,
            skip_tied_lm_head=skip_tied_lm_head,
            tied_lm_head=tied,
            skip_attn_out=skip_attn_out,
            skip_mlp_out=skip_mlp_out,
        ):
            out.append(name)
    return out


def block_index_for_layer_name(name: str, layers_prefix: str) -> Optional[int]:
    prefix = layers_prefix + "."
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    first = rest.split(".", 1)[0]
    try:
        return int(first)
    except ValueError:
        return None


def set_all_requires_grad(model: nn.Module, value: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(value)


def set_active_trainable_layers(model: nn.Module, active_names: Sequence[str]) -> List[nn.Parameter]:
    set_all_requires_grad(model, False)
    params: List[nn.Parameter] = []
    for name in active_names:
        mod = get_module_by_name(model, name)
        if not isinstance(mod, nn.Linear):
            raise TypeError(f"Selected module is not nn.Linear: {name}: {type(mod)}")
        mod.weight.requires_grad_(True)
        params.append(mod.weight)
        if mod.bias is not None:
            # Normally Mistral linear layers have no bias. Keep frozen for exact sparsity experiments.
            mod.bias.requires_grad_(False)
    return params


# ============================================================
# Sparsity pattern helpers
# ============================================================


def parse_nm_pattern(pattern: str) -> Optional[Tuple[int, int]]:
    pattern = str(pattern).strip().lower()
    if pattern in ("", "none", "unstructured"):
        return None
    if ":" not in pattern:
        raise ValueError("Pattern must be 'unstructured' or N:M, e.g. '2:4'.")
    a, b = pattern.split(":")
    n = int(a)
    m = int(b)
    if n < 0 or m <= 0 or n > m:
        raise ValueError(f"Invalid N:M pattern {pattern}. Need 0 <= N <= M.")
    return n, m


def exact_unstructured_project(
    v: torch.Tensor,
    sparsity: float,
    fisher: Optional[torch.Tensor] = None,
    fisher_floor: float = 1.0e-12,
    min_keep: int = 1,
) -> Tuple[torch.Tensor, int, int]:
    """
    Project v into an exact unstructured sparsity set, per tensor.

    If fisher is None:
        keep largest |v|^2.
    If fisher is provided:
        keep largest fisher_i * |v_i|^2.

    Returns:
        z, kept_count, total_count
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError("sparsity must be in [0, 1).")

    total = int(v.numel())
    if total == 0:
        return v.clone(), 0, 0

    n_keep = int(round((1.0 - sparsity) * total))
    n_keep = max(int(min_keep), n_keep)
    n_keep = min(total, n_keep)

    if n_keep >= total:
        return finite_or_zero_(v.clone()), total, total

    vf = finite_or_zero_(v.float(), nan=0.0, posinf=0.0, neginf=0.0)
    score = vf.square()

    if fisher is not None:
        ff = finite_or_zero_(fisher.float(), nan=0.0, posinf=0.0, neginf=0.0)
        ff = ff.clamp(min=fisher_floor)
        # Normalize only for numeric scale; it does not change ordering inside this tensor except for clamp.
        mean = ff.mean().clamp(min=fisher_floor)
        ff = (ff / mean).clamp(min=fisher_floor, max=1.0e6)
        score = score * ff

    score = finite_or_zero_(score, nan=0.0, posinf=0.0, neginf=0.0)
    flat_score = score.reshape(-1)

    # CPU topk on a single layer is slower but avoids GPU memory spikes.
    # topk is exact; threshold/quantile approximations are deliberately avoided.
    keep_idx = torch.topk(flat_score, k=n_keep, largest=True, sorted=False).indices
    mask = torch.zeros(total, dtype=torch.bool, device=v.device)
    mask[keep_idx] = True
    mask = mask.view_as(v)

    z = torch.where(mask, vf, torch.zeros_like(vf))
    return z.to(dtype=torch.float32), int(n_keep), total


def nm_project(
    v: torch.Tensor,
    pattern: Tuple[int, int],
    fisher: Optional[torch.Tensor] = None,
    fisher_floor: float = 1.0e-12,
) -> Tuple[torch.Tensor, int, int]:
    """
    Project matrix v with N:M zeros per row-wise group along columns.
    Pattern n:m means n zeros and m-n kept values in every full group.
    """
    if v.ndim != 2:
        raise ValueError("N:M projection currently expects a 2D Linear weight matrix.")
    n_zero, m = pattern
    rows, cols = v.shape
    vf = finite_or_zero_(v.float(), nan=0.0, posinf=0.0, neginf=0.0)
    score = vf.square()
    if fisher is not None:
        ff = finite_or_zero_(fisher.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=fisher_floor)
        ff = (ff / ff.mean().clamp(min=fisher_floor)).clamp(min=fisher_floor, max=1.0e6)
        score = score * ff
    score = finite_or_zero_(score, nan=0.0, posinf=0.0, neginf=0.0)

    mask = torch.ones_like(vf, dtype=torch.bool)
    kept = 0
    total = rows * cols

    for g0 in range(0, cols, m):
        g1 = min(g0 + m, cols)
        group_cols = g1 - g0
        if group_cols == m:
            prune = n_zero
        else:
            prune = int(round((n_zero / float(m)) * group_cols))
        prune = max(0, min(prune, group_cols))
        keep = group_cols - prune
        if keep <= 0:
            mask[:, g0:g1] = False
            continue
        if keep >= group_cols:
            kept += rows * group_cols
            continue
        local_score = score[:, g0:g1]
        keep_idx = torch.topk(local_score, k=keep, dim=1, largest=True, sorted=False).indices
        local_mask = torch.zeros((rows, group_cols), dtype=torch.bool, device=v.device)
        row_idx = torch.arange(rows, device=v.device).view(-1, 1).expand_as(keep_idx)
        local_mask[row_idx, keep_idx] = True
        mask[:, g0:g1] = local_mask
        kept += rows * keep

    z = torch.where(mask, vf, torch.zeros_like(vf))
    return z.to(dtype=torch.float32), int(kept), int(total)


def project_tensor(
    v: torch.Tensor,
    sparsity: float,
    pattern: str,
    fisher: Optional[torch.Tensor],
    fisher_floor: float,
    min_keep: int,
) -> Tuple[torch.Tensor, int, int]:
    nm = parse_nm_pattern(pattern)
    if nm is None:
        return exact_unstructured_project(
            v=v,
            sparsity=sparsity,
            fisher=fisher,
            fisher_floor=fisher_floor,
            min_keep=min_keep,
        )
    return nm_project(v=v, pattern=nm, fisher=fisher, fisher_floor=fisher_floor)


# ============================================================
# Dynamic sparsity allocation helpers
# ============================================================


def get_layer_sparsity(layer_name: str, default_sparsity: float, sparsity_map: Optional[Dict[str, float]]) -> float:
    if sparsity_map is None:
        return float(default_sparsity)
    return float(sparsity_map.get(layer_name, default_sparsity))


def _weighted_mean(values: Sequence[float], weights: Sequence[int]) -> float:
    denom = float(sum(int(w) for w in weights))
    if denom <= 0:
        return 0.0
    return float(sum(float(v) * int(w) for v, w in zip(values, weights)) / denom)


def allocate_keep_ratios_from_sensitivity(
    names: Sequence[str],
    numels: Dict[str, int],
    sensitivities: Dict[str, float],
    target_sparsity: float,
    sparsity_min: float,
    sparsity_max: float,
    alpha: float,
    eps: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Allocate per-layer keep ratios from sensitivity while preserving global keep ratio.

    Higher sensitivity -> higher keep ratio -> lower sparsity.

    We solve, with clamping:
        sum_l r_l N_l = r_global sum_l N_l,
        r_l = C * (D_l + eps)^alpha for unclamped layers,
        r_l in [1 - sparsity_max, 1 - sparsity_min].
    """
    if not names:
        return {}, {}

    target_keep = 1.0 - float(target_sparsity)
    keep_min = max(0.0, min(1.0, 1.0 - float(sparsity_max)))
    keep_max = max(0.0, min(1.0, 1.0 - float(sparsity_min)))
    if keep_min > keep_max:
        keep_min, keep_max = keep_max, keep_min

    total_n = int(sum(int(numels[n]) for n in names))
    target_keep_total = float(target_keep) * float(total_n)

    raw: Dict[str, float] = {}
    for n in names:
        d = float(sensitivities.get(n, 0.0))
        if not math.isfinite(d) or d < 0.0:
            d = 0.0
        raw[n] = float((d + float(eps)) ** float(alpha))

    if not any(math.isfinite(v) and v > 0.0 for v in raw.values()):
        uniform_s = float(target_sparsity)
        return {n: uniform_s for n in names}, {
            "target_sparsity": float(target_sparsity),
            "actual_global_sparsity": float(target_sparsity),
            "fallback": "uniform_all_zero_sensitivity",
        }

    free: Set[str] = set(names)
    keep_ratio: Dict[str, float] = {}
    fixed_keep_total = 0.0

    for _ in range(len(names) + 2):
        if not free:
            break
        raw_weighted_sum = sum(float(numels[n]) * raw[n] for n in free)
        if raw_weighted_sum <= 0.0 or not math.isfinite(raw_weighted_sum):
            c = 0.0
        else:
            c = (target_keep_total - fixed_keep_total) / raw_weighted_sum

        changed = False
        for n in list(free):
            r = c * raw[n]
            if r < keep_min:
                keep_ratio[n] = keep_min
                fixed_keep_total += keep_min * float(numels[n])
                free.remove(n)
                changed = True
            elif r > keep_max:
                keep_ratio[n] = keep_max
                fixed_keep_total += keep_max * float(numels[n])
                free.remove(n)
                changed = True
        if not changed:
            for n in list(free):
                keep_ratio[n] = max(keep_min, min(keep_max, c * raw[n]))
            free.clear()
            break

    actual_keep_total = sum(keep_ratio[n] * float(numels[n]) for n in names)
    residual = target_keep_total - actual_keep_total
    adjustable = [n for n in names if keep_min + 1e-12 < keep_ratio[n] < keep_max - 1e-12]
    if adjustable and abs(residual) > 1e-6:
        n = max(adjustable, key=lambda x: numels[x])
        keep_ratio[n] = max(keep_min, min(keep_max, keep_ratio[n] + residual / float(numels[n])))

    sparsity_map = {n: float(1.0 - keep_ratio[n]) for n in names}
    actual_zero_total = sum(sparsity_map[n] * float(numels[n]) for n in names)
    actual_s = actual_zero_total / max(1.0, float(total_n))

    vals = [sparsity_map[n] for n in names]
    meta = {
        "target_sparsity": float(target_sparsity),
        "actual_global_sparsity_before_rounding": float(actual_s),
        "sparsity_min": float(min(vals)),
        "sparsity_max": float(max(vals)),
        "sparsity_weighted_mean": float(_weighted_mean(vals, [numels[n] for n in names])),
        "keep_min": float(keep_min),
        "keep_max": float(keep_max),
        "alpha": float(alpha),
        "eps": float(eps),
        "num_layers": int(len(names)),
    }
    return sparsity_map, meta


def summarize_sparsity_map(
    title: str,
    sparsity_map: Dict[str, float],
    numels: Dict[str, int],
    limit: int = 12,
) -> None:
    if not sparsity_map:
        return
    total_n = sum(int(numels.get(n, 0)) for n in sparsity_map)
    global_s = sum(float(sparsity_map[n]) * int(numels.get(n, 0)) for n in sparsity_map) / max(1, total_n)
    print(f"\n{title}")
    print(f"  layers={len(sparsity_map)} weighted_global_sparsity≈{100.0 * global_s:.4f}%")
    low = sorted(sparsity_map.items(), key=lambda kv: kv[1])[:limit]
    high = sorted(sparsity_map.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    print("  most protected / lowest sparsity:")
    for n, sp in low:
        print(f"    {100.0*sp:6.2f}%  {n}  N={numels.get(n, 0):,}")
    print("  most pruned / highest sparsity:")
    for n, sp in high:
        print(f"    {100.0*sp:6.2f}%  {n}  N={numels.get(n, 0):,}")


def build_static_dynamic_sparsity_map(
    model: nn.Module,
    selected_names: Sequence[str],
    mode: str,
    target_sparsity: float,
    protect_sparsity: float,
    protect_suffixes: str,
    boundary_blocks: int,
    layers_prefix: str,
    num_decoder_layers: int,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Simple static dynamic profiles for quick ablations."""
    numels = {n: int(get_module_by_name(model, n).weight.numel()) for n in selected_names}
    mode = str(mode).lower().strip()
    suffixes = tuple(x.strip() for x in str(protect_suffixes).split(",") if x.strip())

    protected: Set[str] = set()
    if mode == "module_protect":
        protected = {n for n in selected_names if n.endswith(suffixes)}
    elif mode == "boundary_protect":
        for n in selected_names:
            bi = block_index_for_layer_name(n, layers_prefix)
            if bi is not None and (bi < int(boundary_blocks) or bi >= int(num_decoder_layers) - int(boundary_blocks)):
                protected.add(n)
    else:
        raise ValueError(f"Unsupported static dynamic sparsity mode: {mode}")

    other = [n for n in selected_names if n not in protected]
    if not protected or not other:
        print(f"[warn] dynamic {mode}: protected={len(protected)}, other={len(other)}; falling back to uniform.")
        return {n: float(target_sparsity) for n in selected_names}, {"fallback": "uniform_empty_group"}

    total_n = sum(numels.values())
    protected_n = sum(numels[n] for n in protected)
    other_n = sum(numels[n] for n in other)
    target_zero = float(target_sparsity) * float(total_n)
    protected_zero = float(protect_sparsity) * float(protected_n)
    other_s = (target_zero - protected_zero) / max(1.0, float(other_n))
    other_s = max(0.0, min(0.999, other_s))

    out = {n: (float(protect_sparsity) if n in protected else float(other_s)) for n in selected_names}
    actual_s = sum(out[n] * numels[n] for n in selected_names) / max(1, total_n)
    meta = {
        "mode": mode,
        "target_sparsity": float(target_sparsity),
        "actual_global_sparsity_before_rounding": float(actual_s),
        "protected_layers": int(len(protected)),
        "other_layers": int(len(other)),
        "protected_sparsity": float(protect_sparsity),
        "other_sparsity": float(other_s),
        "protect_suffixes": list(suffixes),
        "boundary_blocks": int(boundary_blocks),
    }
    return out, meta


def compute_fisher_damage_sparsity_map(
    model: nn.Module,
    selected_names: Sequence[str],
    store: Any,
    opt_store: Optional[Any],
    target_sparsity: float,
    base_metric_sparsity: float,
    sparsity_min: float,
    sparsity_max: float,
    alpha: float,
    eps: float,
    fisher_floor: float,
    projection_device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Compute ELSA-native dynamic sparsity from Fisher-weighted pruning damage."""
    print("\nComputing dynamic ELSA/Fisher sparsity profile...")
    print(f"  target global sparsity : {100.0 * float(target_sparsity):.2f}%")
    print(f"  metric base sparsity   : {100.0 * float(base_metric_sparsity):.2f}%")
    print(f"  clamp sparsity range   : [{100.0 * float(sparsity_min):.2f}%, {100.0 * float(sparsity_max):.2f}%]")
    print(f"  alpha                  : {float(alpha):.4f}")

    t0 = time.time()
    numels: Dict[str, int] = {}
    sensitivities: Dict[str, float] = {}
    layer_meta: Dict[str, Any] = {}
    used_fisher = 0

    for idx, name in enumerate(selected_names, start=1):
        mod = get_module_by_name(model, name)
        n = int(mod.weight.numel())
        numels[name] = n

        z = store.load_tensor(name, "z", device=projection_device, dtype=torch.float32)
        u = store.load_tensor(name, "u", device=projection_device, dtype=torch.float32)
        v = finite_or_zero_(z + u, nan=0.0, posinf=0.0, neginf=0.0)
        del z, u

        fisher = None
        fisher_available = False
        if opt_store is not None and opt_store.exists(name, "adam_v"):
            try:
                fisher = opt_store.load_tensor(name, "adam_v", device=projection_device, dtype=torch.float32)
                fisher = finite_or_zero_(fisher, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=float(fisher_floor))
                if float(fisher.max().item()) > 0.0:
                    fisher = (fisher / fisher.mean().clamp(min=float(fisher_floor))).clamp(min=float(fisher_floor), max=1.0e6)
                    fisher_available = True
                    used_fisher += 1
                else:
                    fisher = None
            except Exception as exc:
                print(f"  [warn] could not load Fisher/Adam v for {name}: {exc}; using magnitude damage.")
                fisher = None

        score = v.float().square()
        if fisher is not None:
            score.mul_(fisher)
        score = finite_or_zero_(score, nan=0.0, posinf=0.0, neginf=0.0)
        flat = score.reshape(-1)
        total_score = float(flat.sum().item())

        keep = int(round((1.0 - float(base_metric_sparsity)) * n))
        keep = max(1, min(n, keep))
        prune = max(0, n - keep)
        if prune <= 0 or keep >= n:
            damage_sum = 0.0
            damage_mean = 0.0
            keep_score = total_score
        else:
            keep_vals = torch.topk(flat, k=keep, largest=True, sorted=False).values
            keep_score = float(keep_vals.sum().item())
            damage_sum = max(0.0, total_score - keep_score)
            damage_mean = damage_sum / max(1, prune)
            del keep_vals

        if not math.isfinite(damage_mean):
            damage_mean = 0.0
        sensitivities[name] = float(damage_mean)
        layer_meta[name] = {
            "numel": int(n),
            "base_metric_sparsity": float(base_metric_sparsity),
            "base_keep": int(keep),
            "base_prune": int(prune),
            "damage_sum": float(damage_sum),
            "damage_mean": float(damage_mean),
            "total_score": float(total_score),
            "kept_score_at_base": float(keep_score),
            "used_fisher": bool(fisher_available),
        }
        print(f"  [{idx}/{len(selected_names)}] {name}: D_mean={damage_mean:.6e} used_fisher={fisher_available} N={n:,}")

        del v, fisher, score, flat
        gc.collect()
        if projection_device.type == "cuda":
            torch.cuda.empty_cache()

    sparsity_map, alloc_meta = allocate_keep_ratios_from_sensitivity(
        names=list(selected_names),
        numels=numels,
        sensitivities=sensitivities,
        target_sparsity=float(target_sparsity),
        sparsity_min=float(sparsity_min),
        sparsity_max=float(sparsity_max),
        alpha=float(alpha),
        eps=float(eps),
    )

    for name in selected_names:
        layer_meta[name]["allocated_sparsity"] = float(sparsity_map[name])
        layer_meta[name]["allocated_keep_ratio"] = float(1.0 - sparsity_map[name])

    meta = {
        "mode": "fisher_damage",
        "seconds": float(time.time() - t0),
        "target_sparsity": float(target_sparsity),
        "base_metric_sparsity": float(base_metric_sparsity),
        "sparsity_min": float(sparsity_min),
        "sparsity_max": float(sparsity_max),
        "alpha": float(alpha),
        "eps": float(eps),
        "used_fisher_layers": int(used_fisher),
        "total_layers": int(len(selected_names)),
        "allocation": alloc_meta,
        "layers": layer_meta,
    }
    summarize_sparsity_map("Dynamic ELSA/Fisher sparsity allocation", sparsity_map, numels)
    print(f"Dynamic profile computed in {fmt_time(meta['seconds'])}.")
    return sparsity_map, meta


# ============================================================
# ADMM state store
# ============================================================


def safe_state_name(layer_name: str) -> str:
    return layer_name.replace(".", "__").replace("/", "__slash__")


@dataclass
class TensorStateInfo:
    format: str
    dtype: str
    shape: List[int]
    scale: Optional[float] = None


class ADMMStateStore:
    """
    Disk-backed storage for z and u ADMM variables.

    Supported formats:
        fp32, fp16, bf16: torch.save(tensor)
        int8: torch.save({'q': int8 tensor, 'scale': scalar, ...})

    int8 is used as ELSA-L-like low-precision state storage:
        q = round(x / scale), scale = max(abs(x)) / 127
        x ~= q * scale
    """

    def __init__(self, root: str | Path, state_format: str = "int8"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_format = state_format.lower().strip()
        if self.state_format not in ("int8", "fp16", "bf16", "fp32"):
            raise ValueError("state_format must be one of: int8, fp16, bf16, fp32")

    def layer_dir(self, layer_name: str) -> Path:
        d = self.root / safe_state_name(layer_name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def path(self, layer_name: str, slot: str) -> Path:
        return self.layer_dir(layer_name) / f"{slot}.pt"

    def exists(self, layer_name: str, slot: str) -> bool:
        return self.path(layer_name, slot).exists()

    @staticmethod
    def _quantize_int8(t: torch.Tensor) -> Dict[str, Any]:
        x = finite_or_zero_(t.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        maxabs = float(x.abs().max().item()) if x.numel() else 0.0
        if not math.isfinite(maxabs) or maxabs <= 0.0:
            scale = 1.0
            q = torch.zeros_like(x, dtype=torch.int8)
        else:
            scale = maxabs / 127.0
            q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
        return {
            "format": "int8_scale",
            "q": q,
            "scale": float(scale),
            "shape": list(x.shape),
        }

    def save_tensor(self, layer_name: str, slot: str, t: torch.Tensor) -> None:
        p = self.path(layer_name, slot)
        p.parent.mkdir(parents=True, exist_ok=True)
        if self.state_format == "int8":
            obj = self._quantize_int8(t)
        elif self.state_format == "fp16":
            obj = finite_or_zero_(t.detach().cpu().to(torch.float16), nan=0.0, posinf=0.0, neginf=0.0)
        elif self.state_format == "bf16":
            obj = finite_or_zero_(t.detach().cpu().to(torch.bfloat16), nan=0.0, posinf=0.0, neginf=0.0)
        elif self.state_format == "fp32":
            obj = finite_or_zero_(t.detach().cpu().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise AssertionError(self.state_format)
        torch.save(obj, p)

    def load_tensor(
        self,
        layer_name: str,
        slot: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        p = self.path(layer_name, slot)
        if not p.exists():
            raise FileNotFoundError(f"Missing ADMM state: {p}")
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, dict) and obj.get("format") == "int8_scale":
            q = obj["q"].to(device=device)
            scale = float(obj["scale"])
            out = q.float().mul(scale)
            return finite_or_zero_(out.to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
        if torch.is_tensor(obj):
            return finite_or_zero_(obj.to(device=device, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
        raise ValueError(f"Unsupported state object at {p}: {type(obj)}")

    def save_json(self, name: str, data: Dict[str, Any]) -> None:
        with open(self.root / name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def load_json(self, name: str) -> Dict[str, Any]:
        with open(self.root / name, "r", encoding="utf-8") as f:
            return json.load(f)


# ============================================================
# Calibration token loader
# ============================================================


class InfiniteTokenLoader:
    """
    Infinite calibration loader with optional random crops.

    Important for LLM compression:
        A calibration tensor often has shape [N, 2048]. If we train with
        --max_seq_len 128 and simply truncate to [:, :128], we only ever see
        the first 128 tokens of every sample. That is a very small and biased
        subset. This loader keeps the full calibration tensor and samples
        random contiguous crops of length max_seq_len, so a 128x2048 file can
        provide many more distinct token positions.
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        seed: int,
        max_seq_len: int = 0,
        crop_mode: str = "random",
        drop_last: bool = False,
    ):
        if tokens.ndim != 2:
            raise ValueError("tokens must be [N, T]")
        self.tokens = tokens.contiguous()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.max_seq_len = int(max_seq_len)
        self.crop_mode = str(crop_mode).lower().strip()
        if self.crop_mode not in ("random", "prefix", "sliding", "none"):
            raise ValueError("crop_mode must be random, prefix, sliding, or none")
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.pos = 0
        self.step = 0
        self.order = torch.arange(tokens.size(0), dtype=torch.long)
        self.crop_gen = torch.Generator(device="cpu")
        self.crop_gen.manual_seed(self.seed + 104729)
        self._reshuffle()

    def _reshuffle(self) -> None:
        if self.shuffle:
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed + self.epoch)
            self.order = torch.randperm(self.tokens.size(0), generator=g)
        else:
            self.order = torch.arange(self.tokens.size(0), dtype=torch.long)
        self.pos = 0
        self.epoch += 1

    def _crop_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if self.max_seq_len <= 0:
            return batch.contiguous()
        B, T = batch.shape
        L = min(self.max_seq_len, T)
        if L >= T:
            return batch.contiguous()

        if self.crop_mode == "none":
            return batch.contiguous()
        if self.crop_mode == "prefix":
            return batch[:, :L].contiguous()

        max_start = T - L
        out = torch.empty((B, L), dtype=batch.dtype)
        if self.crop_mode == "random":
            starts = torch.randint(0, max_start + 1, (B,), generator=self.crop_gen)
        elif self.crop_mode == "sliding":
            # Deterministic coverage of the long sequence. Different rows are offset.
            base = (self.step * L) % (max_start + 1)
            starts = torch.tensor([(base + i * 997) % (max_start + 1) for i in range(B)], dtype=torch.long)
        else:
            raise AssertionError(self.crop_mode)

        for i, st in enumerate(starts.tolist()):
            out[i].copy_(batch[i, st:st + L])
        self.step += 1
        return out.contiguous()

    def next(self) -> torch.Tensor:
        n = self.tokens.size(0)
        if self.pos >= n:
            self._reshuffle()
        end = min(self.pos + self.batch_size, n)
        idx = self.order[self.pos:end]
        self.pos = end
        if idx.numel() < self.batch_size and self.drop_last:
            self._reshuffle()
            end = min(self.pos + self.batch_size, n)
            idx = self.order[self.pos:end]
            self.pos = end
        batch = self.tokens[idx]
        return self._crop_batch(batch)


# ============================================================
# Safe FP32 AdamW for active parameters
# ============================================================


class SafeAdamWFP32:
    """
    Minimal AdamW optimizer with FP32 states for a small active parameter window.

    This avoids relying on FP16 optimizer states. It is intentionally simple and robust.
    The model weights may be FP16/BF16, but Adam's exp_avg and exp_avg_sq are FP32.
    """

    def __init__(
        self,
        named_params: List[Tuple[str, nn.Parameter]],
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        state_device: torch.device | str = "cuda",
        max_state_abs: float = 1.0e6,
        use_master_weights: bool = True,
    ):
        self.named_params = [(n, p) for n, p in named_params if p.requires_grad]
        self.use_master_weights = bool(use_master_weights)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.state_device = torch.device(state_device)
        self.max_state_abs = float(max_state_abs)
        self.step_num = 0
        self.state: Dict[int, Dict[str, torch.Tensor]] = {}
        for _, p in self.named_params:
            self.state[id(p)] = {
                "exp_avg": torch.zeros_like(p.detach(), dtype=torch.float32, device=self.state_device),
                "exp_avg_sq": torch.zeros_like(p.detach(), dtype=torch.float32, device=self.state_device),
            }
            if self.use_master_weights:
                # FP32 active-window master copy. This is important for low-LR refinement:
                # repeated 1e-7...1e-6 updates can disappear if every update is rounded
                # through FP16 weights before the next step/projection.
                self.state[id(p)]["master"] = finite_or_zero_(
                    p.detach().to(device=self.state_device, dtype=torch.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clone()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for _, p in self.named_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    def get_exp_avg_sq(self, p: nn.Parameter, device: torch.device | str = "cpu") -> torch.Tensor:
        st = self.state[id(p)]["exp_avg_sq"]
        return st.detach().to(device=device, dtype=torch.float32)

    def has_master_param(self, p: nn.Parameter) -> bool:
        return self.use_master_weights and ("master" in self.state.get(id(p), {}))

    def get_master_param(self, p: nn.Parameter, device: torch.device | str = "cpu") -> torch.Tensor:
        if self.has_master_param(p):
            return self.state[id(p)]["master"].detach().to(device=device, dtype=torch.float32)
        return p.detach().to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def set_master_param(self, p: nn.Parameter, value: torch.Tensor) -> None:
        vf = finite_or_zero_(value.detach().to(device=self.state_device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if self.has_master_param(p):
            self.state[id(p)]["master"].copy_(vf)
        p.data.copy_(vf.to(device=p.device, dtype=p.dtype))

    @torch.no_grad()
    def step(
        self,
        lr: float,
        weight_decay: float = 0.0,
        update_clip: float = 0.0,
        weight_clip: float = 0.0,
    ) -> Dict[str, float]:
        self.step_num += 1
        lr = float(lr)
        beta1, beta2 = self.beta1, self.beta2
        bc1 = 1.0 - beta1 ** self.step_num
        bc2 = 1.0 - beta2 ** self.step_num
        total_update_norm_sq = 0.0
        total_param = 0
        skipped = 0

        for _, p in self.named_params:
            if p.grad is None:
                skipped += 1
                continue
            g = p.grad.detach().to(device=self.state_device, dtype=torch.float32)
            if not torch.isfinite(g).all():
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

            st = self.state[id(p)]
            m = st["exp_avg"]
            v = st["exp_avg_sq"]
            m.mul_(beta1).add_(g, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            if self.max_state_abs > 0:
                m.clamp_(min=-self.max_state_abs, max=self.max_state_abs)
                v.clamp_(min=0.0, max=self.max_state_abs)

            m_hat = m / bc1
            v_hat = v / bc2
            update = m_hat / (v_hat.sqrt().add_(self.eps))

            if weight_decay > 0:
                update = update.add(p.detach().to(device=self.state_device, dtype=torch.float32), alpha=float(weight_decay))

            if update_clip > 0:
                update = update.clamp(min=-float(update_clip), max=float(update_clip))

            if self.use_master_weights and "master" in st:
                master = st["master"]
            else:
                master = p.detach().to(device=self.state_device, dtype=torch.float32)

            master.add_(update, alpha=-lr)
            if weight_clip > 0:
                master.clamp_(min=-float(weight_clip), max=float(weight_clip))
            if not torch.isfinite(master).all():
                master.copy_(torch.nan_to_num(master, nan=0.0, posinf=float(weight_clip or 1.0), neginf=-float(weight_clip or 1.0)))

            total_update_norm_sq += float(update.float().square().sum().item())
            total_param += update.numel()
            p.data.copy_(master.to(device=p.device, dtype=p.dtype))

        return {
            "step": float(self.step_num),
            "update_rms": math.sqrt(total_update_norm_sq / max(1, total_param)),
            "skipped_params": float(skipped),
        }

    def load_state_from_store(
        self,
        store: Optional[ADMMStateStore],
        strict_shape: bool = True,
    ) -> Dict[str, float]:
        """
        Load per-layer Adam m/v and step counters. The names in self.named_params
        should be the selected layer names, not 'name.weight'.
        """
        if store is None:
            return {"loaded_m": 0.0, "loaded_v": 0.0, "loaded_steps": 0.0, "start_step": float(self.step_num)}

        loaded_m = 0
        loaded_v = 0
        loaded_steps = []
        for layer_name, p in self.named_params:
            st = self.state[id(p)]
            if store.exists(layer_name, "adam_m"):
                m = store.load_tensor(layer_name, "adam_m", device=self.state_device, dtype=torch.float32)
                if (not strict_shape) or tuple(m.shape) == tuple(p.shape):
                    st["exp_avg"].copy_(finite_or_zero_(m, nan=0.0, posinf=0.0, neginf=0.0))
                    loaded_m += 1
                del m
            if store.exists(layer_name, "adam_v"):
                v = store.load_tensor(layer_name, "adam_v", device=self.state_device, dtype=torch.float32)
                if (not strict_shape) or tuple(v.shape) == tuple(p.shape):
                    st["exp_avg_sq"].copy_(finite_or_zero_(v, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0))
                    loaded_v += 1
                del v
            step_path = store.path(layer_name, "adam_step")
            if step_path.exists():
                try:
                    obj = torch.load(step_path, map_location="cpu")
                    if isinstance(obj, dict) and "step" in obj:
                        loaded_steps.append(int(obj["step"]))
                except Exception:
                    pass

        if loaded_steps:
            # Active layers in one window should normally share the same local Adam count.
            # Use the minimum to avoid over-correcting bias for any layer with less history.
            self.step_num = int(min(loaded_steps))

        return {
            "loaded_m": float(loaded_m),
            "loaded_v": float(loaded_v),
            "loaded_steps": float(len(loaded_steps)),
            "start_step": float(self.step_num),
        }

    def save_state_to_store(self, store: Optional[ADMMStateStore]) -> Dict[str, float]:
        if store is None:
            return {"saved_m": 0.0, "saved_v": 0.0, "saved_steps": 0.0}
        saved_m = 0
        saved_v = 0
        saved_steps = 0
        for layer_name, p in self.named_params:
            st = self.state[id(p)]
            store.save_tensor(layer_name, "adam_m", st["exp_avg"].detach().cpu())
            store.save_tensor(layer_name, "adam_v", st["exp_avg_sq"].detach().cpu().clamp(min=0.0))
            torch.save({"step": int(self.step_num)}, store.path(layer_name, "adam_step"))
            saved_m += 1
            saved_v += 1
            saved_steps += 1
        return {"saved_m": float(saved_m), "saved_v": float(saved_v), "saved_steps": float(saved_steps)}

    def release(self) -> None:
        self.state.clear()
        self.named_params.clear()


# ============================================================
# Loss / ADMM penalty / gradient safety
# ============================================================


def compute_lm_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    amp_dtype: torch.dtype,
    device: torch.device,
    autocast_enabled: bool = True,
) -> torch.Tensor:
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled and device.type == "cuda"):
        out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits

    # Compute CE in FP32. This is a crucial stability guard for FP16 Mistral.
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    vocab = shift_logits.size(-1)
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.reshape(-1),
        reduction="mean",
    )
    return loss


def compute_admm_penalty(
    model: nn.Module,
    active_names: Sequence[str],
    active_state: Dict[str, Dict[str, torch.Tensor]],
    lambda_value: float,
    normalization: str,
    diff_clip: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if lambda_value <= 0.0 or not active_names:
        # Need a tensor attached to current device for easy addition.
        first = next(model.parameters())
        return first.new_tensor(0.0, dtype=torch.float32), {"admm_rms": 0.0, "admm_absmax": 0.0}

    total_sq: Optional[torch.Tensor] = None
    total_numel = 0
    max_abs = 0.0

    for name in active_names:
        mod = get_module_by_name(model, name)
        p = mod.weight
        z = active_state[name]["z"].to(device=p.device)
        u = active_state[name]["u"].to(device=p.device)
        diff = p.float() - z.float() + u.float()
        if diff_clip > 0:
            diff = diff.clamp(min=-float(diff_clip), max=float(diff_clip))
        diff = finite_or_zero_(diff, nan=0.0, posinf=0.0, neginf=0.0)
        sq = diff.square().sum()
        total_sq = sq if total_sq is None else total_sq + sq
        total_numel += diff.numel()
        try:
            max_abs = max(max_abs, float(diff.abs().max().item()))
        except RuntimeError:
            pass

    assert total_sq is not None
    if normalization == "mean":
        base = total_sq / max(1, total_numel)
    elif normalization == "sum":
        base = total_sq
    else:
        raise ValueError("normalization must be 'mean' or 'sum'")

    penalty = 0.5 * float(lambda_value) * base
    rms = math.sqrt(float((total_sq / max(1, total_numel)).detach().cpu().item()))
    return penalty, {"admm_rms": rms, "admm_absmax": max_abs}


@torch.no_grad()
def sanitize_and_clip_gradients(
    named_params: List[Tuple[str, nn.Parameter]],
    max_grad_norm: float,
    grad_value_clip: float,
) -> Dict[str, float]:
    total_sq = 0.0
    nonfinite = 0
    count = 0

    for _, p in named_params:
        if p.grad is None:
            continue
        g = p.grad
        if not torch.isfinite(g).all():
            nonfinite += int((~torch.isfinite(g)).sum().item())
            g.data = torch.nan_to_num(g.data, nan=0.0, posinf=0.0, neginf=0.0)
        if grad_value_clip > 0:
            g.data.clamp_(min=-float(grad_value_clip), max=float(grad_value_clip))
        gf = g.detach().float()
        total_sq += float(gf.square().sum().item())
        count += gf.numel()

    grad_norm = math.sqrt(max(0.0, total_sq))
    scale = 1.0
    if max_grad_norm > 0.0 and grad_norm > max_grad_norm:
        scale = float(max_grad_norm / (grad_norm + 1.0e-12))
        for _, p in named_params:
            if p.grad is not None:
                p.grad.data.mul_(scale)
        grad_norm = float(max_grad_norm)

    return {
        "grad_norm": float(grad_norm),
        "grad_rms": math.sqrt(total_sq / max(1, count)),
        "grad_scale": float(scale),
        "nonfinite_grad_values": float(nonfinite),
    }


def lambda_schedule_value(kind: str, final_lambda: float, progress: float, warmup_frac: float) -> float:
    """Monotone ADMM penalty schedule.

    Previous versions warmed up lambda, then restarted the post-warmup cosine/linear
    schedule near zero. That can remove the ADMM constraint exactly when training
    begins to rely on it. This version never decreases after warmup.

    Semantics:
      - warmup_frac > 0: linearly warm up from 0 to final_lambda, then hold final_lambda.
      - warmup_frac = 0 and kind == linear/cosine: monotone increase from 0 to final_lambda
        over the whole run.
      - kind == constant: final_lambda from the beginning when warmup_frac=0.
    """
    progress = min(1.0, max(0.0, float(progress)))
    final_lambda = float(final_lambda)
    warmup_frac = min(0.999, max(0.0, float(warmup_frac)))
    kind = kind.lower().strip()

    if final_lambda <= 0.0:
        return 0.0

    if warmup_frac > 0.0:
        if progress < warmup_frac:
            return final_lambda * (progress / max(warmup_frac, 1.0e-12))
        return final_lambda

    if kind == "constant":
        return final_lambda
    if kind == "linear":
        return final_lambda * progress
    if kind == "cosine":
        return final_lambda * 0.5 * (1.0 - math.cos(math.pi * progress))
    raise ValueError("lambda_schedule must be constant, linear, or cosine")


def lr_schedule_value(kind: str, base_lr: float, progress: float, min_lr_ratio: float) -> float:
    progress = min(1.0, max(0.0, float(progress)))
    kind = kind.lower().strip()
    min_lr = float(base_lr) * float(min_lr_ratio)
    if kind == "constant":
        return float(base_lr)
    if kind == "linear":
        return min_lr + (float(base_lr) - min_lr) * (1.0 - progress)
    if kind == "cosine":
        return min_lr + (float(base_lr) - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError("lr_schedule must be constant, linear, or cosine")


# ============================================================
# ADMM initialization and projection updates
# ============================================================


@torch.no_grad()
def initialize_admm_states(
    model: nn.Module,
    selected_names: Sequence[str],
    store: ADMMStateStore,
    sparsity: float,
    pattern: str,
    min_keep_per_tensor: int,
    overwrite: bool,
    objective_aware_initial: bool = False,
    x_store: Optional[ADMMStateStore] = None,
    sparsity_map: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Initializes z = projection(W) and u = 0 for every selected layer.

    The initial projection is magnitude-based by default because no Fisher estimate is available yet.
    """
    print("\nInitializing ADMM states...")
    meta: Dict[str, Any] = {
        "layers": {},
        "total_weights": 0,
        "total_kept": 0,
        "sparsity": float(sparsity),
        "pattern": str(pattern),
        "state_format": str(store.state_format),
    }
    t0 = time.time()

    for idx, name in enumerate(selected_names, start=1):
        z_exists = store.exists(name, "z")
        u_exists = store.exists(name, "u")
        if z_exists and u_exists and not overwrite:
            print(f"  [{idx}/{len(selected_names)}] keeping existing state: {name}")
            continue

        mod = get_module_by_name(model, name)
        if not isinstance(mod, nn.Linear):
            raise TypeError(f"Selected module is not nn.Linear: {name}")

        w_cpu = finite_or_zero_(mod.weight.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        target_sparsity = get_layer_sparsity(name, sparsity, sparsity_map)
        z, kept, total = project_tensor(
            v=w_cpu,
            sparsity=target_sparsity,
            pattern=pattern,
            fisher=None,
            fisher_floor=1.0e-12,
            min_keep=min_keep_per_tensor,
        )
        u = torch.zeros_like(w_cpu, dtype=torch.float32)
        store.save_tensor(name, "z", z)
        store.save_tensor(name, "u", u)
        if x_store is not None:
            # Persist the ADMM x variable as well. During a single uninterrupted run,
            # x lives in the model weights. Persisting it makes resume/revisit behavior
            # much closer to full ADMM: each block returns to its own previous x state.
            x_store.save_tensor(name, "x", w_cpu)

        layer_sparsity = 1.0 - kept / max(1, total)
        meta["layers"][name] = {
            "shape": list(w_cpu.shape),
            "total": int(total),
            "kept": int(kept),
            "sparsity": float(layer_sparsity),
            "target_sparsity": float(target_sparsity),
        }
        meta["total_weights"] += int(total)
        meta["total_kept"] += int(kept)
        print(
            f"  [{idx}/{len(selected_names)}] {name}: shape={tuple(w_cpu.shape)} "
            f"kept={kept:,}/{total:,} sparsity={100.0 * layer_sparsity:.2f}%"
        )
        del w_cpu, z, u
        gc.collect()

    if meta["total_weights"] > 0:
        meta["actual_sparsity"] = 1.0 - meta["total_kept"] / meta["total_weights"]
    else:
        meta["actual_sparsity"] = None
    meta["seconds"] = time.time() - t0
    store.save_json("admm_init_meta.json", meta)
    print(f"ADMM state initialization finished in {fmt_time(meta['seconds'])}.")
    return meta


@torch.no_grad()
def load_active_states(
    store: ADMMStateStore,
    active_names: Sequence[str],
    device: torch.device,
    active_state_dtype: torch.dtype,
) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in active_names:
        z = store.load_tensor(name, "z", device=device, dtype=active_state_dtype)
        u = store.load_tensor(name, "u", device=device, dtype=active_state_dtype)
        out[name] = {"z": z, "u": u}
    return out


@torch.no_grad()
def save_active_states(
    store: ADMMStateStore,
    active_state: Dict[str, Dict[str, torch.Tensor]],
) -> None:
    for name, st in active_state.items():
        store.save_tensor(name, "z", st["z"])
        store.save_tensor(name, "u", st["u"])



@torch.no_grad()
def load_active_x_states_into_model(
    model: nn.Module,
    active_names: Sequence[str],
    x_store: Optional[ADMMStateStore],
    admm_store: ADMMStateStore,
    device: torch.device,
    fallback_to_z: bool,
) -> Dict[str, Any]:
    """
    Restore the ADMM x variable for active layers before optimizing a window.

    Full ELSA has a persistent x for all parameters. In the windowed single-GPU
    version, inactive layers live in model memory during one uninterrupted run,
    but explicit x persistence is needed for robust resume and for mathematically
    cleaner repeated block-coordinate ADMM sweeps.
    """
    loaded_x = 0
    loaded_z_fallback = 0
    missing = 0
    for name in active_names:
        mod = get_module_by_name(model, name)
        if not isinstance(mod, nn.Linear):
            continue
        source = None
        if x_store is not None and x_store.exists(name, "x"):
            source = x_store.load_tensor(name, "x", device=device, dtype=torch.float32)
            loaded_x += 1
        elif fallback_to_z and admm_store.exists(name, "z"):
            source = admm_store.load_tensor(name, "z", device=device, dtype=torch.float32)
            loaded_z_fallback += 1
        else:
            missing += 1

        if source is not None:
            source = finite_or_zero_(source, nan=0.0, posinf=0.0, neginf=0.0)
            mod.weight.data.copy_(source.to(device=mod.weight.device, dtype=mod.weight.dtype))
            del source

    return {"loaded_x": loaded_x, "loaded_z_fallback": loaded_z_fallback, "missing_x": missing}


@torch.no_grad()
def save_active_x_states_from_model(
    model: nn.Module,
    active_names: Sequence[str],
    x_store: Optional[ADMMStateStore],
) -> None:
    if x_store is None:
        return
    for name in active_names:
        mod = get_module_by_name(model, name)
        if isinstance(mod, nn.Linear):
            x_store.save_tensor(name, "x", finite_or_zero_(mod.weight.detach().float().cpu()))


@torch.no_grad()
def admm_project_active_layers(
    model: nn.Module,
    active_names: Sequence[str],
    optimizer: Optional[SafeAdamWFP32],
    store: ADMMStateStore,
    active_state: Dict[str, Dict[str, torch.Tensor]],
    sparsity: float,
    pattern: str,
    objective_aware: bool,
    projection_device: torch.device,
    fisher_floor: float,
    min_keep_per_tensor: int,
    dual_clip: float,
    dual_lr: float,
    copy_z_to_x: bool,
    fisher_warmup_steps: int = 0,
    sparsity_map: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Performs ADMM z/u update for active layers.

        v = x + u
        z = projection(v)
        u = u + dual_lr * (x - z)

    If objective_aware=True, projection score is fisher * v^2 using Adam second moment.
    """
    print("\n  ADMM projection update...")
    t0 = time.time()
    total = 0
    kept = 0
    layer_logs = {}

    for idx, name in enumerate(active_names, start=1):
        mod = get_module_by_name(model, name)
        p = mod.weight

        # Move to CPU for exact top-k projection by default.
        # If FP32 active master weights are enabled, project the master copy rather than
        # the rounded FP16 parameter tensor. This preserves small accumulated updates.
        if optimizer is not None and hasattr(optimizer, "has_master_param") and optimizer.has_master_param(p):
            x = finite_or_zero_(optimizer.get_master_param(p, device=projection_device), nan=0.0, posinf=0.0, neginf=0.0)
        else:
            x = finite_or_zero_(p.detach().float().to(projection_device), nan=0.0, posinf=0.0, neginf=0.0)
        u_old = active_state[name]["u"].detach().float().to(projection_device)
        v = finite_or_zero_(x + u_old, nan=0.0, posinf=0.0, neginf=0.0)

        fisher = None
        if objective_aware and optimizer is not None and optimizer.step_num >= int(fisher_warmup_steps):
            try:
                fisher = optimizer.get_exp_avg_sq(p, device=projection_device)
                fisher = finite_or_zero_(fisher, nan=0.0, posinf=0.0, neginf=0.0)
                # Avoid an all-zero Fisher in the very first steps.
                if float(fisher.max().item()) <= 0.0:
                    fisher = None
            except Exception as exc:
                print(f"    [warn] could not read fisher for {name}: {exc}. Falling back to magnitude projection.")
                fisher = None

        target_sparsity = get_layer_sparsity(name, sparsity, sparsity_map)
        z_new, layer_kept, layer_total = project_tensor(
            v=v,
            sparsity=target_sparsity,
            pattern=pattern,
            fisher=fisher,
            fisher_floor=fisher_floor,
            min_keep=min_keep_per_tensor,
        )

        u_new = finite_or_zero_(u_old + float(dual_lr) * (x - z_new), nan=0.0, posinf=0.0, neginf=0.0)
        if dual_clip > 0:
            u_new.clamp_(min=-float(dual_clip), max=float(dual_clip))

        if copy_z_to_x:
            # Not default ELSA, but useful for stabilizing very small GPUs if requested.
            if optimizer is not None and hasattr(optimizer, "set_master_param"):
                optimizer.set_master_param(p, z_new)
            else:
                p.data.copy_(z_new.to(device=p.device, dtype=p.dtype))
            x = z_new

        store.save_tensor(name, "z", z_new)
        store.save_tensor(name, "u", u_new)

        # Keep active state on GPU updated for the next minibatches.
        active_state[name]["z"] = z_new.to(device=p.device, dtype=active_state[name]["z"].dtype)
        active_state[name]["u"] = u_new.to(device=p.device, dtype=active_state[name]["u"].dtype)

        total += layer_total
        kept += layer_kept
        layer_sparsity = 1.0 - layer_kept / max(1, layer_total)
        u_stats = safe_tensor_stats(u_new)
        z_absmax = float(z_new.abs().max().item()) if z_new.numel() else 0.0
        layer_logs[name] = {
            "kept": int(layer_kept),
            "total": int(layer_total),
            "sparsity": float(layer_sparsity),
            "u_absmax": float(u_stats["absmax"]),
            "z_absmax": float(z_absmax),
            "objective_aware": bool(objective_aware and fisher is not None),
            "target_sparsity": float(target_sparsity),
        }

        print(
            f"    [{idx}/{len(active_names)}] {name}: kept={layer_kept:,}/{layer_total:,} "
            f"sparsity={100.0 * layer_sparsity:.2f}% target={100.0 * target_sparsity:.2f}% "
            f"u_absmax={u_stats['absmax']:.4e} "
            f"proj={'fisher' if fisher is not None else 'magnitude'}"
        )

        del x, u_old, v, z_new, u_new, fisher
        gc.collect()
        if projection_device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    actual_sparsity = 1.0 - kept / max(1, total)
    print(f"  projection done: sparsity={100.0 * actual_sparsity:.2f}% elapsed={fmt_time(elapsed)}")
    return {
        "kept": int(kept),
        "total": int(total),
        "sparsity": float(actual_sparsity),
        "seconds": float(elapsed),
        "layers": layer_logs,
    }


# ============================================================
# Window planning
# ============================================================


def make_block_windows(
    decoder_layers: nn.ModuleList,
    layers_prefix: str,
    selected_names: Sequence[str],
    active_block_window: int,
) -> List[List[str]]:
    by_block: Dict[int, List[str]] = {i: [] for i in range(len(decoder_layers))}
    nonblock: List[str] = []

    for name in selected_names:
        idx = block_index_for_layer_name(name, layers_prefix)
        if idx is None or idx not in by_block:
            nonblock.append(name)
        else:
            by_block[idx].append(name)

    windows: List[List[str]] = []
    if active_block_window <= 0:
        windows.append(list(selected_names))
    else:
        for start in range(0, len(decoder_layers), active_block_window):
            active: List[str] = []
            for b in range(start, min(start + active_block_window, len(decoder_layers))):
                active.extend(by_block[b])
            if active:
                windows.append(active)
        if nonblock:
            windows.append(nonblock)

    return windows



# ============================================================
# Lightweight final sparse probes
# ============================================================

@torch.no_grad()
def quick_calibration_loss(
    model: nn.Module,
    tokens: torch.Tensor,
    batch_size: int,
    max_seq_len: int,
    batches: int,
    seed: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    crop_mode: str = "random",
) -> Dict[str, float]:
    """Small deterministic probe on calibration tokens after final z is applied."""
    if batches <= 0:
        return {"probe_batches": 0.0, "probe_loss": float("nan"), "probe_ppl": float("nan")}
    loader = InfiniteTokenLoader(
        tokens=tokens,
        batch_size=batch_size,
        shuffle=True,
        seed=seed + 99991,
        max_seq_len=max_seq_len,
        crop_mode=crop_mode,
        drop_last=False,
    )
    model_was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tok = 0
    for _ in range(int(batches)):
        batch = loader.next().to(device, non_blocking=True)
        if batch.size(1) < 2:
            continue
        loss = compute_lm_loss(model, batch, amp_dtype=amp_dtype, device=device, autocast_enabled=True)
        ntok = int(batch.numel() - batch.size(0))
        total_loss += float(loss.detach().cpu().item()) * ntok
        total_tok += ntok
        del batch, loss
    if model_was_training:
        model.train()
    mean = total_loss / max(1, total_tok)
    return {
        "probe_batches": float(batches),
        "probe_tokens": float(total_tok),
        "probe_loss": float(mean),
        "probe_ppl": float(math.exp(min(20.0, mean))),
    }


# ============================================================
# Finalization and saving
# ============================================================


@torch.no_grad()
def apply_z_to_model_and_count(
    model: nn.Module,
    selected_names: Sequence[str],
    store: ADMMStateStore,
    device: torch.device,
) -> Dict[str, Any]:
    print("\nApplying final sparse z weights to model...")
    total = 0
    nonzero = 0
    layer_meta: Dict[str, Any] = {}
    for idx, name in enumerate(selected_names, start=1):
        mod = get_module_by_name(model, name)
        z = store.load_tensor(name, "z", device=device, dtype=torch.float32)
        z = finite_or_zero_(z, nan=0.0, posinf=0.0, neginf=0.0)
        mod.weight.data.copy_(z.to(device=mod.weight.device, dtype=mod.weight.dtype))
        nz = int((z != 0).sum().item())
        n = int(z.numel())
        total += n
        nonzero += nz
        sp = 1.0 - nz / max(1, n)
        layer_meta[name] = {"shape": list(z.shape), "nonzero": nz, "total": n, "sparsity": sp}
        print(f"  [{idx}/{len(selected_names)}] {name}: nonzero={nz:,}/{n:,} sparsity={100.0 * sp:.2f}%")
        del z
        clean_cuda()
    actual = 1.0 - nonzero / max(1, total)
    print(f"Final selected-layer sparsity: {100.0 * actual:.2f}% ({total - nonzero:,}/{total:,} zero)")
    return {"total": total, "nonzero": nonzero, "sparsity": actual, "layers": layer_meta}


def save_outputs(
    model: nn.Module,
    tokenizer: Any,
    out: str,
    save_mode: str,
    meta: Dict[str, Any],
    max_shard_size: str,
) -> None:
    out_path = Path(out)
    save_mode = save_mode.lower().strip()

    if save_mode == "hf_pretrained":
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving Hugging Face model to: {out_path}")
        model.save_pretrained(str(out_path), max_shard_size=max_shard_size, safe_serialization=False)
        if tokenizer is not None:
            tokenizer.save_pretrained(str(out_path))
        with open(out_path / "elsa_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        return

    if save_mode == "full_pt":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving full PyTorch checkpoint to: {out_path}")
        ckpt = {
            "format": "hf_windowed_elsa_admm",
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "meta": meta,
        }
        torch.save(ckpt, out_path)
        return

    if save_mode == "meta_only":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving metadata only to: {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        return

    raise ValueError("save_mode must be hf_pretrained, full_pt, or meta_only")


# ============================================================
# Main training function
# ============================================================


def run(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    script_t0 = time.time()

    if args.sparsity < 0.0 or args.sparsity >= 1.0:
        raise ValueError("--sparsity must be in [0, 1).")


    if getattr(args, "hard_sparse_forward", False):
        # Practical single-GPU mode: keep the model forward path close to the final saved sparse model.
        # This is less pure ADMM than persistent dense x, but avoids the failure mode where LM loss is
        # optimized on dense-ish x while final evaluation uses sparse z.
        args.copy_z_to_x_after_projection = True
        args.no_persist_x_state = True
        args.load_z_into_x_if_missing = True
        args.apply_initial_z_to_model = True
        if args.penalty_normalization == "mean":
            args.penalty_normalization = "sum"

    if args.model_dtype == "bfloat16" and torch.cuda.is_available():
        # Consumer Ampere can be inconsistent with BF16; do not block, only warn.
        print("[warn] BF16 on RTX 3090 may be unsupported or slower. float16 is usually safer on 3090.")

    torch.set_float32_matmul_precision(args.matmul_precision)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    main_device = torch.device(args.device)
    model_dtype = parse_dtype(args.model_dtype)
    active_state_dtype = parse_dtype(args.active_state_dtype)
    projection_device = torch.device(args.projection_device)

    print("=" * 100)
    print("Windowed ELSA / ELSA-L ADMM Sparsification")
    print("=" * 100)
    print(f"Model id             : {args.model_id}")
    print(f"Device               : {main_device}")
    print(f"Model dtype          : {model_dtype}")
    print(f"Active state dtype   : {active_state_dtype}")
    print(f"ADMM z/u format      : {args.state_format}")
    print(f"Persistent x format  : {args.x_state_format if not args.no_persist_x_state else 'disabled'}")
    print(f"Adam/Fisher format   : {args.optimizer_state_format if not args.no_persist_optimizer_state else 'disabled'}")
    print(f"Projection device    : {projection_device}")
    print(f"Target sparsity      : {100.0 * args.sparsity:.2f}%")
    print(f"Pattern              : {args.pattern}")
    print(f"CUDA memory          : {cuda_mem()}")

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.to(main_device)

    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # Keep dropout disabled if present. Mistral has essentially no dropout; train() is not needed for loss.
    if args.model_train_mode:
        model.train()
    else:
        model.eval()

    print(f"Model loaded. CUDA memory: {cuda_mem()}")

    print("\nLoading calibration tokens...")
    calib_tokens = load_calibration_tokens(args.calib)
    print(f"Loaded calibration tokens: {tuple(calib_tokens.shape)}")
    # Do NOT truncate here. InfiniteTokenLoader performs random/prefix/sliding crops.
    # Keeping the full [N, 2048] tensor is important; otherwise --crop_mode random
    # degenerates into repeatedly using the first --max_seq_len tokens.
    if args.max_seq_len > 0 and calib_tokens.size(1) > args.max_seq_len:
        if str(args.crop_mode).lower().strip() in ("random", "sliding", "prefix"):
            print(
                f"Keeping full calibration length {calib_tokens.size(1)}; "
                f"loader will use --crop_mode={args.crop_mode} with --max_seq_len={args.max_seq_len}."
            )
        else:
            print(
                f"Using full calibration length {calib_tokens.size(1)} because --crop_mode=none; "
                f"--max_seq_len={args.max_seq_len} will not crop."
            )
    if calib_tokens.size(1) < 2:
        raise ValueError("Need sequence length >= 2 for next-token prediction loss.")

    suffixes = parse_suffixes(args.suffixes)
    selected_names = find_selected_linear_names(
        model=model,
        include=args.include,
        exclude=args.exclude,
        suffixes=suffixes,
        compress_lm_head=bool(args.compress_lm_head),
        skip_tied_lm_head=bool(args.skip_tied_lm_head),
        skip_attn_out=bool(args.skip_attn_out),
        skip_mlp_out=bool(args.skip_mlp_out),
    )
    if not selected_names:
        raise RuntimeError("No nn.Linear layers selected for sparsification.")

    layers_prefix, decoder_layers = find_decoder_layers(model)
    print(f"\nDecoder layers found: {layers_prefix}, count={len(decoder_layers)}")
    print(f"Selected linears: {len(selected_names)}")
    total_selected = 0
    for name in selected_names:
        mod = get_module_by_name(model, name)
        n = int(mod.weight.numel())
        total_selected += n
        print(f"  - {name}: shape={tuple(mod.weight.shape)} weights={n:,}")
    print(f"Total selected weights: {total_selected:,}")

    windows = make_block_windows(
        decoder_layers=decoder_layers,
        layers_prefix=layers_prefix,
        selected_names=selected_names,
        active_block_window=int(args.active_block_window),
    )
    print(f"\nActive windows: {len(windows)}")
    for i, w in enumerate(windows):
        n = sum(int(get_module_by_name(model, name).weight.numel()) for name in w)
        print(f"  window {i:02d}: layers={len(w)} weights={n:,}")

    if args.active_block_window <= 0:
        print("\n[warn] --active_block_window 0 trains all selected layers at once. This is usually too large for RTX 3090.")

    selected_numels = {name: int(get_module_by_name(model, name).weight.numel()) for name in selected_names}
    layer_sparsity_map: Optional[Dict[str, float]] = None
    dynamic_profile_history: List[Dict[str, Any]] = []
    if str(args.dynamic_sparsity).lower().strip() in ("module_protect", "boundary_protect"):
        layer_sparsity_map, static_meta = build_static_dynamic_sparsity_map(
            model=model,
            selected_names=selected_names,
            mode=str(args.dynamic_sparsity),
            target_sparsity=float(args.sparsity),
            protect_sparsity=float(args.dynamic_protect_sparsity),
            protect_suffixes=str(args.dynamic_protect_suffixes),
            boundary_blocks=int(args.dynamic_boundary_blocks),
            layers_prefix=layers_prefix,
            num_decoder_layers=int(len(decoder_layers)),
        )
        dynamic_profile_history.append(static_meta)
        summarize_sparsity_map("Initial static dynamic sparsity map", layer_sparsity_map, selected_numels)
    elif str(args.dynamic_sparsity).lower().strip() == "fisher_damage":
        print("\nDynamic ELSA/Fisher sparsity is enabled. Pass 1 starts uniform; profile will be computed after the requested warmup pass.")

    state_dir = Path(args.state_dir) if args.state_dir else Path(args.out).with_suffix("") / "elsa_admm_state"
    if args.reset_state and state_dir.exists():
        print(f"\nDeleting old state dir: {state_dir}")
        shutil.rmtree(state_dir)
    store = ADMMStateStore(root=state_dir, state_format=args.state_format)
    x_store = None
    if not bool(args.no_persist_x_state):
        x_store = ADMMStateStore(root=state_dir / "x_state", state_format=args.x_state_format)
    opt_store = None
    if not bool(args.no_persist_optimizer_state):
        opt_store = ADMMStateStore(root=state_dir / "optimizer_state", state_format=args.optimizer_state_format)

    init_meta = initialize_admm_states(
        model=model,
        selected_names=selected_names,
        store=store,
        sparsity=float(args.sparsity),
        pattern=args.pattern,
        min_keep_per_tensor=int(args.min_keep_per_tensor),
        overwrite=bool(args.reset_state),
        x_store=x_store,
        sparsity_map=layer_sparsity_map,
    )

    initial_sparse_meta = None
    initial_probe = None
    if bool(args.apply_initial_z_to_model):
        print("\nApplying initial sparse-forward z weights to the full selected model before training...")
        initial_sparse_meta = apply_z_to_model_and_count(
            model=model,
            selected_names=selected_names,
            store=store,
            device=main_device,
        )
        if int(args.initial_probe_batches) > 0:
            initial_probe = quick_calibration_loss(
                model=model,
                tokens=calib_tokens,
                batch_size=int(args.batch_size),
                max_seq_len=int(args.max_seq_len),
                batches=int(args.initial_probe_batches),
                seed=int(args.seed) + 17,
                device=main_device,
                amp_dtype=model_dtype,
                crop_mode=str(args.crop_mode),
            )
            print(
                f"Initial sparse calibration probe: loss={initial_probe['probe_loss']:.6f} "
                f"ppl~{initial_probe['probe_ppl']:.3f} tokens={initial_probe['probe_tokens']:.0f}"
            )

    token_loader = InfiniteTokenLoader(
        tokens=calib_tokens,
        batch_size=int(args.batch_size),
        shuffle=not bool(args.no_shuffle),
        seed=int(args.seed),
        max_seq_len=int(args.max_seq_len),
        crop_mode=str(args.crop_mode),
        drop_last=False,
    )

    set_all_requires_grad(model, False)

    total_steps_planned = int(args.admm_passes) * len(windows) * int(args.steps_per_window)
    print("\nTraining plan:")
    print(f"  passes            : {args.admm_passes}")
    print(f"  steps/window      : {args.steps_per_window}")
    print(f"  total planned     : {total_steps_planned}")
    print(f"  projection interval: {args.projection_interval}")
    print(f"  base LR           : {args.lr}")
    print(f"  final lambda      : {args.admm_lambda}")
    print(f"  lambda schedule   : {args.lambda_schedule}")
    print(f"  penalty norm      : {args.penalty_normalization}")
    print(f"  hard sparse fwd   : {bool(args.hard_sparse_forward)}")
    print(f"  dynamic sparsity  : {args.dynamic_sparsity}")
    print(f"  crop mode         : {args.crop_mode}")
    print(f"  grad accum steps  : {args.grad_accum_steps}")
    print(f"  FP32 master x     : {not bool(args.no_master_weights)}")
    print(f"  CUDA memory       : {cuda_mem()}")

    train_log: List[Dict[str, Any]] = []
    projection_log: List[Dict[str, Any]] = []
    global_step = 0

    try:
        for pass_idx in range(int(args.admm_passes)):
            print("\n" + "=" * 100)
            print(f"ADMM PASS {pass_idx + 1}/{args.admm_passes}")
            print("=" * 100)

            for win_idx, active_names in enumerate(windows):
                print("\n" + "-" * 100)
                print(f"WINDOW {win_idx + 1}/{len(windows)} | active layers={len(active_names)}")
                for name in active_names:
                    print(f"  active: {name}")
                print(f"CUDA before window: {cuda_mem()}")

                x_load_stats = load_active_x_states_into_model(
                    model=model,
                    active_names=active_names,
                    x_store=x_store,
                    admm_store=store,
                    device=main_device,
                    fallback_to_z=bool(args.load_z_into_x_if_missing),
                )
                if x_load_stats["loaded_x"] or x_load_stats["loaded_z_fallback"]:
                    print(
                        f"  restored x states: x={x_load_stats['loaded_x']} "
                        f"z_fallback={x_load_stats['loaded_z_fallback']} missing={x_load_stats['missing_x']}"
                    )

                active_params = set_active_trainable_layers(model, active_names)
                named_active_params = [(name, get_module_by_name(model, name).weight) for name in active_names]

                active_state = load_active_states(
                    store=store,
                    active_names=active_names,
                    device=main_device,
                    active_state_dtype=active_state_dtype,
                )

                optimizer = SafeAdamWFP32(
                    named_params=named_active_params,
                    betas=(float(args.beta1), float(args.beta2)),
                    eps=float(args.adam_eps),
                    state_device=main_device,
                    max_state_abs=float(args.optimizer_state_clip),
                    use_master_weights=not bool(args.no_master_weights),
                )
                opt_load_stats = optimizer.load_state_from_store(opt_store)
                if opt_load_stats["loaded_v"] > 0:
                    print(
                        f"  restored Adam/Fisher states: m={opt_load_stats['loaded_m']:.0f} "
                        f"v={opt_load_stats['loaded_v']:.0f} steps={opt_load_stats['loaded_steps']:.0f} "
                        f"start_step={opt_load_stats['start_step']:.0f}"
                    )

                clean_cuda()
                print(f"CUDA after optimizer/state load: {cuda_mem()}")

                last_projection_step = -1
                for local_step in range(int(args.steps_per_window)):
                    progress = global_step / max(1, total_steps_planned - 1)
                    lr_t = lr_schedule_value(args.lr_schedule, float(args.lr), progress, float(args.min_lr_ratio))
                    lambda_t = lambda_schedule_value(
                        args.lambda_schedule,
                        float(args.admm_lambda),
                        progress,
                        float(args.lambda_warmup_frac),
                    )

                    optimizer.zero_grad(set_to_none=True)

                    step_t0 = time.time()
                    oom = False
                    try:
                        accum_steps = max(1, int(args.grad_accum_steps))
                        lm_loss_values: List[float] = []
                        bad_loss = False

                        # Average LM gradient over multiple random crops. This keeps memory
                        # the same as batch_size=1 but improves gradient/Fisher estimates.
                        for accum_idx in range(accum_steps):
                            batch = token_loader.next().to(main_device, non_blocking=True)
                            lm_loss_i = compute_lm_loss(
                                model=model,
                                input_ids=batch,
                                amp_dtype=model_dtype,
                                device=main_device,
                                autocast_enabled=not bool(args.no_autocast),
                            )
                            if not torch.isfinite(lm_loss_i).all():
                                print(
                                    f"\n[warn] non-finite LM loss at global_step={global_step}, "
                                    f"accum={accum_idx + 1}/{accum_steps}; skipping step."
                                )
                                bad_loss = True
                                del batch, lm_loss_i
                                break

                            lm_loss_values.append(float(lm_loss_i.detach().cpu().item()))
                            (lm_loss_i / float(accum_steps)).backward()
                            del batch, lm_loss_i

                        if bad_loss or not lm_loss_values:
                            optimizer.zero_grad(set_to_none=True)
                            clean_cuda()
                            global_step += 1
                            continue

                        penalty, pen_stats = compute_admm_penalty(
                            model=model,
                            active_names=active_names,
                            active_state=active_state,
                            lambda_value=lambda_t,
                            normalization=args.penalty_normalization,
                            diff_clip=float(args.admm_diff_clip),
                        )

                        if not torch.isfinite(penalty).all():
                            print(
                                f"\n[warn] non-finite ADMM penalty at global_step={global_step}; skipping step."
                            )
                            optimizer.zero_grad(set_to_none=True)
                            clean_cuda()
                            global_step += 1
                            continue

                        # Add the ADMM constraint once per optimizer step, not once per
                        # accumulated microbatch. If lambda is zero during warmup, the
                        # penalty is a constant zero tensor without a gradient graph.
                        if penalty.requires_grad:
                            penalty.backward()

                        lm_loss_scalar = float(sum(lm_loss_values) / max(1, len(lm_loss_values)))
                        penalty_scalar = float(penalty.detach().cpu().item())
                        total_loss_scalar = lm_loss_scalar + penalty_scalar
                        lm_loss = torch.tensor(lm_loss_scalar, device=main_device, dtype=torch.float32)
                        loss = torch.tensor(total_loss_scalar, device=main_device, dtype=torch.float32)
                    except torch.cuda.OutOfMemoryError:
                        oom = True
                        print(f"\n[OOM] step {global_step}: clearing cache and skipping. CUDA: {cuda_mem()}")
                        optimizer.zero_grad(set_to_none=True)
                        clean_cuda()
                    except RuntimeError as exc:
                        msg = str(exc).lower()
                        if "out of memory" in msg:
                            oom = True
                            print(f"\n[OOM] step {global_step}: {exc}; clearing cache and skipping. CUDA: {cuda_mem()}")
                            optimizer.zero_grad(set_to_none=True)
                            clean_cuda()
                        else:
                            raise

                    if oom:
                        global_step += 1
                        continue

                    grad_stats = sanitize_and_clip_gradients(
                        named_params=named_active_params,
                        max_grad_norm=float(args.max_grad_norm),
                        grad_value_clip=float(args.grad_value_clip),
                    )

                    if args.skip_step_on_nonfinite_grad and grad_stats["nonfinite_grad_values"] > 0:
                        print(
                            f"\n[warn] nonfinite gradients at step {global_step}; "
                            f"count={grad_stats['nonfinite_grad_values']:.0f}; skipping optimizer step."
                        )
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        continue

                    opt_stats = optimizer.step(
                        lr=lr_t,
                        weight_decay=float(args.weight_decay),
                        update_clip=float(args.update_value_clip),
                        weight_clip=float(args.weight_clip),
                    )
                    optimizer.zero_grad(set_to_none=True)

                    elapsed = time.time() - step_t0
                    log_item = {
                        "global_step": int(global_step),
                        "pass": int(pass_idx),
                        "window": int(win_idx),
                        "local_step": int(local_step),
                        "lr": float(lr_t),
                        "lambda": float(lambda_t),
                        "lm_loss": float(lm_loss.detach().cpu().item()),
                        "admm_penalty": float(penalty.detach().cpu().item()),
                        "total_loss": float(loss.detach().cpu().item()),
                        "elapsed": float(elapsed),
                        **pen_stats,
                        **grad_stats,
                        **opt_stats,
                    }
                    train_log.append(log_item)

                    if (global_step % int(args.log_interval) == 0) or local_step == 0:
                        ppl_est = math.exp(min(20.0, log_item["lm_loss"]))
                        print(
                            f"step={global_step:06d} pass={pass_idx + 1}/{args.admm_passes} "
                            f"win={win_idx + 1}/{len(windows)} local={local_step + 1}/{args.steps_per_window} "
                            f"lm={log_item['lm_loss']:.5f} ppl~{ppl_est:.2f} "
                            f"pen={log_item['admm_penalty']:.5e} lambda={lambda_t:.3e} lr={lr_t:.3e} "
                            f"gnorm={log_item['grad_norm']:.3e} admm_rms={log_item['admm_rms']:.3e} "
                            f"upd_rms={log_item['update_rms']:.3e} time={elapsed:.2f}s "
                            f"cuda=[{cuda_mem()}]"
                        )

                    do_projection = False
                    if int(args.projection_interval) > 0:
                        if (local_step + 1) % int(args.projection_interval) == 0:
                            do_projection = True
                    if do_projection:
                        proj_meta = admm_project_active_layers(
                            model=model,
                            active_names=active_names,
                            optimizer=optimizer,
                            store=store,
                            active_state=active_state,
                            sparsity=float(args.sparsity),
                            pattern=args.pattern,
                            objective_aware=not bool(args.no_objective_aware_projection),
                            projection_device=projection_device,
                            fisher_floor=float(args.fisher_floor),
                            min_keep_per_tensor=int(args.min_keep_per_tensor),
                            dual_clip=float(args.dual_clip),
                            dual_lr=float(args.dual_lr),
                            copy_z_to_x=bool(args.copy_z_to_x_after_projection),
                            fisher_warmup_steps=int(args.fisher_warmup_steps),
                            sparsity_map=layer_sparsity_map,
                        )
                        proj_meta.update({
                            "global_step": int(global_step),
                            "pass": int(pass_idx),
                            "window": int(win_idx),
                            "local_step": int(local_step),
                        })
                        projection_log.append(proj_meta)
                        last_projection_step = local_step
                        clean_cuda()

                    global_step += 1

                if args.project_at_window_end and last_projection_step != int(args.steps_per_window) - 1:
                    proj_meta = admm_project_active_layers(
                        model=model,
                        active_names=active_names,
                        optimizer=optimizer,
                        store=store,
                        active_state=active_state,
                        sparsity=float(args.sparsity),
                        pattern=args.pattern,
                        objective_aware=not bool(args.no_objective_aware_projection),
                        projection_device=projection_device,
                        fisher_floor=float(args.fisher_floor),
                        min_keep_per_tensor=int(args.min_keep_per_tensor),
                        dual_clip=float(args.dual_clip),
                        dual_lr=float(args.dual_lr),
                        copy_z_to_x=bool(args.copy_z_to_x_after_projection),
                        fisher_warmup_steps=int(args.fisher_warmup_steps),
                        sparsity_map=layer_sparsity_map,
                    )
                    proj_meta.update({
                        "global_step": int(global_step),
                        "pass": int(pass_idx),
                        "window": int(win_idx),
                        "local_step": int(args.steps_per_window),
                        "window_end": True,
                    })
                    projection_log.append(proj_meta)

                # Persist active x/z/u and optimizer/Fisher states before releasing window memory.
                save_active_states(store, active_state)
                save_active_x_states_from_model(model, active_names, x_store)
                opt_save_stats = optimizer.save_state_to_store(opt_store)
                if opt_save_stats["saved_v"] > 0:
                    print(
                        f"  saved Adam/Fisher states: m={opt_save_stats['saved_m']:.0f} "
                        f"v={opt_save_stats['saved_v']:.0f} steps={opt_save_stats['saved_steps']:.0f}"
                    )
                optimizer.release()
                del optimizer, active_state, active_params, named_active_params
                set_all_requires_grad(model, False)
                clean_cuda()

                # Save logs after each window so progress is not lost on long runs.
                store.save_json("train_log.json", {"items": train_log})
                store.save_json("projection_log.json", {"items": projection_log})
                print(f"Finished window {win_idx + 1}. CUDA: {cuda_mem()}")

            if int(args.pass_probe_batches) > 0:
                pass_probe = quick_calibration_loss(
                    model=model,
                    tokens=calib_tokens,
                    batch_size=int(args.batch_size),
                    max_seq_len=int(args.max_seq_len),
                    batches=int(args.pass_probe_batches),
                    seed=int(args.seed) + 1000 + int(pass_idx),
                    device=main_device,
                    amp_dtype=model_dtype,
                    crop_mode=str(args.crop_mode),
                )
                print(
                    f"Pass {pass_idx + 1} sparse calibration probe: "
                    f"loss={pass_probe['probe_loss']:.6f} ppl~{pass_probe['probe_ppl']:.3f} "
                    f"tokens={pass_probe['probe_tokens']:.0f}"
                )
                store.save_json(f"pass_{pass_idx + 1}_probe.json", pass_probe)

            # Dynamic Fisher-damage sparsity profile: compute after a completed pass,
            # then use the resulting per-layer sparsities for later projections.
            if (
                str(args.dynamic_sparsity).lower().strip() == "fisher_damage"
                and (pass_idx + 1) >= int(args.dynamic_start_pass)
                and (pass_idx + 1) < int(args.admm_passes)
                and (layer_sparsity_map is None or bool(args.dynamic_update_each_pass))
            ):
                metric_s = float(args.dynamic_metric_sparsity) if float(args.dynamic_metric_sparsity) >= 0.0 else float(args.sparsity)
                layer_sparsity_map, dyn_meta = compute_fisher_damage_sparsity_map(
                    model=model,
                    selected_names=selected_names,
                    store=store,
                    opt_store=opt_store,
                    target_sparsity=float(args.sparsity),
                    base_metric_sparsity=metric_s,
                    sparsity_min=float(args.dynamic_sparsity_min),
                    sparsity_max=float(args.dynamic_sparsity_max),
                    alpha=float(args.dynamic_alpha),
                    eps=float(args.dynamic_eps),
                    fisher_floor=float(args.fisher_floor),
                    projection_device=projection_device,
                )
                dyn_meta["computed_after_pass"] = int(pass_idx + 1)
                dynamic_profile_history.append(dyn_meta)
                store.save_json(f"dynamic_sparsity_profile_after_pass_{pass_idx + 1}.json", dyn_meta)
                store.save_json("dynamic_sparsity_latest.json", dyn_meta)
                clean_cuda()

    except KeyboardInterrupt:
        print("\n[interrupt] Training interrupted by user. Applying current z states and saving metadata.")

    final_sparse_meta = apply_z_to_model_and_count(
        model=model,
        selected_names=selected_names,
        store=store,
        device=main_device,
    )

    final_probe = quick_calibration_loss(
        model=model,
        tokens=calib_tokens,
        batch_size=int(args.batch_size),
        max_seq_len=int(args.max_seq_len),
        batches=int(args.final_probe_batches),
        seed=int(args.seed),
        device=main_device,
        amp_dtype=model_dtype,
        crop_mode=str(args.crop_mode),
    )
    if int(args.final_probe_batches) > 0:
        print(
            f"Final sparse calibration probe: loss={final_probe['probe_loss']:.6f} "
            f"ppl~{final_probe['probe_ppl']:.3f} tokens={final_probe['probe_tokens']:.0f}"
        )

    final_meta: Dict[str, Any] = {
        "format": "hf_windowed_elsa_admm",
        "method": "windowed_elsa_l_admm_single_gpu",
        "important_note": (
            "This is an ELSA/ELSA-L style implementation adapted for single-GPU memory limits. "
            "With active_block_window > 0, only a window of decoder blocks is optimized at a time. "
            "Use active_block_window=0 only on hardware capable of full selected-layer training."
        ),
        "model_id": str(args.model_id),
        "args": vars(args),
        "selected_layer_names": list(selected_names),
        "suffixes": list(suffixes),
        "layers_prefix": str(layers_prefix),
        "num_decoder_layers": int(len(decoder_layers)),
        "num_windows": int(len(windows)),
        "init_meta": init_meta,
        "initial_sparse_meta": initial_sparse_meta,
        "initial_sparse_probe": initial_probe,
        "final_sparse_meta": final_sparse_meta,
        "final_sparse_probe": final_probe,
        "dynamic_sparsity_mode": str(args.dynamic_sparsity),
        "dynamic_sparsity_profile_history_tail": dynamic_profile_history[-3:],
        "final_layer_sparsity_map": layer_sparsity_map,
        "train_log_tail": train_log[-50:],
        "projection_log_tail": projection_log[-20:],
        "state_dir": str(state_dir),
        "x_state_dir": str(x_store.root) if x_store is not None else None,
        "optimizer_state_dir": str(opt_store.root) if opt_store is not None else None,
        "script_seconds": float(time.time() - script_t0),
        "cuda_memory_end": cuda_mem(),
    }
    store.save_json("final_meta.json", final_meta)

    save_outputs(
        model=model,
        tokenizer=tokenizer,
        out=args.out,
        save_mode=args.save_mode,
        meta=final_meta,
        max_shard_size=args.max_shard_size,
    )

    print("\nDone.")
    print(f"Total time: {fmt_time(time.time() - script_t0)}")
    print(f"Final selected-layer sparsity: {100.0 * final_sparse_meta['sparsity']:.2f}%")
    print(f"CUDA memory: {cuda_mem()}")


# ============================================================
# CLI
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Windowed ELSA / ELSA-L ADMM sparsification for HF Mistral/LLaMA/Qwen models."
    )

    # Core I/O, similar style to your SparseGPT script.
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--state_dir", type=str, default="")
    parser.add_argument("--reset_state", action="store_true")
    parser.add_argument("--save_mode", type=str, default="hf_pretrained", choices=["hf_pretrained", "full_pt", "meta_only"])
    parser.add_argument("--max_shard_size", type=str, default="2GB")

    # Sparsity.
    parser.add_argument("--sparsity", type=float, default=0.70)
    parser.add_argument("--pattern", type=str, default="unstructured")
    parser.add_argument("--min_keep_per_tensor", type=int, default=1)

    # Dynamic non-uniform sparsity allocation.
    parser.add_argument("--dynamic_sparsity", type=str, default="none", choices=["none", "fisher_damage", "module_protect", "boundary_protect"],
                        help="Use non-uniform per-tensor sparsity while preserving requested global sparsity.")
    parser.add_argument("--dynamic_start_pass", type=int, default=1,
                        help="For fisher_damage, compute first dynamic profile after this completed pass.")
    parser.add_argument("--dynamic_update_each_pass", action="store_true",
                        help="Recompute fisher_damage profile after every pass from dynamic_start_pass onward.")
    parser.add_argument("--dynamic_alpha", type=float, default=0.65)
    parser.add_argument("--dynamic_eps", type=float, default=1.0e-12)
    parser.add_argument("--dynamic_sparsity_min", type=float, default=0.70)
    parser.add_argument("--dynamic_sparsity_max", type=float, default=0.88)
    parser.add_argument("--dynamic_metric_sparsity", type=float, default=-1.0,
                        help="Base sparsity used to measure Fisher pruning damage. <0 means use --sparsity.")
    parser.add_argument("--dynamic_protect_sparsity", type=float, default=0.75)
    parser.add_argument("--dynamic_protect_suffixes", type=str, default="v_proj,o_proj,down_proj")
    parser.add_argument("--dynamic_boundary_blocks", type=int, default=2)

    # Selection, similar to your script.
    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")
    parser.add_argument(
        "--suffixes",
        type=str,
        default="",
        help="Comma-separated Linear suffixes. Default: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--compress_lm_head", action="store_true")
    parser.add_argument("--skip_tied_lm_head", action="store_true")
    parser.add_argument("--skip_attn_out", action="store_true")
    parser.add_argument("--skip_mlp_out", action="store_true")

    # Training plan.
    parser.add_argument("--admm_passes", type=int, default=2)
    parser.add_argument("--steps_per_window", type=int, default=16)
    parser.add_argument(
        "--active_block_window",
        type=int,
        default=1,
        help="Number of decoder blocks trained at once. 1 is safest for RTX 3090. 0 means all selected layers at once.",
    )
    parser.add_argument("--projection_interval", type=int, default=8)
    parser.add_argument("--project_at_window_end", action="store_true")
    parser.set_defaults(project_at_window_end=True)

    # Optimizer and schedules.
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["constant", "linear", "cosine"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1.0e-8)

    # ADMM.
    parser.add_argument("--admm_lambda", type=float, default=1.0e-3)
    parser.add_argument("--lambda_schedule", type=str, default="cosine", choices=["constant", "linear", "cosine"])
    parser.add_argument("--lambda_warmup_frac", type=float, default=0.05)
    parser.add_argument("--penalty_normalization", type=str, default="sum", choices=["mean", "sum"])
    parser.add_argument("--dual_lr", type=float, default=1.0)
    parser.add_argument("--dual_clip", type=float, default=10.0)
    parser.add_argument("--admm_diff_clip", type=float, default=10.0)
    parser.add_argument("--no_objective_aware_projection", action="store_true")
    parser.add_argument("--fisher_floor", type=float, default=1.0e-12)
    parser.add_argument(
        "--copy_z_to_x_after_projection",
        action="store_true",
        help="Stability mode: copy sparse z into trainable x after each projection. Not default ELSA; can help if FP16 explodes.",
    )
    parser.add_argument(
        "--hard_sparse_forward",
        action="store_true",
        help=(
            "Recommended single-GPU mode: after every projection copy z->x, disable persistent dense x, "
            "and use sum-normalized ADMM unless explicitly changed. This avoids training dense x but saving sparse z."
        ),
    )
    parser.add_argument(
        "--fisher_warmup_steps",
        type=int,
        default=8,
        help="Use magnitude projection until the active Adam/Fisher state has at least this many local steps.",
    )

    # Numerical safety.
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--grad_value_clip", type=float, default=1.0)
    parser.add_argument("--update_value_clip", type=float, default=1.0)
    parser.add_argument("--weight_clip", type=float, default=0.0)
    parser.add_argument("--optimizer_state_clip", type=float, default=1.0e6)
    parser.add_argument("--skip_step_on_nonfinite_grad", action="store_true")
    parser.set_defaults(skip_step_on_nonfinite_grad=False)
    parser.add_argument(
        "--no_master_weights",
        action="store_true",
        help="Disable FP32 active-window master weights. Saves VRAM, but low-LR refinement is less numerically accurate.",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Accumulate LM gradients over this many random crops before one optimizer/projection step. Improves gradient/Fisher quality without increasing VRAM.",
    )

    # Data.
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--crop_mode", type=str, default="random", choices=["random", "prefix", "sliding", "none"], help="How to crop long calibration sequences when max_seq_len is shorter than calibration length.")
    parser.add_argument("--apply_initial_z_to_model", action="store_true", help="Apply initial sparse z to all selected layers before training. Enabled automatically by --hard_sparse_forward.")
    parser.add_argument("--initial_probe_batches", type=int, default=0, help="Evaluate a tiny calibration loss immediately after initial z is applied; 0 disables.")
    parser.add_argument("--pass_probe_batches", type=int, default=0, help="Evaluate a tiny calibration loss after each full ADMM pass; 0 disables.")
    parser.add_argument("--final_probe_batches", type=int, default=4, help="Evaluate a tiny calibration loss after final z is applied; 0 disables.")
    parser.add_argument("--no_shuffle", action="store_true")

    # Device/dtype/loading.
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--active_state_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--state_format", type=str, default="fp16", choices=["int8", "fp16", "bf16", "fp32"],
                        help="Disk format for ADMM z/u states. int8 is memory/disk efficient; fp16 is usually better quality.")
    parser.add_argument("--x_state_format", type=str, default="fp16", choices=["int8", "fp16", "bf16", "fp32"],
                        help="Disk format for persistent ADMM x states. fp16 is recommended for resume quality.")
    parser.add_argument("--optimizer_state_format", type=str, default="fp16", choices=["int8", "fp16", "bf16", "fp32"],
                        help="Disk format for persistent Adam/Fisher m/v states. int8 saves disk; fp16 is more accurate.")
    parser.add_argument("--no_persist_x_state", action="store_true",
                        help="Disable saving/loading the ADMM x variable per layer. Not recommended for closest-to-ELSA sweeps/resume.")
    parser.add_argument("--no_persist_optimizer_state", action="store_true",
                        help="Disable saving/loading Adam m/v states across block revisits. Faster/less disk, but less ELSA-faithful.")
    parser.add_argument("--load_z_into_x_if_missing", action="store_true",
                        help="If persistent x is missing, initialize active x from sparse z instead of current model weights. Useful for old state dirs/resume.")
    parser.add_argument("--no_load_z_into_x_if_missing", action="store_false", dest="load_z_into_x_if_missing",
                        help="If persistent x is missing, keep current model weights instead of falling back to sparse z.")
    parser.set_defaults(load_z_into_x_if_missing=True)
    parser.add_argument("--projection_device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--no_autocast", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--model_train_mode", action="store_true", help="Use model.train(); default is eval() to disable dropout.")
    parser.add_argument("--matmul_precision", type=str, default="high", choices=["highest", "high", "medium"])

    # Logging/repro.
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log_interval", type=int, default=1)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
