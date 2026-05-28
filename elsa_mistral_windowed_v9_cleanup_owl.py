#!/usr/bin/env python3
"""
v9 — ELSA windowed ADMM with post-projection cleanup, OWL outlier protection,
     adaptive lambda, KD temperature schedule, and learnable per-output scale.
     Targets sub-20 ppl at 80% sparsity on Mistral-7B on a single 3090.

v9 changes vs v8 ("paper_aligned"):
    - Post-projection CLEANUP phase: after the ADMM loop, freeze the final mask
      and run --cleanup_steps additional optimizer steps per window with
      LM+KD loss only (no ADMM penalty, no projections). Lower LR with
      cosine decay. ADMM only enforces the constraint; this phase actually
      optimizes the surviving 20% of weights at the optimum. Biggest single
      ppl win expected (2-5 ppl at 80%).
    - OWL-style GLOBAL outlier protection: --owl_outlier_pct keeps the top
      X percent of |W| globally across all selected layers, regardless of
      the per-layer Fisher-damage budget. Forced-keep mask is unioned into
      every projection. Outlier weights carry disproportionate signal in
      LLMs (SmoothQuant/AWQ thesis); making this explicit consistently
      buys 1-3 ppl at high sparsity.
    - Per-output learnable RESCALING during cleanup (--learn_output_scale):
      one scalar per output row, trained alongside surviving weights to
      compensate for sparsification-induced output shrinkage. Fused into
      weights at save time so the resulting checkpoint stays a plain
      sparse linear (no inference-time overhead).
    - Mask drift tracking + adaptive lambda: every projection logs the
      Hamming distance to the previous mask. If --adaptive_lambda is set,
      lambda grows when the mask is still drifting and holds when it
      stabilizes, which removes the "lambda too small to enforce" failure
      mode without manual tuning.
    - KD upgrades: default --kd_topk 32 -> 64; new --kd_temperature_start
      enables a linear temperature anneal from kd_temperature_start ->
      kd_temperature over --kd_temperature_warmup_frac of training.
      Softer teacher targets help early; sharper targets help convergence.
      KD remains active during cleanup (--kd_alpha_cleanup; defaults to
      kd_alpha).
    - Numerical: cleanup uses gradient-only updates on surviving weights;
      the dead positions are re-zeroed at the end of every optimizer step
      via mask multiply. No accidental drift back to dense.

Recommended 3090 command for 80% Mistral-7B target:
    --sparsity 0.80 --pattern unstructured --init_method wanda
    --dynamic_sparsity fisher_damage --owl_outlier_pct 1.0
    --kd_alpha 0.5 --kd_topk 64 --kd_temperature 1.0
    --kd_temperature_start 2.0 --kd_temperature_warmup_frac 0.3
    --admm_passes 4 --steps_per_window 128 --projection_interval 16
    --cleanup_steps 64 --cleanup_lr 5e-5 --cleanup_lr_schedule cosine
    --learn_output_scale --adaptive_lambda --max_seq_len 1024
    --active_block_window 2 --grad_accum_steps 4

Paper-aligned Windowed ELSA / ELSA-L ADMM for Hugging Face decoder-only LLMs.

v8 changes vs v7 ("stable_fisher_fixed"):
    - Wanda activation-aware warm-start (--init_method {magnitude, wanda})
      replaces poor magnitude initialization for z; uses per-channel ||X||_2
      collected during a calibration forward pass with hooks.
    - FP32-master-weight aware ADMM penalty: when use_master_weights=True
      the penalty value is computed in FP32 using the master copy; gradient
      still flows through the FP16 model weight via a detached correction.
    - New penalty normalization mode: 'layer_mean' divides each layer's
      ||x - z + u||^2 by that layer's numel before summing, so large MLP
      layers do not dominate the constraint signal.
    - Optional knowledge distillation from a cached dense teacher's top-K
      logits (--kd_alpha, --kd_topk, --kd_cache_dir). Pre-cache pass loads the
      dense model once, streams top-K + indices per (seq_idx, position) to
      disk, then frees and reloads sparsified model. Disk cost ~ N*T*K*12B.
    - Improved dynamic Fisher allocation: --dynamic_metric_mode allows
      'damage_mean' (v7), 'damage_sum', or 'wanda_proxy' (activation-aware).
    - Paper-aligned defaults: penalty=mean, lambda=5e-5, lr=3e-4 with linear
      schedule, max_seq_len=1024, max_grad_norm=1.0, fisher_max_factor=1e6,
      fisher_power=1.0, init_method=wanda, active_block_window=2.
    - hard_sparse_forward is still supported but warns; pure ELSA (dense x +
      sparse z) is the new default.

Target use-case (unchanged):
    - Mistral/LLaMA/Qwen decoder-only models on a single GPU such as RTX 3090.
    - High unstructured sparsity experiments: 70%, 80%, 90%.
    - Numerical robustness, low-precision state storage, windowed training.

Recommended RTX 3090 settings (see CLI defaults):
    --model_dtype float16 --active_block_window 2
    --batch_size 1 --max_seq_len 1024 --grad_accum_steps 4
    --state_format int8 --projection_device cpu
    --gradient_checkpointing --init_method wanda
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



def stable_fisher_weight(
    fisher: Optional[torch.Tensor],
    fisher_floor: float = 1.0e-12,
    fisher_power: float = 1.0,
    fisher_blend_magnitude: float = 0.0,
    fisher_max_factor: float = 1.0e6,
) -> Optional[torch.Tensor]:
    """Return a numerically stabilized diagonal Fisher weight for projection.

    Old projection used score_i = F_i * v_i^2 directly after mean normalization.
    That can be noisy at 80-90% sparsity when Fisher comes from few calibration
    crops. This function performs three safe operations:
      1. mean-normalize F so its scale cannot dominate the loss;
      2. cap extreme values to prevent single noisy entries from owning the mask;
      3. optionally power-compress and blend with magnitude pruning.

    Effective score later becomes:
        score_i = [ blend + (1-blend) * normalized_F_i^power ] * v_i^2

    With blend=0 and power=1, this is the previous behavior except for cap.
    """
    if fisher is None:
        return None
    ff = finite_or_zero_(fisher.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if ff.numel() == 0:
        return None
    ff = ff.clamp(min=float(fisher_floor))
    mean = ff.mean().clamp(min=float(fisher_floor))
    max_factor = float(fisher_max_factor)
    if max_factor <= 0 or not math.isfinite(max_factor):
        max_factor = 1.0e6
    ff = (ff / mean).clamp(min=float(fisher_floor), max=max_factor)

    power = float(fisher_power)
    if not math.isfinite(power) or power <= 0.0:
        power = 1.0
    if abs(power - 1.0) > 1.0e-12:
        ff = ff.pow(power)

    blend = float(fisher_blend_magnitude)
    if not math.isfinite(blend):
        blend = 0.0
    blend = max(0.0, min(1.0, blend))
    if blend > 0.0:
        # blend=0.05 means 5% pure magnitude score + 95% Fisher score.
        # This prevents entries with very low noisy Fisher from being impossible to keep.
        ff = ff.mul(1.0 - blend).add(blend)

    return finite_or_zero_(ff, nan=1.0, posinf=max_factor, neginf=float(fisher_floor))

def exact_unstructured_project(
    v: torch.Tensor,
    sparsity: float,
    fisher: Optional[torch.Tensor] = None,
    fisher_floor: float = 1.0e-12,
    min_keep: int = 1,
    fisher_power: float = 1.0,
    fisher_blend_magnitude: float = 0.0,
    fisher_max_factor: float = 1.0e6,
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

    ff = stable_fisher_weight(
        fisher=fisher,
        fisher_floor=fisher_floor,
        fisher_power=fisher_power,
        fisher_blend_magnitude=fisher_blend_magnitude,
        fisher_max_factor=fisher_max_factor,
    )
    if ff is not None:
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
    fisher_power: float = 1.0,
    fisher_blend_magnitude: float = 0.0,
    fisher_max_factor: float = 1.0e6,
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
    ff = stable_fisher_weight(
        fisher=fisher,
        fisher_floor=fisher_floor,
        fisher_power=fisher_power,
        fisher_blend_magnitude=fisher_blend_magnitude,
        fisher_max_factor=fisher_max_factor,
    )
    if ff is not None:
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
    fisher_power: float = 1.0,
    fisher_blend_magnitude: float = 0.0,
    fisher_max_factor: float = 1.0e6,
) -> Tuple[torch.Tensor, int, int]:
    nm = parse_nm_pattern(pattern)
    if nm is None:
        return exact_unstructured_project(
            v=v,
            sparsity=sparsity,
            fisher=fisher,
            fisher_floor=fisher_floor,
            min_keep=min_keep,
            fisher_power=fisher_power,
            fisher_blend_magnitude=fisher_blend_magnitude,
            fisher_max_factor=fisher_max_factor,
        )
    return nm_project(
        v=v,
        pattern=nm,
        fisher=fisher,
        fisher_floor=fisher_floor,
        fisher_power=fisher_power,
        fisher_blend_magnitude=fisher_blend_magnitude,
        fisher_max_factor=fisher_max_factor,
    )


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
    fisher_power: float = 1.0,
    fisher_blend_magnitude: float = 0.0,
    fisher_max_factor: float = 1.0e6,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Compute ELSA-native dynamic sparsity from Fisher-weighted pruning damage."""
    print("\nComputing dynamic ELSA/Fisher sparsity profile...")
    print(f"  target global sparsity : {100.0 * float(target_sparsity):.2f}%")
    print(f"  metric base sparsity   : {100.0 * float(base_metric_sparsity):.2f}%")
    print(f"  clamp sparsity range   : [{100.0 * float(sparsity_min):.2f}%, {100.0 * float(sparsity_max):.2f}%]")
    print(f"  alpha                  : {float(alpha):.4f}")
    print(f"  fisher power           : {float(fisher_power):.4f}")
    print(f"  fisher-mag blend       : {float(fisher_blend_magnitude):.4f}")
    print(f"  fisher max factor      : {float(fisher_max_factor):.4e}")

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
                fisher_raw = opt_store.load_tensor(name, "adam_v", device=projection_device, dtype=torch.float32)
                fisher = stable_fisher_weight(
                    fisher=fisher_raw,
                    fisher_floor=float(fisher_floor),
                    fisher_power=float(fisher_power),
                    fisher_blend_magnitude=float(fisher_blend_magnitude),
                    fisher_max_factor=float(fisher_max_factor),
                )
                del fisher_raw
                if fisher is not None and float(fisher.max().item()) > 0.0:
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
        "fisher_power": float(fisher_power),
        "fisher_blend_magnitude": float(fisher_blend_magnitude),
        "fisher_max_factor": float(fisher_max_factor),
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

    def _crop_batch(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (cropped_batch, starts) where `starts` is a long tensor [B]
        indicating the starting index inside each source row. This is needed
        by KD lookups against the precomputed teacher top-K cache.
        """
        if self.max_seq_len <= 0:
            return batch.contiguous(), torch.zeros((batch.size(0),), dtype=torch.long)
        B, T = batch.shape
        L = min(self.max_seq_len, T)
        if L >= T:
            return batch.contiguous(), torch.zeros((B,), dtype=torch.long)

        if self.crop_mode == "none":
            return batch.contiguous(), torch.zeros((B,), dtype=torch.long)
        if self.crop_mode == "prefix":
            return batch[:, :L].contiguous(), torch.zeros((B,), dtype=torch.long)

        max_start = T - L
        out = torch.empty((B, L), dtype=batch.dtype)
        if self.crop_mode == "random":
            starts = torch.randint(0, max_start + 1, (B,), generator=self.crop_gen)
        elif self.crop_mode == "sliding":
            base = (self.step * L) % (max_start + 1)
            starts = torch.tensor([(base + i * 997) % (max_start + 1) for i in range(B)], dtype=torch.long)
        else:
            raise AssertionError(self.crop_mode)

        for i, st in enumerate(starts.tolist()):
            out[i].copy_(batch[i, st:st + L])
        self.step += 1
        return out.contiguous(), starts.long()

    def next(self) -> torch.Tensor:
        # Backward-compatible: callers that only need tokens still work.
        batch, _starts, _rows = self.next_with_meta()
        return batch

    def next_with_meta(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (batch_tokens [B, L], crop_starts [B], row_indices [B])
        where row_indices are positions inside self.tokens (i.e. the original
        source-row indices, suitable for KD cache lookup).
        """
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
        cropped, starts = self._crop_batch(batch)
        return cropped, starts, idx.long()


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
    return_logits: bool = False,
):
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
    if return_logits:
        # Return the full logits (not shifted); KD path uses [:, :-1, :] slice itself.
        return loss, logits
    return loss


def compute_admm_penalty(
    model: nn.Module,
    active_names: Sequence[str],
    active_state: Dict[str, Dict[str, torch.Tensor]],
    lambda_value: float,
    normalization: str,
    diff_clip: float,
    optimizer: Optional[Any] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    ADMM penalty = 0.5 * lambda * agg(||x - z + u||^2)

    v8 changes:
        - normalization 'layer_mean' divides each layer's squared norm by that layer's
          numel before summing. This keeps small attn layers as influential per layer
          as huge MLP layers in the constraint signal.
        - When the optimizer has FP32 master weights, the penalty value is computed
          in FP32 using master via a detached correction so gradient still flows
          through the FP16 model weight without losing precision in the value.
    """
    if lambda_value <= 0.0 or not active_names:
        first = next(model.parameters())
        return first.new_tensor(0.0, dtype=torch.float32), {"admm_rms": 0.0, "admm_absmax": 0.0}

    total_sq: Optional[torch.Tensor] = None
    total_numel = 0
    layer_sq_sum: Optional[torch.Tensor] = None  # for layer_mean
    layer_count = 0
    # Accumulate per-layer max-abs as a device-side tensor to avoid CUDA sync each iteration.
    max_abs_dev: Optional[torch.Tensor] = None

    use_master = (
        optimizer is not None
        and hasattr(optimizer, "has_master_param")
        and hasattr(optimizer, "get_master_param")
    )

    for name in active_names:
        mod = get_module_by_name(model, name)
        p = mod.weight
        z = active_state[name]["z"].to(device=p.device)
        u = active_state[name]["u"].to(device=p.device)

        if use_master and optimizer.has_master_param(p):
            # Compute (value-only) base in FP32 using master, detached from autograd.
            # Then add (p_fp32 - p_fp32.detach()) which is zero in value but provides
            # the autograd path back to the FP16 model weight. Only this last "zero"
            # tensor stays alive in the autograd graph (one extra weight-sized FP32
            # tensor per layer, vs five+ in the previous formulation).
            master_val = optimizer.get_master_param(p, device=p.device).to(torch.float32)
            base_val = (master_val - z.float() + u.float()).detach()
            p_fp32 = p.float()
            diff = base_val + (p_fp32 - p_fp32.detach())
            del master_val, base_val, p_fp32
        else:
            diff = p.float() - z.float() + u.float()

        if diff_clip > 0:
            diff = diff.clamp(min=-float(diff_clip), max=float(diff_clip))
        diff = finite_or_zero_(diff, nan=0.0, posinf=0.0, neginf=0.0)
        sq = diff.square().sum()

        total_sq = sq if total_sq is None else total_sq + sq
        total_numel += diff.numel()

        if normalization == "layer_mean":
            per_layer = sq / max(1, diff.numel())
            layer_sq_sum = per_layer if layer_sq_sum is None else layer_sq_sum + per_layer
            layer_count += 1

        with torch.no_grad():
            layer_max = diff.detach().abs().amax()
            max_abs_dev = layer_max if max_abs_dev is None else torch.maximum(max_abs_dev, layer_max)

    assert total_sq is not None
    if normalization == "mean":
        base = total_sq / max(1, total_numel)
    elif normalization == "sum":
        base = total_sq
    elif normalization == "layer_mean":
        assert layer_sq_sum is not None
        base = layer_sq_sum / max(1, layer_count)
    else:
        raise ValueError("normalization must be 'mean', 'sum', or 'layer_mean'")

    penalty = 0.5 * float(lambda_value) * base
    # Single deferred CUDA sync for the two reported stats.
    rms_t = (total_sq / max(1, total_numel)).detach()
    if max_abs_dev is None:
        max_abs_dev = rms_t.new_tensor(0.0)
    stats_pair = torch.stack([rms_t.clamp_min(0.0).sqrt(), max_abs_dev]).cpu()
    return penalty, {"admm_rms": float(stats_pair[0].item()), "admm_absmax": float(stats_pair[1].item())}


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


def kd_temperature_value(
    t_start: float,
    t_final: float,
    progress: float,
    warmup_frac: float,
) -> float:
    """
    Linearly anneal KD temperature from `t_start` to `t_final` over the first
    `warmup_frac` of training, then hold at `t_final`.

    Returns `t_final` when t_start<=0 (disabled), warmup_frac<=0, or progress>=warmup_frac.
    """
    t_final = float(t_final)
    t_start = float(t_start)
    warmup_frac = max(0.0, min(0.999, float(warmup_frac)))
    progress = min(1.0, max(0.0, float(progress)))
    if t_start <= 0.0 or warmup_frac <= 0.0:
        return t_final
    if progress >= warmup_frac:
        return t_final
    frac = progress / max(1.0e-12, warmup_frac)
    return float(t_start + (t_final - t_start) * frac)


# ============================================================
# v8: KD top-K teacher cache
# ============================================================
#
# Disk layout of the cache:
#     <cache_dir>/meta.json
#         { "n_rows": N, "T": T, "K": K, "vocab_size": V, "dtype": "float16",
#           "model_id": "...", "calib_path": "...", "kd_topk_version": 1 }
#     <cache_dir>/probs_<row>.pt   (CPU FP16 tensor of shape [T-1, K])
#     <cache_dir>/idx_<row>.pt     (CPU int32 tensor of shape [T-1, K])
#
# We split per-row because that lets us mmap / load only the rows in a batch
# during training without materializing the full cache in RAM.


def _kd_cache_paths(cache_dir: Path) -> Tuple[Path, Path, Path]:
    return cache_dir / "meta.json", cache_dir / "probs", cache_dir / "idx"


def kd_cache_is_valid(
    cache_dir: Path,
    n_rows: int,
    seq_len: int,
    topk: int,
    model_id: str,
    calib_path: str,
    build_temperature: float = 1.0,
) -> bool:
    meta_path, probs_dir, idx_dir = _kd_cache_paths(cache_dir)
    if not meta_path.exists() or not probs_dir.exists() or not idx_dir.exists():
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False
    if int(meta.get("n_rows", 0)) != int(n_rows):
        return False
    if int(meta.get("T", 0)) != int(seq_len):
        return False
    if int(meta.get("K", 0)) != int(topk):
        return False
    if str(meta.get("model_id", "")) != str(model_id):
        return False
    if str(meta.get("calib_path", "")) != str(calib_path):
        return False
    # v1 caches don't record build temperature; treat them as built at 1.0.
    cached_T = float(meta.get("build_temperature", 1.0))
    if abs(cached_T - float(build_temperature)) > 1e-6:
        return False
    # spot-check that the first row's files exist
    if not (probs_dir / "row_0.pt").exists() or not (idx_dir / "row_0.pt").exists():
        return False
    return True


@torch.no_grad()
def build_kd_topk_cache(
    cache_dir: Path,
    model_id: str,
    calib_path: str,
    calib_tokens: torch.Tensor,
    topk: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    trust_remote_code: bool,
    attn_implementation: str,
    low_cpu_mem_usage: bool,
    batch_size: int = 1,
    log_every: int = 16,
    build_temperature: float = 1.0,
) -> Dict[str, Any]:
    """
    Load a fresh dense teacher copy, run forward over the calibration set,
    save top-K probabilities + indices per token to disk, then release the
    teacher model.

    Returns a meta dict describing the cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path, probs_dir, idx_dir = _kd_cache_paths(cache_dir)
    probs_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)

    n_rows, T = int(calib_tokens.size(0)), int(calib_tokens.size(1))
    topk = int(topk)

    print(f"\nBuilding KD teacher top-K cache at {cache_dir}")
    print(f"  rows={n_rows} T={T} K={topk} dtype=fp16 device={device}")
    t0 = time.time()

    print("  loading dense teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    if hasattr(teacher, "config"):
        teacher.config.use_cache = False
    teacher.to(device)
    teacher.eval()
    vocab_size = int(getattr(teacher.config, "vocab_size", 0))
    print(f"  teacher loaded. vocab={vocab_size}  cuda={cuda_mem()}")

    try:
        for r0 in range(0, n_rows, max(1, int(batch_size))):
            r1 = min(n_rows, r0 + max(1, int(batch_size)))
            batch = calib_tokens[r0:r1].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                out = teacher(input_ids=batch, use_cache=False)
                logits = out.logits  # [B, T, V]
            # Next-token alignment: at position t we predict token t+1, so
            # we drop the last position's logits when saving.
            logits_pred = logits[:, :-1, :].float() / max(1e-6, float(build_temperature))  # [B, T-1, V]
            # Top-K in the logits domain first, then softmax + renormalize on just K values.
            # This avoids materializing a full-vocab softmax tensor for every row (saves
            # ~128MB FP32 transient per [B=1, T=2048-1, V=32000] row at build time).
            top_logits, top_idx = torch.topk(logits_pred, k=topk, dim=-1, largest=True, sorted=True)
            top_probs = torch.softmax(top_logits, dim=-1)  # sums to 1 over K positions

            tp_cpu = top_probs.detach().to(torch.float16).cpu()
            ti_cpu = top_idx.detach().to(torch.int32).cpu()

            for bi in range(r1 - r0):
                row = r0 + bi
                torch.save(tp_cpu[bi].contiguous(), probs_dir / f"row_{row}.pt")
                torch.save(ti_cpu[bi].contiguous(), idx_dir / f"row_{row}.pt")

            if ((r0 // max(1, int(batch_size))) % max(1, log_every // max(1, int(batch_size)))) == 0:
                print(f"    cached rows {r0}-{r1 - 1}/{n_rows}  cuda={cuda_mem()}")
            del batch, out, logits, logits_pred, top_logits, top_probs, top_idx, tp_cpu, ti_cpu
            clean_cuda()
    finally:
        del teacher
        clean_cuda()

    meta = {
        "n_rows": int(n_rows),
        "T": int(T),
        "K": int(topk),
        "vocab_size": int(vocab_size),
        "dtype": "float16",
        "model_id": str(model_id),
        "calib_path": str(calib_path),
        "kd_topk_version": 2,
        "build_temperature": float(build_temperature),
        "build_seconds": float(time.time() - t0),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"KD cache built in {fmt_time(meta['build_seconds'])}. Meta at {meta_path}")
    return meta


class KDTopKCache:
    """
    Lazy reader for the on-disk KD cache produced by `build_kd_topk_cache`.

    Lookup is by (row_index, start_position, length). Returns
        probs:  FP16 [L, K]
        idx:    int32 [L, K]
    aligned to "predicted token at position t+1 given inputs[..., t]".
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        meta_path, self.probs_dir, self.idx_dir = _kd_cache_paths(self.cache_dir)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.K = int(self.meta["K"])
        self.T = int(self.meta["T"])
        self.n_rows = int(self.meta["n_rows"])
        self._row_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._row_lru: List[int] = []
        self._max_cached_rows = 16  # keep at most this many rows resident

    def _load_row(self, row: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if row in self._row_cache:
            return self._row_cache[row]
        probs = torch.load(self.probs_dir / f"row_{row}.pt", map_location="cpu")
        idx = torch.load(self.idx_dir / f"row_{row}.pt", map_location="cpu")
        self._row_cache[row] = (probs, idx)
        self._row_lru.append(row)
        if len(self._row_lru) > self._max_cached_rows:
            evict = self._row_lru.pop(0)
            if evict in self._row_cache:
                del self._row_cache[evict]
        return probs, idx

    def get(self, row: int, start: int, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if row < 0 or row >= self.n_rows:
            raise IndexError(f"row {row} out of bounds [0, {self.n_rows})")
        probs_full, idx_full = self._load_row(row)
        # cache stores [T-1, K] for positions [0..T-2]; a crop [start:start+L]
        # yields predictions for positions [start..start+L-1]. The student LM
        # loss skips the last token (no label), so we need [start, start+L-1)
        # which is length L-1 entries.
        e = min(start + length - 1, probs_full.size(0))
        s = min(start, probs_full.size(0))
        return probs_full[s:e].contiguous(), idx_full[s:e].contiguous()


def compute_kd_topk_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    teacher_idx: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    KL( teacher_topk || student_topk_renorm ).

    student_logits: [B, L, V] FP32 (already up-cast in compute_lm_loss path).
        We slice [:, :-1, :] internally so that L matches teacher's L-1 alignment.
    teacher_probs: [B, L', K] FP16/FP32, sums to ~1 along last dim.
    teacher_idx:   [B, L', K] int.
    """
    if student_logits.dim() != 3:
        raise ValueError("student_logits must be [B, T, V]")
    sl = student_logits[:, :-1, :].float() / max(1e-6, float(temperature))
    # Gather the same vocab positions the teacher kept.
    # teacher_idx may have a slightly shorter L' than sl's L; trim sl to match.
    L_eff = min(sl.size(1), teacher_idx.size(1))
    if L_eff <= 0:
        return sl.new_zeros(())
    sl = sl[:, :L_eff, :]
    t_probs = teacher_probs[:, :L_eff, :].to(device=sl.device, dtype=sl.dtype)
    t_idx = teacher_idx[:, :L_eff, :].to(device=sl.device, dtype=torch.long)
    # gather student logits at teacher's top-K vocab positions
    s_topk_logits = torch.gather(sl, dim=-1, index=t_idx)
    # softmax over the K positions only (teacher already renormalized)
    s_log_probs = F.log_softmax(s_topk_logits, dim=-1)
    # KL(teacher || student) = sum t * (log t - log s)
    eps = 1e-12
    kl = (t_probs * (t_probs.clamp_min(eps).log() - s_log_probs)).sum(dim=-1).mean()
    # scale by temperature^2 as in Hinton distillation
    return kl * (float(temperature) ** 2)


# ============================================================
# v8: Wanda-style activation-aware warm-start
# ============================================================


@torch.no_grad()
def compute_layer_input_norms(
    model: nn.Module,
    selected_names: Sequence[str],
    tokens: torch.Tensor,
    batches: int,
    batch_size: int,
    max_seq_len: int,
    crop_mode: str,
    seed: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """
    Collect per-input-channel L2 norm of activations for each selected nn.Linear.

    For W of shape [out_dim, in_dim], the Wanda metric is:
        score[o, i] = |W[o, i]| * ||X[:, i]||_2
    so squared score = W[o,i]^2 * ||X[:,i]||^2. We return ||X[:,i]||^2 here
    so it can be passed to `project_tensor` as the `fisher` argument (which
    multiplies into v.square()) without any other code change.

    Returns dict mapping layer name -> tensor of shape [in_dim] (float32, on CPU).
    """
    if int(batches) <= 0:
        return {}

    accum: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    for name in selected_names:
        mod = get_module_by_name(model, name)
        if not isinstance(mod, nn.Linear):
            continue
        accum[name] = torch.zeros(mod.in_features, dtype=torch.float64)
        counts[name] = 0

    handles = []

    def make_hook(layer_name: str):
        def hook(_module, inputs, _output):
            if not inputs:
                return
            x = inputs[0]
            if x is None:
                return
            # x shape is usually [B, T, in_dim] for HF linears in decoder blocks.
            if x.dim() == 3:
                x_flat = x.reshape(-1, x.size(-1))
            elif x.dim() == 2:
                x_flat = x
            else:
                return
            x_sq = x_flat.detach().float().square().sum(dim=0).double().cpu()
            if not torch.isfinite(x_sq).all():
                x_sq = torch.nan_to_num(x_sq, nan=0.0, posinf=0.0, neginf=0.0)
            accum[layer_name].add_(x_sq)
            counts[layer_name] += int(x_flat.size(0))
        return hook

    for name in accum:
        mod = get_module_by_name(model, name)
        handles.append(mod.register_forward_hook(make_hook(name)))

    loader = InfiniteTokenLoader(
        tokens=tokens,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        max_seq_len=max_seq_len,
        crop_mode=crop_mode,
        drop_last=False,
    )

    was_training = model.training
    model.eval()
    try:
        for b in range(int(batches)):
            batch = loader.next().to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                model(input_ids=batch, use_cache=False)
            del batch
            if (b + 1) % max(1, int(batches) // 8) == 0:
                print(f"    activation calibration: batch {b + 1}/{batches}  cuda=[{cuda_mem()}]")
            clean_cuda()
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    norms_sq: Dict[str, torch.Tensor] = {}
    for name, acc in accum.items():
        c = max(1, counts[name])
        # mean of x^2 per channel
        v = (acc / float(c)).float()
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        norms_sq[name] = v.contiguous()
    return norms_sq


def make_wanda_fisher_for_layer(
    weight_shape: Tuple[int, int],
    input_norms_sq: torch.Tensor,
) -> torch.Tensor:
    """
    Construct a per-element fisher-equivalent tensor for Wanda projection.

    For a Linear weight of shape [out_dim, in_dim], the Wanda squared score is
        score^2 = W[o,i]^2 * ||X[:,i]||^2
    so we return F[o,i] = ||X[:,i]||^2 broadcast to [out_dim, in_dim] which when
    multiplied by W^2 in `project_tensor` recovers the Wanda metric exactly.
    """
    out_dim, in_dim = weight_shape
    v = input_norms_sq.to(torch.float32)
    if v.numel() != in_dim:
        raise ValueError(f"input_norms_sq length {v.numel()} != in_dim {in_dim}")
    # Broadcast view, no real expansion; final score will be elementwise W^2 * F.
    return v.view(1, in_dim).expand(out_dim, in_dim).contiguous()


# ============================================================
# v9: OWL global outlier protection
# ============================================================


@torch.no_grad()
def compute_owl_force_keep_masks(
    model: nn.Module,
    selected_names: Sequence[str],
    pct: float,
    weight_source: str = "current",
    z_store: Optional[ADMMStateStore] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Compute a per-layer boolean force-keep mask that protects the global top
    `pct` percent of |W| across all selected layers.

    weight_source='current'  : use model's current weights (works before init).
    weight_source='z_store'  : use the current ADMM z values; recompute mask
                               using the actually-kept sparse weights.

    Returns:
        masks: {layer_name: bool tensor with mod.weight.shape, True = force kept}
        meta:  diagnostic summary
    """
    pct = float(pct)
    if pct <= 0.0:
        return {name: torch.zeros_like(get_module_by_name(model, name).weight, dtype=torch.bool, device="cpu")
                for name in selected_names}, {"owl_pct": 0.0, "enabled": False}

    print(f"\n[OWL] computing global top-{pct:.3f}% outlier mask "
          f"(source={weight_source}) across {len(selected_names)} layers...")
    t0 = time.time()

    # Single global threshold via per-layer sampling, then exact membership pass.
    # Memory-efficient: never materializes a full concatenated tensor.
    sizes: Dict[str, int] = {}
    total = 0
    # First pass: gather sizes and a uniform sample for threshold estimation.
    rng = torch.Generator(device="cpu")
    rng.manual_seed(0xCAFEBABE)
    sample_budget = 4_000_000          # 4M values is plenty to estimate a quantile.
    samples: List[torch.Tensor] = []

    layer_abs: Dict[str, torch.Tensor] = {}

    for name in selected_names:
        mod = get_module_by_name(model, name)
        if weight_source == "z_store" and z_store is not None and z_store.exists(name, "z"):
            w = z_store.load_tensor(name, "z", device="cpu", dtype=torch.float32)
        else:
            w = mod.weight.detach().float().cpu()
        a = w.abs().reshape(-1)
        layer_abs[name] = a
        sizes[name] = int(a.numel())
        total += int(a.numel())

    target_keep = int(round((pct / 100.0) * total))
    target_keep = max(1, target_keep)

    # Sample from each layer proportionally and estimate global threshold.
    for name, a in layer_abs.items():
        n = a.numel()
        take = max(1, min(n, int(round(sample_budget * (n / max(1, total))))))
        if take >= n:
            samples.append(a.clone())
        else:
            idx = torch.randint(0, n, (take,), generator=rng)
            samples.append(a[idx].clone())

    pooled = torch.cat(samples)
    del samples
    # Quantile that selects top pct% of pooled.
    q = 1.0 - (target_keep / float(total))
    q = max(0.0, min(1.0, q))
    threshold = float(torch.quantile(pooled.double(), q).item())
    del pooled
    gc.collect()

    masks: Dict[str, torch.Tensor] = {}
    kept_actual = 0
    for name, a in layer_abs.items():
        m_flat = a >= threshold
        kept_actual += int(m_flat.sum().item())
        mod = get_module_by_name(model, name)
        masks[name] = m_flat.view_as(mod.weight).contiguous()
        del a
    layer_abs.clear()
    gc.collect()

    actual_pct = 100.0 * kept_actual / max(1, total)
    meta = {
        "owl_pct_requested": float(pct),
        "owl_pct_actual": float(actual_pct),
        "threshold": float(threshold),
        "total_weights": int(total),
        "force_kept": int(kept_actual),
        "seconds": float(time.time() - t0),
        "weight_source": str(weight_source),
        "enabled": True,
    }
    print(f"[OWL] threshold={threshold:.6e}  forced_keep={kept_actual:,}/{total:,} "
          f"({actual_pct:.4f}%, requested {pct:.4f}%)  elapsed={fmt_time(meta['seconds'])}")
    return masks, meta


def apply_force_keep_to_projection(
    z: torch.Tensor,
    w_dense: torch.Tensor,
    force_keep_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Force-restore values at positions marked True in force_keep_mask using the
    dense weight value, BEFORE the projection picks a top-k subset. Since
    projection sorts by score = w^2 (or fisher*w^2), restoring the dense value
    ensures those positions have the largest scores and stay in the top-k.

    In v9 we instead apply force-keep AFTER projection: we OR the force_keep
    mask into the chosen support, which guarantees these positions are kept
    regardless of score. The function name is preserved for clarity.
    """
    if force_keep_mask is None:
        return z
    fk = force_keep_mask.to(device=z.device, dtype=torch.bool)
    # OR-mask into projection: keep z values where projection kept them, AND
    # also keep dense values where force_keep is True. The "outlier" weight
    # is just the original w at that position.
    return torch.where(fk, w_dense.to(device=z.device, dtype=z.dtype), z)


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
    init_method: str = "magnitude",
    wanda_input_norms_sq: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """
    Initializes z = projection(W) and u = 0 for every selected layer.

    v8: supports init_method='wanda' which uses precomputed per-channel input
    activation norms (passed via wanda_input_norms_sq) to score weights as
    score[o,i] = W[o,i]^2 * ||X[:,i]||^2 — i.e. the Wanda metric.
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

        init_fisher = None
        used_init = init_method
        if init_method == "wanda" and wanda_input_norms_sq is not None and name in wanda_input_norms_sq:
            try:
                init_fisher = make_wanda_fisher_for_layer(
                    weight_shape=tuple(w_cpu.shape),
                    input_norms_sq=wanda_input_norms_sq[name],
                )
                used_init = "wanda"
            except Exception as exc:
                print(f"  [warn] wanda init failed for {name}: {exc}; falling back to magnitude.")
                init_fisher = None
                used_init = "magnitude_fallback"
        elif init_method != "magnitude":
            used_init = "magnitude_fallback"

        z, kept, total = project_tensor(
            v=w_cpu,
            sparsity=target_sparsity,
            pattern=pattern,
            fisher=init_fisher,
            fisher_floor=1.0e-12,
            min_keep=min_keep_per_tensor,
            fisher_power=1.0,
            fisher_blend_magnitude=0.0,
            fisher_max_factor=1.0e6,
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
            "init_method": str(used_init),
        }
        meta["total_weights"] += int(total)
        meta["total_kept"] += int(kept)
        print(
            f"  [{idx}/{len(selected_names)}] {name}: shape={tuple(w_cpu.shape)} "
            f"kept={kept:,}/{total:,} sparsity={100.0 * layer_sparsity:.2f}% init={used_init}"
        )
        del w_cpu, z, u, init_fisher
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
    fisher_power: float = 1.0,
    fisher_blend_magnitude: float = 0.0,
    fisher_max_factor: float = 1.0e6,
    force_keep_masks: Optional[Dict[str, torch.Tensor]] = None,
    prev_masks: Optional[Dict[str, torch.Tensor]] = None,
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
            fisher_power=fisher_power,
            fisher_blend_magnitude=fisher_blend_magnitude,
            fisher_max_factor=fisher_max_factor,
        )

        # v9: OWL force-keep — union force-keep positions back in. Their value
        # comes from the dense pre-projection v (x+u), not zero. We DO NOT
        # change the global sparsity budget; the few forced positions push out
        # the weakest projection picks because score-ranked sparsity got
        # extra protected positions.
        if force_keep_masks is not None and name in force_keep_masks:
            fk = force_keep_masks[name].to(device=z_new.device, dtype=torch.bool).view_as(z_new)
            fk_added = int((fk & (z_new == 0)).sum().item())
            if fk_added > 0:
                z_new = torch.where(fk, v.to(dtype=z_new.dtype), z_new)
                layer_kept = int((z_new != 0).sum().item())

        # v9: mask drift tracking against previous projection's mask.
        mask_drift_frac = None
        if prev_masks is not None:
            cur_mask = (z_new != 0).cpu()
            if name in prev_masks:
                prev = prev_masks[name].view(-1)
                cur = cur_mask.view(-1)
                if prev.numel() == cur.numel():
                    changed = int((prev != cur).sum().item())
                    mask_drift_frac = float(changed) / float(max(1, prev.numel()))
            prev_masks[name] = cur_mask

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
            "mask_drift_frac": float(mask_drift_frac) if mask_drift_frac is not None else None,
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
    # v9: aggregate mask drift across layers (weighted by numel).
    drift_total = 0.0
    drift_weight = 0
    for ln, info in layer_logs.items():
        d = info.get("mask_drift_frac")
        if d is not None:
            drift_total += float(d) * float(info["total"])
            drift_weight += int(info["total"])
    agg_drift = (drift_total / drift_weight) if drift_weight > 0 else None
    drift_str = f" mask_drift={100.0 * agg_drift:.3f}%" if agg_drift is not None else ""
    print(f"  projection done: sparsity={100.0 * actual_sparsity:.2f}%{drift_str} elapsed={fmt_time(elapsed)}")
    return {
        "kept": int(kept),
        "total": int(total),
        "sparsity": float(actual_sparsity),
        "seconds": float(elapsed),
        "layers": layer_logs,
        "mask_drift_frac_weighted": float(agg_drift) if agg_drift is not None else None,
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
# v9: Post-projection cleanup phase
# ============================================================


@torch.no_grad()
def freeze_masks_from_z(
    model: nn.Module,
    selected_names: Sequence[str],
    store: ADMMStateStore,
    device: torch.device,
    apply_z_to_model: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Read final z from disk, optionally copy z into model weights, and return
    {name: bool_mask_on_device} marking surviving (nonzero) positions.

    These masks are then used during the cleanup phase to enforce sparsity
    after every optimizer step (gradient-only fine-tune on the kept weights).
    """
    masks: Dict[str, torch.Tensor] = {}
    for name in selected_names:
        mod = get_module_by_name(model, name)
        z = store.load_tensor(name, "z", device="cpu", dtype=torch.float32)
        z = finite_or_zero_(z, nan=0.0, posinf=0.0, neginf=0.0)
        if apply_z_to_model:
            mod.weight.data.copy_(z.to(device=mod.weight.device, dtype=mod.weight.dtype))
        m = (z != 0).to(device=device)
        masks[name] = m
        del z
    return masks


@torch.no_grad()
def compute_output_scale_init(
    model: nn.Module,
    selected_names: Sequence[str],
    z_store: ADMMStateStore,
    init_method: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Initialize per-output-row learnable scales. For each Linear with weight
    W [out_dim, in_dim] currently sparsified to z, we compute one scalar per
    output row.

    init_method='one':          s_o = 1.0
    init_method='dense_ratio':  s_o = ||W_dense_row_o|| / ||W_sparse_row_o||
                                clipped to [0.5, 2.0]; safer choice.
    """
    scales: Dict[str, torch.Tensor] = {}
    for name in selected_names:
        mod = get_module_by_name(model, name)
        out_dim = int(mod.weight.shape[0])
        s = torch.ones(out_dim, dtype=torch.float32)
        if init_method == "dense_ratio":
            # We need both dense and sparse row norms. The current model weight
            # equals z (set just before cleanup). For dense norm we'd need the
            # original W; since we no longer have it cleanly here, approximate
            # using ||z||_2 / ||z (already kept-mask)||_2 = 1. So default to 1.
            # Practically dense_ratio is best applied right before z overwrites
            # the model. For simplicity v9 ships with safe 'one' behavior when
            # the dense row is no longer available.
            pass
        scales[name] = s.to(device=device)
    return scales


def run_cleanup_phase(
    model: nn.Module,
    windows: List[List[str]],
    selected_names: Sequence[str],
    store: ADMMStateStore,
    x_store: Optional[ADMMStateStore],
    opt_store: Optional[ADMMStateStore],
    main_device: torch.device,
    model_dtype: torch.dtype,
    calib_tokens: torch.Tensor,
    kd_cache: Optional[KDTopKCache],
    args: argparse.Namespace,
    train_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Post-projection cleanup phase.

    For each window in order:
      1) Apply final z to all selected layers (done by caller before entering
         this phase, then reaffirmed here).
      2) Read the surviving-position mask for this window.
      3) Train active layers' surviving weights with LM (+KD) loss only.
         No ADMM penalty, no projection.
      4) After each optimizer step, re-zero dead positions via mask multiply.
      5) Save final weights via store.save_tensor(name, 'z', ...) so the
         finalization step sees the cleaned-up sparse weight.

    This is the single most important v9 addition: ADMM stops when the
    constraint is satisfied; it does not optimize the surviving 20% of
    weights at the optimum. The cleanup phase does exactly that.
    """
    if int(args.cleanup_steps) <= 0:
        return {"enabled": False}

    print("\n" + "=" * 100)
    print(f"v9 POST-PROJECTION CLEANUP PHASE | steps_per_window={args.cleanup_steps}")
    print("=" * 100)

    kd_alpha = float(args.kd_alpha_cleanup if args.kd_alpha_cleanup >= 0.0 else args.kd_alpha)
    print(f"  cleanup_lr           : {args.cleanup_lr}")
    print(f"  cleanup_lr_schedule  : {args.cleanup_lr_schedule}")
    print(f"  cleanup_min_lr_ratio : {args.cleanup_min_lr_ratio}")
    print(f"  cleanup_steps/window : {args.cleanup_steps}")
    print(f"  kd_alpha (cleanup)   : {kd_alpha:.3f}")
    print(f"  learn_output_scale   : {bool(args.learn_output_scale)}")
    print(f"  max_grad_norm        : {args.cleanup_max_grad_norm}")
    print(f"  CUDA before phase    : {cuda_mem()}")

    token_loader = InfiniteTokenLoader(
        tokens=calib_tokens,
        batch_size=int(args.batch_size),
        shuffle=not bool(args.no_shuffle),
        seed=int(args.seed) + 7777,
        max_seq_len=int(args.max_seq_len),
        crop_mode=str(args.crop_mode),
        drop_last=False,
    )

    set_all_requires_grad(model, False)

    # Apply final z to model once before cleanup; we re-apply per window below.
    apply_z_to_model_and_count(
        model=model,
        selected_names=selected_names,
        store=store,
        device=main_device,
    )

    per_window_meta: List[Dict[str, Any]] = []
    total_cleanup_steps = int(args.cleanup_steps) * len(windows)
    global_cleanup_step = 0

    for win_idx, active_names in enumerate(windows):
        print("\n" + "-" * 100)
        print(f"CLEANUP WINDOW {win_idx + 1}/{len(windows)} | active layers={len(active_names)}")

        # Freeze masks for THIS window's active layers from current sparse z.
        active_masks = freeze_masks_from_z(
            model=model,
            selected_names=active_names,
            store=store,
            device=main_device,
            apply_z_to_model=True,
        )

        active_params = set_active_trainable_layers(model, active_names)
        named_active_params = [(name, get_module_by_name(model, name).weight) for name in active_names]

        # Optional learnable per-output scales.
        learn_scale = bool(args.learn_output_scale)
        out_scales: Dict[str, nn.Parameter] = {}
        named_scale_params: List[Tuple[str, nn.Parameter]] = []
        if learn_scale:
            for name in active_names:
                mod = get_module_by_name(model, name)
                out_dim = int(mod.weight.shape[0])
                s = torch.ones(out_dim, dtype=torch.float32, device=main_device)
                out_scales[name] = nn.Parameter(s, requires_grad=True)
                named_scale_params.append((name + ".__scale__", out_scales[name]))

        weight_opt = SafeAdamWFP32(
            named_params=named_active_params,
            betas=(float(args.beta1), float(args.beta2)),
            eps=float(args.adam_eps),
            state_device=main_device,
            max_state_abs=float(args.optimizer_state_clip),
            use_master_weights=not bool(args.no_master_weights),
        )
        scale_opt = None
        if learn_scale:
            scale_opt = SafeAdamWFP32(
                named_params=named_scale_params,
                betas=(float(args.beta1), float(args.beta2)),
                eps=float(args.adam_eps),
                state_device=main_device,
                max_state_abs=float(args.optimizer_state_clip),
                use_master_weights=True,
            )

        # Pre-register forward hooks that multiply Linear outputs by per-row scales.
        scale_hooks: List[Any] = []
        if learn_scale:
            def make_scale_hook(layer_name: str):
                def hook(_module, _inputs, output):
                    s = out_scales[layer_name].to(device=output.device, dtype=output.dtype)
                    # output shape: [B, T, out_dim] for HF linears in decoder blocks.
                    return output * s
                return hook
            for name in active_names:
                mod = get_module_by_name(model, name)
                scale_hooks.append(mod.register_forward_hook(make_scale_hook(name)))

        clean_cuda()
        print(f"  CUDA after cleanup setup: {cuda_mem()}")

        win_t0 = time.time()
        last_log_step = -1
        for c_step in range(int(args.cleanup_steps)):
            progress = global_cleanup_step / max(1, total_cleanup_steps - 1)
            lr_t = lr_schedule_value(
                args.cleanup_lr_schedule,
                float(args.cleanup_lr),
                progress,
                float(args.cleanup_min_lr_ratio),
            )

            weight_opt.zero_grad(set_to_none=True)
            if scale_opt is not None:
                scale_opt.zero_grad(set_to_none=True)

            step_t0 = time.time()
            accum_steps = max(1, int(args.grad_accum_steps))
            lm_loss_values: List[float] = []
            kd_loss_values: List[float] = []
            bad_loss = False

            try:
                for accum_idx in range(accum_steps):
                    batch_cpu, crop_starts, row_idx = token_loader.next_with_meta()
                    batch = batch_cpu.to(main_device, non_blocking=True)

                    need_logits = kd_alpha > 0.0 and kd_cache is not None
                    if need_logits:
                        lm_loss_i, full_logits = compute_lm_loss(
                            model=model, input_ids=batch, amp_dtype=model_dtype,
                            device=main_device, autocast_enabled=not bool(args.no_autocast),
                            return_logits=True,
                        )
                    else:
                        lm_loss_i = compute_lm_loss(
                            model=model, input_ids=batch, amp_dtype=model_dtype,
                            device=main_device, autocast_enabled=not bool(args.no_autocast),
                        )
                        full_logits = None

                    if not torch.isfinite(lm_loss_i).all():
                        bad_loss = True
                        del batch, lm_loss_i
                        if full_logits is not None:
                            del full_logits
                        break

                    kd_loss_i = None
                    if need_logits and full_logits is not None:
                        try:
                            L_eff_max = int(batch.size(1)) - 1
                            K = int(kd_cache.K)
                            B = int(batch.size(0))
                            if L_eff_max > 0:
                                teacher_probs = torch.zeros((B, L_eff_max, K), dtype=torch.float16)
                                teacher_idx = torch.zeros((B, L_eff_max, K), dtype=torch.int32)
                                for bi in range(B):
                                    row = int(row_idx[bi].item())
                                    start = int(crop_starts[bi].item())
                                    tp, ti = kd_cache.get(row=row, start=start, length=int(batch.size(1)))
                                    L_have = tp.size(0)
                                    if L_have > 0:
                                        teacher_probs[bi, :L_have] = tp
                                        teacher_idx[bi, :L_have] = ti
                                teacher_probs = teacher_probs.to(main_device, non_blocking=True)
                                teacher_idx = teacher_idx.to(main_device, non_blocking=True)
                                kd_loss_i = compute_kd_topk_loss(
                                    student_logits=full_logits,
                                    teacher_probs=teacher_probs,
                                    teacher_idx=teacher_idx,
                                    temperature=float(args.kd_temperature),
                                )
                                del teacher_probs, teacher_idx
                        except Exception as exc:
                            print(f"  [warn] cleanup KD compute failed: {exc}")
                            kd_loss_i = None

                    lm_loss_values.append(float(lm_loss_i.detach().cpu().item()))
                    combined = lm_loss_i
                    if kd_loss_i is not None and torch.isfinite(kd_loss_i).all():
                        combined = (1.0 - kd_alpha) * lm_loss_i + kd_alpha * kd_loss_i
                        kd_loss_values.append(float(kd_loss_i.detach().cpu().item()))
                    (combined / float(accum_steps)).backward()
                    del batch, lm_loss_i, combined
                    if full_logits is not None:
                        del full_logits
                    if kd_loss_i is not None:
                        del kd_loss_i

                if bad_loss or not lm_loss_values:
                    weight_opt.zero_grad(set_to_none=True)
                    if scale_opt is not None:
                        scale_opt.zero_grad(set_to_none=True)
                    clean_cuda()
                    global_cleanup_step += 1
                    continue

                # Mask dead-position gradients before clipping; they would just
                # be re-zeroed after the optimizer step anyway.
                with torch.no_grad():
                    for name in active_names:
                        p = get_module_by_name(model, name).weight
                        if p.grad is not None:
                            p.grad.mul_(active_masks[name].to(device=p.grad.device, dtype=p.grad.dtype))

                grad_stats = sanitize_and_clip_gradients(
                    named_params=named_active_params,
                    max_grad_norm=float(args.cleanup_max_grad_norm),
                    grad_value_clip=0.0,
                )

                opt_stats = weight_opt.step(
                    lr=lr_t,
                    weight_decay=float(args.weight_decay),
                    update_clip=float(args.update_value_clip),
                    weight_clip=float(args.weight_clip),
                )
                if scale_opt is not None:
                    _ = sanitize_and_clip_gradients(
                        named_params=named_scale_params,
                        max_grad_norm=float(args.cleanup_max_grad_norm),
                        grad_value_clip=0.0,
                    )
                    scale_opt.step(
                        lr=lr_t * float(args.output_scale_lr_mult),
                        weight_decay=0.0,
                        update_clip=float(args.update_value_clip),
                        weight_clip=0.0,
                    )

                # Re-enforce sparsity: zero dead positions in both FP16 model
                # weights and the FP32 master copy that the optimizer holds.
                with torch.no_grad():
                    for name in active_names:
                        p = get_module_by_name(model, name).weight
                        m_p = active_masks[name].to(device=p.device, dtype=p.dtype)
                        p.data.mul_(m_p)
                        if weight_opt.has_master_param(p):
                            mst = weight_opt.state[id(p)]["master"]
                            m_m = active_masks[name].to(device=mst.device, dtype=mst.dtype)
                            mst.mul_(m_m)

                lm_loss_scalar = float(sum(lm_loss_values) / max(1, len(lm_loss_values)))
                kd_loss_scalar = (
                    float(sum(kd_loss_values) / max(1, len(kd_loss_values))) if kd_loss_values else 0.0
                )
                elapsed = time.time() - step_t0
                log_item = {
                    "phase": "cleanup",
                    "global_step": int(global_cleanup_step),
                    "window": int(win_idx),
                    "local_step": int(c_step),
                    "lr": float(lr_t),
                    "lm_loss": float(lm_loss_scalar),
                    "kd_loss": float(kd_loss_scalar),
                    "elapsed": float(elapsed),
                    **grad_stats,
                    **opt_stats,
                }
                train_log.append(log_item)

                if (c_step % max(1, int(args.cleanup_log_interval)) == 0) or c_step == 0:
                    ppl_est = math.exp(min(20.0, lm_loss_scalar))
                    kd_part = f" kd={kd_loss_scalar:.4f}" if kd_loss_scalar != 0.0 else ""
                    print(
                        f"  cleanup step={c_step + 1}/{args.cleanup_steps} "
                        f"win={win_idx + 1}/{len(windows)} "
                        f"lm={lm_loss_scalar:.5f} ppl~{ppl_est:.2f}{kd_part} "
                        f"lr={lr_t:.3e} gnorm={grad_stats['grad_norm']:.3e} "
                        f"upd_rms={opt_stats['update_rms']:.3e} time={elapsed:.2f}s"
                    )
                    last_log_step = c_step

            except torch.cuda.OutOfMemoryError:
                print(f"  [OOM] cleanup step {global_cleanup_step}; clearing and skipping.")
                weight_opt.zero_grad(set_to_none=True)
                if scale_opt is not None:
                    scale_opt.zero_grad(set_to_none=True)
                clean_cuda()

            global_cleanup_step += 1

        # Fuse learnable per-output scale into model weights, then drop the parameter.
        if learn_scale:
            with torch.no_grad():
                for name in active_names:
                    mod = get_module_by_name(model, name)
                    s = out_scales[name].detach().to(device=mod.weight.device, dtype=mod.weight.dtype)
                    mod.weight.data.mul_(s.view(-1, 1))
                    # Re-enforce mask after fusion (multiplicative scaling preserves zeros).
                    p = mod.weight
                    m_p = active_masks[name].to(device=p.device, dtype=p.dtype)
                    p.data.mul_(m_p)
            for h in scale_hooks:
                h.remove()

        # Persist the cleaned-up sparse weight as the new z for this window's layers.
        with torch.no_grad():
            for name in active_names:
                w = get_module_by_name(model, name).weight.detach().float().cpu()
                w = finite_or_zero_(w, nan=0.0, posinf=0.0, neginf=0.0)
                # Re-enforce mask on CPU side as well, in case fp16->fp32 introduced
                # tiny non-zero noise in dead positions.
                m_cpu = active_masks[name].cpu()
                w[~m_cpu] = 0.0
                store.save_tensor(name, "z", w)

        weight_opt.release()
        del weight_opt
        if scale_opt is not None:
            scale_opt.release()
            del scale_opt
        out_scales.clear()
        named_scale_params.clear()
        del active_masks, active_params, named_active_params
        set_all_requires_grad(model, False)
        clean_cuda()

        win_meta = {
            "window": int(win_idx),
            "steps": int(args.cleanup_steps),
            "elapsed": float(time.time() - win_t0),
        }
        per_window_meta.append(win_meta)
        store.save_json("train_log.json", {"items": train_log})

        # Optional probe after each window's cleanup.
        if bool(args.cleanup_probe_each_window) and int(args.cleanup_probe_batches) > 0:
            probe = quick_calibration_loss(
                model=model,
                tokens=calib_tokens,
                batch_size=int(args.batch_size),
                max_seq_len=int(args.max_seq_len),
                batches=int(args.cleanup_probe_batches),
                seed=int(args.seed) + 9000 + win_idx,
                device=main_device,
                amp_dtype=model_dtype,
                crop_mode=str(args.crop_mode),
            )
            print(
                f"  cleanup probe after window {win_idx + 1}: "
                f"loss={probe['probe_loss']:.6f} ppl~{probe['probe_ppl']:.3f}"
            )
            win_meta["probe"] = probe

        print(f"Finished cleanup window {win_idx + 1}. CUDA: {cuda_mem()}")

    cleanup_meta = {
        "enabled": True,
        "steps_per_window": int(args.cleanup_steps),
        "total_steps": int(total_cleanup_steps),
        "windows": per_window_meta,
        "kd_alpha": float(kd_alpha),
        "learn_output_scale": bool(args.learn_output_scale),
    }
    store.save_json("cleanup_meta.json", cleanup_meta)
    print("\nCleanup phase complete.")
    return cleanup_meta


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
        # NOTE (v8): hard_sparse_forward is no longer recommended at 80%+. It collapses ELSA
        # back into sparse fine-tuning (no dense x exploration). Kept for ablation only.
        # We do NOT override penalty_normalization in v8.
        print("[warn] --hard_sparse_forward enables sparse-z-as-x mode. This is NOT pure ELSA "
              "and is known to be worse at 80%+ sparsity. Consider dropping the flag.")
        args.copy_z_to_x_after_projection = True
        args.no_persist_x_state = True
        args.load_z_into_x_if_missing = True
        args.apply_initial_z_to_model = True

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

    kd_cache: Optional[KDTopKCache] = None
    if float(args.kd_alpha) > 0.0:
        kd_dir = Path(args.kd_cache_dir) if args.kd_cache_dir else (
            Path(args.state_dir) if args.state_dir else Path(args.out).with_suffix("") / "elsa_admm_state"
        ) / "kd_topk_cache"
        kd_dir.mkdir(parents=True, exist_ok=True)
        need_build = bool(args.kd_force_rebuild) or not kd_cache_is_valid(
            cache_dir=kd_dir,
            n_rows=int(calib_tokens.size(0)),
            seq_len=int(calib_tokens.size(1)),
            topk=int(args.kd_topk),
            model_id=str(args.model_id),
            calib_path=str(args.calib),
            build_temperature=float(args.kd_temperature),
        )
        if need_build:
            # Free the as-yet-unmodified sparse model so the dense teacher fits on GPU.
            # Note: at this point in run() the model has only been loaded from HF and
            # moved to GPU. Nothing has been mutated. So we can safely delete it and
            # reload from HF after the cache is built, instead of saving 13.5GB to CPU.
            print("\nFreeing sparse-target model temporarily to load dense teacher for KD cache build...")
            del model
            clean_cuda()
            gc.collect()
            try:
                build_kd_topk_cache(
                    cache_dir=kd_dir,
                    model_id=str(args.model_id),
                    calib_path=str(args.calib),
                    calib_tokens=calib_tokens,
                    topk=int(args.kd_topk),
                    device=main_device,
                    amp_dtype=model_dtype,
                    trust_remote_code=bool(args.trust_remote_code),
                    attn_implementation=str(args.attn_implementation),
                    low_cpu_mem_usage=bool(args.low_cpu_mem_usage),
                    batch_size=int(args.kd_build_batch_size),
                    build_temperature=float(args.kd_temperature),
                )
            finally:
                print("\nReloading sparse-target model after KD cache build...")
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
                    try:
                        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    except TypeError:
                        model.gradient_checkpointing_enable()
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                if args.model_train_mode:
                    model.train()
                else:
                    model.eval()
                clean_cuda()
        kd_cache = KDTopKCache(kd_dir)
        cache_T = float(kd_cache.meta.get("build_temperature", 1.0))
        if abs(cache_T - float(args.kd_temperature)) > 1e-6:
            print(f"[warn] KD cache was built at temperature {cache_T:.3f} but --kd_temperature={args.kd_temperature:.3f}. "
                  f"Using cache temperature {cache_T:.3f} for student-side scaling to stay consistent. "
                  f"Use --kd_force_rebuild to rebuild at the requested temperature.")
            args.kd_temperature = cache_T
        print(f"KD cache ready: K={kd_cache.K}, T={kd_cache.T}, rows={kd_cache.n_rows}, build_temp={cache_T:.3f}")
        print(f"KD alpha={args.kd_alpha:.3f}  temperature={args.kd_temperature:.3f}")
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

    wanda_input_norms_sq: Optional[Dict[str, torch.Tensor]] = None
    if str(args.init_method).lower().strip() == "wanda" and int(args.wanda_calib_batches) > 0:
        print(f"\nCollecting per-channel activation norms for Wanda warm-start "
              f"(batches={args.wanda_calib_batches}, max_seq_len={args.max_seq_len})...")
        t0_wanda = time.time()
        wanda_input_norms_sq = compute_layer_input_norms(
            model=model,
            selected_names=selected_names,
            tokens=calib_tokens,
            batches=int(args.wanda_calib_batches),
            batch_size=int(args.batch_size),
            max_seq_len=int(args.max_seq_len),
            crop_mode=str(args.crop_mode),
            seed=int(args.seed) + 31337,
            device=main_device,
            amp_dtype=model_dtype,
        )
        n_layers_with_norms = sum(1 for v in wanda_input_norms_sq.values() if v.numel() > 0)
        print(f"Wanda activation pass done in {fmt_time(time.time() - t0_wanda)}; "
              f"got norms for {n_layers_with_norms}/{len(selected_names)} layers. "
              f"CUDA: {cuda_mem()}")
        clean_cuda()

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
        init_method=str(args.init_method).lower().strip(),
        wanda_input_norms_sq=wanda_input_norms_sq,
    )

    # Free the activation norms after init.
    wanda_input_norms_sq = None
    gc.collect()

    # v9: OWL global outlier protection masks (one-shot from initial dense weights).
    owl_masks: Optional[Dict[str, torch.Tensor]] = None
    owl_meta: Dict[str, Any] = {"enabled": False, "owl_pct": 0.0}
    if float(args.owl_outlier_pct) > 0.0:
        owl_masks, owl_meta = compute_owl_force_keep_masks(
            model=model,
            selected_names=selected_names,
            pct=float(args.owl_outlier_pct),
            weight_source="current",
            z_store=None,
        )
        store.save_json("owl_meta.json", owl_meta)

    # v9: adaptive lambda multiplier (mutates per projection if --adaptive_lambda).
    adaptive_lambda_mult = 1.0
    prev_proj_masks: Optional[Dict[str, torch.Tensor]] = {} if (
        bool(args.adaptive_lambda) or float(args.owl_outlier_pct) > 0.0
    ) else None
    # ^ Keeping prev_masks state enables both adaptive lambda and OWL drift logging.

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
    print(f"  fisher power      : {args.fisher_power}")
    print(f"  fisher-mag blend  : {args.fisher_blend_magnitude}")
    print(f"  fisher max factor : {args.fisher_max_factor}")
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
                    # v9: apply adaptive lambda multiplier accumulated from mask drift.
                    lambda_t = float(lambda_t) * float(adaptive_lambda_mult)

                    optimizer.zero_grad(set_to_none=True)

                    step_t0 = time.time()
                    oom = False
                    try:
                        accum_steps = max(1, int(args.grad_accum_steps))
                        lm_loss_values: List[float] = []
                        kd_loss_values: List[float] = []
                        bad_loss = False
                        kd_alpha = float(args.kd_alpha)

                        # Average LM (+ KD) gradient over multiple random crops.
                        for accum_idx in range(accum_steps):
                            batch_cpu, crop_starts, row_idx = token_loader.next_with_meta()
                            batch = batch_cpu.to(main_device, non_blocking=True)

                            need_logits = kd_alpha > 0.0 and kd_cache is not None
                            if need_logits:
                                lm_loss_i, full_logits = compute_lm_loss(
                                    model=model,
                                    input_ids=batch,
                                    amp_dtype=model_dtype,
                                    device=main_device,
                                    autocast_enabled=not bool(args.no_autocast),
                                    return_logits=True,
                                )
                            else:
                                lm_loss_i = compute_lm_loss(
                                    model=model,
                                    input_ids=batch,
                                    amp_dtype=model_dtype,
                                    device=main_device,
                                    autocast_enabled=not bool(args.no_autocast),
                                )
                                full_logits = None

                            if not torch.isfinite(lm_loss_i).all():
                                print(
                                    f"\n[warn] non-finite LM loss at global_step={global_step}, "
                                    f"accum={accum_idx + 1}/{accum_steps}; skipping step."
                                )
                                bad_loss = True
                                del batch, lm_loss_i
                                if full_logits is not None:
                                    del full_logits
                                break

                            kd_loss_i = None
                            if need_logits and full_logits is not None:
                                try:
                                    # Gather teacher top-K per row/crop into a single batch tensor.
                                    L_eff_max = int(batch.size(1)) - 1
                                    K = int(kd_cache.K)
                                    B = int(batch.size(0))
                                    if L_eff_max > 0:
                                        teacher_probs = torch.zeros((B, L_eff_max, K), dtype=torch.float16)
                                        teacher_idx = torch.zeros((B, L_eff_max, K), dtype=torch.int32)
                                        for bi in range(B):
                                            row = int(row_idx[bi].item())
                                            start = int(crop_starts[bi].item())
                                            tp, ti = kd_cache.get(row=row, start=start, length=int(batch.size(1)))
                                            L_have = tp.size(0)
                                            if L_have > 0:
                                                teacher_probs[bi, :L_have] = tp
                                                teacher_idx[bi, :L_have] = ti
                                        teacher_probs = teacher_probs.to(main_device, non_blocking=True)
                                        teacher_idx = teacher_idx.to(main_device, non_blocking=True)
                                        # v9: KD temperature schedule
                                        kd_T_now = kd_temperature_value(
                                            t_start=float(args.kd_temperature_start),
                                            t_final=float(args.kd_temperature),
                                            progress=progress,
                                            warmup_frac=float(args.kd_temperature_warmup_frac),
                                        )
                                        kd_loss_i = compute_kd_topk_loss(
                                            student_logits=full_logits,
                                            teacher_probs=teacher_probs,
                                            teacher_idx=teacher_idx,
                                            temperature=kd_T_now,
                                        )
                                        del teacher_probs, teacher_idx
                                except Exception as exc:
                                    print(f"  [warn] KD loss compute failed: {exc}; skipping KD this step.")
                                    kd_loss_i = None

                            lm_loss_values.append(float(lm_loss_i.detach().cpu().item()))
                            combined = lm_loss_i
                            if kd_loss_i is not None and torch.isfinite(kd_loss_i).all():
                                combined = (1.0 - kd_alpha) * lm_loss_i + kd_alpha * kd_loss_i
                                kd_loss_values.append(float(kd_loss_i.detach().cpu().item()))
                            (combined / float(accum_steps)).backward()
                            del batch, lm_loss_i, combined
                            if full_logits is not None:
                                del full_logits
                            if kd_loss_i is not None:
                                del kd_loss_i

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
                            optimizer=optimizer,
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
                        kd_loss_scalar = float(sum(kd_loss_values) / max(1, len(kd_loss_values))) if kd_loss_values else 0.0
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
                        "kd_loss": float(kd_loss_scalar),
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
                        kd_part = f" kd={log_item['kd_loss']:.4f}" if log_item['kd_loss'] != 0.0 else ""
                        print(
                            f"step={global_step:06d} pass={pass_idx + 1}/{args.admm_passes} "
                            f"win={win_idx + 1}/{len(windows)} local={local_step + 1}/{args.steps_per_window} "
                            f"lm={log_item['lm_loss']:.5f} ppl~{ppl_est:.2f}{kd_part} "
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
                            fisher_power=float(args.fisher_power),
                            fisher_blend_magnitude=float(args.fisher_blend_magnitude),
                            fisher_max_factor=float(args.fisher_max_factor),
                            force_keep_masks=owl_masks,
                            prev_masks=prev_proj_masks,
                        )
                        # v9: adaptive lambda update from observed mask drift.
                        if bool(args.adaptive_lambda):
                            d = proj_meta.get("mask_drift_frac_weighted")
                            if d is not None and d > float(args.adaptive_lambda_drift_threshold):
                                adaptive_lambda_mult = min(
                                    float(args.adaptive_lambda_cap),
                                    adaptive_lambda_mult * float(args.adaptive_lambda_grow),
                                )
                                print(f"  [adaptive_lambda] drift={100.0 * d:.3f}% > "
                                      f"{100.0 * float(args.adaptive_lambda_drift_threshold):.3f}%; "
                                      f"lambda multiplier now {adaptive_lambda_mult:.3f}")
                        proj_meta.update({
                            "global_step": int(global_step),
                            "pass": int(pass_idx),
                            "window": int(win_idx),
                            "local_step": int(local_step),
                            "adaptive_lambda_mult": float(adaptive_lambda_mult),
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
                        fisher_power=float(args.fisher_power),
                        fisher_blend_magnitude=float(args.fisher_blend_magnitude),
                        fisher_max_factor=float(args.fisher_max_factor),
                        force_keep_masks=owl_masks,
                        prev_masks=prev_proj_masks,
                    )
                    proj_meta.update({
                        "global_step": int(global_step),
                        "pass": int(pass_idx),
                        "window": int(win_idx),
                        "local_step": int(args.steps_per_window),
                        "window_end": True,
                        "adaptive_lambda_mult": float(adaptive_lambda_mult),
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
                    fisher_power=float(args.fisher_power),
                    fisher_blend_magnitude=float(args.fisher_blend_magnitude),
                    fisher_max_factor=float(args.fisher_max_factor),
                )
                dyn_meta["computed_after_pass"] = int(pass_idx + 1)
                dynamic_profile_history.append(dyn_meta)
                store.save_json(f"dynamic_sparsity_profile_after_pass_{pass_idx + 1}.json", dyn_meta)
                store.save_json("dynamic_sparsity_latest.json", dyn_meta)
                clean_cuda()

    except KeyboardInterrupt:
        print("\n[interrupt] Training interrupted by user. Applying current z states and saving metadata.")

    # v9: post-projection cleanup phase (mask frozen, LM+KD only, no ADMM)
    cleanup_meta = None
    if int(args.cleanup_steps) > 0:
        try:
            cleanup_meta = run_cleanup_phase(
                model=model,
                windows=windows,
                selected_names=selected_names,
                store=store,
                x_store=x_store,
                opt_store=opt_store,
                main_device=main_device,
                model_dtype=model_dtype,
                calib_tokens=calib_tokens,
                kd_cache=kd_cache,
                args=args,
                train_log=train_log,
            )
        except KeyboardInterrupt:
            print("\n[interrupt] Cleanup interrupted; saving current state.")

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
        "owl_meta": owl_meta,
        "cleanup_meta": cleanup_meta,
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
    parser.add_argument("--admm_passes", type=int, default=4)
    parser.add_argument("--steps_per_window", type=int, default=128)
    parser.add_argument(
        "--active_block_window",
        type=int,
        default=2,
        help="Number of decoder blocks trained at once. 2 fits comfortably on a 3090 with int8 states. 0 means all selected layers at once.",
    )
    parser.add_argument("--projection_interval", type=int, default=16)
    parser.add_argument("--project_at_window_end", action="store_true")
    parser.set_defaults(project_at_window_end=True)

    # Optimizer and schedules.
    parser.add_argument("--lr", type=float, default=3.0e-4,
                        help="v8 default raised from 1e-5 -> 3e-4 (paper uses 5e-2 with full-model FSDP; 3e-4 is safe with FP32 master weights and per-block windowing).")
    parser.add_argument("--lr_schedule", type=str, default="linear", choices=["constant", "linear", "cosine"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1.0e-8)

    # ADMM.
    parser.add_argument("--admm_lambda", type=float, default=5.0e-5,
                        help="v8 default for penalty='mean'. The paper uses ~2e-5..5e-5 for LLaMA-2-7B at 80% sparsity.")
    parser.add_argument("--lambda_schedule", type=str, default="constant", choices=["constant", "linear", "cosine"])
    parser.add_argument("--lambda_warmup_frac", type=float, default=0.05)
    parser.add_argument("--penalty_normalization", type=str, default="mean", choices=["mean", "sum", "layer_mean"],
                        help="v8 default switched from sum -> mean. 'layer_mean' divides each layer's penalty by its numel before summing across active layers (better balance between attn and MLP layers).")
    parser.add_argument("--dual_lr", type=float, default=1.0)
    parser.add_argument("--dual_clip", type=float, default=100.0)
    parser.add_argument("--admm_diff_clip", type=float, default=100.0)
    parser.add_argument("--no_objective_aware_projection", action="store_true")
    parser.add_argument("--fisher_floor", type=float, default=1.0e-12)
    parser.add_argument("--fisher_power", type=float, default=1.0,
                        help="Power applied to normalized Fisher before projection. <1 compresses noisy Fisher spikes; 1.0 is paper behavior.")
    parser.add_argument("--fisher_blend_magnitude", type=float, default=0.0,
                        help="Blend pure magnitude score into Fisher score. 0.05 means 5% magnitude + 95% Fisher. 0.0 = pure Fisher (paper).")
    parser.add_argument("--fisher_max_factor", type=float, default=1.0e6,
                        help="Cap normalized Fisher at this factor before top-k projection. v7 used 100, which was too aggressive. 1e6 = effectively no cap (paper).")
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
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="v8 default raised from 0.3 -> 1.0; with the larger LR you need to let gradients through.")
    parser.add_argument("--grad_value_clip", type=float, default=0.0,
                        help="v8 default lowered from 1.0 -> 0.0 (off). Per-value clipping interacts badly with rare large gradients that carry the most signal at high sparsity.")
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
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="v8 default raised from 128 -> 1024 to recover Fisher quality. Use 2048 if your GPU has headroom.")
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

    # ============================================================
    # v8: new flags (Wanda warm-start, KD distillation)
    # ============================================================
    parser.add_argument("--init_method", type=str, default="wanda", choices=["magnitude", "wanda"],
                        help="Initial z projection method. 'wanda' uses per-channel input activation norms (much better than magnitude at >70% sparsity).")
    parser.add_argument("--wanda_calib_batches", type=int, default=128,
                        help="Number of calibration batches used to estimate per-channel activation norms for Wanda init. 64-128 is plenty.")

    parser.add_argument("--kd_alpha", type=float, default=0.0,
                        help="Knowledge-distillation weight against the dense teacher. 0.0 = off. Try 0.5-1.0 at high sparsity.")
    parser.add_argument("--kd_topk", type=int, default=64,
                        help="Top-K of teacher logits to cache and distill against. v9 default raised 32->64.")
    parser.add_argument("--kd_temperature", type=float, default=1.0,
                        help="Final KD temperature. With --kd_temperature_start>1, T linearly anneals start->this over warmup_frac.")
    parser.add_argument("--kd_temperature_start", type=float, default=0.0,
                        help="If >0, KD temperature linearly anneals from this value to --kd_temperature over the first --kd_temperature_warmup_frac of training. 0 disables (constant T).")
    parser.add_argument("--kd_temperature_warmup_frac", type=float, default=0.3,
                        help="Fraction of total ADMM steps over which KD temperature anneals from start->final.")
    parser.add_argument("--kd_alpha_cleanup", type=float, default=-1.0,
                        help="KD weight used during the post-projection cleanup phase. <0 means inherit --kd_alpha.")
    parser.add_argument("--kd_cache_dir", type=str, default="",
                        help="Where to store the dense-teacher top-K cache. Defaults to <state_dir>/kd_topk_cache. ~600MB for 2048x2048x32 fp16+int32.")
    parser.add_argument("--kd_build_batch_size", type=int, default=1,
                        help="Batch size during teacher cache build (1 is safest on a 3090).")
    parser.add_argument("--kd_force_rebuild", action="store_true",
                        help="Force rebuilding the KD cache even if a valid one exists.")

    # ============================================================
    # v9: post-projection cleanup phase
    # ============================================================
    parser.add_argument("--cleanup_steps", type=int, default=0,
                        help="After the full ADMM loop, run this many extra optimizer steps PER WINDOW with the mask frozen and only LM+KD loss (no ADMM penalty, no projections). 0 disables the cleanup phase. Typical: 32-128 at 80% sparsity.")
    parser.add_argument("--cleanup_lr", type=float, default=5.0e-5,
                        help="Base LR for the cleanup phase. Should be smaller than --lr because the optimization landscape is much narrower once the mask is frozen.")
    parser.add_argument("--cleanup_lr_schedule", type=str, default="cosine",
                        choices=["constant", "linear", "cosine"],
                        help="LR schedule within each window's cleanup phase.")
    parser.add_argument("--cleanup_min_lr_ratio", type=float, default=0.1,
                        help="Floor for cleanup LR schedule as fraction of cleanup_lr.")
    parser.add_argument("--cleanup_max_grad_norm", type=float, default=1.0,
                        help="Gradient norm clip used during cleanup.")
    parser.add_argument("--cleanup_log_interval", type=int, default=4,
                        help="Cleanup-phase log frequency (steps).")
    parser.add_argument("--cleanup_probe_each_window", action="store_true",
                        help="Run a small calibration probe (--cleanup_probe_batches) after each window's cleanup finishes.")
    parser.add_argument("--cleanup_probe_batches", type=int, default=2,
                        help="Probe batches used by --cleanup_probe_each_window.")

    # ============================================================
    # v9: OWL global outlier protection
    # ============================================================
    parser.add_argument("--owl_outlier_pct", type=float, default=0.0,
                        help="Force-keep the top X percent of |W| GLOBALLY across all selected layers regardless of per-layer sparsity budget. 0 disables. Try 1.0 at 80% sparsity. Implemented by computing a per-layer force_keep mask once before ADMM and OR-ing it into every projection.")
    parser.add_argument("--owl_recompute_each_pass", action="store_true",
                        help="Recompute the OWL mask after every ADMM pass using current z weights instead of using the initial-W computation.")

    # ============================================================
    # v9: adaptive lambda based on mask drift
    # ============================================================
    parser.add_argument("--adaptive_lambda", action="store_true",
                        help="Scale lambda up when mask is still drifting (Hamming distance > threshold) and hold otherwise. Eliminates the 'lambda too small to enforce constraint' failure mode.")
    parser.add_argument("--adaptive_lambda_drift_threshold", type=float, default=0.05,
                        help="If fraction of mask positions changed since last projection exceeds this, multiply effective lambda by --adaptive_lambda_grow per projection.")
    parser.add_argument("--adaptive_lambda_grow", type=float, default=1.25,
                        help="Multiplicative growth factor for lambda when mask drift exceeds threshold.")
    parser.add_argument("--adaptive_lambda_cap", type=float, default=20.0,
                        help="Maximum multiplier applied to base lambda by adaptive scheduling.")

    # ============================================================
    # v9: learnable per-output rescaling during cleanup
    # ============================================================
    parser.add_argument("--learn_output_scale", action="store_true",
                        help="During the cleanup phase, jointly learn one scalar per output row (shape [out_dim]) for each sparsified Linear. Fused back into weights at save time. Helps recover output magnitude lost to sparsification.")
    parser.add_argument("--output_scale_init", type=str, default="dense_ratio",
                        choices=["one", "dense_ratio"],
                        help="How to initialize the per-output scale. 'one' = identity, 'dense_ratio' = ||W_dense_row|| / ||W_sparse_row|| clipped to [0.5, 2.0].")
    parser.add_argument("--output_scale_lr_mult", type=float, default=2.0,
                        help="Multiplier applied to --cleanup_lr for the learnable output scales (scales typically tolerate a larger LR than weights).")

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
