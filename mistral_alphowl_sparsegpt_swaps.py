#!/usr/bin/env python3
"""
Mistral/LLaMA/Qwen blockwise high-sparsity pruning without GPTQ.

Method:
  1. Dense calibration prepass:
       - collect OWL-style layer outlier ratios
       - estimate AlphaPruning PL_Alpha_Hill weight-spectrum metrics
       - allocate non-uniform sparsity per layer/block

  2. Blockwise pruning:
       - collect Hessian / Gram G = X^T X for each selected Linear layer
       - build row-balanced SparseGPT warm-start mask
       - refine mask with SparseSwaps-style 1-swaps
       - run fixed-mask SparseGPT/OBS reconstruction
       - propagate compressed block outputs to next block

No quantization is performed.

Designed for:
  - mistralai/Mistral-7B-Instruct-v0.3
  - LLaMA-style decoder-only HF models
  - Qwen2.5-style decoder-only HF models

Example:
  python mistral_alphowl_sparsegpt_swaps.py \
    --model_id mistralai/Mistral-7B-Instruct-v0.3 \
    --calib data/calib_wikitext103_train_128x1024_mistral.pt \
    --out compressed/mistral_s70_alphowl_swaps_sparse.pt \
    --target_sparsity 0.70 \
    --batch_size 1 \
    --model_dtype float16 \
    --hidden_cache_dtype float16 \
    --hessian_dtype float32 \
    --max_seq_len 1024 \
    --percdamp 0.05 \
    --blocksize 128 \
    --mask_score sparsegpt \
    --swap_iters 10 \
    --swap_candidates 64 \
    --tau 0.18 \
    --alpha_weight 0.55 \
    --owl_weight 0.45 \
    --mixed_block_matrix \
    --suffix_delta_strength 0.35 \
    --attn_implementation eager \
    --keep_dequantized_state_dict

For a faster first run:
  --swap_iters 3 --swap_candidates 32

For a stronger but slower run:
  --swap_iters 25 --swap_candidates 96
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Basic utilities
# =============================================================================

def now() -> float:
    return time.time()


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


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_mem() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
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


def sanitize_tensor(
    x: torch.Tensor,
    clamp_abs: Optional[float] = None,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    if clamp_abs is not None and clamp_abs > 0:
        x = x.clamp(min=-float(clamp_abs), max=float(clamp_abs))
    return x


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


# =============================================================================
# HF model helpers
# =============================================================================

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
            first = mod[0].__class__.__name__.lower()
            if "decoder" in first or "layer" in first or "block" in first:
                return name, mod

    raise RuntimeError("Could not find decoder block ModuleList.")


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


def layer_suffix(name: str) -> str:
    return name.split(".")[-1]


def should_compress_name(
    name: str,
    include: str,
    exclude: str,
    suffixes: Tuple[str, ...],
    compress_lm_head: bool,
    skip_tied_lm_head: bool,
    tied_lm_head: bool,
) -> bool:
    if include and include not in name:
        return False
    if exclude and exclude in name:
        return False
    if name == "lm_head":
        if skip_tied_lm_head and tied_lm_head:
            return False
        return compress_lm_head
    return name.endswith(suffixes)


def find_selected_linear_names(
    model: nn.Module,
    include: str,
    exclude: str,
    suffixes: Tuple[str, ...],
    compress_lm_head: bool,
    skip_tied_lm_head: bool,
) -> List[str]:
    tied = model_has_tied_lm_head_hf(model)
    out: List[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if should_compress_name(
            name=name,
            include=include,
            exclude=exclude,
            suffixes=suffixes,
            compress_lm_head=compress_lm_head,
            skip_tied_lm_head=skip_tied_lm_head,
            tied_lm_head=tied,
        ):
            out.append(name)
    return out


def block_linear_items(
    block: nn.Module,
    block_prefix: str,
    selected_set: Set[str],
) -> List[Tuple[str, nn.Linear]]:
    out: List[Tuple[str, nn.Linear]] = []
    for subname, mod in block.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        full_name = f"{block_prefix}.{subname}" if subname else block_prefix
        if full_name in selected_set:
            out.append((full_name, mod))
    return out


# =============================================================================
# Decoder block forward helpers
# =============================================================================

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
    mask = torch.full((seq_len, seq_len), fill_value=mask_value, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask.expand(batch_size, 1, seq_len, seq_len)


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
    position_embeddings = maybe_make_position_embeddings(backbone, hidden_states, position_ids)
    causal_mask = make_4d_causal_attention_mask(batch_size, seq_len, dtype, device)

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
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if torch.is_tensor(out):
        return out
    raise RuntimeError(f"Unsupported decoder block output type: {type(out)}")


@torch.no_grad()
def compute_initial_hidden_cache(
    model: nn.Module,
    calib_tokens: torch.Tensor,
    batch_size: int,
    main_device: torch.device,
    storage_dtype: torch.dtype,
) -> torch.Tensor:
    emb = get_embedding_layer(model)
    emb.eval()

    outs: List[torch.Tensor] = []
    n = calib_tokens.size(0)
    t0 = now()
    print("\nComputing initial embedding hidden cache...")

    for i in range(0, n, batch_size):
        input_ids = calib_tokens[i:i + batch_size].to(main_device)
        h = emb(input_ids)
        h = sanitize_tensor(h, clamp_abs=1.0e4)
        outs.append(h.detach().to("cpu", dtype=storage_dtype))
        done = min(i + batch_size, n)
        print(
            f"\r embeddings: {done}/{n} ({100.0 * done / n:.1f}%) "
            f"elapsed={fmt_time(now() - t0)}",
            end="",
            flush=True,
        )
    print()

    hidden = torch.cat(outs, dim=0).contiguous()
    print(f"Initial hidden cache: shape={tuple(hidden.shape)} dtype={hidden.dtype}")
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
    t0 = now()
    autocast_enabled = main_device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16)

    for i in range(0, n, batch_size):
        h = hidden_cache[i:i + batch_size].to(main_device, dtype=amp_dtype)
        with torch.autocast(device_type=main_device.type, dtype=amp_dtype, enabled=autocast_enabled):
            out = call_decoder_block(block, h, backbone)

        if not torch.isfinite(out).all():
            bad = torch.isfinite(out).logical_not().sum().item()
            total = out.numel()
            print(f"\n[warn] Non-finite output in {desc}: {bad:,}/{total:,}; sanitizing.")
            out = sanitize_tensor(out, clamp_abs=1.0e4)

        outs.append(out.detach().to("cpu", dtype=storage_dtype))
        done = min(i + batch_size, n)
        print(
            f"\r {desc}: {done}/{n} ({100.0 * done / n:.1f}%) "
            f"elapsed={fmt_time(now() - t0)}",
            end="",
            flush=True,
        )
    print()

    return torch.cat(outs, dim=0).contiguous()


# =============================================================================
# Mask packing
# =============================================================================

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
    shifts = torch.arange(8, dtype=torch.uint8)
    bits = ((packed.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 1).bool()
    return bits.view(rows, packed_cols * 8)[:, :original_cols].contiguous()


# =============================================================================
# Allocation statistics: OWL + AlphaPruning
# =============================================================================

@dataclass
class LayerAllocationStat:
    name: str
    block_idx: int
    suffix: str
    shape: Tuple[int, int]
    nparams: int
    owl_outlier_ratio: float
    alpha_hill: float


class InputSecondMomentCollector:
    """
    Collects sum_j x_j^2 for a Linear layer input.
    Used for OWL-style outlier score:
        A_ij = |W_ij| * ||X_j||_2
    """

    def __init__(self, layer: nn.Linear, name: str):
        self.layer = layer
        self.name = name
        self.in_features = int(layer.in_features)
        self.x2sum = torch.zeros(self.in_features, dtype=torch.float64, device="cpu")
        self.nsamples = 0
        self.handle = None
        self.warned_nonfinite = False

    def _hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        x = x.detach().reshape(-1, x.size(-1))

        if not torch.isfinite(x).all():
            if not self.warned_nonfinite:
                bad = torch.isfinite(x).logical_not().sum().item()
                total = x.numel()
                print(f"\n[warn] Non-finite activation entering {self.name}: {bad:,}/{total:,}; sanitizing.")
                self.warned_nonfinite = True
            x = sanitize_tensor(x, clamp_abs=1.0e4)

        x_cpu = x.float().to("cpu")
        self.x2sum += (x_cpu * x_cpu).sum(dim=0, dtype=torch.float64)
        self.nsamples += x_cpu.size(0)

    def register(self) -> None:
        self.handle = self.layer.register_forward_pre_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def compute_owl_outlier_ratio(
    W: torch.Tensor,
    col_l2: torch.Tensor,
    outlier_multiplier: float,
    row_chunk: int = 1024,
) -> float:
    """
    OWL-style Layerwise Outlier Distribution.

    score_ij = |W_ij| * ||X_j||_2
    outlier_ratio = mean(score_ij > M * mean(score))
    """
    device = W.device
    col_l2_dev = col_l2.to(device=device, dtype=torch.float32).clamp_min(1.0e-12)

    rows = W.size(0)
    cols = W.size(1)
    total_elems = rows * cols

    total_sum = 0.0
    for r0 in range(0, rows, row_chunk):
        r1 = min(r0 + row_chunk, rows)
        chunk = W[r0:r1].detach().float()
        score = chunk.abs() * col_l2_dev.view(1, -1)
        score = sanitize_tensor(score)
        total_sum += float(score.sum().item())

    mean_score = total_sum / max(total_elems, 1)
    if not math.isfinite(mean_score) or mean_score <= 0:
        return 0.0

    threshold = float(outlier_multiplier) * mean_score
    out_count = 0
    for r0 in range(0, rows, row_chunk):
        r1 = min(r0 + row_chunk, rows)
        chunk = W[r0:r1].detach().float()
        score = sanitize_tensor(chunk.abs() * col_l2_dev.view(1, -1))
        out_count += int((score > threshold).sum().item())

    return out_count / float(total_elems)


@torch.no_grad()
def estimate_pl_alpha_hill(
    W: torch.Tensor,
    rank: int = 256,
    niter: int = 2,
    tail_fraction: float = 0.25,
    device: Optional[torch.device] = None,
    eps: float = 1.0e-12,
) -> float:
    """
    Approximate PL_Alpha_Hill from the top singular-value tail.

    Full ESD estimation is too expensive for all Mistral matrices on 3090/32GB,
    so this uses randomized low-rank SVD to estimate the upper tail.

    Lower alpha_hill => heavier-tailed => more important => less sparsity.
    """
    try:
        if device is None:
            device = W.device

        Wf = W.detach().to(device=device, dtype=torch.float32)
        Wf = sanitize_tensor(Wf)

        min_dim = min(Wf.shape)
        if min_dim < 16:
            return 2.0

        q = min(int(rank), min_dim - 1)
        q = max(q, 16)

        try:
            # Randomized SVD. This is much cheaper than full SVD.
            _U, S, _V = torch.svd_lowrank(Wf, q=q, niter=int(niter))
        except Exception:
            # Fallback: if the matrix is small enough, exact SVD; otherwise sampled columns.
            if min_dim <= 2048:
                S = torch.linalg.svdvals(Wf)
            else:
                gen = torch.Generator(device=Wf.device)
                gen.manual_seed(1234)
                if Wf.size(1) >= Wf.size(0):
                    idx = torch.randperm(Wf.size(1), device=Wf.device, generator=gen)[:min(Wf.size(1), 4096)]
                    S = torch.linalg.svdvals(Wf[:, idx])
                else:
                    idx = torch.randperm(Wf.size(0), device=Wf.device, generator=gen)[:min(Wf.size(0), 4096)]
                    S = torch.linalg.svdvals(Wf[idx, :])

        lambdas = (S.float() ** 2).clamp_min(eps)
        vals = torch.sort(lambdas, descending=True).values
        n = vals.numel()
        if n < 8:
            return 2.0

        k = int(round(float(tail_fraction) * n))
        k = max(4, min(k, n - 1))

        threshold = vals[k].clamp_min(eps)
        denom = torch.log((vals[:k] / threshold).clamp_min(1.0 + eps)).sum().item()

        if not math.isfinite(denom) or denom <= eps:
            return 10.0

        alpha = 1.0 + float(k) / denom
        if not math.isfinite(alpha):
            return 10.0
        return float(max(0.5, min(alpha, 20.0)))

    except Exception as exc:
        print(f"[warn] Alpha metric failed for shape={tuple(W.shape)}: {type(exc).__name__}: {exc}")
        return 5.0


def normalize_01(values: Sequence[float]) -> List[float]:
    vals = [float(v) for v in values]
    lo = min(vals)
    hi = max(vals)
    if hi - lo < 1.0e-12:
        return [0.5 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


@torch.no_grad()
def collect_allocation_stats_prepass(
    model: nn.Module,
    calib_tokens: torch.Tensor,
    selected_layer_names: List[str],
    batch_size: int,
    main_device: torch.device,
    model_dtype: torch.dtype,
    hidden_cache_dtype: torch.dtype,
    alpha_rank: int,
    alpha_niter: int,
    alpha_device: torch.device,
    owl_outlier_multiplier: float,
) -> List[LayerAllocationStat]:
    """
    Dense prepass for non-uniform sparsity allocation.

    It collects only cheap x^2 statistics, not full Hessians.
    """
    selected_set = set(selected_layer_names)
    layers_prefix, decoder_layers = find_decoder_layers(model)
    backbone = get_backbone(model)

    hidden_cache = compute_initial_hidden_cache(
        model=model,
        calib_tokens=calib_tokens,
        batch_size=batch_size,
        main_device=main_device,
        storage_dtype=hidden_cache_dtype,
    )

    stats: List[LayerAllocationStat] = []

    print("\n" + "=" * 100)
    print("Dense allocation-statistics prepass: OWL outlier ratios + AlphaPruning spectra")
    print("=" * 100)

    for block_idx, block in enumerate(decoder_layers):
        block_prefix = f"{layers_prefix}.{block_idx}"
        items = block_linear_items(block, block_prefix, selected_set)

        print("\n" + "-" * 100)
        print(f"[allocation prepass] block {block_idx}/{len(decoder_layers) - 1}; selected linears={len(items)}")

        collectors: Dict[str, InputSecondMomentCollector] = {}
        for name, layer in items:
            collectors[name] = InputSecondMomentCollector(layer, name)
            collectors[name].register()

        hidden_cache = run_block_to_cache(
            block=block,
            backbone=backbone,
            hidden_cache=hidden_cache,
            batch_size=batch_size,
            main_device=main_device,
            amp_dtype=model_dtype,
            storage_dtype=hidden_cache_dtype,
            desc=f"allocation dense block {block_idx}",
        )

        for col in collectors.values():
            col.remove()

        for name, layer in items:
            col = collectors[name]
            if col.nsamples <= 0:
                print(f"[warn] no activation samples for {name}; setting OWL ratio to 0.")
                col_l2 = torch.ones(layer.in_features, dtype=torch.float32)
            else:
                col_l2 = torch.sqrt(col.x2sum.float().clamp_min(1.0e-12))

            print(f"  metric {name}: shape={tuple(layer.weight.shape)}")
            t_metric = now()

            owl = compute_owl_outlier_ratio(
                W=layer.weight.detach(),
                col_l2=col_l2,
                outlier_multiplier=owl_outlier_multiplier,
                row_chunk=1024,
            )
            alpha = estimate_pl_alpha_hill(
                W=layer.weight.detach(),
                rank=alpha_rank,
                niter=alpha_niter,
                tail_fraction=0.25,
                device=alpha_device,
            )

            print(
                f"    OWL ratio={owl:.6e}  "
                f"PL_Alpha_Hill≈{alpha:.4f}  "
                f"time={fmt_time(now() - t_metric)}"
            )

            stats.append(
                LayerAllocationStat(
                    name=name,
                    block_idx=block_idx,
                    suffix=layer_suffix(name),
                    shape=tuple(layer.weight.shape),
                    nparams=int(layer.weight.numel()),
                    owl_outlier_ratio=float(owl),
                    alpha_hill=float(alpha),
                )
            )

        del collectors
        cleanup()

    return stats


def default_suffix_deltas() -> Dict[str, float]:
    """
    Small Mistral-friendly intra-block bias.

    Negative = protect layer.
    Positive = prune more.

    This is intentionally mild. The main allocation still comes from Alpha/OWL.
    """
    return {
        "q_proj": -0.045,
        "k_proj": -0.045,
        "v_proj": -0.020,
        "o_proj": -0.060,
        "gate_proj": +0.045,
        "up_proj": +0.045,
        "down_proj": -0.030,
    }


def solve_offset_for_target(
    raw: Dict[str, float],
    nparams: Dict[str, int],
    target: float,
    min_sparsity: float,
    max_sparsity: float,
) -> Dict[str, float]:
    total = sum(nparams.values())
    if total <= 0:
        raise ValueError("No parameters to allocate.")

    min_possible = sum(min_sparsity * n for n in nparams.values()) / total
    max_possible = sum(max_sparsity * n for n in nparams.values()) / total

    if target < min_possible - 1.0e-9 or target > max_possible + 1.0e-9:
        raise ValueError(
            f"Target sparsity {target} infeasible with caps "
            f"[{min_sparsity}, {max_sparsity}]. Feasible average range: "
            f"[{min_possible}, {max_possible}]"
        )

    def avg_with_offset(offset: float) -> float:
        return sum(
            max(min_sparsity, min(max_sparsity, raw[name] + offset)) * nparams[name]
            for name in raw
        ) / total

    lo = -1.0
    hi = +1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        avg = avg_with_offset(mid)
        if avg < target:
            lo = mid
        else:
            hi = mid

    off = 0.5 * (lo + hi)
    return {
        name: float(max(min_sparsity, min(max_sparsity, raw[name] + off)))
        for name in raw
    }


def allocate_nonuniform_sparsities(
    stats: List[LayerAllocationStat],
    target_sparsity: float,
    tau: float,
    alpha_weight: float,
    owl_weight: float,
    min_sparsity: float,
    max_sparsity: float,
    mixed_block_matrix: bool,
    suffix_delta_strength: float,
) -> Dict[str, float]:
    """
    Alpha/OWL mixed allocation.

    Alpha part:
      lower PL_Alpha_Hill => more important => lower sparsity.

    OWL part:
      higher outlier ratio => more important => lower sparsity.

    The final prunability score is normalized and mapped into a bounded
    sparsity range, then shifted to exactly match the global target.
    """
    if not stats:
        raise ValueError("No allocation stats.")

    names = [s.name for s in stats]
    nparams = {s.name: s.nparams for s in stats}

    # ----- Per-block metrics
    block_ids = sorted(set(s.block_idx for s in stats))
    block_alpha: Dict[int, float] = {}
    block_owl: Dict[int, float] = {}
    block_params: Dict[int, int] = {}

    for b in block_ids:
        ss = [s for s in stats if s.block_idx == b]
        p_total = sum(s.nparams for s in ss)
        block_params[b] = p_total
        block_alpha[b] = sum(s.alpha_hill * s.nparams for s in ss) / max(p_total, 1)
        block_owl[b] = sum(s.owl_outlier_ratio * s.nparams for s in ss) / max(p_total, 1)

    norm_block_alpha_vals = normalize_01([block_alpha[b] for b in block_ids])
    norm_block_owl_vals = normalize_01([block_owl[b] for b in block_ids])
    norm_block_alpha = {b: norm_block_alpha_vals[i] for i, b in enumerate(block_ids)}
    norm_block_owl = {b: norm_block_owl_vals[i] for i, b in enumerate(block_ids)}

    # alpha high => less heavy-tailed => more prunable
    # owl high => more outlier-heavy => less prunable
    block_prunability_raw: Dict[int, float] = {}
    for b in block_ids:
        a = norm_block_alpha[b]
        o = 1.0 - norm_block_owl[b]
        denom = max(alpha_weight + owl_weight, 1.0e-12)
        block_prunability_raw[b] = (alpha_weight * a + owl_weight * o) / denom

    norm_block_prunability_vals = normalize_01([block_prunability_raw[b] for b in block_ids])
    block_prunability = {b: norm_block_prunability_vals[i] for i, b in enumerate(block_ids)}

    # ----- Optional per-matrix refinement inside each block
    layer_alpha_norm_all = {s.name: v for s, v in zip(stats, normalize_01([s.alpha_hill for s in stats]))}
    layer_owl_norm_all = {s.name: v for s, v in zip(stats, normalize_01([s.owl_outlier_ratio for s in stats]))}

    suffix_deltas = default_suffix_deltas()

    raw: Dict[str, float] = {}
    for s in stats:
        # Base block allocation around target:
        # prunability=0 -> target*(1 - tau)
        # prunability=1 -> target*(1 + tau)
        p_block = block_prunability[s.block_idx]
        base = target_sparsity * (1.0 + float(tau) * (2.0 * p_block - 1.0))

        if mixed_block_matrix:
            a = layer_alpha_norm_all[s.name]
            o = 1.0 - layer_owl_norm_all[s.name]
            denom = max(alpha_weight + owl_weight, 1.0e-12)
            p_layer = (alpha_weight * a + owl_weight * o) / denom
            # Mild refinement. Keeps Alpha's high-sparsity per-block behavior
            # while letting q/k/o/down get protected if their own metrics say so.
            base += 0.5 * target_sparsity * float(tau) * (2.0 * (p_layer - 0.5))

        if suffix_delta_strength > 0:
            base += float(suffix_delta_strength) * suffix_deltas.get(s.suffix, 0.0)

        raw[s.name] = float(base)

    allocated = solve_offset_for_target(
        raw=raw,
        nparams=nparams,
        target=target_sparsity,
        min_sparsity=min_sparsity,
        max_sparsity=max_sparsity,
    )

    total = sum(nparams.values())
    actual = sum(allocated[n] * nparams[n] for n in names) / total

    print("\n" + "=" * 100)
    print("Final non-uniform sparsity allocation")
    print("=" * 100)
    print(f"Target weighted sparsity: {100.0 * target_sparsity:.4f}%")
    print(f"Actual weighted sparsity: {100.0 * actual:.4f}%")
    print(f"tau={tau}, alpha_weight={alpha_weight}, owl_weight={owl_weight}")
    print(f"mixed_block_matrix={mixed_block_matrix}, suffix_delta_strength={suffix_delta_strength}")
    print()

    for s in stats:
        print(
            f"{s.name:<60s} "
            f"s={100.0 * allocated[s.name]:6.2f}% "
            f"alpha={s.alpha_hill:7.3f} "
            f"owl={s.owl_outlier_ratio:10.4e} "
            f"shape={s.shape}"
        )

    return allocated


# =============================================================================
# Hessian collection
# =============================================================================

def choose_hessian_device(
    layer: nn.Linear,
    main_device: torch.device,
    large_layer_cpu_threshold: int,
) -> torch.device:
    if large_layer_cpu_threshold <= 0:
        return torch.device("cpu")
    if int(layer.in_features) > int(large_layer_cpu_threshold):
        return torch.device("cpu")
    return main_device


class HessianCollector:
    """
    Collects H = 2 X^T X for one Linear layer.
    """

    def __init__(
        self,
        layer: nn.Linear,
        hessian_device: torch.device,
        dtype: torch.dtype,
        name: str,
        activation_clamp: float = 1.0e4,
    ):
        self.layer = layer
        self.hessian_device = hessian_device
        self.dtype = dtype
        self.name = name
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
                print(f"\n[warn] Non-finite activation entering {self.name}: {bad:,}/{total:,}; sanitizing.")
                self.warned_nonfinite_input = True
            x = sanitize_tensor(x, clamp_abs=self.activation_clamp)

        x = x.to(device=self.hessian_device, dtype=self.dtype)
        local_H = 2.0 * x.T.matmul(x)

        if not torch.isfinite(local_H).all():
            if not self.warned_nonfinite_hessian:
                bad = torch.isfinite(local_H).logical_not().sum().item()
                total = local_H.numel()
                print(
                    f"\n[warn] Non-finite local Hessian for {self.name}: "
                    f"{bad:,}/{total:,}; skipping this local contribution."
                )
                self.warned_nonfinite_hessian = True
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
    block_linear_items_: List[Tuple[str, nn.Linear]],
    hidden_cache: torch.Tensor,
    batch_size: int,
    main_device: torch.device,
    hessian_dtype: torch.dtype,
    amp_dtype: torch.dtype,
    large_layer_cpu_threshold: int,
) -> Dict[str, Tuple[torch.Tensor, int, torch.device]]:
    collectors: Dict[str, HessianCollector] = {}

    for name, layer in block_linear_items_:
        hdev = choose_hessian_device(layer, main_device, large_layer_cpu_threshold)
        collectors[name] = HessianCollector(
            layer=layer,
            hessian_device=hdev,
            dtype=hessian_dtype,
            name=name,
        )

    for c in collectors.values():
        c.register()

    n = hidden_cache.size(0)
    t0 = now()
    autocast_enabled = main_device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16)

    print(f" Collecting Hessians for {len(block_linear_items_)} linears...")

    try:
        for i in range(0, n, batch_size):
            h = hidden_cache[i:i + batch_size].to(main_device, dtype=amp_dtype)
            with torch.autocast(device_type=main_device.type, dtype=amp_dtype, enabled=autocast_enabled):
                out = call_decoder_block(block, h, backbone)

            if not torch.isfinite(out).all():
                bad = torch.isfinite(out).logical_not().sum().item()
                total = out.numel()
                print(f"\n[warn] Non-finite block output during Hessian pass: {bad:,}/{total:,}")

            done = min(i + batch_size, n)
            print(
                f"\r block Hessian pass: {done}/{n} ({100.0 * done / n:.1f}%) "
                f"elapsed={fmt_time(now() - t0)}",
                end="",
                flush=True,
            )
        print()
    finally:
        for c in collectors.values():
            c.remove()

    out: Dict[str, Tuple[torch.Tensor, int, torch.device]] = {}
    for name, c in collectors.items():
        out[name] = (c.H, c.nsamples, c.hessian_device)
    return out


# =============================================================================
# Stable Hessian inverse
# =============================================================================

@torch.no_grad()
def stable_hessian_inverse(
    H: torch.Tensor,
    percdamp: float,
    force_float64: bool = True,
    max_tries: int = 14,
    eig_fallback_max_dim: int = 8192,
) -> Tuple[torch.Tensor, float]:
    """
    Robust inverse of damped H.

    Returns:
      Hinv float32 on H.device
      used_damp scalar
    """
    if H.ndim != 2 or H.size(0) != H.size(1):
        raise ValueError(f"H must be square, got {tuple(H.shape)}")

    device = H.device
    n = H.size(0)

    H = sanitize_tensor(H)
    dtype = torch.float64 if force_float64 else torch.float32
    H_work = H.to(dtype=dtype)
    H_work = 0.5 * (H_work + H_work.T)

    ar = torch.arange(n, device=device)
    diag = torch.diag(H_work)
    diag = sanitize_tensor(diag)
    H_work[ar, ar] = diag

    diag_abs = torch.diag(H_work).abs()
    diag_mean = max(float(diag_abs.mean().item()), 1.0e-12)
    diag_max = max(float(diag_abs.max().item()), diag_mean, 1.0e-12)
    base = max(diag_mean, 1.0e-6 * diag_max, 1.0e-12)

    multipliers = [
        1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
        200.0, 500.0, 1000.0, 2000.0, 5000.0,
        10000.0, 20000.0,
    ][:max_tries]

    last_exc: Optional[Exception] = None

    for mult in multipliers:
        used_damp = float(percdamp) * float(mult) * base
        H_try = H_work.clone()
        H_try[ar, ar] += used_damp

        try:
            chol = torch.linalg.cholesky(H_try)
            Hinv = torch.cholesky_inverse(chol)
            Hinv = 0.5 * (Hinv + Hinv.T)
            Hinv = sanitize_tensor(Hinv).to(torch.float32)
            return Hinv, used_damp
        except RuntimeError as exc:
            last_exc = exc
            del H_try
            cleanup()

    if n <= eig_fallback_max_dim:
        print("[warn] Cholesky retries failed; using eigenvalue fallback.")
        evals = torch.linalg.eigvalsh(H_work)
        min_eval = float(evals.min().item())
        max_eval = float(evals.abs().max().item())
        shift = max(0.0, -min_eval) + max(float(percdamp) * base, 1.0e-6 * max(max_eval, 1.0))
        H_try = H_work.clone()
        H_try[ar, ar] += shift
        chol = torch.linalg.cholesky(H_try)
        Hinv = torch.cholesky_inverse(chol)
        Hinv = 0.5 * (Hinv + Hinv.T)
        return sanitize_tensor(Hinv).to(torch.float32), float(shift)

    raise RuntimeError(
        f"Cholesky failed for dim={n} after damping retries. "
        f"Last error: {last_exc}. Try larger --percdamp, e.g. 0.1."
    )


# =============================================================================
# Mask generation and SparseSwaps refinement
# =============================================================================

@torch.no_grad()
def row_balanced_topk_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Per-output-row balanced mask.

    Every row keeps exactly round((1-s) * cols) weights.
    """
    score = sanitize_tensor(score)
    rows, cols = score.shape
    keep = int(round((1.0 - float(sparsity)) * cols))
    keep = max(1, min(cols, keep))

    if keep >= cols:
        return torch.ones((rows, cols), dtype=torch.bool, device=score.device)

    idx = torch.topk(score, k=keep, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros((rows, cols), dtype=torch.bool, device=score.device)
    row_idx = torch.arange(rows, device=score.device).view(-1, 1).expand_as(idx)
    mask[row_idx, idx] = True
    return mask


@torch.no_grad()
def make_warmstart_mask(
    W: torch.Tensor,
    Hinv_diag: torch.Tensor,
    H_diag: torch.Tensor,
    sparsity: float,
    mask_score: str,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """
    Warm-start mask.

    sparsegpt score:
        score_ij = W_ij^2 / Hinv_jj

    wanda_diag score:
        score_ij = |W_ij| * sqrt(H_jj / 2)
    """
    mask_score = mask_score.lower().strip()
    if mask_score == "sparsegpt":
        denom = Hinv_diag.abs().clamp_min(eps).view(1, -1)
        score = (W.float() ** 2) / denom
    elif mask_score == "wanda_diag":
        xnorm = torch.sqrt((H_diag.float() / 2.0).clamp_min(eps)).view(1, -1)
        score = W.float().abs() * xnorm
    elif mask_score == "magnitude":
        score = W.float().abs()
    else:
        raise ValueError(f"Unknown --mask_score: {mask_score}")

    return row_balanced_topk_mask(score, sparsity=sparsity)


@torch.no_grad()
def sparse_swaps_refine_mask(
    W: torch.Tensor,
    G: torch.Tensor,
    mask: torch.Tensor,
    max_iters: int,
    candidates: int,
    eps: float,
    row_print_every: int = 2048,
) -> torch.Tensor:
    """
    SparseSwaps-style row-wise 1-swap local search.

    For each row:
      residual r = sum_{p pruned} w_p phi_p
      c_j = <phi_j, r> = [G @ ((1-m) * w)]_j

    Swap p from pruned -> kept, u from kept -> pruned:
      r' = r - w_p phi_p + w_u phi_u

      ΔL =
        -2 w_p c_p + w_p^2 G_pp
        +2 w_u c_u + w_u^2 G_uu
        -2 w_p w_u G_pu

    This implementation restricts the exact pair search to the best candidate
    p/u pools for speed. Within the candidate pools, the ΔL calculation is exact.
    """
    if max_iters <= 0:
        return mask

    W = sanitize_tensor(W.float())
    G = sanitize_tensor(G.float())
    mask = mask.bool().clone()

    rows, cols = W.shape
    candidates = int(max(1, min(candidates, cols)))

    Gdiag = torch.diag(G).float().clamp_min(1.0e-12)

    # Precompute initial c for all rows:
    # C = (W * pruned_mask) @ G
    # This is memory-heavy but much faster than per-row recomputation.
    print(f"  SparseSwaps: computing initial C = residual_weights @ G, rows={rows}, cols={cols}")
    residual_weights = torch.where(mask, torch.zeros_like(W), W)
    C = residual_weights.matmul(G)
    C = sanitize_tensor(C)

    accepted_total = 0
    t0 = now()

    for i in range(rows):
        w = W[i]
        m = mask[i]
        c = C[i]

        for _it in range(max_iters):
            pruned = (~m).nonzero(as_tuple=False).flatten()
            kept = m.nonzero(as_tuple=False).flatten()

            if pruned.numel() == 0 or kept.numel() == 0:
                break

            # Candidate p: currently pruned, try to unprune/remove from residual.
            wp_all = w[pruned]
            cp_all = c[pruned]
            remove_delta = -2.0 * wp_all * cp_all + (wp_all * wp_all) * Gdiag[pruned]

            kp = min(candidates, pruned.numel())
            p_local = torch.topk(remove_delta, k=kp, largest=False, sorted=False).indices
            p_idx = pruned[p_local]
            p_delta = remove_delta[p_local]
            wp = w[p_idx]

            # Candidate u: currently kept, try to prune/add to residual.
            wu_all = w[kept]
            cu_all = c[kept]
            add_delta = 2.0 * wu_all * cu_all + (wu_all * wu_all) * Gdiag[kept]

            ku = min(candidates, kept.numel())
            u_local = torch.topk(add_delta, k=ku, largest=False, sorted=False).indices
            u_idx = kept[u_local]
            u_delta = add_delta[u_local]
            wu = w[u_idx]

            # Exact pair delta inside candidate pools.
            # delta[p,u] = remove_delta[p] + add_delta[u] - 2 wp wu G[p,u]
            G_pu = G[p_idx][:, u_idx]
            delta = (
                p_delta.view(-1, 1)
                + u_delta.view(1, -1)
                - 2.0 * (wp.view(-1, 1) * wu.view(1, -1) * G_pu)
            )
            delta = sanitize_tensor(delta)

            best_val, flat_idx = torch.min(delta.reshape(-1), dim=0)
            best = float(best_val.item())

            if not math.isfinite(best) or best >= -float(eps):
                break

            p_pos = int(flat_idx.item() // ku)
            u_pos = int(flat_idx.item() % ku)
            p = int(p_idx[p_pos].item())
            u = int(u_idx[u_pos].item())

            wp_scalar = w[p].clone()
            wu_scalar = w[u].clone()

            # Perform swap.
            m[p] = True
            m[u] = False

            # c <- c + w_u G[:,u] - w_p G[:,p]
            c.add_(G[:, u], alpha=float(wu_scalar.item()))
            c.add_(G[:, p], alpha=-float(wp_scalar.item()))

            accepted_total += 1

        mask[i] = m

        if row_print_every > 0 and (i + 1) % row_print_every == 0:
            print(
                f"\r  SparseSwaps rows {i + 1}/{rows}; "
                f"accepted={accepted_total:,}; elapsed={fmt_time(now() - t0)}",
                end="",
                flush=True,
            )

    print(
        f"\r  SparseSwaps rows {rows}/{rows}; "
        f"accepted={accepted_total:,}; elapsed={fmt_time(now() - t0)}"
    )

    return mask


# =============================================================================
# Fixed-mask SparseGPT / OBS reconstruction
# =============================================================================

@torch.no_grad()
def fixed_mask_obs_reconstruction(
    W: torch.Tensor,
    Hinv: torch.Tensor,
    mask: torch.Tensor,
    blocksize: int,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """
    SparseGPT/OBS reconstruction with fixed mask.

    The mask is never changed. Only surviving weights are updated via OBS-style
    error compensation.
    """
    W = sanitize_tensor(W.float()).clone()
    Hinv = sanitize_tensor(Hinv.float())
    mask = mask.bool()

    rows, cols = W.shape
    Q = torch.zeros_like(W)

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2].contiguous()

        for local_i in range(count):
            global_col = i1 + local_i
            d = Hinv1[local_i, local_i].abs().clamp_min(eps)

            w = W1[:, local_i]
            keep = mask[:, global_col]
            q = torch.where(keep, w, torch.zeros_like(w))

            Q1[:, local_i] = q
            Q[:, global_col] = q

            err = (w - q) / d
            err = sanitize_tensor(err, clamp_abs=1.0e8)
            Err1[:, local_i] = err

            if local_i + 1 < count:
                W1[:, local_i + 1:count] -= (
                    err.unsqueeze(1)
                    @ Hinv1[local_i, local_i + 1:count].unsqueeze(0)
                )
                W1[:, local_i + 1:count] = sanitize_tensor(W1[:, local_i + 1:count], clamp_abs=1.0e8)

        W[:, i1:i2] = Q1

        if i2 < cols:
            W[:, i2:cols] -= Err1.matmul(Hinv[i1:i2, i2:cols])
            W[:, i2:cols] = sanitize_tensor(W[:, i2:cols], clamp_abs=1.0e8)

    Q = sanitize_tensor(Q * mask.to(Q.dtype))
    return Q


@dataclass
class SparseLayerResult:
    mask: torch.Tensor
    dense_sparse_weight: torch.Tensor
    original_shape: Tuple[int, int]
    sparsity: float
    target_sparsity: float
    pruned_count: int
    total_count: int
    swaps_enabled: bool
    swap_iters: int
    swap_candidates: int
    used_damp: float


@torch.no_grad()
def compress_linear_alphowl_sparsegpt_swaps(
    layer: nn.Linear,
    H: torch.Tensor,
    target_sparsity: float,
    percdamp: float,
    blocksize: int,
    mask_score: str,
    act_order: bool,
    swap_iters: int,
    swap_candidates: int,
    swap_eps: float,
    compress_device: torch.device,
    inverse_float64: bool,
    eig_fallback_max_dim: int,
) -> SparseLayerResult:
    """
    One Linear layer:
      1. H inverse
      2. row-balanced SparseGPT/Wanda warm-start mask
      3. SparseSwaps mask refinement
      4. fixed-mask OBS reconstruction
    """
    original_device = layer.weight.device
    original_dtype = layer.weight.dtype

    W_orig = layer.weight.detach().to(device=compress_device, dtype=torch.float32).clone()
    W_orig = sanitize_tensor(W_orig)

    rows, cols = W_orig.shape

    if H.shape != (cols, cols):
        raise ValueError(f"H shape mismatch for layer. Expected {(cols, cols)}, got {tuple(H.shape)}")

    H = H.to(compress_device, dtype=torch.float32)
    H = sanitize_tensor(H)
    H = 0.5 * (H + H.T)

    # Dead-input safeguard.
    diagH = torch.diag(H)
    dead = diagH <= 1.0e-12
    if dead.any():
        n_dead = int(dead.sum().item())
        print(f"  [warn] {n_dead} dead/near-dead Hessian columns; zeroing corresponding weights.")
        W_orig[:, dead] = 0.0
        ar_dead = dead.nonzero(as_tuple=False).flatten()
        H[ar_dead, :] = 0.0
        H[:, ar_dead] = 0.0
        H[ar_dead, ar_dead] = 1.0

    t_inv = now()
    Hinv, used_damp = stable_hessian_inverse(
        H=H,
        percdamp=percdamp,
        force_float64=inverse_float64,
        eig_fallback_max_dim=eig_fallback_max_dim,
    )
    print(f"  H inverse: used_damp={used_damp:.6e}, time={fmt_time(now() - t_inv)}")

    G = (H * 0.5).float()
    H_diag = torch.diag(H).float()
    Hinv_diag = torch.diag(Hinv).float().abs().clamp_min(1.0e-12)

    if act_order:
        perm = torch.argsort(H_diag.abs(), descending=True)
        invperm = torch.argsort(perm)
        W = W_orig[:, perm].contiguous()
        G = G[perm][:, perm].contiguous()
        Hinv = Hinv[perm][:, perm].contiguous()
        H_diag = H_diag[perm].contiguous()
        Hinv_diag = Hinv_diag[perm].contiguous()
    else:
        invperm = None
        W = W_orig

    print(f"  target sparsity={100.0 * target_sparsity:.2f}%")
    print(f"  mask_score={mask_score}")
    print(f"  row-balanced keep/row={int(round((1.0 - target_sparsity) * cols))}/{cols}")

    t_mask = now()
    mask = make_warmstart_mask(
        W=W,
        Hinv_diag=Hinv_diag,
        H_diag=H_diag,
        sparsity=target_sparsity,
        mask_score=mask_score,
    )
    print(f"  warm-start mask time={fmt_time(now() - t_mask)}")

    if swap_iters > 0:
        t_swaps = now()
        mask = sparse_swaps_refine_mask(
            W=W,
            G=G,
            mask=mask,
            max_iters=swap_iters,
            candidates=swap_candidates,
            eps=swap_eps,
        )
        print(f"  SparseSwaps total time={fmt_time(now() - t_swaps)}")

    t_obs = now()
    Q = fixed_mask_obs_reconstruction(
        W=W,
        Hinv=Hinv,
        mask=mask,
        blocksize=blocksize,
    )
    print(f"  fixed-mask OBS time={fmt_time(now() - t_obs)}")

    if act_order:
        assert invperm is not None
        Q = Q[:, invperm].contiguous()
        mask = mask[:, invperm].contiguous()

    Q = sanitize_tensor(Q * mask.to(Q.dtype))

    layer.weight.data.copy_(Q.to(device=original_device, dtype=original_dtype))

    total = rows * cols
    kept = int(mask.sum().item())
    pruned = total - kept
    actual = pruned / float(total)

    return SparseLayerResult(
        mask=mask.detach().cpu(),
        dense_sparse_weight=Q.detach().cpu(),
        original_shape=(rows, cols),
        sparsity=float(actual),
        target_sparsity=float(target_sparsity),
        pruned_count=int(pruned),
        total_count=int(total),
        swaps_enabled=bool(swap_iters > 0),
        swap_iters=int(swap_iters),
        swap_candidates=int(swap_candidates),
        used_damp=float(used_damp),
    )


# =============================================================================
# Checkpoint helpers
# =============================================================================

def get_sparse_weight_keys(sparse_layers: Dict[str, Dict[str, Any]]) -> Set[str]:
    return {f"{name}.weight" for name in sparse_layers.keys()}


def build_partial_noncompressed_state_dict(
    model: nn.Module,
    sparse_layers: Dict[str, Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    sparse_weight_keys = get_sparse_weight_keys(sparse_layers)
    out: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if k in sparse_weight_keys:
            continue
        out[k] = v.detach().cpu()
    return out


def save_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    out_path: str,
    model_id: str,
    meta: Dict[str, Any],
    sparse_layers: Dict[str, Dict[str, Any]],
    keep_dequantized_state_dict: bool,
) -> None:
    if keep_dequantized_state_dict:
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    else:
        model_state = build_partial_noncompressed_state_dict(model, sparse_layers)

    ckpt = {
        "format": "hf_alphowl_sparsegpt_sparseswaps",
        "model_id": model_id,
        "config": model.config.to_dict() if hasattr(model, "config") else None,
        "model": model_state,
        "alphowl_sparse_layers": sparse_layers,
        "alphowl_sparse_meta": meta,
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
    print(f"Saved checkpoint in {fmt_time(now() - t0)}: {out}")

    meta_path = Path(str(out) + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta JSON: {meta_path}")


# =============================================================================
# Blockwise compression
# =============================================================================

@torch.no_grad()
def compress_model_blockwise(
    model: nn.Module,
    tokenizer: Any,
    calib_tokens: torch.Tensor,
    selected_layer_names: List[str],
    layer_sparsities: Dict[str, float],
    args: argparse.Namespace,
    main_device: torch.device,
    model_dtype: torch.dtype,
    hidden_cache_dtype: torch.dtype,
    hessian_dtype: torch.dtype,
    script_t0: float,
) -> Dict[str, Dict[str, Any]]:
    selected_set = set(selected_layer_names)
    layers_prefix, decoder_layers = find_decoder_layers(model)
    backbone = get_backbone(model)

    print("\n" + "=" * 100)
    print("Blockwise Alpha/OWL + SparseGPT + SparseSwaps pruning")
    print("=" * 100)
    print(f"Decoder layers path: {layers_prefix}")
    print(f"Number of blocks: {len(decoder_layers)}")

    hidden_cache = compute_initial_hidden_cache(
        model=model,
        calib_tokens=calib_tokens,
        batch_size=args.batch_size,
        main_device=main_device,
        storage_dtype=hidden_cache_dtype,
    )

    sparse_layers: Dict[str, Dict[str, Any]] = {}

    for block_idx, block in enumerate(decoder_layers):
        block_prefix = f"{layers_prefix}.{block_idx}"
        items = block_linear_items(block, block_prefix, selected_set)

        print("\n" + "=" * 100)
        print(f"BLOCK {block_idx}/{len(decoder_layers) - 1}")
        print(f"Selected linears: {len(items)}")
        print(f"Hidden cache entering block: shape={tuple(hidden_cache.shape)} dtype={hidden_cache.dtype}")
        print(f"CUDA memory: {cuda_mem()}")

        for name, layer in items:
            print(
                f" - {name}: shape={tuple(layer.weight.shape)} "
                f"target_sparsity={100.0 * layer_sparsities[name]:.2f}%"
            )

        if not items:
            hidden_cache = run_block_to_cache(
                block=block,
                backbone=backbone,
                hidden_cache=hidden_cache,
                batch_size=args.batch_size,
                main_device=main_device,
                amp_dtype=model_dtype,
                storage_dtype=hidden_cache_dtype,
                desc=f"block {block_idx} dense forward",
            )
            continue

        hessians = collect_block_hessians_once(
            block=block,
            backbone=backbone,
            block_linear_items_=items,
            hidden_cache=hidden_cache,
            batch_size=args.batch_size,
            main_device=main_device,
            hessian_dtype=hessian_dtype,
            amp_dtype=model_dtype,
            large_layer_cpu_threshold=args.large_layer_cpu_threshold,
        )

        for local_idx, (layer_name, layer) in enumerate(items, start=1):
            print("\n" + "-" * 100)
            print(f"[{local_idx}/{len(items)}] Compressing {layer_name}")
            print(f" shape={tuple(layer.weight.shape)} dtype={layer.weight.dtype}")

            H, nsamples, hdev = hessians[layer_name]
            print(f" H shape={tuple(H.shape)} samples={nsamples} device={hdev}")

            if nsamples <= 0:
                raise RuntimeError(f"No Hessian samples collected for {layer_name}.")

            compress_device = hdev
            t_layer = now()

            result = compress_linear_alphowl_sparsegpt_swaps(
                layer=layer,
                H=H,
                target_sparsity=float(layer_sparsities[layer_name]),
                percdamp=float(args.percdamp),
                blocksize=int(args.blocksize),
                mask_score=str(args.mask_score),
                act_order=bool(args.act_order),
                swap_iters=int(args.swap_iters),
                swap_candidates=int(args.swap_candidates),
                swap_eps=float(args.swap_eps),
                compress_device=compress_device,
                inverse_float64=bool(args.inverse_float64),
                eig_fallback_max_dim=int(args.eig_fallback_max_dim),
            )

            mask_packed = pack_bool_mask_rows(result.mask)
            dense_sparse = result.dense_sparse_weight.cpu()

            kept_values = dense_sparse[result.mask.bool()].to(dtype=parse_dtype(args.value_dtype)).contiguous()

            layer_state: Dict[str, Any] = {
                "shape": list(result.original_shape),
                "mask": mask_packed,
                "mask_packing": "packedbits",
                "values": kept_values,
                "values_format": "kept_1d_row_major",
                "value_dtype": str(args.value_dtype),
                "sparsity": float(result.sparsity),
                "target_sparsity": float(result.target_sparsity),
                "pruned_count": int(result.pruned_count),
                "total_count": int(result.total_count),
                "method": "alphowl_rowbalanced_sparsegpt_warmstart_sparseswaps_fixedmask_obs",
                "swaps_enabled": bool(result.swaps_enabled),
                "swap_iters": int(result.swap_iters),
                "swap_candidates": int(result.swap_candidates),
                "used_damp": float(result.used_damp),
            }

            if args.store_debug_dense_weight:
                layer_state["dense_sparse_weight"] = dense_sparse.to(torch.float16)

            sparse_layers[layer_name] = layer_state

            print(
                f" saved layer: actual_sparsity={100.0 * result.sparsity:.2f}% "
                f"values={tuple(kept_values.shape)} "
                f"mask_packed={tuple(mask_packed.shape)} "
                f"time={fmt_time(now() - t_layer)}"
            )

            del H
            del hessians[layer_name]
            del result
            cleanup()

        print("\n Running compressed block to create next hidden cache...")
        hidden_cache = run_block_to_cache(
            block=block,
            backbone=backbone,
            hidden_cache=hidden_cache,
            batch_size=args.batch_size,
            main_device=main_device,
            amp_dtype=model_dtype,
            storage_dtype=hidden_cache_dtype,
            desc=f"block {block_idx} compressed forward",
        )

        if args.save_partial_every_block:
            partial_path = str(args.out) + ".partial.pt"
            meta_partial = build_meta(args, sparse_layers, layer_sparsities, script_t0, partial=True, completed_block=block_idx)
            print(f"\nSaving partial checkpoint after block {block_idx}: {partial_path}")
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                out_path=partial_path,
                model_id=str(args.model_id),
                meta=meta_partial,
                sparse_layers=sparse_layers,
                keep_dequantized_state_dict=False,
            )

        cleanup()
        print(f"Finished block {block_idx}. Total elapsed={fmt_time(now() - script_t0)}")
        print(f"CUDA memory: {cuda_mem()}")

    return sparse_layers


def build_meta(
    args: argparse.Namespace,
    sparse_layers: Dict[str, Dict[str, Any]],
    layer_sparsities: Dict[str, float],
    script_t0: float,
    partial: bool = False,
    completed_block: Optional[int] = None,
) -> Dict[str, Any]:
    total_pruned = sum(int(st["pruned_count"]) for st in sparse_layers.values())
    total_weights = sum(int(st["total_count"]) for st in sparse_layers.values())
    actual = total_pruned / float(total_weights) if total_weights else 0.0

    mask_bytes = sum(int(st["mask"].numel() * st["mask"].element_size()) for st in sparse_layers.values())
    values_bytes = sum(int(st["values"].numel() * st["values"].element_size()) for st in sparse_layers.values())

    meta: Dict[str, Any] = {
        "method": "alphowl_rowbalanced_sparsegpt_sparseswaps_fixedmask_obs_no_quant",
        "model_id": str(args.model_id),
        "target_sparsity": float(args.target_sparsity),
        "actual_total_sparsity": float(actual),
        "total_pruned_weights": int(total_pruned),
        "total_compressed_weights": int(total_weights),
        "percdamp": float(args.percdamp),
        "blocksize": int(args.blocksize),
        "mask_score": str(args.mask_score),
        "act_order": bool(args.act_order),
        "swap_iters": int(args.swap_iters),
        "swap_candidates": int(args.swap_candidates),
        "swap_eps": float(args.swap_eps),
        "tau": float(args.tau),
        "alpha_weight": float(args.alpha_weight),
        "owl_weight": float(args.owl_weight),
        "mixed_block_matrix": bool(args.mixed_block_matrix),
        "suffix_delta_strength": float(args.suffix_delta_strength),
        "min_sparsity": float(args.min_sparsity),
        "max_sparsity": float(args.max_sparsity),
        "alpha_rank": int(args.alpha_rank),
        "alpha_niter": int(args.alpha_niter),
        "owl_outlier_multiplier": float(args.owl_outlier_multiplier),
        "model_dtype": str(args.model_dtype),
        "hidden_cache_dtype": str(args.hidden_cache_dtype),
        "hessian_dtype": str(args.hessian_dtype),
        "value_dtype": str(args.value_dtype),
        "max_seq_len": int(args.max_seq_len),
        "calibration_source": str(args.calib),
        "large_layer_cpu_threshold": int(args.large_layer_cpu_threshold),
        "layer_sparsities": {k: float(v) for k, v in layer_sparsities.items()},
        "mask_bytes": int(mask_bytes),
        "values_bytes": int(values_bytes),
        "raw_stored_bytes": int(mask_bytes + values_bytes),
        "script_seconds": float(now() - script_t0),
        "checkpoint_note": (
            "If keep_dequantized_state_dict=True, ckpt['model'] contains full dense tensors "
            "with zeros at pruned positions. Otherwise compressed weights are stored as "
            "packed mask + kept values in ckpt['alphowl_sparse_layers']."
        ),
        "partial": bool(partial),
    }
    if completed_block is not None:
        meta["completed_block"] = int(completed_block)
    return meta


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    # Model / IO
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="eager")

    # Compression target
    parser.add_argument("--target_sparsity", type=float, default=0.70)
    parser.add_argument("--min_sparsity", type=float, default=0.05)
    parser.add_argument("--max_sparsity", type=float, default=0.92)

    # Data / dtype
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hidden_cache_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hessian_dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--value_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max_seq_len", type=int, default=1024)

    # Layer selection
    parser.add_argument("--suffixes", type=str, default="")
    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")
    parser.add_argument("--compress_lm_head", action="store_true")
    parser.add_argument("--skip_tied_lm_head", action="store_true")

    # Allocation
    parser.add_argument("--tau", type=float, default=0.18)
    parser.add_argument("--alpha_weight", type=float, default=0.55)
    parser.add_argument("--owl_weight", type=float, default=0.45)
    parser.add_argument("--mixed_block_matrix", action="store_true")
    parser.add_argument("--suffix_delta_strength", type=float, default=0.35)
    parser.add_argument("--alpha_rank", type=int, default=256)
    parser.add_argument("--alpha_niter", type=int, default=2)
    parser.add_argument("--alpha_device", type=str, default="cuda")
    parser.add_argument("--owl_outlier_multiplier", type=float, default=5.0)

    # SparseGPT / OBS
    parser.add_argument("--percdamp", type=float, default=0.05)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--mask_score", type=str, default="sparsegpt", choices=["sparsegpt", "wanda_diag", "magnitude"])
    parser.add_argument("--act_order", action="store_true")
    parser.add_argument("--inverse_float64", action="store_true", default=True)
    parser.add_argument("--no_inverse_float64", dest="inverse_float64", action="store_false")
    parser.add_argument("--eig_fallback_max_dim", type=int, default=8192)
    parser.add_argument(
        "--large_layer_cpu_threshold",
        type=int,
        default=8192,
        help="If in_features exceeds this threshold, Hessian and compression math use CPU. Use 0 to force CPU.",
    )

    # SparseSwaps
    parser.add_argument("--swap_iters", type=int, default=10)
    parser.add_argument("--swap_candidates", type=int, default=64)
    parser.add_argument("--swap_eps", type=float, default=1.0e-10)

    # Saving
    parser.add_argument("--keep_dequantized_state_dict", action="store_true")
    parser.add_argument("--store_debug_dense_weight", action="store_true")
    parser.add_argument("--save_partial_every_block", action="store_true", default=True)
    parser.add_argument("--no_save_partial_every_block", dest="save_partial_every_block", action="store_false")
    parser.add_argument("--save_hf_dir", type=str, default="")

    args = parser.parse_args()

    script_t0 = now()

    if not (0.0 <= args.target_sparsity < 1.0):
        raise ValueError("--target_sparsity must be in [0, 1).")
    if not (0.0 <= args.min_sparsity <= args.max_sparsity < 1.0):
        raise ValueError("Need 0 <= --min_sparsity <= --max_sparsity < 1.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    main_device = torch.device(args.device)
    model_dtype = parse_dtype(args.model_dtype)
    hidden_cache_dtype = parse_dtype(args.hidden_cache_dtype)
    hessian_dtype = parse_dtype(args.hessian_dtype)
    alpha_device = torch.device(args.alpha_device if torch.cuda.is_available() or args.alpha_device == "cpu" else "cpu")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("=" * 100)
    print("Mistral Alpha/OWL + row-balanced SparseGPT + SparseSwaps + fixed-mask OBS")
    print("=" * 100)
    print(f"model_id: {args.model_id}")
    print(f"calib: {args.calib}")
    print(f"out: {args.out}")
    print(f"device: {main_device}")
    print(f"model_dtype: {model_dtype}")
    print(f"hidden_cache_dtype: {hidden_cache_dtype}")
    print(f"hessian_dtype: {hessian_dtype}")
    print(f"target_sparsity: {args.target_sparsity}")
    print(f"swap_iters: {args.swap_iters}")
    print(f"swap_candidates: {args.swap_candidates}")
    print(f"CUDA memory: {cuda_mem()}")

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
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()
    model.to(main_device)

    print(f"Loaded model in {fmt_time(now() - t0)}")
    print(f"CUDA memory after load: {cuda_mem()}")

    calib_tokens = load_calibration_tokens(args.calib)
    print(f"Loaded calibration tokens: {tuple(calib_tokens.shape)}")

    if args.max_seq_len > 0 and calib_tokens.size(1) > args.max_seq_len:
        calib_tokens = calib_tokens[:, :args.max_seq_len].contiguous()
        print(f"Trimmed calibration sequence length to --max_seq_len={args.max_seq_len}")

    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
        max_pos = int(model.config.max_position_embeddings)
        if calib_tokens.size(1) > max_pos:
            calib_tokens = calib_tokens[:, :max_pos].contiguous()
            print(f"Trimmed calibration sequence length to model max_position_embeddings={max_pos}")

    suffixes = parse_suffixes(args.suffixes)
    selected_layer_names = find_selected_linear_names(
        model=model,
        include=str(args.include),
        exclude=str(args.exclude),
        suffixes=suffixes,
        compress_lm_head=bool(args.compress_lm_head),
        skip_tied_lm_head=bool(args.skip_tied_lm_head),
    )

    if not selected_layer_names:
        raise RuntimeError("No Linear layers selected for compression.")

    print(f"\nSelected Linear layers: {len(selected_layer_names)}")
    for name in selected_layer_names:
        mod = get_module_by_name(model, name)
        print(f" - {name}: {tuple(mod.weight.shape)}")

    # -------------------------------------------------------------------------
    # >>> CRUCIAL PART 1:
    # Non-uniform allocation prepass using AlphaPruning + OWL metrics.
    # -------------------------------------------------------------------------
    allocation_stats = collect_allocation_stats_prepass(
        model=model,
        calib_tokens=calib_tokens,
        selected_layer_names=selected_layer_names,
        batch_size=int(args.batch_size),
        main_device=main_device,
        model_dtype=model_dtype,
        hidden_cache_dtype=hidden_cache_dtype,
        alpha_rank=int(args.alpha_rank),
        alpha_niter=int(args.alpha_niter),
        alpha_device=alpha_device,
        owl_outlier_multiplier=float(args.owl_outlier_multiplier),
    )

    layer_sparsities = allocate_nonuniform_sparsities(
        stats=allocation_stats,
        target_sparsity=float(args.target_sparsity),
        tau=float(args.tau),
        alpha_weight=float(args.alpha_weight),
        owl_weight=float(args.owl_weight),
        min_sparsity=float(args.min_sparsity),
        max_sparsity=float(args.max_sparsity),
        mixed_block_matrix=bool(args.mixed_block_matrix),
        suffix_delta_strength=float(args.suffix_delta_strength),
    )

    # -------------------------------------------------------------------------
    # >>> CRUCIAL PART 2:
    # Blockwise prune with row-balanced warm-start + SparseSwaps + fixed-mask OBS.
    # -------------------------------------------------------------------------
    try:
        sparse_layers = compress_model_blockwise(
            model=model,
            tokenizer=tokenizer,
            calib_tokens=calib_tokens,
            selected_layer_names=selected_layer_names,
            layer_sparsities=layer_sparsities,
            args=args,
            main_device=main_device,
            model_dtype=model_dtype,
            hidden_cache_dtype=hidden_cache_dtype,
            hessian_dtype=hessian_dtype,
            script_t0=script_t0,
        )
    except Exception as exc:
        print("\n[error] Compression failed.")
        print(f"[error] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise

    meta = build_meta(
        args=args,
        sparse_layers=sparse_layers,
        layer_sparsities=layer_sparsities,
        script_t0=script_t0,
        partial=False,
    )

    print("\nSaving final checkpoint...")
    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        out_path=str(args.out),
        model_id=str(args.model_id),
        meta=meta,
        sparse_layers=sparse_layers,
        keep_dequantized_state_dict=bool(args.keep_dequantized_state_dict),
    )

    if args.save_hf_dir:
        print(f"\nSaving full dense-zero HF model to: {args.save_hf_dir}")
        Path(args.save_hf_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_hf_dir)
        tokenizer.save_pretrained(args.save_hf_dir)

    print("\nDone.")
    print(f"Compressed layers: {len(sparse_layers)}")
    print(f"Total selected weights: {meta['total_compressed_weights']:,}")
    print(f"Pruned weights: {meta['total_pruned_weights']:,}")
    print(f"Actual sparsity: {100.0 * meta['actual_total_sparsity']:.4f}%")
    print(f"Mask bytes: {meta['mask_bytes']:,}")
    print(f"Values bytes: {meta['values_bytes']:,}")
    print(f"Raw stored bytes: {meta['raw_stored_bytes']:,}")
    print(f"Total script time: {fmt_time(now() - script_t0)}")
    print(f"CUDA memory: {cuda_mem()}")


if __name__ == "__main__":
    main()