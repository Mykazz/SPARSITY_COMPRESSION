#!/usr/bin/env python3
"""
alpha_oats_sparse_lowrank_hf.py

Blockwise Hugging Face compression for decoder-only LLMs, designed for high-sparsity
Mistral/LLaMA/Qwen experiments on a single RTX 3090-class GPU.

Main idea:
    1. AlphaPruning-like non-uniform per-layer compression allocation.
    2. OATS-like outlier-aware sparse + low-rank decomposition:
           W_c = (S_scaled + L_scaled) D^{-1}
       where D = sqrt(diag(X^T X)).
    3. Optional SparseSwaps-style mask refinement with the Gram matrix G = X^T X.
    4. Blockwise hidden-cache propagation, so later blocks see compressed previous blocks.
    5. Saves both:
           - factorized sparse+lowrank state
           - optionally dense dequantized state_dict for immediate eval.

This is not a custom-kernel runtime. It is a mathematical compression/evaluation script.

Recommended first runs:

70% effective compression, likely should beat your pure 70% sparse+GPTQ PPL:
    python alpha_oats_sparse_lowrank_hf.py \\
      --model_id mistralai/Mistral-7B-Instruct-v0.3 \\
      --calib data/calib_wikitext103_train_128x1024_mistral.pt \\
      --out compressed/mistral_alpha_oats_c70.pt \\
      --target_compression 0.70 \\
      --rank_ratio 0.30 \\
      --oats_iters 4 \\
      --alpha_strength 0.10 \\
      --batch_size 1 \\
      --max_seq_len 1024 \\
      --model_dtype bfloat16 \\
      --hidden_cache_dtype float16 \\
      --compress_device cuda \\
      --keep_dequantized_state_dict

80% effective compression:
    python alpha_oats_sparse_lowrank_hf.py \\
      --model_id mistralai/Mistral-7B-Instruct-v0.3 \\
      --calib data/calib_wikitext103_train_128x1024_mistral.pt \\
      --out compressed/mistral_alpha_oats_c80.pt \\
      --target_compression 0.80 \\
      --rank_ratio 0.35 \\
      --oats_iters 5 \\
      --alpha_strength 0.14 \\
      --batch_size 1 \\
      --max_seq_len 1024 \\
      --model_dtype bfloat16 \\
      --hidden_cache_dtype float16 \\
      --compress_device cuda \\
      --keep_dequantized_state_dict

90% effective compression:
    python alpha_oats_sparse_lowrank_hf.py \\
      --model_id mistralai/Mistral-7B-Instruct-v0.3 \\
      --calib data/calib_wikitext103_train_128x1024_mistral.pt \\
      --out compressed/mistral_alpha_oats_c90.pt \\
      --target_compression 0.90 \\
      --rank_ratio 0.50 \\
      --oats_iters 6 \\
      --alpha_strength 0.18 \\
      --batch_size 1 \\
      --max_seq_len 1024 \\
      --model_dtype bfloat16 \\
      --hidden_cache_dtype float16 \\
      --compress_device cuda \\
      --keep_dequantized_state_dict

Optional mask refinement:
    Add:
      --swap_iters 1 --swap_candidate_p 96 --swap_candidate_u 96 --gram_device cuda

Important:
    target_compression is effective compression:
        1 - (sparse_nonzeros + rank * (out_features + in_features)) / dense_params

    So 0.80 means only about 20% equivalent parameter budget is kept,
    split between sparse values and low-rank factors.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Basic utilities
# ============================================================

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


def cuda_mem() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB"


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        raise ValueError(f"Expected calibration tokens [N, T], got {tuple(tokens.shape)}")
    return tokens.long()


# ============================================================
# Module helpers
# ============================================================

def get_module_by_name(root: nn.Module, full_name: str) -> nn.Module:
    obj = root
    if not full_name:
        return obj
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


def layer_suffix(name: str) -> str:
    return name.split(".")[-1]


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
            cname = mod[0].__class__.__name__.lower()
            if "decoder" in cname or "layer" in cname or "block" in cname:
                return name, mod

    raise RuntimeError("Could not find decoder block ModuleList.")


def should_compress_name(
    name: str,
    suffixes: Tuple[str, ...],
    include: str,
    exclude: str,
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


def find_selected_linear_names(
    model: nn.Module,
    suffixes: Tuple[str, ...],
    include: str,
    exclude: str,
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
        if tied and skip_tied_lm_head and name == "lm_head":
            print("[info] skipping tied lm_head")
            continue
        if should_compress_name(
            name=name,
            suffixes=suffixes,
            include=include,
            exclude=exclude,
            compress_lm_head=compress_lm_head,
            skip_attn_out=skip_attn_out,
            skip_mlp_out=skip_mlp_out,
        ):
            out.append(name)
    return out


# ============================================================
# HF block forward helpers
# ============================================================

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


def call_decoder_block(block: nn.Module, hidden_states: torch.Tensor, backbone: nn.Module) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    position_ids = make_position_ids(batch_size, seq_len, device)
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
    return out


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
    print("\nComputing initial hidden cache...")
    t0 = now()

    for i in range(0, n, batch_size):
        input_ids = calib_tokens[i:i + batch_size].to(main_device)
        h = emb(input_ids)
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
    t0 = now()

    for i in range(0, n, batch_size):
        h = hidden_cache[i:i + batch_size].to(main_device)
        with torch.autocast(device_type=main_device.type, dtype=amp_dtype, enabled=autocast_enabled):
            out = call_decoder_block(block, h, backbone)
            if not torch.isfinite(out).all():
                print(f"\n[warn] non-finite output in {desc}; sanitizing.")
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


# ============================================================
# AlphaPruning-like allocation
# ============================================================

def parse_suffix_bias(raw: str) -> Dict[str, float]:
    """
    Example:
        q_proj:-0.08,k_proj:-0.08,v_proj:-0.10,o_proj:-0.15,gate_proj:0.05,up_proj:0.05,down_proj:-0.10

    Negative means protect the layer from compression.
    Positive means compress more aggressively.
    """
    default = {
        "q_proj": -0.06,
        "k_proj": -0.06,
        "v_proj": -0.08,
        "o_proj": -0.12,
        "gate_proj": 0.05,
        "up_proj": 0.05,
        "down_proj": -0.08,
    }
    raw = raw.strip()
    if not raw:
        return default

    out: Dict[str, float] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        k, v = item.split(":")
        out[k.strip()] = float(v)
    return out


@torch.no_grad()
def sample_matrix_for_alpha(
    W: torch.Tensor,
    max_rows: int,
    max_cols: int,
) -> torch.Tensor:
    """
    CRUCIAL:
    Full SVD/eigen-analysis of every 7B matrix is expensive on 32 GB RAM.
    This deterministic submatrix approximation gives a stable ranking signal.
    """
    W = W.detach().float().cpu()
    rows, cols = W.shape

    if rows > max_rows:
        ridx = torch.linspace(0, rows - 1, steps=max_rows).round().long()
        W = W.index_select(0, ridx)
    if cols > max_cols:
        cidx = torch.linspace(0, cols - 1, steps=max_cols).round().long()
        W = W.index_select(1, cidx)

    return sanitize_tensor(W)


@torch.no_grad()
def pl_alpha_hill_from_weight(
    W: torch.Tensor,
    max_rows: int = 2048,
    max_cols: int = 1024,
    top_frac: float = 0.20,
    eps: float = 1.0e-12,
) -> float:
    """
    Approximate PL_Alpha_Hill from sampled ESD.

    Lower alpha => heavier tail => more important => should receive lower compression.
    """
    Ws = sample_matrix_for_alpha(W, max_rows=max_rows, max_cols=max_cols)
    if Ws.numel() == 0:
        return 3.0

    # Use smaller covariance side.
    if Ws.size(0) >= Ws.size(1):
        C = Ws.T @ Ws
    else:
        C = Ws @ Ws.T

    C = 0.5 * (C + C.T)
    evals = torch.linalg.eigvalsh(C).float()
    evals = evals[torch.isfinite(evals)]
    evals = evals.clamp(min=eps)
    evals, _ = torch.sort(evals)

    n = evals.numel()
    if n < 16:
        return 3.0

    k = int(max(8, min(n - 1, round(top_frac * n))))
    top = evals[-k:]
    threshold = evals[-k - 1].clamp(min=eps)
    denom = torch.log((top / threshold).clamp(min=1.0 + 1.0e-7)).sum().item()
    if denom <= eps or not math.isfinite(denom):
        return 3.0

    alpha = 1.0 + k / denom
    if not math.isfinite(alpha):
        return 3.0
    return float(alpha)


def project_compressions_to_target(
    raw: Dict[str, float],
    weights: Dict[str, int],
    target: float,
    min_comp: float,
    max_comp: float,
) -> Dict[str, float]:
    """
    Shift all layer compressions by a scalar offset, with caps, until weighted average target is met.
    """
    names = list(raw.keys())

    def clipped(offset: float) -> Dict[str, float]:
        return {n: min(max_comp, max(min_comp, raw[n] + offset)) for n in names}

    def avg(d: Dict[str, float]) -> float:
        total = sum(weights[n] for n in names)
        return sum(d[n] * weights[n] for n in names) / max(total, 1)

    lo, hi = -1.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = avg(clipped(mid))
        if val < target:
            lo = mid
        else:
            hi = mid

    out = clipped(0.5 * (lo + hi))
    return out


@torch.no_grad()
def compute_alpha_compression_rates(
    model: nn.Module,
    selected_layer_names: Sequence[str],
    target_compression: float,
    alpha_strength: float,
    suffix_bias: Dict[str, float],
    min_compression: float,
    max_compression: float,
    alpha_sample_rows: int,
    alpha_sample_cols: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    CRUCIAL:
    This is the AlphaPruning-inspired part.
    It computes an approximate heavy-tail metric and uses it to make sparsity non-uniform.
    """
    layers_prefix, decoder_layers = find_decoder_layers(model)
    selected_set = set(selected_layer_names)

    block_metric: Dict[int, float] = {}
    layer_alpha: Dict[str, float] = {}

    print("\nComputing approximate AlphaPruning metrics...")
    t0 = now()

    for bi, block in enumerate(decoder_layers):
        alphas: List[float] = []
        block_prefix = f"{layers_prefix}.{bi}"

        for subname, mod in block.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            full_name = f"{block_prefix}.{subname}" if subname else block_prefix
            if full_name not in selected_set:
                continue

            alpha = pl_alpha_hill_from_weight(
                mod.weight.detach(),
                max_rows=alpha_sample_rows,
                max_cols=alpha_sample_cols,
            )
            layer_alpha[full_name] = alpha
            alphas.append(alpha)

        if alphas:
            block_metric[bi] = float(sum(alphas) / len(alphas))
            print(f" block {bi:02d}: alpha={block_metric[bi]:.4f}")

    vals = list(block_metric.values())
    if not vals:
        raise RuntimeError("No block alpha metrics were computed.")

    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((x - mean) ** 2 for x in vals) / max(len(vals), 1))
    std = max(std, 1.0e-6)

    raw: Dict[str, float] = {}
    weights: Dict[str, int] = {}

    for name in selected_layer_names:
        mod = get_module_by_name(model, name)
        weights[name] = int(mod.weight.numel())

        parts = name.split(".")
        try:
            bi = int(parts[parts.index("layers") + 1])
        except Exception:
            bi = 0

        z = (block_metric.get(bi, mean) - mean) / std
        suf = layer_suffix(name)

        # higher alpha => lighter tail => more compression
        raw[name] = target_compression + alpha_strength * z + suffix_bias.get(suf, 0.0)

    rates = project_compressions_to_target(
        raw=raw,
        weights=weights,
        target=target_compression,
        min_comp=min_compression,
        max_comp=max_compression,
    )

    actual = sum(rates[n] * weights[n] for n in rates) / sum(weights.values())
    print(f"Alpha allocation done in {fmt_time(now() - t0)}")
    print(f"Weighted effective compression target: {target_compression:.4f}")
    print(f"Weighted effective compression actual: {actual:.4f}")
    print("\nLayer compression rates:")
    for name in selected_layer_names:
        print(f" {name:<60s} comp={100.0 * rates[name]:6.2f}% alpha={layer_alpha.get(name, float('nan')):.4f}")

    return rates, layer_alpha


# ============================================================
# Input statistics collection
# ============================================================

class LinearInputStatsCollector:
    def __init__(
        self,
        layer: nn.Linear,
        name: str,
        collect_gram: bool,
        gram_device: torch.device,
        diag_dtype: torch.dtype = torch.float32,
        gram_dtype: torch.dtype = torch.float32,
        activation_clamp: float = 1.0e4,
    ):
        self.layer = layer
        self.name = name
        self.in_features = int(layer.in_features)
        self.collect_gram = bool(collect_gram)
        self.gram_device = gram_device
        self.diag_dtype = diag_dtype
        self.gram_dtype = gram_dtype
        self.activation_clamp = float(activation_clamp)

        self.diag = torch.zeros(self.in_features, dtype=diag_dtype, device="cpu")
        self.G: Optional[torch.Tensor] = None
        if self.collect_gram:
            self.G = torch.zeros(
                (self.in_features, self.in_features),
                dtype=gram_dtype,
                device=gram_device,
            )

        self.nsamples = 0
        self.handle = None
        self.warned = False

    def _hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        x = inputs[0]
        if not torch.is_tensor(x):
            return

        x = x.detach().reshape(-1, x.size(-1))
        if not torch.isfinite(x).all():
            if not self.warned:
                print(f"\n[warn] non-finite activation entering {self.name}; sanitizing.")
                self.warned = True
            x = sanitize_tensor(x, clamp_abs=self.activation_clamp)

        x32 = x.float()
        self.diag += (x32 * x32).sum(dim=0).detach().cpu().to(self.diag_dtype)

        if self.collect_gram:
            assert self.G is not None
            xg = x32.to(device=self.gram_device, dtype=self.gram_dtype)
            self.G += xg.T @ xg

        self.nsamples += int(x.size(0))

    def register(self) -> None:
        self.handle = self.layer.register_forward_pre_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def collect_block_input_stats(
    block: nn.Module,
    backbone: nn.Module,
    layer_infos: List[Tuple[str, nn.Linear]],
    hidden_cache: torch.Tensor,
    batch_size: int,
    main_device: torch.device,
    amp_dtype: torch.dtype,
    collect_gram: bool,
    gram_device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    collectors: Dict[str, LinearInputStatsCollector] = {}

    for name, layer in layer_infos:
        collectors[name] = LinearInputStatsCollector(
            layer=layer,
            name=name,
            collect_gram=collect_gram,
            gram_device=gram_device,
        )

    for c in collectors.values():
        c.register()

    n = hidden_cache.size(0)
    autocast_enabled = main_device.type == "cuda"
    t0 = now()
    print(f" Collecting input stats for {len(layer_infos)} layers...")

    try:
        for i in range(0, n, batch_size):
            h = hidden_cache[i:i + batch_size].to(main_device)
            with torch.autocast(device_type=main_device.type, dtype=amp_dtype, enabled=autocast_enabled):
                _ = call_decoder_block(block, h, backbone)
            done = min(i + batch_size, n)
            print(
                f"\r stats pass: {done}/{n} ({100.0 * done / n:.1f}%) "
                f"elapsed={fmt_time(now() - t0)}",
                end="",
                flush=True,
            )
        print()
    finally:
        for c in collectors.values():
            c.remove()

    out: Dict[str, Dict[str, Any]] = {}
    for name, c in collectors.items():
        out[name] = {
            "diag": c.diag.detach().cpu(),
            "G": c.G.detach().cpu() if c.G is not None and c.G.device.type != "cpu" else c.G,
            "nsamples": c.nsamples,
        }

    return out


# ============================================================
# Sparse + low-rank OATS helpers
# ============================================================

@dataclass
class OATSLayerResult:
    mask: torch.Tensor
    sparse_values: torch.Tensor
    lowrank_left: torch.Tensor
    lowrank_right: torch.Tensor
    dense_compressed_weight: torch.Tensor
    shape: Tuple[int, int]
    rank: int
    sparse_nnz: int
    total_params: int
    effective_compression: float
    sparse_fraction: float
    rank_ratio: float
    oats_iters: int


def compute_rank_and_sparse_nnz(
    rows: int,
    cols: int,
    target_compression: float,
    rank_ratio: float,
) -> Tuple[int, int]:
    dense_params = rows * cols
    kept_budget = max(1.0, (1.0 - target_compression) * dense_params)

    rank_params_budget = rank_ratio * kept_budget
    rank = int(math.floor(rank_params_budget / max(rows + cols, 1)))
    rank = max(0, min(rank, min(rows, cols) - 1))

    sparse_budget = kept_budget - rank * (rows + cols)
    sparse_nnz = int(max(0, min(dense_params, math.floor(sparse_budget))))
    return rank, sparse_nnz


@torch.no_grad()
def rowwise_topk_mask(score: torch.Tensor, total_keep: int) -> torch.Tensor:
    """
    Row-wise hard thresholding.

    CRUCIAL:
    Equal-ish row budget is intentionally used because:
      - it is more stable for LLM pruning than fully global thresholding,
      - it enables SparseSwaps-style row decoupling.
    """
    score = sanitize_tensor(score.float())
    rows, cols = score.shape
    total = rows * cols
    total_keep = int(max(0, min(total_keep, total)))

    if total_keep <= 0:
        return torch.zeros_like(score, dtype=torch.bool)
    if total_keep >= total:
        return torch.ones_like(score, dtype=torch.bool)

    base = total_keep // rows
    rem = total_keep - base * rows
    base = int(max(0, min(base, cols)))

    mask = torch.zeros_like(score, dtype=torch.bool)

    if base >= cols:
        return torch.ones_like(score, dtype=torch.bool)

    if base > 0:
        idx = torch.topk(score, k=base, dim=1, largest=True, sorted=False).indices
        row_idx = torch.arange(rows, device=score.device).view(-1, 1).expand_as(idx)
        mask[row_idx, idx] = True

    if rem > 0 and base < cols:
        # Candidate next-best per row.
        vals_extra, idx_extra = torch.topk(score, k=base + 1, dim=1, largest=True, sorted=True)
        next_val = vals_extra[:, base]
        next_idx = idx_extra[:, base]
        extra_rows = torch.topk(next_val, k=rem, largest=True, sorted=False).indices
        mask[extra_rows, next_idx[extra_rows]] = True

    return mask


@torch.no_grad()
def truncated_svd_factors(
    A: torch.Tensor,
    rank: int,
    backend: str,
    oversample: int,
    niter: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns A_fac, B_fac such that L ~= A_fac @ B_fac.

    A_fac: [rows, rank]
    B_fac: [rank, cols]
    """
    rows, cols = A.shape
    rank = int(max(0, min(rank, min(rows, cols) - 1)))

    if rank <= 0:
        return (
            torch.empty((rows, 0), device=A.device, dtype=A.dtype),
            torch.empty((0, cols), device=A.device, dtype=A.dtype),
        )

    A = sanitize_tensor(A.float())

    if backend == "exact":
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        U = U[:, :rank].contiguous()
        S = S[:rank].contiguous()
        Vh = Vh[:rank, :].contiguous()
        return U * S.view(1, -1), Vh

    # Randomized SVD through pca_lowrank.
    q = int(min(min(rows, cols), max(rank + oversample, rank + 2)))
    U, S, V = torch.pca_lowrank(A, q=q, center=False, niter=niter)
    U = U[:, :rank].contiguous()
    S = S[:rank].contiguous()
    V = V[:, :rank].contiguous()
    return U * S.view(1, -1), V.T.contiguous()


@torch.no_grad()
def sparse_swaps_refine_mask(
    W_res: torch.Tensor,
    mask: torch.Tensor,
    G: torch.Tensor,
    max_iters: int,
    candidate_p: int,
    candidate_u: int,
    eps: float = 0.0,
) -> torch.Tensor:
    """
    Approximate SparseSwaps local search.

    W_res is the residual to sparsify:
        W_res = W - L

    mask=True means kept in sparse matrix.
    For each row, swap one pruned and one kept coordinate if the exact quadratic loss decreases.

    This version restricts each row to a candidate pool for 3090 practicality.
    """
    if max_iters <= 0:
        return mask

    device = W_res.device
    W_res = sanitize_tensor(W_res.float()).to(device)
    mask = mask.bool().to(device)
    G = sanitize_tensor(G.float()).to(device)
    diagG = torch.diag(G).contiguous()

    rows, cols = W_res.shape
    t0 = now()

    print(
        f"   SparseSwaps refine: rows={rows}, cols={cols}, "
        f"iters={max_iters}, cand_p={candidate_p}, cand_u={candidate_u}"
    )

    for i in range(rows):
        w = W_res[i]
        m = mask[i].clone()

        if m.sum().item() == 0 or m.sum().item() == cols:
            mask[i] = m
            continue

        rvec = torch.where(~m, w, torch.zeros_like(w))
        c = G.mv(rvec)

        for _ in range(max_iters):
            P = torch.nonzero(~m, as_tuple=False).flatten()
            U = torch.nonzero(m, as_tuple=False).flatten()
            if P.numel() == 0 or U.numel() == 0:
                break

            wp_all = w[P]
            wu_all = w[U]

            # Loss change of removing pruned p from residual, alone.
            p_delta = -2.0 * wp_all * c[P] + (wp_all * wp_all) * diagG[P]
            # Loss change of adding kept u into residual, alone.
            u_delta = 2.0 * wu_all * c[U] + (wu_all * wu_all) * diagG[U]

            kp = min(candidate_p, P.numel())
            ku = min(candidate_u, U.numel())

            p_idx_local = torch.topk(p_delta, k=kp, largest=False, sorted=False).indices
            u_idx_local = torch.topk(u_delta, k=ku, largest=False, sorted=False).indices

            Pc = P[p_idx_local]
            Uc = U[u_idx_local]
            wp = w[Pc]
            wu = w[Uc]

            # Full exact one-swap delta:
            # Δ = p_delta + u_delta - 2 w_p w_u G[p,u]
            cross = G[Pc][:, Uc]
            delta = (
                p_delta[p_idx_local].view(-1, 1)
                + u_delta[u_idx_local].view(1, -1)
                - 2.0 * wp.view(-1, 1) * wu.view(1, -1) * cross
            )

            best_val, best_flat = torch.min(delta.reshape(-1), dim=0)
            if float(best_val.item()) < -float(eps):
                pi = int(best_flat.item() // ku)
                ui = int(best_flat.item() % ku)
                p = int(Pc[pi].item())
                u = int(Uc[ui].item())

                # p becomes kept, u becomes pruned.
                m[p] = True
                m[u] = False

                # Update correlation vector:
                # r' = r - w_p phi_p + w_u phi_u
                c = c - w[p] * G[:, p] + w[u] * G[:, u]
            else:
                break

        mask[i] = m

        if (i + 1) % 512 == 0:
            print(
                f"\r   SparseSwaps rows: {i + 1}/{rows} "
                f"elapsed={fmt_time(now() - t0)}",
                end="",
                flush=True,
            )

    print()
    return mask.detach()


@torch.no_grad()
def oats_compress_linear(
    layer: nn.Linear,
    input_diag: torch.Tensor,
    input_gram: Optional[torch.Tensor],
    target_compression: float,
    rank_ratio: float,
    oats_iters: int,
    svd_backend: str,
    svd_oversample: int,
    svd_niter: int,
    compress_device: torch.device,
    value_dtype: torch.dtype,
    swap_iters: int,
    swap_candidate_p: int,
    swap_candidate_u: int,
) -> OATSLayerResult:
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(layer)}")

    orig_device = layer.weight.device
    orig_dtype = layer.weight.dtype

    W = layer.weight.detach().to(device=compress_device, dtype=torch.float32).clone()
    W = sanitize_tensor(W)
    rows, cols = W.shape

    D = input_diag.detach().float().to(compress_device).clamp(min=1.0e-12).sqrt()
    D = D / D.mean().clamp(min=1.0e-12)
    D = D.clamp(min=1.0e-6, max=1.0e6)

    rank, sparse_nnz_target = compute_rank_and_sparse_nnz(
        rows=rows,
        cols=cols,
        target_compression=target_compression,
        rank_ratio=rank_ratio,
    )

    print(f"   shape                 : {rows} x {cols}")
    print(f"   target compression    : {100.0 * target_compression:.2f}%")
    print(f"   rank ratio            : {rank_ratio:.3f}")
    print(f"   chosen low-rank rank  : {rank}")
    print(f"   sparse nnz target     : {sparse_nnz_target:,}")
    print(f"   OATS iterations       : {oats_iters}")
    print(f"   SVD backend           : {svd_backend}")

    # CRUCIAL:
    # Outlier-aware scaling: preserve channels with large activation second moment.
    A = W * D.view(1, -1)

    S_scaled = torch.zeros_like(A)
    A_fac = torch.empty((rows, 0), dtype=torch.float32, device=compress_device)
    B_fac = torch.empty((0, cols), dtype=torch.float32, device=compress_device)
    mask = torch.zeros_like(A, dtype=torch.bool)

    t0 = now()

    for it in range(1, oats_iters + 1):
        R_for_lowrank = A - S_scaled
        A_fac, B_fac = truncated_svd_factors(
            R_for_lowrank,
            rank=rank,
            backend=svd_backend,
            oversample=svd_oversample,
            niter=svd_niter,
        )

        if rank > 0:
            L_scaled = A_fac @ B_fac
        else:
            L_scaled = torch.zeros_like(A)

        residual_for_sparse = A - L_scaled
        mask = rowwise_topk_mask(residual_for_sparse.abs(), sparse_nnz_target)
        S_scaled = residual_for_sparse * mask.to(residual_for_sparse.dtype)

        approx = S_scaled + L_scaled
        rel = torch.linalg.norm((A - approx).float()) / torch.linalg.norm(A.float()).clamp(min=1.0e-12)
        print(f"    OATS iter {it:02d}/{oats_iters}: scaled relative Fro error={float(rel):.6e}")

        del R_for_lowrank, L_scaled, residual_for_sparse, approx
        cleanup()

    # Unscale sparse term.
    S_unscaled = S_scaled / D.view(1, -1)

    # Unscale low-rank right factor:
    # L_unscaled = A_fac @ (B_fac D^{-1})
    B_unscaled = B_fac / D.view(1, -1)
    L_unscaled = A_fac @ B_unscaled if rank > 0 else torch.zeros_like(W)

    # Optional SparseSwaps refinement of sparse mask with low-rank fixed.
    if swap_iters > 0:
        if input_gram is None:
            print("   [warn] swap_iters > 0 but no Gram matrix was collected; skipping swaps.")
        else:
            G = input_gram.to(device=compress_device, dtype=torch.float32)
            W_res = W - L_unscaled
            mask = sparse_swaps_refine_mask(
                W_res=W_res,
                mask=mask,
                G=G,
                max_iters=swap_iters,
                candidate_p=swap_candidate_p,
                candidate_u=swap_candidate_u,
            )
            S_unscaled = W_res * mask.to(W_res.dtype)
            del G, W_res
            cleanup()

    W_compressed = S_unscaled + L_unscaled
    W_compressed = sanitize_tensor(W_compressed)

    layer.weight.data.copy_(W_compressed.to(device=orig_device, dtype=orig_dtype))

    sparse_nnz = int(mask.sum().item())
    total_params = rows * cols
    lowrank_params = rank * (rows + cols)
    effective_kept = sparse_nnz + lowrank_params
    effective_compression = 1.0 - effective_kept / float(total_params)

    sparse_values = S_unscaled[mask].detach().cpu().to(value_dtype).contiguous()
    lowrank_left = A_fac.detach().cpu().to(value_dtype).contiguous()
    lowrank_right = B_unscaled.detach().cpu().to(value_dtype).contiguous()

    print(f"   final sparse nnz      : {sparse_nnz:,}")
    print(f"   low-rank params       : {lowrank_params:,}")
    print(f"   effective compression : {100.0 * effective_compression:.2f}%")
    print(f"   layer time            : {fmt_time(now() - t0)}")

    result = OATSLayerResult(
        mask=mask.detach().cpu(),
        sparse_values=sparse_values,
        lowrank_left=lowrank_left,
        lowrank_right=lowrank_right,
        dense_compressed_weight=W_compressed.detach().cpu().to(value_dtype),
        shape=(rows, cols),
        rank=rank,
        sparse_nnz=sparse_nnz,
        total_params=total_params,
        effective_compression=float(effective_compression),
        sparse_fraction=1.0 - sparse_nnz / float(total_params),
        rank_ratio=float(rank_ratio),
        oats_iters=int(oats_iters),
    )

    del W, A, S_scaled, S_unscaled, L_unscaled, W_compressed, A_fac, B_fac, B_unscaled
    cleanup()

    return result


# ============================================================
# Checkpoint helpers
# ============================================================

def get_compressed_weight_keys(compressed_layers: Dict[str, Any]) -> set[str]:
    return {name + ".weight" for name in compressed_layers.keys()}


def build_partial_noncompressed_state_dict(
    model: nn.Module,
    compressed_layers: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    compressed_weight_keys = get_compressed_weight_keys(compressed_layers)
    out: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if k in compressed_weight_keys:
            continue
        out[k] = v.detach().cpu()
    return out


def save_oats_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    out_path: str,
    model_id: str,
    meta: Dict[str, Any],
    oats_layers: Dict[str, Any],
    keep_dequantized_state_dict: bool,
) -> None:
    if keep_dequantized_state_dict:
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    else:
        model_state = build_partial_noncompressed_state_dict(model, oats_layers)

    ckpt = {
        "format": "hf_alpha_oats_sparse_lowrank",
        "model_id": model_id,
        "config": model.config.to_dict() if hasattr(model, "config") else None,
        "model": model_state,
        "alpha_oats_meta": meta,
        "alpha_oats_layers": oats_layers,
        "compression_meta": meta,
    }
    try:
        ckpt["tokenizer_name_or_path"] = tokenizer.name_or_path
    except Exception:
        ckpt["tokenizer_name_or_path"] = model_id

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving checkpoint to {out} ...")
    t0 = now()
    torch.save(ckpt, out)
    print(f"Saved in {fmt_time(now() - t0)}")

    meta_path = Path(str(out) + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta JSON to {meta_path}")


# ============================================================
# Main blockwise compressor
# ============================================================

@torch.no_grad()
def compress_hf_model_blockwise(
    model: nn.Module,
    tokenizer: Any,
    calib_tokens: torch.Tensor,
    selected_layer_names: List[str],
    compression_rates: Dict[str, float],
    batch_size: int,
    main_device: torch.device,
    model_dtype: torch.dtype,
    hidden_cache_dtype: torch.dtype,
    value_dtype: torch.dtype,
    target_compression: float,
    rank_ratio: float,
    oats_iters: int,
    svd_backend: str,
    svd_oversample: int,
    svd_niter: int,
    compress_device: torch.device,
    collect_gram: bool,
    gram_device: torch.device,
    swap_iters: int,
    swap_candidate_p: int,
    swap_candidate_u: int,
    store_debug_dense_weight: bool,
) -> Dict[str, Any]:
    selected_set = set(selected_layer_names)
    layers_prefix, decoder_layers = find_decoder_layers(model)
    backbone = model.model if hasattr(model, "model") else model

    print(f"\nDecoder layers found: {layers_prefix}")
    print(f"Number of decoder blocks: {len(decoder_layers)}")

    hidden_cache = compute_initial_hidden_cache(
        model=model,
        calib_tokens=calib_tokens,
        batch_size=batch_size,
        main_device=main_device,
        storage_dtype=hidden_cache_dtype,
    )

    oats_layers: Dict[str, Any] = {}
    script_t0 = now()

    for bi, block in enumerate(decoder_layers):
        block_prefix = f"{layers_prefix}.{bi}"
        layer_infos: List[Tuple[str, nn.Linear]] = []

        for subname, mod in block.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            full_name = f"{block_prefix}.{subname}" if subname else block_prefix
            if full_name in selected_set:
                layer_infos.append((full_name, mod))

        print("\n" + "=" * 100)
        print(f"BLOCK {bi}/{len(decoder_layers) - 1}")
        print(f"Selected linear layers: {len(layer_infos)}")
        print(f"Hidden cache entering block: {tuple(hidden_cache.shape)}, dtype={hidden_cache.dtype}")
        print(f"CUDA memory: {cuda_mem()}")
        for name, layer in layer_infos:
            print(f" - {name}: {tuple(layer.weight.shape)} comp={100.0 * compression_rates[name]:.2f}%")

        if not layer_infos:
            hidden_cache = run_block_to_cache(
                block=block,
                backbone=backbone,
                hidden_cache=hidden_cache,
                batch_size=batch_size,
                main_device=main_device,
                amp_dtype=model_dtype,
                storage_dtype=hidden_cache_dtype,
                desc=f"block {bi} dense forward",
            )
            continue

        stats = collect_block_input_stats(
            block=block,
            backbone=backbone,
            layer_infos=layer_infos,
            hidden_cache=hidden_cache,
            batch_size=batch_size,
            main_device=main_device,
            amp_dtype=model_dtype,
            collect_gram=collect_gram,
            gram_device=gram_device,
        )

        for li, (layer_name, layer) in enumerate(layer_infos, start=1):
            print("\n" + "-" * 100)
            print(f"[{li}/{len(layer_infos)}] Compressing {layer_name}")
            print(f"CUDA memory before layer: {cuda_mem()}")

            layer_stats = stats[layer_name]
            nsamples = int(layer_stats["nsamples"])
            if nsamples == 0:
                raise RuntimeError(f"No activation samples collected for {layer_name}")

            input_diag = layer_stats["diag"]
            input_gram = layer_stats["G"] if collect_gram else None

            result = oats_compress_linear(
                layer=layer,
                input_diag=input_diag,
                input_gram=input_gram,
                target_compression=float(compression_rates[layer_name]),
                rank_ratio=rank_ratio,
                oats_iters=oats_iters,
                svd_backend=svd_backend,
                svd_oversample=svd_oversample,
                svd_niter=svd_niter,
                compress_device=compress_device,
                value_dtype=value_dtype,
                swap_iters=swap_iters,
                swap_candidate_p=swap_candidate_p,
                swap_candidate_u=swap_candidate_u,
            )

            state: Dict[str, Any] = {
                "shape": list(result.shape),
                "mask": pack_bool_mask_rows(result.mask),
                "mask_packing": "packedbits",
                "sparse_values": result.sparse_values,
                "values_format": "kept_1d_rowmajor",
                "lowrank_left": result.lowrank_left,
                "lowrank_right": result.lowrank_right,
                "lowrank_format": "left_right",
                "value_dtype": dtype_to_name(value_dtype),
                "rank": int(result.rank),
                "sparse_nnz": int(result.sparse_nnz),
                "total_params": int(result.total_params),
                "effective_compression": float(result.effective_compression),
                "sparse_fraction": float(result.sparse_fraction),
                "rank_ratio": float(result.rank_ratio),
                "oats_iters": int(result.oats_iters),
                "target_compression": float(compression_rates[layer_name]),
            }

            if store_debug_dense_weight:
                state["dense_debug_weight"] = result.dense_compressed_weight

            oats_layers[layer_name] = state

            print(
                f" saved:\n"
                f"   mask packed       : {tuple(state['mask'].shape)}\n"
                f"   sparse_values     : {tuple(state['sparse_values'].shape)}\n"
                f"   lowrank_left      : {tuple(state['lowrank_left'].shape)}\n"
                f"   lowrank_right     : {tuple(state['lowrank_right'].shape)}"
            )

            del result
            cleanup()

        del stats
        cleanup()

        print("\nRunning compressed block to create next hidden cache...")
        hidden_cache = run_block_to_cache(
            block=block,
            backbone=backbone,
            hidden_cache=hidden_cache,
            batch_size=batch_size,
            main_device=main_device,
            amp_dtype=model_dtype,
            storage_dtype=hidden_cache_dtype,
            desc=f"block {bi} compressed forward",
        )

        print(f"Finished block {bi}. Elapsed: {fmt_time(now() - script_t0)}")
        print(f"CUDA memory: {cuda_mem()}")

    return oats_layers


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--target_compression", type=float, default=0.70)
    parser.add_argument("--rank_ratio", type=float, default=0.30)
    parser.add_argument("--oats_iters", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compress_device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gram_device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--model_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--hidden_cache_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--value_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--svd_backend", type=str, default="randomized", choices=["randomized", "exact"])
    parser.add_argument("--svd_oversample", type=int, default=24)
    parser.add_argument("--svd_niter", type=int, default=2)

    parser.add_argument("--swap_iters", type=int, default=0)
    parser.add_argument("--swap_candidate_p", type=int, default=96)
    parser.add_argument("--swap_candidate_u", type=int, default=96)

    parser.add_argument("--alpha_strength", type=float, default=0.10)
    parser.add_argument("--min_compression", type=float, default=0.30)
    parser.add_argument("--max_compression", type=float, default=0.94)
    parser.add_argument("--suffix_bias", type=str, default="")
    parser.add_argument("--alpha_sample_rows", type=int, default=2048)
    parser.add_argument("--alpha_sample_cols", type=int, default=1024)

    parser.add_argument("--suffixes", type=str, default="")
    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")
    parser.add_argument("--compress_lm_head", action="store_true")
    parser.add_argument("--skip_tied_lm_head", action="store_true")
    parser.add_argument("--skip_attn_out", action="store_true")
    parser.add_argument("--skip_mlp_out", action="store_true")

    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="eager")

    parser.add_argument("--keep_dequantized_state_dict", action="store_true")
    parser.add_argument("--store_debug_dense_weight", action="store_true")

    args = parser.parse_args()

    script_t0 = now()

    if not (0.0 <= args.target_compression < 1.0):
        raise ValueError("--target_compression must be in [0, 1).")
    if not (0.0 <= args.rank_ratio <= 1.0):
        raise ValueError("--rank_ratio must be in [0, 1].")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if args.compress_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--compress_device cuda requested but CUDA unavailable.")
    if args.gram_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--gram_device cuda requested but CUDA unavailable.")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    main_device = torch.device(args.device)
    compress_device = torch.device(args.compress_device)
    gram_device = torch.device(args.gram_device)

    model_dtype = parse_dtype(args.model_dtype)
    hidden_cache_dtype = parse_dtype(args.hidden_cache_dtype)
    value_dtype = parse_dtype(args.value_dtype)

    suffixes = parse_suffixes(args.suffixes)
    suffix_bias = parse_suffix_bias(args.suffix_bias)

    print("=" * 100)
    print("HF Alpha-OATS Sparse+LowRank Compression")
    print("=" * 100)
    print(f"model_id                 : {args.model_id}")
    print(f"calib                    : {args.calib}")
    print(f"out                      : {args.out}")
    print(f"device                   : {main_device}")
    print(f"compress_device          : {compress_device}")
    print(f"gram_device              : {gram_device}")
    print(f"model_dtype              : {model_dtype}")
    print(f"hidden_cache_dtype       : {hidden_cache_dtype}")
    print(f"value_dtype              : {value_dtype}")
    print(f"target_compression       : {args.target_compression}")
    print(f"rank_ratio               : {args.rank_ratio}")
    print(f"oats_iters               : {args.oats_iters}")
    print(f"svd_backend              : {args.svd_backend}")
    print(f"swap_iters               : {args.swap_iters}")
    print(f"alpha_strength           : {args.alpha_strength}")
    print(f"min/max compression      : {args.min_compression}/{args.max_compression}")
    print(f"suffix_bias              : {suffix_bias}")
    print(f"suffixes                 : {suffixes}")
    print(f"keep_dequantized_state   : {args.keep_dequantized_state_dict}")
    print(f"CUDA memory              : {cuda_mem()}")

    print("\nLoading tokenizer/model...")
    t0 = now()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=bool(args.trust_remote_code),
    )
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

    print(f"Loaded model in {fmt_time(now() - t0)}")
    print(f"CUDA memory after load: {cuda_mem()}")

    calib_tokens = load_calibration_tokens(args.calib)
    print(f"Loaded calibration tokens: {tuple(calib_tokens.shape)}")

    if args.max_seq_len > 0 and calib_tokens.size(1) > args.max_seq_len:
        calib_tokens = calib_tokens[:, :args.max_seq_len]
        print(f"Trimmed calibration sequence length to {args.max_seq_len}")

    if hasattr(model, "config") and hasattr(model.config, "max_position_embeddings"):
        max_pos = int(model.config.max_position_embeddings)
        if calib_tokens.size(1) > max_pos:
            calib_tokens = calib_tokens[:, :max_pos]
            print(f"Trimmed calibration sequence length to model max_position_embeddings={max_pos}")

    selected_layer_names = find_selected_linear_names(
        model=model,
        suffixes=suffixes,
        include=args.include,
        exclude=args.exclude,
        compress_lm_head=bool(args.compress_lm_head),
        skip_tied_lm_head=bool(args.skip_tied_lm_head),
        skip_attn_out=bool(args.skip_attn_out),
        skip_mlp_out=bool(args.skip_mlp_out),
    )

    if not selected_layer_names:
        raise RuntimeError("No nn.Linear layers selected for compression.")

    print(f"\nSelected linear layers: {len(selected_layer_names)}")
    for name in selected_layer_names:
        mod = get_module_by_name(model, name)
        print(f" - {name}: {tuple(mod.weight.shape)}")

    compression_rates, alpha_metrics = compute_alpha_compression_rates(
        model=model,
        selected_layer_names=selected_layer_names,
        target_compression=float(args.target_compression),
        alpha_strength=float(args.alpha_strength),
        suffix_bias=suffix_bias,
        min_compression=float(args.min_compression),
        max_compression=float(args.max_compression),
        alpha_sample_rows=int(args.alpha_sample_rows),
        alpha_sample_cols=int(args.alpha_sample_cols),
    )

    collect_gram = bool(args.swap_iters > 0)

    oats_layers = compress_hf_model_blockwise(
        model=model,
        tokenizer=tokenizer,
        calib_tokens=calib_tokens,
        selected_layer_names=selected_layer_names,
        compression_rates=compression_rates,
        batch_size=int(args.batch_size),
        main_device=main_device,
        model_dtype=model_dtype,
        hidden_cache_dtype=hidden_cache_dtype,
        value_dtype=value_dtype,
        target_compression=float(args.target_compression),
        rank_ratio=float(args.rank_ratio),
        oats_iters=int(args.oats_iters),
        svd_backend=str(args.svd_backend),
        svd_oversample=int(args.svd_oversample),
        svd_niter=int(args.svd_niter),
        compress_device=compress_device,
        collect_gram=collect_gram,
        gram_device=gram_device,
        swap_iters=int(args.swap_iters),
        swap_candidate_p=int(args.swap_candidate_p),
        swap_candidate_u=int(args.swap_candidate_u),
        store_debug_dense_weight=bool(args.store_debug_dense_weight),
    )

    total_dense = sum(int(st["total_params"]) for st in oats_layers.values())
    total_sparse = sum(int(st["sparse_nnz"]) for st in oats_layers.values())
    total_lowrank = sum(int(st["rank"]) * (int(st["shape"][0]) + int(st["shape"][1])) for st in oats_layers.values())
    total_effective_kept = total_sparse + total_lowrank
    actual_effective_compression = 1.0 - total_effective_kept / float(max(total_dense, 1))

    meta = {
        "method": "alpha_oats_sparse_lowrank_blockwise",
        "model_id": str(args.model_id),
        "target_compression": float(args.target_compression),
        "actual_effective_compression": float(actual_effective_compression),
        "rank_ratio": float(args.rank_ratio),
        "oats_iters": int(args.oats_iters),
        "svd_backend": str(args.svd_backend),
        "svd_oversample": int(args.svd_oversample),
        "svd_niter": int(args.svd_niter),
        "swap_iters": int(args.swap_iters),
        "swap_candidate_p": int(args.swap_candidate_p),
        "swap_candidate_u": int(args.swap_candidate_u),
        "alpha_strength": float(args.alpha_strength),
        "alpha_metrics": {k: float(v) for k, v in alpha_metrics.items()},
        "compression_rates": {k: float(v) for k, v in compression_rates.items()},
        "suffix_bias": suffix_bias,
        "suffixes": list(suffixes),
        "model_dtype": str(args.model_dtype),
        "hidden_cache_dtype": str(args.hidden_cache_dtype),
        "value_dtype": str(args.value_dtype),
        "calibration_source": str(args.calib),
        "max_seq_len": int(args.max_seq_len),
        "compressed_layers": int(len(oats_layers)),
        "total_dense_params_selected": int(total_dense),
        "total_sparse_values": int(total_sparse),
        "total_lowrank_params": int(total_lowrank),
        "total_effective_kept_params": int(total_effective_kept),
        "keep_dequantized_state_dict": bool(args.keep_dequantized_state_dict),
        "store_debug_dense_weight": bool(args.store_debug_dense_weight),
        "script_seconds": float(now() - script_t0),
        "note": (
            "W_compressed = S + L. Sparse S is stored by packed mask plus kept values. "
            "Low-rank L is stored as lowrank_left @ lowrank_right. "
            "Effective compression counts sparse values plus low-rank factor parameters. "
            "This checkpoint can also include dense dequantized compressed weights if "
            "--keep_dequantized_state_dict is set."
        ),
    }

    save_oats_checkpoint(
        model=model,
        tokenizer=tokenizer,
        out_path=str(args.out),
        model_id=str(args.model_id),
        meta=meta,
        oats_layers=oats_layers,
        keep_dequantized_state_dict=bool(args.keep_dequantized_state_dict),
    )

    print("\nDone.")
    print(f"Compressed layers              : {len(oats_layers)}")
    print(f"Selected dense params          : {total_dense:,}")
    print(f"Sparse values                  : {total_sparse:,}")
    print(f"Low-rank params                : {total_lowrank:,}")
    print(f"Effective kept params          : {total_effective_kept:,}")
    print(f"Actual effective compression   : {100.0 * actual_effective_compression:.2f}%")
    print(f"Total script time              : {fmt_time(now() - script_t0)}")
    print(f"CUDA memory                    : {cuda_mem()}")

    if args.keep_dequantized_state_dict:
        print("Checkpoint includes dense compressed model weights for direct evaluation.")
    else:
        print("Checkpoint stores factorized sparse+lowrank layers; compressed weights omitted from ckpt['model'].")


if __name__ == "__main__":
    main()