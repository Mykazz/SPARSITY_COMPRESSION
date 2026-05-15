#!/usr/bin/env python3
"""
AlphaPruning + SparseGPT-style pruning for Hugging Face causal LMs.

Designed for:
    - mistralai/Mistral-7B-Instruct-v0.3
    - meta-llama/Llama-*
    - Qwen/Qwen*
    - other decoder-only AutoModelForCausalLM models with transformer blocks

Core method:
    1. Compute AlphaPruning heavy-tail metric from each selected Linear weight matrix.
       We use singular values of W and the Hill estimator on eigenvalues of W^T W.
       Lower alpha => more heavy-tailed => more important => less sparsity.

    2. Map block metrics to non-uniform block sparsities:
           s_b = map(alpha_b) with weighted normalization to target global sparsity.

    3. Sequential blockwise SparseGPT-style pruning:
           H = 2 X^T X
           H_damped = H + damp I
           H_inv_chol = chol(inv(H_damped))
           prune mask by SparseGPT saliency:
               score_ij = W_ij^2 / H_inv_chol_diag_j^2
           then do GPTQ/SparseGPT column-wise compensation:
               err = (w - q) / d
               W_future -= err @ H_inv_chol[j, future]

Important:
    - This is pure pruning, not INT4 quantization.
    - It saves compact sparse layers as:
          packed bit mask + kept values
    - Runtime kernels are not optimized. This is for mathematically correct experiments.
    - For inference/eval, a matching runtime wrapper is included in this file.
      You can import load_alpha_sparse_checkpoint(...) in eval code.

Recommended first experiments:
    60%, 65%, 70%, 75%, 80% global sparsity
    Compare against your uniform SparseGPT/GPTQ results.

Example:
    /venv/main/bin/python alpha_sparsegpt_hf.py \\
      --model_id mistralai/Mistral-7B-Instruct-v0.3 \\
      --calib data/calib_wikitext103_train_128x2048_mistral.pt \\
      --out compressed/mistral_alpha_sparsegpt_s70_bf16_1024.pt \\
      --target_sparsity 0.70 \\
      --alpha_s1 0.50 \\
      --alpha_s2 0.90 \\
      --model_dtype bfloat16 \\
      --hidden_cache_dtype float16 \\
      --hessian_dtype float32 \\
      --batch_size 1 \\
      --max_seq_len 1024 \\
      --blocksize 128 \\
      --percdamp 0.1 \\
      --large_layer_cpu_threshold 8192 \\
      --attn_implementation eager \\
      2>&1 | tee logs/mistral_alpha_sparsegpt_s70_bf16_1024.log
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Constants
# =============================================================================

DEFAULT_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


# =============================================================================
# Small utilities
# =============================================================================

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


def cuda_mem_string() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return f"alloc={alloc:.2f}GB reserved={reserv:.2f}GB peak={peak:.2f}GB"


def print_cuda_memory(prefix: str = "CUDA memory") -> None:
    if torch.cuda.is_available():
        print(f"{prefix}: {cuda_mem_string()}")


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


def get_first_parameter_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_first_parameter_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        try:
            return next(module.buffers()).dtype
        except StopIteration:
            return torch.float16


def parse_suffixes(raw: str) -> Tuple[str, ...]:
    raw = raw.strip()
    if not raw:
        return DEFAULT_SUFFIXES
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def ensure_dir_for_file(path: str) -> None:
    p = Path(path)
    if p.parent:
        p.parent.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Calibration tokens
# =============================================================================

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
# Packed mask helpers
# =============================================================================

def pack_bool_mask_rows(mask: torch.Tensor) -> torch.Tensor:
    """
    Pack bool mask [rows, cols] into uint8 [rows, ceil(cols/8)].

    Bit convention:
        bit 0 -> col 0 inside byte
        bit 1 -> col 1 inside byte
        ...
        bit 7 -> col 7 inside byte
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


# =============================================================================
# Compact sparse runtime layer
# =============================================================================

class AlphaSparseLinear(nn.Module):
    """
    Runtime wrapper for compact pure sparse layer.

    Stored:
        mask_packed : uint8 [out_features, ceil(in_features / 8)]
        values      : kept nonzero values in row-major mask order

    Runtime:
        W = zeros(out, in)
        W[mask] = values
        y = F.linear(x, W, bias)

    This is mathematically correct but not inference-kernel optimized.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask_packed: torch.Tensor,
        values: torch.Tensor,
        value_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
        cache_dequantized: bool = False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.value_dtype_name = str(value_dtype).replace("torch.", "")
        self.cache_dequantized = bool(cache_dequantized)

        self.register_buffer("mask_packed", mask_packed.contiguous().to(torch.uint8))
        self.register_buffer("values", values.contiguous())

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
        else:
            self.bias = None

        self._cached_weight: Optional[torch.Tensor] = None

    @torch.no_grad()
    def dequantize_weight(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        device = self.mask_packed.device
        if dtype is None:
            dtype = self.values.dtype

        mask = unpack_bool_mask_rows(self.mask_packed, self.in_features).to(device)
        w = torch.zeros(
            (self.out_features, self.in_features),
            device=device,
            dtype=dtype,
        )
        w[mask] = self.values.to(device=device, dtype=dtype)
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


@torch.no_grad()
def apply_alpha_sparse_layers_to_model(
    model: nn.Module,
    ckpt: Dict[str, Any],
    device: torch.device,
    cache_dequantized: bool = False,
) -> nn.Module:
    """
    Load compact AlphaSparseGPT layers into an existing dense HF model.
    Useful for evaluation.
    """
    if "alpha_sparsegpt_layers" not in ckpt:
        raise ValueError("Checkpoint has no 'alpha_sparsegpt_layers'.")

    layers = ckpt["alpha_sparsegpt_layers"]

    for layer_name, st in layers.items():
        old = get_module_by_name(model, layer_name)
        if not isinstance(old, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {layer_name}, got {type(old)}")

        shape = tuple(st["shape"])
        out_features, in_features = shape

        bias = old.bias.detach().clone() if old.bias is not None else None

        qlayer = AlphaSparseLinear(
            in_features=in_features,
            out_features=out_features,
            mask_packed=st["mask_packed"].to(device),
            values=st["values"].to(device),
            value_dtype=parse_dtype(st.get("value_dtype", "float16")),
            bias=bias.to(device) if bias is not None else None,
            cache_dequantized=cache_dequantized,
        )

        set_module_by_name(model, layer_name, qlayer)

    model.eval()
    return model


def load_alpha_sparse_checkpoint(
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
    cache_dequantized: bool = False,
) -> Tuple[nn.Module, Any, Dict[str, Any]]:
    """
    Convenience loader for eval scripts.

    Example:
        model, tokenizer, ckpt = load_alpha_sparse_checkpoint(...)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_id = ckpt["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    model.to(device)

    if "model" in ckpt and ckpt["model"]:
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"Loaded ckpt['model'] non-compressed params.")
        print(f"  missing keys    : {len(missing)}")
        print(f"  unexpected keys : {len(unexpected)}")

    model = apply_alpha_sparse_layers_to_model(
        model=model,
        ckpt=ckpt,
        device=device,
        cache_dequantized=cache_dequantized,
    )

    return model, tokenizer, ckpt


# =============================================================================
# Decoder block discovery and forward helpers
# =============================================================================

def get_base_model(model: nn.Module) -> nn.Module:
    """
    Most HF causal LMs store decoder under model.model.
    Fallbacks included for GPT-like names.
    """
    for attr in ("model", "transformer", "gpt_neox"):
        if hasattr(model, attr):
            return getattr(model, attr)
    return model


def find_decoder_layers(model: nn.Module) -> Tuple[str, nn.ModuleList]:
    """
    Common decoder layer paths:
        model.model.layers
        model.transformer.h
        model.gpt_neox.layers
    """
    candidates = [
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "model.decoder.layers",
    ]

    for path in candidates:
        try:
            obj = get_module_by_name(model, path)
            if isinstance(obj, (nn.ModuleList, list, tuple)) and len(obj) > 0:
                return path, obj
        except Exception:
            pass

    # Generic fallback: choose largest ModuleList
    best_name = None
    best_mod = None
    best_len = 0

    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > best_len:
            best_name = name
            best_mod = mod
            best_len = len(mod)

    if best_mod is None:
        raise RuntimeError("Could not find decoder blocks.")

    return str(best_name), best_mod


def get_input_embeddings_module(model: nn.Module) -> nn.Module:
    emb = model.get_input_embeddings()
    if emb is None:
        raise RuntimeError("model.get_input_embeddings() returned None.")
    return emb


def maybe_get_norm(model: nn.Module) -> Optional[nn.Module]:
    base = get_base_model(model)
    for name in ("norm", "final_layernorm", "ln_f"):
        if hasattr(base, name):
            return getattr(base, name)
    return None


def make_position_ids(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)


def make_cache_position(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, device=device, dtype=torch.long)


def call_rotary_emb_if_available(model: nn.Module, hidden: torch.Tensor, position_ids: torch.Tensor):
    """
    Newer Mistral/LLaMA implementations pass position_embeddings=(cos, sin)
    into decoder blocks.
    """
    base = get_base_model(model)
    rotary = getattr(base, "rotary_emb", None)
    if rotary is None:
        return None

    try:
        return rotary(hidden, position_ids)
    except TypeError:
        try:
            return rotary(hidden, seq_len=hidden.size(1))
        except TypeError:
            return None


@torch.no_grad()
def forward_decoder_block(
    model: nn.Module,
    block: nn.Module,
    hidden: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Robust manual decoder block forward.

    Works with Mistral/LLaMA-style blocks that may require:
        hidden_states
        attention_mask
        position_ids
        cache_position
        position_embeddings
        use_cache
    """
    block_dtype = get_first_parameter_dtype(block)
    block_device = get_first_parameter_device(block)

    hidden = hidden.to(device=block_device, dtype=block_dtype)

    bsz, seq_len, _ = hidden.shape

    if position_ids is None:
        position_ids = make_position_ids(bsz, seq_len, block_device)
    else:
        position_ids = position_ids.to(block_device)

    cache_position = make_cache_position(seq_len, block_device)
    position_embeddings = call_rotary_emb_if_available(model, hidden, position_ids)

    sig = inspect.signature(block.forward)
    kwargs: Dict[str, Any] = {}

    if "hidden_states" in sig.parameters:
        kwargs["hidden_states"] = hidden
    else:
        # Some older blocks take hidden as positional arg.
        kwargs = {}

    if "attention_mask" in sig.parameters:
        kwargs["attention_mask"] = attention_mask

    if "position_ids" in sig.parameters:
        kwargs["position_ids"] = position_ids

    if "cache_position" in sig.parameters:
        kwargs["cache_position"] = cache_position

    if "position_embeddings" in sig.parameters and position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings

    if "use_cache" in sig.parameters:
        kwargs["use_cache"] = False

    if "past_key_value" in sig.parameters:
        kwargs["past_key_value"] = None

    if "past_key_values" in sig.parameters:
        kwargs["past_key_values"] = None

    try:
        if kwargs:
            out = block(**kwargs)
        else:
            out = block(hidden)
    except TypeError:
        # fallback for older signatures
        try:
            out = block(hidden, attention_mask=attention_mask, position_ids=position_ids)
        except TypeError:
            out = block(hidden, position_ids=position_ids)

    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return out


@torch.no_grad()
def compute_initial_hidden_cache(
    model: nn.Module,
    calib_tokens: torch.Tensor,
    batch_size: int,
    device: torch.device,
    hidden_cache_dtype: torch.dtype,
) -> torch.Tensor:
    emb = get_input_embeddings_module(model)
    emb_device = get_first_parameter_device(emb)

    n = calib_tokens.size(0)
    outs = []

    t0 = now()
    print("\nComputing initial embedding hidden cache...")

    for i in range(0, n, batch_size):
        toks = calib_tokens[i:i + batch_size].to(emb_device)
        h = emb(toks)
        outs.append(h.detach().to("cpu", dtype=hidden_cache_dtype))

        done = min(i + batch_size, n)
        print(
            f"\r  embeddings: {done}/{n} ({100.0 * done / n:.1f}%) "
            f"elapsed={format_seconds(now() - t0)}",
            end="",
            flush=True,
        )

    print()
    hidden = torch.cat(outs, dim=0).contiguous()
    print(f"Initial hidden cache shape: {tuple(hidden.shape)}, dtype={hidden.dtype}")
    return hidden


@torch.no_grad()
def run_block_on_hidden_cache(
    model: nn.Module,
    block: nn.Module,
    hidden_cache: torch.Tensor,
    batch_size: int,
    hidden_cache_dtype: torch.dtype,
    desc: str,
) -> torch.Tensor:
    n = hidden_cache.size(0)
    outs = []
    t0 = now()

    for i in range(0, n, batch_size):
        hidden = hidden_cache[i:i + batch_size]
        bsz, seq_len, _ = hidden.shape
        block_device = get_first_parameter_device(block)

        position_ids = make_position_ids(bsz, seq_len, block_device)

        out = forward_decoder_block(
            model=model,
            block=block,
            hidden=hidden.to(block_device),
            position_ids=position_ids,
            attention_mask=None,
        )

        outs.append(out.detach().to("cpu", dtype=hidden_cache_dtype))

        done = min(i + batch_size, n)
        print(
            f"\r  {desc}: {done}/{n} ({100.0 * done / n:.1f}%) "
            f"elapsed={format_seconds(now() - t0)}",
            end="",
            flush=True,
        )

    print()
    return torch.cat(outs, dim=0).contiguous()


# =============================================================================
# AlphaPruning metric
# =============================================================================

@torch.no_grad()
def hill_alpha_from_eigs(eigs: torch.Tensor, tail_fraction: float = 0.20, min_tail: int = 32) -> float:
    """
    Hill estimator on top-tail eigenvalues.

    AlphaPruning uses PL_Alpha_Hill:
        alpha = 1 + k / sum_i log(lambda_i / lambda_min_tail)

    Lower alpha => heavier tail => more important => prune less.
    """
    eigs = eigs.detach().float().cpu()
    eigs = eigs[torch.isfinite(eigs)]
    eigs = eigs[eigs > 0]

    if eigs.numel() < max(4, min_tail):
        return float("nan")

    eigs, _ = torch.sort(eigs)
    n = eigs.numel()

    k = int(round(tail_fraction * n))
    k = max(min_tail, k)
    k = min(k, n - 1)

    tail = eigs[-k:]
    threshold = eigs[-k - 1].clamp(min=1e-30)

    logs = torch.log((tail / threshold).clamp(min=1.0 + 1e-12))
    denom = logs.sum().item()

    if not math.isfinite(denom) or denom <= 1e-12:
        return float("nan")

    return float(1.0 + k / denom)


@torch.no_grad()
def compute_weight_alpha_metric(
    layer: nn.Linear,
    tail_fraction: float,
    max_svd_dim: int,
    device: torch.device,
) -> float:
    """
    Compute PL_Alpha_Hill metric from W.

    For W [out, in], nonzero eigenvalues of W^T W equal singular_values(W)^2.
    We use torch.linalg.svdvals. For very large matrices, optionally subsample columns/rows.
    """
    W = layer.weight.detach()

    # Move to CPU or chosen device. CPU is safer for memory; CUDA is faster if available.
    W = W.to(device=device, dtype=torch.float32)

    rows, cols = W.shape

    # Optional deterministic subsampling for very large matrices to keep spectral analysis practical.
    if max_svd_dim > 0:
        if rows > max_svd_dim:
            idx = torch.linspace(0, rows - 1, max_svd_dim, device=W.device).long()
            W = W.index_select(0, idx)
        if cols > max_svd_dim:
            idx = torch.linspace(0, cols - 1, max_svd_dim, device=W.device).long()
            W = W.index_select(1, idx)

    try:
        s = torch.linalg.svdvals(W)
        eigs = s.square()
        alpha = hill_alpha_from_eigs(eigs, tail_fraction=tail_fraction)
    except RuntimeError as exc:
        print(f"[warn] SVD failed for shape {tuple(layer.weight.shape)}: {exc}")
        alpha = float("nan")

    del W
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return alpha


@torch.no_grad()
def compute_alpha_metrics_for_blocks(
    model: nn.Module,
    decoder_layers: Sequence[nn.Module],
    decoder_path: str,
    suffixes: Tuple[str, ...],
    include: str,
    exclude: str,
    tail_fraction: float,
    max_svd_dim: int,
    alpha_device: torch.device,
) -> Tuple[Dict[str, float], Dict[int, float], Dict[int, int], Dict[int, List[str]]]:
    """
    Compute per-layer and per-block AlphaPruning metrics.

    Pure AlphaPruning allocation uses block-level metric:
        alpha_block = parameter-weighted mean alpha over selected matrices in block.
    """
    layer_alpha: Dict[str, float] = {}
    block_alpha: Dict[int, float] = {}
    block_params: Dict[int, int] = {}
    block_layer_names: Dict[int, List[str]] = {}

    print("\nComputing AlphaPruning heavy-tail metrics...")
    t0 = now()

    for bidx, block in enumerate(decoder_layers):
        names = find_selected_linear_names_in_block(
            model=model,
            block_index=bidx,
            block=block,
            decoder_path=decoder_path,
            suffixes=suffixes,
            include=include,
            exclude=exclude,
        )

        block_layer_names[bidx] = names

        weighted_sum = 0.0
        total_params = 0

        print(f"\n  Block {bidx}: {len(names)} selected Linear layers")

        for lname in names:
            layer = get_module_by_name(model, lname)
            if not isinstance(layer, nn.Linear):
                continue

            params = layer.weight.numel()
            alpha = compute_weight_alpha_metric(
                layer=layer,
                tail_fraction=tail_fraction,
                max_svd_dim=max_svd_dim,
                device=alpha_device,
            )

            if not math.isfinite(alpha):
                print(f"    {lname}: alpha=nan -> ignored")
                continue

            layer_alpha[lname] = alpha
            weighted_sum += alpha * params
            total_params += params

            print(f"    {lname}: alpha={alpha:.6f}, params={params:,}")

        if total_params > 0:
            block_alpha[bidx] = weighted_sum / total_params
            block_params[bidx] = total_params
        else:
            block_alpha[bidx] = float("nan")
            block_params[bidx] = 0

        print(
            f"  Block {bidx} weighted alpha: {block_alpha[bidx]:.6f}, "
            f"selected params={block_params[bidx]:,}"
        )

    print(f"\nAlpha metric computation time: {format_seconds(now() - t0)}")
    return layer_alpha, block_alpha, block_params, block_layer_names


def solve_weighted_sparsities_with_clamp(
    base: torch.Tensor,
    weights: torch.Tensor,
    target: float,
    min_sparsity: float,
    max_sparsity: float,
) -> torch.Tensor:
    """
    Find sparsities s_i = clamp(a * base_i, min, max) so weighted mean equals target.

    This is robust version of AlphaPruning's eta normalization.
    """
    base = base.float()
    weights = weights.float()

    if weights.sum().item() <= 0:
        raise ValueError("weights sum is zero.")

    target = float(target)

    min_possible = float((torch.full_like(base, min_sparsity) * weights).sum() / weights.sum())
    max_possible = float((torch.full_like(base, max_sparsity) * weights).sum() / weights.sum())

    if target < min_possible - 1e-8 or target > max_possible + 1e-8:
        raise ValueError(
            f"Target sparsity {target} not achievable with clamps "
            f"[{min_sparsity}, {max_sparsity}]. Achievable range: "
            f"[{min_possible}, {max_possible}]"
        )

    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        s = torch.clamp(mid * base, min_sparsity, max_sparsity)
        avg = float((s * weights).sum() / weights.sum())
        if avg < target:
            lo = mid
        else:
            hi = mid

    return torch.clamp(hi * base, min_sparsity, max_sparsity)


def allocate_alpha_sparsities(
    block_alpha: Dict[int, float],
    block_params: Dict[int, int],
    target_sparsity: float,
    alpha_s1: float,
    alpha_s2: float,
    min_sparsity: float,
    max_sparsity: float,
) -> Dict[int, float]:
    """
    AlphaPruning mapping:
        lower alpha => lower sparsity
        higher alpha => higher sparsity

    base_i in [s1, s2], then eta scaling to hit global target.
    """
    valid = [
        b for b, a in block_alpha.items()
        if math.isfinite(a) and block_params.get(b, 0) > 0
    ]

    if not valid:
        raise RuntimeError("No valid AlphaPruning block metrics.")

    alphas = torch.tensor([block_alpha[b] for b in valid], dtype=torch.float32)
    weights = torch.tensor([block_params[b] for b in valid], dtype=torch.float32)

    amin = float(alphas.min())
    amax = float(alphas.max())

    if abs(amax - amin) < 1e-12:
        base = torch.full_like(alphas, fill_value=target_sparsity)
    else:
        normalized = (alphas - amin) / (amax - amin)
        base = normalized * (alpha_s2 - alpha_s1) + alpha_s1

    sparsities = solve_weighted_sparsities_with_clamp(
        base=base,
        weights=weights,
        target=target_sparsity,
        min_sparsity=min_sparsity,
        max_sparsity=max_sparsity,
    )

    out: Dict[int, float] = {}
    for b, s in zip(valid, sparsities.tolist()):
        out[b] = float(s)

    # Blocks with no selected params get zero.
    for b in block_alpha:
        if b not in out:
            out[b] = 0.0

    actual = sum(out[b] * block_params.get(b, 0) for b in out) / max(
        1, sum(block_params.get(b, 0) for b in out)
    )

    print("\n=== AlphaPruning sparsity allocation ===")
    print(f"Target global sparsity : {target_sparsity:.4f}")
    print(f"Actual weighted target : {actual:.4f}")
    print(f"Alpha range            : [{amin:.6f}, {amax:.6f}]")
    print(f"Base interval s1/s2    : [{alpha_s1:.4f}, {alpha_s2:.4f}]")
    print(f"Clamp min/max          : [{min_sparsity:.4f}, {max_sparsity:.4f}]")
    print()

    for b in sorted(out):
        print(
            f"  block {b:02d}: alpha={block_alpha[b]:.6f} "
            f"params={block_params.get(b, 0):,} sparsity={out[b]:.4f}"
        )

    return out


# =============================================================================
# Linear selection
# =============================================================================

def find_selected_linear_names_in_block(
    model: nn.Module,
    block_index: int,
    block: nn.Module,
    decoder_path: str,
    suffixes: Tuple[str, ...],
    include: str,
    exclude: str,
) -> List[str]:
    out = []

    block_prefix = f"{decoder_path}.{block_index}"

    for local_name, mod in block.named_modules():
        if not isinstance(mod, nn.Linear):
            continue

        full_name = block_prefix if local_name == "" else f"{block_prefix}.{local_name}"

        if include and include not in full_name:
            continue
        if exclude and exclude in full_name:
            continue
        if not full_name.endswith(suffixes):
            continue

        out.append(full_name)

    return out


def build_partial_state_dict_excluding_sparse_layers(
    model: nn.Module,
    sparse_layers: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    excluded = {f"{name}.weight" for name in sparse_layers.keys()}
    sd = model.state_dict()
    out = {}

    for k, v in sd.items():
        if k in excluded:
            continue
        out[k] = v.detach().cpu()

    return out


# =============================================================================
# Hessian collection
# =============================================================================

class HessianCollector:
    def __init__(
        self,
        layer: nn.Linear,
        hessian_device: torch.device,
        hessian_dtype: torch.dtype,
        sanitize: bool = True,
        clamp_abs: float = 0.0,
    ):
        self.layer = layer
        self.in_features = int(layer.in_features)
        self.hessian_device = hessian_device
        self.hessian_dtype = hessian_dtype
        self.sanitize = bool(sanitize)
        self.clamp_abs = float(clamp_abs)

        self.H = torch.zeros(
            (self.in_features, self.in_features),
            device=hessian_device,
            dtype=hessian_dtype,
        )
        self.nsamples = 0
        self.nonfinite_batches = 0
        self.handle = None

    def hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        x = inputs[0]
        if not torch.is_tensor(x):
            return

        x = x.detach().reshape(-1, x.size(-1))

        if self.sanitize:
            bad = ~torch.isfinite(x)
            if bad.any():
                self.nonfinite_batches += 1
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.clamp_abs > 0:
            x = x.clamp(min=-self.clamp_abs, max=self.clamp_abs)

        x = x.to(device=self.hessian_device, dtype=self.hessian_dtype)

        # CRUCIAL: SparseGPT Hessian convention.
        self.H.addmm_(x.t(), x, beta=1.0, alpha=2.0)
        self.nsamples += x.size(0)

    def register(self):
        self.handle = self.layer.register_forward_pre_hook(self.hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def choose_hessian_device(
    layer: nn.Linear,
    main_device: torch.device,
    large_layer_cpu_threshold: int,
) -> torch.device:
    """
    For Mistral:
        q/k/v/o/gate/up have in_features 4096 -> CUDA OK.
        down_proj has in_features 14336 -> H is huge, CPU safer.
    """
    if large_layer_cpu_threshold <= 0:
        return torch.device("cpu")
    if int(layer.in_features) > large_layer_cpu_threshold:
        return torch.device("cpu")
    return main_device


@torch.no_grad()
def collect_block_hessians(
    model: nn.Module,
    block: nn.Module,
    block_layer_names: List[str],
    hidden_cache: torch.Tensor,
    batch_size: int,
    main_device: torch.device,
    hessian_dtype: torch.dtype,
    large_layer_cpu_threshold: int,
    sanitize_activations: bool,
    activation_clamp_abs: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], Dict[str, torch.device]]:
    collectors: Dict[str, HessianCollector] = {}
    hessian_devices: Dict[str, torch.device] = {}

    for lname in block_layer_names:
        layer = get_module_by_name(model, lname)
        if not isinstance(layer, nn.Linear):
            continue

        hdev = choose_hessian_device(
            layer=layer,
            main_device=main_device,
            large_layer_cpu_threshold=large_layer_cpu_threshold,
        )
        hessian_devices[lname] = hdev

        collector = HessianCollector(
            layer=layer,
            hessian_device=hdev,
            hessian_dtype=hessian_dtype,
            sanitize=sanitize_activations,
            clamp_abs=activation_clamp_abs,
        )
        collector.register()
        collectors[lname] = collector

    n = hidden_cache.size(0)
    t0 = now()

    for i in range(0, n, batch_size):
        hidden = hidden_cache[i:i + batch_size]
        bsz, seq_len, _ = hidden.shape

        block_device = get_first_parameter_device(block)
        position_ids = make_position_ids(bsz, seq_len, block_device)

        _ = forward_decoder_block(
            model=model,
            block=block,
            hidden=hidden.to(block_device),
            position_ids=position_ids,
            attention_mask=None,
        )

        done = min(i + batch_size, n)
        print(
            f"\r  block Hessian pass: {done}/{n} ({100.0 * done / n:.1f}%) "
            f"elapsed={format_seconds(now() - t0)}",
            end="",
            flush=True,
        )

    print()

    Hs: Dict[str, torch.Tensor] = {}
    nsamples: Dict[str, int] = {}

    for lname, collector in collectors.items():
        collector.remove()

        if collector.nonfinite_batches > 0:
            print(
                f"  [warn] {lname}: sanitized non-finite activations in "
                f"{collector.nonfinite_batches} hook calls."
            )

        H = collector.H
        if not torch.isfinite(H).all():
            bad = int((~torch.isfinite(H)).sum().item())
            print(f"  [warn] {lname}: H has {bad} non-finite entries; replacing with 0.")
            H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

        Hs[lname] = H
        nsamples[lname] = collector.nsamples

    return Hs, nsamples, hessian_devices


# =============================================================================
# SparseGPT pruning
# =============================================================================

@dataclass
class SparseGPTResult:
    mask: torch.Tensor
    sparse_weight: torch.Tensor
    sparsity: float
    pruned_count: int
    total_count: int
    used_damp: float
    h_inverse_seconds: float
    prune_seconds: float


@torch.no_grad()
def robust_cholesky_inverse_chol(
    H: torch.Tensor,
    percdamp: float,
    min_damp: float = 1e-8,
    max_tries: int = 10,
) -> Tuple[torch.Tensor, float]:
    """
    Return upper Cholesky factor of inverse damped Hessian:
        Hinv_chol = chol(inv(H + damp I), upper=True)

    This is what SparseGPT-style compensation uses.
    """
    H = H.to(torch.float64)
    H = 0.5 * (H + H.T)

    diag = torch.diag(H)
    if not torch.isfinite(diag).all():
        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        diag = torch.diag(H)

    diag_abs = diag.abs()
    mean_diag = float(diag_abs.mean().item())
    max_diag = float(diag_abs.max().item())

    base = max(mean_diag, min_damp)
    n = H.size(0)
    ar = torch.arange(n, device=H.device)

    multipliers = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]
    multipliers = multipliers[:max_tries]

    last_error: Optional[Exception] = None

    for mult in multipliers:
        damp = max(percdamp * mult * base, min_damp)

        H_try = H.clone()
        H_try[ar, ar] += damp

        try:
            chol = torch.linalg.cholesky(H_try)
            Hinv = torch.cholesky_inverse(chol)
            Hinv = 0.5 * (Hinv + Hinv.T)
            Hinv_chol = torch.linalg.cholesky(Hinv, upper=True)
            return Hinv_chol, damp
        except RuntimeError as exc:
            last_error = exc

    # Fallback based on max diagonal.
    base2 = max(max_diag, min_damp)
    for mult in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
        damp = max(percdamp * mult * base2, min_damp)

        H_try = H.clone()
        H_try[ar, ar] += damp

        try:
            chol = torch.linalg.cholesky(H_try)
            Hinv = torch.cholesky_inverse(chol)
            Hinv = 0.5 * (Hinv + Hinv.T)
            Hinv_chol = torch.linalg.cholesky(Hinv, upper=True)
            return Hinv_chol, damp
        except RuntimeError as exc:
            last_error = exc

    # Last-resort eigenvalue projection. Numerically expensive but robust.
    print("      [warn] Cholesky failed; using eigenvalue projection fallback.")
    evals, evecs = torch.linalg.eigh(H)
    floor = max(percdamp * base, min_damp)
    evals = torch.clamp(evals, min=floor)
    Hinv = (evecs * (1.0 / evals).unsqueeze(0)) @ evecs.T
    Hinv = 0.5 * (Hinv + Hinv.T)
    Hinv_chol = torch.linalg.cholesky(Hinv, upper=True)
    return Hinv_chol, floor


@torch.no_grad()
def make_sparsegpt_mask(
    W: torch.Tensor,
    Hinv_chol: torch.Tensor,
    sparsity: float,
    row_wise: bool = True,
) -> torch.Tensor:
    """
    SparseGPT saliency:
        score_ij = W_ij^2 / diag(Hinv_chol)_j^2

    Smaller score -> prune.
    """
    rows, cols = W.shape

    if sparsity <= 0:
        return torch.ones((rows, cols), dtype=torch.bool, device=W.device)
    if sparsity >= 1:
        return torch.zeros((rows, cols), dtype=torch.bool, device=W.device)

    diag = torch.diag(Hinv_chol).to(device=W.device, dtype=W.dtype)
    denom = diag.square().clamp(min=1e-12).view(1, -1)
    score = W.square() / denom

    if row_wise:
        n_prune = int(round(sparsity * cols))
        n_prune = max(0, min(n_prune, cols))

        if n_prune == 0:
            return torch.ones_like(W, dtype=torch.bool)
        if n_prune == cols:
            return torch.zeros_like(W, dtype=torch.bool)

        prune_idx = torch.topk(
            score,
            k=n_prune,
            dim=1,
            largest=False,
            sorted=False,
        ).indices

        mask = torch.ones_like(W, dtype=torch.bool)
        row_idx = torch.arange(rows, device=W.device).view(-1, 1).expand_as(prune_idx)
        mask[row_idx, prune_idx] = False
        return mask

    total = rows * cols
    n_prune = int(round(sparsity * total))
    n_prune = max(0, min(n_prune, total))

    if n_prune == 0:
        return torch.ones_like(W, dtype=torch.bool)
    if n_prune == total:
        return torch.zeros_like(W, dtype=torch.bool)

    flat = score.reshape(-1)
    prune_idx = torch.topk(flat, k=n_prune, largest=False, sorted=False).indices
    mask_flat = torch.ones(total, dtype=torch.bool, device=W.device)
    mask_flat[prune_idx] = False
    return mask_flat.view(rows, cols)


@torch.no_grad()
def sparsegpt_prune_linear(
    layer: nn.Linear,
    H: torch.Tensor,
    sparsity: float,
    percdamp: float,
    blocksize: int,
    row_wise: bool,
    compress_device: torch.device,
) -> SparseGPTResult:
    """
    Pure SparseGPT-style pruning with column-wise error compensation.

    CRUCIAL:
        q = w if kept
        q = 0 if pruned
        err = (w - q) / Hinv_chol[j,j]
        W_future -= err @ Hinv_chol[j, future]
    """
    t_start = now()

    orig_device = layer.weight.device
    orig_dtype = layer.weight.dtype

    W_orig = layer.weight.detach().to(device=compress_device, dtype=torch.float32).clone()
    rows, cols = W_orig.shape

    H = H.to(compress_device)
    if H.dtype != torch.float64:
        H = H.to(torch.float64)

    # Stabilize dead columns.
    diag = torch.diag(H)
    dead = diag.abs() <= 1e-12
    if dead.any():
        ar = torch.arange(cols, device=H.device)
        H[dead, :] = 0
        H[:, dead] = 0
        H[ar[dead], ar[dead]] = 1.0
        W_orig[:, dead] = 0.0

    inv_t0 = now()
    Hinv_chol, used_damp = robust_cholesky_inverse_chol(H, percdamp=percdamp)
    h_inverse_seconds = now() - inv_t0

    Hinv_chol = Hinv_chol.to(device=compress_device, dtype=torch.float32)

    mask = make_sparsegpt_mask(
        W=W_orig,
        Hinv_chol=Hinv_chol,
        sparsity=sparsity,
        row_wise=row_wise,
    )

    W = W_orig.clone()
    Q = torch.zeros_like(W)

    prune_t0 = now()

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)

        Hinv1 = Hinv_chol[i1:i2, i1:i2].contiguous()

        for i in range(count):
            col = i1 + i
            d = Hinv1[i, i]

            if d.abs().item() < 1e-12:
                d = torch.tensor(1e-12, device=W.device, dtype=W.dtype)

            w = W1[:, i]
            keep = mask[:, col]

            # q is pruned weight column.
            q = torch.where(keep, w, torch.zeros_like(w))
            Q1[:, i] = q
            Q[:, col] = q

            # CRUCIAL SparseGPT compensation.
            err = (w - q) / d
            Err1[:, i] = err

            if i + 1 < count:
                W1[:, i + 1:count] -= (
                    err.unsqueeze(1)
                    @ Hinv1[i, i + 1:count].unsqueeze(0)
                )

        W[:, i1:i2] = Q1

        if i2 < cols:
            W[:, i2:cols] -= Err1 @ Hinv_chol[i1:i2, i2:cols]

        done = i2
        print(
            f"\r      SparseGPT columns: {done}/{cols} "
            f"({100.0 * done / cols:.1f}%)",
            end="",
            flush=True,
        )

    print()

    sparse_weight = Q * mask.to(Q.dtype)

    # Replace real model weight so future blocks see compressed previous blocks.
    layer.weight.data.copy_(sparse_weight.to(device=orig_device, dtype=orig_dtype))

    kept = int(mask.sum().item())
    total = rows * cols
    pruned = total - kept
    actual_sparsity = pruned / float(total)

    return SparseGPTResult(
        mask=mask.detach().cpu(),
        sparse_weight=sparse_weight.detach().cpu(),
        sparsity=actual_sparsity,
        pruned_count=pruned,
        total_count=total,
        used_damp=float(used_damp),
        h_inverse_seconds=float(h_inverse_seconds),
        prune_seconds=float(now() - prune_t0),
    )


# =============================================================================
# Main blockwise compression
# =============================================================================

@torch.no_grad()
def compress_model_alpha_sparsegpt(
    model: nn.Module,
    calib_tokens: torch.Tensor,
    decoder_path: str,
    decoder_layers: Sequence[nn.Module],
    block_layer_names: Dict[int, List[str]],
    block_sparsities: Dict[int, float],
    batch_size: int,
    main_device: torch.device,
    hidden_cache_dtype: torch.dtype,
    hessian_dtype: torch.dtype,
    percdamp: float,
    blocksize: int,
    row_wise: bool,
    large_layer_cpu_threshold: int,
    sanitize_activations: bool,
    activation_clamp_abs: float,
    value_dtype: torch.dtype,
) -> Dict[str, Any]:
    sparse_layers: Dict[str, Any] = {}

    hidden_cache = compute_initial_hidden_cache(
        model=model,
        calib_tokens=calib_tokens,
        batch_size=batch_size,
        device=main_device,
        hidden_cache_dtype=hidden_cache_dtype,
    )

    total_blocks = len(decoder_layers)
    global_t0 = now()

    for bidx, block in enumerate(decoder_layers):
        block_t0 = now()
        selected = block_layer_names.get(bidx, [])
        block_sparsity = float(block_sparsities.get(bidx, 0.0))

        print("\n" + "=" * 100)
        print(f"BLOCK {bidx}/{total_blocks - 1}")
        print(f"Block target sparsity : {block_sparsity:.4f}")
        print(f"Selected linears      : {len(selected)}")
        print(f"Hidden cache entering : {tuple(hidden_cache.shape)}, dtype={hidden_cache.dtype}")
        print_cuda_memory("CUDA memory")

        for lname in selected:
            layer = get_module_by_name(model, lname)
            print(
                f"  - {lname}: shape={tuple(layer.weight.shape)} "
                f"dtype={layer.weight.dtype}"
            )

        if selected:
            print("  Collecting Hessians for selected Linear layers...")
            Hs, nsamples, hessian_devices = collect_block_hessians(
                model=model,
                block=block,
                block_layer_names=selected,
                hidden_cache=hidden_cache,
                batch_size=batch_size,
                main_device=main_device,
                hessian_dtype=hessian_dtype,
                large_layer_cpu_threshold=large_layer_cpu_threshold,
                sanitize_activations=sanitize_activations,
                activation_clamp_abs=activation_clamp_abs,
            )

            for lidx, lname in enumerate(selected, start=1):
                layer = get_module_by_name(model, lname)
                if not isinstance(layer, nn.Linear):
                    continue

                H = Hs[lname]
                math_device = hessian_devices[lname]

                print("\n" + "-" * 100)
                print(f"  [{lidx}/{len(selected)}] AlphaSparseGPT pruning: {lname}")
                print(f"      shape        : {tuple(layer.weight.shape)}")
                print(f"      in_features  : {layer.in_features}")
                print(f"      out_features : {layer.out_features}")
                print(f"      sparsity     : {block_sparsity:.4f}")
                print(f"      H shape      : {tuple(H.shape)}")
                print(f"      H samples    : {nsamples[lname]}")
                print(f"      math device  : {math_device}")
                print(f"      percdamp     : {percdamp}")
                print(f"      blocksize    : {blocksize}")
                print(f"      row_wise     : {row_wise}")

                result = sparsegpt_prune_linear(
                    layer=layer,
                    H=H,
                    sparsity=block_sparsity,
                    percdamp=percdamp,
                    blocksize=blocksize,
                    row_wise=row_wise,
                    compress_device=math_device,
                )

                mask_packed = pack_bool_mask_rows(result.mask.bool())
                values = result.sparse_weight[result.mask.bool()].to(value_dtype).contiguous()

                layer_state = {
                    "shape": list(result.sparse_weight.shape),
                    "mask_packed": mask_packed.cpu(),
                    "values": values.cpu(),
                    "value_dtype": str(value_dtype).replace("torch.", ""),
                    "sparsity": float(result.sparsity),
                    "target_sparsity": float(block_sparsity),
                    "pruned_count": int(result.pruned_count),
                    "total_count": int(result.total_count),
                    "used_damp": float(result.used_damp),
                    "h_inverse_seconds": float(result.h_inverse_seconds),
                    "prune_seconds": float(result.prune_seconds),
                    "format": "mask_packed_plus_values_row_major",
                }

                sparse_layers[lname] = layer_state

                dense_bytes = result.total_count * 2
                sparse_bytes = int(mask_packed.numel()) + int(values.numel()) * torch.tensor([], dtype=value_dtype).element_size()

                print(f"      actual sparsity : {100.0 * result.sparsity:.2f}%")
                print(f"      used damping    : {result.used_damp:.6e}")
                print(f"      H inverse time  : {format_seconds(result.h_inverse_seconds)}")
                print(f"      pruning time    : {format_seconds(result.prune_seconds)}")
                print(f"      mask packed     : {tuple(mask_packed.shape)}")
                print(f"      kept values     : {tuple(values.shape)}")
                print(f"      dense bf16 est  : {dense_bytes:,} bytes")
                print(f"      sparse raw est  : {sparse_bytes:,} bytes")

                del H
                del result

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print("\n  Running pruned block to create next hidden cache...")
        hidden_cache = run_block_on_hidden_cache(
            model=model,
            block=block,
            hidden_cache=hidden_cache,
            batch_size=batch_size,
            hidden_cache_dtype=hidden_cache_dtype,
            desc=f"block {bidx} forward",
        )

        print(f"Finished block {bidx}. Block time: {format_seconds(now() - block_t0)}")
        print(f"Total elapsed: {format_seconds(now() - global_t0)}")
        print_cuda_memory("CUDA memory")

    return sparse_layers


# =============================================================================
# Checkpoint save
# =============================================================================

def save_alpha_sparse_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    out_path: str,
    model_id: str,
    meta: Dict[str, Any],
    sparse_layers: Dict[str, Any],
    keep_dequantized_state_dict: bool,
) -> None:
    ensure_dir_for_file(out_path)

    if keep_dequantized_state_dict:
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        model_field = "full_dense_pruned_state_dict"
    else:
        model_state = build_partial_state_dict_excluding_sparse_layers(model, sparse_layers)
        model_field = "non_sparse_parameters_only"

    ckpt = {
        "format": "hf_alpha_sparsegpt",
        "model_id": model_id,
        "config": model.config.to_dict() if hasattr(model, "config") else None,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", model_id),
        "alpha_sparsegpt_meta": meta,
        "alpha_sparsegpt_layers": sparse_layers,
        "model": model_state,
        "model_field_contents": model_field,
    }

    t0 = now()
    torch.save(ckpt, out_path)
    print(f"Saved checkpoint in {format_seconds(now() - t0)}: {out_path}")

    meta_path = out_path + ".meta.json"
    meta_json = dict(meta)
    meta_json["model_field_contents"] = model_field
    meta_json["compressed_layers"] = len(sparse_layers)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2)

    print(f"Saved meta JSON: {meta_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AlphaPruning + SparseGPT-style pure pruning for HF causal LMs."
    )

    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--calib", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--target_sparsity", type=float, required=True)
    parser.add_argument("--alpha_s1", type=float, default=0.50)
    parser.add_argument("--alpha_s2", type=float, default=0.90)
    parser.add_argument("--min_sparsity", type=float, default=0.05)
    parser.add_argument("--max_sparsity", type=float, default=0.95)

    parser.add_argument("--alpha_tail_fraction", type=float, default=0.20)
    parser.add_argument(
        "--alpha_max_svd_dim",
        type=int,
        default=4096,
        help="Subsample rows/cols for spectral metric if larger than this. 0 disables subsampling.",
    )
    parser.add_argument(
        "--alpha_device",
        type=str,
        default="cpu",
        help="cpu is safer; cuda is faster but uses VRAM.",
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--model_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--hidden_cache_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--hessian_dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
    )
    parser.add_argument(
        "--value_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Stored sparse surviving value dtype.",
    )

    parser.add_argument("--max_seq_len", type=int, default=0)
    parser.add_argument("--percdamp", type=float, default=0.1)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--row_wise", action="store_true", default=True)
    parser.add_argument("--global_within_layer", action="store_true")

    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")
    parser.add_argument("--suffixes", type=str, default="")

    parser.add_argument(
        "--large_layer_cpu_threshold",
        type=int,
        default=8192,
        help="If layer.in_features > threshold, Hessian/pruning runs on CPU. 0 forces CPU.",
    )

    parser.add_argument("--sanitize_activations", action="store_true", default=True)
    parser.add_argument(
        "--activation_clamp_abs",
        type=float,
        default=0.0,
        help="If >0, clamp activations before Hessian accumulation.",
    )

    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="")

    parser.add_argument(
        "--keep_dequantized_state_dict",
        action="store_true",
        help="Stores full dense pruned model state. Easier but much larger.",
    )

    args = parser.parse_args()

    script_t0 = now()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is not available.")

    if not (0.0 <= args.target_sparsity < 1.0):
        raise ValueError("--target_sparsity must be in [0, 1).")
    if args.blocksize <= 0:
        raise ValueError("--blocksize must be positive.")
    if args.global_within_layer:
        args.row_wise = False

    main_device = torch.device(args.device)
    model_dtype = parse_dtype(args.model_dtype)
    hidden_cache_dtype = parse_dtype(args.hidden_cache_dtype)
    hessian_dtype = parse_dtype(args.hessian_dtype)
    value_dtype = parse_dtype(args.value_dtype)
    alpha_device = torch.device(args.alpha_device)

    suffixes = parse_suffixes(args.suffixes)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("=" * 100)
    print("AlphaPruning + SparseGPT pure pruning")
    print("=" * 100)
    print(f"model_id                  : {args.model_id}")
    print(f"calib                     : {args.calib}")
    print(f"out                       : {args.out}")
    print(f"target_sparsity           : {args.target_sparsity}")
    print(f"alpha_s1/s2               : {args.alpha_s1}, {args.alpha_s2}")
    print(f"min/max sparsity          : {args.min_sparsity}, {args.max_sparsity}")
    print(f"alpha_tail_fraction       : {args.alpha_tail_fraction}")
    print(f"alpha_max_svd_dim         : {args.alpha_max_svd_dim}")
    print(f"alpha_device              : {alpha_device}")
    print(f"device                    : {main_device}")
    print(f"model_dtype               : {model_dtype}")
    print(f"hidden_cache_dtype        : {hidden_cache_dtype}")
    print(f"hessian_dtype             : {hessian_dtype}")
    print(f"value_dtype               : {value_dtype}")
    print(f"batch_size                : {args.batch_size}")
    print(f"max_seq_len               : {args.max_seq_len}")
    print(f"percdamp                  : {args.percdamp}")
    print(f"blocksize                 : {args.blocksize}")
    print(f"row_wise within layer     : {args.row_wise}")
    print(f"large_layer_cpu_threshold : {args.large_layer_cpu_threshold}")
    print(f"suffixes                  : {suffixes}")

    print("\nLoading tokenizer/model...")
    load_t0 = now()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
        "trust_remote_code": args.trust_remote_code,
    }

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()
    model.to(main_device)

    if hasattr(model, "config"):
        model.config.use_cache = False

    print(f"Loaded model in {format_seconds(now() - load_t0)}")
    print_cuda_memory("CUDA memory after model load")

    calib_tokens = load_calibration_tokens(args.calib)
    print(f"\nLoaded calibration tokens: {tuple(calib_tokens.shape)}")

    if args.max_seq_len > 0 and calib_tokens.size(1) > args.max_seq_len:
        calib_tokens = calib_tokens[:, :args.max_seq_len].contiguous()
        print(f"Trimmed calibration sequence length to {args.max_seq_len}")

    decoder_path, decoder_layers = find_decoder_layers(model)
    print(f"\nDecoder layers found: {decoder_path}")
    print(f"Number of decoder blocks: {len(decoder_layers)}")

    layer_alpha, block_alpha, block_params, block_layer_names = compute_alpha_metrics_for_blocks(
        model=model,
        decoder_layers=decoder_layers,
        decoder_path=decoder_path,
        suffixes=suffixes,
        include=args.include,
        exclude=args.exclude,
        tail_fraction=args.alpha_tail_fraction,
        max_svd_dim=args.alpha_max_svd_dim,
        alpha_device=alpha_device,
    )

    block_sparsities = allocate_alpha_sparsities(
        block_alpha=block_alpha,
        block_params=block_params,
        target_sparsity=args.target_sparsity,
        alpha_s1=args.alpha_s1,
        alpha_s2=args.alpha_s2,
        min_sparsity=args.min_sparsity,
        max_sparsity=args.max_sparsity,
    )

    sparse_layers = compress_model_alpha_sparsegpt(
        model=model,
        calib_tokens=calib_tokens,
        decoder_path=decoder_path,
        decoder_layers=decoder_layers,
        block_layer_names=block_layer_names,
        block_sparsities=block_sparsities,
        batch_size=args.batch_size,
        main_device=main_device,
        hidden_cache_dtype=hidden_cache_dtype,
        hessian_dtype=hessian_dtype,
        percdamp=args.percdamp,
        blocksize=args.blocksize,
        row_wise=args.row_wise,
        large_layer_cpu_threshold=args.large_layer_cpu_threshold,
        sanitize_activations=bool(args.sanitize_activations),
        activation_clamp_abs=float(args.activation_clamp_abs),
        value_dtype=value_dtype,
    )

    total_pruned = sum(int(v["pruned_count"]) for v in sparse_layers.values())
    total_weights = sum(int(v["total_count"]) for v in sparse_layers.values())
    actual_sparsity = total_pruned / float(total_weights) if total_weights > 0 else 0.0

    packed_mask_bytes = sum(int(v["mask_packed"].numel()) for v in sparse_layers.values())
    value_bytes = sum(
        int(v["values"].numel()) * torch.tensor([], dtype=value_dtype).element_size()
        for v in sparse_layers.values()
    )
    raw_sparse_bytes = packed_mask_bytes + value_bytes
    dense_bf16_bytes = total_weights * 2

    meta = {
        "method": "AlphaPruning_blockwise_allocation_plus_SparseGPT_pure_pruning",
        "model_id": args.model_id,
        "target_sparsity": float(args.target_sparsity),
        "actual_sparsity": float(actual_sparsity),
        "alpha_s1": float(args.alpha_s1),
        "alpha_s2": float(args.alpha_s2),
        "min_sparsity": float(args.min_sparsity),
        "max_sparsity": float(args.max_sparsity),
        "alpha_tail_fraction": float(args.alpha_tail_fraction),
        "alpha_max_svd_dim": int(args.alpha_max_svd_dim),
        "percdamp": float(args.percdamp),
        "blocksize": int(args.blocksize),
        "row_wise": bool(args.row_wise),
        "calibration_source": args.calib,
        "model_dtype": args.model_dtype,
        "hidden_cache_dtype": args.hidden_cache_dtype,
        "hessian_dtype": args.hessian_dtype,
        "value_dtype": args.value_dtype,
        "suffixes": list(suffixes),
        "decoder_path": decoder_path,
        "block_alpha": {str(k): float(v) for k, v in block_alpha.items()},
        "block_sparsities": {str(k): float(v) for k, v in block_sparsities.items()},
        "layer_alpha": {str(k): float(v) for k, v in layer_alpha.items()},
        "compressed_layers": int(len(sparse_layers)),
        "total_selected_weights": int(total_weights),
        "pruned_weights": int(total_pruned),
        "kept_weights": int(total_weights - total_pruned),
        "packed_mask_bytes": int(packed_mask_bytes),
        "value_bytes": int(value_bytes),
        "raw_sparse_bytes": int(raw_sparse_bytes),
        "dense_bf16_bytes": int(dense_bf16_bytes),
        "raw_compression_vs_bf16": float(dense_bf16_bytes / raw_sparse_bytes) if raw_sparse_bytes > 0 else None,
        "large_layer_cpu_threshold": int(args.large_layer_cpu_threshold),
        "attn_implementation": args.attn_implementation,
        "note": (
            "Pure AlphaPruning allocation is blockwise: lower PL_Alpha_Hill receives lower sparsity. "
            "Pruning inside each selected Linear uses SparseGPT saliency and column-wise Hessian compensation. "
            "Checkpoint stores compact sparse layers as packed mask plus kept values."
        ),
    }

    print("\nSaving checkpoint...")
    save_alpha_sparse_checkpoint(
        model=model,
        tokenizer=tokenizer,
        out_path=args.out,
        model_id=args.model_id,
        meta=meta,
        sparse_layers=sparse_layers,
        keep_dequantized_state_dict=bool(args.keep_dequantized_state_dict),
    )

    print("\nDone.")
    print(f"Compressed layers              : {len(sparse_layers)}")
    print(f"Total selected weights         : {total_weights:,}")
    print(f"Kept weights                   : {total_weights - total_pruned:,}")
    print(f"Pruned weights                 : {total_pruned:,}")
    print(f"Actual sparsity                : {100.0 * actual_sparsity:.2f}%")
    print(f"Packed mask bytes              : {packed_mask_bytes:,}")
    print(f"Surviving value bytes          : {value_bytes:,}")
    print(f"Raw sparse stored bytes        : {raw_sparse_bytes:,}")
    print(f"Dense BF16 bytes               : {dense_bf16_bytes:,}")

    if raw_sparse_bytes > 0:
        print(f"Raw compression vs BF16        : {dense_bf16_bytes / raw_sparse_bytes:.2f}x")

    print(f"Total script time              : {format_seconds(now() - script_t0)}")
    print_cuda_memory("CUDA memory")

    if args.keep_dequantized_state_dict:
        print("Checkpoint includes full dense pruned state_dict.")
    else:
        print("Checkpoint includes only non-sparse parameters plus compact sparse layer storage.")


if __name__ == "__main__":
    main()