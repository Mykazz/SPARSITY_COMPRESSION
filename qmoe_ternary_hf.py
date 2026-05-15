#!/usr/bin/env python3
"""
Dense-QMoE-style GPTQ compression for Hugging Face dense causal LMs.

Supports:
    - Mistral / Llama-like decoder-only HF models
    - Qwen2-like decoder-only HF models, with best-effort block forward handling

Implements:
    - sequential blockwise calibration
    - full Hessian H = 2 X^T X per Linear layer
    - collect Hessians for all selected Linear layers in a transformer block together
    - GPTQ column-wise compensation
    - ternary or 2-bit quantization grid
    - groupwise learned/search min-max grid
    - raw qidx storage, no dictionary/entropy coding yet

Important:
    - This is mathematically focused, not kernel optimized.
    - Runtime reconstruction is dense.
    - Dictionary coding is lossless and should be added only after quality is acceptable.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# General helpers
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


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_memory_string() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"

    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    return f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB max={peak:.2f}GB"


def get_model_param_dtype(model: nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def get_module_param_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        try:
            return next(module.buffers()).dtype
        except StopIteration:
            return fallback


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


def load_calibration_tokens(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and "tokens" in obj:
        tokens = obj["tokens"]
    elif torch.is_tensor(obj):
        tokens = obj
    else:
        raise ValueError("Calibration file must contain dict['tokens'] or a tensor.")

    if tokens.ndim != 2:
        raise ValueError(f"Expected calibration tokens [N, T], got {tuple(tokens.shape)}")

    return tokens.long()


def autocast_context_for_device(device: torch.device, compute_dtype: torch.dtype):
    enabled = device.type == "cuda" and compute_dtype in (torch.float16, torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=enabled)


# ============================================================
# HF architecture helpers
# ============================================================

def get_base_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    return model


def find_decoder_layers(model: nn.Module) -> Tuple[str, nn.ModuleList]:
    candidates = [
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
    ]

    for name in candidates:
        try:
            mod = get_module_by_name(model, name)
            if isinstance(mod, nn.ModuleList) and len(mod) > 0:
                return name, mod
        except Exception:
            pass

    best_name = ""
    best_mod = None
    best_len = 0

    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > best_len:
            if len(mod) == 0:
                continue
            has_linear = any(isinstance(m, nn.Linear) for m in mod[0].modules())
            if has_linear:
                best_name = name
                best_mod = mod
                best_len = len(mod)

    if best_mod is None:
        raise RuntimeError("Could not find decoder block ModuleList.")

    return best_name, best_mod


def make_position_ids(batch: int, seqlen: int, device: torch.device) -> torch.Tensor:
    return torch.arange(
        0,
        seqlen,
        dtype=torch.long,
        device=device,
    ).view(1, -1).expand(batch, -1)


@torch.no_grad()
def compute_position_embeddings(
    model: nn.Module,
    hidden: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    model_compute_dtype: torch.dtype,
) -> Optional[Any]:
    """
    Newer HF Mistral/Llama/Qwen decoder layers expect:
        position_embeddings=(cos, sin)

    In recent transformers versions, model.forward computes this once as:
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
    and passes it into every decoder layer.

    This function recreates that for manual blockwise forward.
    """
    if position_ids is None:
        return None

    base = get_base_model(model)

    if not hasattr(base, "rotary_emb"):
        return None

    rotary_emb = base.rotary_emb
    device = hidden.device

    with autocast_context_for_device(device, model_compute_dtype):
        try:
            return rotary_emb(hidden, position_ids)
        except TypeError:
            try:
                return rotary_emb(hidden)
            except TypeError:
                return None


@torch.no_grad()
def compute_initial_hidden_cache(
    model: nn.Module,
    tokens: torch.Tensor,
    batch_size: int,
    device: torch.device,
    model_compute_dtype: torch.dtype,
    hidden_cache_dtype: torch.dtype,
) -> torch.Tensor:
    base = get_base_model(model)

    if hasattr(base, "embed_tokens"):
        embed = base.embed_tokens
    elif hasattr(base, "wte"):
        embed = base.wte
    else:
        embed = model.get_input_embeddings()

    if embed is None:
        raise RuntimeError("Could not find input embeddings.")

    n, seqlen = tokens.shape
    chunks: List[torch.Tensor] = []
    t0 = now()

    for i in range(0, n, batch_size):
        ids = tokens[i:i + batch_size].to(device)

        with autocast_context_for_device(device, model_compute_dtype):
            h = embed(ids)

            if hasattr(base, "wpe"):
                pos = torch.arange(0, ids.size(1), device=device).view(1, -1)
                h = h + base.wpe(pos)

        chunks.append(h.detach().to("cpu", dtype=hidden_cache_dtype))

        done = min(i + batch_size, n)
        print(
            f"\r  embeddings: {done}/{n} ({100.0 * done / n:.1f}%) "
            f"elapsed={format_seconds(now() - t0)}",
            end="",
            flush=True,
        )

    print()
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def forward_decoder_block(
    model: nn.Module,
    block: nn.Module,
    hidden: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    model_compute_dtype: torch.dtype,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Robust manual decoder-block forward.

    Key fixes:
        1. hidden is cast to the block/model weight dtype.
        2. position_embeddings=(cos, sin) is computed and passed for newer Mistral/Llama.
    """
    device = hidden.device
    block_dtype = get_module_param_dtype(block, fallback=model_compute_dtype)

    if block_dtype in (torch.float16, torch.bfloat16, torch.float32):
        hidden = hidden.to(dtype=block_dtype)
    else:
        hidden = hidden.to(dtype=model_compute_dtype)

    cache_position = None
    if position_ids is not None:
        cache_position = position_ids[0]

    position_embeddings = compute_position_embeddings(
        model=model,
        hidden=hidden,
        position_ids=position_ids,
        model_compute_dtype=model_compute_dtype,
    )

    call_variants: List[Dict[str, Any]] = []

    # Newer Mistral/Llama/Qwen path.
    if position_embeddings is not None:
        call_variants.extend(
            [
                dict(
                    hidden_states=hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                ),
                dict(
                    hidden_states=hidden,
                    attention_mask=None,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                ),
                dict(
                    hidden_states=hidden,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                ),
                dict(
                    hidden_states=hidden,
                    attention_mask=None,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                ),
                dict(
                    hidden_states=hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    use_cache=False,
                ),
                dict(
                    hidden_states=hidden,
                    attention_mask=None,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    use_cache=False,
                ),
            ]
        )

    # Older fallback path.
    call_variants.extend(
        [
            dict(
                hidden_states=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ),
            dict(
                hidden_states=hidden,
                attention_mask=None,
                position_ids=position_ids,
                use_cache=False,
            ),
            dict(
                hidden_states=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ),
            dict(
                hidden_states=hidden,
                attention_mask=None,
                position_ids=position_ids,
            ),
            dict(
                hidden_states=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=False,
            ),
            dict(
                hidden_states=hidden,
                attention_mask=None,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=False,
            ),
        ]
    )

    last_error: Optional[Exception] = None

    with autocast_context_for_device(device, model_compute_dtype):
        for kwargs in call_variants:
            try:
                out = block(**kwargs)
                if isinstance(out, tuple):
                    return out[0]
                return out
            except (TypeError, ValueError) as exc:
                last_error = exc
                continue

        try:
            out = block(hidden)
            if isinstance(out, tuple):
                return out[0]
            return out
        except Exception as exc:
            raise RuntimeError(
                "Could not run decoder block. "
                f"Last argument-mismatch error: {last_error}. "
                f"Final error: {exc}"
            )


# ============================================================
# Layer selection
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


def should_compress_layer(
    full_name: str,
    include: str,
    exclude: str,
    suffixes: Tuple[str, ...],
    skip_attn_out: bool,
    skip_mlp_out: bool,
) -> bool:
    if include and include not in full_name:
        return False

    if exclude and exclude in full_name:
        return False

    if skip_attn_out and full_name.endswith("o_proj"):
        return False

    if skip_mlp_out and full_name.endswith("down_proj"):
        return False

    return full_name.endswith(suffixes)


def selected_linear_layers_in_block(
    block: nn.Module,
    block_prefix: str,
    include: str,
    exclude: str,
    suffixes: Tuple[str, ...],
    skip_attn_out: bool,
    skip_mlp_out: bool,
) -> List[Tuple[str, nn.Linear]]:
    out: List[Tuple[str, nn.Linear]] = []

    for subname, mod in block.named_modules():
        if not isinstance(mod, nn.Linear):
            continue

        full_name = f"{block_prefix}.{subname}" if subname else block_prefix

        if should_compress_layer(
            full_name=full_name,
            include=include,
            exclude=exclude,
            suffixes=suffixes,
            skip_attn_out=skip_attn_out,
            skip_mlp_out=skip_mlp_out,
        ):
            out.append((full_name, mod))

    return out


# ============================================================
# Hessian collection
# ============================================================

class BlockHessianCollector:
    """
    Collects full Hessians for multiple Linear layers in one block pass.

    H = 2 X^T X
    """

    def __init__(
        self,
        named_layers: List[Tuple[str, nn.Linear]],
        hessian_dtype: torch.dtype,
        main_device: torch.device,
        large_layer_cpu_threshold: int,
        sanitize_inputs: bool = True,
    ):
        self.named_layers = named_layers
        self.hessian_dtype = hessian_dtype
        self.main_device = main_device
        self.large_layer_cpu_threshold = int(large_layer_cpu_threshold)
        self.sanitize_inputs = bool(sanitize_inputs)

        self.H: Dict[str, torch.Tensor] = {}
        self.nsamples: Dict[str, int] = {}
        self.handles: List[Any] = []
        self.hessian_devices: Dict[str, torch.device] = {}

        for name, layer in named_layers:
            dev = self.choose_hessian_device(layer)
            self.hessian_devices[name] = dev

            self.H[name] = torch.zeros(
                (int(layer.in_features), int(layer.in_features)),
                dtype=hessian_dtype,
                device=dev,
            )

            self.nsamples[name] = 0

    def choose_hessian_device(self, layer: nn.Linear) -> torch.device:
        if self.large_layer_cpu_threshold <= 0:
            return torch.device("cpu")

        if int(layer.in_features) > self.large_layer_cpu_threshold:
            return torch.device("cpu")

        return self.main_device

    def make_hook(self, name: str):
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            x = inputs[0]

            if not torch.is_tensor(x):
                return

            x = x.detach().reshape(-1, x.shape[-1])

            if self.sanitize_inputs:
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            dev = self.hessian_devices[name]

            x = x.to(device=dev, dtype=self.hessian_dtype)

            local = 2.0 * x.t().matmul(x)
            local = torch.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0)

            self.H[name] += local
            self.nsamples[name] += x.shape[0]

        return hook

    def register(self) -> None:
        for name, layer in self.named_layers:
            self.handles.append(layer.register_forward_pre_hook(self.make_hook(name)))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


@torch.no_grad()
def collect_block_hessians(
    model: nn.Module,
    block: nn.Module,
    named_layers: List[Tuple[str, nn.Linear]],
    hidden_cache: torch.Tensor,
    batch_size: int,
    device: torch.device,
    model_compute_dtype: torch.dtype,
    hessian_dtype: torch.dtype,
    large_layer_cpu_threshold: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], Dict[str, torch.device]]:
    collector = BlockHessianCollector(
        named_layers=named_layers,
        hessian_dtype=hessian_dtype,
        main_device=device,
        large_layer_cpu_threshold=large_layer_cpu_threshold,
        sanitize_inputs=True,
    )

    collector.register()

    n, seqlen, _ = hidden_cache.shape
    t0 = now()

    try:
        for i in range(0, n, batch_size):
            h = hidden_cache[i:i + batch_size].to(
                device=device,
                dtype=model_compute_dtype,
            )

            position_ids = make_position_ids(h.size(0), seqlen, device)

            _ = forward_decoder_block(
                model=model,
                block=block,
                hidden=h,
                position_ids=position_ids,
                model_compute_dtype=model_compute_dtype,
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

    finally:
        collector.remove()

    return collector.H, collector.nsamples, collector.hessian_devices


# ============================================================
# Group helpers
# ============================================================

def get_num_groups(cols: int, groupsize: int) -> int:
    if groupsize == -1 or groupsize >= cols:
        return 1
    return math.ceil(cols / groupsize)


def get_group_bounds(group_idx: int, cols: int, groupsize: int) -> Tuple[int, int]:
    if groupsize == -1 or groupsize >= cols:
        return 0, cols
    g0 = group_idx * groupsize
    g1 = min((group_idx + 1) * groupsize, cols)
    return g0, g1


def get_group_index(col_idx: int, cols: int, groupsize: int) -> int:
    if groupsize == -1 or groupsize >= cols:
        return 0
    return col_idx // groupsize


def parse_rho_grid(raw: str) -> List[float]:
    raw = raw.strip()

    if raw:
        return [float(x.strip()) for x in raw.split(",") if x.strip()]

    vals: List[float] = []
    x = 0.50
    while x <= 1.000001:
        vals.append(round(x, 4))
        x += 0.05

    return vals


# ============================================================
# Grid quantization
# ============================================================

@torch.no_grad()
def make_candidate_levels(
    Wg: torch.Tensor,
    rho_neg: float,
    rho_pos: float,
    mode: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Wg:
        [rows, group_cols]

    ternary:
        symbol 0 -> 0
        symbol 1 -> a_minus
        symbol 2 -> a_plus

    2bit:
        4-level asymmetric grid, nearest-to-zero level forced to 0.
    """
    w_min = Wg.amin(dim=1)
    w_max = Wg.amax(dim=1)

    a_minus = torch.minimum(w_min * float(rho_neg), torch.zeros_like(w_min))
    a_plus = torch.maximum(w_max * float(rho_pos), torch.zeros_like(w_max))

    a_minus = torch.where(a_minus.abs() < eps, w_min, a_minus)
    a_plus = torch.where(a_plus.abs() < eps, w_max, a_plus)

    if mode == "ternary":
        zero = torch.zeros_like(a_minus)
        return torch.stack([zero, a_minus, a_plus], dim=1)

    if mode == "2bit":
        l0 = a_minus
        l3 = a_plus
        l1 = a_minus + (a_plus - a_minus) / 3.0
        l2 = a_minus + 2.0 * (a_plus - a_minus) / 3.0

        levels = torch.stack([l0, l1, l2, l3], dim=1)

        nearest_zero = levels.abs().argmin(dim=1)
        rows = torch.arange(levels.size(0), device=levels.device)
        levels[rows, nearest_zero] = 0.0

        return levels

    raise ValueError("--quant_mode must be 'ternary' or '2bit'.")


@torch.no_grad()
def quantize_matrix_to_levels(
    Wg: torch.Tensor,
    levels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dist = (Wg.unsqueeze(-1) - levels.unsqueeze(1)).abs()
    qidx = dist.argmin(dim=-1).to(torch.uint8)

    qval = torch.gather(
        levels,
        dim=1,
        index=qidx.long(),
    )

    return qidx, qval


@torch.no_grad()
def grid_search_levels_for_group(
    Wg: torch.Tensor,
    Hdiag_g: torch.Tensor,
    mode: str,
    rho_values: List[float],
) -> torch.Tensor:
    """
    Choose grid levels rowwise by minimizing:

        sum_j H_jj (w_j - q_j)^2
    """
    device = Wg.device

    Hdiag_g = Hdiag_g.to(device=device, dtype=torch.float32).abs().clamp(min=1e-8)

    rows = Wg.size(0)
    nlevels = 3 if mode == "ternary" else 4

    best_err = torch.full((rows,), float("inf"), device=device, dtype=torch.float32)
    best_levels = torch.zeros((rows, nlevels), device=device, dtype=torch.float32)

    Wg32 = Wg.float()

    for rn in rho_values:
        for rp in rho_values:
            levels = make_candidate_levels(
                Wg=Wg32,
                rho_neg=rn,
                rho_pos=rp,
                mode=mode,
            )

            _, qval = quantize_matrix_to_levels(Wg32, levels)

            err = ((Wg32 - qval).pow(2) * Hdiag_g.view(1, -1)).sum(dim=1)

            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_levels = torch.where(better.view(-1, 1), levels, best_levels)

    return best_levels


@torch.no_grad()
def quantize_column_to_levels(
    w: torch.Tensor,
    levels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dist = (w.view(-1, 1) - levels).abs()
    qidx = dist.argmin(dim=1).to(torch.uint8)
    qval = levels.gather(1, qidx.long().view(-1, 1)).squeeze(1)
    return qidx, qval


# ============================================================
# Robust Hessian inverse
# ============================================================

@torch.no_grad()
def robust_cholesky_inverse_upper(
    H: torch.Tensor,
    percdamp: float,
    max_tries: int = 12,
) -> Tuple[torch.Tensor, float]:
    """
    Returns:
        U = cholesky(inv(H + damp I), upper=True)

    GPTQ update uses:
        U[j,j] and U[j,j+1:]
    """
    H64 = H.to(torch.float64)
    H64 = 0.5 * (H64 + H64.T)
    H64 = torch.nan_to_num(H64, nan=0.0, posinf=0.0, neginf=0.0)

    n = H64.size(0)
    ar = torch.arange(n, device=H64.device)

    diag = torch.diag(H64)
    diag_abs = diag.abs()

    mean_base = max(float(diag_abs.mean().item()), 1e-8)
    max_base = max(float(diag_abs.max().item()), 1e-8)

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

    last_exc: Optional[Exception] = None

    for base in (mean_base, max_base):
        for mult in multipliers:
            damp = float(percdamp) * mult * base

            H_try = H64.clone()
            H_try[ar, ar] += damp

            try:
                chol = torch.linalg.cholesky(H_try)
                Hinv = torch.cholesky_inverse(chol)
                Hinv = 0.5 * (Hinv + Hinv.T)
                U = torch.linalg.cholesky(Hinv, upper=True)
                return U.to(torch.float32), damp

            except RuntimeError as exc:
                last_exc = exc

    raise RuntimeError(
        "Cholesky failed after adaptive damping retries. "
        f"Last error: {last_exc}"
    )


# ============================================================
# Layer compression
# ============================================================

@dataclass
class DenseQMoEResult:
    qidx: torch.Tensor
    levels: torch.Tensor
    dense_dequant: torch.Tensor
    shape: Tuple[int, int]
    groupsize: int
    quant_mode: str
    natural_zero_fraction: float
    hessian_samples: int


@torch.no_grad()
def dense_qmoe_gptq_linear(
    layer: nn.Linear,
    H: torch.Tensor,
    hessian_samples: int,
    quant_mode: str,
    groupsize: int,
    blocksize: int,
    percdamp: float,
    rho_values: List[float],
    compress_device: torch.device,
    store_dense_dequant: bool,
) -> DenseQMoEResult:
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(layer)}")

    if quant_mode not in ("ternary", "2bit"):
        raise ValueError("--quant_mode must be 'ternary' or '2bit'.")

    original_device = layer.weight.device
    original_dtype = layer.weight.dtype

    W_orig = layer.weight.detach().to(device=compress_device, dtype=torch.float32).clone()
    rows, cols = W_orig.shape

    if H.shape != (cols, cols):
        raise ValueError(f"H shape mismatch. Expected {(cols, cols)}, got {tuple(H.shape)}")

    H = H.to(compress_device)
    H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

    nlevels = 3 if quant_mode == "ternary" else 4
    ngroups = get_num_groups(cols, groupsize)

    print(f"      quant mode   : {quant_mode}")
    print(f"      shape        : {tuple(W_orig.shape)}")
    print(f"      groupsize    : {groupsize}")
    print(f"      groups       : {ngroups}")
    print(f"      nlevels      : {nlevels}")
    print(f"      blocksize    : {blocksize}")
    print(f"      rho values   : {rho_values}")
    print(f"      device       : {compress_device}")

    inv_t0 = now()
    U, used_damp = robust_cholesky_inverse_upper(H=H, percdamp=percdamp)
    U = U.to(compress_device)

    print(f"      used damping : {used_damp:.6e}")
    print(f"      inverse time : {format_seconds(now() - inv_t0)}")

    Hdiag = torch.diag(H).float().abs().clamp(min=1e-8).to(compress_device)

    W = W_orig.clone()
    Q = torch.zeros_like(W)

    qidx = torch.zeros(
        (rows, cols),
        dtype=torch.uint8,
        device=compress_device,
    )

    levels_all = torch.zeros(
        (rows, ngroups, nlevels),
        dtype=torch.float32,
        device=compress_device,
    )

    current_group = -1
    current_levels: Optional[torch.Tensor] = None

    layer_t0 = now()

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        U1 = U[i1:i2, i1:i2].contiguous()

        for local_j in range(count):
            j = i1 + local_j
            g = get_group_index(j, cols, groupsize)

            # CRUCIAL:
            # Grid is recomputed at group boundary using current GPTQ-updated W.
            if g != current_group:
                g0, g1 = get_group_bounds(g, cols, groupsize)

                current_levels = grid_search_levels_for_group(
                    Wg=W[:, g0:g1],
                    Hdiag_g=Hdiag[g0:g1],
                    mode=quant_mode,
                    rho_values=rho_values,
                )

                levels_all[:, g, :] = current_levels
                current_group = g

            assert current_levels is not None

            d = U1[local_j, local_j]

            if d.abs().item() < 1e-12:
                d = torch.tensor(1e-12, device=compress_device, dtype=U1.dtype)

            w = W1[:, local_j]

            qj, qv = quantize_column_to_levels(
                w=w,
                levels=current_levels,
            )

            qidx[:, j] = qj
            Q1[:, local_j] = qv
            Q[:, j] = qv

            # CRUCIAL GPTQ update:
            # err = (w - q) / U[j,j]
            # future weights -= err @ U[j,future]
            err = (w - qv) / d
            Err1[:, local_j] = err

            if local_j + 1 < count:
                W1[:, local_j + 1:count] -= (
                    err.unsqueeze(1)
                    @ U1[local_j, local_j + 1:count].unsqueeze(0)
                )

        W[:, i1:i2] = Q1

        if i2 < cols:
            W[:, i2:cols] -= Err1 @ U[i1:i2, i2:cols]

        print(
            f"\r      GPTQ columns: {i2}/{cols} "
            f"({100.0 * i2 / cols:.1f}%) "
            f"elapsed={format_seconds(now() - layer_t0)}",
            end="",
            flush=True,
        )

    print()

    natural_zero_fraction = float((qidx == 0).sum().item()) / float(qidx.numel())

    # CRUCIAL:
    # Sequential compression effect.
    # Future block/layer Hessians see already-compressed previous weights.
    layer.weight.data.copy_(
        Q.to(device=original_device, dtype=original_dtype)
    )

    dense_dequant = Q.detach().cpu() if store_dense_dequant else torch.empty(0)

    return DenseQMoEResult(
        qidx=qidx.detach().cpu(),
        levels=levels_all.detach().cpu(),
        dense_dequant=dense_dequant,
        shape=(rows, cols),
        groupsize=groupsize,
        quant_mode=quant_mode,
        natural_zero_fraction=natural_zero_fraction,
        hessian_samples=int(hessian_samples),
    )


# ============================================================
# Runtime wrapper for later eval support
# ============================================================

class DenseQMoEQuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        qidx: torch.Tensor,
        levels: torch.Tensor,
        groupsize: int,
        quant_mode: str,
        bias: Optional[torch.Tensor] = None,
        cache_dequantized: bool = False,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.groupsize = int(groupsize)
        self.quant_mode = str(quant_mode)
        self.cache_dequantized = bool(cache_dequantized)

        self.register_buffer("qidx", qidx.contiguous().to(torch.uint8))
        self.register_buffer("levels", levels.contiguous().to(torch.float32))

        col_group_idx = torch.tensor(
            [
                get_group_index(c, self.in_features, self.groupsize)
                for c in range(self.in_features)
            ],
            dtype=torch.long,
        )

        self.register_buffer("col_group_idx", col_group_idx)

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
        else:
            self.bias = None

        self._cached_weight: Optional[torch.Tensor] = None

    @torch.no_grad()
    def dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        levels_expanded = self.levels[:, self.col_group_idx, :]
        q = self.qidx.long().unsqueeze(-1)

        w = torch.gather(
            levels_expanded,
            dim=2,
            index=q,
        ).squeeze(-1)

        return w.to(dtype=dtype)

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


# ============================================================
# Run compressed block to get next hidden cache
# ============================================================

@torch.no_grad()
def run_block_to_hidden_cache(
    model: nn.Module,
    block: nn.Module,
    hidden_cache: torch.Tensor,
    batch_size: int,
    device: torch.device,
    model_compute_dtype: torch.dtype,
    hidden_cache_dtype: torch.dtype,
) -> torch.Tensor:
    n, seqlen, _ = hidden_cache.shape
    chunks: List[torch.Tensor] = []

    t0 = now()

    for i in range(0, n, batch_size):
        h = hidden_cache[i:i + batch_size].to(
            device=device,
            dtype=model_compute_dtype,
        )

        position_ids = make_position_ids(h.size(0), seqlen, device)

        out = forward_decoder_block(
            model=model,
            block=block,
            hidden=h,
            position_ids=position_ids,
            model_compute_dtype=model_compute_dtype,
            attention_mask=None,
        )

        chunks.append(out.detach().to("cpu", dtype=hidden_cache_dtype))

        done = min(i + batch_size, n)

        print(
            f"\r  compressed block forward: {done}/{n} ({100.0 * done / n:.1f}%) "
            f"elapsed={format_seconds(now() - t0)}",
            end="",
            flush=True,
        )

    print()

    return torch.cat(chunks, dim=0)


# ============================================================
# Blockwise compression
# ============================================================

@torch.no_grad()
def compress_model_blockwise(
    model: nn.Module,
    calib_tokens: torch.Tensor,
    batch_size: int,
    device: torch.device,
    model_compute_dtype: torch.dtype,
    hidden_cache_dtype: torch.dtype,
    hessian_dtype: torch.dtype,
    quant_mode: str,
    groupsize: int,
    blocksize: int,
    percdamp: float,
    rho_values: List[float],
    suffixes: Tuple[str, ...],
    include: str,
    exclude: str,
    skip_attn_out: bool,
    skip_mlp_out: bool,
    large_layer_cpu_threshold: int,
    store_dense_dequant: bool,
) -> Dict[str, Any]:
    decoder_name, decoder_layers = find_decoder_layers(model)

    print(f"\nDecoder layers found: {decoder_name}")
    print(f"Number of decoder blocks: {len(decoder_layers)}")

    print("\nComputing initial embedding hidden cache...")

    hidden_cache = compute_initial_hidden_cache(
        model=model,
        tokens=calib_tokens,
        batch_size=batch_size,
        device=device,
        model_compute_dtype=model_compute_dtype,
        hidden_cache_dtype=hidden_cache_dtype,
    )

    print(f"Initial hidden cache shape: {tuple(hidden_cache.shape)}, dtype={hidden_cache.dtype}")

    qmoe_layers: Dict[str, Any] = {}
    total_t0 = now()

    for bi, block in enumerate(decoder_layers):
        block_prefix = f"{decoder_name}.{bi}"

        named_layers = selected_linear_layers_in_block(
            block=block,
            block_prefix=block_prefix,
            include=include,
            exclude=exclude,
            suffixes=suffixes,
            skip_attn_out=skip_attn_out,
            skip_mlp_out=skip_mlp_out,
        )

        print("\n" + "=" * 100)
        print(f"BLOCK {bi}/{len(decoder_layers) - 1}")
        print(f"Selected linear layers in block: {len(named_layers)}")
        print(f"Hidden cache entering block: {tuple(hidden_cache.shape)}, dtype={hidden_cache.dtype}")
        print(f"CUDA memory: {cuda_memory_string()}")

        for name, layer in named_layers:
            print(f"  - {name}: {tuple(layer.weight.shape)} dtype={layer.weight.dtype}")

        if named_layers:
            print("  Collecting Hessians for all selected Linear layers in this block...")

            Hs, nsamples, hessian_devices = collect_block_hessians(
                model=model,
                block=block,
                named_layers=named_layers,
                hidden_cache=hidden_cache,
                batch_size=batch_size,
                device=device,
                model_compute_dtype=model_compute_dtype,
                hessian_dtype=hessian_dtype,
                large_layer_cpu_threshold=large_layer_cpu_threshold,
            )

            for li, (layer_name, layer) in enumerate(named_layers, start=1):
                print("\n" + "-" * 100)
                print(f"  [{li}/{len(named_layers)}] Compressing {layer_name}")
                print(f"      weight shape : {tuple(layer.weight.shape)}")
                print(f"      weight dtype  : {layer.weight.dtype}")
                print(f"      H shape      : {tuple(Hs[layer_name].shape)}")
                print(f"      H samples    : {nsamples[layer_name]}")
                print(f"      math device  : {hessian_devices[layer_name]}")

                result = dense_qmoe_gptq_linear(
                    layer=layer,
                    H=Hs[layer_name],
                    hessian_samples=nsamples[layer_name],
                    quant_mode=quant_mode,
                    groupsize=groupsize,
                    blocksize=blocksize,
                    percdamp=percdamp,
                    rho_values=rho_values,
                    compress_device=hessian_devices[layer_name],
                    store_dense_dequant=store_dense_dequant,
                )

                layer_state: Dict[str, Any] = {
                    "format": "dense_qmoe_gptq_layer",
                    "quant_mode": result.quant_mode,
                    "shape": list(result.shape),
                    "groupsize": int(result.groupsize),
                    "qidx": result.qidx,
                    "levels": result.levels,
                    "natural_zero_fraction": float(result.natural_zero_fraction),
                    "hessian_samples": int(result.hessian_samples),
                }

                if store_dense_dequant:
                    layer_state["dense_dequant"] = result.dense_dequant

                qmoe_layers[layer_name] = layer_state

                rows, cols = result.shape
                nparams = rows * cols
                qidx_bytes = nparams
                levels_bytes = int(result.levels.numel()) * 4
                dense_bf16_bytes = nparams * 2
                stored_bytes = qidx_bytes + levels_bytes

                print(f"      natural zero fraction : {100.0 * result.natural_zero_fraction:.2f}%")
                print(f"      qidx shape            : {tuple(result.qidx.shape)}")
                print(f"      levels shape          : {tuple(result.levels.shape)}")
                print(f"      raw qidx+levels bytes : {stored_bytes:,}")
                print(f"      dense bf16 bytes      : {dense_bf16_bytes:,}")
                print(f"      raw compression est.  : {dense_bf16_bytes / stored_bytes:.2f}x")

                del result
                cleanup_cuda()

            del Hs
            cleanup_cuda()

        print("\n  Running compressed block to create next hidden cache...")

        hidden_cache = run_block_to_hidden_cache(
            model=model,
            block=block,
            hidden_cache=hidden_cache,
            batch_size=batch_size,
            device=device,
            model_compute_dtype=model_compute_dtype,
            hidden_cache_dtype=hidden_cache_dtype,
        )

        print(f"Finished block {bi}. Total elapsed: {format_seconds(now() - total_t0)}")

    return qmoe_layers


# ============================================================
# Save helpers
# ============================================================

def build_partial_state_dict(
    model: nn.Module,
    compressed_layers: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    compressed_weight_keys = {f"{name}.weight" for name in compressed_layers.keys()}
    out: Dict[str, torch.Tensor] = {}

    for k, v in model.state_dict().items():
        if k in compressed_weight_keys:
            continue
        out[k] = v.detach().cpu()

    return out


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
    parser.add_argument("--hessian_dtype", type=str, default="float32", choices=["float32", "float64"])

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=1024)

    parser.add_argument("--quant_mode", type=str, default="ternary", choices=["ternary", "2bit"])
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.1)

    parser.add_argument(
        "--rho_grid",
        type=str,
        default="",
        help=(
            "Comma-separated shrink grid. Empty means 0.50,0.55,...,1.00. "
            "Fast test: 0.5,0.6,0.7,0.8,0.9,1.0"
        ),
    )

    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")
    parser.add_argument("--suffixes", type=str, default="")
    parser.add_argument("--skip_attn_out", action="store_true")
    parser.add_argument("--skip_mlp_out", action="store_true")

    parser.add_argument(
        "--large_layer_cpu_threshold",
        type=int,
        default=8192,
        help=(
            "If layer.in_features > threshold, Hessian/inverse/compression runs on CPU. "
            "Mistral down_proj has in_features=14336, so default sends it to CPU."
        ),
    )

    parser.add_argument("--store_dense_dequant", action="store_true")
    parser.add_argument("--keep_dequantized_state_dict", action="store_true")

    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="eager")

    args = parser.parse_args()

    script_t0 = now()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable.")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    device = torch.device(args.device)
    model_dtype = parse_dtype(args.model_dtype)
    hidden_cache_dtype = parse_dtype(args.hidden_cache_dtype)
    hessian_dtype = parse_dtype(args.hessian_dtype)
    rho_values = parse_rho_grid(args.rho_grid)
    suffixes = parse_suffixes(args.suffixes)

    print("=" * 100)
    print("Dense-QMoE-style GPTQ compression")
    print("=" * 100)
    print(f"model_id                  : {args.model_id}")
    print(f"calib                     : {args.calib}")
    print(f"out                       : {args.out}")
    print(f"device                    : {device}")
    print(f"model_dtype               : {model_dtype}")
    print(f"hidden_cache_dtype        : {hidden_cache_dtype}")
    print(f"hessian_dtype             : {hessian_dtype}")
    print(f"quant_mode                : {args.quant_mode}")
    print(f"groupsize                 : {args.groupsize}")
    print(f"blocksize                 : {args.blocksize}")
    print(f"percdamp                  : {args.percdamp}")
    print(f"rho_values                : {rho_values}")
    print(f"large_layer_cpu_threshold : {args.large_layer_cpu_threshold}")
    print(f"suffixes                  : {suffixes}")

    print("\nLoading tokenizer/model...")
    load_t0 = now()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=bool(args.trust_remote_code),
    )

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
    model.to(device)

    model_compute_dtype = get_model_param_dtype(model)

    print(f"Loaded model in {format_seconds(now() - load_t0)}")
    print(f"Model compute dtype: {model_compute_dtype}")
    print(f"CUDA memory: {cuda_memory_string()}")

    calib_tokens = load_calibration_tokens(args.calib)
    print(f"Loaded calibration tokens: {tuple(calib_tokens.shape)}")

    if args.max_seq_len > 0 and calib_tokens.size(1) > args.max_seq_len:
        calib_tokens = calib_tokens[:, :args.max_seq_len]
        print(f"Trimmed calibration sequence length to {args.max_seq_len}")

    qmoe_layers = compress_model_blockwise(
        model=model,
        calib_tokens=calib_tokens,
        batch_size=args.batch_size,
        device=device,
        model_compute_dtype=model_compute_dtype,
        hidden_cache_dtype=hidden_cache_dtype,
        hessian_dtype=hessian_dtype,
        quant_mode=args.quant_mode,
        groupsize=args.groupsize,
        blocksize=args.blocksize,
        percdamp=args.percdamp,
        rho_values=rho_values,
        suffixes=suffixes,
        include=args.include,
        exclude=args.exclude,
        skip_attn_out=bool(args.skip_attn_out),
        skip_mlp_out=bool(args.skip_mlp_out),
        large_layer_cpu_threshold=int(args.large_layer_cpu_threshold),
        store_dense_dequant=bool(args.store_dense_dequant),
    )

    total_params = 0
    total_zeros = 0
    total_qidx_bytes = 0
    total_levels_bytes = 0
    total_dense_bf16_bytes = 0

    for st in qmoe_layers.values():
        rows, cols = st["shape"]
        n = rows * cols
        total_params += n
        total_zeros += int(round(float(st["natural_zero_fraction"]) * n))
        total_qidx_bytes += n
        total_levels_bytes += int(st["levels"].numel()) * 4
        total_dense_bf16_bytes += n * 2

    raw_stored_bytes = total_qidx_bytes + total_levels_bytes
    natural_zero_fraction = total_zeros / float(total_params) if total_params else 0.0

    meta = {
        "format": "hf_dense_qmoe_gptq",
        "method": "dense_qmoe_style_gptq_groupwise_grid_search_blockwise",
        "model_id": str(args.model_id),
        "calib": str(args.calib),
        "quant_mode": str(args.quant_mode),
        "groupsize": int(args.groupsize),
        "blocksize": int(args.blocksize),
        "percdamp": float(args.percdamp),
        "rho_values": list(rho_values),
        "model_dtype": str(args.model_dtype),
        "model_compute_dtype": str(model_compute_dtype),
        "hidden_cache_dtype": str(args.hidden_cache_dtype),
        "hessian_dtype": str(args.hessian_dtype),
        "large_layer_cpu_threshold": int(args.large_layer_cpu_threshold),
        "selected_layers": int(len(qmoe_layers)),
        "total_params_compressed": int(total_params),
        "total_natural_zero_fraction": float(natural_zero_fraction),
        "raw_qidx_bytes": int(total_qidx_bytes),
        "levels_bytes": int(total_levels_bytes),
        "raw_stored_bytes": int(raw_stored_bytes),
        "dense_bf16_bytes": int(total_dense_bf16_bytes),
        "raw_compression_vs_bf16": (
            float(total_dense_bf16_bytes / raw_stored_bytes) if raw_stored_bytes else None
        ),
        "keep_dequantized_state_dict": bool(args.keep_dequantized_state_dict),
        "store_dense_dequant": bool(args.store_dense_dequant),
        "note": (
            "This checkpoint stores raw qidx uint8 and per-row/per-group quantization levels. "
            "No entropy/dictionary coding is applied yet. Runtime reconstruction is dense and "
            "not kernel optimized."
        ),
    }

    if args.keep_dequantized_state_dict:
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        model_field_contents = "full_dense_dequantized_state_dict"
    else:
        model_state = build_partial_state_dict(model, qmoe_layers)
        model_field_contents = "non_compressed_parameters_only"

    meta["model_field_contents"] = model_field_contents

    ckpt = {
        "format": "hf_dense_qmoe_gptq",
        "model_id": args.model_id,
        "config": model.config.to_dict() if hasattr(model, "config") else None,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", args.model_id),
        "dense_qmoe_meta": meta,
        "qmoe_layers": qmoe_layers,
        "model": model_state,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("\nSaving checkpoint...")
    save_t0 = now()
    torch.save(ckpt, out)
    print(f"Saved checkpoint in {format_seconds(now() - save_t0)}: {out}")

    meta_path = out.with_suffix(out.suffix + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved meta JSON: {meta_path}")

    print("\nDone.")
    print(f"Compressed layers       : {len(qmoe_layers)}")
    print(f"Compressed params       : {total_params:,}")
    print(f"Natural zero fraction   : {100.0 * natural_zero_fraction:.2f}%")
    print(f"Raw qidx bytes          : {total_qidx_bytes:,}")
    print(f"Levels bytes            : {total_levels_bytes:,}")
    print(f"Raw stored bytes        : {raw_stored_bytes:,}")
    print(f"Dense BF16 bytes        : {total_dense_bf16_bytes:,}")

    if raw_stored_bytes:
        print(f"Raw compression vs BF16 : {total_dense_bf16_bytes / raw_stored_bytes:.2f}x")

    print(f"Total script time       : {format_seconds(now() - script_t0)}")
    print(f"CUDA memory             : {cuda_memory_string()}")


if __name__ == "__main__":
    main()