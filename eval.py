#!/usr/bin/env python3
"""
Evaluate Hugging Face causal LM perplexity on raw text.

Supports:

1. Original HF model loaded in:
    - NF4 4-bit BitsAndBytes
    - FP16/BF16/FP32

2. Older compressed checkpoint format:
    ckpt["layers"]

3. Newer joint SparseGPT+GPTQ HF checkpoint format:
    ckpt["joint_sparsegpt_gptq_layers"]

Evaluation convention:

    x = tokens[start:end]
    y = tokens[start+1:end+1]

    logits = model(x)
    loss = CE(logits[t], y[t])

Important:
    - x and y are externally shifted.
    - Do NOT shift logits again.
    - Overlap tokens are used as context but not counted twice.
    - perplexity = exp(mean negative log-likelihood)

For your checkpoint:

    compressed/mistral_gptq_int4_nosparsity_blockwise_bf16_1024.pt

use:

    --load_mode fp
    --dtype bfloat16 or float16
    --compressed compressed/mistral_gptq_int4_nosparsity_blockwise_bf16_1024.pt

Do NOT use --load_mode nf4 for your custom GPTQ checkpoint unless you intentionally
want to load the non-compressed parts from an NF4 model.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


IGNORE_INDEX = -1


# ============================================================
# Time / memory helpers
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


def get_cuda_memory_string() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return (
        f"allocated={allocated:.2f}GB "
        f"reserved={reserved:.2f}GB "
        f"max_allocated={max_allocated:.2f}GB"
    )


def print_cuda_memory(prefix: str) -> None:
    if torch.cuda.is_available():
        print(f"{prefix}: {get_cuda_memory_string()}")


def get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return fallback


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Packing / unpacking helpers
# ============================================================

def unpack_4bit_rows(packed: torch.Tensor, original_cols: int) -> torch.Tensor:
    """
    Unpack packed INT4 weights.

    Input:
        packed shape = [rows, ceil(cols / 2)]

    Output:
        uint8 qweight shape = [rows, original_cols]
    """
    if packed.dtype != torch.uint8:
        packed = packed.to(torch.uint8)

    rows, packed_cols = packed.shape

    out = torch.empty(
        (rows, packed_cols * 2),
        dtype=torch.uint8,
        device=packed.device,
    )

    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = (packed >> 4) & 0x0F

    return out[:, :original_cols]


def unpack_bool_mask_rows(packed: torch.Tensor, original_cols: int) -> torch.Tensor:
    """
    Unpack packed sparse mask.

    Input:
        packed shape = [rows, ceil(cols / 8)]

    Output:
        bool mask shape = [rows, original_cols]
    """
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


def maybe_unpack_qweight(
    qweight_stored: torch.Tensor,
    bits: int,
    packing: str,
    original_shape: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    rows, cols = original_shape

    qweight_stored = qweight_stored.to(device)

    if packing == "uint8":
        if qweight_stored.shape != (rows, cols):
            raise ValueError(
                f"Expected qweight shape {(rows, cols)}, got {tuple(qweight_stored.shape)}"
            )
        return qweight_stored.to(torch.uint8)

    if packing == "packed4":
        if bits != 4:
            raise ValueError("packing='packed4' only works with bits=4.")
        return unpack_4bit_rows(qweight_stored.to(torch.uint8), original_cols=cols)

    raise ValueError(f"Unsupported qweight packing: {packing}")


def maybe_unpack_mask(
    mask_stored: Optional[torch.Tensor],
    mask_packing: str,
    original_shape: Tuple[int, int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if mask_stored is None:
        return None

    rows, cols = original_shape
    mask_stored = mask_stored.to(device)

    if mask_packing == "bool":
        mask = mask_stored.bool()
        if mask.shape != (rows, cols):
            raise ValueError(f"Expected mask shape {(rows, cols)}, got {tuple(mask.shape)}")
        return mask

    if mask_packing == "packedbits":
        return unpack_bool_mask_rows(mask_stored.to(torch.uint8), original_cols=cols)

    raise ValueError(f"Unsupported mask packing: {mask_packing}")


def get_group_index(col_idx: int, cols: int, groupsize: int) -> int:
    if groupsize == -1 or groupsize >= cols:
        return 0
    return col_idx // groupsize


# ============================================================
# Runtime layer for compressed sparse quantized checkpoint
# ============================================================

class SparseQuantLinear(nn.Module):
    """
    Runtime wrapper for compressed sparse quantized linear layers.

    Stored form:
        qweight     : packed4 or uint8
        scales      : [out_features, num_groups]
        zero_points : [out_features, num_groups]
        mask        : packedbits, bool, or None

    Runtime reconstruction:
        q = unpack(qweight)
        W = ((q - zero_point) * scale)
        if mask exists:
            W = W * mask

    This is mathematically correct but not kernel-optimized.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        mask: Optional[torch.Tensor],
        groupsize: int,
        packing: str,
        mask_packing: str,
        bias: Optional[torch.Tensor] = None,
        cache_dequantized: bool = False,
        dequant_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bits = int(bits)
        self.groupsize = int(groupsize)
        self.packing = str(packing)
        self.mask_packing = str(mask_packing)
        self.cache_dequantized = bool(cache_dequantized)
        self.dequant_dtype = dequant_dtype

        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scales", scales.contiguous().to(torch.float32))
        self.register_buffer("zero_points", zero_points.contiguous().to(torch.float32))

        if mask is None:
            self.mask = None
        else:
            self.register_buffer("mask", mask.contiguous())

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
    def dequantize_weight(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        device = self.qweight.device
        out_dtype = dtype if dtype is not None else self.dequant_dtype

        q = maybe_unpack_qweight(
            qweight_stored=self.qweight,
            bits=self.bits,
            packing=self.packing,
            original_shape=(self.out_features, self.in_features),
            device=device,
        ).to(torch.float32)

        scale_expanded = self.scales[:, self.col_group_idx]
        zero_expanded = self.zero_points[:, self.col_group_idx]

        w = (q - zero_expanded) * scale_expanded

        if self.mask is not None:
            mask = maybe_unpack_mask(
                mask_stored=self.mask,
                mask_packing=self.mask_packing,
                original_shape=(self.out_features, self.in_features),
                device=device,
            )
            assert mask is not None
            w = w * mask.to(dtype=w.dtype)

        return w.to(dtype=out_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache_dequantized:
            if self._cached_weight is None or self._cached_weight.device != x.device:
                self._cached_weight = self.dequantize_weight(dtype=x.dtype).to(x.device)

            w = self._cached_weight.to(dtype=x.dtype)
        else:
            w = self.dequantize_weight(dtype=x.dtype).to(device=x.device)

        bias = self.bias

        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)

        return F.linear(x, w, bias)


SparseInt4Linear = SparseQuantLinear


# ============================================================
# Module helpers
# ============================================================

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


# ============================================================
# Model loading
# ============================================================

def parse_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.lower().strip()

    if dtype_str in ("float16", "fp16", "half"):
        return torch.float16
    if dtype_str in ("bfloat16", "bf16"):
        return torch.bfloat16
    if dtype_str in ("float32", "fp32"):
        return torch.float32

    raise ValueError(f"Unsupported dtype: {dtype_str}")


def load_hf_model(
    model_id: str,
    load_mode: str,
    dtype_str: str,
    device: str,
    trust_remote_code: bool,
    low_cpu_mem_usage: bool,
    attn_implementation: Optional[str],
):
    """
    load_mode:
        nf4 : BitsAndBytes NF4 4-bit
        fp  : normal HF fp16/bf16/fp32 model
    """
    t0 = now()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = parse_torch_dtype(dtype_str)

    model_kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "trust_remote_code": trust_remote_code,
    }

    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if load_mode == "nf4":
        print("Loading original model in BitsAndBytes NF4 4-bit...")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            **model_kwargs,
        )

    elif load_mode == "fp":
        print(f"Loading original model in {dtype_str}...")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            **model_kwargs,
        )

        model.to(torch.device(device))

    else:
        raise ValueError("--load_mode must be 'nf4' or 'fp'.")

    if hasattr(model, "config"):
        model.config.use_cache = False

    model.eval()

    elapsed = now() - t0
    print(f"Model loaded in {format_seconds(elapsed)}")
    print(f"Model first parameter device: {get_model_device(model)}")
    print_cuda_memory("CUDA memory after model load")

    return tokenizer, model


# ============================================================
# Compressed checkpoint application
# ============================================================

def detect_compressed_layers_key(ckpt: Dict[str, Any]) -> str:
    if "layers" in ckpt:
        return "layers"

    if "joint_sparsegpt_gptq_layers" in ckpt:
        return "joint_sparsegpt_gptq_layers"

    raise ValueError(
        "Compressed checkpoint must contain either key:\n"
        "  - 'layers'\n"
        "  - 'joint_sparsegpt_gptq_layers'"
    )


def get_checkpoint_global_meta(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    if "joint_sparsegpt_gptq_meta" in ckpt:
        return ckpt["joint_sparsegpt_gptq_meta"]

    if "compression_meta" in ckpt:
        return ckpt["compression_meta"]

    return ckpt


def normalize_layer_state(st: Dict[str, Any]) -> Dict[str, Any]:
    shape = tuple(st["shape"])
    out_features, in_features = shape

    bits = int(st.get("bits", 4))
    groupsize = int(st["groupsize"])

    packing = str(st.get("packing", "packed4"))
    mask_packing = str(st.get("mask_packing", "packedbits"))

    qweight = st["qweight"]
    scales = st["scales"]
    zero_points = st["zero_points"]

    mask = st.get("mask", None)

    # Some future no-sparsity checkpoints may store has_mask=False.
    if st.get("has_mask", True) is False:
        mask = None

    bias = st.get("bias", None)

    return {
        "shape": shape,
        "out_features": out_features,
        "in_features": in_features,
        "bits": bits,
        "groupsize": groupsize,
        "packing": packing,
        "mask_packing": mask_packing,
        "qweight": qweight,
        "scales": scales,
        "zero_points": zero_points,
        "mask": mask,
        "bias": bias,
    }


def optionally_load_noncompressed_state_dict(
    model: nn.Module,
    ckpt: Dict[str, Any],
    load_partial_state_dict: bool,
) -> None:
    """
    Newer HF joint checkpoint may store non-compressed parameters in ckpt["model"].

    Your checkpoint says:
        ckpt['model'] with ONLY NON-COMPRESSED parameters
        compressed linear weights are omitted from ckpt['model']

    Usually loading this is okay but not strictly needed when the base HF model_id is the same.
    """
    if not load_partial_state_dict:
        return

    if "model" not in ckpt or not ckpt["model"]:
        print("[info] No ckpt['model'] state_dict to load.")
        return

    print("Loading non-compressed parameters from ckpt['model'] with strict=False...")
    t0 = now()

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)

    print(f"Partial state_dict loaded in {format_seconds(now() - t0)}")

    if missing:
        print(f"[warn] missing keys: {len(missing)}")
        print(f"[warn] first missing keys: {missing[:10]}")

    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
        print(f"[warn] first unexpected keys: {unexpected[:10]}")


def apply_sparse_quant_checkpoint(
    model: nn.Module,
    compressed_path: str,
    cache_dequantized: bool = False,
    show_progress: bool = True,
    load_partial_state_dict: bool = False,
    dequant_dtype: torch.dtype = torch.float16,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Replace selected Linear layers with SparseQuantLinear.
    """
    print(f"Loading compressed checkpoint: {compressed_path}")
    t0 = now()

    ckpt = torch.load(compressed_path, map_location="cpu")

    print(f"Compressed checkpoint loaded in {format_seconds(now() - t0)}")

    optionally_load_noncompressed_state_dict(
        model=model,
        ckpt=ckpt,
        load_partial_state_dict=load_partial_state_dict,
    )

    layers_key = detect_compressed_layers_key(ckpt)
    layers = ckpt[layers_key]

    if not isinstance(layers, dict) or len(layers) == 0:
        raise ValueError(f"Compressed checkpoint has empty or invalid ckpt['{layers_key}'].")

    fallback_device = get_model_device(model)
    items = list(layers.items())

    print(f"Applying compressed layers from ckpt['{layers_key}']")
    print(f"Compressed layers: {len(items):,}")
    print(f"Cache dequantized: {cache_dequantized}")
    print(f"Dequant dtype    : {dequant_dtype}")

    iterator = items

    if show_progress and tqdm is not None:
        iterator = tqdm(
            items,
            desc="Applying compressed layers",
            unit="layer",
            dynamic_ncols=True,
        )

    apply_t0 = now()

    for idx, (layer_name, st_raw) in enumerate(iterator, start=1):
        st = normalize_layer_state(st_raw)

        old_layer = get_module_by_name(model, layer_name)
        target_device = get_module_device(old_layer, fallback=fallback_device)

        bias = st["bias"]

        if bias is None and hasattr(old_layer, "bias") and old_layer.bias is not None:
            bias = old_layer.bias.detach().cpu()

        qlayer = SparseQuantLinear(
            in_features=st["in_features"],
            out_features=st["out_features"],
            bits=st["bits"],
            qweight=st["qweight"].to(target_device),
            scales=st["scales"].to(target_device),
            zero_points=st["zero_points"].to(target_device),
            mask=st["mask"].to(target_device) if torch.is_tensor(st["mask"]) else None,
            groupsize=st["groupsize"],
            packing=st["packing"],
            mask_packing=st["mask_packing"],
            bias=bias.to(target_device) if torch.is_tensor(bias) else None,
            cache_dequantized=cache_dequantized,
            dequant_dtype=dequant_dtype,
        )

        set_module_by_name(model, layer_name, qlayer)

        del old_layer

        if idx % 16 == 0:
            cleanup_cuda()

        if tqdm is None and show_progress:
            elapsed = now() - apply_t0
            rate = idx / elapsed if elapsed > 0 else 0.0
            remaining = len(items) - idx
            eta = remaining / rate if rate > 0 else float("inf")

            print(
                f"\rApplying compressed layers: {idx}/{len(items)} "
                f"({100.0 * idx / len(items):5.1f}%) "
                f"rate={rate:.2f} layer/s "
                f"elapsed={format_seconds(elapsed)} "
                f"eta={format_seconds(eta)}",
                end="",
                flush=True,
            )

    if tqdm is None and show_progress:
        print()

    cleanup_cuda()
    model.eval()

    print(f"Compressed layers applied in {format_seconds(now() - apply_t0)}")
    print_cuda_memory("CUDA memory after applying compressed checkpoint")

    return model, ckpt


# ============================================================
# Tokenization / data
# ============================================================

def encode_text_file(
    tokenizer,
    input_file: str,
    add_bos_token: bool = False,
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    t0 = now()

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    read_time = now() - t0

    tok_t0 = now()

    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if add_bos_token:
        if tokenizer.bos_token_id is not None:
            ids = [tokenizer.bos_token_id] + ids
        else:
            print("[warn] --add_bos_token requested, but tokenizer has no bos_token_id.")

    tok_time = now() - tok_t0

    if len(ids) < 2:
        raise ValueError("Input text is too short after tokenization; need at least 2 tokens.")

    stats = {
        "read_seconds": read_time,
        "tokenize_seconds": tok_time,
        "raw_chars": len(text),
        "tokens": len(ids),
        "chars_per_second_read": len(text) / read_time if read_time > 0 else float("inf"),
        "tokens_per_second_tokenize": len(ids) / tok_time if tok_time > 0 else float("inf"),
    }

    return np.array(ids, dtype=np.int64), text, stats


def iterate_eval_windows(
    token_ids: np.ndarray,
    block_size: int,
    stride: Optional[int],
):
    if stride is None:
        stride = block_size

    if stride <= 0:
        raise ValueError("stride must be positive.")

    if stride > block_size:
        raise ValueError("stride must be <= block_size, otherwise some tokens are skipped.")

    n = len(token_ids)

    if n < 2:
        return

    prev_end = None

    for start in range(0, n - 1, stride):
        end = min(start + block_size, n - 1)

        x = token_ids[start:end]
        y = token_ids[start + 1:end + 1]

        if len(x) == 0:
            continue

        if prev_end is None:
            score_from = 0
        else:
            score_from = max(0, prev_end - start)

        yield x, y, score_from, start

        prev_end = end

        if end >= n - 1:
            break


def count_eval_windows(
    token_ids: np.ndarray,
    block_size: int,
    stride: Optional[int],
) -> int:
    return sum(1 for _ in iterate_eval_windows(token_ids, block_size, stride))


# ============================================================
# Evaluation math
# ============================================================

def make_autocast_context(device: str, dtype_str: str):
    if "cuda" not in device:
        return nullcontext()

    if dtype_str == "float32":
        return nullcontext()

    ptdtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_str]

    return torch.amp.autocast(device_type="cuda", dtype=ptdtype)


def safe_perplexity(mean_loss: float) -> float:
    return math.exp(mean_loss) if mean_loss < 20 else float("inf")


@torch.no_grad()
def evaluate_metrics(
    model,
    token_ids: np.ndarray,
    block_size: int,
    batch_size: int,
    device: str,
    dtype_str: str,
    stride: Optional[int],
    pad_token_id: int,
    show_progress: bool = True,
    progress_update_every: int = 1,
) -> Dict[str, Any]:
    ctx = make_autocast_context(device, dtype_str)

    if stride is None:
        stride = block_size

    if progress_update_every <= 0:
        progress_update_every = 1

    total_expected_windows = count_eval_windows(token_ids, block_size, stride)

    total_nll = 0.0
    total_tokens = 0
    total_correct = 0
    total_batches = 0
    total_windows = 0
    processed_windows = 0

    batch_x: List[np.ndarray] = []
    batch_y: List[np.ndarray] = []
    batch_score_from: List[int] = []

    eval_t0 = now()

    pbar = None

    if show_progress and tqdm is not None:
        pbar = tqdm(
            total=total_expected_windows,
            desc="Evaluating",
            unit="win",
            dynamic_ncols=True,
            smoothing=0.05,
        )

    def current_stats() -> Dict[str, Any]:
        elapsed = now() - eval_t0

        mean_loss = total_nll / total_tokens if total_tokens > 0 else float("nan")
        ppl = safe_perplexity(mean_loss) if total_tokens > 0 else float("nan")
        acc = total_correct / total_tokens if total_tokens > 0 else float("nan")

        tok_s = total_tokens / elapsed if elapsed > 0 else 0.0
        win_s = processed_windows / elapsed if elapsed > 0 else 0.0
        batch_s = total_batches / elapsed if elapsed > 0 else 0.0

        remaining_windows = total_expected_windows - processed_windows
        eta = remaining_windows / win_s if win_s > 0 else float("inf")

        pct = 100.0 * processed_windows / max(total_expected_windows, 1)

        return {
            "elapsed": elapsed,
            "eta": eta,
            "mean_loss": mean_loss,
            "ppl": ppl,
            "acc": acc,
            "tok_s": tok_s,
            "win_s": win_s,
            "batch_s": batch_s,
            "pct": pct,
        }

    def update_tqdm(n_windows: int) -> None:
        if pbar is None:
            return

        pbar.update(n_windows)

        if total_tokens > 0 and total_batches % progress_update_every == 0:
            st = current_stats()

            postfix = {
                "done": f"{st['pct']:.1f}%",
                "tok": f"{total_tokens:,}",
                "tok/s": f"{st['tok_s']:.0f}",
                "loss": f"{st['mean_loss']:.4f}",
                "ppl": f"{st['ppl']:.3f}",
                "acc": f"{100.0 * st['acc']:.2f}%",
                "eta": format_seconds(st["eta"]),
            }

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                postfix["vram"] = f"{allocated:.1f}G"

            pbar.set_postfix(postfix, refresh=False)

    def print_plain_progress(force: bool = False) -> None:
        if pbar is not None:
            return
        if not show_progress:
            return
        if not force and total_batches % progress_update_every != 0:
            return

        st = current_stats()

        mem = ""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            mem = f" vram={allocated:.1f}/{reserved:.1f}G"

        print(
            f"\rEvaluating: {processed_windows}/{total_expected_windows} "
            f"({st['pct']:5.1f}%) "
            f"tok={total_tokens:,} "
            f"tok/s={st['tok_s']:.0f} "
            f"win/s={st['win_s']:.2f} "
            f"loss={st['mean_loss']:.4f} "
            f"ppl={st['ppl']:.3f} "
            f"acc={100.0 * st['acc']:.2f}% "
            f"elapsed={format_seconds(st['elapsed'])} "
            f"eta={format_seconds(st['eta'])}"
            f"{mem}",
            end="",
            flush=True,
        )

    def process_batch(
        batch_x_local: List[np.ndarray],
        batch_y_local: List[np.ndarray],
        batch_score_from_local: List[int],
    ) -> None:
        nonlocal total_nll, total_tokens, total_correct, total_batches

        if not batch_x_local:
            return

        max_len = max(len(arr) for arr in batch_x_local)

        xb = torch.full(
            (len(batch_x_local), max_len),
            pad_token_id,
            dtype=torch.long,
        )

        yb = torch.full(
            (len(batch_y_local), max_len),
            IGNORE_INDEX,
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            (len(batch_x_local), max_len),
            dtype=torch.long,
        )

        for i, (x_arr, y_arr, score_from) in enumerate(
            zip(batch_x_local, batch_y_local, batch_score_from_local)
        ):
            L = len(x_arr)

            xb[i, :L] = torch.tensor(x_arr, dtype=torch.long)
            yb[i, :L] = torch.tensor(y_arr, dtype=torch.long)
            attention_mask[i, :L] = 1

            if score_from > 0:
                yb[i, :score_from] = IGNORE_INDEX

        target_device = get_model_device(model)

        xb = xb.to(target_device, non_blocking=True)
        yb = yb.to(target_device, non_blocking=True)
        attention_mask = attention_mask.to(target_device, non_blocking=True)

        with ctx:
            outputs = model(
                input_ids=xb,
                attention_mask=attention_mask,
                use_cache=False,
            )

            logits = outputs.logits

            # CRUCIAL:
            # x and y are already externally shifted.
            # Do NOT shift logits/labels again.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                yb.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="mean",
            )

        valid_mask = yb != IGNORE_INDEX
        scored_tokens = int(valid_mask.sum().item())

        if scored_tokens == 0:
            return

        total_nll += float(loss.item()) * scored_tokens
        total_tokens += scored_tokens
        total_batches += 1

        preds = logits.argmax(dim=-1)
        total_correct += int(((preds == yb) & valid_mask).sum().item())

        del logits, outputs, xb, yb, attention_mask, valid_mask, preds

    print("\nStarting evaluation...")
    print(f"Total expected windows: {total_expected_windows:,}")
    print(f"Block size            : {block_size}")
    print(f"Stride                : {stride}")
    print(f"Batch size            : {batch_size}")
    print_cuda_memory("CUDA memory at eval start")

    for x, y, score_from, _start in iterate_eval_windows(token_ids, block_size, stride):
        batch_x.append(x)
        batch_y.append(y)
        batch_score_from.append(score_from)
        total_windows += 1

        if len(batch_x) == batch_size:
            windows_in_batch = len(batch_x)

            process_batch(batch_x, batch_y, batch_score_from)

            processed_windows += windows_in_batch
            update_tqdm(windows_in_batch)
            print_plain_progress()

            batch_x.clear()
            batch_y.clear()
            batch_score_from.clear()

    if batch_x:
        windows_in_batch = len(batch_x)

        process_batch(batch_x, batch_y, batch_score_from)

        processed_windows += windows_in_batch
        update_tqdm(windows_in_batch)
        print_plain_progress(force=True)

    if pbar is not None:
        if total_tokens > 0:
            st = current_stats()

            postfix = {
                "done": f"{st['pct']:.1f}%",
                "tok": f"{total_tokens:,}",
                "tok/s": f"{st['tok_s']:.0f}",
                "loss": f"{st['mean_loss']:.4f}",
                "ppl": f"{st['ppl']:.3f}",
                "acc": f"{100.0 * st['acc']:.2f}%",
            }

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                postfix["vram"] = f"{allocated:.1f}G"

            pbar.set_postfix(postfix, refresh=True)

        pbar.close()
    elif show_progress:
        print()

    if total_tokens == 0:
        raise RuntimeError("No tokens evaluated. Use a longer file or reduce block_size.")

    mean_loss = total_nll / total_tokens
    elapsed = now() - eval_t0

    return {
        "mean_loss": mean_loss,
        "perplexity": safe_perplexity(mean_loss),
        "bits_per_token": mean_loss / math.log(2.0),
        "top1_accuracy": total_correct / total_tokens,
        "tokens_evaluated": total_tokens,
        "batches_evaluated": total_batches,
        "windows_evaluated": total_windows,
        "processed_windows": processed_windows,
        "expected_windows": total_expected_windows,
        "stride_used": stride,
        "evaluation_mode": "non-overlapping" if stride == block_size else "overlapping",
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0.0,
        "windows_per_second": processed_windows / elapsed if elapsed > 0 else 0.0,
        "batches_per_second": total_batches / elapsed if elapsed > 0 else 0.0,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate HF original or compressed sparse quantized checkpoint perplexity."
    )

    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--compressed", type=str, default=None)

    parser.add_argument(
        "--load_mode",
        type=str,
        default="fp",
        choices=["nf4", "fp"],
        help="nf4 = BitsAndBytes NF4 4-bit baseline, fp = normal HF fp16/bf16/fp32 model.",
    )

    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Autocast dtype and FP load dtype when --load_mode fp.",
    )

    parser.add_argument("--add_bos_token", action="store_true")

    parser.add_argument(
        "--cache_dequantized",
        action="store_true",
        help=(
            "Cache dequantized compressed weights. Faster, but uses much more VRAM. "
            "For Mistral-7B custom INT4 this may OOM on 24GB."
        ),
    )

    parser.add_argument(
        "--load_partial_state_dict",
        action="store_true",
        help=(
            "For newer compressed checkpoints, load ckpt['model'] non-compressed params with strict=False. "
            "Usually optional if --model_id is the same as compression."
        ),
    )

    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--progress_update_every", type=int, default=1)
    parser.add_argument("--save_json", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=[None, "eager", "sdpa", "flash_attention_2"],
        help="Optional HF attention implementation.",
    )

    args = parser.parse_args()

    script_t0 = now()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    if args.block_size <= 0:
        raise ValueError("--block_size must be positive.")

    if args.stride is not None and args.stride > args.block_size:
        raise ValueError("--stride must be <= --block_size.")

    show_progress = not args.no_progress

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("=" * 100)
    print("HF Causal LM Perplexity Evaluation")
    print("=" * 100)
    print(f"Model id       : {args.model_id}")
    print(f"Input file     : {args.input_file}")
    print(f"Compressed     : {args.compressed}")
    print(f"Load mode      : {args.load_mode}")
    print(f"Device         : {args.device}")
    print(f"Dtype/autocast : {args.dtype}")
    print(f"Block size     : {args.block_size}")
    print(f"Stride         : {args.stride if args.stride is not None else args.block_size}")
    print(f"Batch size     : {args.batch_size}")

    tokenizer, model = load_hf_model(
        model_id=args.model_id,
        load_mode=args.load_mode,
        dtype_str=args.dtype,
        device=args.device,
        trust_remote_code=bool(args.trust_remote_code),
        low_cpu_mem_usage=bool(args.low_cpu_mem_usage),
        attn_implementation=args.attn_implementation,
    )

    compressed_info = None
    model_variant = f"original_{args.load_mode}"

    if args.compressed is not None:
        print("\nApplying compressed checkpoint...")

        dequant_dtype = parse_torch_dtype(args.dtype)

        model, compressed_info = apply_sparse_quant_checkpoint(
            model=model,
            compressed_path=args.compressed,
            cache_dequantized=args.cache_dequantized,
            show_progress=show_progress,
            load_partial_state_dict=bool(args.load_partial_state_dict),
            dequant_dtype=dequant_dtype,
        )

        model_variant = "compressed_joint_sparsegpt_gptq"

    else:
        print("\nEvaluating original baseline.")

    print("\nTokenizing input file...")

    token_ids, raw_text, tok_stats = encode_text_file(
        tokenizer=tokenizer,
        input_file=args.input_file,
        add_bos_token=args.add_bos_token,
    )

    print(f"Input read time       : {format_seconds(tok_stats['read_seconds'])}")
    print(f"Tokenization time     : {format_seconds(tok_stats['tokenize_seconds'])}")
    print(f"Raw chars             : {tok_stats['raw_chars']:,}")
    print(f"Tokenized length      : {tok_stats['tokens']:,}")
    print(f"Tokenization speed    : {tok_stats['tokens_per_second_tokenize']:,.0f} tok/s")

    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    if pad_token_id is None:
        pad_token_id = 0

    effective_stride = args.stride if args.stride is not None else args.block_size
    expected_windows = count_eval_windows(token_ids, args.block_size, effective_stride)

    print("\n=== Evaluation Setup ===")
    print(f"Input file        : {args.input_file}")
    print(f"Raw chars         : {len(raw_text):,}")
    print(f"Tokenized length  : {len(token_ids):,}")
    print(f"Expected windows  : {expected_windows:,}")
    print(f"Model id          : {args.model_id}")
    print(f"Model variant     : {model_variant}")
    print(f"Compressed ckpt   : {args.compressed}")
    print(f"Device requested  : {args.device}")
    print(f"Model device      : {get_model_device(model)}")
    print(f"Dtype/autocast    : {args.dtype}")
    print(f"Block size        : {args.block_size}")
    print(f"Stride            : {effective_stride}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Pad token id      : {pad_token_id}")
    print(f"BOS token id      : {tokenizer.bos_token_id}")
    print(f"EOS token id      : {tokenizer.eos_token_id}")
    print(f"Add BOS token     : {args.add_bos_token}")
    print(f"Cache dequantized : {args.cache_dequantized}")
    print(f"Progress          : {show_progress}")
    print_cuda_memory("CUDA memory before eval")

    if compressed_info is not None:
        global_meta = get_checkpoint_global_meta(compressed_info)
        layers_key = detect_compressed_layers_key(compressed_info)

        print("\n=== Compression Info ===")
        print(f"Layer key         : {layers_key}")
        print(f"Format            : {compressed_info.get('format', 'unknown')}")
        print(f"Method            : {global_meta.get('method', compressed_info.get('method', 'unknown'))}")
        print(f"Bits              : {global_meta.get('bits', compressed_info.get('bits', 'unknown'))}")
        print(f"Groupsize         : {global_meta.get('groupsize', compressed_info.get('groupsize', 'unknown'))}")
        print(f"Target sparsity   : {global_meta.get('sparsity', compressed_info.get('sparsity', 'unknown'))}")

        actual_sparsity = global_meta.get(
            "actual_total_sparsity",
            compressed_info.get("actual_sparsity", 0.0),
        )

        print(f"Actual sparsity   : {100.0 * float(actual_sparsity):.2f}%")
        print(f"Compressed layers : {len(compressed_info.get(layers_key, {}))}")
        print(f"Compression dtype : {global_meta.get('model_dtype', 'unknown')}")
        print(f"Calibration       : {global_meta.get('calibration_source', 'unknown')}")

    metrics = evaluate_metrics(
        model=model,
        token_ids=token_ids,
        block_size=args.block_size,
        batch_size=args.batch_size,
        device=args.device,
        dtype_str=args.dtype,
        stride=args.stride,
        pad_token_id=pad_token_id,
        show_progress=show_progress,
        progress_update_every=args.progress_update_every,
    )

    print("\n=== Evaluation Metrics ===")
    print(f"Perplexity            : {metrics['perplexity']:.6f}")
    print(f"Mean loss (nats/token): {metrics['mean_loss']:.6f}")
    print(f"Bits per token        : {metrics['bits_per_token']:.6f}")
    print(f"Top-1 accuracy        : {metrics['top1_accuracy']:.6%}")
    print(f"Tokens evaluated      : {metrics['tokens_evaluated']:,}")
    print(f"Batches evaluated     : {metrics['batches_evaluated']:,}")
    print(f"Windows evaluated     : {metrics['windows_evaluated']:,}")
    print(f"Processed windows     : {metrics['processed_windows']:,}/{metrics['expected_windows']:,}")
    print(f"Stride used           : {metrics['stride_used']}")
    print(f"Evaluation mode       : {metrics['evaluation_mode']}")
    print(f"Elapsed eval time     : {format_seconds(metrics['elapsed_seconds'])}")
    print(f"Tokens/sec            : {metrics['tokens_per_second']:,.2f}")
    print(f"Windows/sec           : {metrics['windows_per_second']:,.2f}")
    print(f"Batches/sec           : {metrics['batches_per_second']:,.2f}")
    print_cuda_memory("CUDA memory after eval")

    total_script_seconds = now() - script_t0
    print(f"Total script time     : {format_seconds(total_script_seconds)}")

    if args.save_json is not None:
        global_meta = get_checkpoint_global_meta(compressed_info) if compressed_info else {}

        out = {
            "input_file": args.input_file,
            "model_id": args.model_id,
            "model_variant": model_variant,
            "compressed": args.compressed,
            "raw_chars": len(raw_text),
            "tokenized_length": int(len(token_ids)),
            "block_size": int(args.block_size),
            "stride": int(effective_stride),
            "batch_size": int(args.batch_size),
            "load_mode": args.load_mode,
            "dtype": args.dtype,
            "add_bos_token": bool(args.add_bos_token),
            "cache_dequantized": bool(args.cache_dequantized),
            "tokenization_stats": tok_stats,
            "metrics": metrics,
            "total_script_seconds": total_script_seconds,
            "cuda_memory": get_cuda_memory_string() if torch.cuda.is_available() else None,
            "compression_info": {
                k: v
                for k, v in global_meta.items()
                if k not in ("layers", "joint_sparsegpt_gptq_layers", "model")
            },
        }

        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        print(f"\nSaved JSON metrics to: {json_path}")


if __name__ == "__main__":
    main()