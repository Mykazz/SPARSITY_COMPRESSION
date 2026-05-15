#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


IGNORE_INDEX = -1


# ============================================================
# Helpers
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
    raise ValueError(f"Unsupported dtype: {name}")


def cuda_mem_string() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return f"alloc={alloc:.2f}GB reserved={reserv:.2f}GB peak={peak:.2f}GB"


def get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# Packed sparse mask
# ============================================================

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


# ============================================================
# Runtime AlphaSparse layer
# ============================================================

class AlphaSparseLinear(nn.Module):
    """
    Runtime wrapper for AlphaSparseGPT checkpoint.

    Stored:
        mask_packed : [out, ceil(in / 8)]
        values      : row-major kept values

    Runtime:
        W = zeros(out, in)
        W[mask] = values
        y = F.linear(x, W, bias)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask_packed: torch.Tensor,
        values: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        cache_dequantized: bool = False,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.cache_dequantized = bool(cache_dequantized)

        self.register_buffer("mask_packed", mask_packed.contiguous().to(torch.uint8))
        self.register_buffer("values", values.contiguous())

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
        else:
            self.bias = None

        self._cached_weight: Optional[torch.Tensor] = None

    @torch.no_grad()
    def dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        device = self.mask_packed.device

        mask = unpack_bool_mask_rows(
            self.mask_packed,
            original_cols=self.in_features,
        ).to(device)

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


# ============================================================
# Checkpoint loading
# ============================================================

@torch.no_grad()
def apply_alpha_sparse_checkpoint(
    model: nn.Module,
    ckpt: Dict[str, Any],
    device: torch.device,
    cache_dequantized: bool,
    skip_model_state: bool,
) -> nn.Module:
    if ckpt.get("format") != "hf_alpha_sparsegpt":
        raise ValueError(f"Expected format='hf_alpha_sparsegpt', got {ckpt.get('format')}")

    if not skip_model_state and "model" in ckpt and ckpt["model"]:
        print("Loading non-sparse parameters from ckpt['model']...")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"  missing keys    : {len(missing)}")
        print(f"  unexpected keys : {len(unexpected)}")

    layers = ckpt["alpha_sparsegpt_layers"]

    items = list(layers.items())
    iterator = items
    if tqdm is not None:
        iterator = tqdm(items, desc="Applying AlphaSparse layers", unit="layer", dynamic_ncols=True)

    total_weights = 0
    total_pruned = 0
    total_kept = 0

    for layer_name, st in iterator:
        old = get_module_by_name(model, layer_name)
        if not isinstance(old, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {layer_name}, got {type(old)}")

        shape = tuple(st["shape"])
        out_features, in_features = shape

        bias = old.bias.detach().clone() if old.bias is not None else None

        layer = AlphaSparseLinear(
            in_features=in_features,
            out_features=out_features,
            mask_packed=st["mask_packed"].to(device),
            values=st["values"].to(device),
            bias=bias.to(device) if bias is not None else None,
            cache_dequantized=cache_dequantized,
        )

        set_module_by_name(model, layer_name, layer)

        total_weights += int(st.get("total_count", out_features * in_features))
        total_pruned += int(st.get("pruned_count", 0))
        total_kept += int(st.get("total_count", out_features * in_features)) - int(st.get("pruned_count", 0))

    print(f"Applied AlphaSparse layers : {len(layers)}")
    print(f"Total selected weights     : {total_weights:,}")
    print(f"Kept weights               : {total_kept:,}")
    print(f"Pruned weights             : {total_pruned:,}")
    if total_weights:
        print(f"Actual sparsity            : {100.0 * total_pruned / total_weights:.2f}%")

    return model


def load_model_and_alpha_checkpoint(
    model_id: str,
    ckpt_path: str,
    dtype: torch.dtype,
    device: torch.device,
    attn_implementation: str,
    trust_remote_code: bool,
    low_cpu_mem_usage: bool,
    cache_dequantized: bool,
    skip_model_state: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }

    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    print("Loading tokenizer/model...")
    t0 = now()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    model.to(device)

    if hasattr(model, "config"):
        model.config.use_cache = False

    print(f"Loaded model in {format_seconds(now() - t0)}")
    if torch.cuda.is_available():
        print(f"CUDA memory after model load: {cuda_mem_string()}")

    print(f"Loading AlphaSparse checkpoint: {ckpt_path}")
    t0 = now()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint loaded in {format_seconds(now() - t0)}")

    model = apply_alpha_sparse_checkpoint(
        model=model,
        ckpt=ckpt,
        device=device,
        cache_dequantized=cache_dequantized,
        skip_model_state=skip_model_state,
    )

    if torch.cuda.is_available():
        print(f"CUDA memory after applying AlphaSparse: {cuda_mem_string()}")

    return tokenizer, model, ckpt


# ============================================================
# Data / windows
# ============================================================

def encode_text_file(tokenizer, input_file: str, add_bos_token: bool) -> Tuple[np.ndarray, str]:
    text = Path(input_file).read_text(encoding="utf-8")

    ids = tokenizer.encode(text, add_special_tokens=False)

    if add_bos_token:
        if tokenizer.bos_token_id is not None:
            ids = [tokenizer.bos_token_id] + ids
        else:
            print("[warn] add_bos_token requested, but tokenizer has no bos_token_id.")

    if len(ids) < 2:
        raise ValueError("Input text is too short after tokenization.")

    return np.array(ids, dtype=np.int64), text


def iterate_eval_windows(token_ids: np.ndarray, block_size: int, stride: Optional[int]):
    if stride is None:
        stride = block_size

    if stride <= 0:
        raise ValueError("stride must be positive.")
    if stride > block_size:
        raise ValueError("stride must be <= block_size.")

    n = len(token_ids)
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


def count_eval_windows(token_ids: np.ndarray, block_size: int, stride: Optional[int]) -> int:
    return sum(1 for _ in iterate_eval_windows(token_ids, block_size, stride))


def make_autocast_context(device: str, dtype_str: str):
    if "cuda" not in device:
        return nullcontext()
    if dtype_str == "float32":
        return nullcontext()

    dtype = parse_dtype(dtype_str)
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def safe_ppl(loss: float) -> float:
    return math.exp(loss) if loss < 20 else float("inf")


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    token_ids: np.ndarray,
    block_size: int,
    stride: Optional[int],
    batch_size: int,
    pad_token_id: int,
    dtype_str: str,
    device_str: str,
    show_progress: bool,
) -> Dict[str, Any]:
    if stride is None:
        stride = block_size

    total_windows = count_eval_windows(token_ids, block_size, stride)
    ctx = make_autocast_context(device_str, dtype_str)

    total_nll = 0.0
    total_tokens = 0
    total_correct = 0
    processed_windows = 0
    batches = 0

    batch_x: List[np.ndarray] = []
    batch_y: List[np.ndarray] = []
    batch_score_from: List[int] = []

    t0 = now()

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=total_windows, desc="Evaluating", unit="win", dynamic_ncols=True)

    def process_batch():
        nonlocal total_nll, total_tokens, total_correct, processed_windows, batches

        if not batch_x:
            return

        max_len = max(len(x) for x in batch_x)
        bs = len(batch_x)

        xb = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
        yb = torch.full((bs, max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)

        for i, (x_arr, y_arr, score_from) in enumerate(zip(batch_x, batch_y, batch_score_from)):
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
            out = model(
                input_ids=xb,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = out.logits

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                yb.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="mean",
            )

        valid = yb != IGNORE_INDEX
        scored = int(valid.sum().item())

        if scored == 0:
            return

        total_nll += float(loss.item()) * scored
        total_tokens += scored
        batches += 1

        preds = logits.argmax(dim=-1)
        total_correct += int(((preds == yb) & valid).sum().item())

        processed_windows += bs

        elapsed = now() - t0
        mean_loss = total_nll / max(total_tokens, 1)
        ppl = safe_ppl(mean_loss)
        tok_s = total_tokens / elapsed if elapsed > 0 else 0.0

        if pbar is not None:
            pbar.update(bs)
            pbar.set_postfix(
                {
                    "tok": f"{total_tokens:,}",
                    "tok/s": f"{tok_s:.0f}",
                    "loss": f"{mean_loss:.4f}",
                    "ppl": f"{ppl:.3f}",
                    "vram": f"{torch.cuda.memory_allocated()/1024**3:.1f}G" if torch.cuda.is_available() else "cpu",
                },
                refresh=False,
            )
        elif show_progress:
            pct = 100.0 * processed_windows / max(total_windows, 1)
            print(
                f"\rEvaluating: {processed_windows}/{total_windows} "
                f"({pct:.1f}%) tok={total_tokens:,} tok/s={tok_s:.0f} "
                f"loss={mean_loss:.4f} ppl={ppl:.3f}",
                end="",
                flush=True,
            )

    for x, y, score_from, _start in iterate_eval_windows(token_ids, block_size, stride):
        batch_x.append(x)
        batch_y.append(y)
        batch_score_from.append(score_from)

        if len(batch_x) == batch_size:
            process_batch()
            batch_x.clear()
            batch_y.clear()
            batch_score_from.clear()

    if batch_x:
        process_batch()

    if pbar is not None:
        pbar.close()
    elif show_progress:
        print()

    if total_tokens == 0:
        raise RuntimeError("No tokens evaluated.")

    elapsed = now() - t0
    mean_loss = total_nll / total_tokens

    return {
        "mean_loss": mean_loss,
        "perplexity": safe_ppl(mean_loss),
        "bits_per_token": mean_loss / math.log(2.0),
        "top1_accuracy": total_correct / total_tokens,
        "tokens_evaluated": total_tokens,
        "windows_evaluated": processed_windows,
        "expected_windows": total_windows,
        "batches_evaluated": batches,
        "stride_used": stride,
        "evaluation_mode": "non-overlapping" if stride == block_size else "overlapping",
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0.0,
        "windows_per_second": processed_windows / elapsed if elapsed > 0 else 0.0,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--alpha_sparse", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--add_bos_token", action="store_true")
    parser.add_argument("--cache_dequantized", action="store_true")
    parser.add_argument("--skip_model_state", action="store_true")
    parser.add_argument("--no_progress", action="store_true")

    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")

    parser.add_argument("--save_json", type=str, default=None)

    args = parser.parse_args()

    script_t0 = now()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    dtype = parse_dtype(args.dtype)
    device = torch.device(args.device)

    print("=" * 100)
    print("AlphaSparseGPT perplexity evaluation")
    print("=" * 100)
    print(f"model_id          : {args.model_id}")
    print(f"alpha_sparse ckpt : {args.alpha_sparse}")
    print(f"input_file        : {args.input_file}")
    print(f"dtype             : {dtype}")
    print(f"block_size        : {args.block_size}")
    print(f"stride            : {args.stride}")
    print(f"batch_size        : {args.batch_size}")
    print(f"cache_dequantized : {args.cache_dequantized}")
    print(f"skip_model_state  : {args.skip_model_state}")

    tokenizer, model, ckpt = load_model_and_alpha_checkpoint(
        model_id=args.model_id,
        ckpt_path=args.alpha_sparse,
        dtype=dtype,
        device=device,
        attn_implementation=args.attn_implementation,
        trust_remote_code=bool(args.trust_remote_code),
        low_cpu_mem_usage=bool(args.low_cpu_mem_usage),
        cache_dequantized=bool(args.cache_dequantized),
        skip_model_state=bool(args.skip_model_state),
    )

    print("\nTokenizing eval file...")
    token_ids, raw_text = encode_text_file(
        tokenizer=tokenizer,
        input_file=args.input_file,
        add_bos_token=bool(args.add_bos_token),
    )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    expected_windows = count_eval_windows(token_ids, args.block_size, args.stride)

    print(f"Raw chars        : {len(raw_text):,}")
    print(f"Tokenized length : {len(token_ids):,}")
    print(f"Windows          : {expected_windows:,}")
    print(f"Pad token id     : {pad_token_id}")
    if torch.cuda.is_available():
        print(f"CUDA memory      : {cuda_mem_string()}")

    metrics = evaluate(
        model=model,
        token_ids=token_ids,
        block_size=args.block_size,
        stride=args.stride,
        batch_size=args.batch_size,
        pad_token_id=pad_token_id,
        dtype_str=args.dtype,
        device_str=args.device,
        show_progress=not args.no_progress,
    )

    print("\n=== Metrics ===")
    print(f"Perplexity             : {metrics['perplexity']:.6f}")
    print(f"Mean loss              : {metrics['mean_loss']:.6f}")
    print(f"Bits/token             : {metrics['bits_per_token']:.6f}")
    print(f"Top-1 accuracy         : {metrics['top1_accuracy']:.4%}")
    print(f"Tokens evaluated       : {metrics['tokens_evaluated']:,}")
    print(f"Windows evaluated      : {metrics['windows_evaluated']:,}/{metrics['expected_windows']:,}")
    print(f"Eval time              : {format_seconds(metrics['elapsed_seconds'])}")
    print(f"Tokens/sec             : {metrics['tokens_per_second']:.2f}")
    print(f"Total script time      : {format_seconds(now() - script_t0)}")
    if torch.cuda.is_available():
        print(f"CUDA memory            : {cuda_mem_string()}")

    if args.save_json is not None:
        meta = ckpt.get("alpha_sparsegpt_meta", {})

        out = {
            "model_id": args.model_id,
            "model_variant": "alpha_sparsegpt",
            "alpha_sparse": args.alpha_sparse,
            "input_file": args.input_file,
            "raw_chars": len(raw_text),
            "tokenized_length": int(len(token_ids)),
            "block_size": int(args.block_size),
            "stride": int(args.stride),
            "batch_size": int(args.batch_size),
            "dtype": args.dtype,
            "cache_dequantized": bool(args.cache_dequantized),
            "metrics": metrics,
            "compression_info": meta,
            "cuda_memory": cuda_mem_string() if torch.cuda.is_available() else None,
            "total_script_seconds": now() - script_t0,
        }

        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        print(f"\nSaved JSON: {json_path}")


if __name__ == "__main__":
    main()