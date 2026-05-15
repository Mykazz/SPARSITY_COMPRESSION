#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
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
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
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
    raise ValueError(f"Unsupported dtype: {name}")


def cuda_mem() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    p = torch.cuda.max_memory_allocated() / 1024**3
    return f"alloc={a:.2f}GB reserved={r:.2f}GB peak={p:.2f}GB"


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
# Packed mask helpers
# ============================================================

def unpack_bool_mask_rows(packed: torch.Tensor, original_cols: int) -> torch.Tensor:
    """
    Unpack uint8 [rows, ceil(cols / 8)] into bool [rows, cols].

    True  = kept weight
    False = pruned weight
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


# ============================================================
# Wanda++ sparse runtime layer
# ============================================================

class WandaSparseLinear(nn.Module):
    """
    Runtime layer for Wanda++ sparse storage:

        mask   : packedbits [out, ceil(in / 8)]
        values : surviving weights in row-major order

    Reconstruct:

        W = zeros(out, in)
        W[mask] = values

    This is mathematically correct but not kernel optimized.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        packed_mask: torch.Tensor,
        values: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        cache_dequantized: bool = False,
        layer_name: str = "",
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.cache_dequantized = bool(cache_dequantized)
        self.layer_name = str(layer_name)

        self.register_buffer("packed_mask", packed_mask.contiguous().to(torch.uint8))
        self.register_buffer("values", values.contiguous())

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
        else:
            self.bias = None

        self._cached_weight: Optional[torch.Tensor] = None

    @torch.no_grad()
    def dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        mask = unpack_bool_mask_rows(
            self.packed_mask,
            original_cols=self.in_features,
        )

        kept = int(mask.sum().item())

        if kept != self.values.numel():
            raise RuntimeError(
                f"{self.layer_name}: mask/value mismatch. "
                f"mask kept={kept}, values={self.values.numel()}"
            )

        w = torch.zeros(
            (self.out_features, self.in_features),
            device=self.values.device,
            dtype=dtype,
        )

        w[mask] = self.values.to(dtype=dtype)

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
# Checkpoint application
# ============================================================

@torch.no_grad()
def apply_wandapp_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    device: torch.device,
    cache_dequantized: bool,
    skip_model_state: bool,
) -> Dict[str, Any]:
    print(f"Loading Wanda++ checkpoint: {ckpt_path}")
    t0 = now()

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if ckpt.get("format") != "hf_wandapp_pruned":
        raise ValueError(f"Expected format='hf_wandapp_pruned', got {ckpt.get('format')}")

    if "wandapp_layers" not in ckpt:
        raise ValueError("Checkpoint missing key 'wandapp_layers'.")

    print(f"Checkpoint loaded in {format_seconds(now() - t0)}")

    if not skip_model_state and "model" in ckpt and ckpt["model"]:
        print("Loading non-compressed parameters from ckpt['model']...")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"  missing keys    : {len(missing)}")
        print(f"  unexpected keys : {len(unexpected)}")
    else:
        print("[info] Skipping ckpt['model'] partial state_dict load.")

    layers = ckpt["wandapp_layers"]
    items = list(layers.items())

    iterator = tqdm(items, desc="Applying Wanda++ layers", unit="layer") if tqdm else items

    total_values = 0
    total_kept = 0
    total_weights = 0

    for layer_name, st in iterator:
        old_layer = get_module_by_name(model, layer_name)

        if not isinstance(old_layer, nn.Linear):
            raise TypeError(f"{layer_name} is not nn.Linear: {type(old_layer)}")

        out_features, in_features = tuple(st["shape"])

        packed_mask = st["mask"]
        values = st["values"]

        kept_count = int(st.get("kept_count", values.numel()))
        total_count = int(st.get("total_count", out_features * in_features))

        if values.numel() != kept_count:
            print(
                f"[warn] {layer_name}: values numel {values.numel()} != kept_count {kept_count}"
            )

        bias = old_layer.bias.detach().cpu() if old_layer.bias is not None else None

        qlayer = WandaSparseLinear(
            in_features=in_features,
            out_features=out_features,
            packed_mask=packed_mask.to(device),
            values=values.to(device),
            bias=bias.to(device) if bias is not None else None,
            cache_dequantized=cache_dequantized,
            layer_name=layer_name,
        )

        set_module_by_name(model, layer_name, qlayer)

        total_values += values.numel()
        total_kept += kept_count
        total_weights += total_count

    model.eval()

    actual_sparsity = 1.0 - total_kept / float(total_weights)

    print(f"Applied Wanda++ layers : {len(items)}")
    print(f"Total selected weights : {total_weights:,}")
    print(f"Kept weights           : {total_kept:,}")
    print(f"Values stored          : {total_values:,}")
    print(f"Actual sparsity        : {100.0 * actual_sparsity:.2f}%")
    print(f"CUDA memory            : {cuda_mem()}")

    return ckpt


# ============================================================
# Data and eval windows
# ============================================================

def encode_text_file(tokenizer, input_file: str, add_bos_token: bool) -> Tuple[np.ndarray, str]:
    text = Path(input_file).read_text(encoding="utf-8")

    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if add_bos_token:
        if tokenizer.bos_token_id is not None:
            ids = [tokenizer.bos_token_id] + ids
        else:
            print("[warn] tokenizer has no BOS token")

    if len(ids) < 2:
        raise ValueError("Input file is too short after tokenization.")

    return np.asarray(ids, dtype=np.int64), text


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


def count_windows(token_ids: np.ndarray, block_size: int, stride: Optional[int]) -> int:
    return sum(1 for _ in iterate_eval_windows(token_ids, block_size, stride))


def make_autocast(device: str, dtype: torch.dtype):
    enabled = device.startswith("cuda") and dtype in (torch.float16, torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def safe_ppl(loss: float) -> float:
    return math.exp(loss) if loss < 20 else float("inf")


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    token_ids: np.ndarray,
    block_size: int,
    stride: Optional[int],
    batch_size: int,
    pad_token_id: int,
    dtype: torch.dtype,
    device: str,
) -> Dict[str, Any]:
    if stride is None:
        stride = block_size

    total_windows = count_windows(token_ids, block_size, stride)
    ctx = make_autocast(device, dtype)

    total_nll = 0.0
    total_tokens = 0
    total_correct = 0
    batches = 0
    processed_windows = 0

    bx: List[np.ndarray] = []
    by: List[np.ndarray] = []
    bscore: List[int] = []

    t0 = now()

    pbar = tqdm(total=total_windows, desc="Evaluating", unit="win") if tqdm else None

    def process_batch() -> None:
        nonlocal total_nll, total_tokens, total_correct, batches

        if not bx:
            return

        max_len = max(len(x) for x in bx)

        xb = torch.full((len(bx), max_len), pad_token_id, dtype=torch.long)
        yb = torch.full((len(bx), max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(bx), max_len), dtype=torch.long)

        for i, (x_arr, y_arr, score_from) in enumerate(zip(bx, by, bscore)):
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

            # CRUCIAL:
            # x and y are externally shifted.
            # Do NOT shift logits/labels again.
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
        total_correct += int(((logits.argmax(dim=-1) == yb) & valid).sum().item())
        batches += 1

    for x, y, score_from, _ in iterate_eval_windows(token_ids, block_size, stride):
        bx.append(x)
        by.append(y)
        bscore.append(score_from)

        if len(bx) == batch_size:
            nwin = len(bx)
            process_batch()
            processed_windows += nwin

            bx.clear()
            by.clear()
            bscore.clear()

            if pbar:
                elapsed = now() - t0
                mean_loss = total_nll / total_tokens if total_tokens else float("nan")
                pbar.update(nwin)
                pbar.set_postfix(
                    {
                        "tok": f"{total_tokens:,}",
                        "tok/s": f"{total_tokens / elapsed:.0f}" if elapsed > 0 else "0",
                        "loss": f"{mean_loss:.4f}",
                        "ppl": f"{safe_ppl(mean_loss):.3f}",
                        "vram": f"{torch.cuda.memory_allocated() / 1024**3:.1f}G" if torch.cuda.is_available() else "CPU",
                    }
                )

    if bx:
        nwin = len(bx)
        process_batch()
        processed_windows += nwin
        if pbar:
            pbar.update(nwin)

    if pbar:
        pbar.close()

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
        "stride": stride,
        "block_size": block_size,
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0.0,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--wandapp", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--add_bos_token", action="store_true")
    parser.add_argument("--cache_dequantized", action="store_true")
    parser.add_argument("--skip_model_state", action="store_true")

    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="eager")

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
    print("Wanda++ pruned HF model perplexity evaluation")
    print("=" * 100)
    print(f"model_id          : {args.model_id}")
    print(f"wandapp checkpoint: {args.wandapp}")
    print(f"input_file        : {args.input_file}")
    print(f"dtype             : {dtype}")
    print(f"block_size        : {args.block_size}")
    print(f"stride            : {args.stride}")
    print(f"batch_size        : {args.batch_size}")
    print(f"cache_dequantized : {args.cache_dequantized}")
    print(f"skip_model_state  : {args.skip_model_state}")

    print("\nLoading tokenizer/model...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=bool(args.trust_remote_code),
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": bool(args.low_cpu_mem_usage),
        "trust_remote_code": bool(args.trust_remote_code),
    }

    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)

    if hasattr(model, "config"):
        model.config.use_cache = False

    model.eval()
    model.to(device)

    print(f"CUDA memory after model load: {cuda_mem()}")

    ckpt = apply_wandapp_checkpoint(
        model=model,
        ckpt_path=args.wandapp,
        device=device,
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

    print(f"Raw chars        : {len(raw_text):,}")
    print(f"Tokenized length : {len(token_ids):,}")
    print(f"Windows          : {count_windows(token_ids, args.block_size, args.stride):,}")
    print(f"Pad token id     : {pad_token_id}")
    print(f"CUDA memory      : {cuda_mem()}")

    metrics = evaluate(
        model=model,
        token_ids=token_ids,
        block_size=args.block_size,
        stride=args.stride,
        batch_size=args.batch_size,
        pad_token_id=pad_token_id,
        dtype=dtype,
        device=args.device,
    )

    total_time = now() - script_t0

    print("\n=== Metrics ===")
    print(f"Perplexity             : {metrics['perplexity']:.6f}")
    print(f"Mean loss              : {metrics['mean_loss']:.6f}")
    print(f"Bits/token             : {metrics['bits_per_token']:.6f}")
    print(f"Top-1 accuracy         : {100.0 * metrics['top1_accuracy']:.4f}%")
    print(f"Tokens evaluated       : {metrics['tokens_evaluated']:,}")
    print(f"Windows evaluated      : {metrics['windows_evaluated']:,}/{metrics['expected_windows']:,}")
    print(f"Eval time              : {format_seconds(metrics['elapsed_seconds'])}")
    print(f"Tokens/sec             : {metrics['tokens_per_second']:.2f}")
    print(f"Total script time      : {format_seconds(total_time)}")
    print(f"CUDA memory            : {cuda_mem()}")

    if args.save_json:
        meta = ckpt.get("wandapp_meta", {})

        out = {
            "model_id": args.model_id,
            "variant": "wandapp_pruned",
            "wandapp": args.wandapp,
            "input_file": args.input_file,
            "raw_chars": len(raw_text),
            "tokenized_length": int(len(token_ids)),
            "dtype": args.dtype,
            "block_size": int(args.block_size),
            "stride": int(args.stride),
            "batch_size": int(args.batch_size),
            "cache_dequantized": bool(args.cache_dequantized),
            "metrics": metrics,
            "wandapp_meta": meta,
            "total_script_seconds": total_time,
            "cuda_memory": cuda_mem(),
        }

        p = Path(args.save_json)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        print(f"\nSaved JSON: {p}")


if __name__ == "__main__":
    main()