#!/usr/bin/env python3
"""
Create evaluation raw-text files and calibration token tensors for HF causal LMs.

Outputs:

1. Evaluation .txt file:
    Used by perplexity evaluator.
    Shared across models is okay.

2. Calibration .pt file:
    Contains:
        {
            "tokens": LongTensor [samples, seq_len],
            "model_id": ...,
            ...
        }

Recommended:

Evaluation:
    WikiText-103 test, 250k tokens

Calibration:
    C4 train, 64-128 samples x 2048 tokens
    or WikiText-103 train if you want faster setup.

Examples:

    python make_data.py \
      --model_id mistralai/Mistral-7B-Instruct-v0.3 \
      --make_eval \
      --eval_dataset wikitext103-test \
      --eval_tokens 250000 \
      --eval_out data/eval_wikitext103_test_shared_250k.txt

    python make_data.py \
      --model_id mistralai/Mistral-7B-Instruct-v0.3 \
      --make_calib \
      --calib_dataset c4-train \
      --calib_samples 128 \
      --calib_seq_len 2048 \
      --calib_out data/calib_c4_128x2048_mistral.pt \
      --calib_mode random \
      --seed 1234

Faster calibration:

    python make_data.py \
      --model_id mistralai/Mistral-7B-Instruct-v0.3 \
      --make_calib \
      --calib_dataset wikitext103-train \
      --calib_samples 128 \
      --calib_seq_len 2048 \
      --calib_out data/calib_wikitext103_train_128x2048_mistral.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


# ============================================================
# Small utilities
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


def fmt_int(x: float) -> str:
    return f"{int(x):,}"


def is_wikitext_header(text: str) -> bool:
    text = text.strip()
    return text.startswith("=") and text.endswith("=")


def print_progress(
    prefix: str,
    rows: int,
    chars: int,
    tokens: Optional[int],
    target_tokens: Optional[int],
    start_time: float,
    force_newline: bool = False,
) -> None:
    elapsed = time.time() - start_time
    char_s = chars / elapsed if elapsed > 0 else 0.0
    tok_s = tokens / elapsed if tokens is not None and elapsed > 0 else 0.0

    if tokens is not None and target_tokens is not None and target_tokens > 0:
        pct = 100.0 * min(tokens / target_tokens, 1.0)
        remaining = max(target_tokens - tokens, 0)
        eta = remaining / tok_s if tok_s > 0 else float("inf")
        token_part = (
            f"tokens={fmt_int(tokens)}/{fmt_int(target_tokens)} "
            f"({pct:5.1f}%) tok/s={tok_s:,.0f} eta={fmt_time(eta)}"
        )
    elif tokens is not None:
        token_part = f"tokens={fmt_int(tokens)} tok/s={tok_s:,.0f}"
    else:
        token_part = "tokens=?"

    msg = (
        f"\r{prefix}: rows={fmt_int(rows)} "
        f"chars={fmt_int(chars)} char/s={char_s:,.0f} "
        f"{token_part} elapsed={fmt_time(elapsed)}"
    )

    print(msg, end="\n" if force_newline else "", flush=True)


# ============================================================
# Dataset loading
# ============================================================

def load_text_dataset(dataset_name: str, streaming: bool):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "wikitext103-train":
        return load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split="train",
            streaming=streaming,
        )

    if dataset_name == "wikitext103-validation":
        return load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split="validation",
            streaming=streaming,
        )

    if dataset_name == "wikitext103-test":
        return load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split="test",
            streaming=streaming,
        )

    if dataset_name == "c4-train":
        return load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=True,
        )

    if dataset_name == "c4-validation":
        return load_dataset(
            "allenai/c4",
            "en",
            split="validation",
            streaming=True,
        )

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def iter_clean_texts(dataset_name: str, streaming: bool) -> Iterable[str]:
    ds = load_text_dataset(dataset_name, streaming=streaming)

    for row in ds:
        text = str(row["text"]).strip()

        if not text:
            continue

        if dataset_name.startswith("wikitext") and is_wikitext_header(text):
            continue

        if dataset_name.startswith("c4") and len(text) < 200:
            continue

        yield text


# ============================================================
# Token collection with visible metrics
# ============================================================

def collect_token_ids_from_dataset(
    tokenizer,
    dataset_name: str,
    target_tokens: int,
    streaming: bool,
    tokenize_every_chars: int,
    progress_every_rows: int,
    max_extra_factor: float,
) -> List[int]:
    """
    Collect at least target_tokens token ids.

    CRUCIAL:
        This prints progress and tokenizes periodically so you can see if the
        process is actually moving.
    """

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive.")

    texts: List[str] = []
    rows = 0
    chars = 0
    last_tokenized_chars = 0
    current_tokens = 0
    start_time = time.time()

    soft_char_limit = int(target_tokens * 8 * max_extra_factor)
    soft_char_limit = max(soft_char_limit, 200_000)

    print(
        f"[info] target_tokens={target_tokens:,}, "
        f"initial_soft_char_limit={soft_char_limit:,}, "
        f"tokenize_every_chars={tokenize_every_chars:,}"
    )

    for text in iter_clean_texts(dataset_name, streaming=streaming):
        texts.append(text)
        rows += 1
        chars += len(text)

        should_tokenize = (chars - last_tokenized_chars) >= tokenize_every_chars
        should_print = rows % progress_every_rows == 0

        if should_tokenize or should_print:
            joined = "\n\n".join(texts)

            if should_tokenize:
                ids = tokenizer.encode(joined, add_special_tokens=False)
                current_tokens = len(ids)
                last_tokenized_chars = chars

                print_progress(
                    prefix=f"Collecting {dataset_name}",
                    rows=rows,
                    chars=chars,
                    tokens=current_tokens,
                    target_tokens=target_tokens,
                    start_time=start_time,
                )

                if current_tokens >= target_tokens:
                    print_progress(
                        prefix=f"Collecting {dataset_name}",
                        rows=rows,
                        chars=chars,
                        tokens=current_tokens,
                        target_tokens=target_tokens,
                        start_time=start_time,
                        force_newline=True,
                    )
                    return ids[:target_tokens]

            elif should_print:
                print_progress(
                    prefix=f"Collecting {dataset_name}",
                    rows=rows,
                    chars=chars,
                    tokens=current_tokens if current_tokens > 0 else None,
                    target_tokens=target_tokens,
                    start_time=start_time,
                )

        if chars >= soft_char_limit:
            joined = "\n\n".join(texts)
            ids = tokenizer.encode(joined, add_special_tokens=False)
            current_tokens = len(ids)
            last_tokenized_chars = chars

            print_progress(
                prefix=f"Collecting {dataset_name}",
                rows=rows,
                chars=chars,
                tokens=current_tokens,
                target_tokens=target_tokens,
                start_time=start_time,
            )

            if current_tokens >= target_tokens:
                print_progress(
                    prefix=f"Collecting {dataset_name}",
                    rows=rows,
                    chars=chars,
                    tokens=current_tokens,
                    target_tokens=target_tokens,
                    start_time=start_time,
                    force_newline=True,
                )
                return ids[:target_tokens]

            old_limit = soft_char_limit
            soft_char_limit = int(soft_char_limit * 1.5)
            print(
                f"\n[info] Token count still low: {current_tokens:,}/{target_tokens:,}. "
                f"Increasing char limit {old_limit:,} -> {soft_char_limit:,}"
            )

    joined = "\n\n".join(texts)
    ids = tokenizer.encode(joined, add_special_tokens=False)

    print_progress(
        prefix=f"Collecting {dataset_name}",
        rows=rows,
        chars=chars,
        tokens=len(ids),
        target_tokens=target_tokens,
        start_time=start_time,
        force_newline=True,
    )

    if len(ids) < target_tokens:
        raise RuntimeError(
            f"Dataset ended before enough tokens were collected. "
            f"Need {target_tokens:,}, got {len(ids):,}."
        )

    return ids[:target_tokens]


# ============================================================
# Evaluation text generation
# ============================================================

def make_eval_file(
    tokenizer,
    dataset_name: str,
    eval_tokens: int,
    out_path: str,
    add_bos_token: bool,
    streaming: bool,
    tokenize_every_chars: int,
    progress_every_rows: int,
) -> None:
    print("=" * 100)
    print("Creating evaluation text")
    print(f"Dataset      : {dataset_name}")
    print(f"Target tokens: {eval_tokens:,}")

    ids = collect_token_ids_from_dataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        target_tokens=eval_tokens,
        streaming=streaming,
        tokenize_every_chars=tokenize_every_chars,
        progress_every_rows=progress_every_rows,
        max_extra_factor=1.25,
    )

    if add_bos_token:
        if tokenizer.bos_token_id is None:
            print("[warn] --eval_add_bos requested, but tokenizer has no bos_token_id.")
        else:
            ids = [int(tokenizer.bos_token_id)] + ids

    text = tokenizer.decode(
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    retok = tokenizer.encode(text, add_special_tokens=False)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    meta = {
        "type": "eval_text",
        "dataset": dataset_name,
        "model_id": tokenizer.name_or_path,
        "target_tokens": int(eval_tokens),
        "actual_ids_before_decode": int(len(ids)),
        "retokenized_tokens": int(len(retok)),
        "characters": int(len(text)),
        "add_bos_token": bool(add_bos_token),
        "output": str(out),
    }

    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nSaved evaluation file")
    print(f"  text file          : {out}")
    print(f"  metadata           : {meta_path}")
    print(f"  target ids         : {len(ids):,}")
    print(f"  retokenized tokens : {len(retok):,}")
    print(f"  characters         : {len(text):,}")


# ============================================================
# Calibration generation
# ============================================================

def make_contiguous_chunks(
    ids: List[int],
    samples: int,
    seq_len: int,
) -> torch.Tensor:
    needed = samples * seq_len

    if len(ids) < needed:
        raise RuntimeError(f"Need {needed:,} tokens, got {len(ids):,}.")

    chunks = []

    for i in range(samples):
        start = i * seq_len
        end = start + seq_len
        chunks.append(ids[start:end])

    return torch.tensor(chunks, dtype=torch.long)


def make_random_chunks(
    ids: List[int],
    samples: int,
    seq_len: int,
    seed: int,
) -> torch.Tensor:
    if len(ids) < seq_len + 1:
        raise RuntimeError(f"Need at least {seq_len + 1:,} tokens, got {len(ids):,}.")

    rng = random.Random(seed)
    max_start = len(ids) - seq_len

    chunks = []

    for _ in range(samples):
        start = rng.randint(0, max_start)
        end = start + seq_len
        chunks.append(ids[start:end])

    return torch.tensor(chunks, dtype=torch.long)


def make_calibration_file(
    tokenizer,
    dataset_name: str,
    samples: int,
    seq_len: int,
    out_path: str,
    mode: str,
    seed: int,
    token_pool_multiplier: int,
    streaming: bool,
    tokenize_every_chars: int,
    progress_every_rows: int,
) -> None:
    print("=" * 100)
    print("Creating calibration tokens")
    print(f"Dataset   : {dataset_name}")
    print(f"Samples   : {samples}")
    print(f"Seq len   : {seq_len}")
    print(f"Mode      : {mode}")
    print(f"Seed      : {seed}")

    needed = samples * seq_len

    if mode == "random":
        target_tokens = max(needed, token_pool_multiplier * needed)
    elif mode == "contiguous":
        target_tokens = needed
    else:
        raise ValueError("mode must be random or contiguous.")

    print(f"Needed calibration tokens : {needed:,}")
    print(f"Token pool target         : {target_tokens:,}")

    ids = collect_token_ids_from_dataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        target_tokens=target_tokens,
        streaming=streaming,
        tokenize_every_chars=tokenize_every_chars,
        progress_every_rows=progress_every_rows,
        max_extra_factor=1.25,
    )

    if mode == "random":
        tokens = make_random_chunks(
            ids=ids,
            samples=samples,
            seq_len=seq_len,
            seed=seed,
        )
    else:
        tokens = make_contiguous_chunks(
            ids=ids,
            samples=samples,
            seq_len=seq_len,
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "tokens": tokens,
        "model_id": tokenizer.name_or_path,
        "dataset": dataset_name,
        "samples": int(samples),
        "seq_len": int(seq_len),
        "mode": mode,
        "seed": int(seed),
        "token_pool_size": int(len(ids)),
        "calibration_tokens": int(tokens.numel()),
    }

    torch.save(ckpt, out)

    meta = {
        k: v
        for k, v in ckpt.items()
        if k != "tokens"
    }
    meta["shape"] = list(tokens.shape)
    meta["output"] = str(out)

    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nSaved calibration file")
    print(f"  calibration file : {out}")
    print(f"  metadata         : {meta_path}")
    print(f"  shape            : {tuple(tokens.shape)}")
    print(f"  calib tokens     : {tokens.numel():,}")
    print(f"  token pool size  : {len(ids):,}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--make_eval", action="store_true")
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="wikitext103-test",
        choices=[
            "wikitext103-train",
            "wikitext103-validation",
            "wikitext103-test",
            "c4-train",
            "c4-validation",
        ],
    )
    parser.add_argument("--eval_tokens", type=int, default=250_000)
    parser.add_argument("--eval_out", type=str, default="data/eval_wikitext103_test_250k.txt")
    parser.add_argument("--eval_add_bos", action="store_true")

    parser.add_argument("--make_calib", action="store_true")
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="c4-train",
        choices=[
            "wikitext103-train",
            "wikitext103-validation",
            "wikitext103-test",
            "c4-train",
            "c4-validation",
        ],
    )
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--calib_seq_len", type=int, default=2048)
    parser.add_argument("--calib_out", type=str, default="data/calib_c4_128x2048.pt")
    parser.add_argument("--calib_mode", type=str, default="random", choices=["random", "contiguous"])
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument(
        "--token_pool_multiplier",
        type=int,
        default=2,
        help=(
            "Only used for random calibration. "
            "2 means collect 2x samples*seq_len tokens then randomly sample chunks. "
            "Use 1 for fastest; 4 for more diversity."
        ),
    )

    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming for non-C4 datasets too. C4 is always streaming.",
    )

    parser.add_argument(
        "--tokenize_every_chars",
        type=int,
        default=250_000,
        help="Retokenize accumulated text every N new characters to report progress.",
    )

    parser.add_argument(
        "--progress_every_rows",
        type=int,
        default=100,
        help="Print progress every N rows even if not retokenizing.",
    )

    args = parser.parse_args()

    if not args.make_eval and not args.make_calib:
        raise ValueError("Nothing to do. Use --make_eval and/or --make_calib.")

    if args.calib_samples <= 0:
        raise ValueError("--calib_samples must be positive.")

    if args.calib_seq_len <= 0:
        raise ValueError("--calib_seq_len must be positive.")

    if args.eval_tokens <= 0:
        raise ValueError("--eval_tokens must be positive.")

    print(f"Loading tokenizer: {args.model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.make_eval:
        make_eval_file(
            tokenizer=tokenizer,
            dataset_name=args.eval_dataset,
            eval_tokens=args.eval_tokens,
            out_path=args.eval_out,
            add_bos_token=bool(args.eval_add_bos),
            streaming=bool(args.streaming),
            tokenize_every_chars=int(args.tokenize_every_chars),
            progress_every_rows=int(args.progress_every_rows),
        )

    if args.make_calib:
        make_calibration_file(
            tokenizer=tokenizer,
            dataset_name=args.calib_dataset,
            samples=int(args.calib_samples),
            seq_len=int(args.calib_seq_len),
            out_path=args.calib_out,
            mode=args.calib_mode,
            seed=int(args.seed),
            token_pool_multiplier=int(args.token_pool_multiplier),
            streaming=bool(args.streaming),
            tokenize_every_chars=int(args.tokenize_every_chars),
            progress_every_rows=int(args.progress_every_rows),
        )


if __name__ == "__main__":
    main()
