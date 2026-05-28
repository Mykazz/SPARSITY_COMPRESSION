#!/usr/bin/env python3
"""
Fast calibration-token builder for Hugging Face causal LMs.

Creates a .pt file compatible with elsa_mistral_windowed_stable.py and your
SparseGPT-style scripts: either a tensor [N, T] or dict['tokens'] [N, T].

Default source is WikiText-103 train. It tokenizes documents until enough tokens
are collected, then writes contiguous chunks. This is usually fast enough and
avoids tokenizing the whole dataset when you only need e.g. 512x2048 tokens.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterable, List

import torch
from transformers import AutoTokenizer


def now() -> float:
    return time.time()


def fmt_time(s: float) -> str:
    s = int(max(0, s))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def iter_hf_dataset_texts(dataset: str, config: str, split: str, text_field: str) -> Iterable[str]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "The 'datasets' package is required for HF dataset loading. Install with:\n"
            "  /venv/main/bin/python -m pip install datasets\n"
            f"Original error: {exc}"
        )
    ds = load_dataset(dataset, config, split=split)
    for item in ds:
        text = item.get(text_field, "")
        if isinstance(text, str) and text.strip():
            yield text


def iter_text_file(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line.rstrip("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build calibration token .pt for Mistral/LLaMA/Qwen.")
    ap.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--num_sequences", type=int, default=512)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--add_eos_between_docs", action="store_true")
    ap.set_defaults(add_eos_between_docs=True)

    src = ap.add_argument_group("source")
    src.add_argument("--source_text_file", type=str, default="", help="Optional local text file. If set, HF dataset is not used.")
    src.add_argument("--dataset", type=str, default="wikitext")
    src.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    src.add_argument("--split", type=str, default="train")
    src.add_argument("--text_field", type=str, default="text")

    ap.add_argument("--shuffle_docs", action="store_true", help="Materialize and shuffle docs before tokenizing; slower and uses more RAM.")
    ap.add_argument("--save_plain_tensor", action="store_true", help="Save just tensor instead of dict with metadata.")
    args = ap.parse_args()

    random.seed(args.seed)
    t0 = now()
    need = int(args.num_sequences) * int(args.seq_len)
    if need <= 0:
        raise ValueError("num_sequences * seq_len must be positive")

    print("=" * 100)
    print("Calibration token builder")
    print("=" * 100)
    print(f"Model/tokenizer : {args.model_id}")
    print(f"Output          : {args.out}")
    print(f"Target shape    : ({args.num_sequences}, {args.seq_len}) = {need:,} tokens")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    eos = tokenizer.eos_token_id

    if args.source_text_file:
        print(f"Source          : local text file {args.source_text_file}")
        texts_iter = iter_text_file(args.source_text_file)
    else:
        print(f"Source          : HF dataset {args.dataset}/{args.dataset_config} split={args.split}")
        texts_iter = iter_hf_dataset_texts(args.dataset, args.dataset_config, args.split, args.text_field)

    if args.shuffle_docs:
        print("Materializing documents for shuffle...")
        docs = list(texts_iter)
        random.shuffle(docs)
        texts_iter = iter(docs)
        print(f"Documents       : {len(docs):,}")

    ids: List[int] = []
    docs_seen = 0
    tok_t0 = now()
    for text in texts_iter:
        docs_seen += 1
        piece = tokenizer.encode(text, add_special_tokens=False)
        if piece:
            ids.extend(piece)
            if args.add_eos_between_docs and eos is not None:
                ids.append(int(eos))
        if len(ids) >= need:
            break
        if docs_seen % 10000 == 0:
            rate = len(ids) / max(1e-9, now() - tok_t0)
            print(f"  docs={docs_seen:,} tokens={len(ids):,}/{need:,} tok/s={rate:,.0f}")

    if len(ids) < need:
        raise RuntimeError(f"Not enough tokens: got {len(ids):,}, need {need:,}")

    ids = ids[:need]
    tokens = torch.tensor(ids, dtype=torch.long).view(args.num_sequences, args.seq_len).contiguous()

    meta = {
        "model_id": args.model_id,
        "shape": list(tokens.shape),
        "tokens": int(tokens.numel()),
        "source_text_file": args.source_text_file or None,
        "dataset": args.dataset if not args.source_text_file else None,
        "dataset_config": args.dataset_config if not args.source_text_file else None,
        "split": args.split if not args.source_text_file else None,
        "docs_seen": docs_seen,
        "seed": args.seed,
        "elapsed_seconds": now() - t0,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_plain_tensor:
        torch.save(tokens, out)
    else:
        torch.save({"tokens": tokens, "meta": meta}, out)
    with open(str(out) + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved           : {out}")
    print(f"Meta            : {out}.meta.json")
    print(f"Docs seen       : {docs_seen:,}")
    print(f"Elapsed         : {fmt_time(now() - t0)}")
    print(f"Tokens/sec      : {tokens.numel() / max(1e-9, now() - tok_t0):,.0f}")


if __name__ == "__main__":
    main()
