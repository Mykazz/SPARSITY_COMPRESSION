#!/usr/bin/env python3
"""
Standard-protocol perplexity eval: WikiText-2-raw test + C4 validation @ 2048,
matching the SparseGPT / Wanda / GPTQ convention used by most pruning papers
(incl. the ELSA table you're comparing to).

What it does:
  1. Builds & caches the two eval sets the standard way (only if missing):
       - WikiText-2-raw test: "\n\n".join(test['text']) -> tokenize -> reshape
         into non-overlapping seqlen windows.
       - C4 'en' validation: 256 random seqlen-segments (seed 0), Wanda-style.
     Caches are tokenized tensors keyed by --data_tag (share them between the
     dense base and your sparse checkpoint — same tokenizer => same cache).
  2. Loads the model and reports perplexity on both, GPTQ convention
       ppl = exp( sum_i (mean_CE_i * seqlen) / (nwindows * seqlen) ).

Run it once on the dense base and once on each sparse checkpoint (same
--data_tag) and compare to the paper's Wiki / C4 columns.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def build_wikitext2(tokenizer, cache_pt: Path):
    """Tokenized WikiText-2-raw test as a single [1, N] tensor (standard join)."""
    if cache_pt.exists():
        return torch.load(cache_pt)
    from datasets import load_dataset
    print("  building WikiText-2-raw test cache...")
    test = None
    last_exc = None
    for ds_id in ("Salesforce/wikitext", "wikitext", "EleutherAI/wikitext_document_level"):
        try:
            test = load_dataset(ds_id, "wikitext-2-raw-v1", split="test")
            break
        except Exception as exc:  # newer datasets rejects the bare 'wikitext' id
            last_exc = exc
            test = None
    if test is None:
        raise RuntimeError(f"Could not load wikitext-2-raw-v1 test. Last error: {last_exc}")
    text = "\n\n".join(test["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids  # [1, N]
    cache_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ids, cache_pt)
    print(f"  cached {tuple(ids.shape)} -> {cache_pt}")
    return ids


def build_c4(tokenizer, seqlen: int, nsamples: int, seed: int, cache_pt: Path):
    """C4 'en' validation: nsamples random seqlen-token segments (Wanda/GPTQ style)."""
    if cache_pt.exists():
        return torch.load(cache_pt)
    from datasets import load_dataset
    print("  building C4 validation cache (downloads one ~300MB shard)...")
    segs = []
    # Primary: non-streaming validation shard -> random nsamples segments (Wanda/GPTQ convention).
    try:
        val = load_dataset(
            "allenai/c4", "en",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        random.seed(seed)
        tries = 0
        while len(segs) < nsamples:
            tries += 1
            i = random.randint(0, len(val) - 1)
            enc = tokenizer(val[i]["text"], return_tensors="pt").input_ids
            if enc.shape[1] <= seqlen:
                continue
            j = random.randint(0, enc.shape[1] - seqlen - 1)
            segs.append(enc[:, j:j + seqlen])
            if len(segs) % 64 == 0:
                print(f"    sampled {len(segs)}/{nsamples} C4 segments (in {tries} tries)")
    except Exception as exc:
        # Fallback: stream the validation split and take sequential segments (deterministic).
        print(f"    [c4] non-streaming load failed ({exc}); using streaming sequential fallback...")
        segs = []
        buf: list[int] = []
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        for ex in ds:
            buf.extend(tokenizer(ex["text"], add_special_tokens=False).input_ids)
            while len(buf) >= seqlen:
                segs.append(torch.tensor(buf[:seqlen], dtype=torch.long).unsqueeze(0))
                buf = buf[seqlen:]
                if len(segs) >= nsamples:
                    break
            if len(segs) >= nsamples:
                break
    ids = torch.cat(segs, dim=0)  # [nsamples, seqlen]
    cache_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ids, cache_pt)
    print(f"  cached {tuple(ids.shape)} -> {cache_pt}")
    return ids


@torch.no_grad()
def eval_ppl(model, ids: torch.Tensor, seqlen: int, device, label: str) -> dict:
    """GPTQ-convention perplexity over non-overlapping seqlen windows."""
    if ids.dim() == 2 and ids.shape[0] == 1:           # wikitext long sequence
        n = ids.shape[1] // seqlen
        windows = ids[:, : n * seqlen].reshape(n, seqlen)
    else:                                               # c4 [nsamples, seqlen]
        windows = ids
        n = windows.shape[0]
    nlls = []
    t0 = time.time()
    for i in range(n):
        batch = windows[i:i + 1].to(device)
        logits = model(batch).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = batch[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        nlls.append(loss.double() * seqlen)             # GPTQ convention
        if (i + 1) % 25 == 0 or (i + 1) == n:
            cur = float(torch.exp(torch.stack(nlls).sum() / ((i + 1) * seqlen)))
            print(f"    [{label}] {i + 1}/{n} windows  running ppl={cur:.4f}", flush=True)
    ppl = float(torch.exp(torch.stack(nlls).sum() / (n * seqlen)))
    return {"perplexity": ppl, "windows": int(n), "seqlen": int(seqlen),
            "seconds": float(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True, help="HF repo or local compressed/ checkpoint folder")
    ap.add_argument("--data_tag", required=True,
                    help="Tokenizer family tag for cache reuse, e.g. 'llama2' or 'mistral'. "
                         "Use the SAME tag for the dense base and its sparse checkpoints.")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--c4_samples", type=int, default=256)
    ap.add_argument("--c4_seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--attn_implementation", type=str, default="sdpa")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--low_cpu_mem_usage", action="store_true")
    ap.add_argument("--cache_dir", type=str, default="data/eval_cache")
    ap.add_argument("--skip_c4", action="store_true")
    ap.add_argument("--skip_wikitext", action="store_true")
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir)
    wiki_pt = cache_dir / f"wikitext2_test_{args.data_tag}_sl{args.seqlen}.pt"
    c4_pt = cache_dir / f"c4_val_{args.data_tag}_{args.c4_samples}x{args.seqlen}_seed{args.c4_seed}.pt"

    print("=" * 90)
    print("Standard-protocol PPL eval (WikiText-2-raw + C4)  |  GPTQ convention")
    print(f"  model    : {args.model_id}")
    print(f"  seqlen   : {args.seqlen}   data_tag: {args.data_tag}   dtype: {args.dtype}")
    print("=" * 90)

    # Tokenizer + eval-set caches first (no model resident during dataset build).
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code, use_fast=True)
    wiki_ids = None if args.skip_wikitext else build_wikitext2(tokenizer, wiki_pt)
    c4_ids = None if args.skip_c4 else build_c4(tokenizer, args.seqlen, args.c4_samples, args.c4_seed, c4_pt)

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=parse_dtype(args.dtype),
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.to(device).eval()

    results = {"model_id": args.model_id, "data_tag": args.data_tag, "seqlen": args.seqlen,
               "dtype": args.dtype}
    if wiki_ids is not None:
        print("\nEvaluating WikiText-2-raw test...")
        results["wikitext2"] = eval_ppl(model, wiki_ids, args.seqlen, device, "wiki2")
    if c4_ids is not None:
        print("\nEvaluating C4 validation...")
        results["c4"] = eval_ppl(model, c4_ids, args.seqlen, device, "c4")

    print("\n" + "=" * 90)
    print("RESULTS")
    if "wikitext2" in results:
        print(f"  WikiText-2-raw  ppl = {results['wikitext2']['perplexity']:.4f}  "
              f"({results['wikitext2']['windows']} windows)")
    if "c4" in results:
        print(f"  C4 validation   ppl = {results['c4']['perplexity']:.4f}  "
              f"({results['c4']['windows']} windows)")
    print("=" * 90)

    out_json = args.out_json or f"results/{Path(args.model_id).name}_wiki2c4_sl{args.seqlen}.json"
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
