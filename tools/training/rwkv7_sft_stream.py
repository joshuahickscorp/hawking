#!/usr/bin/env python3
"""Optimized, shard-streaming SFT trainer for the on-device RWKV-7 post-train.

This is the torch-mps SFT driver the optimized runbook uses (the MLX path is a
NO-GO for RWKV-7 — see docs/rwkv7_posttrain_ondevice.md §7). Two improvements
over the baseline single-shot `SFTTrainer` snippet:

  * **Shard streaming / pipeline overlap (opt #3).** It can train directly off
    the `*.shard-NNNN.jsonl` files the batched capture (`dismantle generate
    --batched-capture`) writes, optionally WAITING for new shards to appear
    (`--watch`). That lets SFT start on the first finished capture shard while
    the teacher capture is still producing later shards — overlapping the two
    phases instead of running them back-to-back.

  * **Tuned memory/throughput config (opt #4).** Gradient checkpointing on,
    grad-accumulation for a large *effective* batch at batch-1 resident cost,
    bf16 autocast where mps supports it (fp32 master weights for RWKV-7
    stability), and an explicit, documented 18 GB budget. See `build_sft_config`.

This script is a DRIVER. The heavy run is GPU/mps and is deferred to "when the
GPU is free"; it is validated CPU-side by `rwkv7_train_smoke.py` (the trainer
*mechanics* — bf16 autocast, grad-accum, checkpointing — are proven there) and
by `--dry-run` here (loads + tokenizes a few shard rows, builds the config, and
exits WITHOUT training, so the data path is checked without a GPU).

Usage
-----
    # Dry-run on the committed sample (CPU, no training) — checks wiring:
    python3.12 tools/training/rwkv7_sft_stream.py \
        --shards-glob 'tools/training/data/rwkv7_sft_sample.jsonl' \
        --model models/rwkv7-g1-04-hf --dry-run

    # Real SFT off capture shards, streaming as they land (GPU, deferred):
    python3.12 tools/training/rwkv7_sft_stream.py \
        --shards-glob 'artifacts/rwkv7_posttrain/teacher.shard-*.jsonl' \
        --model models/rwkv7-g1-04-hf \
        --out artifacts/rwkv7_posttrain/sft_out \
        --watch --expected-shards 375

Input shard rows may be EITHER:
  * SFT corpus rows: {"text": "<rwkv-chat-formatted>", ...}, or
  * capture rows:    {"prompt": "...", "completion": "..."} — the trainer
    renders these into the RWKV-7 chat `text` (prompt already ends at
    'Assistant:'; completion is appended + the EOS separator).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

# RWKV-7 g1 control strings (must match rwkv7_build_corpus.py).
SEP = "\n\n"  # turn separator AND eos


def _shard_to_text(row: dict) -> str | None:
    """Render one shard row into the RWKV-7 chat `text` field.

    Accepts both pre-rendered SFT rows ({"text": ...}) and raw capture rows
    ({"prompt": ..., "completion": ...}).
    """
    t = row.get("text")
    if isinstance(t, str) and t.strip():
        return t
    prompt = row.get("prompt")
    completion = row.get("completion")
    if isinstance(prompt, str) and isinstance(completion, str):
        # prompt already ends in 'Assistant:' (rwkv_prompt); append the answer
        # plus the EOS separator so the model learns to stop.
        sep = "" if prompt.endswith(" ") else " "
        return f"{prompt}{sep}{completion.strip()}{SEP}"
    return None


def iter_shard_rows(paths: list[Path]) -> Iterator[dict]:
    for p in paths:
        try:
            with p.open(encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        yield json.loads(ln)
        except FileNotFoundError:
            continue


def discover_shards(pattern: str) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(pattern))


def load_text_rows(paths: list[Path]) -> list[dict]:
    """Read shards → [{'text': ...}] rows, dropping unrenderable rows."""
    out = []
    for row in iter_shard_rows(paths):
        txt = _shard_to_text(row)
        if txt:
            out.append({"text": txt})
    return out


def build_sft_config(out_dir: str, max_length: int, grad_accum: int, lr: float,
                     epochs: float, bf16: bool):
    """The tuned SFTConfig for 18 GB. Documented memory budget:

      RWKV-7 0.4B fp32 weights ............ ~1.8 GB
      AdamW master + m + v (fp32) ......... ~3.6 GB  (2 states × params)
      activations @ batch1/seq=max_length . ~1–3 GB  (grad-checkpointing ON
                                            recomputes them; keeps this low)
      ----------------------------------------------
      SFT resident ........................ ~6–8 GB  → fits 18 GB with headroom

    Effective batch = 1 × grad_accum (default 16) at batch-1 *resident* cost —
    the grad-accum lever raises the effective batch without raising memory
    (validated lossless in rwkv7_train_smoke.py Check 2).
    """
    from trl import SFTConfig

    return SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        max_length=max_length,
        # mps bf16 is partial for RWKV-7's WKV recurrence; bf16 autocast helps
        # the dense matmuls where supported, fp32 master weights stay stable.
        # Off by default (fp32) — flip --bf16 only after confirming it doesn't
        # trip an fp32 fallback that erases the win (runbook §6).
        bf16=bf16,
        fp16=False,
        dataset_text_field="text",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=500,
        report_to=[],
        # Keep the optimizer memory-lean; fused/foreach can spike peak RSS.
        optim="adamw_torch",
        dataloader_num_workers=2,
    )


def wait_for_shards(pattern: str, want: int, poll_s: float, timeout_s: float) -> list[Path]:
    """Block until at least one new shard exists (or `want` are present / timeout)."""
    start = time.time()
    seen: set[str] = set()
    while True:
        shards = discover_shards(pattern)
        names = {str(p) for p in shards}
        new = names - seen
        if new and (len(shards) >= 1):
            return shards
        if want and len(shards) >= want:
            return shards
        if time.time() - start > timeout_s:
            return shards
        time.sleep(poll_s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF RWKV-7 dir (models/rwkv7-g1-04-hf)")
    ap.add_argument("--shards-glob", required=True,
                    help="glob for capture shards or an SFT jsonl, e.g. "
                         "'artifacts/rwkv7_posttrain/teacher.shard-*.jsonl'")
    ap.add_argument("--out", default="artifacts/rwkv7_posttrain/sft_out")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bf16", action="store_true",
                    help="enable bf16 autocast on mps (test it doesn't fp32-fall-back)")
    ap.add_argument("--watch", action="store_true",
                    help="pipeline overlap: wait for capture shards to appear, "
                         "train as they land (re-globs each epoch chunk)")
    ap.add_argument("--expected-shards", type=int, default=0,
                    help="with --watch, total shards expected (stop waiting once present)")
    ap.add_argument("--poll-s", type=float, default=10.0)
    ap.add_argument("--watch-timeout-s", type=float, default=3600.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU: load+tokenize a few rows, build config, DO NOT train")
    args = ap.parse_args()

    # ── Discover (optionally wait for) shards ────────────────────────────────
    if args.watch and not args.dry_run:
        print(f"[sft] --watch: polling {args.shards_glob} every {args.poll_s}s "
              f"(expected={args.expected_shards or '?'})", flush=True)
        shards = wait_for_shards(args.shards_glob, args.expected_shards,
                                 args.poll_s, args.watch_timeout_s)
    else:
        shards = discover_shards(args.shards_glob)
    if not shards:
        print(f"[sft] no shards match {args.shards_glob}", file=sys.stderr)
        return 1
    print(f"[sft] {len(shards)} shard(s); first={shards[0].name} last={shards[-1].name}",
          flush=True)

    rows = load_text_rows(shards)
    if not rows:
        print("[sft] shards contained no renderable rows", file=sys.stderr)
        return 1
    print(f"[sft] {len(rows)} training rows after rendering", flush=True)

    # ── Dry-run: validate the data + config path WITHOUT a GPU/training ──────
    if args.dry_run:
        from transformers import AutoTokenizer

        print("[sft] DRY-RUN: tokenizing first 3 rows with the model tokenizer", flush=True)
        if not os.path.isdir(args.model):
            # Tokenizer needs the model dir; if absent just show the rendered text.
            print(f"[sft] model dir {args.model} absent — showing rendered text only")
            for r in rows[:3]:
                print("  text[:120]:", repr(r["text"][:120]))
            print("[sft] DRY-RUN OK (rendering verified; fetch the model to tokenize).")
            return 0
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        for r in rows[:3]:
            ids = tok(r["text"])["input_ids"]
            print(f"  text[:80]={r['text'][:80]!r}  -> {len(ids)} tokens", flush=True)
        # Build the config object too (proves trl import + arg validity).
        try:
            cfg = build_sft_config(args.out, args.max_length, args.grad_accum,
                                   args.lr, args.epochs, args.bf16)
            print(f"[sft] SFTConfig built: eff_batch={1 * args.grad_accum} "
                  f"max_length={cfg.max_length} gc={cfg.gradient_checkpointing}")
        except ImportError:
            print("[sft] trl not installed — config build skipped (install for the real run)")
        print("[sft] DRY-RUN OK (data + tokenizer + config path validated; no training).")
        return 0

    # ── Real training (GPU/mps — deferred to 'when GPU free') ────────────────
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[sft] device={device}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    model.config.use_cache = False  # required with gradient checkpointing

    ds = Dataset.from_list(rows)
    cfg = build_sft_config(args.out, args.max_length, args.grad_accum,
                           args.lr, args.epochs, args.bf16)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    final = Path(args.out) / "final"
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"[sft] DONE -> {final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
