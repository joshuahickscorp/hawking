"""
eval_acceptance.py — held-out acceptance-rate evaluator for the EAGLE-3 head.

The single most important piece of validation infrastructure. Without it
you find out at training hour 40 that the loss is fine but acceptance is
12% — wasted 40 hours.

Pipeline (matches the training data path so eval is apples-to-apples):

  1. Take a held-out JSONL of N prompts that were NOT in the training set
     (seed + dataset filter ensure disjoint). Default `held_out_500.jsonl`.

  2. For each held-out prompt, run dismantle `capture-hidden` on it to
     get the EXACT same (hidden_state, prev_token, ground_truth_next_token)
     records the training pipeline saw. Cached to a held-out shard so
     re-eval doesn't re-pay the capture cost.

  3. Load the trained EAGLE head from a checkpoint .npz.

  4. For each record:
       - Get target's greedy next_token (== ground_truth from teacher
         forcing? No — teacher forcing gives the corpus token, NOT the
         target's argmax. We need the model's argmax to evaluate against,
         since spec-decode's "acceptance" is draft_token == target_argmax.)
       - To get target argmax: re-run lm_head on the captured hidden ourselves.
         Same lm_head weights are in v2lite_frozen.npz already, so this
         is just `(hidden @ lm_head).argmax(-1)`. Cheap.
       - Run draft head on (prev_token, target_hidden) to get its top-K.
       - Record matches:
           top1_match: draft top-1 == target argmax
           top3_match: target argmax in draft top-3
           top5_match: target argmax in draft top-5

  5. Aggregate per position:
       - p_accept[pos] = mean over records of top1_match[pos]
       - Overall acceptance = mean over all records (weighted by mask)

  6. Write reports/path_to_90/stage3_c2/eval_<ckpt-step>.json plus a
     plot-friendly per-position CSV.

The pass bar (architecture.md): >= 40% token+1 acceptance overall on
the held-out slice. Below that, retraining with more data is the first
move; switching architecture is the second.

Usage:

    PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

    # First-time: capture held-out hidden states (~5 min for 500 prompts)
    $PY tools/training/mlx_eagle/eval_acceptance.py prep \\
      --n 500 \\
      --seed 42 \\
      --out-jsonl tests/data/held_out_500.jsonl \\
      --out-shard training_data/c2_hidden/held_out_500.bin

    # Then evaluate against a trained checkpoint
    $PY tools/training/mlx_eagle/eval_acceptance.py eval \\
      --ckpt tools/training/mlx_eagle/ckpt/latest.npz \\
      --shard training_data/c2_hidden/held_out_500.bin \\
      --out reports/path_to_90/stage3_c2/eval_latest.json

    # CI-style: run every K minutes during training (separate shell)
    while true; do
      $PY tools/training/mlx_eagle/eval_acceptance.py eval \\
        --ckpt tools/training/mlx_eagle/ckpt/latest.npz \\
        --shard training_data/c2_hidden/held_out_500.bin \\
        --out reports/path_to_90/stage3_c2/eval_$(date +%H%M).json
      sleep 600
    done
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_BINARY = REPO_ROOT / "target/release/dismantle"
DEFAULT_WEIGHTS = REPO_ROOT / "models/deepseek-v2-lite-q4.gguf"
DEFAULT_PROFILE = REPO_ROOT / "profiles/deepseek-v2-lite-q4.m3pro18.json"
DEFAULT_FROZEN = REPO_ROOT / "tools/training/mlx_eagle/v2lite_frozen.npz"


# ---------------------------------------------------------------------------
# prep: pull N held-out samples disjoint from training set + capture
# ---------------------------------------------------------------------------
def cmd_prep(args: argparse.Namespace) -> int:
    """Generate a held-out JSONL disjoint from the training set, then
    capture hidden states for it via `dismantle capture-hidden`."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        return 2

    out_jsonl = pathlib.Path(args.out_jsonl)
    out_shard = pathlib.Path(args.out_shard)

    # Load existing training-set IDs to ensure disjoint sampling.
    train_ids = set()
    for tp in args.exclude_ids_from:
        if not pathlib.Path(tp).exists():
            print(f"WARN: --exclude-ids-from {tp} missing; skipping", file=sys.stderr)
            continue
        for line in open(tp):
            train_ids.add(json.loads(line)["id"])
    print(f"[prep] excluding {len(train_ids):,} ids already in training set",
          file=sys.stderr)

    print(f"[prep] streaming {args.dataset} split={args.split}", file=sys.stderr)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    def _extract(row):
        if "data" in row and isinstance(row["data"], list) and row["data"]:
            return row["data"][0].strip()
        if "conversations" in row and isinstance(row["conversations"], list):
            for turn in row["conversations"]:
                if isinstance(turn, dict) and "value" in turn:
                    return str(turn["value"]).strip()
        for k in ("text", "instruction", "prompt", "input"):
            if k in row and isinstance(row[k], str) and row[k].strip():
                return row[k].strip()
        return None

    import random
    rng = random.Random(args.seed)
    candidates: List[Tuple[int, str]] = []
    pool_target = max(args.n * 4, 2000)
    for i, row in enumerate(ds):
        if len(candidates) >= pool_target:
            break
        t = _extract(row)
        if not t:
            continue
        if not (args.min_chars <= len(t) <= args.max_chars):
            continue
        # Construct the same id format prep uses.
        sid = f"{args.id_prefix}_{i}_{len(candidates)}"
        if sid in train_ids:
            continue
        candidates.append((i, t))
    if len(candidates) < args.n:
        print(f"ERROR: only {len(candidates)} held-out candidates; need {args.n}",
              file=sys.stderr)
        return 2
    sampled = rng.sample(candidates, args.n)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for slot, (orig_idx, text) in enumerate(sampled):
            sid = f"heldout_{orig_idx}_{slot}"
            f.write(json.dumps({"id": sid, "text": text}, ensure_ascii=False) + "\n")
    print(f"[prep] wrote {out_jsonl} ({args.n} held-out samples)", file=sys.stderr)

    # Capture hidden states.
    binary = pathlib.Path(args.binary)
    weights = pathlib.Path(args.weights)
    profile = pathlib.Path(args.kernel_profile) if args.kernel_profile else None
    cmd = [
        str(binary), "capture-hidden",
        "--weights", str(weights),
        "--samples", str(out_jsonl),
        "--out", str(out_shard),
        "--max-tokens", str(args.max_tokens),
        "--no-lm-head",
    ]
    if profile:
        cmd += ["--kernel-profile", str(profile)]
    if args.resume:
        cmd += ["--resume"]
    print(f"[prep] running: {' '.join(cmd)}", file=sys.stderr)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[prep] capture failed exit={rc}", file=sys.stderr)
        return rc
    print(f"[prep] held-out shard ready: {out_shard}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Shard reader (matches data.py / inspect)
# ---------------------------------------------------------------------------
def _read_shard_records(shard_path: pathlib.Path, hidden_dim: int):
    """Yield (sample_id, pos, prev_tok, next_tok, hidden_np_float32)."""
    with open(shard_path, "rb") as f:
        hdr = f.read(16)
        if hdr[:4] != b"DCAP":
            raise ValueError(f"bad magic in {shard_path}")
        hd_shard = struct.unpack("<I", hdr[8:12])[0]
        if hd_shard != hidden_dim:
            raise ValueError(f"shard hidden_dim={hd_shard} != expected {hidden_dim}")
        hb_bytes = hidden_dim * 2
        while True:
            lb = f.read(2)
            if not lb:
                return
            (id_len,) = struct.unpack("<H", lb)
            sid = f.read(id_len).decode()
            pos, prev_tok, next_tok = struct.unpack("<III", f.read(12))
            hb = f.read(hb_bytes)
            yield sid, pos, prev_tok, next_tok, np.frombuffer(hb, dtype=np.float16).astype(np.float32)


# ---------------------------------------------------------------------------
# eval: load ckpt, run head on each held-out record, compute acceptance
# ---------------------------------------------------------------------------
def cmd_eval(args: argparse.Namespace) -> int:
    if mx is None:
        print("ERROR: pip install mlx", file=sys.stderr)
        return 2

    from tools.training.mlx_eagle.model import load_head_from_npz
    from tools.training.mlx_eagle.train import load_checkpoint

    # Load frozen weights + head architecture.
    print(f"[eval] loading frozen weights from {args.frozen}", file=sys.stderr)
    head = load_head_from_npz(args.frozen)
    cfg = head.cfg

    # Restore params from checkpoint. We need an optimizer to call
    # load_checkpoint, but for inference we ignore opt state.
    import mlx.nn as nn
    import mlx.optimizers as optim
    opt = optim.AdamW(learning_rate=0.0)
    # Dummy step to allocate opt state, then load over it.
    dummy_prev = mx.zeros((1, 1), dtype=mx.int32)
    dummy_hid = mx.zeros((1, 1, cfg.hidden_dim))
    dummy_next = mx.zeros((1, 1), dtype=mx.int32)
    dummy_mask = mx.ones((1, 1))
    def _dummy_loss(h, prev, hid, nxt, m):
        logits, _ = h(prev, hid, return_hidden=True)
        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), nxt.reshape(-1), reduction="mean"
        )
    gf = nn.value_and_grad(head, _dummy_loss)
    _, g = gf(head, dummy_prev, dummy_hid, dummy_next, dummy_mask)
    opt.update(head, g)
    mx.eval(head.parameters(), opt.state)
    meta = load_checkpoint(head, opt, pathlib.Path(args.ckpt))
    print(f"[eval] loaded ckpt step={meta['step']} epoch={meta['epoch']}", file=sys.stderr)

    # Reload frozen lm_head as numpy for the target-argmax pass (we don't
    # need MLX for that — just one big GEMV per record on CPU).
    npz = np.load(args.frozen, allow_pickle=True)
    lm_head_np = npz["lm_head"].astype(np.float32)  # (H, V)
    # Verify orientation matches head.__call__'s `x @ self._lm_head`.

    # Iterate shard, batch into chunks of args.batch_size, run head, compare.
    print(f"[eval] reading shard {args.shard}", file=sys.stderr)
    accept_by_pos: Dict[int, List[int]] = defaultdict(list)
    overall = {"top1": 0, "top3": 0, "top5": 0, "total": 0, "skip_bos": 0}
    skip_bos = args.skip_bos_positions
    t0 = time.time()

    # Accumulate a buffer of records and flush in batches.
    buf_prev: List[int] = []
    buf_hidden: List[np.ndarray] = []
    buf_pos: List[int] = []

    def _flush():
        if not buf_prev:
            return
        B = len(buf_prev)
        prev_arr = mx.array(np.array(buf_prev, dtype=np.int32).reshape(B, 1))
        hid_arr = mx.array(np.stack(buf_hidden, axis=0).reshape(B, 1, cfg.hidden_dim))
        logits, _ = head(prev_arr, hid_arr, return_hidden=True)
        mx.eval(logits)
        # Draft top-5 per record.
        draft_logits = np.array(logits).reshape(B, cfg.vocab_size)
        topk = np.argpartition(-draft_logits, kth=4, axis=1)[:, :5]
        # Order top-5 properly (argpartition isn't sorted).
        top5 = np.array([row[np.argsort(-draft_logits[i, row])] for i, row in enumerate(topk)])
        # Target argmax via numpy.
        hidden_np = np.stack(buf_hidden, axis=0)  # (B, H)
        tgt_logits = hidden_np @ lm_head_np  # (B, V)
        tgt_argmax = tgt_logits.argmax(axis=1)
        # Score per record.
        for i in range(B):
            pos = buf_pos[i]
            tgt = int(tgt_argmax[i])
            top1 = int(top5[i, 0])
            in3 = tgt in set(top5[i, :3].tolist())
            in5 = tgt in set(top5[i, :5].tolist())
            if pos < skip_bos:
                overall["skip_bos"] += 1
                continue
            overall["total"] += 1
            if top1 == tgt:
                overall["top1"] += 1
            if in3:
                overall["top3"] += 1
            if in5:
                overall["top5"] += 1
            accept_by_pos[pos].append(1 if top1 == tgt else 0)

    n_records = 0
    for sid, pos, prev_tok, next_tok, hidden in _read_shard_records(
        pathlib.Path(args.shard), cfg.hidden_dim
    ):
        buf_prev.append(prev_tok)
        buf_hidden.append(hidden)
        buf_pos.append(pos)
        n_records += 1
        if len(buf_prev) >= args.batch_size:
            _flush()
            buf_prev.clear()
            buf_hidden.clear()
            buf_pos.clear()
        if args.max_records > 0 and n_records >= args.max_records:
            break
    _flush()

    elapsed = time.time() - t0
    total = overall["total"] or 1
    accept_top1 = overall["top1"] / total
    accept_top3 = overall["top3"] / total
    accept_top5 = overall["top5"] / total

    # Per-position roll-up.
    pos_summary = {}
    for p, vals in sorted(accept_by_pos.items()):
        pos_summary[str(p)] = {
            "n": len(vals),
            "accept_top1": sum(vals) / len(vals),
        }

    result = {
        "ckpt": str(args.ckpt),
        "shard": str(args.shard),
        "step": meta["step"],
        "epoch": meta["epoch"],
        "n_records_scored": overall["total"],
        "n_records_skipped_bos": overall["skip_bos"],
        "skip_bos_positions": skip_bos,
        "accept_top1": accept_top1,
        "accept_top3": accept_top3,
        "accept_top5": accept_top5,
        "elapsed_s": elapsed,
        "records_per_sec": n_records / elapsed,
        "pass_bar_top1_0.40": "PASS" if accept_top1 >= 0.40 else "FAIL",
        "pass_bar_top1_0.50": "PASS" if accept_top1 >= 0.50 else "FAIL",
        "pass_bar_top1_0.70": "PASS" if accept_top1 >= 0.70 else "FAIL",
        "per_position": pos_summary,
    }

    out_path = pathlib.Path(args.out) if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[eval] wrote {out_path}", file=sys.stderr)

    # Print headline to stdout for CI grep.
    print(f"\n=== EAGLE-3 head acceptance (ckpt step {meta['step']}) ===")
    print(f"  records scored      : {overall['total']:,} (skipped BOS: {overall['skip_bos']:,})")
    print(f"  accept top-1        : {accept_top1*100:6.2f}%   {result['pass_bar_top1_0.40']} @0.40  {result['pass_bar_top1_0.50']} @0.50  {result['pass_bar_top1_0.70']} @0.70")
    print(f"  accept top-3        : {accept_top3*100:6.2f}%")
    print(f"  accept top-5        : {accept_top5*100:6.2f}%")
    print(f"  eval throughput     : {n_records/elapsed:.0f} records/sec")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prep", help="Build held-out JSONL + capture hidden")
    pp.add_argument("--n", type=int, default=500)
    pp.add_argument("--seed", type=int, default=42)
    pp.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    pp.add_argument("--split", default="train_sft")
    pp.add_argument("--id-prefix", default="ultrachat")
    pp.add_argument("--min-chars", type=int, default=200)
    pp.add_argument("--max-chars", type=int, default=2000)
    pp.add_argument("--exclude-ids-from", nargs="*",
                    default=["tests/data/ultrachat_55k_union.jsonl"],
                    help="JSONL files whose ids should NOT appear in the held-out set.")
    pp.add_argument("--out-jsonl", default=str(REPO_ROOT / "tests/data/held_out_500.jsonl"))
    pp.add_argument("--out-shard", default=str(REPO_ROOT / "training_data/c2_hidden/held_out_500.bin"))
    pp.add_argument("--max-tokens", type=int, default=128)
    pp.add_argument("--binary", default=str(DEFAULT_BINARY))
    pp.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    pp.add_argument("--kernel-profile", default=str(DEFAULT_PROFILE))
    pp.add_argument("--resume", action="store_true")
    pp.set_defaults(func=cmd_prep)

    pe = sub.add_parser("eval", help="Run head against held-out shard")
    pe.add_argument("--ckpt", required=True, help="Path to trained head .npz")
    pe.add_argument("--shard", required=True, help="Held-out hidden-state .bin")
    pe.add_argument("--frozen", default=str(DEFAULT_FROZEN))
    pe.add_argument("--out", default=None, help="Write JSON result here.")
    pe.add_argument("--batch-size", type=int, default=128,
                    help="Forward-pass batch size for the head. Large is fine; "
                         "head is small and inference-only.")
    pe.add_argument("--skip-bos-positions", type=int, default=3,
                    help="Drop positions 0..N-1 from scoring (matches training mask).")
    pe.add_argument("--max-records", type=int, default=0,
                    help="Cap records evaluated (0=all). 50000 is sufficient for "
                         "tight acceptance estimates; saves wall on huge shards.")
    pe.set_defaults(func=cmd_eval)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
