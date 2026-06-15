#!/usr/bin/env python3
"""Hard-negative miner for Eagle5 dense (Qwen) training corpora.

Loads a trained Eagle5 head + frozen weights + an existing parquet corpus,
scores each row by how badly the head fails to match the verifier's argmax
at the captured residual, then writes a NEW parquet directory containing
only the top-N hardest rows (whole rows; schema preserved exactly so the
existing trainer can consume the output without changes).

Scoring modes
-------------
* ``depth1_miss`` (default): per-row mean of ``1.0 - I[head_argmax == verifier_argmax]``
  evaluated at every captured position. Cheap, deterministic, and aligns
  with the depth-1 acceptance gate the runtime cares about.
* ``rollout_ce``: per-row mean cross-entropy of the head's draft logits
  against the verifier argmax over a short autoregressive rollout of the
  head fed by its own draft tokens. Higher cost; useful when you already
  have decent depth-1 acceptance and want to surface multi-step failures.

Output
------
``<out_dir>/shard_*.parquet`` plus a sidecar ``mine_manifest.json`` describing
the head used, the score distribution, and which input shards were sampled.
The manifest is what the notebook diffs to decide if a mine is reusable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - mirrors trainer behavior
    print("ERROR: torch not installed. `pip install torch`.", file=sys.stderr)
    sys.exit(1)

from eagle5_train_pytorch import (  # noqa: E402
    N_HEADS,
    RMS_EPS,
    _extract_row,
    _rms_norm,
    build_head,
)
from eagle5_tau_eval_pytorch import _load_head  # noqa: E402


def _list_shards(corpus_dir: Path) -> list[Path]:
    shards = sorted(corpus_dir.glob("shard_*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards found in {corpus_dir}")
    return shards


def _row_to_dict(table: pa.Table, i: int) -> dict:
    return {c: table[c][i].as_py() for c in table.column_names}


def _row_to_tensors(
    row: dict,
    max_row_tokens: int,
    device: str,
) -> Optional[dict[str, torch.Tensor]]:
    ex = _extract_row(row, max_row_tokens=max_row_tokens)
    if ex is None:
        return None
    prev = torch.from_numpy(ex["prev_tokens"].astype(np.int64)).to(device).unsqueeze(0)
    nxt = torch.from_numpy(ex["next_tokens"].astype(np.int64)).to(device).unsqueeze(0)
    res = torch.from_numpy(ex["residual"].astype(np.float32)).to(device).unsqueeze(0)
    inter = torch.from_numpy(ex["intermediate"].astype(np.float32)).to(device).unsqueeze(0)
    return {"prev": prev, "next": nxt, "residual": res, "intermediate": inter, "n": int(prev.shape[1])}


@torch.inference_mode()
def _score_row_depth1(head, lm_head_f: torch.Tensor, row_t: dict) -> float:
    """Per-row hardness as 1 - mean(head_argmax == verifier_argmax)."""
    token_logits, _s, _draft_h, _calib = head(
        row_t["prev"], row_t["residual"], row_t["intermediate"]
    )
    head_arg = token_logits[0].float().argmax(dim=-1)  # (S,)
    baseline = _rms_norm(row_t["residual"], head._output_norm, RMS_EPS)
    verifier_logits = torch.matmul(baseline[0].float(), lm_head_f)
    verifier_arg = verifier_logits.argmax(dim=-1)  # (S,)
    match = (head_arg == verifier_arg).float().mean().item()
    return float(1.0 - match)


@torch.inference_mode()
def _score_row_rollout(head, lm_head_f: torch.Tensor, row_t: dict, depth: int) -> float:
    """Per-row hardness as mean CE of head-from-own-drafts vs verifier argmax.

    Walks ``depth`` steps starting from offsets every ``depth`` tokens. At
    each rollout step the head sees its previously-drafted token, the next
    captured residual, and the next captured intermediate. The label is the
    verifier's argmax at that position. The score is averaged over all
    rollout starts in the row, then again over the row's lengths.
    """
    S = row_t["n"]
    if S < depth + 1:
        return _score_row_depth1(head, lm_head_f, row_t)
    losses: list[float] = []
    prev = row_t["prev"][0]
    residual = row_t["residual"][0]
    intermediate = row_t["intermediate"][0]
    baseline_all = _rms_norm(residual.unsqueeze(0), head._output_norm, RMS_EPS)[0].float()
    verifier_arg_all = torch.matmul(baseline_all, lm_head_f).argmax(dim=-1)
    for start in range(0, S - depth, max(1, depth)):
        cur_prev = prev[start:start + 1].view(1, 1)
        for d in range(depth):
            pos = start + d
            if pos >= S:
                break
            res_d = residual[pos:pos + 1, :].view(1, 1, -1)
            inter_d = intermediate[pos:pos + 1, :].view(1, 1, -1)
            token_logits, _s, _draft_h, _calib = head(cur_prev, res_d, inter_d)
            log_probs = F.log_softmax(token_logits[:, 0, :].float(), dim=-1)
            target = verifier_arg_all[pos]
            losses.append(float(-log_probs[0, target].item()))
            cur_prev = token_logits[:, 0, :].float().argmax(dim=-1).view(1, 1)
    if not losses:
        return _score_row_depth1(head, lm_head_f, row_t)
    return float(sum(losses) / len(losses))


def _score_shard(
    shard: Path,
    head,
    lm_head_f: torch.Tensor,
    device: str,
    *,
    max_row_tokens: int,
    score_kind: str,
    rollout_depth: int,
) -> list[dict]:
    """Return [{row_index, score, n_tokens}] for every usable row in the shard."""
    table = pq.read_table(shard)
    results: list[dict] = []
    for i in range(table.num_rows):
        row = _row_to_dict(table, i)
        row_t = _row_to_tensors(row, max_row_tokens=max_row_tokens, device=device)
        if row_t is None:
            continue
        if score_kind == "rollout_ce":
            score = _score_row_rollout(head, lm_head_f, row_t, rollout_depth)
        else:
            score = _score_row_depth1(head, lm_head_f, row_t)
        results.append({"row": i, "score": score, "n": row_t["n"]})
    return results


def _write_filtered_shards(
    src_shards: list[Path],
    keep_by_shard: dict[Path, list[int]],
    out_dir: Path,
    rows_per_shard: int,
) -> dict:
    """Re-emit chosen rows into shard_*.parquet under out_dir.

    Preserves the input schema exactly by slicing the source tables. Output
    shards are packed to ~`rows_per_shard` rows each so the trainer's shard
    sampler keeps roughly the same shape as before.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer: list[pa.Table] = []
    buffered_rows = 0
    shard_idx = 0
    schema_ref: Optional[pa.Schema] = None
    written = 0

    def _flush() -> None:
        nonlocal buffer, buffered_rows, shard_idx
        if not buffer:
            return
        merged = pa.concat_tables(buffer, promote=False)
        out_path = out_dir / f"shard_{shard_idx:05d}.parquet"
        pq.write_table(merged, out_path, compression="zstd")
        shard_idx += 1
        buffer = []
        buffered_rows = 0

    for shard in src_shards:
        rows = keep_by_shard.get(shard) or []
        if not rows:
            continue
        table = pq.read_table(shard)
        if schema_ref is None:
            schema_ref = table.schema
        elif not schema_ref.equals(table.schema):
            # Different schemas would corrupt concatenation; skip safely.
            print(f"[mine] WARN skipping shard with mismatched schema: {shard}")
            continue
        idx_array = pa.array(sorted(set(int(r) for r in rows)), type=pa.int64())
        sliced = table.take(idx_array)
        if sliced.num_rows == 0:
            continue
        buffer.append(sliced)
        buffered_rows += sliced.num_rows
        written += sliced.num_rows
        if buffered_rows >= rows_per_shard:
            _flush()
    _flush()
    return {"rows_written": written, "shards_written": shard_idx, "schema": str(schema_ref)}


def main() -> None:
    ap = argparse.ArgumentParser(prog="eagle5_hard_neg_miner")
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="head checkpoint (safetensors or latest.npz)")
    ap.add_argument("--frozen", required=True, type=Path,
                    help="frozen .npz for the same model the head was trained on")
    ap.add_argument("--corpus-dir", required=True, type=Path,
                    help="existing parquet corpus to mine from")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="destination directory for shard_*.parquet")
    ap.add_argument("--keep-fraction", type=float, default=0.25,
                    help="fraction of input rows to retain as hard examples (0-1)")
    ap.add_argument("--keep-min-rows", type=int, default=2000,
                    help="floor on number of retained rows even if keep-fraction is tiny")
    ap.add_argument("--keep-max-rows", type=int, default=12000,
                    help="hard ceiling on retained rows so a mine fits in Drive")
    ap.add_argument("--shards-to-scan", type=int, default=0,
                    help="if >0, sample this many input shards instead of scanning all")
    ap.add_argument("--rows-per-output-shard", type=int, default=200,
                    help="approximate row count per emitted shard_*.parquet")
    ap.add_argument("--max-row-tokens", type=int, default=384,
                    help="row truncation used when scoring; should match trainer setting")
    ap.add_argument("--score", choices=["depth1_miss", "rollout_ce"], default="depth1_miss")
    ap.add_argument("--rollout-depth", type=int, default=4)
    ap.add_argument("--num-blocks", type=int, default=1)
    ap.add_argument("--head-heads", type=int, default=N_HEADS)
    ap.add_argument("--head-ff-mult", type=float, default=4.0)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not 0.0 < args.keep_fraction <= 1.0:
        raise SystemExit("--keep-fraction must be in (0, 1]")
    if args.keep_min_rows > args.keep_max_rows:
        raise SystemExit("--keep-min-rows cannot exceed --keep-max-rows")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[mine] WARN: CUDA unavailable; falling back to CPU", flush=True)
        device = "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    head = _load_head(
        args.ckpt,
        args.frozen,
        device,
        num_blocks=args.num_blocks,
        n_heads=args.head_heads,
        ff_mult=args.head_ff_mult,
    ).eval()
    lm_head_f = head._lm_head.float()

    shards = _list_shards(args.corpus_dir)
    rng = random.Random(args.seed)
    if args.shards_to_scan > 0 and args.shards_to_scan < len(shards):
        shards = rng.sample(shards, args.shards_to_scan)
    print(f"[mine] scanning {len(shards)} shards from {args.corpus_dir}", flush=True)

    t0 = time.time()
    per_shard_scores: dict[Path, list[dict]] = {}
    scored_rows = 0
    for shard in shards:
        results = _score_shard(
            shard,
            head,
            lm_head_f,
            device,
            max_row_tokens=args.max_row_tokens,
            score_kind=args.score,
            rollout_depth=args.rollout_depth,
        )
        per_shard_scores[shard] = results
        scored_rows += len(results)
        if len(per_shard_scores) % 32 == 0:
            elapsed = time.time() - t0
            rate = scored_rows / max(elapsed, 1e-6)
            print(
                f"[mine] scored {len(per_shard_scores)}/{len(shards)} shards, "
                f"{scored_rows} rows, {rate:.1f} rows/s",
                flush=True,
            )

    if scored_rows == 0:
        raise SystemExit("no usable rows scored; check corpus/head/frozen combo")

    flat: list[tuple[Path, int, float]] = []
    for shard, results in per_shard_scores.items():
        for r in results:
            flat.append((shard, int(r["row"]), float(r["score"])))
    flat.sort(key=lambda t: t[2], reverse=True)

    target = int(math.ceil(scored_rows * args.keep_fraction))
    target = max(args.keep_min_rows, min(args.keep_max_rows, target))
    target = min(target, len(flat))
    keep = flat[:target]
    cutoff_score = keep[-1][2] if keep else float("inf")

    keep_by_shard: dict[Path, list[int]] = {}
    for shard, row_idx, _score in keep:
        keep_by_shard.setdefault(shard, []).append(row_idx)

    write_stats = _write_filtered_shards(
        list(per_shard_scores.keys()),
        keep_by_shard,
        args.out_dir,
        rows_per_shard=max(50, int(args.rows_per_output_shard)),
    )

    scores_only = [s for _sh, _r, s in flat]
    score_summary = {
        "n_scored": scored_rows,
        "n_kept": len(keep),
        "cutoff_score": cutoff_score,
        "score_min": float(min(scores_only)),
        "score_max": float(max(scores_only)),
        "score_mean": float(sum(scores_only) / len(scores_only)),
        "score_p50": float(sorted(scores_only)[len(scores_only) // 2]),
        "score_p90": float(sorted(scores_only)[int(len(scores_only) * 0.9)]),
        "score_p99": float(sorted(scores_only)[int(len(scores_only) * 0.99)]),
    }
    manifest = {
        "schema": "dismantle-eagle5-hard-neg-mine-v1",
        "created_at_unix": int(time.time()),
        "ckpt": str(args.ckpt),
        "frozen": str(args.frozen),
        "corpus_dir": str(args.corpus_dir),
        "out_dir": str(args.out_dir),
        "score_kind": args.score,
        "rollout_depth": args.rollout_depth,
        "keep_fraction": args.keep_fraction,
        "keep_min_rows": args.keep_min_rows,
        "keep_max_rows": args.keep_max_rows,
        "shards_scanned": [str(p) for p in per_shard_scores.keys()],
        "max_row_tokens": args.max_row_tokens,
        "score_summary": score_summary,
        "write_stats": write_stats,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "mine_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"[mine] kept {len(keep)}/{scored_rows} rows "
        f"(cutoff={cutoff_score:.4f}, mean={score_summary['score_mean']:.4f}) "
        f"→ {write_stats['shards_written']} shards, "
        f"{write_stats['rows_written']} rows in {args.out_dir}",
        flush=True,
    )
    print(f"[mine] manifest written: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
