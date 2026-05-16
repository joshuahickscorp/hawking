"""
spec_decode_stub.py — end-to-end draft+verify integration stub.

PURPOSE: validate the spec-decode wins arithmetic against the REAL trained
head + the REAL captured target distribution, BEFORE writing the C3
Rust wire-up. Catches misalignments (tokenizer, KV layout, sampling) at
session-scope cost so they don't bite the multi-week C3 effort.

What this stub does (no Rust changes, runs entirely in Python/MLX):

  1. Load trained EagleHead checkpoint via MLX.
  2. Load held-out shard's (sample_id, pos, prev_token, target_hidden,
     ground_truth_next) records — same data eval_acceptance.py uses.
  3. For each held-out sample (which is a SEQUENCE of records ordered by
     pos), simulate K-step speculative decoding:
       a. Treat the sample's records as the ground-truth target trajectory.
       b. At each "spec step", starting from the last accepted position p:
           - Run head on (records[p].prev_token, records[p].target_hidden)
             to get its top-1 next-token prediction `draft[0]`.
           - To get `draft[1]`: we need a hidden state at position p+1, but
             we only HAVE it from the captured corpus if record p+1 exists.
             Use it (this is the stub limitation — real spec-decode would
             have the draft head produce its own forward hidden; here we
             use the corpus's target hidden as a proxy).
           - Continue for K-1 draft proposals.
       c. Verify: compare each draft[i] against the target's argmax-at-
          pos-p+i+1 (computed via frozen lm_head on the corpus hidden).
          Accept the longest matching prefix.
       d. Record: accepted count, draft cost, equivalent target forwards saved.

  4. Compute the win metric:
       tokens_per_target_forward = (1 + mean_accepts) / K
       The current verify cost ≈ K × single-forward, so this number > 1.0
       means spec-decode is a win, < 1.0 means a regression.

  5. Also compute a "Path B" projection: if verify cost dropped to ~1.5×
     single-forward (per stage3_spec/audit.md), what would speedup look like?

Stub limitations (called out so we don't pretend this is the full system):

  - **Draft head consumes corpus target hidden, not its own multi-step
    hidden.** Real spec-decode at K>1 has the head produce its own
    rolled-forward hidden state; here we use the captured corpus hidden
    as a proxy. This OVER-estimates acceptance because the draft never
    has to deal with its own prediction errors compounding.

  - **No real wall-clock measurement of the target forward.** We assume
    K=4 verify cost ≈ 4× single-forward per the current `forward_tokens_
    batched_for_test` impl. Use eval_acceptance.py + real benchmarks
    in C3 to get true wall-clock.

  - **No actual KV management.** The held-out capture already advanced
    its KV correctly during teacher-forced capture; we just replay it.

  - **K is fixed across the stub run.** Real spec-decode could adaptively
    choose K based on streak length; not modeled here.

The stub is intentionally pessimistic in some ways (single-token-at-a-time
draft generation overhead, no KV reuse for the draft head) and optimistic
in others (corpus-hidden draft input). The headline number it produces
("tokens_per_target_forward at K=4") is a useful estimate but should be
verified against the real Rust C3 wire-up before any perf claims ship.

Usage:

    PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
    $PY tools/training/mlx_eagle/spec_decode_stub.py \\
      --ckpt tools/training/mlx_eagle/ckpt/latest.npz \\
      --shard training_data/c2_hidden/held_out_500.bin \\
      --k 4 \\
      --max-samples 100 \\
      --out reports/path_to_90/stage3_c2/spec_stub_k4.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_FROZEN = REPO_ROOT / "tools/training/mlx_eagle/v2lite_frozen.npz"


def _group_records_by_sample(
    shard_path: pathlib.Path,
    hidden_dim: int,
    max_samples: int = 0,
):
    """Return dict[sample_id] -> list of (pos, prev_tok, next_tok, hidden) sorted by pos.

    If max_samples > 0, stops scanning once a new sample_id would push the
    distinct-sample count past `max_samples + 1` (1 extra to ensure the
    last admitted sample is complete). Saves wall on huge shards.
    """
    by_sample: Dict[str, List[Tuple[int, int, int, np.ndarray]]] = defaultdict(list)
    hb_bytes = hidden_dim * 2
    cap = max_samples + 1 if max_samples > 0 else None
    with open(shard_path, "rb") as f:
        hdr = f.read(16)
        assert hdr[:4] == b"DCAP"
        hd = struct.unpack("<I", hdr[8:12])[0]
        assert hd == hidden_dim, f"shard hidden_dim={hd} != expected {hidden_dim}"
        while True:
            lb = f.read(2)
            if not lb:
                break
            (id_len,) = struct.unpack("<H", lb)
            sid = f.read(id_len).decode()
            pos, prev_tok, next_tok = struct.unpack("<III", f.read(12))
            # Early stop: if we've already seen `cap` distinct ids and this
            # one is new, we've collected enough — skip the rest.
            if cap is not None and sid not in by_sample and len(by_sample) >= cap:
                break
            hb = f.read(hb_bytes)
            arr = np.frombuffer(hb, dtype=np.float16).astype(np.float32)
            by_sample[sid].append((pos, prev_tok, next_tok, arr))
    # Sort each sample by position.
    for sid in by_sample:
        by_sample[sid].sort(key=lambda r: r[0])
    return by_sample


def _target_argmax(hidden_np: np.ndarray, lm_head_np: np.ndarray) -> int:
    """argmax of `hidden @ lm_head` — what the target model would greedily pick."""
    return int((hidden_np @ lm_head_np).argmax())


def simulate(args: argparse.Namespace) -> int:
    if mx is None:
        print("ERROR: pip install mlx", file=sys.stderr)
        return 2

    from tools.training.mlx_eagle.model import load_head_from_npz
    from tools.training.mlx_eagle.train import load_checkpoint
    import mlx.nn as nn
    import mlx.optimizers as optim

    # Load frozen weights + head architecture.
    print(f"[stub] loading frozen weights from {args.frozen}", file=sys.stderr)
    head = load_head_from_npz(args.frozen)
    cfg = head.cfg

    # Resume checkpoint via the same dummy-step pattern as eval_acceptance.
    opt = optim.AdamW(learning_rate=0.0)
    dummy = lambda: (
        mx.zeros((1, 1), dtype=mx.int32),
        mx.zeros((1, 1, cfg.hidden_dim)),
        mx.zeros((1, 1), dtype=mx.int32),
        mx.ones((1, 1)),
    )
    def _loss_fn(h, prev, hid, nxt, m):
        logits, _ = h(prev, hid, return_hidden=True)
        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), nxt.reshape(-1), reduction="mean"
        )
    grad_fn = nn.value_and_grad(head, _loss_fn)
    _l, _g = grad_fn(head, *dummy())
    opt.update(head, _g)
    mx.eval(head.parameters(), opt.state)
    ckmeta = load_checkpoint(head, opt, pathlib.Path(args.ckpt))
    print(f"[stub] loaded ckpt step={ckmeta['step']} epoch={ckmeta['epoch']}",
          file=sys.stderr)

    # Frozen lm_head as numpy for target-argmax computation.
    npz = np.load(args.frozen, allow_pickle=True)
    lm_head_np = npz["lm_head"].astype(np.float32)

    # Load held-out records grouped by sample. Early-stop reading when we
    # have args.max_samples + 1 distinct sample_ids — avoids scanning the
    # tail of huge shards.
    print(f"[stub] reading shard {args.shard} (cap: {args.max_samples} samples)",
          file=sys.stderr)
    by_sample = _group_records_by_sample(
        pathlib.Path(args.shard), cfg.hidden_dim,
        max_samples=args.max_samples,
    )
    print(f"[stub] {len(by_sample)} samples loaded", file=sys.stderr)

    samples = sorted(by_sample.keys())
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    K = args.k

    # ---- Simulate spec decode per sample ----
    total_accepts: List[int] = []  # one accept-count per spec step
    draft_wall_total = 0.0
    total_target_forwards_baseline = 0  # = total decode tokens (1 per token)
    total_target_forwards_spec = 0      # 1 per spec step regardless of K
    total_tokens_emitted = 0

    head_dtype = mx.float32  # head is small; fp32 inference is fine
    t_start = time.time()

    for s_idx, sid in enumerate(samples):
        recs = by_sample[sid]
        if len(recs) < K + 1:
            continue  # too short to evaluate K-step spec decode

        # Skip BOS positions per training convention.
        start_pos = args.skip_bos_positions
        # Walk the sample as a decode trajectory.
        p = start_pos
        while p + K < len(recs):
            # Build a (K,) batch of draft head queries.
            # Each query: prev = recs[p+i].prev_token, hidden = recs[p+i].target_hidden
            # The stub uses corpus target_hidden as a proxy for what the head
            # would produce on its own forward; this OVER-ESTIMATES acceptance.
            prevs = np.array([recs[p + i][1] for i in range(K)], dtype=np.int32)
            hids = np.stack([recs[p + i][3] for i in range(K)])  # (K, H)
            prev_arr = mx.array(prevs.reshape(K, 1))
            hid_arr = mx.array(hids.reshape(K, 1, cfg.hidden_dim))
            t_draft0 = time.time()
            logits, _ = head(prev_arr, hid_arr, return_hidden=True)
            mx.eval(logits)
            draft_wall_total += time.time() - t_draft0
            draft_argmax = np.array(logits).reshape(K, cfg.vocab_size).argmax(axis=1)

            # Target argmax at the K verify positions (would be lm_head(target_hidden)).
            # The "next" position is p+i+1 because verify produces logits FOR p+i+1.
            # Records at pos p have hidden that PREDICTS the token at pos p+1.
            # So verify_argmax[i] = argmax(recs[p+i].hidden @ lm_head).
            tgt_argmax = np.array([
                _target_argmax(recs[p + i][3], lm_head_np) for i in range(K)
            ])

            # Longest matching prefix.
            accepted = 0
            for i in range(K):
                if int(draft_argmax[i]) == int(tgt_argmax[i]):
                    accepted += 1
                else:
                    break

            total_accepts.append(accepted)
            total_target_forwards_baseline += accepted + 1
            total_target_forwards_spec += 1
            total_tokens_emitted += accepted + 1
            p += accepted + 1

        if (s_idx + 1) % 20 == 0:
            mean_a = np.mean(total_accepts) if total_accepts else 0.0
            print(
                f"  [{s_idx+1}/{len(samples)}] mean_accepts={mean_a:.3f} "
                f"steps={len(total_accepts):,}",
                file=sys.stderr,
            )

    elapsed = time.time() - t_start

    # ---- Headline metrics ----
    n_steps = len(total_accepts)
    if n_steps == 0:
        print("[stub] ERROR: no eligible samples (too short?)", file=sys.stderr)
        return 1
    mean_accepts = float(np.mean(total_accepts))
    accept_dist = {str(k): int(sum(1 for a in total_accepts if a == k)) for k in range(K + 1)}

    # tokens_per_target_forward under current verify cost (K× single-forward).
    # In wall-clock terms: per spec step, target costs K × t_fwd, draft costs t_draft.
    # Token output per step = 1 + mean_accepts.
    # So tokens/sec_with_spec = (1 + mean_a) / (K * t_fwd + t_draft) -- vs baseline (1 / t_fwd).
    # Without losing the t_fwd term, the dimensionless headline is:
    tokens_per_target_forward = (1 + mean_accepts) / K  # ignores draft cost
    speedup_K_verify = tokens_per_target_forward  # vs 1.0 baseline

    # "Path B" projection: if verify cost dropped to ~1.5x single-forward
    # (per stage3_spec/audit.md), what would speedup be?
    tokens_per_target_forward_pathB = (1 + mean_accepts) / 1.5
    speedup_pathB = tokens_per_target_forward_pathB

    # Draft cost per accept: total_draft_wall / tokens emitted.
    draft_ms_per_token = (draft_wall_total / total_tokens_emitted) * 1000 if total_tokens_emitted else 0.0

    result = {
        "ckpt": str(args.ckpt),
        "shard": str(args.shard),
        "step": ckmeta["step"],
        "K": K,
        "skip_bos_positions": args.skip_bos_positions,
        "n_samples_evaluated": len(samples),
        "n_spec_steps": n_steps,
        "n_tokens_emitted": total_tokens_emitted,
        "mean_accepts_per_step": mean_accepts,
        "accept_distribution": accept_dist,
        "draft_ms_per_token": draft_ms_per_token,
        "eval_wall_s": elapsed,
        "headline_metrics": {
            "tokens_per_target_forward_K_verify": tokens_per_target_forward,
            "speedup_vs_no_spec_K_verify": speedup_K_verify,
            "tokens_per_target_forward_pathB_verify": tokens_per_target_forward_pathB,
            "speedup_vs_no_spec_pathB_verify": speedup_pathB,
        },
        "stub_limitations_acknowledged": [
            "draft head uses corpus target_hidden, not its own multi-step hidden -- over-estimates acceptance",
            "no real wall-clock for target forward; assumed K x single-forward (current Rust verify cost)",
            "K is fixed, no adaptive K based on streak length",
            "no KV reuse for the draft head -- under-estimates draft throughput",
        ],
    }

    out_path = pathlib.Path(args.out) if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[stub] wrote {out_path}", file=sys.stderr)

    print(f"\n=== spec-decode stub (ckpt step {ckmeta['step']}, K={K}) ===")
    print(f"  samples evaluated   : {len(samples)}")
    print(f"  spec steps          : {n_steps:,}")
    print(f"  tokens emitted      : {total_tokens_emitted:,}")
    print(f"  mean accepts/step   : {mean_accepts:.3f}  (max possible: {K})")
    print(f"  accept distribution : {accept_dist}")
    print()
    print(f"  CURRENT verify cost (K x single-forward, per audit.md):")
    print(f"    tokens/target_forward = {tokens_per_target_forward:.3f}")
    print(f"    speedup vs no-spec    = {speedup_K_verify:.3f}x   {'WIN' if speedup_K_verify > 1.0 else 'REGRESSION'}")
    print()
    print(f"  PROJECTED Path B verify (~1.5x single-forward):")
    print(f"    tokens/target_forward = {tokens_per_target_forward_pathB:.3f}")
    print(f"    speedup vs no-spec    = {speedup_pathB:.3f}x")
    print()
    print(f"  draft cost (head fwd) : {draft_ms_per_token:.2f} ms/token emitted")
    print(f"  eval wall             : {elapsed:.1f}s")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--ckpt", required=True, help="Trained head .npz")
    p.add_argument("--shard", required=True, help="Held-out hidden-state .bin")
    p.add_argument("--frozen", default=str(DEFAULT_FROZEN))
    p.add_argument("--k", type=int, default=4, help="Spec-decode verify window size")
    p.add_argument("--max-samples", type=int, default=100,
                   help="Cap eval samples (0 = all in shard)")
    p.add_argument("--skip-bos-positions", type=int, default=3)
    p.add_argument("--out", default=None, help="Write JSON result here.")
    args = p.parse_args()
    return simulate(args)


if __name__ == "__main__":
    sys.exit(main())
