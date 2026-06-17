"""Build DPO pairs for RWKV-7 SimPO training.

Reads dpo_chosen.jsonl (gold "chosen" answers), generates "rejected" samples
from the SFT model via dismantle generate at temperature 0.7, writes dpo.jsonl
with {user, chosen, rejected} rows.

Parallel mode (default): spawns N_WORKERS subprocesses, each handling a shard
of prompts and writing to a shard file. Shards are merged at the end. Resumable:
any user prompts already in --out are skipped. The 0.4B model uses only ~15% of
GPU bandwidth per instance, so 4-6 workers run near-independently on M3 Pro.

Usage:
    python3.12 rwkv7_dpo_build_pairs.py \
        --chosen artifacts/rwkv7_posttrain/dpo_chosen.jsonl \
        --gguf models/rwkv7-g1-04-sft-Q4_K_M.gguf \
        --out artifacts/rwkv7_posttrain/dpo.jsonl \
        --workers 4 --max-new-tokens 200 --temperature 0.7
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _gen(bin_path: str, gguf: str, user: str, max_new_tokens: int,
         temperature: float, seed: int) -> str:
    prompt = f"<|rwkv_tokenizer_end_of_text|>User: {user}\n\nAssistant:"
    try:
        r = subprocess.run(
            [bin_path, "generate", "--weights", gguf, "--prompt", prompt,
             "--max-new-tokens", str(max_new_tokens),
             "--temperature", str(temperature),
             "--seed", str(seed)],
            capture_output=True, text=True, timeout=60,
        )
        out = r.stdout.strip()
        return out.split("\n\n", 1)[0].strip()[:600]
    except subprocess.TimeoutExpired:
        return ""


def _worker(args: tuple) -> None:
    """Worker: generate rejected samples for a shard of rows, write to shard file."""
    bin_path, gguf, shard_rows, shard_path, max_new_tokens, temperature, worker_id = args
    n_written = 0
    t0 = time.time()
    # Shard file is append-only; done set checked at start only (resume handled by caller)
    with open(shard_path, "a") as f:
        for global_idx, r in shard_rows:
            rejected = _gen(bin_path, gguf, r["user"], max_new_tokens, temperature, seed=global_idx)
            if not rejected:
                continue
            f.write(json.dumps({"user": r["user"], "chosen": r["chosen"], "rejected": rejected}) + "\n")
            f.flush()
            n_written += 1
            if n_written % 25 == 0 or n_written == 1:
                elapsed = time.time() - t0
                rate = n_written / elapsed if elapsed > 0 else 1
                rem = (len(shard_rows) - n_written) / rate
                print(f"  [worker {worker_id}] {n_written}/{len(shard_rows)} done, ~{rem/60:.0f}min rem",
                      flush=True)
    print(f"  [worker {worker_id}] DONE — {n_written} pairs written to {shard_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chosen", default=str(ROOT / "artifacts/rwkv7_posttrain/dpo_chosen.jsonl"))
    ap.add_argument("--gguf", default=str(ROOT / "models/rwkv7-g1-04-sft-Q4_K_M.gguf"))
    ap.add_argument("--bin", default=str(ROOT / "target/release/dismantle"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/rwkv7_posttrain/dpo.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.chosen)]
    if args.max_rows:
        rows = rows[: args.max_rows]

    # Resume: skip any user prompts already in the output file (from prior runs/shards)
    done_users: set[str] = set()
    out_path = Path(args.out)
    if out_path.exists():
        for line in out_path.open():
            try:
                done_users.add(json.loads(line)["user"])
            except Exception:
                pass
    # Also scan any live shard files from a prior aborted parallel run
    shard_paths = [Path(args.out + f".shard{i}") for i in range(args.workers)]
    for sp in shard_paths:
        if sp.exists():
            for line in sp.open():
                try:
                    done_users.add(json.loads(line)["user"])
                except Exception:
                    pass

    pending = [(i, r) for i, r in enumerate(rows) if r["user"] not in done_users]
    n_skip = len(rows) - len(pending)
    print(f"[pairs] {len(rows)} total, {n_skip} already done, {len(pending)} pending "
          f"({args.workers} workers)", flush=True)

    if not pending:
        print("[pairs] nothing to do — all pairs already built")
        _merge(shard_paths, out_path, done_from_main=done_users)
        return

    # Distribute pending rows round-robin across workers
    shards: list[list[tuple[int, dict]]] = [[] for _ in range(args.workers)]
    for j, item in enumerate(pending):
        shards[j % args.workers].append(item)

    worker_args = [
        (args.bin, args.gguf, shards[i], str(shard_paths[i]),
         args.max_new_tokens, args.temperature, i)
        for i in range(args.workers) if shards[i]
    ]

    t0 = time.time()
    with mp.Pool(len(worker_args)) as pool:
        pool.map(_worker, worker_args)

    elapsed = time.time() - t0
    print(f"[pairs] parallel generation done in {elapsed/60:.1f}min, merging shards...", flush=True)
    _merge(shard_paths, out_path, done_from_main=done_users)


def _merge(shard_paths: list[Path], out_path: Path, done_from_main: set[str]) -> None:
    """Append any new shard rows into the main output file, dedup by user."""
    existing: set[str] = set(done_from_main)
    n_merged = 0
    with out_path.open("a") as fout:
        for sp in shard_paths:
            if not sp.exists():
                continue
            for line in sp.open():
                try:
                    r = json.loads(line)
                    if r["user"] not in existing:
                        fout.write(line)
                        existing.add(r["user"])
                        n_merged += 1
                except Exception:
                    pass
            sp.unlink()  # clean up shard file
    total = len(existing)
    print(f"[pairs] merged {n_merged} new rows — {total} total in {out_path}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # safe on macOS (avoids fork+MPS issues)
    main()
