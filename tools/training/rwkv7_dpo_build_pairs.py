"""Build DPO pairs for RWKV-7 SimPO training.

Reads dpo_chosen.jsonl (gold "chosen" answers), generates "rejected" samples
from the SFT model via dismantle generate at temperature 0.7, writes dpo.jsonl
with {user, chosen, rejected} rows.

Usage:
    python3.12 rwkv7_dpo_build_pairs.py \
        --chosen artifacts/rwkv7_posttrain/dpo_chosen.jsonl \
        --gguf models/rwkv7-g1-04-sft-Q4_K_M.gguf \
        --out artifacts/rwkv7_posttrain/dpo.jsonl \
        --max-new-tokens 200 --temperature 0.7
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def gen_rejected(bin_path: str, gguf: str, user: str, max_new_tokens: int, temperature: float, seed: int) -> str:
    prompt = f"<|rwkv_tokenizer_end_of_text|>User: {user}\n\nAssistant:"
    r = subprocess.run(
        [bin_path, "generate", "--weights", gguf, "--prompt", prompt,
         "--max-new-tokens", str(max_new_tokens), "--temperature", str(temperature),
         "--seed", str(seed)],
        capture_output=True, text=True, timeout=60,
    )
    out = r.stdout.strip()
    # trim at the first blank-line turn break (same as coherence measure)
    return out.split("\n\n", 1)[0].strip()[:600]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chosen", default=str(ROOT / "artifacts/rwkv7_posttrain/dpo_chosen.jsonl"))
    ap.add_argument("--gguf", default=str(ROOT / "models/rwkv7-g1-04-sft-Q4_K_M.gguf"))
    ap.add_argument("--bin", default=str(ROOT / "target/release/dismantle"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/rwkv7_posttrain/dpo.jsonl"))
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    ap.add_argument("--resume", action="store_true", help="skip already-written rows")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.chosen)]
    if args.max_rows:
        rows = rows[: args.max_rows]

    # resume: skip rows already in output
    done_users: set[str] = set()
    if args.resume and Path(args.out).exists():
        for l in open(args.out):
            try:
                done_users.add(json.loads(l)["user"])
            except Exception:
                pass
        print(f"[resume] {len(done_users)} already done, skipping")

    n_written = 0
    n_skip = 0
    t0 = time.time()
    with open(args.out, "a") as fout:
        for i, r in enumerate(rows):
            user = r["user"]
            chosen = r["chosen"]
            if user in done_users:
                n_skip += 1
                continue
            try:
                rejected = gen_rejected(args.bin, args.gguf, user, args.max_new_tokens,
                                        args.temperature, seed=i)
            except subprocess.TimeoutExpired:
                print(f"  [warn] timeout on row {i}, skipping", flush=True)
                continue
            if not rejected:
                print(f"  [warn] empty rejected on row {i}, skipping", flush=True)
                continue
            fout.write(json.dumps({"user": user, "chosen": chosen, "rejected": rejected}) + "\n")
            fout.flush()
            n_written += 1
            if n_written % 50 == 0 or i < 5:
                elapsed = time.time() - t0
                remaining = (len(rows) - i - 1) * (elapsed / max(n_written, 1))
                print(f"  [{i+1}/{len(rows)}] {n_written} written, "
                      f"~{remaining/60:.0f}min remaining — {user[:50]!r}", flush=True)

    total = time.time() - t0
    print(f"[done] {n_written} pairs written to {args.out} ({total/60:.1f}min)")


if __name__ == "__main__":
    main()
