"""STaR (Self-Taught Reasoner) self-improvement loop for RWKV-7.

Each round:
  1. Generate N_SAMPLES completions per prompt (via dismantle binary, temperature sweep)
  2. Score each completion: distinct-2 + length normalisation + ROUGE-L vs gold
  3. Keep the best-scoring completion per prompt (drop if below QUALITY_FLOOR)
  4. Append winners to the running SFT corpus and run one SFT epoch
  5. Eval wikitext-2 PPL and log to round events.jsonl

Rounds are resumable: if <out>/round_N/events.jsonl already has a final_ppl entry,
that round is skipped on re-run.

Usage:
    python3 rwkv7_star_loop.py \
        --gguf models/rwkv7-g1-04-sft-Q4_K_M.gguf \
        --model artifacts/rwkv7_posttrain/sft_out/final/state_dict.pt \
        --hf-dir models/rwkv7-g1-04-hf \
        --corpus artifacts/rwkv7_posttrain/sft.jsonl \
        --out artifacts/rwkv7_posttrain/star \
        --rounds 3 --n-samples 3 --workers 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Quality scoring (reference-free + reference-aware)
# ---------------------------------------------------------------------------

def _bigrams(words: list[str]) -> list[tuple[str, str]]:
    return [(words[i], words[i + 1]) for i in range(len(words) - 1)]


def _lcs(a: list[str], b: list[str]) -> int:
    """Length of longest common subsequence (O(|a||b|))."""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(hyp: str, ref: str) -> float:
    hw = hyp.lower().split()
    rw = ref.lower().split()
    if not hw or not rw:
        return 0.0
    lcs = _lcs(hw[:80], rw[:80])  # cap to keep O(n²) cheap
    p = lcs / len(hw)
    r = lcs / len(rw)
    if p + r < 1e-9:
        return 0.0
    return 2 * p * r / (p + r)


def distinct_2(text: str) -> float:
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    bgs = _bigrams(words)
    return len(set(bgs)) / len(bgs)


def length_norm(text: str) -> float:
    import math
    words = len(text.split())
    return 1.0 - math.exp(-words / 60.0)  # saturates ~1.0 at 200+ words


def score_completion(text: str, gold: str) -> float:
    if not text or len(text.split()) < 5:
        return 0.0
    d2 = distinct_2(text)
    ln = length_norm(text)
    rl = rouge_l(text, gold)
    return 0.35 * d2 + 0.35 * ln + 0.30 * rl


# ---------------------------------------------------------------------------
# Generation (subprocess, matches dpo_build_pairs.py pattern)
# ---------------------------------------------------------------------------

def _gen_one(bin_path: str, gguf: str, user: str, temperature: float,
             seed: int, max_new_tokens: int = 200) -> str:
    prompt = f"<|rwkv_tokenizer_end_of_text|>User: {user}\n\nAssistant:"
    try:
        r = subprocess.run(
            [bin_path, "generate", "--weights", gguf,
             "--prompt", prompt,
             "--max-new-tokens", str(max_new_tokens),
             "--temperature", str(temperature),
             "--seed", str(seed)],
            capture_output=True, text=True, timeout=90,
        )
        out = r.stdout.strip()
        # Strip the echo'd prompt if present
        if "Assistant:" in out:
            out = out.split("Assistant:", 1)[-1]
        return out.split("\n\n", 1)[0].strip()[:800]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


# ---------------------------------------------------------------------------
# Worker: generate + score shard, write best completion per row
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> None:
    (bin_path, gguf, shard, shard_path,
     n_samples, max_new_tokens, quality_floor, worker_id) = args

    temps = [0.7, 0.9, 1.0][:n_samples]
    written = 0
    t0 = time.time()
    with open(shard_path, "a") as f:
        for global_idx, row in shard:
            msgs = row.get("messages") or []
            user = next((m["content"] for m in msgs if m["role"] == "user"), None)
            gold = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            if not user or not gold:
                continue

            best_score, best_text = -1.0, ""
            for s_i, temp in enumerate(temps):
                text = _gen_one(bin_path, gguf, user, temp,
                                seed=global_idx * 17 + s_i, max_new_tokens=max_new_tokens)
                sc = score_completion(text, gold)
                if sc > best_score:
                    best_score, best_text = sc, text

            if best_score < quality_floor or not best_text:
                continue

            winner = {
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": best_text},
                ],
                "star_score": round(best_score, 4),
                "source": "star",
            }
            f.write(json.dumps(winner, ensure_ascii=False) + "\n")
            written += 1

    elapsed = time.time() - t0
    print(f"[worker {worker_id}] done: {written}/{len(shard)} winners in {elapsed:.0f}s",
          flush=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _find_bin() -> str:
    candidates = [
        ROOT / "target/release/hawking",
        ROOT / "target/debug/dismantle",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    sys.exit("[star] dismantle binary not found — run: cargo build -p hawking --release")


def _run_sft(model_path: str, hf_dir: str, data_path: str, out_dir: str,
             device: str, lr: float, last_n_layers: int) -> int:
    cmd = [
        sys.executable, str(HERE / "rwkv7_sft_torch.py"),
        "--model", model_path,
        "--hf-dir", hf_dir,
        "--data", data_path,
        "--out", out_dir,
        "--device", device,
        "--lr", str(lr),
        "--last-n-layers", str(last_n_layers),
        "--epochs", "1",
        "--grad-accum", "16",
    ]
    print("[star] SFT command:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _run_ppl(model_path: str, hf_dir: str) -> float:
    """Return wikitext-2 PPL (single 4k window) or -1 on failure."""
    cmd = [
        sys.executable, str(HERE / "rwkv7_eval_ppl.py"),
        "--model", model_path,
        "--hf-dir", hf_dir,
        "--corpus", "wikitext2",
        "--tokens", "4096",
        "--stride", "5000",
        "--device", "cpu",
        "--run-id", "star_inline",
        "--out", "/dev/null",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        for line in (r.stdout + r.stderr).splitlines():
            if '"ppl"' in line:
                return json.loads(line)["ppl"]
        return -1.0
    except Exception:
        return -1.0


def _merge_corpus(base: str, augmented: str, out: str) -> int:
    rows = []
    for src in [base, augmented]:
        p = Path(src)
        if p.exists():
            rows.extend(json.loads(l) for l in p.read_text().splitlines() if l.strip())
    Path(out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="GGUF weights for generation")
    ap.add_argument("--model", required=True, help="Initial state_dict.pt or safetensors")
    ap.add_argument("--hf-dir", default=str(ROOT / "models/rwkv7-g1-04-hf"))
    ap.add_argument("--corpus", default=str(ROOT / "artifacts/rwkv7_posttrain/sft.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/rwkv7_posttrain/star"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=3,
                    help="Completions generated per prompt (up to 3 temperatures)")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = full corpus")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--quality-floor", type=float, default=0.30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--last-n-layers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate only for the first 20 rows then exit")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bin_path = _find_bin()
    base_corpus = args.corpus

    corpus_rows = [json.loads(l) for l in Path(args.corpus).read_text().splitlines() if l.strip()]
    if args.max_rows:
        corpus_rows = corpus_rows[: args.max_rows]
    if args.dry_run:
        corpus_rows = corpus_rows[:20]
        print(f"[star] dry-run: using {len(corpus_rows)} rows")

    current_model = args.model

    for rnd in range(1, args.rounds + 1):
        rnd_dir = Path(args.out) / f"round_{rnd}"
        events_path = rnd_dir / "events.jsonl"
        os.makedirs(rnd_dir, exist_ok=True)

        # Resume: if PPL already logged, skip this round
        if events_path.exists():
            for line in events_path.read_text().splitlines():
                try:
                    ev = json.loads(line)
                    if "final_ppl" in ev:
                        print(f"[star] round {rnd} already complete "
                              f"(PPL={ev['final_ppl']:.2f}), skipping")
                        current_model = str(rnd_dir / "sft_out/final/state_dict.pt")
                        break
                except json.JSONDecodeError:
                    pass
            else:
                pass  # incomplete — fall through and redo
            if (rnd_dir / "sft_out/final/state_dict.pt").exists():
                continue

        print(f"\n[star] ===== Round {rnd}/{args.rounds} =====", flush=True)
        t_round = time.time()

        # --- Phase 1: Generate + score ---
        gen_path = rnd_dir / "generated.jsonl"
        if not gen_path.exists():
            print(f"[star] generating {args.n_samples} completions × "
                  f"{len(corpus_rows)} prompts via {args.workers} workers…", flush=True)
            indexed = list(enumerate(corpus_rows))
            shards = [indexed[i::args.workers] for i in range(args.workers)]
            shard_paths = [rnd_dir / f"shard_{i}.jsonl" for i in range(args.workers)]
            worker_args = [
                (bin_path, args.gguf, shards[i], str(shard_paths[i]),
                 args.n_samples, args.max_new_tokens, args.quality_floor, i)
                for i in range(args.workers)
            ]
            with mp.Pool(args.workers) as pool:
                pool.map(_worker, worker_args)
            # Merge shards
            winners = []
            for sp in shard_paths:
                if sp.exists():
                    winners.extend(json.loads(l)
                                   for l in sp.read_text().splitlines() if l.strip())
            gen_path.write_text("\n".join(json.dumps(w, ensure_ascii=False)
                                          for w in winners) + "\n")
            print(f"[star] {len(winners)} winners from {len(corpus_rows)} prompts "
                  f"(accept rate {len(winners)/len(corpus_rows)*100:.1f}%)", flush=True)
        else:
            winners = [json.loads(l) for l in gen_path.read_text().splitlines() if l.strip()]
            print(f"[star] loaded {len(winners)} pre-generated winners", flush=True)

        with open(events_path, "a") as ef:
            ef.write(json.dumps({
                "round": rnd, "phase": "generate",
                "n_corpus": len(corpus_rows),
                "n_winners": len(winners),
                "accept_rate": len(winners) / max(len(corpus_rows), 1),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }) + "\n")

        # --- Phase 2: Merge corpus ---
        augmented_corpus = str(rnd_dir / "augmented_sft.jsonl")
        n_total = _merge_corpus(base_corpus, str(gen_path), augmented_corpus)
        print(f"[star] merged corpus: {n_total} rows "
              f"({len(corpus_rows)} base + {len(winners)} star)", flush=True)

        if args.dry_run:
            print("[star] dry-run: skipping SFT + PPL eval")
            break

        # --- Phase 3: SFT epoch ---
        sft_out = str(rnd_dir / "sft_out")
        rc = _run_sft(
            current_model, args.hf_dir, augmented_corpus, sft_out,
            args.device, args.lr, args.last_n_layers,
        )
        if rc != 0:
            print(f"[star] SFT FAILED (rc={rc}) — aborting round {rnd}", flush=True)
            break

        next_model = str(rnd_dir / "sft_out/final/state_dict.pt")
        if not Path(next_model).exists():
            print(f"[star] SFT produced no checkpoint at {next_model}", flush=True)
            break

        # --- Phase 4: PPL eval ---
        print("[star] evaluating PPL…", flush=True)
        ppl = _run_ppl(next_model, args.hf_dir)
        elapsed = time.time() - t_round

        print(f"[star] round {rnd} done: PPL={ppl:.2f} (prev model={current_model}), "
              f"wall={elapsed/60:.1f}min", flush=True)

        with open(events_path, "a") as ef:
            ef.write(json.dumps({
                "round": rnd, "phase": "done",
                "final_ppl": ppl,
                "sft_out": next_model,
                "wall_s": round(elapsed, 1),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }) + "\n")

        current_model = next_model

    print(f"\n[star] loop complete — final model: {current_model}", flush=True)


if __name__ == "__main__":
    main()
