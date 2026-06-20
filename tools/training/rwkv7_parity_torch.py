"""RWKV-7 pure-torch forward parity gate.

Loads the BF16 safetensors (cast fp32) into the pure-torch RWKV7Model, runs a
full-sequence prefill on the committed fixture prompts, and checks the
last-position logits + greedy continuation against the gold reference dumped
from dismantle's Rust oracle (bit-exact vs llama.cpp).

Gold source (set via --gold-dir, default /tmp/rwkv_ref):
  <stem>.prompt_ids  : space-separated prompt token ids
  <stem>.gen_ids     : space-separated gold greedy continuation ids
  <stem>.logits0     : 65536 float32 (little-endian) last-position logits

PASS (CPU fp32): argmax(logits[-1]) == gold argmax; max|Δlogit| <= LOGIT_TOL;
greedy gen_ids exactly equal gold for N tokens. (MPS uses a looser logit tol.)

Usage:
  python rwkv7_parity_torch.py                       # CPU gate, default gold
  python rwkv7_parity_torch.py --device mps          # MPS gate
  python rwkv7_parity_torch.py --layerwise rwkv_ref  # per-block diff vs Rust dump
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from rwkv7_load_weights import load_rwkv7
from rwkv7_torch_model import RWKV7Config

DEFAULT_MODEL = "models/rwkv7-g1-04-hf/model.safetensors"
LOGIT_TOL_CPU = 0.03
LOGIT_TOL_MPS = 0.10


def read_ids(path: Path) -> list[int]:
    return [int(t) for t in path.read_text().split()]


def read_logits(path: Path, vocab: int = 65536) -> np.ndarray:
    a = np.fromfile(path, dtype="<f4")
    assert a.size == vocab, f"{path}: expected {vocab} f32, got {a.size}"
    return a


def greedy(model, prompt_ids: list[int], n: int, device: str):
    """Full-sequence prefill then greedy-decode n tokens. Returns (gen_ids,
    last_position_logits_at_prefill)."""
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids)  # [1, T, vocab]
    logits0 = logits[0, -1, :].float().cpu().numpy()
    gen = []
    cur = list(prompt_ids)
    nxt = int(np.argmax(logits0))
    gen.append(nxt)
    for _ in range(1, n):
        cur.append(nxt)
        ids = torch.tensor([cur], dtype=torch.long, device=device)
        with torch.no_grad():
            lg = model(ids)
        nxt = int(torch.argmax(lg[0, -1, :]).item())
        gen.append(nxt)
    return gen, logits0


def run_gate(model_path: str, gold_dir: Path, stems: list[str], device: str, logit_tol: float):
    model = load_rwkv7(model_path, RWKV7Config(), device=device, dtype=torch.float32)
    all_ok = True
    for stem in stems:
        prompt_ids = read_ids(gold_dir / f"{stem}.prompt_ids")
        gold_gen = read_ids(gold_dir / f"{stem}.gen_ids")
        gold_logits = read_logits(gold_dir / f"{stem}.logits0")
        n = len(gold_gen)

        gen, logits0 = greedy(model, prompt_ids, n, device)

        my_argmax = int(np.argmax(logits0))
        gold_argmax = int(np.argmax(gold_logits))
        max_dlogit = float(np.max(np.abs(logits0 - gold_logits)))
        argmax_ok = my_argmax == gold_argmax
        gen_ok = gen == gold_gen
        tol_ok = max_dlogit <= logit_tol
        matched = 0
        for x, y in zip(gen, gold_gen):
            if x != y:
                break
            matched += 1

        ok = argmax_ok and gen_ok and tol_ok
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"[{stem}] {status} ({device})")
        print(f"    max|Δlogit| = {max_dlogit:.6f}  (tol {logit_tol})  {'ok' if tol_ok else 'OVER'}")
        print(f"    argmax: mine={my_argmax} gold={gold_argmax}  {'ok' if argmax_ok else 'MISMATCH'}")
        print(f"    gen_ids: {matched}/{n} match  {'ok' if gen_ok else 'MISMATCH'}")
        if not gen_ok:
            print(f"      mine={gen}")
            print(f"      gold={gold_gen}")
    return all_ok


def run_layerwise(model_path: str, dump_dir: Path, stem: str, device: str):
    """Compare per-block hidden states torch-vs-Rust to localize first divergence.

    Expects Rust per-layer dumps at <dump_dir>/<stem>.hidden_<L>.f32 (T*n_embd
    float32, the post-block-L hidden over all prompt positions). Falls back to a
    summary if dumps are absent."""
    model = load_rwkv7(model_path, RWKV7Config(), device=device, dtype=torch.float32)
    prompt_ids = read_ids(dump_dir / f"{stem}.prompt_ids")
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, hiddens = model(ids, return_hidden=True)
    T = len(prompt_ids)
    n = model.cfg.n_embd
    print(f"layerwise [{stem}] T={T} n_embd={n}")
    found = False
    for li, h in enumerate(hiddens):
        ref_path = dump_dir / f"{stem}.hidden_{li}.f32"
        if not ref_path.exists():
            continue
        found = True
        ref = torch.from_numpy(np.fromfile(ref_path, dtype="<f4")).reshape(T, n)
        mine = h[0].float().cpu()
        d = (mine - ref).abs().max().item()
        dlast = (mine[-1] - ref[-1]).abs().max().item()
        print(f"  block {li:2d}: max|Δ| all={d:.6e}  last-pos={dlast:.6e}")
    if not found:
        print("  (no Rust per-layer dumps found; printing torch hidden norms by block)")
        for li, h in enumerate(hiddens):
            print(f"  block {li:2d}: hidden[-1] L2={h[0,-1].float().norm().item():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gold-dir", default="/tmp/rwkv_ref")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--stems", nargs="+", default=["ref04f32a", "ref04f32b"])
    ap.add_argument("--tol", type=float, default=None, help="override logit tol")
    ap.add_argument("--layerwise", default=None, metavar="DUMP_DIR",
                    help="run layerwise diff against Rust per-layer dumps in DUMP_DIR")
    ap.add_argument("--layerwise-stem", default="ref04f32a")
    args = ap.parse_args()

    if args.layerwise is not None:
        run_layerwise(args.model, Path(args.layerwise), args.layerwise_stem, args.device)
        return

    tol = args.tol if args.tol is not None else (LOGIT_TOL_MPS if args.device == "mps" else LOGIT_TOL_CPU)
    ok = run_gate(args.model, Path(args.gold_dir), args.stems, args.device, tol)
    print()
    print(f"GATE {'GREEN' if ok else 'RED'} ({args.device}, tol {tol})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
