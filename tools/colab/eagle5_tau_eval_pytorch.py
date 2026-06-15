#!/usr/bin/env python3
"""PyTorch τ-at-depth-K eval for Colab-trained Qwen-3B Eagle5 heads.

This is the Colab counterpart to ``tools/training/eagle5_tau_eval.py``. It
loads a trained PyTorch/safetensors Eagle5 head, rolls it autoregressively for
K draft positions, and compares each draft token to the frozen full-model
argmax computed from the captured residual at that position.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from eagle5_train_pytorch import N_HEADS, RMS_EPS, _extract_row, _rms_norm, build_head


def _infer_head_knobs(ckpt: Path, requested_blocks: int, requested_heads: int, requested_ff_mult: float):
    num_blocks = requested_blocks
    n_heads = requested_heads
    ff_mult = requested_ff_mult
    if ckpt.suffix == ".safetensors":
        try:
            from safetensors import safe_open

            with safe_open(str(ckpt), framework="pt", device="cpu") as f:
                meta = f.metadata() or {}
            num_blocks = int(meta.get("num_blocks", num_blocks))
            n_heads = int(meta.get("n_heads", n_heads))
            ff_mult = float(meta.get("ff_mult", ff_mult))
        except Exception as e:
            print(f"[tau] WARN: could not read safetensors metadata: {e}", flush=True)
    else:
        try:
            with np.load(ckpt) as z:
                if "__num_blocks__" in z.files:
                    num_blocks = int(np.asarray(z["__num_blocks__"]).item())
                if "__n_heads__" in z.files:
                    n_heads = int(np.asarray(z["__n_heads__"]).item())
                if "__ff_mult_x1000__" in z.files:
                    ff_mult = float(np.asarray(z["__ff_mult_x1000__"]).item()) / 1000.0
        except Exception as e:
            print(f"[tau] WARN: could not read npz metadata: {e}", flush=True)
    return num_blocks, n_heads, ff_mult


def _load_head(
    ckpt: Path,
    frozen: Path,
    device: str,
    *,
    num_blocks: int = 1,
    n_heads: int = N_HEADS,
    ff_mult: float = 4.0,
):
    num_blocks, n_heads, ff_mult = _infer_head_knobs(ckpt, num_blocks, n_heads, ff_mult)
    head = build_head(
        frozen,
        with_sparsity=False,
        device=device,
        num_blocks=num_blocks,
        n_heads=n_heads,
        ff_mult=ff_mult,
    ).eval()
    if ckpt.suffix == ".safetensors":
        from safetensors import safe_open

        state = {}
        with safe_open(str(ckpt), framework="pt", device=device) as f:
            for k in f.keys():
                if not k.startswith("_"):
                    state[k] = f.get_tensor(k)
    else:
        z = np.load(ckpt)
        state = {
            k: torch.from_numpy(np.asarray(z[k])).to(device)
            for k in z.files
            if not k.startswith("__")
        }

    current = head.state_dict()
    filtered = {}
    for k, v in state.items():
        if k in current:
            filtered[k] = v.to(device=device, dtype=current[k].dtype)
    missing, unexpected = head.load_state_dict(filtered, strict=False)
    missing_trainable = [k for k in missing if k in dict(head.named_parameters())]
    if missing_trainable:
        raise RuntimeError(f"checkpoint missing trainable keys: {missing_trainable[:8]}")
    if unexpected:
        print(f"[tau] WARN: unexpected checkpoint keys: {unexpected[:8]}", flush=True)
    return head


def _read_windows_from_shard(
    shard: Path,
    depth: int,
    max_row_tokens: int,
) -> list[dict[str, np.ndarray]]:
    table = pq.read_table(shard)
    out = []
    for i in range(table.num_rows):
        row = {c: table[c][i].as_py() for c in table.column_names}
        ex = _extract_row(row, max_row_tokens=max_row_tokens)
        if ex is None:
            continue
        n = len(ex["prev_tokens"])
        for off in range(0, n - depth + 1, depth):
            out.append(
                {
                    "prev": ex["prev_tokens"][off : off + depth],
                    "next": ex["next_tokens"][off : off + depth],
                    "residual": ex["residual"][off : off + depth],
                    "intermediate": ex["intermediate"][off : off + depth],
                }
            )
    return out


def _load_eval_windows(
    corpus_dir: Path,
    depth: int,
    max_windows: int,
    max_row_tokens: int,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    shards = sorted(corpus_dir.glob("shard_*.parquet"))
    if not shards:
        raise RuntimeError(f"no parquet shards found in {corpus_dir}")
    rng = random.Random(seed)
    rng.shuffle(shards)

    windows: list[dict[str, np.ndarray]] = []
    seen: set[bytes] = set()
    max_workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for shard_windows in pool.map(
            lambda s: _read_windows_from_shard(s, depth, max_row_tokens), shards
        ):
            rng.shuffle(shard_windows)
            for w in shard_windows:
                fp = w["prev"][:64].tobytes()
                if fp in seen:
                    continue
                seen.add(fp)
                windows.append(w)
                if len(windows) >= max_windows:
                    return windows
    return windows


@torch.inference_mode()
def evaluate(args) -> dict:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[tau] WARN: CUDA unavailable; falling back to CPU", flush=True)
        device = "cpu"
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print("[tau] WARN: MPS unavailable; falling back to CPU", flush=True)
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
    )
    lm_head_f = head._lm_head.float()
    windows = _load_eval_windows(
        args.corpus,
        args.depth,
        args.max_windows,
        args.max_row_tokens,
        args.seed,
    )
    if not windows:
        raise RuntimeError("no usable eval windows loaded")

    W = len(windows)
    prev = torch.from_numpy(np.stack([w["prev"] for w in windows]).astype(np.int64)).to(device)
    nxt = torch.from_numpy(np.stack([w["next"] for w in windows]).astype(np.int64)).to(device)
    residual = torch.from_numpy(
        np.stack([w["residual"] for w in windows]).astype(np.float32)
    ).to(device)
    inter = torch.from_numpy(
        np.stack([w["intermediate"] for w in windows]).astype(np.float32)
    ).to(device)

    accepted_len = torch.zeros(W, device=device, dtype=torch.int32)
    per_pos_accept = torch.zeros(args.depth, device=device, dtype=torch.int64)
    cur_prev = prev[:, :1]
    # Chained-hidden state for --chain-hidden: depth 0 uses the real captured
    # residual; deeper depths feed the head's own draft_hidden forward
    # (intermediate=0) — EXACTLY what the runtime does. This makes the reported
    # accepted-prefix the RUNTIME-PREDICTIVE number (the speedup driver),
    # whereas feeding the real per-depth residual is optimistic (the runtime
    # never has it). See memory/eagle5_corrected_pipeline_2026_05_29.md.
    chain_res = residual[:, 0:1, :]
    chain_inter = inter[:, 0:1, :]

    for d in range(args.depth):
        if args.chain_hidden:
            residual_d = chain_res
            inter_d = chain_inter
        else:
            residual_d = residual[:, d : d + 1, :]
            inter_d = inter[:, d : d + 1, :]
        token_logits, _sparsity, _draft_h, _calib = head(cur_prev, residual_d, inter_d)
        head_arg = token_logits[:, 0, :].float().argmax(dim=-1)
        if args.chain_hidden:
            chain_res = _draft_h.detach()
            chain_inter = torch.zeros_like(chain_res)

        # Acceptance target. THE CRITICAL FIX (2026-05-29): the runtime
        # verifier accepts a draft only when it equals the model's REAL next
        # token. The legacy metric compared against
        # argmax(RMSNorm(captured_residual) @ lm_head) — a self-referential
        # proxy derived from the head's own input — which inflates τ to ~100%
        # while real acceptance is ~0%. target-mode:
        #   corpus (default) — the captured real next token (ground truth)
        #   proxy            — legacy self-referential baseline
        if args.target_mode == "corpus":
            target_arg = nxt[:, d]
        else:
            baseline = _rms_norm(residual_d, head._output_norm, RMS_EPS).reshape(
                W, head.hidden_dim
            )
            target_logits = torch.matmul(baseline.float(), lm_head_f)
            target_arg = target_logits.argmax(dim=-1)

        still_accepting = accepted_len == d
        accepted_step = still_accepting & (head_arg == target_arg)
        accepted_len += accepted_step.to(torch.int32)
        per_pos_accept[d] = accepted_step.sum()
        cur_prev = head_arg.reshape(W, 1)

    accepted_cpu = accepted_len.detach().cpu().numpy()
    per_pos = (per_pos_accept.detach().cpu().numpy() / float(W)).tolist()
    tau = float(accepted_cpu.mean())
    projected_tps = (
        float(args.base_tps)
        * (1.0 + tau)
        * float(args.w4a8_multiplier)
        * float(args.spec_efficiency)
    )
    return {
        "schema": "eagle5-tau-pytorch-v1",
        "ckpt": str(args.ckpt),
        "frozen": str(args.frozen),
        "corpus": str(args.corpus),
        "windows": int(W),
        "depth": int(args.depth),
        "tau": tau,
        "depth1_accept_rate": float(per_pos[0]) if per_pos else 0.0,
        "full_accept_rate": float((accepted_cpu == args.depth).mean()),
        "per_pos_accept_rate": per_pos,
        "projection": {
            "base_tps": float(args.base_tps),
            "w4a8_multiplier": float(args.w4a8_multiplier),
            "spec_efficiency": float(args.spec_efficiency),
            "projected_dec_tps": projected_tps,
            "formula": "base_tps * (1 + tau) * w4a8_multiplier * spec_efficiency",
        },
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--frozen", required=True, type=Path)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--max-windows", type=int, default=4000)
    p.add_argument("--max-row-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    p.add_argument("--target-mode", choices=("corpus", "proxy"), default="corpus",
                   help="corpus = real next token (ground truth); proxy = legacy "
                        "self-referential baseline argmax")
    p.add_argument("--chain-hidden", action="store_true",
                   help="Feed the head's own draft_hidden forward as the next-depth "
                        "residual (matches the runtime + --rollout-chain-hidden). "
                        "Reports the RUNTIME-PREDICTIVE accepted-prefix; without it "
                        "the eval uses real per-depth residuals (optimistic).")
    p.add_argument("--num-blocks", type=int, default=1)
    p.add_argument("--head-heads", type=int, default=N_HEADS)
    p.add_argument("--head-ff-mult", type=float, default=4.0)
    p.add_argument("--base-tps", type=float, default=26.6)
    p.add_argument("--w4a8-multiplier", type=float, default=1.25)
    p.add_argument("--spec-efficiency", type=float, default=0.85)
    args = p.parse_args()

    if args.depth <= 0 or args.max_windows <= 0:
        raise SystemExit("--depth and --max-windows must be positive")
    result = evaluate(args)
    _write_json_atomic(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        f"[tau] tau@{args.depth}={result['tau']:.3f} "
        f"depth1={result['depth1_accept_rate']:.2%} "
        f"projected={result['projection']['projected_dec_tps']:.1f} dec_tps "
        f"→ {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
