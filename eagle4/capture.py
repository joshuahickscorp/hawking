"""Capture V2-Lite per-layer hidden + routing state → parquet.

Per token we record: hidden_low/mid/high (layers 2/13/25), shared_hidden
(last MoE layer's shared-expert output), router_logits_per_layer [26x64],
routed_mask_per_layer [26x64], prev_token, next_token.

Each MoE layer's `mlp` attribute is wrapped to record input + pre-softmax
gate logits while delegating to the real MLP — preserves mlx-lm's fused
forward speed. ~745 records/sec on M3 Pro.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load

# V2-Lite constants
FUSION_LAYERS = (2, 13, 25)
N_MOE_LAYERS = 26
N_ROUTED = 64
TOP_K = 6
HIDDEN = 2048
SHARD_ROWS = 8192

SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("position", pa.int32()),
        ("prev_token", pa.int32()),
        ("next_token", pa.int32()),
        ("hidden_low", pa.binary()),
        ("hidden_mid", pa.binary()),
        ("hidden_high", pa.binary()),
        ("shared_hidden", pa.binary()),
        ("router_logits_per_layer", pa.binary()),
        ("routed_mask_per_layer", pa.binary()),
    ]
)


def _install_hooks(model):
    """Wrap each MoE layer's .mlp to capture (mlp_input, gate_logits)."""
    buf: dict[int, tuple] = {}
    moe_idx = 0

    class Hooked:
        def __init__(self, real, idx):
            self._real = real
            self._idx = idx
            self.gate = real.gate
            self.shared_experts = real.shared_experts
            self.switch_mlp = real.switch_mlp
            self.config = real.config
            self.num_experts_per_tok = real.num_experts_per_tok
            self.sharding_group = real.sharding_group

        def __call__(self, x):
            buf[self._idx] = (x, x @ self._real.gate.weight.T)
            return self._real(x)

    for layer in model.model.pipeline_layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp = Hooked(layer.mlp, moe_idx)
            moe_idx += 1
    return buf


def _format(tokenizer, messages, max_ctx):
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    except Exception:
        return None
    if len(ids) < 8:
        return None
    return ids[:max_ctx]


def capture(out_dir: Path, n_records: int, skip_n: int, max_ctx: int, model_id: str, dataset_id: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[capture] loading {model_id}", flush=True)
    model, tok = load(model_id)
    inner = model.model
    capture_buf = _install_hooks(model)
    print(f"[capture] hooks on {N_MOE_LAYERS} MoE layers", flush=True)

    ds = load_dataset(dataset_id, split="train_sft", streaming=True)

    rows: list[dict] = []
    shard = 0
    total = 0
    convs = 0
    t0 = time.time()
    for idx, sample in enumerate(ds):
        if idx < skip_n:
            continue
        if total >= n_records:
            break
        msgs = sample.get("messages")
        if not msgs:
            continue
        ids = _format(tok, msgs, max_ctx)
        if ids is None:
            continue
        input_ids = mx.array([ids], dtype=mx.int32)

        # Run V2-Lite forward; hooks fill capture_buf as MoE layers execute.
        h = inner.embed_tokens(input_ids)
        attn_mask = create_attention_mask(h, None)
        fusion: dict[int, mx.array] = {}
        for li, layer in enumerate(inner.pipeline_layers):
            h = layer(h, attn_mask, None)
            if li in FUSION_LAYERS:
                fusion[li] = h

        low = fusion[FUSION_LAYERS[0]][0].astype(mx.float16)
        mid = fusion[FUSION_LAYERS[1]][0].astype(mx.float16)
        hi = fusion[FUSION_LAYERS[2]][0].astype(mx.float16)
        last_in, _ = capture_buf[N_MOE_LAYERS - 1]
        shared_h = inner.pipeline_layers[26].mlp.shared_experts(last_in)[0].astype(mx.float16)
        rl = mx.stack([capture_buf[mi][1][0].astype(mx.float16) for mi in range(N_MOE_LAYERS)], axis=0)
        mx.eval(low, mid, hi, shared_h, rl)

        rl_np = np.array(rl)
        top6 = np.argpartition(-rl_np, kth=TOP_K - 1, axis=-1)[..., :TOP_K]
        mask_np = np.zeros((N_MOE_LAYERS, rl_np.shape[1], N_ROUTED), dtype=np.uint8)
        np.put_along_axis(mask_np, top6, 1, axis=-1)

        low_b = bytes(memoryview(low))
        mid_b = bytes(memoryview(mid))
        hi_b = bytes(memoryview(hi))
        sh_b = bytes(memoryview(shared_h))
        Hb = HIDDEN * 2

        sample_id = f"sample_{idx}"
        for pos in range(len(ids) - 1):
            rows.append(
                {
                    "sample_id": sample_id,
                    "position": int(pos),
                    "prev_token": int(ids[pos]),
                    "next_token": int(ids[pos + 1]),
                    "hidden_low": low_b[pos * Hb : (pos + 1) * Hb],
                    "hidden_mid": mid_b[pos * Hb : (pos + 1) * Hb],
                    "hidden_high": hi_b[pos * Hb : (pos + 1) * Hb],
                    "shared_hidden": sh_b[pos * Hb : (pos + 1) * Hb],
                    "router_logits_per_layer": rl_np[:, pos, :].astype(np.float16).tobytes(),
                    "routed_mask_per_layer": mask_np[:, pos, :].tobytes(),
                }
            )
            total += 1
            if total >= n_records:
                break

        while len(rows) >= SHARD_ROWS:
            path = out_dir / f"shard_{shard:05d}.parquet"
            pq.write_table(pa.Table.from_pylist(rows[:SHARD_ROWS], schema=SCHEMA), path, compression="zstd")
            print(f"[capture] wrote {path.name} ({SHARD_ROWS} rows)", flush=True)
            rows = rows[SHARD_ROWS:]
            shard += 1

        convs += 1
        if convs % 5 == 0:
            elapsed = time.time() - t0
            rate = total / max(elapsed, 1e-3)
            eta = (n_records - total) / max(rate, 1e-3) / 60
            print(f"[capture] conv={convs} rec={total}/{n_records} {rate:.0f} rec/s eta={eta:.1f}m", flush=True)

    if rows:
        path = out_dir / f"shard_{shard:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path, compression="zstd")
        shard += 1

    elapsed = time.time() - t0
    print(f"[capture] done: {total} records in {shard} shards over {convs} convs ({elapsed:.1f}s, {total/max(elapsed,1):.0f} rec/s)", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(prog="eagle4-capture")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-records", type=int, default=100_000)
    p.add_argument("--skip-n", type=int, default=0)
    p.add_argument("--max-ctx", type=int, default=256)
    p.add_argument("--model", default="mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx")
    p.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    args = p.parse_args()
    capture(args.out_dir, args.n_records, args.skip_n, args.max_ctx, args.model, args.dataset)
    sys.stdout.flush()
    os._exit(0)  # bypass MLX module-reassignment teardown which hangs on exit


if __name__ == "__main__":
    main()
