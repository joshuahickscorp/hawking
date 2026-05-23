#!/usr/bin/env python3
"""Build the V2-Lite calibration corpus.

Runs DeepSeek-V2-Lite-Chat forward over a chat-style corpus and captures the
per-token, per-layer intermediates needed by Phase 1 levers in the dismantle
execution plan:

- mixed-precision quant calibration (per-tensor sensitivity, GPTQ/AWQ)
- eagle5 activation-sparsity predictor training
- quant-quality regression eval
- vocab-prune calibration (top-K vocab over chat-decode tokens)

One run, offline. Output lands in artifacts/calibration/v2_lite_corpus/ as
int8-quantized parquet shards. Idempotent per-shard — crashed runs resume.

Backend: HuggingFace transformers + MPS. See tools/training/README.md for
why this and not MLX/llama-cpp/dismantle's own engine.

Usage:
    python3 tools/training/build_corpus.py \\
        --model deepseek-ai/DeepSeek-V2-Lite-Chat \\
        --dataset HuggingFaceH4/ultrachat_200k \\
        --max-sequences 10000 \\
        --out artifacts/calibration/v2_lite_corpus

Plan: dismantle-execution-plan-enchanted-salamander.md, task #7.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

# Heavy imports are guarded so `--help` doesn't pay for them.
def _require(*mods: str) -> None:
    import importlib.util
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    if missing:
        sys.stderr.write(
            "error: missing python deps: " + ", ".join(missing) + "\n"
            "       pip install -r tools/training/requirements.txt\n"
        )
        sys.exit(2)


@dataclasses.dataclass
class Args:
    model: str
    dataset: str
    dataset_split: str
    max_sequences: int
    max_tokens_per_seq: int
    shard_size: int
    out: Path
    device: str
    dtype: str
    quantize_intermediates: str
    skip_existing: bool
    capture: tuple[str, ...]
    batch_size: int
    skip_rows: int


def parse_args() -> Args:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    p.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--dataset-split", default="train_sft")
    p.add_argument("--max-sequences", type=int, default=10_000)
    p.add_argument("--max-tokens-per-seq", type=int, default=2048)
    p.add_argument("--shard-size", type=int, default=128,
                   help="sequences per parquet shard")
    p.add_argument("--out", type=Path,
                   default=Path("artifacts/calibration/v2_lite_corpus"))
    p.add_argument("--device", default="mps",
                   choices=["mps", "cpu", "cuda"],
                   help="MPS is the default on M3 Pro; CUDA for cross-machine runs")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--quantize-intermediates", default="int8",
                   choices=["int8", "none"],
                   help="int8 cuts shard size 4x with negligible calibration drift")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="skip shards that already exist on disk (resume default)")
    p.add_argument("--skip-rows", type=int, default=0,
                   help="skip the first N rows of the source dataset before sampling; "
                        "pair with the watchdog so resumption walks forward instead of "
                        "re-sampling row 0 each restart (see wall-clock audit 2026-05-22)")
    p.add_argument("--capture", default="all",
                   help="comma-separated subset of: residual_in,expert_idx,"
                        "routing_logits,intermediate,h_high,output_logits,all")
    p.add_argument("--batch-size", type=int, default=1,
                   help="sequences per forward pass. >1 amortizes disk-offload "
                        "expert streaming across the batch")
    a = p.parse_args()
    capture = (
        # "all" intentionally omits output_logits + h_high — both are huge
        # (vocab × tokens, hidden × tokens) and only needed for vocab-prune,
        # which uses a small dedicated subset. Pass --capture explicitly to
        # include them.
        ("residual_in", "expert_idx", "routing_logits", "intermediate")
        if a.capture == "all"
        else tuple(s.strip() for s in a.capture.split(",") if s.strip())
    )
    return Args(
        model=a.model,
        dataset=a.dataset,
        dataset_split=a.dataset_split,
        max_sequences=a.max_sequences,
        max_tokens_per_seq=a.max_tokens_per_seq,
        shard_size=a.shard_size,
        out=a.out,
        device=a.device,
        dtype=a.dtype,
        quantize_intermediates=a.quantize_intermediates,
        skip_existing=a.skip_existing,
        capture=capture,
        batch_size=a.batch_size,
        skip_rows=a.skip_rows,
    )


def iter_chat_sequences(
    dataset_name: str,
    split: str,
    limit: int,
    skip_rows: int = 0,
) -> Iterator[str]:
    """Yield rendered chat strings up to `limit`, skipping the first
    `skip_rows` of the underlying dataset.

    Wall-clock optimization #5 (2026-05-22): the autonomous corpus
    watchdog at `tools/training/run_corpus_autonomous.sh` historically
    re-started this iterator at row 0 on every crash + restart, which
    produced ~60% duplicate sequences across the 4,512-row 2026-05-21
    corpus. Future captures should pass `--skip-rows N` where N is
    `heartbeat['rows_consumed']` so resumption walks forward instead of
    re-sampling the same prefix. See `feedback_wall_clock_audit.md`.
    """
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset(dataset_name, split=split, streaming=True)
    yielded = 0
    skipped = 0
    for row in ds:
        if skipped < skip_rows:
            skipped += 1
            continue
        if yielded >= limit:
            return
        # ultrachat_200k schema: row["messages"] = list[{role, content}]
        messages = row.get("messages") or row.get("conversations") or []
        if not messages:
            continue
        # Render as a single string. The actual chat-template should be
        # applied per-tokenizer downstream — here we want raw text variety,
        # not chat-completion targets.
        parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if content:
                parts.append(f"{role}: {content}")
        if parts:
            yield "\n".join(parts)
            yielded += 1


def quantize_int8(arr):
    """Symmetric per-tensor int8 quantization. Returns (q, scale)."""
    import numpy as np
    max_abs = float(np.abs(arr).max())
    if max_abs == 0.0:
        return np.zeros_like(arr, dtype=np.int8), 0.0
    scale = max_abs / 127.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q, scale


def capture_one_sequence(
    text: str,
    model,
    tokenizer,
    cap_set: set[str],
    max_tokens: int,
    device: str,
):
    """Run one sequence forward and return per-token, per-layer dicts."""
    import torch  # type: ignore[import-not-found]

    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_tokens, add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # output_hidden_states gives us residual_in / h_high at layer boundaries.
    # output_router_logits gives routing_logits + expert_idx.
    # For "intermediate" (post-SiLU gate-up) we register forward hooks on
    # the expert FFN modules. See the model's modeling_deepseek.py for the
    # exact attribute path; this scaffold uses a generic name fallback.
    captured_intermediates: list[dict] = []
    hooks = []

    # See capture_batch for the async-hook rationale: store device tensors
    # in the hook, defer .cpu().numpy() to a single post-forward sweep.
    def _make_hook(layer_idx: int):
        def _hook(_module, _inp, out):
            if isinstance(out, tuple):
                arrs = [o.detach() for o in out if hasattr(o, "detach")]
                payload = arrs[0] if len(arrs) == 1 else arrs
            else:
                payload = out.detach() if hasattr(out, "detach") else out
            captured_intermediates.append({"layer": layer_idx, "raw_t": payload})
        return _hook

    if "intermediate" in cap_set:
        for li, layer in enumerate(getattr(model, "model", model).layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "experts"):
                hooks.append(mlp.experts[0].register_forward_hook(_make_hook(li)))

    captured_gates: list[dict] = []
    want_routing = ("routing_logits" in cap_set or "expert_idx" in cap_set)
    if want_routing:
        def _make_gate_hook(layer_idx: int):
            def _gate_hook(_module, _inp, out):
                if isinstance(out, tuple) and len(out) >= 2:
                    captured_gates.append({
                        "layer": layer_idx,
                        "topk_idx_t": out[0].detach(),
                        "topk_weight_t": out[1].detach(),
                    })
            return _gate_hook
        for li, layer in enumerate(getattr(model, "model", model).layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "gate") and hasattr(mlp, "experts"):
                hooks.append(mlp.gate.register_forward_hook(_make_gate_hook(li)))

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=("residual_in" in cap_set or "h_high" in cap_set),
            use_cache=False,
            return_dict=True,
        )

    for h in hooks:
        h.remove()

    # Post-forward batched transfer.
    for g in captured_gates:
        g["topk_idx"] = g.pop("topk_idx_t").cpu().numpy()
        g["topk_weight"] = g.pop("topk_weight_t").float().cpu().numpy()
    for it in captured_intermediates:
        rt = it.pop("raw_t")
        if isinstance(rt, list):
            it["raw"] = [(x.float().cpu().numpy() if hasattr(x, "detach") else x)
                         for x in rt]
        else:
            it["raw"] = (rt.float().cpu().numpy() if hasattr(rt, "detach") else rt)

    # Convert to numpy for parquet write-out. Quantize per --quantize.
    import numpy as np
    sample = {
        "tokens": input_ids.cpu().numpy().astype(np.int32).squeeze(0),
        "n_tokens": int(input_ids.shape[1]),
    }

    if "output_logits" in cap_set and hasattr(out, "logits"):
        sample["output_logits"] = out.logits.detach().cpu().numpy().squeeze(0)

    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        if "residual_in" in cap_set:
            sample["residual_in_per_layer"] = [
                h.detach().cpu().numpy().squeeze(0) for h in out.hidden_states[:-1]
            ]
        if "h_high" in cap_set:
            sample["h_high"] = out.hidden_states[-1].detach().cpu().numpy().squeeze(0)

    if captured_gates:
        if "expert_idx" in cap_set:
            sample["expert_idx_per_layer"] = [g["topk_idx"] for g in captured_gates]
        if "routing_logits" in cap_set:
            # MoEGate exposes only post-topk weights (n_tokens, top_k), not full
            # per-expert scores. Store the topk weights aligned with expert_idx.
            sample["routing_topk_weight_per_layer"] = [
                g["topk_weight"] for g in captured_gates
            ]

    if captured_intermediates and "intermediate" in cap_set:
        sample["intermediate_per_layer"] = captured_intermediates

    return sample


def capture_batch(
    texts: list[str],
    model,
    tokenizer,
    cap_set: set[str],
    max_tokens: int,
    device: str,
) -> list[dict]:
    """Batched forward — runs one model() call across N sequences then demuxes
    per-sample arrays. Amortizes disk-offloaded expert streaming across the
    batch. Quality-equivalent to capture_one_sequence; only the wall-clock cost
    of the streaming reads is shared.
    """
    import torch  # type: ignore[import-not-found]
    import numpy as np

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(texts, return_tensors="pt", truncation=True,
                    max_length=max_tokens, padding=True, add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    B, S = input_ids.shape
    seq_lens = attention_mask.sum(dim=1).cpu().tolist()

    captured_gates: list[dict] = []
    captured_intermediates: list[dict] = []
    hooks = []

    # ASYNC HOOKS: store device tensors only during forward; defer the
    # MPS→CPU→numpy transfer until after model() returns. With 27 layers ×
    # 2 hook types, the prior code created 50+ synchronous transfer barriers
    # inside the forward pass, blocking MPS pipelining. Now we batch all
    # transfers post-forward in one sweep.
    want_routing = ("routing_logits" in cap_set or "expert_idx" in cap_set)
    if want_routing:
        def _make_gate_hook(layer_idx: int):
            def _gate_hook(_module, _inp, out):
                if isinstance(out, tuple) and len(out) >= 2:
                    captured_gates.append({
                        "layer": layer_idx,
                        "topk_idx_t": out[0].detach(),
                        "topk_weight_t": out[1].detach(),
                    })
            return _gate_hook
        for li, layer in enumerate(getattr(model, "model", model).layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "gate") and hasattr(mlp, "experts"):
                hooks.append(mlp.gate.register_forward_hook(_make_gate_hook(li)))

    if "intermediate" in cap_set:
        def _make_hook(layer_idx: int):
            def _hook(_module, _inp, out):
                if isinstance(out, tuple):
                    arrs = [o.detach() for o in out if hasattr(o, "detach")]
                    payload = arrs[0] if len(arrs) == 1 else arrs
                else:
                    payload = out.detach() if hasattr(out, "detach") else out
                captured_intermediates.append({"layer": layer_idx, "raw_t": payload})
            return _hook
        for li, layer in enumerate(getattr(model, "model", model).layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "experts"):
                hooks.append(mlp.experts[0].register_forward_hook(_make_hook(li)))

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=("residual_in" in cap_set or "h_high" in cap_set),
            use_cache=False,
            return_dict=True,
        )

    for h in hooks:
        h.remove()

    # Batch-resolve all deferred device→numpy transfers in one post-forward
    # sweep. Order doesn't matter for correctness; what matters is that the
    # MPS pipeline isn't stalled mid-forward.
    for g in captured_gates:
        g["topk_idx"] = g.pop("topk_idx_t").cpu().numpy()
        g["topk_weight"] = g.pop("topk_weight_t").float().cpu().numpy()
    for it in captured_intermediates:
        rt = it.pop("raw_t")
        if isinstance(rt, list):
            it["raw"] = [(x.float().cpu().numpy() if hasattr(x, "detach") else x)
                         for x in rt]
        else:
            it["raw"] = (rt.float().cpu().numpy() if hasattr(rt, "detach") else rt)

    # Demux per-sample.
    samples: list[dict] = []
    input_ids_np = input_ids.detach().cpu().numpy().astype(np.int32)
    hidden_per_layer = None
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        hidden_per_layer = [h.detach().cpu().numpy() for h in out.hidden_states]

    for b in range(B):
        L = int(seq_lens[b])
        sample: dict = {
            "tokens": input_ids_np[b, :L],
            "n_tokens": L,
        }
        if "output_logits" in cap_set and hasattr(out, "logits"):
            sample["output_logits"] = out.logits[b, :L, :].detach().cpu().numpy()
        if hidden_per_layer is not None:
            if "residual_in" in cap_set:
                sample["residual_in_per_layer"] = [
                    h[b, :L, :] for h in hidden_per_layer[:-1]
                ]
            if "h_high" in cap_set:
                sample["h_high"] = hidden_per_layer[-1][b, :L, :]
        if captured_gates:
            # MoEGate output shape is [B*S, top_k] — reshape to [B, S, K] then
            # slice per-sample to true length.
            if "expert_idx" in cap_set:
                sample["expert_idx_per_layer"] = []
            if "routing_logits" in cap_set:
                sample["routing_topk_weight_per_layer"] = []
            for g in captured_gates:
                idx = g["topk_idx"]
                wt = g["topk_weight"]
                if idx.ndim == 2 and idx.shape[0] == B * S:
                    idx_bs = idx.reshape(B, S, -1)[b, :L, :]
                    wt_bs = wt.reshape(B, S, -1)[b, :L, :]
                else:
                    # Already-shaped or unexpected — fall back to first-N take.
                    idx_bs = idx[:L]
                    wt_bs = wt[:L]
                if "expert_idx" in cap_set:
                    sample["expert_idx_per_layer"].append(idx_bs)
                if "routing_logits" in cap_set:
                    sample["routing_topk_weight_per_layer"].append(wt_bs)
        # Intermediate is best-effort (one expert per MoE layer); attribute the
        # batch-level capture to sample 0 only to avoid duplicating storage.
        if b == 0 and captured_intermediates and "intermediate" in cap_set:
            sample["intermediate_per_layer"] = captured_intermediates
        samples.append(sample)

    return samples


def write_shard(samples: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    # Parquet schemas tolerate ragged arrays via list-of-list columns.
    # The exact column set adapts to whatever was actually captured.
    tbl = pa.Table.from_pylist([_flatten_for_parquet(s) for s in samples])
    pq.write_table(tbl, path, compression="zstd")


def _flatten_for_parquet(sample: dict) -> dict:
    """Convert numpy + nested structures into parquet-safe py types."""
    import numpy as np

    def _conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k: _conv(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_conv(x) for x in v]
        if isinstance(v, tuple):
            return [_conv(x) for x in v]
        if hasattr(v, "detach"):
            return v.detach().float().cpu().numpy().tolist()
        return v

    return {k: _conv(v) for k, v in sample.items()}


def main() -> int:
    args = parse_args()
    _require("torch", "transformers", "datasets", "pyarrow", "numpy", "tqdm")

    import torch  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
    from tqdm import tqdm  # type: ignore[import-not-found]

    if args.device == "mps" and not torch.backends.mps.is_available():
        print("warning: --device mps but MPS not available; falling back to cpu",
              file=sys.stderr)
        args.device = "cpu"

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_split": args.dataset_split,
        "max_sequences": args.max_sequences,
        "max_tokens_per_seq": args.max_tokens_per_seq,
        "shard_size": args.shard_size,
        "dtype": args.dtype,
        "quantize_intermediates": args.quantize_intermediates,
        "capture": list(args.capture),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    cap_set = set(args.capture)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    print(f"loading {args.model} …", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    offload_folder = args.out / "offload"
    offload_folder.mkdir(parents=True, exist_ok=True)
    if args.device == "mps":
        max_memory = {"mps": "3GiB", "cpu": "11GiB"}
    elif args.device == "cuda":
        max_memory = None
    else:
        max_memory = {"cpu": "12GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=str(offload_folder),
        offload_buffers=True,
        low_cpu_mem_usage=True,
    ).eval()
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = True
    try:
        input_device = next(iter(model.hf_device_map.values()))
    except Exception:
        input_device = args.device
    print(f"hf_device_map sample: {list(model.hf_device_map.items())[:4]} …",
          file=sys.stderr)
    print(f"inputs will go to: {input_device}", file=sys.stderr)
    args.device = str(input_device)

    print(f"streaming {args.dataset} [{args.dataset_split}] …", file=sys.stderr)
    seq_iter = iter_chat_sequences(
        args.dataset,
        args.dataset_split,
        args.max_sequences,
        skip_rows=args.skip_rows,
    )

    shard_buf: list[dict] = []
    shard_idx = 0
    yielded = 0
    # Pre-advance shard_idx past any existing shards (resume-from-skip).
    if args.skip_existing:
        while (args.out / f"shard_{shard_idx:04d}.parquet").exists():
            shard_idx += 1
            yielded += args.shard_size
    pbar = tqdm(total=args.max_sequences, desc="sequences", initial=yielded)

    batch_buf: list[str] = []

    def _drain_batch():
        nonlocal shard_buf, shard_idx, yielded
        if not batch_buf:
            return
        try:
            if args.batch_size > 1:
                samples = capture_batch(
                    batch_buf, model, tokenizer, cap_set,
                    args.max_tokens_per_seq, args.device,
                )
            else:
                samples = [capture_one_sequence(
                    batch_buf[0], model, tokenizer, cap_set,
                    args.max_tokens_per_seq, args.device,
                )]
        except Exception as e:  # noqa: BLE001
            print(f"\nwarn: batch skipped ({type(e).__name__}: {e})",
                  file=sys.stderr)
            batch_buf.clear()
            return
        for sample in samples:
            shard_buf.append(sample)
            yielded += 1
            pbar.update(1)
            if len(shard_buf) >= args.shard_size:
                shard_path = args.out / f"shard_{shard_idx:04d}.parquet"
                write_shard(shard_buf, shard_path)
                print(f"\nwrote {shard_path} ({len(shard_buf)} seqs)",
                      file=sys.stderr)
                shard_buf = []
                shard_idx += 1
        batch_buf.clear()

    for text in seq_iter:
        batch_buf.append(text)
        if len(batch_buf) >= args.batch_size:
            _drain_batch()
    _drain_batch()  # final partial batch

    if shard_buf:
        shard_path = args.out / f"shard_{shard_idx:04d}.parquet"
        write_shard(shard_buf, shard_path)
        print(f"wrote final {shard_path} ({len(shard_buf)} seqs)", file=sys.stderr)

    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["sequences_written"] = yielded
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\ndone. {yielded} sequences across {shard_idx + 1} shards in {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
