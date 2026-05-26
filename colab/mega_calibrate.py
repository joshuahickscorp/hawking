#!/usr/bin/env python3
"""Mega-calibration: capture EVERYTHING dismantle needs in one Colab pass.

Output per prompt (int8-quantized parquet):
  * tokens                  : input ids
  * residual_q/intermediate : layer-K activation (for Eagle5 head training)
  * proj_input_mean_abs     : per-channel mean|x| at all 7 projection sites
                              × all layers (for AWQ / SmoothQuant /
                              per-channel W4A8 calibration)
  * proj_input_max_abs      : per-channel max|x| (for clip-style scales)
  * topk_logit_ids/probs    : top-k output logits per token (quality bench
                              ground truth)

One ~6 hr H100 run produces calibration data for FOUR downstream projects:
  1. Eagle5 head training (residual + intermediate)
  2. AWQ smoothing factor calculation (per-channel mean|x|)
  3. Per-channel W4A8 static scales (per-channel max|x|)
  4. Quality benchmarks (top-k logits as reference)

Usage (Colab):
  python mega_calibrate.py \\
      --model Qwen/Qwen2.5-3B-Instruct \\
      --max-sequences 2000 \\
      --capture-layer 32 \\
      --batch-size 4 \\
      --out /content/qwen3b_corpus
"""

from __future__ import annotations
import argparse, gc, os, sys, time, json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def quantize_int8(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric int8 quantize."""
    arr = arr.astype(np.float32)
    max_abs = float(np.abs(arr).max()) if arr.size else 0.0
    if max_abs < 1e-8:
        return np.zeros(arr.shape, dtype=np.int8), 0.0
    scale = max_abs / 127.0
    q = np.round(arr / scale).clip(-127, 127).astype(np.int8)
    return q, scale


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                   help="HF model id. Qwen2.5-3B-Instruct (dismantle target) or "
                        "deepseek-ai/DeepSeek-V2-Lite-Chat or similar.")
    p.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--split", default="train_sft")
    p.add_argument("--max-sequences", type=int, default=2000,
                   help="Total prompts to process. 2000 is the standard "
                        "AWQ/SmoothQuant calibration size.")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--capture-layer", type=int, default=32,
                   help="Layer at which to capture residual+intermediate "
                        "(for Eagle5 head). Qwen-3B has 36 layers → 32 = "
                        "near-top. V2-Lite has 27 → use 25.")
    p.add_argument("--topk-logits", type=int, default=100,
                   help="How many top output logit ids+probs to save per "
                        "token (for quality benchmark ground truth).")
    p.add_argument("--shard-size", type=int, default=16)
    p.add_argument("--load-4bit", action="store_true",
                   help="Load model in 4-bit nf4 via bitsandbytes. Required "
                        "on T4/V100/L4 (sub-32 GB VRAM). On A100/H100 use "
                        "native fp16 instead.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"[mega-cal] model={args.model}", flush=True)
    print(f"[mega-cal] mode={'4-bit nf4' if args.load_4bit else 'native fp16'}",
          flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model_kwargs = dict(
        trust_remote_code=False,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model_kwargs["device_map"] = "auto"
        del model_kwargs["torch_dtype"]
    else:
        model_kwargs["device_map"] = "cuda"

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()
    print(f"[mega-cal] loaded in {time.time() - t0:.1f}s", flush=True)

    # ── Layer-K residual + intermediate (Eagle5 head training) ───────────
    L = args.capture_layer
    layer = model.model.layers[L]
    layer_captures: Dict[str, torch.Tensor] = {}

    def _residual_hook(_m, _i, out):
        layer_captures["residual"] = (out[0] if isinstance(out, tuple) else out).detach()

    def _intermediate_hook(_m, _i, out):
        layer_captures["intermediate"] = (out[0] if isinstance(out, tuple) else out).detach()

    layer.register_forward_hook(_residual_hook)
    mlp = layer.mlp
    inter_target = mlp.experts if hasattr(mlp, "experts") else mlp
    inter_target.register_forward_hook(_intermediate_hook)

    # ── Per-site mean|x| accumulators across ALL layers (for AWQ etc.) ──
    # Sites we care about for AWQ on dense models:
    #   q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    # Captured INPUT activations (not outputs) → that's what AWQ needs.
    SITE_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")
    n_layers = len(model.model.layers)
    # Per-channel running stats: shape (n_layers, n_sites, hidden_dim)
    # We don't know hidden_dim until forward; allocate lazily.
    per_site_running_sum: Dict[tuple[int, str], torch.Tensor] = {}
    per_site_running_max: Dict[tuple[int, str], torch.Tensor] = {}
    per_site_running_count: Dict[tuple[int, str], int] = {}

    def _make_input_hook(li: int, site: str):
        def _hook(_m, inp, _out):
            x = inp[0] if isinstance(inp, tuple) else inp
            if x is None:
                return
            # x is (batch, seq, hidden). Reduce over batch+seq → (hidden,)
            x_abs = x.detach().abs().float()
            x_sum = x_abs.sum(dim=tuple(range(x_abs.ndim - 1)))
            x_max = x_abs.amax(dim=tuple(range(x_abs.ndim - 1)))
            n = int(x_abs.shape[0] * x_abs.shape[1]) if x_abs.ndim >= 2 else 1
            key = (li, site)
            if key not in per_site_running_sum:
                per_site_running_sum[key] = x_sum.cpu()
                per_site_running_max[key] = x_max.cpu()
                per_site_running_count[key] = n
            else:
                per_site_running_sum[key] += x_sum.cpu()
                per_site_running_max[key] = torch.maximum(
                    per_site_running_max[key], x_max.cpu())
                per_site_running_count[key] += n
        return _hook

    # Wire up per-site hooks across all layers
    for li, lyr in enumerate(model.model.layers):
        # Self-attention: q_proj, k_proj, v_proj, o_proj
        if hasattr(lyr, "self_attn"):
            for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                m = getattr(lyr.self_attn, nm, None)
                if m is not None:
                    m.register_forward_pre_hook(_make_input_hook(li, nm))
        # MLP: gate_proj, up_proj, down_proj (dense models)
        # OR fused experts (MoE) — different path, skip per-site for MoE
        if hasattr(lyr, "mlp"):
            for nm in ("gate_proj", "up_proj", "down_proj"):
                m = getattr(lyr.mlp, nm, None)
                if m is not None:
                    m.register_forward_pre_hook(_make_input_hook(li, nm))

    # ── Resume + stream dataset ─────────────────────────────────────────
    existing = sorted(args.out.glob("shard_*.parquet"))
    shard_idx = len(existing)
    yielded = shard_idx * args.shard_size
    print(f"[mega-cal] resume: {yielded} seqs done, starting shard {shard_idx}",
          flush=True)
    if yielded >= args.max_sequences:
        print("[mega-cal] already complete", flush=True)
        return 0

    print(f"[mega-cal] streaming {args.dataset}[{args.split}]", flush=True)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    ds_iter = iter(ds)
    for _ in range(yielded):
        next(ds_iter, None)

    pbar = tqdm(total=args.max_sequences, initial=yielded, desc="seqs",
                file=sys.stderr, mininterval=2.0)
    buf: list[dict] = []

    while yielded < args.max_sequences:
        texts: list[str] = []
        for _ in range(args.batch_size):
            try:
                ex = next(ds_iter)
            except StopIteration:
                break
            content = ex.get("messages")
            if isinstance(content, list):
                content = " ".join(m.get("content", "") for m in content
                                   if isinstance(m, dict))
            elif not isinstance(content, str):
                content = str(content or "")
            if content.strip():
                texts.append(content)
        if not texts:
            break

        enc = tok(texts, return_tensors="pt", truncation=True,
                  max_length=args.max_tokens, padding=True)
        ids = enc["input_ids"].to("cuda")
        attn = enc["attention_mask"].to("cuda")

        with torch.no_grad():
            layer_captures.clear()
            outputs = model(input_ids=ids, attention_mask=attn)
            logits = outputs.logits  # (B, S, vocab)

        # Normalize layer captures shape
        B, S = ids.shape[0], ids.shape[1]
        for key in list(layer_captures.keys()):
            t = layer_captures[key]
            if t.dim() == 2 and t.shape[0] == B * S:
                layer_captures[key] = t.reshape(B, S, t.shape[-1])

        # Compute top-k logits per token
        topk_vals, topk_ids = torch.topk(logits.float().softmax(dim=-1),
                                         k=args.topk_logits, dim=-1)
        topk_vals_np = topk_vals.cpu().numpy().astype(np.float16)
        topk_ids_np = topk_ids.cpu().numpy().astype(np.int32)

        # Build per-row samples
        for b in range(B):
            real_len = int(attn[b].sum().item())
            tokens = ids[b, :real_len].cpu().numpy().astype(np.int32)
            sample = {"tokens": tokens.tobytes(), "n_tokens": int(real_len)}
            for key in ("residual", "intermediate"):
                if key not in layer_captures:
                    continue
                arr = layer_captures[key][b, :real_len, :].float().cpu().numpy()
                q, scale = quantize_int8(arr)
                sample[f"{key}_q"] = q.tobytes()
                sample[f"{key}_scale"] = scale
                sample[f"{key}_shape"] = list(arr.shape)
            # Top-k logits per position
            sample["topk_ids"] = topk_ids_np[b, :real_len, :].tobytes()
            sample["topk_probs"] = topk_vals_np[b, :real_len, :].tobytes()
            sample["topk_shape"] = [int(real_len), int(args.topk_logits)]
            buf.append(sample)
            yielded += 1
            pbar.update(1)

            if len(buf) >= args.shard_size:
                _flush(buf, args.out, shard_idx, pa, pq)
                buf = []
                shard_idx += 1

        del ids, attn, outputs, logits, topk_vals, topk_ids
        layer_captures.clear()
        torch.cuda.empty_cache()
        gc.collect()

    if buf:
        _flush(buf, args.out, shard_idx, pa, pq)
        shard_idx += 1
    pbar.close()

    # ── Save per-site per-layer activation aggregates ────────────────────
    # This is what AWQ / SmoothQuant / per-channel W4A8 calibration needs.
    aggregates_path = args.out / "per_site_activation_stats.npz"
    save_dict: Dict[str, np.ndarray] = {"n_layers": np.array(n_layers)}
    for (li, site), s in per_site_running_sum.items():
        n = per_site_running_count[(li, site)]
        mean_abs = (s / max(n, 1)).numpy().astype(np.float32)
        max_abs = per_site_running_max[(li, site)].numpy().astype(np.float32)
        save_dict[f"layer_{li}_{site}_mean_abs"] = mean_abs
        save_dict[f"layer_{li}_{site}_max_abs"] = max_abs
    np.savez_compressed(aggregates_path, **save_dict)
    print(f"[mega-cal] saved {len(save_dict)} per-site stats → {aggregates_path}",
          flush=True)

    # Manifest
    manifest = {
        "model": args.model,
        "capture_layer": args.capture_layer,
        "n_layers": n_layers,
        "max_sequences": args.max_sequences,
        "yielded": yielded,
        "shards": shard_idx,
        "shard_size": args.shard_size,
        "max_tokens": args.max_tokens,
        "topk_logits": args.topk_logits,
        "sites": list(SITE_NAMES),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[mega-cal] done. {yielded} sequences in {shard_idx} shards", flush=True)
    print(f"[mega-cal] outputs:", flush=True)
    print(f"  per-prompt parquet shards (Eagle5 + logits): {args.out}/shard_*.parquet", flush=True)
    print(f"  per-site activation stats (AWQ/W4A8 cal):    {aggregates_path}", flush=True)
    print(f"  manifest:                                    {args.out}/manifest.json", flush=True)
    return 0


def _flush(buf: list[dict], out: Path, shard_idx: int, pa, pq) -> None:
    final = out / f"shard_{shard_idx:04d}.parquet"
    tmp = out / f"shard_{shard_idx:04d}.parquet.tmp"
    table = pa.Table.from_pylist(buf)
    pq.write_table(table, str(tmp), compression="zstd")
    os.replace(tmp, final)


if __name__ == "__main__":
    sys.exit(main())
