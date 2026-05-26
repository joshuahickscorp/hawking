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
import argparse, gc, json, math, os, shutil, sys, time
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


def _shard_path(root: Path, shard_idx: int) -> Path:
    return root / f"shard_{shard_idx:04d}.parquet"


def _contiguous_resume(
    out: Path,
    shard_size: int,
    sync_dir: Path | None = None,
) -> tuple[int, int]:
    shard_idx = 0
    while _shard_path(out, shard_idx).exists() or (
        sync_dir is not None and _shard_path(sync_dir, shard_idx).exists()
    ):
        shard_idx += 1
    all_names = {p.name for p in out.glob("shard_*.parquet")}
    if sync_dir is not None:
        all_names.update(p.name for p in sync_dir.glob("shard_*.parquet"))
    if len(all_names) != shard_idx:
        print(
            f"[mega-cal] WARN: found {len(all_names)} shards but only "
            f"{shard_idx} are contiguous from shard_0000; resuming at "
            f"shard_{shard_idx:04d}",
            flush=True,
        )
    return shard_idx, shard_idx * shard_size


def _sync_outputs(out: Path, sync_dir: Path, delete_local_shards: bool) -> tuple[int, float]:
    sync_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    copied_bytes = 0
    for src in sorted(out.glob("*")):
        if not src.is_file():
            continue
        if src.suffix == ".tmp" or src.name.endswith(".tmp"):
            continue
        dst = sync_dir / src.name
        src_size = src.stat().st_size
        if not dst.exists() or dst.stat().st_size != src_size:
            shutil.copy2(src, dst)
            copied += 1
            copied_bytes += src_size
        if (
            delete_local_shards
            and src.name.startswith("shard_")
            and src.suffix == ".parquet"
            and dst.exists()
            and dst.stat().st_size == src_size
        ):
            src.unlink()
    return copied, copied_bytes / 1e9


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _stats_path(out: Path, sync_dir: Path | None = None) -> Path:
    local = out / "per_site_activation_stats.npz"
    if local.exists() or sync_dir is None:
        return local
    return sync_dir / "per_site_activation_stats.npz"


def _save_site_stats(
    path: Path,
    *,
    n_layers: int,
    sequences_written: int,
    sites: tuple[str, ...],
    sums: Dict[tuple[int, str], torch.Tensor],
    maxes: Dict[tuple[int, str], torch.Tensor],
    counts: Dict[tuple[int, str], int],
) -> None:
    save_dict: Dict[str, np.ndarray] = {
        "n_layers": np.array(n_layers, dtype=np.int32),
        "sequences_written": np.array(sequences_written, dtype=np.int64),
    }
    for (li, site), s in sums.items():
        n = counts[(li, site)]
        max_abs = maxes[(li, site)].numpy().astype(np.float32)
        sum_abs = s.numpy().astype(np.float64)
        mean_abs = (sum_abs / max(n, 1)).astype(np.float32)
        stem = f"layer_{li}_{site}"
        save_dict[f"{stem}_sum_abs"] = sum_abs
        save_dict[f"{stem}_count"] = np.array(n, dtype=np.int64)
        save_dict[f"{stem}_mean_abs"] = mean_abs
        save_dict[f"{stem}_max_abs"] = max_abs
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **save_dict)
    os.replace(tmp, path)


def _load_site_stats(
    path: Path,
) -> tuple[int, Dict[tuple[int, str], torch.Tensor], Dict[tuple[int, str], torch.Tensor], Dict[tuple[int, str], int]]:
    sums: Dict[tuple[int, str], torch.Tensor] = {}
    maxes: Dict[tuple[int, str], torch.Tensor] = {}
    counts: Dict[tuple[int, str], int] = {}
    if not path.exists():
        return 0, sums, maxes, counts
    with np.load(path) as z:
        sequences_written = (
            int(np.asarray(z["sequences_written"]).item())
            if "sequences_written" in z.files
            else 0
        )
        for key in z.files:
            if not key.startswith("layer_") or not key.endswith("_sum_abs"):
                continue
            parts = key.split("_")
            if len(parts) < 4:
                continue
            li = int(parts[1])
            site = "_".join(parts[2:-2])
            stem = f"layer_{li}_{site}"
            count_key = f"{stem}_count"
            max_key = f"{stem}_max_abs"
            if count_key not in z or max_key not in z:
                continue
            sums[(li, site)] = torch.from_numpy(np.asarray(z[key], dtype=np.float64))
            maxes[(li, site)] = torch.from_numpy(np.asarray(z[max_key], dtype=np.float32))
            counts[(li, site)] = int(np.asarray(z[count_key]).item())
    return sequences_written, sums, maxes, counts


def _load_best_site_stats(
    out: Path,
    sync_dir: Path | None,
) -> tuple[int, Dict[tuple[int, str], torch.Tensor], Dict[tuple[int, str], torch.Tensor], Dict[tuple[int, str], int], Path]:
    candidates = [out / "per_site_activation_stats.npz"]
    if sync_dir is not None:
        candidates.append(sync_dir / "per_site_activation_stats.npz")

    best_path = candidates[0]
    best_seq = 0
    best_sum: Dict[tuple[int, str], torch.Tensor] = {}
    best_max: Dict[tuple[int, str], torch.Tensor] = {}
    best_count: Dict[tuple[int, str], int] = {}
    for candidate in candidates:
        seq, sums, maxes, counts = _load_site_stats(candidate)
        if sums and seq >= best_seq:
            best_path = candidate
            best_seq = seq
            best_sum = sums
            best_max = maxes
            best_count = counts
    return best_seq, best_sum, best_max, best_count, best_path


def _flush(buf: list[dict], out: Path, shard_idx: int, pa, pq) -> Path:
    final = _shard_path(out, shard_idx)
    tmp = out / f"shard_{shard_idx:04d}.parquet.tmp"
    table = pa.Table.from_pylist(buf)
    pq.write_table(table, str(tmp), compression="zstd")
    os.replace(tmp, final)
    return final


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
    p.add_argument("--lm-head-chunk-tokens", type=int, default=128,
                   help="Token positions per lm_head chunk. Smaller is safer "
                        "for VRAM; larger is faster on high-memory GPUs.")
    p.add_argument("--shard-size", type=int, default=16)
    p.add_argument("--load-4bit", action="store_true",
                   help="Load model in 4-bit nf4 via bitsandbytes. Required "
                        "on T4/V100/L4 (sub-32 GB VRAM). On A100/H100 use "
                        "native fp16 instead.")
    p.add_argument("--sync-dir", type=Path,
                   help="Optional durable copy destination, e.g. Google Drive. "
                        "Existing shards there count for resume.")
    p.add_argument("--sync-every", type=int, default=4,
                   help="Copy local outputs to --sync-dir every N new shards.")
    p.add_argument("--delete-local-after-sync", action="store_true",
                   help="After verifying shard copies in --sync-dir, delete "
                        "local shard files to preserve Colab SSD.")
    p.add_argument("--stats-every-shards", type=int, default=1,
                   help="Persist per-site activation stats every N shards. "
                        "Default 1 minimizes disconnect loss.")
    p.add_argument("--allow-missing-stats-resume", action="store_true",
                   help="Allow resuming from existing shards without a matching "
                        "per_site_activation_stats.npz. Unsafe for AWQ/W4A8.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.sync_every = max(1, args.sync_every)
    args.stats_every_shards = max(1, args.stats_every_shards)
    args.lm_head_chunk_tokens = max(1, args.lm_head_chunk_tokens)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.sync_dir is not None:
        args.sync_dir.mkdir(parents=True, exist_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"[mega-cal] model={args.model}", flush=True)
    print(f"[mega-cal] mode={'4-bit nf4' if args.load_4bit else 'native fp16'}",
          flush=True)
    print(
        f"[mega-cal] topk={args.topk_logits}, lm_head_chunk_tokens="
        f"{args.lm_head_chunk_tokens}",
        flush=True,
    )
    print(
        f"[mega-cal] local disk free: {shutil.disk_usage(args.out).free / 1e9:.1f} GB",
        flush=True,
    )
    if args.sync_dir is not None:
        print(
            f"[mega-cal] durable sync: {args.sync_dir} every {args.sync_every} "
            f"shard(s){' with local shard deletion' if args.delete_local_after_sync else ''}",
            flush=True,
        )
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # transformers 5.0 renamed `torch_dtype` → `dtype` on from_pretrained.
    # Probe the installed version and use the right kwarg name so the same
    # script works on both 4.x (Colab "stable") and 5.x (latest Colab images).
    import transformers as _hf
    _hf_major = int(str(_hf.__version__).split(".", 1)[0])
    _dtype_kw = "dtype" if _hf_major >= 5 else "torch_dtype"

    model_kwargs = dict(
        trust_remote_code=False,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model_kwargs[_dtype_kw] = torch.float16
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
        del model_kwargs[_dtype_kw]
    else:
        model_kwargs["device_map"] = "cuda"

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    print(f"[mega-cal] loaded in {time.time() - t0:.1f}s", flush=True)

    # ── Layer-K residual + intermediate (Eagle5 head training) ───────────
    L = args.capture_layer
    backbone = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if backbone is None:
        raise RuntimeError("model has no .model backbone; cannot chunk lm_head safely")
    if lm_head is None:
        raise RuntimeError("model has no .lm_head; cannot produce quality top-k")
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise RuntimeError("model backbone has no .layers; cannot attach capture hooks")
    if L < 0 or L >= len(layers):
        raise ValueError(f"--capture-layer {L} out of range for {len(layers)} layers")
    layer = layers[L]
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
    n_layers = len(layers)
    # Per-channel running stats: shape (n_layers, n_sites, hidden_dim)
    # We don't know hidden_dim until forward; allocate lazily.
    per_site_running_sum: Dict[tuple[int, str], torch.Tensor] = {}
    per_site_running_max: Dict[tuple[int, str], torch.Tensor] = {}
    per_site_running_count: Dict[tuple[int, str], int] = {}
    pending_site_stats: list[tuple[int, str, torch.Tensor, torch.Tensor, int]] = []
    registered_site_hooks = 0

    def _make_input_hook(li: int, site: str):
        def _hook(_m, inp, _out=None):
            x = inp[0] if isinstance(inp, tuple) else inp
            if x is None:
                return
            # x is (batch, seq, hidden). Reduce over batch+seq → (hidden,)
            with torch.no_grad():
                x_abs = x.detach().abs().float()
                reduce_dims = tuple(range(x_abs.ndim - 1))
                x_sum = x_abs.sum(dim=reduce_dims).detach()
                x_max = x_abs.amax(dim=reduce_dims).detach()
                n = int(math.prod(x_abs.shape[:-1])) if x_abs.ndim >= 2 else 1
                pending_site_stats.append((li, site, x_sum, x_max, n))
        return _hook

    # Wire up per-site hooks across all layers
    for li, lyr in enumerate(layers):
        # Self-attention: q_proj, k_proj, v_proj, o_proj
        if hasattr(lyr, "self_attn"):
            for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                m = getattr(lyr.self_attn, nm, None)
                if m is not None:
                    m.register_forward_pre_hook(_make_input_hook(li, nm))
                    registered_site_hooks += 1
        # MLP: gate_proj, up_proj, down_proj (dense models)
        # OR fused experts (MoE) — different path, skip per-site for MoE
        if hasattr(lyr, "mlp"):
            for nm in ("gate_proj", "up_proj", "down_proj"):
                m = getattr(lyr.mlp, nm, None)
                if m is not None:
                    m.register_forward_pre_hook(_make_input_hook(li, nm))
                    registered_site_hooks += 1
    if registered_site_hooks == 0:
        raise RuntimeError("no AWQ/W4A8 input hooks registered; unsupported model layout")
    print(
        f"[mega-cal] registered {registered_site_hooks} AWQ/W4A8 input hooks",
        flush=True,
    )

    # ── Resume + stream dataset ─────────────────────────────────────────
    shard_idx, yielded = _contiguous_resume(args.out, args.shard_size, args.sync_dir)
    stats_seen, loaded_sum, loaded_max, loaded_count, stats_loaded_from = _load_best_site_stats(
        args.out, args.sync_dir
    )
    if loaded_sum:
        per_site_running_sum.update(loaded_sum)
        per_site_running_max.update(loaded_max)
        per_site_running_count.update(loaded_count)
        print(
            f"[mega-cal] loaded activation stats through {stats_seen} seqs "
            f"from {stats_loaded_from}",
            flush=True,
        )
    elif yielded > 0 and not args.allow_missing_stats_resume:
        raise RuntimeError(
            f"found {yielded} completed seqs but no resumable "
            f"per_site_activation_stats.npz. Re-run from an empty output dir, "
            f"restore stats from Drive, or pass --allow-missing-stats-resume "
            f"if you only need parquet shards."
        )
    if loaded_sum and stats_seen < yielded and not args.allow_missing_stats_resume:
        raise RuntimeError(
            f"activation stats only cover {stats_seen} seqs but shards cover "
            f"{yielded}. To keep AWQ/W4A8 stats correct, restore matching stats "
            f"or restart from the first shard after {stats_seen} seqs."
        )
    if loaded_sum and stats_seen > yielded:
        print(
            f"[mega-cal] WARN: activation stats cover {stats_seen} seqs but "
            f"contiguous shards cover {yielded}; continuing from shard state",
            flush=True,
        )
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
    first_batch = True
    shards_since_sync = 0

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
        B, S = ids.shape[0], ids.shape[1]

        if first_batch:
            print(
                f"[mega-cal] first forward: batch={B}, padded_tokens={S}, "
                f"target_seqs={args.max_sequences}",
                flush=True,
            )

        with torch.inference_mode():
            layer_captures.clear()
            pending_site_stats.clear()
            outputs = backbone(
                input_ids=ids,
                attention_mask=attn,
                use_cache=False,
                return_dict=True,
            )
            hidden = (
                outputs.last_hidden_state
                if hasattr(outputs, "last_hidden_state")
                else outputs[0]
            )
            del outputs

        # Normalize layer captures shape
        for key in list(layer_captures.keys()):
            t = layer_captures[key]
            if t.dim() == 2 and t.shape[0] == B * S:
                layer_captures[key] = t.reshape(B, S, t.shape[-1])
            elif t.dim() == 3:
                pass
            else:
                if yielded == 0:
                    print(
                        f"[mega-cal] WARN: dropping '{key}' with shape "
                        f"{tuple(t.shape)} (expected 3D or flat 2D)",
                        flush=True,
                    )
                del layer_captures[key]

        missing = {"residual", "intermediate"} - set(layer_captures)
        if missing:
            raise RuntimeError(
                f"capture hooks missing required tensors: {sorted(missing)}; "
                f"got {sorted(layer_captures)}"
            )

        batch_site_hook_count = len(pending_site_stats)
        if batch_site_hook_count == 0:
            raise RuntimeError(
                "AWQ/W4A8 input hooks registered but did not fire during the "
                "forward pass; aborting before writing incomplete stats"
            )

        for li, site, x_sum, x_max, n in pending_site_stats:
            key = (li, site)
            x_sum_cpu = x_sum.cpu()
            x_max_cpu = x_max.cpu()
            if key not in per_site_running_sum:
                per_site_running_sum[key] = x_sum_cpu
                per_site_running_max[key] = x_max_cpu
                per_site_running_count[key] = n
            else:
                per_site_running_sum[key] += x_sum_cpu
                per_site_running_max[key] = torch.maximum(
                    per_site_running_max[key], x_max_cpu)
                per_site_running_count[key] += n
        pending_site_stats.clear()

        capture_shapes = {k: tuple(v.shape) for k, v in sorted(layer_captures.items())}
        layer_captures_np = {
            k: v.detach().cpu().float().numpy()
            for k, v in layer_captures.items()
        }
        layer_captures.clear()
        torch.cuda.empty_cache()

        # Compute top-k probabilities in LM-head chunks instead of materializing
        # the full (B, S, vocab) softmax tensor.
        flat_hidden = hidden.reshape(B * S, hidden.shape[-1])
        topk_ids_parts: list[np.ndarray] = []
        topk_vals_parts: list[np.ndarray] = []
        chunk = max(1, int(args.lm_head_chunk_tokens))
        lm_head_param = next(lm_head.parameters(), None)
        lm_head_device = (
            lm_head_param.device
            if lm_head_param is not None and lm_head_param.device.type != "meta"
            else flat_hidden.device
        )
        with torch.inference_mode():
            for start in range(0, flat_hidden.shape[0], chunk):
                h = flat_hidden[start:start + chunk]
                if h.device != lm_head_device:
                    h = h.to(lm_head_device)
                logits = lm_head(h)
                logits_f = logits.float()
                vals, ids_top = torch.topk(logits_f, k=args.topk_logits, dim=-1)
                probs = (vals - torch.logsumexp(logits_f, dim=-1, keepdim=True)).exp()
                topk_ids_parts.append(ids_top.cpu().numpy().astype(np.int32))
                topk_vals_parts.append(probs.cpu().numpy().astype(np.float16))
                del h, logits, logits_f, vals, ids_top, probs
        topk_ids_np = np.concatenate(topk_ids_parts, axis=0).reshape(B, S, args.topk_logits)
        topk_vals_np = np.concatenate(topk_vals_parts, axis=0).reshape(B, S, args.topk_logits)
        del hidden, flat_hidden, topk_ids_parts, topk_vals_parts

        # Build per-row samples
        for b in range(B):
            real_len = int(attn[b].sum().item())
            tokens = ids[b, :real_len].cpu().numpy().astype(np.int32)
            sample = {"tokens": tokens.tobytes(), "n_tokens": int(real_len)}
            for key in ("residual", "intermediate"):
                arr = layer_captures_np[key][b, :real_len, :]
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
                shards_since_sync += 1
                if shard_idx % args.stats_every_shards == 0:
                    _save_site_stats(
                        args.out / "per_site_activation_stats.npz",
                        n_layers=n_layers,
                        sequences_written=yielded,
                        sites=SITE_NAMES,
                        sums=per_site_running_sum,
                        maxes=per_site_running_max,
                        counts=per_site_running_count,
                    )
                if args.sync_dir is not None and shards_since_sync >= args.sync_every:
                    copied, copied_gb = _sync_outputs(
                        args.out, args.sync_dir, args.delete_local_after_sync
                    )
                    if copied:
                        print(
                            f"[mega-cal] synced {copied} file(s), {copied_gb:.2f} GB "
                            f"→ {args.sync_dir}",
                            flush=True,
                        )
                    shards_since_sync = 0

        if first_batch:
            print(
                "[mega-cal] first batch OK — captures: "
                + ", ".join(f"{k}{capture_shapes[k]}" for k in sorted(capture_shapes))
                + f"; site hooks fired={batch_site_hook_count}",
                flush=True,
            )
            first_batch = False

        del layer_captures_np, ids, attn, topk_vals_np, topk_ids_np
        torch.cuda.empty_cache()
        gc.collect()

    if buf:
        _flush(buf, args.out, shard_idx, pa, pq)
        shard_idx += 1
    pbar.close()

    # ── Save per-site per-layer activation aggregates ────────────────────
    # This is what AWQ / SmoothQuant / per-channel W4A8 calibration needs.
    aggregates_path = args.out / "per_site_activation_stats.npz"
    _save_site_stats(
        aggregates_path,
        n_layers=n_layers,
        sequences_written=yielded,
        sites=SITE_NAMES,
        sums=per_site_running_sum,
        maxes=per_site_running_max,
        counts=per_site_running_count,
    )
    print(f"[mega-cal] saved per-site stats → {aggregates_path}",
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
        "lm_head_chunk_tokens": args.lm_head_chunk_tokens,
        "sites": list(SITE_NAMES),
        "sync_dir": str(args.sync_dir) if args.sync_dir is not None else None,
    }
    _write_json_atomic(args.out / "manifest.json", manifest)
    if args.sync_dir is not None:
        copied, copied_gb = _sync_outputs(
            args.out, args.sync_dir, args.delete_local_after_sync
        )
        if copied:
            print(
                f"[mega-cal] final sync: {copied} file(s), {copied_gb:.2f} GB "
                f"→ {args.sync_dir}",
                flush=True,
            )

    print(f"[mega-cal] done. {yielded} sequences in {shard_idx} shards", flush=True)
    print(f"[mega-cal] outputs:", flush=True)
    print(f"  per-prompt parquet shards (Eagle5 + logits): {args.out}/shard_*.parquet", flush=True)
    print(f"  per-site activation stats (AWQ/W4A8 cal):    {aggregates_path}", flush=True)
    print(f"  manifest:                                    {args.out}/manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
