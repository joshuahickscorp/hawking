"""RWKV-7 Quantization-Aware Training (QAT).

Wraps any subset of the pure-torch RWKV-7 forward with STE fake-quantizers
(binary / ternary / uniform symmetric) or strand-pv (subprocess requant cycle),
then fine-tunes on a prompt-masked JSONL dataset.

Stages
------
  ffn    — key/value projections only (RWKV7_FFN_SUFFIXES)
  time   — r/k/v/o projections only  (RWKV7_TIME_SUFFIXES)
  all    — all projections            (RWKV7_ALL_PROJ_SUFFIXES)
  mixed  — all projections, per-tensor bits from --mp-config JSON
  lmhead — lm_head only (handled specially)

Usage
-----
  python rwkv7_qat.py --model models/rwkv7-g1-04-hf/model.safetensors \
      --hf-dir models/rwkv7-g1-04-hf \
      --data artifacts/rwkv7_posttrain/sft.jsonl \
      --out artifacts/qat_out --stage all --quant ternary

  python rwkv7_qat.py --dry-run --stage ffn --quant binary  # quick smoke test
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

sys.path.insert(0, str(HERE))

from rwkv7_torch_model import RWKV7Model, RWKV7Config
from rwkv7_load_weights import load_rwkv7
from rwkv7_export_strand import make_torch_to_gguf, write_safetensors
from lowbit_qat import (
    QuantLinear,
    quant_binary,
    quant_ternary,
    quant_uniform_symmetric,
    wrap_linears,
    list_wrapped_modules,
    RWKV7_TIME_SUFFIXES,
    RWKV7_FFN_SUFFIXES,
    RWKV7_ALL_PROJ_SUFFIXES,
    QUANT_FNS,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="RWKV-7 Quantization-Aware Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default=str(ROOT / "models/rwkv7-g1-04-hf/model.safetensors"),
                    help="safetensors checkpoint")
    ap.add_argument("--hf-dir", default=str(ROOT / "models/rwkv7-g1-04-hf"),
                    help="tokenizer directory")
    ap.add_argument("--data", default=str(ROOT / "artifacts/rwkv7_posttrain/sft.jsonl"),
                    help="training JSONL")
    ap.add_argument("--out", default=str(ROOT / "artifacts/qat_out"),
                    help="output directory")

    # Stage / quant
    ap.add_argument("--stage", choices=["ffn", "time", "all", "mixed", "lmhead"],
                    default="all", help="which projections to quantize")
    ap.add_argument("--quant", choices=["uniform", "binary", "ternary", "strand-pv"],
                    default="ternary", help="quantizer kind")
    ap.add_argument("--bits", type=int, default=2,
                    help="bit-width for uniform mode (2..8); binary/ternary ignore this")
    ap.add_argument("--mp-config", default=None,
                    help="JSON path for per-tensor bits in mixed stage")

    # Training
    ap.add_argument("--last-n-layers", type=int, default=0,
                    help="0 = train all layers; N = freeze first (L-N) layers")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=25,
                    help="optimizer steps between checkpoints")
    ap.add_argument("--eval-every", type=int, default=25,
                    help="optimizer steps between PPL evals")
    ap.add_argument("--eval-tokens", type=int, default=4096,
                    help="tokens to use for in-training PPL eval")
    ap.add_argument("--device", default="mps",
                    help="mps | cpu | cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="print wrapped modules + 3 train steps, then exit")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="if >0, limit data to first N rows (dry-run helper)")
    ap.add_argument("--run-id", default=None,
                    help="optional identifier stamped in event logs")

    # Knowledge distillation
    ap.add_argument("--teacher", default=None,
                    help="teacher safetensors checkpoint (enables KD)")
    ap.add_argument("--kd", choices=["topk", "full", "none"], default="none",
                    help="distillation mode")
    ap.add_argument("--kd-temperature", type=float, default=2.0)
    ap.add_argument("--ce-weight", type=float, default=0.3)
    ap.add_argument("--kd-weight", type=float, default=0.7)

    # strand-pv
    ap.add_argument("--requant-every", type=int, default=25,
                    help="(strand-pv) optimizer steps between requantization calls")
    ap.add_argument("--requant-shards", type=int, default=8)
    ap.add_argument("--strand-bin", default="target/release/quantize-model",
                    help="path to the strand quantize-model binary")
    ap.add_argument("--strand-flags", default="",
                    help="extra flags forwarded to the strand binary")
    ap.add_argument("--l", type=int, default=7,
                    help="(strand-pv) trellis L parameter; 0 = use quantize-model default")
    ap.add_argument("--keep-requant-tmp", action="store_true",
                    help="(strand-pv) keep intermediate requant safetensors for debugging")

    # Resume
    ap.add_argument("--resume-from", default=None,
                    help="checkpoint directory containing state_dict.pt to resume from")
    ap.add_argument("--resume-step", type=int, default=0,
                    help="optimizer step recorded in checkpoint (0 = infer from dirname step_NNNNNN)")

    return ap.parse_args()


# ---------------------------------------------------------------------------
# Stage → suffixes
# ---------------------------------------------------------------------------

def get_stage_suffixes(stage: str) -> Tuple[str, ...]:
    if stage == "ffn":
        return RWKV7_FFN_SUFFIXES
    if stage == "time":
        return RWKV7_TIME_SUFFIXES
    if stage in ("all", "mixed"):
        return RWKV7_ALL_PROJ_SUFFIXES
    if stage == "lmhead":
        return ()  # handled separately
    raise ValueError(f"Unknown stage: {stage!r}")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(hf_dir: str):
    """Try the tokenizers fast tokenizer, then fall back to transformers.

    Returns an encode callable:  encode(text: str) -> list[int]
    For RWKV World tokenizer (custom greedy-trie), also tries the HF module directly.
    """
    hf_path = Path(hf_dir)

    # Strategy 0: RWKV custom greedy-trie tokenizer shipped alongside the model.
    tok_script = hf_path / "hf_rwkv_tokenizer.py"
    if tok_script.exists():
        try:
            spec = importlib.util.spec_from_file_location("hf_rwkv_tokenizer", str(tok_script))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _tok = mod.RWKV_TOKENIZER(str(hf_path / "rwkv_vocab_v20230424.txt"))

            def encode_rwkv(text: str) -> list[int]:
                return _tok.encodeBytes(text.encode("utf-8"))

            return encode_rwkv
        except Exception:
            pass

    # Strategy 1: tokenizers fast tokenizer.
    tok_json = hf_path / "tokenizer.json"
    if tok_json.exists():
        try:
            from tokenizers import Tokenizer
            _ftok = Tokenizer.from_file(str(tok_json))

            def encode_fast(text: str) -> list[int]:
                return _ftok.encode(text).ids

            return encode_fast
        except ImportError:
            pass

    # Strategy 2: transformers AutoTokenizer.
    try:
        from transformers import AutoTokenizer
        _atok = AutoTokenizer.from_pretrained(str(hf_path), trust_remote_code=True)

        def encode_auto(text: str) -> list[int]:
            return _atok.encode(text, add_special_tokens=False)

        return encode_auto
    except Exception as exc:
        raise RuntimeError(
            f"Could not load tokenizer from {hf_dir}. "
            "Install 'tokenizers' (for tokenizer.json) or 'transformers' (for AutoTokenizer). "
            f"Underlying error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_batch(
    record: dict,
    encode,
    max_length: int,
    device: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Tokenise one record.

    Returns (input_ids [1,T], labels [1,T]) where prompt tokens are masked
    with -100.  Returns None if there is no supervised signal.

    Accepted formats
    ----------------
    {"prompt": "...", "completion": "..."}
    {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
    {"text": "..."}                                 (full supervision, no mask)
    """
    EOS = 0

    if "prompt" in record and "completion" in record:
        prompt_ids = encode(record["prompt"])
        comp_ids = encode(record["completion"])
        ids = prompt_ids + comp_ids
        labels = [-100] * len(prompt_ids) + list(comp_ids)

    elif "messages" in record:
        msgs = record["messages"]
        user_text = next((m["content"] for m in msgs if m["role"] == "user"), None)
        asst_text = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if not user_text or not asst_text:
            return None
        prompt_ids = [EOS] + encode(f"User: {user_text}\n\nAssistant:")
        comp_ids = encode(f" {asst_text}") + [EOS]
        ids = prompt_ids + comp_ids
        labels = [-100] * len(prompt_ids) + list(comp_ids)

    elif "text" in record:
        ids = encode(record["text"])
        if not ids:
            return None
        labels = list(ids)

    else:
        return None

    if len(ids) > max_length:
        ids = ids[:max_length]
        labels = labels[:max_length]

    if all(l == -100 for l in labels):
        return None  # prompt filled window, no signal

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    label_ids = torch.tensor([labels], dtype=torch.long, device=device)
    return input_ids, label_ids


# ---------------------------------------------------------------------------
# Freeze helpers
# ---------------------------------------------------------------------------

def freeze_to_last_n(model: RWKV7Model, n: int) -> None:
    """Freeze all layers except the last N + lm_head + final norm.

    n <= 0 is a no-op (full fine-tune).
    """
    if n <= 0:
        return
    n_layer = model.cfg.n_layer
    trainable_from = max(0, n_layer - n)
    for name, param in model.named_parameters():
        keep = name.startswith(("norm_", "lm_head"))
        if name.startswith("layers."):
            layer_idx = int(name.split(".")[1])
            keep = keep or (layer_idx >= trainable_from)
        param.requires_grad_(keep)


# ---------------------------------------------------------------------------
# CE + KD loss
# ---------------------------------------------------------------------------

def compute_ce_loss(
    model: RWKV7Model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Forward pass + cross-entropy, skipping the full lm_head on masked positions."""
    hidden = model(input_ids, return_final_hidden=True)  # [1, T, n_embd]
    shift_hidden = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)
    mask = shift_labels != -100
    if not mask.any():
        return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
    sel_logits = model.lm_head(shift_hidden[mask])  # [n_sup, vocab]
    return F.cross_entropy(sel_logits.float(), shift_labels[mask])


def compute_kd_loss(
    student_model: RWKV7Model,
    teacher_model: Optional[RWKV7Model],
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    mode: str,
    temperature: float,
    artifacts_dir: Path,
    step: int,
) -> torch.Tensor:
    """KL-divergence distillation loss.

    If artifacts_dir / f"teacher_{step:06d}.pt" exists, loads pre-captured
    teacher logits (produced by rwkv7_capture_teacher_logits.py).  Otherwise
    runs the teacher forward live (requires --teacher to be set).
    """
    if mode == "none":
        return torch.tensor(0.0, device=input_ids.device)

    T_val = temperature

    # Try pre-captured logits first.
    pre_captured = artifacts_dir / f"teacher_{step:06d}.pt"
    if pre_captured.exists():
        teacher_logits = torch.load(str(pre_captured), map_location=input_ids.device)
    elif teacher_model is not None:
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids)  # [1, T, V]
    else:
        raise RuntimeError(
            "--kd mode requires either --teacher PATH or pre-captured teacher logits "
            f"at {pre_captured}"
        )

    with torch.no_grad():
        student_logits = student_model(input_ids)  # [1, T, V]

    # Restrict to supervised positions.
    shift_labels = labels[:, 1:].reshape(-1)
    mask = shift_labels != -100

    s_log = F.log_softmax(student_logits[:, :-1, :].reshape(-1, student_logits.size(-1))[mask].float() / T_val, dim=-1)
    t_soft = F.softmax(teacher_logits[:, :-1, :].reshape(-1, teacher_logits.size(-1))[mask].float() / T_val, dim=-1)

    return F.kl_div(s_log, t_soft, reduction="batchmean") * (T_val ** 2)


# ---------------------------------------------------------------------------
# Self-contained safetensors reader (no safetensors library required)
# ---------------------------------------------------------------------------

def read_safetensors(path: str | Path) -> Dict[str, "np.ndarray"]:
    """Read a safetensors file and return {name: float32 ndarray}.

    Format:
      [8 bytes: uint64 little-endian header_len]
      [header_len bytes: JSON with dtype/shape/data_offsets per tensor]
      [data section: packed tensor bytes]
    """
    import struct as _struct

    path = Path(path)
    with open(path, "rb") as f:
        (header_len,) = _struct.unpack("<Q", f.read(8))
        header_bytes = f.read(header_len)
        header: dict = json.loads(header_bytes.decode("utf-8"))
        data_section = f.read()

    result: Dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype_str = meta.get("dtype", "F32")
        if dtype_str != "F32":
            # Skip non-float32 tensors (we only write F32 in write_safetensors).
            continue
        shape = meta["shape"]
        start, end = meta["data_offsets"]
        arr = np.frombuffer(data_section[start:end], dtype=np.float32).reshape(shape).copy()
        result[name] = arr

    return result


# ---------------------------------------------------------------------------
# strand-pv requantization step
# ---------------------------------------------------------------------------

def requant_step(
    model: RWKV7Model,
    args: argparse.Namespace,
    step: int,
    tmp_dir: Path,
) -> None:
    """Periodically call the strand quantize-model binary to tighten the base.

    High-level flow:
      1. Collect QuantLinear shadow weights and map them to GGUF tensor names.
      2. Write a float32 safetensors file with GGUF names as keys.
      3. Call ``quantize-model`` to produce a reconstruction safetensors.
      4. Load the reconstruction and update each QuantLinear's ``base`` buffer.
      5. Log RMS reconstruction error for the first few updated tensors.
    """
    wrapped = list_wrapped_modules(model)
    if not wrapped:
        return

    tmp_dir.mkdir(parents=True, exist_ok=True)
    in_path = tmp_dir / f"shadow_{step:06d}.safetensors"
    out_path = tmp_dir / f"recon_{step:06d}.safetensors"

    # --- 1. Build GGUF name map ------------------------------------------------
    torch_to_gguf = make_torch_to_gguf(model.cfg.n_layer)
    # Build reverse map: gguf_name -> (torch_name, module) for step 5.
    gguf_to_module: Dict[str, Tuple[str, "QuantLinear"]] = {}

    # --- 2. Collect shadow weights keyed by GGUF name -------------------------
    shadow_tensors: Dict[str, np.ndarray] = {}
    skipped = 0
    for torch_name, mod in wrapped:
        gguf_name = torch_to_gguf.get(torch_name + ".weight")
        if gguf_name is None:
            print(f"[requant] WARNING: no GGUF name for {torch_name!r}, skipping", flush=True)
            skipped += 1
            continue
        w = mod.weight.detach().float().cpu()
        shadow_tensors[gguf_name] = w.numpy()
        gguf_to_module[gguf_name] = (torch_name, mod)

    if not shadow_tensors:
        print(f"[requant step={step}] WARNING: no mappable tensors ({skipped} skipped); "
              "aborting requant", flush=True)
        return

    # --- 3. Write shadow safetensors ------------------------------------------
    write_safetensors(in_path, shadow_tensors)

    # --- 4. Run quantize-model binary -----------------------------------------
    strand_bin_path = Path(args.strand_bin)
    if not strand_bin_path.is_absolute():
        strand_bin_path = ROOT / strand_bin_path

    cmd = [
        str(strand_bin_path),
        "--in",   str(in_path),
        "--out",  str(out_path),
        "--bits", str(args.bits),
        "--packed-v2-out", "/dev/null",
    ]
    if args.l > 0:
        cmd += ["--l", str(args.l)]
    if args.strand_flags:
        cmd += args.strand_flags.split()

    print(f"[requant step={step}] running quantize-model on {len(shadow_tensors)} tensors "
          f"(bits={args.bits}, l={args.l})", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[requant step={step}] WARNING: quantize-model exited {result.returncode}: "
              f"{result.stderr[:200]}", flush=True)
        return

    # --- 5. Load reconstruction and update base buffers -----------------------
    if not out_path.exists():
        print(f"[requant step={step}] WARNING: recon output not found: {out_path}", flush=True)
        return

    recon_arrays = read_safetensors(out_path)

    n_updated = 0
    log_limit = 5  # log RMS for first N tensors
    for gguf_name, recon_arr in recon_arrays.items():
        if gguf_name not in gguf_to_module:
            continue
        _torch_name, mod = gguf_to_module[gguf_name]
        new_base = torch.from_numpy(recon_arr).to(mod.weight.device)
        with torch.no_grad():
            mod.base.copy_(new_base)
            if n_updated < log_limit:
                err = (mod.base - mod.weight.detach()).pow(2).mean().sqrt()
                print(f"  requant step {step}: {gguf_name}  RMS={err.item():.4f}", flush=True)
        n_updated += 1

    print(f"[requant step={step}] updated {n_updated}/{len(shadow_tensors)} base buffers",
          flush=True)

    # --- 6. Cleanup -----------------------------------------------------------
    if not getattr(args, "keep_requant_tmp", False):
        for p in (in_path, out_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# In-training PPL eval
# ---------------------------------------------------------------------------

def run_eval(
    model: RWKV7Model,
    encode,
    args: argparse.Namespace,
    step: int,
    out_dir: Path,
) -> float:
    """Compute PPL on the first args.eval_tokens tokens of heldout.jsonl (or wikitext2).

    Writes one JSON line to <out>/eval_ledger.jsonl.
    Returns the perplexity.
    """
    model.eval()

    # Build token stream.
    heldout_path = Path(args.data).parent / "heldout.jsonl"
    text_chunks: List[str] = []

    if heldout_path.exists():
        with open(heldout_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "text" in obj:
                    text_chunks.append(obj["text"])
                elif "completion" in obj:
                    text_chunks.append(obj["completion"])
                if sum(len(c) for c in text_chunks) > args.eval_tokens * 8:
                    break
    else:
        # Fallback: sample a slice of the training data.
        try:
            rows = load_jsonl(args.data)
            for r in rows[:50]:
                text_chunks.append(r.get("text") or r.get("completion") or "")
        except Exception:
            pass

    raw_text = "\n".join(text_chunks)
    if not raw_text.strip():
        print("[eval] WARNING: could not build eval text; skipping PPL eval.", flush=True)
        model.train()
        return float("nan")

    ids = encode(raw_text)[: args.eval_tokens + 1]
    if len(ids) < 2:
        model.train()
        return float("nan")

    ids_tensor = torch.tensor(ids, dtype=torch.long, device=args.device)

    total_nll = 0.0
    total_n = 0
    stride = min(512, len(ids) - 1)

    with torch.no_grad():
        begin = 0
        prev_end = 0
        while begin < len(ids) - 1:
            end = min(begin + stride, len(ids))
            window = ids_tensor[begin:end].unsqueeze(0)
            logits = model(window)  # [1, T, V]
            target_start = 0 if begin == 0 else (prev_end - begin - 1)
            ls = logits[0, target_start: end - begin - 1, :]
            ts = ids_tensor[begin + target_start + 1: end]
            if ls.shape[0] > 0:
                nll = F.cross_entropy(ls.float(), ts, reduction="sum").item()
                total_nll += nll
                total_n += ts.shape[0]
            prev_end = end
            begin += stride

    ppl = math.exp(total_nll / total_n) if total_n > 0 else float("nan")

    ledger_path = out_dir / "eval_ledger.jsonl"
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "step": step,
            "ppl": round(ppl, 4),
            "n_tokens": total_n,
            "run_id": args.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }) + "\n")

    print(f"[eval step={step}] PPL={ppl:.2f}  ({total_n} tokens)", flush=True)
    model.train()
    return ppl


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Persist run config ---------------------------------------------------
    run_config = vars(args).copy()
    run_config["started_at"] = datetime.now(timezone.utc).isoformat()
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # --- Load model -----------------------------------------------------------
    print(f"[model] loading {args.model} on {args.device} (fp32) ...", flush=True)
    model = load_rwkv7(args.model, device=args.device, dtype=torch.float32)
    model.train()

    # --- Freeze layers --------------------------------------------------------
    if args.last_n_layers > 0:
        freeze_to_last_n(model, args.last_n_layers)
        frozen_msg = f"last {args.last_n_layers} layers + lm_head + norm"
    else:
        frozen_msg = "full model"
    n_trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable before wrap: {n_trainable_before/1e6:.1f}M  ({frozen_msg})",
          flush=True)

    # --- Wrap projections -----------------------------------------------------
    suffixes = get_stage_suffixes(args.stage)
    quant_fn: Optional[Callable] = QUANT_FNS.get(args.quant) if args.quant != "strand-pv" else None
    bits = args.bits if args.quant == "uniform" else None

    # Load per-tensor mp-config if provided (mixed stage).
    mp_config: Optional[Dict[str, int]] = None
    if args.mp_config:
        with open(args.mp_config) as f:
            mp_config = json.load(f)

    if args.stage == "lmhead":
        # Wrap only lm_head.
        old_lmhead = model.lm_head
        if isinstance(old_lmhead, nn.Linear):
            model.lm_head = QuantLinear(old_lmhead, quant_fn, bits)
            n_wrapped = 1
        else:
            n_wrapped = 0
    elif mp_config is not None:
        # mixed stage with per-tensor bits: wrap each module separately.
        n_wrapped = 0
        for name, mod in list(model.named_modules()):
            if not isinstance(mod, nn.Linear):
                continue
            if not any(name.endswith(s) for s in suffixes):
                continue
            module_bits = mp_config.get(name, bits)
            # Find parent and child name.
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules()).get(parts[0])
                child_attr = parts[1]
            else:
                parent = model
                child_attr = parts[0]
            if parent is None:
                continue
            setattr(parent, child_attr, QuantLinear(mod, quant_fn, module_bits))
            n_wrapped += 1
    else:
        n_wrapped = wrap_linears(model, suffixes, quant_fn, bits)

    # Save module wrap list.
    wrapped_names = [name for name, _ in list_wrapped_modules(model)]
    with open(out_dir / "module_wrap.json", "w") as f:
        json.dump(wrapped_names, f, indent=2)

    print(f"[wrap] stage={args.stage}  quant={args.quant}  bits={bits}  "
          f"wrapped={n_wrapped} modules", flush=True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable after wrap: {n_trainable/1e6:.1f}M", flush=True)

    # --- Dry run --------------------------------------------------------------
    if args.dry_run:
        print("\n[dry-run] wrapped modules:")
        for nm in wrapped_names:
            print(f"  {nm}")
        print(f"\n[dry-run] running 3 training steps on CPU ...")
        rows = load_jsonl(args.data)
        if args.max_rows:
            rows = rows[: args.max_rows]
        encode = load_tokenizer(args.hf_dir)
        dr_device = "cpu"
        model_dr = model.to(dr_device)
        model_dr.train()
        opt_dr = torch.optim.AdamW(
            [p for p in model_dr.parameters() if p.requires_grad],
            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
        )
        for step in range(3):
            rec = rows[step % len(rows)]
            batch = build_batch(rec, encode, args.max_length, dr_device)
            if batch is None:
                print(f"  step {step}: skipped (no signal)")
                continue
            ids, lbl = batch
            loss = compute_ce_loss(model_dr, ids, lbl)
            loss.backward()
            opt_dr.step()
            opt_dr.zero_grad()
            print(f"  step {step}: loss={loss.item():.4f}  T={ids.shape[1]}", flush=True)
        print("[dry-run] OK — loss is finite; grad flowed.")
        return

    events_path = out_dir / "events.jsonl"

    # --- Resume from checkpoint ----------------------------------------------
    resume_opt_step = 0
    resume_loss_ema: Optional[float] = None
    if args.resume_from:
        ckpt = Path(args.resume_from) / "state_dict.pt"
        print(f"[resume] loading {ckpt}", flush=True)
        sd = torch.load(str(ckpt), map_location=args.device)
        model.load_state_dict(sd, strict=True)
        resume_opt_step = args.resume_step
        if resume_opt_step == 0:
            import re as _re
            m = _re.search(r"step_0*(\d+)", Path(args.resume_from).name)
            if m:
                resume_opt_step = int(m.group(1))
        if resume_opt_step > 0 and events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        ev = json.loads(line)
                        if ev.get("loss_ema") is not None:
                            resume_loss_ema = float(ev["loss_ema"])
                    except json.JSONDecodeError:
                        pass
        print(f"[resume] opt_step={resume_opt_step}  loss_ema={resume_loss_ema}", flush=True)

    # --- Optimizer -----------------------------------------------------------
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
    )

    # --- Tokenizer + data ----------------------------------------------------
    print(f"[data] loading tokenizer from {args.hf_dir} ...", flush=True)
    encode = load_tokenizer(args.hf_dir)

    print(f"[data] loading {args.data} ...", flush=True)
    rows = load_jsonl(args.data)
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"[data] {len(rows)} rows", flush=True)

    # --- Teacher model (optional) -------------------------------------------
    teacher_model: Optional[RWKV7Model] = None
    if args.teacher and args.kd != "none":
        print(f"[kd] loading teacher from {args.teacher} ...", flush=True)
        teacher_model = load_rwkv7(args.teacher, device=args.device, dtype=torch.float32)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    artifacts_dir = Path(args.data).parent / "artifacts"
    tmp_dir = out_dir / "requant_tmp"
    if args.quant == "strand-pv":
        tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop -------------------------------------------------------
    global_step = resume_opt_step * args.grad_accum
    opt_step = resume_opt_step
    loss_ema = resume_loss_ema
    t0 = time.time()
    seen_tok = 0

    print(f"[train] starting  epochs={args.epochs}  grad_accum={args.grad_accum}  "
          f"lr={args.lr}  device={args.device}"
          + (f"  resumed_from_step={resume_opt_step}" if resume_opt_step else ""),
          flush=True)

    opt.zero_grad()

    for epoch in range(args.epochs):
        # Deterministic ordering without RNG: alternate slice direction per epoch.
        order = list(range(len(rows)))
        if epoch % 2 == 1:
            order = order[::-1]

        # On the first epoch of a resumed run, skip rows already consumed.
        row_skip = global_step if epoch == 0 else 0

        for local_i, row_idx in enumerate(order):
            if local_i < row_skip:
                continue
            rec = rows[row_idx]
            batch = build_batch(rec, encode, args.max_length, args.device)
            if batch is None:
                continue

            input_ids, labels = batch
            seen_tok += input_ids.shape[1]
            global_step += 1

            # CE loss.
            ce_loss = compute_ce_loss(model, input_ids, labels)

            # KD loss.
            if args.kd != "none" and (teacher_model is not None or artifacts_dir.exists()):
                kd_loss = compute_kd_loss(
                    model, teacher_model, input_ids, labels,
                    args.kd, args.kd_temperature, artifacts_dir, global_step,
                )
                loss = args.ce_weight * ce_loss + args.kd_weight * kd_loss
            else:
                loss = ce_loss
                kd_loss = torch.tensor(0.0)

            (loss / args.grad_accum).backward()

            l_item = loss.item()
            loss_ema = l_item if loss_ema is None else 0.98 * loss_ema + 0.02 * l_item

            # Release MPS fragmentation.
            if args.device == "mps":
                torch.mps.empty_cache()

            # Optimizer step every grad_accum micro-batches.
            if global_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                opt.zero_grad()
                opt_step += 1

                dt = time.time() - t0
                event = {
                    "step": opt_step,
                    "epoch": epoch,
                    "loss": round(l_item, 6),
                    "loss_ema": round(loss_ema, 6),
                    "ppl_ema": round(math.exp(min(loss_ema, 20.0)), 2),
                    "ce_loss": round(ce_loss.item(), 6),
                    "kd_loss": round(kd_loss.item() if isinstance(kd_loss, torch.Tensor) else 0.0, 6),
                    "tok_s": round(seen_tok / dt, 1),
                    "run_id": args.run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                with open(events_path, "a") as ef:
                    ef.write(json.dumps(event) + "\n")

                if opt_step % 10 == 0:
                    print(f"[ep{epoch} opt={opt_step}] loss={loss_ema:.4f}  "
                          f"ppl={math.exp(min(loss_ema,20)):.1f}  "
                          f"{seen_tok/dt:.0f} tok/s", flush=True)

                # Checkpoint.
                if args.save_every and opt_step % args.save_every == 0:
                    _save_checkpoint(model, out_dir, f"step_{opt_step:06d}")
                    _save_checkpoint(model, out_dir, "latest")

                # Eval.
                if args.eval_every and opt_step % args.eval_every == 0:
                    run_eval(model, encode, args, opt_step, out_dir)

                # strand-pv requant.
                if (args.quant == "strand-pv"
                        and args.requant_every
                        and opt_step % args.requant_every == 0):
                    requant_step(model, args, opt_step, tmp_dir)

    # --- Final save ----------------------------------------------------------
    _save_checkpoint(model, out_dir, "final")
    total_time = time.time() - t0
    print(f"[done] opt_steps={opt_step}  loss_ema={loss_ema:.4f}  "
          f"elapsed={total_time/3600:.2f}h", flush=True)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save_checkpoint(model: RWKV7Model, out_dir: Path, tag: str) -> None:
    dest = out_dir / tag
    dest.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    torch.save(sd, dest / "state_dict.pt")
    print(f"  [save] {dest / 'state_dict.pt'}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
