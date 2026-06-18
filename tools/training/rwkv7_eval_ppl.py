"""Canonical RWKV-7 perplexity evaluation script.

Usage:
    python rwkv7_eval_ppl.py \
        --model models/rwkv7-g1-04-hf/model.safetensors \
        --hf-dir models/rwkv7-g1-04-hf \
        --corpus wikitext2 \
        --tokens 8192 \
        --stride 512 \
        --out artifacts/rwkv7_posttrain/ppl.jsonl \
        --run-id baseline

Supported corpora: wikitext2 | wikitext103-small | heldout | text-file
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------

def load_tokenizer(hf_dir: str):
    """Try fast tokenizer first, then fall back to transformers AutoTokenizer."""
    hf_path = Path(hf_dir)

    # Strategy 0: RWKV custom greedy-trie tokenizer (avoids triton import).
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

    # Strategy 1: tokenizers library fast tokenizer.
    tok_json = hf_path / "tokenizer.json"
    if tok_json.exists():
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(tok_json))

            def encode_fn(text: str) -> list[int]:
                return tok.encode(text).ids

            return encode_fn
        except ImportError:
            pass

    # Strategy 2: transformers AutoTokenizer.
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(hf_path), trust_remote_code=True)

        def encode_fn(text: str) -> list[int]:
            return tok.encode(text, add_special_tokens=False)

        return encode_fn
    except Exception as exc:
        raise RuntimeError(
            f"Could not load tokenizer from {hf_dir}. "
            "Install either 'tokenizers' (for tokenizer.json) or "
            "'transformers' (for AutoTokenizer). "
            f"Underlying error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(corpus: str, text_file: str | None) -> str:
    if corpus == "wikitext2":
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            return "\n".join(ds["text"])
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load wikitext2. Install 'datasets' and ensure network access. {exc}"
            ) from exc

    if corpus == "wikitext103-small":
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
            return "\n".join(ds["text"])
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load wikitext103-small. Install 'datasets' and ensure network access. {exc}"
            ) from exc

    if corpus == "heldout":
        heldout_path = Path("artifacts/rwkv7_posttrain/heldout.jsonl")
        if not heldout_path.exists():
            raise FileNotFoundError(
                f"Heldout corpus not found at {heldout_path}. "
                "Run the post-training data prep pipeline first."
            )
        lines = []
        with open(heldout_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "text" in obj:
                    lines.append(obj["text"])
        if not lines:
            raise RuntimeError(f"heldout.jsonl at {heldout_path} has no 'text' fields.")
        return "\n".join(lines)

    if corpus == "text-file":
        if not text_file:
            raise ValueError("--text-file PATH is required when --corpus text-file")
        with open(text_file, "r", encoding="utf-8") as f:
            return f.read()

    raise ValueError(f"Unknown corpus: {corpus!r}")


# ---------------------------------------------------------------------------
# PPL computation
# ---------------------------------------------------------------------------

def compute_ppl(
    model,
    token_ids: list[int],
    n_tokens: int,
    stride: int,
    device: str,
    dtype: torch.dtype,
    teacher_logits_path: str | None,
) -> tuple[float, float, int]:
    """Sliding-window perplexity.

    Returns (ppl, nll, n_evaluated_tokens).
    """
    ids = token_ids[:n_tokens]
    total_len = len(ids)
    if total_len < 2:
        raise ValueError(f"Not enough tokens to evaluate: got {total_len}, need >=2.")

    # Pre-load teacher logits if requested (shape [total_len, vocab]).
    teacher_top1: list[int] | None = None
    if teacher_logits_path:
        raw = torch.load(teacher_logits_path, map_location="cpu")
        # Accept either saved logits tensor or pre-softmaxed probs.
        if isinstance(raw, torch.Tensor):
            teacher_top1 = raw.argmax(dim=-1).tolist()
        elif isinstance(raw, dict) and "logits" in raw:
            teacher_top1 = raw["logits"].argmax(dim=-1).tolist()
        else:
            raise RuntimeError(
                f"Unrecognised teacher logits format in {teacher_logits_path}. "
                "Expected a tensor or dict with 'logits' key."
            )

    ids_tensor = torch.tensor(ids, dtype=torch.long, device=device)

    total_nll = 0.0
    total_n = 0
    teacher_agree = 0
    teacher_total = 0

    model.eval()
    # Context window: we use the full stride window starting at offset 0, then
    # slide by stride.  Each window predicts the tokens in (begin+1 .. end].
    begin = 0
    with torch.no_grad():
        while begin < total_len - 1:
            end = min(begin + stride, total_len)
            window_ids = ids_tensor[begin:end].unsqueeze(0)  # [1, T]
            logits = model(window_ids)  # [1, T, vocab]

            # Targets are shifted by one: predict ids[1..T] from context ids[0..T-1].
            # For the first window, all positions count.
            # For subsequent windows, only the new positions beyond the previous end count
            # to avoid double-counting overlap.
            if begin == 0:
                target_start = 0  # first position is ids[1], shifted from logits[:, 0]
            else:
                # Positions [0 .. (prev_end - begin - 1)] were already counted.
                # Clamp to 0: when stride == window size there is no overlap and
                # target_start would be -1, which Python interprets as the last
                # element rather than "start from the beginning".
                target_start = max(0, prev_end - begin - 1)

            # logits[:, t] predicts ids[begin + t + 1].
            # We want target indices begin+target_start+1 .. end (exclusive).
            logits_slice = logits[0, target_start : end - begin - 1, :]  # [n, vocab]
            targets_slice = ids_tensor[begin + target_start + 1 : end]   # [n]

            if logits_slice.shape[0] == 0:
                prev_end = end
                begin += stride
                continue

            nll = F.cross_entropy(
                logits_slice.float(),
                targets_slice,
                reduction="sum",
            ).item()

            n = targets_slice.shape[0]
            total_nll += nll
            total_n += n

            if teacher_top1 is not None:
                # Compare student argmax to teacher top-1 for the counted positions.
                student_top1 = logits_slice.argmax(dim=-1)
                for i, pos in enumerate(range(begin + target_start + 1, end)):
                    if pos < len(teacher_top1):
                        teacher_agree += int(student_top1[i].item() == teacher_top1[pos])
                        teacher_total += 1

            prev_end = end
            begin += stride

    if total_n == 0:
        raise RuntimeError("No tokens were evaluated. Check --tokens and --stride values.")

    avg_nll = total_nll / total_n
    ppl = math.exp(avg_nll)

    if teacher_top1 is not None and teacher_total > 0:
        agreement = teacher_agree / teacher_total
        print(f"  Teacher top-1 agreement: {agreement:.4f}  ({teacher_agree}/{teacher_total})")

    return ppl, avg_nll, total_n


# ---------------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------------

def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str) -> torch.dtype:
    # bfloat16 is not supported on MPS.
    if device == "mps":
        return torch.float32
    return torch.float32


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RWKV-7 perplexity evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to .safetensors checkpoint")
    parser.add_argument("--hf-dir", required=True, help="Directory containing tokenizer files")
    parser.add_argument(
        "--corpus",
        choices=["wikitext2", "wikitext103-small", "heldout", "text-file"],
        default="wikitext2",
    )
    parser.add_argument("--text-file", default=None, help="Path to text file (--corpus text-file)")
    parser.add_argument("--tokens", type=int, default=8192, help="Number of tokens to evaluate")
    parser.add_argument("--stride", type=int, default=512, help="Sliding window stride")
    parser.add_argument("--device", default=None, help="mps | cpu | cuda (auto-detect if omitted)")
    parser.add_argument("--out", default=None, help="JSONL output file (append mode)")
    parser.add_argument("--run-id", default=None, help="Optional identifier for this run")
    parser.add_argument(
        "--teacher-logits",
        default=None,
        help="Path to pre-captured teacher logits for top-1 agreement reporting",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(device)

    print(f"device={device}  dtype={dtype}")

    # Load tokenizer.
    print(f"Loading tokenizer from {args.hf_dir} ...")
    encode = load_tokenizer(args.hf_dir)

    # Load corpus.
    print(f"Loading corpus: {args.corpus} ...")
    text = load_corpus(args.corpus, args.text_file)

    # Tokenize.
    print("Tokenizing ...")
    token_ids = encode(text)
    print(f"  corpus tokens: {len(token_ids):,}  (evaluating first {args.tokens:,})")

    if len(token_ids) < args.tokens:
        print(
            f"  WARNING: corpus has only {len(token_ids):,} tokens; "
            f"evaluating all of them instead of requested {args.tokens:,}."
        )

    # Load model.
    # Import here so the script is importable even when the model isn't on disk.
    sys.path.insert(0, str(Path(__file__).parent))
    from rwkv7_load_weights import load_rwkv7

    print(f"Loading model from {args.model} ...")
    t0 = time.time()
    model_path = args.model
    if model_path.endswith(".pt"):
        # QAT checkpoint: load base safetensors then patch trained weights.
        base_st = str(Path(args.hf_dir) / "model.safetensors")
        print(f"  (QAT .pt: base={base_st}, patching trained weights)")
        model = load_rwkv7(base_st, device="cpu", dtype=torch.float32)
        sd = torch.load(model_path, map_location="cpu")
        weight_only = {k: v for k, v in sd.items() if k.endswith(".weight")}
        missing, unexpected = model.load_state_dict(weight_only, strict=False)
        print(f"  patched {len(weight_only)} weight tensors "
              f"(missing={len(missing)} unexpected={len(unexpected)})")
        model = model.to(device=device, dtype=dtype)
    else:
        model = load_rwkv7(model_path, device=device, dtype=dtype)
    print(f"  loaded in {time.time() - t0:.1f}s")

    # Evaluate.
    print(f"Evaluating (stride={args.stride}) ...")
    t0 = time.time()
    ppl, nll, n_tokens = compute_ppl(
        model,
        token_ids,
        n_tokens=args.tokens,
        stride=args.stride,
        device=device,
        dtype=dtype,
        teacher_logits_path=args.teacher_logits,
    )
    elapsed = time.time() - t0

    print(f"PPL: {ppl:.2f}  NLL: {nll:.4f}  tokens: {n_tokens}  ({elapsed:.1f}s)")

    # Write JSONL output.
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": args.run_id,
            "model": str(args.model),
            "corpus": args.corpus,
            "n_tokens": n_tokens,
            "nll": round(nll, 6),
            "ppl": round(ppl, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"Result appended to {out_path}")


if __name__ == "__main__":
    main()
