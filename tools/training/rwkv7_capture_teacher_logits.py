"""Capture top-k teacher logits from a trained RWKV-7 model for knowledge distillation.

Runs the teacher model over a training dataset (JSONL), collects log-softmax
logits at supervised positions (completion tokens only), and writes shard files
for downstream distillation training.

Usage:
  python rwkv7_capture_teacher_logits.py \\
    --model models/rwkv7-g1-04-hf/model.safetensors \\
    --hf-dir models/rwkv7-g1-04-hf \\
    --data data/train.jsonl \\
    --out /tmp/teacher_logits \\
    --top-k 128 \\
    --max-seq-len 1024 \\
    --shard-size 1000 \\
    --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from rwkv7_load_weights import load_rwkv7
from rwkv7_sft_torch import load_tokenizer
from rwkv7_torch_model import RWKV7Config


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def encode_text(tokenizer, text: str) -> list[int]:
    if hasattr(tokenizer, "encodeBytes"):
        return tokenizer.encodeBytes(text.encode("utf-8"))
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def tokenize_sequence(
    tokenizer,
    item: dict,
    max_seq_len: int,
) -> tuple[list[int], int] | None:
    """Tokenise a JSONL item and return (input_ids, prompt_len).

    Returns None if the result has no supervised tokens (e.g. completion
    is empty after truncation).

    Handles three input shapes:
      {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
      {"prompt": "...", "completion": "..."}
      {"text": "..."}  — treated as pure completion (prompt_len=0)
    """
    if "messages" in item:
        msgs = item.get("messages") or []
        user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        assistant = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        if not user or not assistant:
            return None
        prompt_ids = [0] + encode_text(tokenizer, f"User: {user}\n\nAssistant:")
        completion_ids = encode_text(tokenizer, f" {assistant}") + [0]
    elif "text" in item:
        ids = encode_text(tokenizer, item["text"])
        ids = ids[:max_seq_len]
        if len(ids) < 1:
            return None
        return ids, 0
    else:
        prompt_text = item.get("prompt", "")
        completion_text = item.get("completion", "")

        prompt_ids = encode_text(tokenizer, prompt_text)
        completion_ids = encode_text(tokenizer, completion_text)

    # Truncate the combined sequence to max_seq_len.  Preserve as many
    # completion tokens as possible — only trim the prompt from the front.
    total = len(prompt_ids) + len(completion_ids)
    if total > max_seq_len:
        overflow = total - max_seq_len
        trim_prompt = min(overflow, len(prompt_ids))
        prompt_ids = prompt_ids[trim_prompt:]
        overflow -= trim_prompt
        if overflow > 0:
            completion_ids = completion_ids[:len(completion_ids) - overflow]

    ids = prompt_ids + completion_ids
    prompt_len = len(prompt_ids)

    if len(ids) - prompt_len < 1:
        # No supervised positions remain after truncation.
        return None

    return ids, prompt_len


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_sequence(
    model,
    input_ids: list[int],
    prompt_len: int,
    top_k: int,
    device: str,
) -> dict:
    """Run the teacher forward pass and extract top-k log-probs at supervised
    positions.

    Returns a result dict (see module docstring for schema).
    """
    ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)  # [1, T]
    logits = model(ids_t)  # [1, T, V]
    logits = logits[0]     # [T, V]

    T = logits.shape[0]
    log_probs = F.log_softmax(logits.float(), dim=-1)  # [T, V]

    # Supervised mask: True for completion token positions.
    sup_mask = [False] * prompt_len + [True] * (T - prompt_len)
    sup_indices = [i for i, m in enumerate(sup_mask) if m]
    T_sup = len(sup_indices)

    sup_log_probs = log_probs[sup_indices]  # [T_sup, V]

    # Top-k indices and values.
    top_vals, top_ids = torch.topk(sup_log_probs, k=top_k, dim=-1)  # [T_sup, top_k]

    # Per-position entropy: H = -sum(p * log(p)) = -sum(exp(lp) * lp)
    probs = sup_log_probs.exp()  # [T_sup, V]
    entropy = -(probs * sup_log_probs).sum(dim=-1)  # [T_sup]

    return {
        "input_ids": input_ids,
        "supervised_mask": sup_mask,
        "top_ids": top_ids.cpu(),       # [T_sup, top_k]  int64
        "top_logits": top_vals.cpu(),   # [T_sup, top_k]  float32
        "entropy": entropy.cpu(),       # [T_sup]          float32
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture top-k teacher logits for RWKV-7 knowledge distillation."
    )
    parser.add_argument("--model", required=True, help="Path to safetensors checkpoint")
    parser.add_argument("--hf-dir", required=True, help="HF tokenizer directory")
    parser.add_argument("--data", required=True, help="Input JSONL file")
    parser.add_argument("--out", required=True, help="Output directory for shards")
    parser.add_argument("--top-k", type=int, default=128, dest="top_k")
    parser.add_argument("--max-seq-len", type=int, default=1024, dest="max_seq_len")
    parser.add_argument("--shard-size", type=int, default=1000, dest="shard_size",
                        help="Number of sequences per output shard")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load tokenizer ----
    print(f"Loading tokenizer from {args.hf_dir} ...")
    tokenizer = load_tokenizer(Path(args.hf_dir))

    # ---- Load model ----
    print(f"Loading model from {args.model} onto {args.device} ...")
    model = load_rwkv7(args.model, cfg=RWKV7Config(), device=args.device)
    model.eval()
    if args.device == "mps":
        model = model.to(torch.device("mps"))

    # ---- Count lines (for progress display) ----
    data_path = Path(args.data)
    total_lines = sum(1 for _ in data_path.open())
    print(f"Dataset: {total_lines} lines in {data_path}")

    # ---- Process sequences ----
    shard_records: list[dict] = []
    shard_idx = 0
    seq_idx = 0
    skipped = 0
    supervised_tokens_total = 0

    def flush_shard() -> None:
        nonlocal shard_idx
        shard_path = out_dir / f"shard_{shard_idx:06d}.pt"
        torch.save(shard_records, shard_path)
        print(f"  -> saved {shard_path} ({len(shard_records)} sequences)")
        shard_idx += 1
        shard_records.clear()

    with data_path.open() as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: JSON parse error on line {seq_idx + skipped + 1}: {exc}",
                      file=sys.stderr)
                skipped += 1
                continue

            result = tokenize_sequence(tokenizer, item, args.max_seq_len)
            if result is None:
                skipped += 1
                continue

            input_ids, prompt_len = result

            record = process_sequence(
                model,
                input_ids,
                prompt_len,
                args.top_k,
                args.device,
            )
            record["seq_idx"] = seq_idx

            n_sup = int(record["supervised_mask"].count(True))
            supervised_tokens_total += n_sup

            shard_records.append(record)
            seq_idx += 1

            # Progress report every 100 sequences.
            if seq_idx % 100 == 0:
                print(f"seq {seq_idx}/{total_lines}, "
                      f"supervised_tokens_total={supervised_tokens_total}")

            # Flush when shard is full.
            if len(shard_records) >= args.shard_size:
                flush_shard()

    # Flush any remaining records.
    if shard_records:
        flush_shard()

    n_shards = shard_idx

    # ---- Write manifest ----
    manifest = {
        "n_sequences": seq_idx,
        "n_shards": n_shards,
        "top_k": args.top_k,
        "model": str(Path(args.model).resolve()),
        "data": str(data_path.resolve()),
        "shard_size": args.shard_size,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nDone.")
    print(f"  sequences processed : {seq_idx}")
    print(f"  sequences skipped   : {skipped}")
    print(f"  supervised tokens   : {supervised_tokens_total}")
    print(f"  shards written      : {n_shards}")
    print(f"  manifest            : {manifest_path}")


if __name__ == "__main__":
    main()
