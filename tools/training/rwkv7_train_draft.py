"""Train custom RWKV-7 draft models for speculative decoding.

The script trains one of the compact configs in rwkv7_custom_configs.py from
scratch on the SFT corpus. If a directory of top-k teacher-logit shards is
provided, it adds a truncated top-k distillation term.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

import sys

sys.path.insert(0, str(HERE))

from rwkv7_custom_configs import CUSTOM_VARIANTS, VARIANT_ORDER, estimated_params
from rwkv7_sft_torch import build_examples, load_tokenizer
from rwkv7_torch_model import RWKV7Model


def resolve_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        print("[device] requested mps but it is unavailable; falling back to cpu", flush=True)
        return "cpu"
    return requested


def initialise_from_scratch(model: RWKV7Model) -> None:
    """Depth-scaled random init while keeping norm weights sane."""
    depth_std = 0.02 / math.sqrt(2 * model.cfg.n_layer)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                std = 0.02
                if name.endswith(("attn.o_proj.weight", "ffn.value.weight", "lm_head.weight")):
                    std = depth_std
                nn.init.normal_(p, mean=0.0, std=std)
            elif name.endswith(("_norm_w", "norm_w", "g_norm_w", "pre_norm_w", "attn_norm_w", "ffn_norm_w")):
                p.fill_(1.0)
            else:
                p.zero_()


def lm_loss_from_example(model: RWKV7Model, ids: list[int], labels: list[int], device: str):
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = torch.tensor([labels], dtype=torch.long, device=device)
    hidden = model(x, return_final_hidden=True)
    shift_hidden = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
    shift_labels = y[:, 1:].reshape(-1)
    mask = shift_labels != -100
    n_supervised = int(mask.sum().item())
    if n_supervised == 0:
        return None, 0
    logits = model.lm_head(shift_hidden[mask])
    return F.cross_entropy(logits.float(), shift_labels[mask]), n_supervised


def load_teacher_records(path: Path) -> list[dict]:
    shard_paths = sorted(path.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard_*.pt files found in {path}")
    records: list[dict] = []
    for shard in shard_paths:
        loaded = torch.load(str(shard), map_location="cpu")
        if not isinstance(loaded, list):
            raise RuntimeError(f"{shard} is not a list of teacher records")
        records.extend(loaded)
    if not records:
        raise RuntimeError(f"teacher-logit directory {path} contained no records")
    return records


def _mask_to_positions(mask) -> list[int]:
    if isinstance(mask, torch.Tensor):
        mask = mask.tolist()
    return [i for i, keep in enumerate(mask) if bool(keep)]


def kd_loss_from_record(model: RWKV7Model, record: dict, device: str, alpha: float):
    ids = [int(x) for x in record["input_ids"]]
    supervised_positions = _mask_to_positions(record["supervised_mask"])
    pairs = [(row_idx, pos) for row_idx, pos in enumerate(supervised_positions) if pos < len(ids) - 1]
    if not pairs:
        return None, 0

    row_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
    pos_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
    labels = torch.tensor([ids[p[1] + 1] for p in pairs], dtype=torch.long, device=device)

    x = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = model(x, return_final_hidden=True)[0]
    logits = model.lm_head(hidden.index_select(0, pos_idx))
    ce = F.cross_entropy(logits.float(), labels)

    top_ids = record["top_ids"]
    top_logits = record["top_logits"]
    if not isinstance(top_ids, torch.Tensor):
        top_ids = torch.tensor(top_ids)
    if not isinstance(top_logits, torch.Tensor):
        top_logits = torch.tensor(top_logits)
    top_ids = top_ids.index_select(0, row_idx.cpu()).to(device=device, dtype=torch.long)
    top_logits = top_logits.index_select(0, row_idx.cpu()).to(device=device, dtype=torch.float32)

    student_top_logits = logits.gather(1, top_ids)
    student_top_logprobs = F.log_softmax(student_top_logits.float(), dim=-1)
    teacher_top_logprobs = F.log_softmax(top_logits.float(), dim=-1)
    student_top_probs = student_top_logprobs.exp()
    kl_student_teacher = (student_top_probs * (student_top_logprobs - teacher_top_logprobs)).sum(dim=-1).mean()

    return alpha * ce + (1.0 - alpha) * kl_student_teacher, len(pairs)


def save_checkpoint(model: RWKV7Model, out: Path, tag: str, variant: str, step: int) -> None:
    dest = out / tag
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(
        {k: v.detach().to("cpu", torch.float32) for k, v in model.state_dict().items()},
        dest / "state_dict.pt",
    )
    meta = {
        "variant": variant,
        "step": step,
        "params_M_formula": round(estimated_params(model.cfg) / 1e6, 6),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"  [save] {dest / 'state_dict.pt'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--variant", required=True, choices=VARIANT_ORDER)
    ap.add_argument("--teacher-logits", default=None, help="Directory of shard_*.pt from rwkv7_capture_teacher_logits.py")
    ap.add_argument("--alpha", type=float, default=0.5, help="CE weight in alpha*CE + (1-alpha)*KD")
    ap.add_argument("--hf-dir", default=str(ROOT / "models/rwkv7-g1-04-hf"))
    ap.add_argument("--data", default=str(ROOT / "artifacts/rwkv7_posttrain/sft.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = no optimizer-step cap")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--use-chunked", action="store_true", help="Enable chunked WKV training path")
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1]")

    device = resolve_device(args.device)
    out = Path(args.out) if args.out else ROOT / "artifacts/lowbit_rwkv7/runs" / f"custom_{args.variant}"
    out.mkdir(parents=True, exist_ok=True)

    cfg = replace(CUSTOM_VARIANTS[args.variant], use_chunked=args.use_chunked, chunk_size=args.chunk_size)
    model = RWKV7Model(cfg)
    initialise_from_scratch(model)
    model.grad_checkpoint = True
    model = model.to(device=device, dtype=torch.float32)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[model] {args.variant}: n_embd={cfg.n_embd} layers={cfg.n_layer} "
        f"n_ff={cfg.n_ff} params={n_params/1e6:.1f}M formula={estimated_params(cfg)/1e6:.1f}M",
        flush=True,
    )

    if args.teacher_logits:
        train_items = load_teacher_records(Path(args.teacher_logits))
        if args.max_rows:
            train_items = train_items[: args.max_rows]
        mode = "kd"
        n_tokens = sum(len(r["input_ids"]) for r in train_items)
        print(f"[data] {len(train_items)} teacher records, {n_tokens} tokens, alpha={args.alpha}", flush=True)
    else:
        tok = load_tokenizer(Path(args.hf_dir))
        rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
        if args.max_rows:
            rows = rows[: args.max_rows]
        train_items = build_examples(rows, tok, args.max_length)
        mode = "sft"
        n_tokens = sum(len(ids) for ids, _ in train_items)
        n_sup = sum(sum(1 for y in labels if y != -100) for _, labels in train_items)
        print(f"[data] {len(train_items)} SFT examples, {n_tokens} tokens ({n_sup} supervised)", flush=True)

    if not train_items:
        raise RuntimeError(
            "no training examples after tokenization/truncation; increase --max-length "
            "or check the input data/teacher shards"
        )

    if args.dry_run:
        args.epochs = 1
        args.max_steps = min(args.max_steps or 2, 2)
        args.save_every = 0
        args.log_every = 1

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    n_steps = 0
    pending = 0
    loss_ema = None
    seen_tok = 0
    t0 = time.time()
    stop = False

    for epoch in range(args.epochs):
        generator = torch.Generator().manual_seed(1000 + epoch)
        order = torch.randperm(len(train_items), generator=generator).tolist()
        opt.zero_grad(set_to_none=True)

        for j, idx in enumerate(order):
            item = train_items[idx]
            if mode == "kd":
                loss, supervised = kd_loss_from_record(model, item, device, args.alpha)
                seen_tok += len(item["input_ids"])
            else:
                ids, labels = item
                loss, supervised = lm_loss_from_example(model, ids, labels, device)
                seen_tok += len(ids)
            if loss is None or supervised == 0:
                continue

            (loss / args.grad_accum).backward()
            pending += 1

            l = float(loss.detach().item())
            loss_ema = l if loss_ema is None else 0.98 * loss_ema + 0.02 * l
            if device == "mps":
                torch.mps.empty_cache()

            is_last = j == len(order) - 1
            if pending >= args.grad_accum or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                pending = 0
                n_steps += 1

                if n_steps % args.log_every == 0 or n_steps == 1:
                    dt = max(time.time() - t0, 1e-6)
                    ppl = math.exp(min(loss_ema if loss_ema is not None else 0.0, 20.0))
                    print(
                        f"[ep{epoch} opt={n_steps}] loss={loss_ema:.4f} "
                        f"ppl={ppl:.1f} tok/s={seen_tok/dt:.0f}",
                        flush=True,
                    )

                if args.save_every and n_steps % args.save_every == 0:
                    save_checkpoint(model, out, f"step_{n_steps:06d}", args.variant, n_steps)
                    save_checkpoint(model, out, "latest", args.variant, n_steps)

                if args.max_steps and n_steps >= args.max_steps:
                    stop = True
                    break

        if stop:
            break

    elapsed_h = (time.time() - t0) / 3600.0
    if args.dry_run:
        print(f"[dry-run] OK opt={n_steps} final_loss={loss_ema:.4f} hours={elapsed_h:.2f}", flush=True)
        return
    save_checkpoint(model, out, f"step_{n_steps:06d}", args.variant, n_steps)
    save_checkpoint(model, out, "latest", args.variant, n_steps)
    save_checkpoint(model, out, "final", args.variant, n_steps)
    print(f"[done] {args.variant} opt={n_steps} final_loss={loss_ema:.4f} hours={elapsed_h:.2f}", flush=True)


if __name__ == "__main__":
    main()
