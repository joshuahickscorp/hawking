"""On-device SFT for RWKV-7-0.4B on Apple MPS, using the parity-verified pure-torch
forward (rwkv7_torch_model.py) — fla's RWKV7 recurrence is triton/CUDA-only and
cannot train on MPS.

Trains the g1-0.4B instruct base on the gold corpus (artifacts/rwkv7_posttrain/
sft.jsonl). Prompt-masked causal-LM loss (loss only on the assistant completion +
the token-0 EOS), grad-accum, grad-checkpointing, fp32 master weights.

Tokenization matches the base model's validated behavior:
    input  = [0] + encode("User: {u}\\n\\nAssistant:")
    target = encode(" {a}") + [0]      # leading space + token-0 EOS (model stops on 0)
where token 0 = <|rwkv_tokenizer_end_of_text|> (the doc/turn separator).

Usage:
    python rwkv7_sft_torch.py --dry-run                       # CPU, 3 steps, no save
    python rwkv7_sft_torch.py --device mps --out <dir>        # real SFT
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def load_tokenizer(hf_dir: Path):
    """The standalone RWKV World greedy-trie tokenizer (no fla/triton)."""
    spec = importlib.util.spec_from_file_location(
        "hf_rwkv_tokenizer", str(hf_dir / "hf_rwkv_tokenizer.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RWKV_TOKENIZER(str(hf_dir / "rwkv_vocab_v20230424.txt"))


def build_examples(rows, tok, max_length: int):
    """Yield (input_ids, labels) with the prompt masked to -100."""
    EOS = 0  # <|rwkv_tokenizer_end_of_text|>
    examples = []
    for r in rows:
        msgs = r.get("messages") or []
        u = next((m["content"] for m in msgs if m["role"] == "user"), None)
        a = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if not u or not a:
            continue
        prompt = [EOS] + tok.encodeBytes(f"User: {u}\n\nAssistant:".encode("utf-8"))
        comp = tok.encodeBytes(f" {a}".encode("utf-8")) + [EOS]
        ids = prompt + comp
        labels = [-100] * len(prompt) + comp
        if len(ids) > max_length:
            ids = ids[:max_length]
            labels = labels[:max_length]
        if all(l == -100 for l in labels):  # prompt filled the window — no signal
            continue
        examples.append((ids, labels))
    return examples


def lm_loss(model, ids, labels, device):
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = torch.tensor([labels], dtype=torch.long, device=device)
    # Run the model WITHOUT the lm_head, then project only the supervised positions
    # (the prompt is masked, so ~half the 65K-vocab matmul is otherwise wasted). Same
    # loss as a full cross_entropy(logits, labels, ignore_index=-100).
    hidden = model(x, return_final_hidden=True)  # [1, T, n_embd]
    shift_hidden = hidden[:, :-1, :].reshape(-1, hidden.size(-1))  # predict t+1 from t
    shift_labels = y[:, 1:].reshape(-1)
    mask = shift_labels != -100
    sel_logits = model.lm_head(shift_hidden[mask])  # [num_supervised, vocab]
    return F.cross_entropy(sel_logits.float(), shift_labels[mask])


def freeze_to_last_n(model, n: int):
    """Train only the last `n` blocks + final norm + lm_head; freeze the rest."""
    if n <= 0:
        return
    n_layer = model.cfg.n_layer
    trainable_from = max(0, n_layer - n)
    for name, p in model.named_parameters():
        keep = name.startswith(("norm_", "lm_head"))
        if name.startswith("layers."):
            li = int(name.split(".")[1])
            keep = keep or (li >= trainable_from)
        p.requires_grad_(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/rwkv7-g1-04-hf/model.safetensors"))
    ap.add_argument("--hf-dir", default=str(ROOT / "models/rwkv7-g1-04-hf"))
    ap.add_argument("--data", default=str(ROOT / "artifacts/rwkv7_posttrain/sft.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/rwkv7_posttrain/sft_out"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--last-n-layers", type=int, default=0, help="0 = full fine-tune")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    ap.add_argument("--save-every", type=int, default=400, help="optimizer steps")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(HERE))
    from rwkv7_load_weights import load_rwkv7
    from rwkv7_torch_model import RWKV7Config

    hf_dir = Path(args.hf_dir)
    tok = load_tokenizer(hf_dir)
    rows = [json.loads(l) for l in open(args.data)]
    if args.max_rows:
        rows = rows[: args.max_rows]
    examples = build_examples(rows, tok, args.max_length)
    tot_tok = sum(len(i) for i, _ in examples)
    sup_tok = sum(sum(1 for x in l if x != -100) for _, l in examples)
    print(f"[data] {len(examples)} examples, {tot_tok} tokens ({sup_tok} supervised), "
          f"max_length={args.max_length}")

    print(f"[model] loading {args.model} on {args.device} (fp32)...")
    model = load_rwkv7(args.model, RWKV7Config(), device=args.device, dtype=torch.float32)
    model.grad_checkpoint = True
    model.train()
    if args.last_n_layers:
        freeze_to_last_n(model, args.last_n_layers)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {n_train/1e6:.1f}M "
          f"({'full' if not args.last_n_layers else f'last {args.last_n_layers} layers'})")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    if args.dry_run:
        print("[dry-run] 3 steps on the first examples (CPU), expect finite, ~decreasing loss")
        for step in range(3):
            ids, labels = examples[step % len(examples)]
            loss = lm_loss(model, ids, labels, args.device)
            loss.backward()
            opt.step(); opt.zero_grad()
            print(f"  step {step}: loss={loss.item():.4f}  (T={len(ids)})")
        print("[dry-run] OK — loss is finite; grad flowed; tokenization + masking valid.")
        return

    os.makedirs(args.out, exist_ok=True)
    n_steps = 0
    loss_ema = None
    t0 = time.time()
    seen_tok = 0
    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        # deterministic shuffle by index parity rounds (no RNG dependency)
        order = order[epoch % 2::1]
        opt.zero_grad()
        for i, idx in enumerate(order):
            ids, labels = examples[idx]
            loss = lm_loss(model, ids, labels, args.device) / args.grad_accum
            loss.backward()
            seen_tok += len(ids)
            l = loss.item() * args.grad_accum
            loss_ema = l if loss_ema is None else 0.98 * loss_ema + 0.02 * l
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
                if args.device == "mps":
                    torch.mps.empty_cache()  # per-opt-step: peak is per-example (autograd frees
                    # the graph itself); this only trims the allocator cache, no per-example churn
                n_steps += 1
                if n_steps % args.log_every == 0:
                    dt = time.time() - t0
                    print(f"[ep{epoch} step {n_steps}] loss={loss_ema:.4f} "
                          f"ppl={math.exp(min(loss_ema,20)):.1f} "
                          f"{seen_tok/dt:.0f} tok/s ({i+1}/{len(order)} ex)", flush=True)
                if args.save_every and n_steps % args.save_every == 0:
                    _save(model, args.out, f"step{n_steps}")
    _save(model, args.out, "final")
    print(f"[done] {n_steps} opt-steps, final loss_ema={loss_ema:.4f}, {(time.time()-t0)/3600:.2f}h")


def _save(model, out, tag):
    d = Path(out) / tag
    d.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().to("cpu", torch.float32) for k, v in model.state_dict().items()},
               d / "state_dict.pt")
    print(f"  [save] {d/'state_dict.pt'}", flush=True)


if __name__ == "__main__":
    main()
