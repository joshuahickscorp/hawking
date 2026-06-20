#!/usr/bin/env python3
"""Deterministic prompt sampler for RWKV7 QAT checkpoints.

This is intentionally simple and slow: it reloads the full context on each new
token. That is fine for a small verification sample after a candidate checkpoint
has been frozen, and it avoids depending on any serving stack.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def load_tokenizer(hf_dir: Path):
    tok_script = hf_dir / "hf_rwkv_tokenizer.py"
    if not tok_script.exists():
        raise FileNotFoundError(f"missing RWKV tokenizer shim: {tok_script}")
    spec = importlib.util.spec_from_file_location("hf_rwkv_tokenizer", str(tok_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import tokenizer shim: {tok_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RWKV_TOKENIZER(str(hf_dir / "rwkv_vocab_v20230424.txt"))


def encode_prompt(tok: Any, prompt: str) -> list[int]:
    return tok.encodeBytes(prompt.encode("utf-8"))


def decode_tokens(tok: Any, token_ids: list[int]) -> str:
    raw = tok.decodeBytes(token_ids)
    return raw.decode("utf-8", errors="replace")


def load_prompts(path: Path, max_prompts: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt_rwkv")
            if not prompt:
                user = obj.get("user")
                if user:
                    prompt = f"<|rwkv_tokenizer_end_of_text|>User: {user}\n\nAssistant:"
            if prompt:
                rows.append({
                    "idx": obj.get("idx", len(rows)),
                    "bucket": obj.get("bucket"),
                    "prompt": prompt,
                    "gold": obj.get("gold"),
                })
            if len(rows) >= max_prompts:
                break
    if not rows:
        raise RuntimeError(f"no prompts loaded from {path}")
    return rows


def load_model(model_path: Path, hf_dir: Path, device: str):
    sys.path.insert(0, str(HERE))
    from rwkv7_load_weights import load_rwkv7

    if model_path.suffix == ".pt":
        base_st = hf_dir / "model.safetensors"
        model = load_rwkv7(str(base_st), device="cpu", dtype=torch.float32)
        state = torch.load(model_path, map_location="cpu")
        weights = {k: v for k, v in state.items() if k.endswith(".weight")}
        model.load_state_dict(weights, strict=False)
        model = model.to(device=device, dtype=torch.float32)
    else:
        model = load_rwkv7(str(model_path), device=device, dtype=torch.float32)
    model.eval()
    return model


def sample_one(
    model: torch.nn.Module,
    tok: Any,
    prompt: str,
    max_new_tokens: int,
    max_context: int,
    temperature: float,
    seed: int,
    device: str,
) -> tuple[str, list[int]]:
    ids = encode_prompt(tok, prompt)
    generated: list[int] = []
    rng = random.Random(seed)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx = (ids + generated)[-max_context:]
            input_ids = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input_ids)[0, -1, :].float()
            if temperature <= 0.0:
                next_id = int(torch.argmax(logits).item())
            else:
                torch.manual_seed(rng.randrange(0, 2**31 - 1))
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
            if next_id == 0:
                break
            generated.append(next_id)

    return decode_tokens(tok, generated), generated


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate deterministic samples from an RWKV7 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", required=True, help=".pt QAT checkpoint or safetensors")
    ap.add_argument("--hf-dir", default=str(ROOT / "models/rwkv7-g1-04-hf"))
    ap.add_argument("--prompts-jsonl", default=str(ROOT / "artifacts/rwkv7_posttrain/eval_prompts.jsonl"))
    ap.add_argument("--out", required=True, help="JSONL output path")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-prompts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--max-context", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "mps" and hasattr(torch, "mps"):
        torch.mps.manual_seed(args.seed)

    hf_dir = Path(args.hf_dir)
    model_path = Path(args.model)
    prompts_path = Path(args.prompts_jsonl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[sample] loading tokenizer from {hf_dir}")
    tok = load_tokenizer(hf_dir)
    prompts = load_prompts(prompts_path, args.max_prompts)
    print(f"[sample] loaded {len(prompts)} prompts")

    print(f"[sample] loading model {model_path} on {args.device}")
    t0 = time.time()
    model = load_model(model_path, hf_dir, args.device)
    print(f"[sample] model loaded in {time.time() - t0:.1f}s")

    with out_path.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(prompts):
            t0 = time.time()
            text, token_ids = sample_one(
                model=model,
                tok=tok,
                prompt=str(row["prompt"]),
                max_new_tokens=args.max_new_tokens,
                max_context=args.max_context,
                temperature=args.temperature,
                seed=args.seed + i,
                device=args.device,
            )
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "idx": row["idx"],
                "bucket": row.get("bucket"),
                "prompt": row["prompt"],
                "gold": row.get("gold"),
                "completion": text,
                "n_generated_tokens": len(token_ids),
                "elapsed_seconds": round(time.time() - t0, 3),
                "seed": args.seed + i,
                "temperature": args.temperature,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[sample] {i + 1}/{len(prompts)} idx={row['idx']} tokens={len(token_ids)}")

    print(f"[sample] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
