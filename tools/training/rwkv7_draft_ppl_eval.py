"""Evaluate a custom RWKV-7 draft checkpoint.

Outputs a single JSON line with wikitext2 PPL and greedy draft/teacher accept
rate on SFT prompts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from rwkv7_custom_configs import CUSTOM_VARIANTS
from rwkv7_eval_ppl import compute_ppl, load_corpus, load_tokenizer
from rwkv7_load_weights import load_rwkv7
from rwkv7_torch_model import RWKV7Config, RWKV7Model


def resolve_checkpoint(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "state_dict.pt"
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    return p


def load_meta(ckpt: Path) -> dict:
    meta = ckpt.parent / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    return {}


def infer_step(ckpt: Path, meta: dict) -> int | None:
    if "step" in meta:
        return int(meta["step"])
    for part in (ckpt.parent.name, ckpt.name):
        m = re.search(r"step_(\d+)", part)
        if m:
            return int(m.group(1))
    return None


def load_student(variant: str, ckpt: Path, device: str, use_chunked: bool, chunk_size: int) -> RWKV7Model:
    cfg = replace(CUSTOM_VARIANTS[variant], use_chunked=use_chunked, chunk_size=chunk_size)
    model = RWKV7Model(cfg)
    sd = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def read_sft_prompts(data: Path, encode, limit: int, max_length: int) -> list[list[int]]:
    eos = 0
    prompts: list[list[int]] = []
    with data.open("r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= limit:
                break
            obj = json.loads(line)
            msgs = obj.get("messages") or []
            user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
            if not user:
                continue
            ids = [eos] + encode(f"User: {user}\n\nAssistant:")
            if len(ids) > max_length:
                ids = ids[-max_length:]
            if len(ids) >= 1:
                prompts.append(ids)
    return prompts


@torch.no_grad()
def greedy_accept_rate(
    student: RWKV7Model,
    teacher: RWKV7Model,
    prompts: list[list[int]],
    draft_k: int,
    device: str,
) -> float:
    matched = 0
    total = 0
    for prompt in prompts:
        context = list(prompt)
        for _ in range(draft_k):
            x_student = torch.tensor([context], dtype=torch.long, device=device)
            student_token = int(student(x_student)[0, -1].argmax().item())

            x_teacher = torch.tensor([context], dtype=torch.long, device=device)
            teacher_token = int(teacher(x_teacher)[0, -1].argmax().item())

            matched += int(student_token == teacher_token)
            total += 1
            context.append(student_token)
    return matched / total if total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="state_dict.pt or checkpoint directory")
    ap.add_argument("--variant", default=None, choices=tuple(CUSTOM_VARIANTS))
    ap.add_argument("--teacher", default=str(ROOT / "models/rwkv7-g1-04-hf/model.safetensors"))
    ap.add_argument("--hf-dir", default=str(ROOT / "models/rwkv7-g1-04-hf"))
    ap.add_argument("--data", default=str(ROOT / "artifacts/rwkv7_posttrain/sft.jsonl"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--stride", type=int, default=4096)
    ap.add_argument("--accept-seqs", type=int, default=200)
    ap.add_argument("--accept-max-length", type=int, default=256)
    ap.add_argument("--draft-k", type=int, default=4)
    ap.add_argument("--use-chunked", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=32)
    args = ap.parse_args()

    ckpt = resolve_checkpoint(args.checkpoint)
    meta = load_meta(ckpt)
    variant = args.variant or meta.get("variant")
    if not variant:
        raise ValueError("--variant is required when checkpoint meta.json does not name it")

    print(f"[eval] loading student {variant} from {ckpt}", file=sys.stderr, flush=True)
    student = load_student(variant, ckpt, args.device, args.use_chunked, args.chunk_size)
    params_m = sum(p.numel() for p in student.parameters()) / 1e6

    print("[eval] loading tokenizer and wikitext2", file=sys.stderr, flush=True)
    encode = load_tokenizer(args.hf_dir)
    text = load_corpus("wikitext2", None)
    token_ids = encode(text)
    ppl, _, _ = compute_ppl(
        student,
        token_ids,
        n_tokens=args.tokens,
        stride=args.stride,
        device=args.device,
        dtype=torch.float32,
        teacher_logits_path=None,
    )

    print(f"[eval] loading teacher {args.teacher}", file=sys.stderr, flush=True)
    teacher = load_rwkv7(args.teacher, RWKV7Config(), device=args.device, dtype=torch.float32)
    teacher.eval()
    prompts = read_sft_prompts(Path(args.data), encode, args.accept_seqs, args.accept_max_length)
    accept = greedy_accept_rate(student, teacher, prompts, args.draft_k, args.device)

    record = {
        "variant": variant,
        "wikitext2_ppl": round(float(ppl), 6),
        "draft_accept_rate": round(float(accept), 6),
        "params_M": round(params_m, 6),
        "step": infer_step(ckpt, meta),
    }
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

