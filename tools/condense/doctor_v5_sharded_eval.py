#!/usr/bin/env python3.12
"""Resident, tensor-streamed quality evaluation for sharded reconstruction overrides.

This helper is deliberately *not* an out-of-core model runtime.  It loads one
Qwen2.5 dense parent in BF16 on CPU, then copies each decoded projection into the
already allocated model one tensor at a time.  That avoids a second model-sized
``state_dict`` while retaining the same forward-pass semantics as the original
Pass-B evaluators.  Requests above the resident admission ceiling are refused by
the ladder worker before this program is launched.

The override manifest and every referenced shard are hash-bound.  The program
prints exactly one JSON result line; diagnostics go to stderr.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any


SCHEMA = "hawking.doctor_v5_sharded_override.v1"
RESULT_SCHEMA = "hawking.doctor_v5_sharded_eval.v1"
MAX_JSON_BYTES = 32 * 1024 * 1024
SHA_RE = re.compile(r"[0-9a-f]{64}")

TEXT = """The science of operations, as derived from mathematics more especially,
is a science of itself, and has its own abstract truth and value. The bounds of
arithmetic were outstepped the moment the idea of applying the cards had occurred.
A new, a vast, and a powerful language is developed for the future use of analysis,
in which to wield its truths so that these may become of more speedy and accurate
practical application for the purposes of mankind than the means hitherto in our
possession have rendered possible. Thus not only the mental and the material, but
the theoretical and the practical in the mathematical world, are brought into more
intimate and effective connection with each other. We are not aware of its being
on record that anything partaking in the nature of what we call an analytical
engine has been hitherto proposed, or even thought of, as a practical possibility."""

QA = [
    ("Q: What is the capital of France?\nA:", ["paris"]),
    ("Q: What is the capital of Japan?\nA:", ["tokyo"]),
    ("Q: What is the chemical symbol for water?\nA:", ["h2o"]),
    ("Q: How many days are in a week?\nA:", ["seven", "7"]),
    ("Q: What planet is known as the Red Planet?\nA:", ["mars"]),
]
CLOZE = [
    ("The opposite of hot is", [" cold", " warm", " fast", " loud"], 0),
    ("Water freezes at zero degrees", [" Celsius", " kilometers", " apples"], 0),
    ("A dog is a kind of", [" animal", " mineral", " number"], 0),
    ("The sun rises in the", [" east", " west", " ceiling"], 0),
    ("Two plus two equals", [" four", " seven", " purple"], 0),
]
MATH = [
    ("2 + 2 =", "4"), ("10 - 3 =", "7"), ("6 * 7 =", "42"),
    ("100 / 4 =", "25"), ("9 + 8 =", "17"), ("12 * 12 =", "144"),
]
CODE = [
    ("def add(a, b):\n    return a +", ["b"]),
    ("for i in range(10):\n    print(", ["i"]),
    ("x = [1, 2, 3]\nx.app", ["end"]),
    ("import nu", ["mpy"]),
]


class EvalError(RuntimeError):
    pass


def _hash_file(path: Path) -> tuple[str, int]:
    if path.is_symlink():
        raise EvalError(f"symlinked override forbidden: {path}")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise EvalError(f"override is not a regular file: {path}")
    digest, total = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            total += len(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != \
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) \
            or total != after.st_size:
        raise EvalError(f"override changed while hashing: {path}")
    return digest.hexdigest(), total


def _load_manifest(path: Path) -> list[Path]:
    if path.stat().st_size > MAX_JSON_BYTES or path.is_symlink():
        raise EvalError("override manifest is invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema", "shards"} \
            or value.get("schema") != SCHEMA:
        raise EvalError("override manifest schema/keys are invalid")
    rows = value.get("shards")
    if not isinstance(rows, list) or not rows:
        raise EvalError("override manifest has no shards")
    result: list[Path] = []
    ordinals: list[int] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
                "ordinal", "path", "sha256", "bytes", "tensor_count"}:
            raise EvalError("override shard binding keys are invalid")
        if not isinstance(row["ordinal"], int) or isinstance(row["ordinal"], bool):
            raise EvalError("override ordinal is invalid")
        if not isinstance(row["tensor_count"], int) or row["tensor_count"] < 0:
            raise EvalError("override tensor_count is invalid")
        raw = Path(row["path"])
        shard = raw.resolve(strict=True)
        digest, size = _hash_file(shard)
        if not SHA_RE.fullmatch(str(row["sha256"])) \
                or digest != row["sha256"] or size != row["bytes"]:
            raise EvalError(f"override identity mismatch: {shard}")
        ordinals.append(row["ordinal"])
        result.append(shard)
    if ordinals != list(range(len(ordinals))):
        raise EvalError("override ordinals are not a contiguous ordered sequence")
    return result


def _apply_overrides(model: Any, paths: list[Path], dtype: Any) -> int:
    import torch
    from safetensors import safe_open

    state = model.state_dict(keep_vars=True)
    seen: set[str] = set()
    copied = 0
    with torch.no_grad():
        for path in paths:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in seen:
                        raise EvalError(f"duplicate override tensor: {name}")
                    seen.add(name)
                    destination = state.get(name)
                    if destination is None:
                        raise EvalError(f"override tensor absent from model: {name}")
                    source = handle.get_tensor(name)
                    if tuple(source.shape) != tuple(destination.shape):
                        raise EvalError(f"override shape mismatch: {name}")
                    destination.copy_(source.to(dtype=dtype))
                    copied += 1
                    del source
    if copied == 0:
        raise EvalError("override contains no model tensors")
    return copied


def _load_model(model_dir: Path, manifest: Path | None) -> tuple[Any, Any, Any, int]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_name = os.environ.get("DOCTOR_DTYPE", "bfloat16")
    if dtype_name not in {"bfloat16", "float32"}:
        raise EvalError("resident evaluator permits only bfloat16 or float32")
    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=dtype, attn_implementation="eager",
        low_cpu_mem_usage=True, local_files_only=True,
    ).eval()
    copied = 0
    if manifest is not None:
        copied = _apply_overrides(model, _load_manifest(manifest), dtype)
    return model, tokenizer, torch, copied


def _ppl(model: Any, tokenizer: Any, torch: Any) -> dict[str, Any]:
    text = TEXT
    text_path = os.environ.get("PPL_TEXT")
    if text_path and Path(text_path).is_file():
        text = Path(text_path).read_text(encoding="utf-8", errors="ignore")
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :2048]
    with torch.no_grad():
        out = model(ids, labels=ids)
    loss = float(out.loss.item())
    return {"ppl": math.exp(loss), "loss": loss, "ntok": int(ids.numel())}


def _capability(model: Any, tok: Any, torch: Any) -> dict[str, Any]:
    def greedy(prompt: str, max_new: int = 6) -> str:
        enc = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                enc.input_ids, attention_mask=enc.attention_mask,
                max_new_tokens=max_new, do_sample=False, num_beams=1,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

    def candidate_loss(context: str, candidate: str) -> float:
        ctx = tok(context, return_tensors="pt").input_ids
        full = tok(context + candidate, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(full).logits
        cont = full[0, ctx.shape[1]:]
        if cont.numel() == 0:
            return float("inf")
        lp = torch.log_softmax(logits[0, ctx.shape[1] - 1:-1], dim=-1)
        return float(-lp[range(cont.numel()), cont].mean())

    scores: dict[str, float] = {}
    scores["qa"] = sum(
        any(greedy(prompt).strip().lower().lstrip(".:- ").startswith(a) for a in answers)
        for prompt, answers in QA
    ) / len(QA)
    scores["cloze"] = sum(
        min(range(len(candidates)), key=lambda i: candidate_loss(context, candidates[i])) == correct
        for context, candidates, correct in CLOZE
    ) / len(CLOZE)
    math_hits = 0
    for prompt, answer in MATH:
        words = greedy(prompt).strip().split()
        if words and words[0].rstrip(".") == answer:
            math_hits += 1
    scores["math"] = math_hits / len(MATH)
    scores["code"] = sum(
        any(greedy(prefix, 3).strip().startswith(a) for a in accepted)
        for prefix, accepted in CODE
    ) / len(CODE)
    counts = {"qa": len(QA), "cloze": len(CLOZE), "math": len(MATH), "code": len(CODE)}
    n = sum(counts.values())
    aggregate = sum(scores[key] * counts[key] for key in scores) / n
    return {"per_task": {k: round(v, 4) for k, v in scores.items()},
            "task_n": counts, "aggregate": round(aggregate, 4), "n": n}


def _selftest() -> None:
    assert SHA_RE.fullmatch("0" * 64)
    assert not SHA_RE.fullmatch("0" * 63)
    assert SCHEMA.endswith(".v1") and RESULT_SCHEMA.endswith(".v1")
    print(json.dumps({"status": "ok", "schema": RESULT_SCHEMA}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--mode", choices=("ppl", "capability"), required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--override-manifest", type=Path)
    run.add_argument("--label", required=True)
    sub.add_parser("selftest")
    args = parser.parse_args(argv)
    try:
        if args.command == "selftest":
            _selftest()
            return 0
        model_dir = args.model_dir.resolve(strict=True)
        manifest = args.override_manifest.resolve(strict=True) if args.override_manifest else None
        model, tokenizer, torch, copied = _load_model(model_dir, manifest)
        metrics = _ppl(model, tokenizer, torch) if args.mode == "ppl" \
            else _capability(model, tokenizer, torch)
        print(json.dumps({
            "schema": RESULT_SCHEMA, "mode": args.mode, "label": args.label,
            "model": str(model_dir), "override_manifest": str(manifest) if manifest else None,
            "override_tensor_count": copied, **metrics,
        }, sort_keys=True))
        return 0
    except (EvalError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
