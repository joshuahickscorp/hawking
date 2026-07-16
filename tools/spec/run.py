#!/usr/bin/env python3
"""Canonical speculative-decoding capture and n-gram oracle runner.

This replaces the two shell pipelines with one standard-library driver while
preserving their trace JSONL and n-gram report formats. Heavy model execution
is opt-in; every command supports ``--dry-run``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
ANALYZER = ROOT / "tools/spec/ngram_analysis.py"
LOCKED_ENV = {
    "HAWKING_QWEN_TCB": "1",
    "HAWKING_QWEN_VOCAB_PRUNE": "32000",
    "HAWKING_QWEN_Q4K_LMHEAD": "1",
    "HAWKING_QWEN_FFN_DOWN_Q4K": "1",
    "HAWKING_QWEN_Q4K_PREDEC": "1",
}


def _rooted(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _prompts(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"sample file not found: {path}")
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not prompts:
        raise SystemExit(f"no non-comment prompts in {path}")
    return prompts


def _generation_command(args: argparse.Namespace, prompt: str) -> list[str]:
    command: list[str] = []
    if shutil.which("nice"):
        command += ["nice", "-n", "19"]
    if shutil.which("taskpolicy"):
        command += ["taskpolicy", "-b"]
    command += [
        str(_rooted(args.binary)),
        "generate",
        "--weights",
        str(_rooted(args.weights)),
        "--kernel-profile",
        str(_rooted(args.profile)),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(args.tokens),
        "--temperature",
        "0",
        "--seed",
        "0",
    ]
    return command


def _run_generation(args: argparse.Namespace, prompt: str) -> tuple[list[str], str, int]:
    command = _generation_command(args, prompt)
    env = os.environ.copy()
    env.update(LOCKED_ENV)
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout,
        )
        return command, result.stdout, result.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return command, output + f"\n[TIMEOUT after {args.timeout}s]\n", 124


def _extract_generated(raw: str) -> tuple[list[int], str]:
    match = re.search(r"^\[tokens:\s*([0-9 ]+)\]$", raw, re.MULTILINE)
    if match:
        tokens = [int(value) for value in match.group(1).split()]
        if tokens:
            return tokens, "real_ids"
    response = "\n".join(
        line for line in raw.splitlines() if line.strip() and not line.startswith("[")
    )
    payload = (response or raw).encode("utf-8", errors="replace")
    return list(payload), "utf8_bytes"


def _preflight_generation(args: argparse.Namespace) -> None:
    binary = _rooted(args.binary)
    weights = _rooted(args.weights)
    profile = _rooted(args.profile)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit(f"binary not found/executable: {binary}")
    if not weights.is_file():
        raise SystemExit(f"weights not found: {weights}")
    if not profile.is_file():
        raise SystemExit(f"kernel profile not found: {profile}")


def capture(args: argparse.Namespace) -> int:
    sample_file = _rooted(args.sample_file)
    prompts = _prompts(sample_file)
    out_dir = _rooted(args.out_dir)
    traces_path = out_dir / "traces.jsonl"
    if args.dry_run:
        print(f"capture: {len(prompts)} prompts -> {traces_path}")
        print("env:", " ".join(f"{key}={value}" for key, value in LOCKED_ENV.items()))
        print("example:", shlex.join(_generation_command(args, prompts[0])))
        return 0

    _preflight_generation(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    with traces_path.open("w", encoding="utf-8") as traces:
        for index, prompt in enumerate(prompts):
            command, raw, returncode = _run_generation(args, prompt)
            log_path = out_dir / f"prompt_{index:03d}.log"
            log_path.write_text(raw, encoding="utf-8")
            if returncode:
                print(f"[{index + 1}/{len(prompts)}] FAIL rc={returncode}: {shlex.join(command)}")
                continue
            generated, encoding = _extract_generated(raw)
            if len(generated) < 2:
                print(f"[{index + 1}/{len(prompts)}] FAIL: fewer than two generated tokens")
                continue
            prompt_tokens = list(prompt.encode("utf-8", errors="replace"))
            base = len(prompt_tokens)
            record = {
                "prompt_idx": index,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated,
                "positions": list(range(base, base + len(generated))),
                "n_generated": len(generated),
                "token_encoding": encoding,
            }
            traces.write(json.dumps(record, separators=(",", ":")) + "\n")
            ok += 1
            print(
                f"[{index + 1}/{len(prompts)}] ok "
                f"n_gen={len(generated)} encoding={encoding}"
            )
    print(f"captured={ok} failed={len(prompts) - ok} output={traces_path}")
    return 0 if ok else 1


def _sequence_for_prompt(
    args: argparse.Namespace,
    prompt: str,
    index: int,
    sequence_dir: Path | None,
) -> list[int] | None:
    cached = sequence_dir / f"seq_{index}.txt" if sequence_dir else None
    if args.reuse_seqs:
        if cached and cached.is_file():
            return [int(value) for value in cached.read_text(encoding="utf-8").split()]
        return None
    _command, raw, returncode = _run_generation(args, prompt)
    if returncode:
        return None
    tokens, _encoding = _extract_generated(raw)
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(" ".join(map(str, tokens)) + "\n", encoding="utf-8")
    return tokens


def ngram(args: argparse.Namespace) -> int:
    sample_file = _rooted(args.sample_file)
    prompts = _prompts(sample_file)
    out_path = _rooted(args.out)
    sequence_dir = _rooted(args.seq_dir) if args.seq_dir else None
    if args.reuse_seqs and sequence_dir is None:
        raise SystemExit("--reuse-seqs requires --seq-dir")
    if args.dry_run:
        print(f"ngram: {len(prompts)} prompts -> {out_path}")
        print("orders:", " ".join(map(str, args.ngrams)), "min_freq:", args.min_freq)
        print("example:", shlex.join(_generation_command(args, prompts[0])))
        return 0

    if not args.reuse_seqs:
        _preflight_generation(args)
    sequences: list[list[int]] = []
    for index, prompt in enumerate(prompts):
        tokens = _sequence_for_prompt(args, prompt, index, sequence_dir)
        if tokens and len(tokens) >= 2:
            sequences.append(tokens)
            print(f"[{index + 1}/{len(prompts)}] ok n={len(tokens)}")
        else:
            print(f"[{index + 1}/{len(prompts)}] skipped")
    if not sequences:
        raise SystemExit("no usable sequences collected")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", prefix="hawking-ngram-", suffix=".txt") as handle:
        for tokens in sequences:
            handle.write(" ".join(map(str, tokens)) + "\n")
        handle.flush()
        command = [
            args.python,
            str(ANALYZER),
            "--seqs",
            handle.name,
            "--ngrams",
            *map(str, args.ngrams),
            "--min-freq",
            str(args.min_freq),
            "--json",
            str(out_path),
        ]
        return subprocess.run(command, cwd=ROOT, check=False).returncode


def selftest(_args: argparse.Namespace) -> int:
    real, real_kind = _extract_generated("text\n[tokens: 10 20 30]\n[stats] x")
    assert real == [10, 20, 30] and real_kind == "real_ids"
    fallback, fallback_kind = _extract_generated("[stats] x\nhello")
    assert fallback == list(b"hello") and fallback_kind == "utf8_bytes"

    sys.path.insert(0, str(ANALYZER.parent))
    from ngram_analysis import oracle_accept_rate_incremental

    result = oracle_accept_rate_incremental([1, 2, 1, 2, 1, 2], max_n=3)
    assert result["n_positions"] == 5
    assert 0.0 <= result["overall"] <= 1.0
    prompt = list(b"p")
    positions = list(range(len(prompt), len(prompt) + len(real)))
    assert positions == [1, 2, 3]
    print("spec runner selftest: PASS")
    return 0


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-file", default="tools/spec/sample_prompts.txt")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--binary", default="./target/release/hawking")
    parser.add_argument("--weights", default="models/qwen2.5-3b-instruct-q4_k_m.gguf")
    parser.add_argument("--profile", default="profiles/qwen3b-instruct-q4k.m3pro18.json")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--dry-run", action="store_true")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    capture_parser = commands.add_parser("capture", help="capture replay traces")
    _add_generation_args(capture_parser)
    capture_parser.add_argument("--out-dir", default="traces")
    capture_parser.set_defaults(handler=capture)

    ngram_parser = commands.add_parser("ngram", help="generate and score n-gram sequences")
    _add_generation_args(ngram_parser)
    ngram_parser.add_argument("--ngrams", type=int, nargs="+", default=[2, 3, 4])
    ngram_parser.add_argument("--min-freq", type=int, default=1)
    ngram_parser.add_argument("--out", default="reports/ngram_oracle.json")
    ngram_parser.add_argument("--seq-dir")
    ngram_parser.add_argument(
        "--reuse-seqs",
        action="store_true",
        help="read seq_<index>.txt from --seq-dir instead of running the model",
    )
    ngram_parser.add_argument("--python", default=sys.executable)
    ngram_parser.set_defaults(handler=ngram)

    test_parser = commands.add_parser("selftest", help="run CPU-only parser checks")
    test_parser.set_defaults(handler=selftest)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
