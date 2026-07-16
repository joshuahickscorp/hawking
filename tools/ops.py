#!/usr/bin/env python3
"""Small, parameterized repository operations.

This is the canonical home for operational wrappers whose behavior is mostly a
profile plus a shared transport or process-control primitive.  Keep scientific
and campaign logic in its owning package; use this module to avoid duplicating
shell plumbing around it.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
UNVERIFIED = "UNVERIFIED-PIN-AFTER-FIRST-DOWNLOAD"


@dataclass(frozen=True)
class ModelProfile:
    target: str
    size_gb: int
    reserve_gb: int
    sha256: str
    mirrors: tuple[tuple[str, str], ...]


MODEL_PROFILES = {
    "deepseek-v2-lite-q4": ModelProfile(
        target="deepseek-v2-lite-q4.gguf",
        size_gb=9,
        reserve_gb=0,
        sha256="5d33e5f045c7a03351c319aafc8afdad94b69d07bb68f36dc9bb5af340b343a4",
        mirrors=(
            (
                "mradermacher/DeepSeek-V2-Lite-Chat-GGUF",
                "DeepSeek-V2-Lite-Chat.Q4_K_M.gguf",
            ),
            (
                "legraphista/DeepSeek-V2-Lite-Chat-IMat-GGUF",
                "DeepSeek-V2-Lite-Chat.Q4_K.gguf",
            ),
        ),
    ),
    "mixtral-8x7b-instruct-q4": ModelProfile(
        target="mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
        size_gb=26,
        reserve_gb=30,
        sha256=UNVERIFIED,
        mirrors=(
            (
                "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
                "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
            ),
            (
                "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF",
                "Mixtral-8x7B-Instruct-v0.1-Q4_K_M.gguf",
            ),
        ),
    ),
}


@dataclass(frozen=True)
class CommandStep:
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TaskProfile:
    description: str
    steps: tuple[CommandStep, ...]


def _task(description: str, *argv: str) -> TaskProfile:
    return TaskProfile(description, (CommandStep(tuple(argv)),))


TASK_PROFILES = {
    "bench": {
        "batch": _task(
            "continuous-batching aggregate throughput gate",
            "bash", "tools/bench/batch_aggregate_bench.sh",
        ),
        "clean": _task(
            "authoritative clean benchmark",
            "bash", "tools/bench/clean_bench.sh",
        ),
        "clean-room": _task(
            "clean-room absolute metrics plus energy anchor",
            "bash", "tools/bench/clean_room_batch.sh",
        ),
        "coexist": _task(
            "contamination-robust development benchmark",
            "bash", "tools/bench/coexist_bench.sh",
        ),
        "condense-scorecard": _task(
            "compression, quality, and RAM scorecard",
            "python3", "tools/bench/condense_scorecard.py",
        ),
        "energy": _task(
            "absolute joules per token",
            "bash", "tools/bench/measure_joules.sh",
        ),
        "energy-phases": _task(
            "phase and domain energy attribution",
            "bash", "tools/bench/phase_joules.sh",
        ),
        "oracle-attention": _task(
            "attention working-set concentration oracle",
            "python3", "tools/bench/oracle_attn_mass.py",
        ),
        "oracle-draft": _task(
            "small-dense-draft acceptance oracle",
            "python3", "tools/bench/draft_accept_oracle.py",
        ),
        "oracle-prefix": _task(
            "exact and semantic prefix-cache opportunity oracle",
            "python3", "tools/bench/oracle_prefix_cache.py",
        ),
        "oracle-qtip": _task(
            "lower-bpw QTIP quality oracle",
            "python3", "tools/bench/oracle_qtip_quality.py",
        ),
        "oracle-spec": _task(
            "n-gram and prompt-lookup acceptance oracle",
            "python3", "tools/bench/oracle_spec_accept.py",
        ),
        "oracle-vocab": _task(
            "certified output-vocabulary coverage oracle",
            "python3", "tools/bench/oracle_vocab_coverage.py",
        ),
        "paired": _task(
            "parameterized parity and ABBA performance gate",
            "bash", "tools/bench/paired_lever.sh",
        ),
        "quality": _task(
            "paired token-drift quality gate",
            "bash", "tools/bench/quality_oracle.sh",
        ),
        "ratios": _task(
            "CI warm-median throughput and argmax identity gate",
            "bash", "tools/bench/ratios.sh",
        ),
        "report-card": _task(
            "Hawking/llama serving, energy, and readback report card",
            "bash", "tools/bench/report_card.sh",
        ),
        "serve-matrix": _task(
            "OpenAI-compatible concurrency matrix",
            "python3", "tools/bench/serve_concurrency_matrix.py",
        ),
        "trace-analyze": _task(
            "TCB trace threshold and bandwidth analysis",
            "python3", "tools/bench/analyze_tcb_trace.py",
        ),
    },
    "training": {
        "awq": _task(
            "activation-aware smoothing calibration",
            "python3", "tools/training/awq_calibrate.py",
        ),
        "corpus-analyze": _task(
            "derive vocab, residual, and expert-load artifacts",
            "python3", "tools/training/analyze_corpus.py",
        ),
        "corpus-build": _task(
            "capture the V2-Lite calibration corpus",
            "python3", "tools/training/build_corpus.py",
        ),
        "eagle-quantize": _task(
            "quantize an Eagle5 head and run parity",
            "python3", "tools/training/eagle5_quantize.py",
        ),
        "eagle-tau": _task(
            "evaluate Eagle5 accepted-prefix depth",
            "python3", "tools/training/eagle5_tau_eval.py",
        ),
        "eagle-tau-sweep": _task(
            "Event Horizon layer-triplet tau sweep",
            "python3", "tools/training/eh_eagle_tau_sweep.py",
        ),
        "eagle-train": _task(
            "train the current Eagle5 activation head",
            "python3", "tools/training/eagle5_train.py",
        ),
        "rwkv-corpus": _task(
            "build RWKV-7 SFT and preference corpora",
            "python3", "tools/training/rwkv7_build_corpus.py",
        ),
        "rwkv-eval": _task(
            "canonical RWKV-7 perplexity evaluation",
            "python3", "tools/training/rwkv7_eval_ppl.py",
        ),
        "rwkv-export-hf": _task(
            "export a trained RWKV-7 checkpoint to HF/GGUF",
            "python3", "tools/training/rwkv7_export_hf.py",
        ),
        "rwkv-export-strand": _task(
            "export a QAT checkpoint to STR2/TQ",
            "python3", "tools/training/rwkv7_export_strand.py",
        ),
        "rwkv-qat": _task(
            "low-bit RWKV-7 quantization-aware training",
            "python3", "tools/training/rwkv7_qat.py",
        ),
        "rwkv-sft": _task(
            "parity-verified RWKV-7 SFT",
            "python3", "tools/training/rwkv7_sft_torch.py",
        ),
        "rwkv-spec": _task(
            "draft shrink-frontier and spec-decode hardening report",
            "python3", "tools/training/rwkv7_spec_hardening.py",
        ),
        "rwkv-teacher": _task(
            "capture top-k teacher logits for distillation",
            "python3", "tools/training/rwkv7_capture_teacher_logits.py",
        ),
        "rwkv-train-draft": _task(
            "train a selected compact RWKV-7 draft",
            "python3", "tools/training/rwkv7_train_draft.py",
        ),
        "rwkv-selftest": _task(
            "CPU-only RWKV recurrence and batching parity",
            "python3", "tools/training/test_rwkv7.py",
        ),
    },
    "orchestrator": {
        "eagle-bench": _task(
            "baseline-versus-Eagle verify-window sweep",
            "python3", "tools/orchestrator/bench_head.py",
        ),
        "eagle-frozen": _task(
            "build runtime-faithful frozen Eagle inputs",
            "python3", "tools/orchestrator/build_frozen_gguf.py",
        ),
        "eagle-pack": _task(
            "pack quantized residual captures into parquet",
            "python3", "tools/orchestrator/pack_corpus.py",
        ),
        "eagle-prompts": _task(
            "generate a diverse residual-capture prompt corpus",
            "python3", "tools/orchestrator/gen_prompts.py",
        ),
        "ffn-measure": _task(
            "measure captured FFN sparsity",
            "python3", "tools/orchestrator/measure_ffn_sparsity.py",
        ),
        "ffn-pack": _task(
            "pack FFN capture streams into parquet",
            "python3", "tools/orchestrator/pack_ffn.py",
        ),
    },
    "spec": {
        "capture": _task(
            "capture verifier traces with stable artifact format",
            "python3", "tools/spec/run.py", "capture",
        ),
        "ngram": _task(
            "generate and score the n-gram feasibility oracle",
            "python3", "tools/spec/run.py", "ngram",
        ),
        "replay": _task(
            "replay candidate draft policies offline",
            "python3", "tools/spec/replay_oracle.py",
        ),
        "selftest": _task(
            "CPU-only speculative runner checks",
            "python3", "tools/spec/run.py", "selftest",
        ),
    },
    "strand": {
        "autotune": _task(
            "idle-gated machine-specific performance sweep",
            "bash", "tools/strand/scripts/autotune.sh",
        ),
        "autotune-apply": _task(
            "print opt-in settings from the tuned profile",
            "python3", "tools/strand/scripts/autotune.py", "apply",
        ),
        "eval": _task(
            "canonical WikiText-2 PPL evaluation and ledger",
            "bash", "tools/strand/scripts/strand-eval",
        ),
        "eval-selftest": _task(
            "torch-free STRAND eval and self-location suite",
            "python3", "-m", "unittest", "discover",
            "-s", "tools/strand/tools/strand_eval", "-p", "test_*.py", "-v",
        ),
        "package-macos": _task(
            "build and sign the STRAND macOS archive opener",
            "bash", "tools/strand/scripts/packaging/macos/make-app.sh",
        ),
        "ppl-7b": _task(
            "7B-safe quantize, reconstruct, and PPL workflow",
            "bash", "tools/strand/scripts/strand-7b-ppl.sh",
        ),
        "qat": _task(
            "selective STRAND quantization-aware training",
            "python3", "tools/strand/scripts/strand-qat.py",
        ),
        "replay": _task(
            "idle exactness and provenance invariant sweep",
            "bash", "tools/strand/scripts/replay.sh",
        ),
        "rung-screen": _task(
            "per-tensor rung screen and water-filling allocator",
            "python3", "tools/strand/scripts/rung-screen.py",
        ),
        "selfdesc": _task(
            "stdlib-only STR2 self-description verifier",
            "python3", "tools/strand/scripts/selfdesc-interpreter.py",
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: Sequence[str], *, env: dict[str, str] | None = None) -> int:
    return subprocess.run(command, env=env, check=False).returncode


def _hf_identity() -> str | None:
    if shutil.which("hf") is None:
        return None
    result = subprocess.run(
        ["hf", "auth", "whoami"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.splitlines()[0] if result.returncode == 0 and result.stdout else None


def _verify_existing(profile: ModelProfile, target: Path) -> bool:
    if not target.is_file():
        return False
    print(f"[model] already present: {target}")
    if profile.sha256 == UNVERIFIED:
        print(f"[model] unpinned sha256: {_sha256(target)}")
        return True
    actual = _sha256(target)
    if actual != profile.sha256:
        raise SystemExit(
            "[model] sha256 mismatch\n"
            f"  expected: {profile.sha256}\n"
            f"  actual:   {actual}\n"
            "delete the file or pass --replace to fetch a fresh copy"
        )
    print(f"[model] sha256 OK ({actual})")
    return True


def _finish_download(profile: ModelProfile, target: Path) -> None:
    actual = _sha256(target)
    print(f"[model] complete: {target}")
    print(f"[model] sha256: {actual}")
    if profile.sha256 != UNVERIFIED and actual != profile.sha256:
        target.unlink(missing_ok=True)
        raise SystemExit(
            f"[model] downloaded sha256 mismatch; removed {target}\n"
            f"  expected: {profile.sha256}\n"
            f"  actual:   {actual}"
        )
    if profile.sha256 == UNVERIFIED:
        print("[model] profile is unpinned; record this hash after verification")


def model_fetch(args: argparse.Namespace) -> int:
    profile = MODEL_PROFILES[args.profile]
    MODELS.mkdir(parents=True, exist_ok=True)
    target = MODELS / profile.target
    if target.exists() and not args.replace and _verify_existing(profile, target):
        return 0
    if args.replace:
        target.unlink(missing_ok=True)

    if profile.reserve_gb:
        free_gb = shutil.disk_usage(MODELS).free // 1_000_000_000
        if free_gb < profile.reserve_gb:
            raise SystemExit(
                f"[model] only {free_gb} GB free; {profile.reserve_gb} GB required"
            )
        print(f"[model] {free_gb} GB free")

    identity = _hf_identity()
    if args.transport == "hf" and identity is None:
        raise SystemExit("[model] Hugging Face CLI login required: hf auth login")
    print(f"[model] target: {target} (~{profile.size_gb} GB)")
    print(f"[model] Hugging Face user: {identity or 'not authenticated'}")

    env = os.environ.copy()
    if args.fast:
        if importlib.util.find_spec("hf_transfer") is None:
            raise SystemExit(
                '[model] --fast requires: python3 -m pip install -U '
                '"huggingface_hub[hf_transfer]"'
            )
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    for repo, filename in profile.mirrors:
        print(f"[model] trying {repo} / {filename}")
        source = MODELS / filename
        if identity is not None and args.transport in {"auto", "hf"}:
            if _run(
                ["hf", "download", repo, filename, "--local-dir", str(MODELS)],
                env=env,
            ) == 0 and source.is_file():
                source.replace(target)
                _finish_download(profile, target)
                return 0
        if args.transport in {"auto", "curl"} and shutil.which("curl"):
            url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
            if _run(
                ["curl", "-L", "-C", "-", "--fail", "--progress-bar", "-o", str(target), url]
            ) == 0:
                _finish_download(profile, target)
                return 0
            target.unlink(missing_ok=True)
        print("[model] mirror failed")
    raise SystemExit("[model] all mirrors failed")


def bench_signal(action: str) -> int:
    run_root = ROOT / "artifacts" / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    if action == "pause":
        (run_root / "PAUSE").touch()
        (run_root / "RESUME").unlink(missing_ok=True)
        print("paused — pipeline will halt before the next stage/trial")
    else:
        (run_root / "RESUME").touch()
        print("resume signal sent — pipelines poll every 10 seconds")
    return 0


def _slm_process_group() -> list[int]:
    result = subprocess.run(
        ["pgrep", "-f", "overnight_shift.py"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    first = next((line for line in result.stdout.splitlines() if line.strip()), None)
    if first is None:
        return []
    pgid = subprocess.run(
        ["ps", "-o", "pgid=", "-p", first],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout.strip()
    if not pgid:
        return [int(first)]
    members = subprocess.run(
        ["pgrep", "-g", pgid],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout.split()
    return [int(pid) for pid in members] or [int(first)]


def with_slm_paused(command: list[str]) -> int:
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("bench with-slm-paused requires a command after --")
    pids = _slm_process_group()
    try:
        if pids:
            print(f"[slm] pausing process tree: {' '.join(map(str, pids))}", file=sys.stderr)
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGSTOP)
                except ProcessLookupError:
                    pass
            time.sleep(2)
        else:
            print("[slm] no overnight_shift.py process found", file=sys.stderr)
        return subprocess.run(command, check=False).returncode
    finally:
        if pids:
            print(f"[slm] resuming process tree: {' '.join(map(str, pids))}", file=sys.stderr)
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass


def _profile_command(args: argparse.Namespace) -> int:
    profile = TASK_PROFILES[args.profile_group][args.profile]
    extra = list(args.argv)
    passthrough: list[str] = []
    index = 0
    while index < len(extra):
        item = extra[index]
        if item == "--":
            passthrough.extend(extra[index + 1:])
            break
        if item == "--dry-run":
            args.dry_run = True
        elif item == "--env":
            index += 1
            if index >= len(extra):
                raise SystemExit("--env requires KEY=VALUE")
            args.env.append(extra[index])
        elif item.startswith("--env="):
            args.env.append(item.removeprefix("--env="))
        else:
            passthrough.append(item)
        index += 1
    extra = passthrough
    if extra and len(profile.steps) != 1:
        raise SystemExit("extra arguments are only valid for single-step profiles")
    env = os.environ.copy()
    overrides: dict[str, str] = {}
    for item in args.env:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise SystemExit(f"invalid --env {item!r}; expected KEY=VALUE")
        overrides[key] = value
    env.update(overrides)

    for index, step in enumerate(profile.steps, start=1):
        command = [*step.argv, *(extra if len(profile.steps) == 1 else ())]
        step_env = dict(step.env)
        step_env.update(overrides)
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(step_env.items()))
        rendered = shlex.join(command)
        print(f"[{args.profile_group}:{args.profile} {index}/{len(profile.steps)}] "
              f"{prefix + ' ' if prefix else ''}{rendered}")
        if args.dry_run:
            continue
        run_env = env.copy()
        run_env.update(step.env)
        returncode = subprocess.run(command, cwd=ROOT, env=run_env, check=False).returncode
        if returncode:
            return returncode
    return 0


def _profile_show(args: argparse.Namespace) -> int:
    profile = TASK_PROFILES[args.profile_group][args.profile]
    print(profile.description)
    for step in profile.steps:
        prefix = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in sorted(step.env)
        )
        print(f"{prefix + ' ' if prefix else ''}{shlex.join(step.argv)}")
    return 0


def _profile_list(args: argparse.Namespace) -> int:
    for name, profile in sorted(TASK_PROFILES[args.profile_group].items()):
        print(f"{name:20} {profile.description}")
    return 0


def profiles_selftest(_args: argparse.Namespace) -> int:
    failures: list[str] = []
    for group, profiles in TASK_PROFILES.items():
        if not profiles:
            failures.append(f"{group}: no profiles")
        for name, profile in profiles.items():
            if not profile.steps:
                failures.append(f"{group}:{name}: no steps")
            for step in profile.steps:
                for token in step.argv:
                    if token.startswith("tools/") and token.endswith((".py", ".sh")):
                        if not (ROOT / token).is_file():
                            failures.append(f"{group}:{name}: missing {token}")

    workload_path = ROOT / "tools/bench/workloads.json"
    try:
        workloads = json.loads(workload_path.read_text(encoding="utf-8"))["profiles"]
        expected = {
            "cache_miss_taxonomy",
            "long_prompt_burst",
            "mixed_latency",
            "shared_agent",
            "spec_decode_gate",
        }
        if set(workloads) != expected:
            failures.append("bench workload profile set mismatch")
    except Exception as exc:
        failures.append(f"invalid benchmark workloads: {exc}")

    if failures:
        print("\n".join(f"FAIL: {failure}" for failure in failures), file=sys.stderr)
        return 1
    total = sum(len(profiles) for profiles in TASK_PROFILES.values())
    print(f"ops profile selftest: PASS ({total} profiles)")
    return 0


def _add_profile_commands(parent: argparse.ArgumentParser, group: str) -> None:
    commands = parent.add_subparsers(dest=f"{group}_command", required=True)
    listing = commands.add_parser("profiles", help="list canonical profiles")
    listing.set_defaults(handler=_profile_list, profile_group=group)

    show = commands.add_parser("show", help="show a profile without running it")
    show.add_argument("profile", choices=sorted(TASK_PROFILES[group]))
    show.set_defaults(handler=_profile_show, profile_group=group)

    run = commands.add_parser("run", help="run a canonical profile")
    run.add_argument("profile", choices=sorted(TASK_PROFILES[group]))
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("argv", nargs=argparse.REMAINDER)
    run.set_defaults(handler=_profile_command, profile_group=group)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    model = commands.add_parser("model", help="fetch pinned model profiles")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    fetch = model_commands.add_parser("fetch")
    fetch.add_argument("profile", choices=sorted(MODEL_PROFILES))
    fetch.add_argument("--transport", choices=("auto", "hf", "curl"), default="auto")
    fetch.add_argument("--fast", action="store_true", help="enable hf_transfer")
    fetch.add_argument("--replace", action="store_true")
    fetch.set_defaults(handler=model_fetch)

    bench = commands.add_parser("bench", help="benchmark process controls")
    bench_commands = bench.add_subparsers(dest="bench_command", required=True)
    for action in ("pause", "resume"):
        child = bench_commands.add_parser(action)
        child.set_defaults(handler=lambda _args, value=action: bench_signal(value))
    paused = bench_commands.add_parser("with-slm-paused")
    paused.add_argument("argv", nargs=argparse.REMAINDER)
    paused.set_defaults(handler=lambda args: with_slm_paused(args.argv))

    listing = bench_commands.add_parser("profiles", help="list canonical profiles")
    listing.set_defaults(handler=_profile_list, profile_group="bench")
    show = bench_commands.add_parser("show", help="show a profile without running it")
    show.add_argument("profile", choices=sorted(TASK_PROFILES["bench"]))
    show.set_defaults(handler=_profile_show, profile_group="bench")
    run = bench_commands.add_parser("run", help="run a canonical profile")
    run.add_argument("profile", choices=sorted(TASK_PROFILES["bench"]))
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("argv", nargs=argparse.REMAINDER)
    run.set_defaults(handler=_profile_command, profile_group="bench")

    for group in ("training", "orchestrator", "spec", "strand"):
        parent = commands.add_parser(group, help=f"{group} canonical profiles")
        _add_profile_commands(parent, group)

    selftest = commands.add_parser("selftest", help="validate profile wiring")
    selftest.set_defaults(handler=profiles_selftest)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
