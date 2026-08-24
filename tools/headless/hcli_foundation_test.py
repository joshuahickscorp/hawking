#!/usr/bin/env python3
"""Protected HCLI foundation checks. Plain python3 + assert. No pytest fixtures.

Run:
    python3 tools/headless/hcli_foundation_test.py
    pytest tools/headless/hcli_foundation_test.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

import hcli.cli  # noqa: E402
from hcli.engine import Engine  # noqa: E402
from hcli.events import EventBus  # noqa: E402
from hcli.models import discover_models, resolve_model, selectable_models  # noqa: E402
from hcli.workspace import Workspace  # noqa: E402

KNOWN_MODEL = (
    Path.home()
    / "models"
    / "qwen3.8-27b-abliterated"
    / "Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf"
)
HCLI_BIN = Path.home() / ".local" / "bin" / "hcli"
JHCLI_BIN = Path.home() / ".local" / "bin" / "jhcli"
SOURCE_PARENT = REPO


def _reject(argv):
    import contextlib
    import io

    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            hcli.cli.parse_haider_args(argv)
    except SystemExit:
        return
    raise AssertionError(f"expected rejection for {argv!r}")


def check_grammar():
    empty = hcli.cli.parse_haider_args([])
    assert empty.runtime_count == 1, empty
    assert empty.prompt is None, empty
    assert empty.interactive is True, empty

    prompt_only = hcli.cli.parse_haider_args(["p"])
    assert prompt_only.runtime_count == 1
    assert prompt_only.prompt == "p"
    assert prompt_only.interactive is False

    n_prompt = hcli.cli.parse_haider_args(["3", "p"])
    assert n_prompt.runtime_count == 3
    assert n_prompt.prompt == "p"
    assert n_prompt.interactive is False

    saved = os.environ.pop("HCLI_RESIDENT_RUNTIME_LIMIT", None)
    try:
        max_args = hcli.cli.parse_haider_args(["max", "p"])
    finally:
        if saved is not None:
            os.environ["HCLI_RESIDENT_RUNTIME_LIMIT"] = saved
    assert max_args.prompt == "p"
    assert max_args.interactive is False
    assert max_args.max_source, "max must set max_source"
    assert 1 <= max_args.runtime_count <= 8, max_args.runtime_count
    assert max_args.max_source in {
        "HCLI_RESIDENT_RUNTIME_LIMIT",
        "machine_genome.json",
        "worker-equilibrium.json",
        "fallback",
    }, max_args.max_source

    _reject(["9", "p"])
    _reject(["0", "p"])


def check_projector_exclusion():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf").write_bytes(b"x" * 64)
        (root / "mmproj-model-bf16.gguf").write_bytes(b"x" * 64)
        found = discover_models([tmp])
        selectable = selectable_models(found)
        assert len(selectable) == 1, [m.name for m in found]
        assert selectable[0].selectable
        assert not selectable[0].is_projector
        tagged = [m for m in found if m.is_projector]
        assert len(tagged) == 1, "projector must be recorded, not silently dropped"
        chosen = resolve_model(discovered=found)
        assert chosen is not None
        assert chosen.path == selectable[0].path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "alpha-7B-Q4.gguf").write_bytes(b"x" * 64)
        (root / "beta-7B-Q4.gguf").write_bytes(b"x" * 64)
        found = discover_models([tmp])
        assert len(selectable_models(found)) == 2
        assert resolve_model(discovered=found) is None, (
            "HCLI must not silently choose one of several installed models"
        )


def _shim_python() -> str:
    if HCLI_BIN.is_file():
        for line in HCLI_BIN.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("exec ") and "-m hcli" in stripped:
                rest = stripped[5:].strip()
                if rest.startswith('"'):
                    end = rest.find('"', 1)
                    if end > 1:
                        return rest[1:end]
                parts = rest.split()
                if parts:
                    return parts[0]
    return sys.executable


def _installed_package_is_current() -> bool:
    current = Path.home() / ".local" / "share" / "hcli" / "current"
    cli_py = current / "hcli" / "cli.py"
    if not cli_py.is_file():
        return False
    try:
        text = cli_py.read_text(encoding="utf-8")
    except OSError:
        return False
    return "resolve_resident_runtime_limit" in text and "install-shims" in text


def _try_real_install() -> None:
    try:
        rc = hcli.cli.install_shims()
        if rc == 0:
            print("ok install_shims -> ~/.local/bin/{hcli,jhcli}")
            return
        print(f"WARN install_shims returned {rc}")
    except OSError as exc:
        print(
            "WARN install_shims could not write ~/.local "
            f"({type(exc).__name__}: {exc}). "
            "Create the shims with: "
            "python3 -m hcli install-shims"
        )


def _hcli_invocation() -> list:
    """Prefer the installed binary once it carries this foundation; else source."""
    if HCLI_BIN.is_file() and _installed_package_is_current():
        return [str(HCLI_BIN)]
    python = _shim_python()
    return [python, "-m", "hcli"]


def _run_headless(workdir: Path, prompt: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["HCLI_MODEL_TOKENS"] = "256"
    env["HCLI_READY_TIMEOUT"] = env.get("HCLI_READY_TIMEOUT", "300")
    # llama-server Metal init SIGSEGVs / fails inside the agent sandbox.
    # Force CPU so the live any-folder check can actually reach /health.
    env.setdefault("LLAMA_ARG_DEVICE", "none")
    env.setdefault("LLAMA_ARG_N_GPU_LAYERS", "0")
    env.setdefault("LLAMA_ARG_FIT", "off")
    env.setdefault("LLAMA_ARG_REASONING", "off")
    env.setdefault("LLAMA_ARG_THINK_BUDGET", "0")
    env.setdefault("HCLI_CTX_SIZE", "8192")
    cmd = _hcli_invocation()
    if len(cmd) >= 2 and cmd[-2] == "-m":
        env["PYTHONPATH"] = str(SOURCE_PARENT) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
    return subprocess.run(
        cmd + ["1", prompt],
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def check_any_folder():
    _try_real_install()
    if not KNOWN_MODEL.is_file():
        print(f"SKIP any_folder: model file absent at {KNOWN_MODEL}")
        return
    cmd = _hcli_invocation()
    print(f"any_folder using {cmd}")
    if JHCLI_BIN.is_file():
        if HCLI_BIN.read_text() != JHCLI_BIN.read_text():
            raise AssertionError("jhcli shim is not byte-identical to hcli")
    else:
        print(
            "WARN jhcli is not at ~/.local/bin/jhcli yet; "
            "install-shims creates it next to hcli"
        )

    prompt = (
        "Read-only question. Reply with kind=answer JSON whose content is the "
        "single word ok. Do not change any files."
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        git_dir = tmp_root / "git"
        plain = tmp_root / "plain"
        git_dir.mkdir()
        plain.mkdir()
        for workdir in (git_dir, plain):
            (workdir / "README.md").write_text("# hcli foundation any-folder\n")
        subprocess.run(
            ["git", "init"],
            cwd=str(git_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        for label, workdir in (("git", git_dir), ("non-git", plain)):
            last_err = None
            for attempt in range(1, 4):
                proc = _run_headless(workdir, prompt)
                if proc.returncode == 0:
                    last_err = None
                    break
                last_err = (
                    f"{label} hcli exit {proc.returncode} attempt {attempt}\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
            if last_err:
                raise AssertionError(last_err)
            receipts = list((workdir / ".hcli" / "receipts").glob("*.json"))
            assert receipts, f"{label} wrote no receipt under {workdir}/.hcli/receipts/"


def check_receipt_diagnostics():
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Workspace(tmp)
        engine = Engine(
            workspace=workspace,
            event_bus=EventBus(),
            runtime_count=1,
            model_name="/missing.gguf",
        )

        def boom(prompt, evidence, compiled):
            raise ConnectionError("runtime endpoint unreachable")

        engine._call_model = boom
        result = engine.execute("force a failure")
        assert result.get("status") in {"failed", "cancelled"}
        receipt_path = Path(result["receipt"])
        failed = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert failed.get("error"), failed
        assert failed.get("error_type"), failed
        assert failed.get("error_traceback"), failed

        engine2 = Engine(
            workspace=workspace,
            event_bus=EventBus(),
            runtime_count=1,
            model_name="/missing.gguf",
        )
        engine2._call_model = lambda p, e, c: {
            "kind": "answer",
            "content": "ok",
            "operations": [],
            "tests": [],
        }
        ok = engine2.execute("succeed")
        assert ok.get("status") == "completed"
        success = json.loads(Path(ok["receipt"]).read_text(encoding="utf-8"))
        assert "error" not in success, success
        assert "error_type" not in success, success
        assert "error_traceback" not in success, success


CHECKS = [
    ("grammar", check_grammar),
    ("projector_exclusion", check_projector_exclusion),
    ("any_folder", check_any_folder),
    ("receipt_diagnostics", check_receipt_diagnostics),
]


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"ok {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    return 1 if failed else 0


def test_hcli_foundation():
    """pytest entry: the same checks as running this file directly."""
    rc = main()
    assert rc == 0


if __name__ == "__main__":
    sys.exit(main())
