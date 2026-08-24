from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple


MAX_RUNTIME_COUNT = 8


def _prog_name() -> str:
    base = os.path.basename(sys.argv[0] if sys.argv else "hcli")
    if base in ("hcli", "jhcli"):
        return base
    return "hcli"


def _clamp_runtime_count(n: int) -> int:
    return max(1, min(int(n), MAX_RUNTIME_COUNT))


def _cli_limit_source(raw: str) -> str:
    """Map ``resolve_runtime_limits`` source strings onto the CLI labels."""
    if raw.startswith("env:"):
        return raw.split(":", 1)[-1]
    if "machine_genome.json" in raw:
        return "machine_genome.json"
    if "MACHINE_GENOME.json" in raw:
        return "MACHINE_GENOME.json"
    if "worker-equilibrium.json" in raw:
        return "worker-equilibrium.json"
    return raw


def resolve_resident_runtime_limit(
    start_dir: Optional[str] = None,
) -> Tuple[int, str]:
    """Resolve `hcli max` resident runtime count.

    Adapter over ``hcli.machine.resolve_runtime_limits`` so CLI ``max``
    cannot pick a STALE genome the runtime pool would refuse. Clamps to
    ``MAX_RUNTIME_COUNT`` because the positional N grammar is 1-8.

    Verified caller: ``parse_haider_args`` (token ``max``).
    """
    from .machine import resolve_runtime_limits

    start = start_dir or os.getcwd()
    resolved = resolve_runtime_limits(repo_root=start, start_dir=start)
    return _clamp_runtime_count(resolved.resident_limit), _cli_limit_source(
        resolved.resident_source
    )


def parse_haider_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=_prog_name(),
        description="HCLI — autonomous local model engineering",
    )
    parser.add_argument("n_or_prompt", nargs="?", type=str, default=None,
                        help="Number of runtimes (1-8), 'max', or immediate mission prompt")
    parser.add_argument("prompt", nargs="?", type=str, default=None,
                        help="Immediate mission prompt (when N is supplied positionally)")
    parser.add_argument("--task", type=str, default=None, help="(legacy) mission text")
    parser.add_argument("--task-file", type=str, default=None, help="(legacy) path to mission file")
    parser.add_argument("--model", type=str, default=None, help="Path to GGUF model")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--max-turns", type=int, default=10, help="Max observation turns")
    parser.add_argument("--max-cycles", type=int, default=3, help="Max mission cycles")
    parser.add_argument("--workspace", type=str, default=None, help="Workspace root")

    args = parser.parse_args(argv)

    n = 1
    prompt = None
    max_source: Optional[str] = None

    if args.n_or_prompt is not None:
        token = args.n_or_prompt
        if token.lower() == "max":
            n, max_source = resolve_resident_runtime_limit(
                args.workspace or os.getcwd()
            )
        else:
            try:
                n = int(token)
                if n < 1 or n > MAX_RUNTIME_COUNT:
                    parser.error(f"N must be 1-8, got {n}")
            except ValueError:
                prompt = token

    if args.prompt is not None:
        if prompt is not None:
            parser.error("Too many positional arguments")
        prompt = args.prompt

    if args.task:
        if prompt is not None:
            parser.error("Cannot use both positional prompt and --task")
        prompt = args.task
    if args.task_file:
        if prompt is not None:
            parser.error("Cannot use both positional prompt and --task-file")
        try:
            prompt = open(args.task_file).read().strip()
        except Exception as e:
            parser.error(f"Cannot read task file: {e}")

    args.runtime_count = n
    args.prompt = prompt
    args.interactive = prompt is None
    args.max_source = max_source
    return args


def _shim_python() -> str:
    """Reuse the interpreter the existing hcli shim already execs, if any."""
    existing = Path.home() / ".local" / "bin" / "hcli"
    if existing.is_file():
        try:
            for line in existing.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not (stripped.startswith("exec ") and "-m hcli" in stripped):
                    continue
                rest = stripped[5:].strip()
                candidate = None
                if rest.startswith('"'):
                    end = rest.find('"', 1)
                    if end > 1:
                        candidate = rest[1:end]
                else:
                    parts = rest.split()
                    if parts:
                        candidate = parts[0]
                if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    return candidate
        except OSError:
            pass
    return sys.executable


def install_shims(home: Optional[str] = None) -> int:
    """Install `hcli` and `jhcli` shims plus a timestamped package copy.

    There is no pre-existing install script in this repo; this subcommand is
    the source of the ~/.local/bin shims. Both names exec `python -m hcli`
    with PYTHONPATH pointing at ~/.local/share/hcli/current.
    """
    home_path = Path(home).expanduser() if home else Path.home()
    src = Path(__file__).resolve().parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    share = home_path / ".local" / "share" / "hcli"
    dest_root = share / f"build-{stamp}"
    dest_pkg = dest_root / "hcli"
    share.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src,
        dest_pkg,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    current = share / "current"
    if current.is_symlink() or current.is_file():
        current.unlink()
    elif current.exists():
        raise RuntimeError(f"Refusing to replace non-symlink {current}")
    current.symlink_to(dest_root)

    python = _shim_python()
    script = (
        "#!/bin/sh\n"
        'BASE="$HOME/.local/share/hcli/current"\n'
        'export PYTHONPATH="$BASE${PYTHONPATH:+:$PYTHONPATH}"\n'
        f'exec "{python}" -m hcli "$@"\n'
    )
    bin_dir = home_path / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("hcli", "jhcli"):
        path = bin_dir / name
        path.write_text(script, encoding="utf-8")
        path.chmod(0o755)
        print(f"installed {path}")
    print(f"package {dest_pkg}")
    print(f"python {python}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in ("install-shims", "--install-shims"):
        home = None
        if len(raw) >= 3 and raw[1] == "--home":
            home = raw[2]
        return install_shims(home=home)

    args = parse_haider_args(raw)
    if args.debug:
        print(f"[hcli] args={vars(args)}")
    from .app import App
    app = App(
        workspace=args.workspace or os.getcwd(),
        runtime_count=args.runtime_count,
        model=args.model,
        debug=args.debug,
    )
    return app.run(prompt=args.prompt)
