from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union


MAX_RUNTIME_COUNT = 8

# Written into the build dir by install_shims, read on startup. The stamped
# copy has no .git, so the identity of what was deployed has to be recorded
# at install time; nothing about the copy itself can answer "from where".
INSTALL_STAMP = "install.json"

# Rollback window kept by install_shims. 3 = the build going live, the one it
# replaces, and one more to fall back to. 60 accumulated snapshots (the state
# this reaping was written for) never helped anyone.
KEEP_BUILDS = 3

# Delegation verbs dispatch BEFORE parse_haider_args, the same way
# `install-shims` already does. The single-shot positional grammar
# (`hcli 4 "do the thing"`, `hcli --task ...`) is untouched: it never
# begins with one of these tokens. A prompt that literally starts with
# the bare word "run"/"status"/... must use --task.
DELEGATE_VERBS = ("run", "status", "steer", "result", "abort")
DELEGATE_EXEC_VERB = "__delegate_exec"


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
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model artifact, native profile, or OpenAI-compatible endpoint URL",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Run provider text-only cognition without the HCLI result schema",
    )
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


def package_digest(pkg: Union[str, Path]) -> str:
    """Content hash of the ``*.py`` under a package dir.

    Bytes, not mtimes: a checkout or a rebase rewrites mtimes without
    changing a line, and a staleness warning that cries wolf gets ignored.
    ~2.5MB over ~110 files, ~6ms. ``__pycache__`` is not walked (it is also
    what ``install_shims`` refuses to copy).
    """
    h = hashlib.sha256()
    root = Path(pkg)
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def warn_if_stale() -> None:
    """One line when the running stamped copy no longer matches its source.

    `hcli` execs the deployed snapshot under ~/.local/share/hcli/current,
    so an install left behind by four days of repo work runs code nobody is
    editing any more and says nothing about it. Silent when the digests
    agree, and silent from an editable checkout (no stamp file at all).
    Reinstalling here would be worse than the drift: running a command must
    not rewrite the install underneath it.
    """
    try:
        stamp = json.loads(
            (Path(__file__).resolve().parent.parent / INSTALL_STAMP).read_text(
                encoding="utf-8"
            )
        )
        src = Path(stamp["source"])
        if not src.is_dir() or package_digest(src) == stamp["digest"]:
            return
    except (OSError, ValueError, KeyError):
        return
    print(
        f"[hcli] STALE: running the {stamp.get('installed', '?')} snapshot of "
        f"{src}, which has changed since. Fix: cd {src.parent} && "
        f"PYTHONPATH=. {sys.executable} -m hcli install-shims",
        file=sys.stderr,
    )


DAEMON_SHIM = "hawkingd"


def install_shims(home: Optional[str] = None) -> int:
    """Install the `hcli`/`jhcli` client and the `hawkingd` daemon shims.

    There is no pre-existing install script in this repo; this subcommand is
    the source of the ~/.local/bin shims. The client names exec `python -m
    hcli`; `hawkingd` execs `python -m hcli.hawkingd`, so a supervisor
    or worker shows up in `ps` under the daemon's own name rather than under
    the interpreter hosting it. The daemon is deliberately not named for any
    one model -- a supervisor may hold several bodies, in parallel or in
    succession -- while `hcli` stays the client that talks to it.

    All shims share PYTHONPATH pointing at ~/.local/share/hcli/current.
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
    (dest_root / INSTALL_STAMP).write_text(
        json.dumps(
            {"source": str(src), "digest": package_digest(src), "installed": stamp},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    current = share / "current"
    if current.is_symlink() or current.is_file():
        current.unlink()
    elif current.exists():
        raise RuntimeError(f"Refusing to replace non-symlink {current}")
    current.symlink_to(dest_root)

    # Reap old snapshots. Names sort by timestamp, so newest-first slicing is
    # the age order. The live target is never removed even if something else
    # left the symlink pointing at an older build, and a symlinked build-*
    # entry is skipped rather than followed.
    live = current.resolve()
    builds = sorted(
        (p for p in share.glob("build-*") if p.is_dir() and not p.is_symlink()),
        reverse=True,
    )
    reaped = 0
    for old in builds[KEEP_BUILDS:]:
        if old == live:
            continue
        shutil.rmtree(old)
        reaped += 1
    if reaped:
        print(f"reaped {reaped} old snapshot(s), kept {KEEP_BUILDS}")

    python = _shim_python()

    def _script(module: str) -> str:
        return (
            "#!/bin/sh\n"
            'BASE="$HOME/.local/share/hcli/current"\n'
            'export PYTHONPATH="$BASE${PYTHONPATH:+:$PYTHONPATH}"\n'
            f'exec "{python}" -m {module} "$@"\n'
        )

    bin_dir = home_path / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    # `hawkingd` is a separate entry point, not an alias: resident.py's
    # __main__ routes --supervise/--worker, so `ps` reads
    # "hawkingd --supervise <state>" instead of a bare interpreter line.
    for name, module in (
        ("hcli", "hcli"),
        ("jhcli", "hcli"),
        (DAEMON_SHIM, "hcli.hawkingd"),
    ):
        path = bin_dir / name
        path.write_text(_script(module), encoding="utf-8")
        path.chmod(0o755)
        print(f"installed {path}")
    print(f"package {dest_pkg}")
    print(f"python {python}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    warn_if_stale()
    if raw and raw[0] in ("install-shims", "--install-shims"):
        home = None
        if len(raw) >= 3 and raw[1] == "--home":
            home = raw[2]
        return install_shims(home=home)
    if raw and raw[0] in DELEGATE_VERBS:
        from .delegate import cli_main

        return cli_main(raw)
    if raw and raw[0] == DELEGATE_EXEC_VERB:
        from .delegate import exec_main

        return exec_main(raw[1:])
    if raw and raw[0] == "connectivity":
        from .connectivity import main as connectivity_main

        return connectivity_main(raw[1:])
    if raw and raw[0] == "flash-next":
        from .flash_next import main as flash_next_main

        return flash_next_main(raw[1:])
    if raw and raw[0] == "agentos":
        from .agentos_cli import main as agentos_main

        return agentos_main(raw[1:])
    if raw and raw[0] in ("resident", "daemon"):
        from .agentos.resident import main as resident_main

        return resident_main(raw[1:])

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
    return app.run(prompt=args.prompt, plain=args.plain)
