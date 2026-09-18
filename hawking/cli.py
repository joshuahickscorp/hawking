from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union


# Runtime fan-out is bounded by the machine/runtime admission path, not by a
# fixed Hawking count.  Keep this name as a compatibility export for callers
# that imported it, but make the absence of a count ceiling explicit.
MAX_RUNTIME_COUNT = None

# One public/local Hawking endpoint.  Keep this lightweight configuration at
# the command boundary so client launchers do not need to import the HTTP
# server just to learn where it will listen.
DEFAULT_ENDPOINT_PORT = 8014
DEFAULT_ENDPOINT_HOST = "127.0.0.1"
DEFAULT_ENDPOINT_BASE_URL = f"http://{DEFAULT_ENDPOINT_HOST}:{DEFAULT_ENDPOINT_PORT}"
DEFAULT_ENDPOINT_MODEL = "hawking-auto"

# Written into the build dir by install_shims, read on startup. The stamped
# copy has no .git, so the identity of what was deployed has to be recorded
# at install time; nothing about the copy itself can answer "from where".
INSTALL_STAMP = "install.json"

# Rollback window kept by install_shims. 3 = the build going live, the one it
# replaces, and one more to fall back to. 60 accumulated snapshots (the state
# this reaping was written for) never helped anyone.
KEEP_BUILDS = 3

# Delegation verbs dispatch BEFORE parse_hawking_args, the same way
# `install-shims` already does. The single-shot positional grammar
# (`hawking 4 "do the thing"`, `hawking --task ...`) is untouched: it never
# begins with one of these tokens. A prompt that literally starts with
# the bare word "run"/"status"/... must use --task.
DELEGATE_VERBS = ("run", "status", "steer", "result", "abort")
DELEGATE_EXEC_VERB = "__delegate_exec"

# Public control vocabulary. ``hawking.control`` imports this same tuple for
# parser verification, so a direct command has one spelling and one router.
CONTROL_VERBS: tuple[str, ...] = (
    "tools", "status", "resident", "checkpoint", "recovery", "recovery-gate",
    "research-gate", "perception-gate", "native-gate", "resident-gate",
    "native-mission-gate", "autonomy-gate", "unattended-window",
    "accelerator-regression", "qwen27-runtime-archaeology", "qwen27-mlp-ab",
    "qwen38-fusion-audit", "modellake-census", "modellake-supervise",
    "flash-science", "flash-executable", "flash-tensor-probe",
    "flash-vocabulary-prefilter", "flash-representation-experiment",
    "flash-transform-parity", "flash-loader-roundtrip", "flash-component-body",
    "flash-matrix-body", "flash-vector-body", "flash-router-graph",
    "flash-router-selection", "flash-router-representation-ab",
    "flash-component-campaign", "flash-graph-component", "preboard",
    "initial-charge", "science-maps", "ab-scaffold", "fpga-preboard",
    "architecture-atlas", "architecture-queue", "accelerator-physical-queue",
    "qwen27-token-budget", "architecture-audit", "protected-bench-watch",
    "protected-accelerator-bench", "handoff", "workunit", "background",
)


def _is_control_status(argv: Sequence[str]) -> bool:
    """Keep ``status <mission>`` for delegation and route only local status."""
    if not argv or argv[0] != "status":
        return False
    if len(argv) == 1:
        return True
    index = 1
    value_flags = {"--workspace", "--repo-root"}
    while index < len(argv):
        token = argv[index]
        if token in value_flags:
            index += 2
            continue
        if token.startswith("--workspace=") or token.startswith("--repo-root="):
            index += 1
            continue
        if token in {"-h", "--help"}:
            index += 1
            continue
        return False
    return True


VERB_HELP = """\
commands (each takes its own --help):

  Hawking
    h / chat    open the one Hawking UI in Chat mode
    build       open the same Hawking UI with server-owned Build authority
    web         compatibility alias for Chat (deprecated user-facing spelling)
    serve       the OpenAI-compatible endpoint only, no browser
    use         list every Hawking body, or switch which one answers
    stop        stop what web/serve started
    report      measure the loaded resident: cold, warm, decode

  work
    run         start a delegated task        status <mission>  how a task is doing
    steer       redirect a running task       result   fetch its result
    abort       stop a task

  system
    tools, status, checkpoint, background, workunit
    gates and Flash/ModelLake commands run directly as `hawking <verb>`
    resident    the long-running daemon (alias: daemon)
    connectivity, flash-next, install-shims

  with no verb, the first argument is a prompt:
    hawking "explain DeltaNet"
    hawking 4 "explain DeltaNet"      (4 runtimes)
"""


def _prog_name() -> str:
    base = os.path.basename(sys.argv[0] if sys.argv else "hawking")
    if base in ("hawking", "h"):
        return base
    return "hawking"


def _clamp_runtime_count(n: int) -> int:
    return max(1, int(n))


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
    """Resolve `hawking max` resident runtime count.

    Adapter over ``hawking.machine.resolve_runtime_limits`` so CLI ``max``
    cannot pick a STALE genome the runtime pool would refuse. The positional N
    grammar accepts any positive integer; ``max`` still uses the measured
    resident limit returned by ``hawking.machine``.

    Verified caller: ``parse_hawking_args`` (token ``max``).
    """
    from .machine import resolve_runtime_limits

    start = start_dir or os.getcwd()
    resolved = resolve_runtime_limits(repo_root=start, start_dir=start)
    return _clamp_runtime_count(resolved.resident_limit), _cli_limit_source(
        resolved.resident_source
    )


def parse_hawking_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=_prog_name(),
        description="HAWKING — autonomous local model engineering",
        # Every verb main() dispatches, listed. `hawking --help` used to advertise
        # only the positional prompt form, so `web`, `use`, `serve`, `report`,
        # `agentos`, `resident` and the delegation verbs were undiscoverable
        # from the tool itself -- a command that exists and is unfindable is
        # the same defect as one that is advertised and does not work.
        epilog=VERB_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("n_or_prompt", nargs="?", type=str, default=None,
                        help="Positive number of runtimes, 'max', or immediate mission prompt")
    parser.add_argument("prompt", nargs="?", type=str, default=None,
                        help="Immediate mission prompt (when N is supplied positionally)")
    parser.add_argument("--task", type=str, default=None, help="(legacy) mission text")
    parser.add_argument("--task-file", type=str, default=None, help="(legacy) path to mission file")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Explicitly admitted Hawking-native artifact or profile",
    )
    parser.add_argument("--resolved-action-json", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Run provider text-only cognition without the HAWKING result schema",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    # THESE BOUND NOTHING, AND SAYING SO IS THE POINT. Both were parsed and read
    # by no code anywhere in the package: the headless path calls
    # controller.execute() exactly once, and there is no cycle loop or turn loop
    # to cap. Two unattended tools passed --max-cycles believing it was a
    # ceiling, which is the dangerous direction of this bug -- a caller that
    # thinks a run is capped at 2 cycles and is in fact uncapped. Kept in the
    # parser ONLY so the refusal can explain itself instead of argparse saying
    # "unrecognized arguments".
    parser.add_argument("--max-turns", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--max-cycles", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--workspace", type=str, default=None, help="Workspace root")

    args = parser.parse_args(argv)
    args.resolved_action = None
    if args.resolved_action_json:
        from .catalog import resolved_action_from_json
        try:
            args.resolved_action = resolved_action_from_json(
                args.resolved_action_json, action="execute"
            )
            if args.model and os.path.realpath(os.path.expanduser(args.model)) != args.resolved_action.path:
                raise ValueError("--model does not match the resolved action path")
            args.model = args.resolved_action.path
        except (LookupError, PermissionError, ValueError) as exc:
            parser.error(str(exc))
    elif args.model:
        # The public command accepts only a catalog-admitted native execution
        # binding.  Reference providers have their own explicit science tools;
        # they must not reach the ordinary Hawking execution path by spelling a
        # directory or endpoint here.
        from .catalog import resolve_action
        try:
            resolved = resolve_action(args.model, "execute")
        except (LookupError, PermissionError, ValueError) as exc:
            parser.error(str(exc))
        if resolved is None:
            parser.error(
                f"--model {args.model!r} does not name an admitted Hawking-native "
                "artifact for execute; use an immutable admitted artifact id or profile"
            )
        args.resolved_action = resolved
        args.model = resolved.path
    dead = [name for name, value in (("--max-turns", args.max_turns),
                                     ("--max-cycles", args.max_cycles))
            if value is not None]
    if dead:
        parser.error(
            f"{' and '.join(dead)} bound nothing and never did: this path runs "
            f"one mission, once, and no cycle or turn loop exists to cap. "
            f"Accepting the flag would tell you a run is bounded when it is not. "
            f"For a bounded long run use `hawking resident start` (--max-restarts, "
            f"--interval-s, --swap-ceiling), or bound the work itself in the goal.")

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
                if n < 1:
                    parser.error(f"N must be a positive integer, got {n}")
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
    """The interpreter the shims will exec.

    PREFER THE ONE RUNNING THIS INSTALL. Reusing whatever the previous shim
    execed sounds conservative and is not: the shim on this machine pointed at a
    venv with no `mlx`, so every MLX body in the catalog -- 53 of the 54 -- would
    have failed at load through `hawking` while working perfectly through
    `python -m hawking`. The person running install-shims chose an interpreter by
    running it; honour that choice, and fall back to the old shim's only when
    this one cannot be located.
    """
    current = sys.executable
    if current and os.path.isfile(current) and os.access(current, os.X_OK):
        return current
    existing = Path.home() / ".local" / "bin" / "hawking"
    if existing.is_file():
        try:
            for line in existing.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not (stripped.startswith("exec ") and "-m hawking" in stripped):
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


def _file_digest(path: Path) -> str:
    """Return the content address for one deployable native artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_gravityd_binary(source_package: Union[str, Path]) -> Optional[Path]:
    """Find a deliberately built Gravity daemon beside a source checkout.

    Installation is not allowed to quietly invoke Cargo: package construction
    must remain reproducible and bounded.  A caller builds the exact native
    artifact it wants to ship, then this function packages that artifact with
    the Python control plane.  The direct-root location is the immutable
    snapshot layout used after installation; the remaining locations are the
    normal source-checkout build outputs.
    """
    source_root = Path(source_package).resolve().parent
    explicit = os.environ.get("HAWKING_GRAVITYD_BIN")
    candidates = [Path(explicit).expanduser()] if explicit else []
    candidates.extend((
        source_root / "hawking-gravityd",
        source_root / "workspace" / "ops" / "build" / "rust" / "release" / "hawking-gravityd",
        source_root / "workspace" / "ops" / "build" / "rust" / "debug" / "hawking-gravityd",
        source_root / "target" / "release" / "hawking-gravityd",
        source_root / "target" / "debug" / "hawking-gravityd",
    ))
    return next((path for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)


def native_hawking_binary(source_package: Union[str, Path]) -> Optional[Path]:
    """Find the Rust HAWKING authority to bundle with an installed snapshot.

    The Python compatibility skin may be launched from any directory.  A
    deployed snapshot therefore needs to carry the exact process-authority
    binary it was built and verified with; relying on the caller's checkout
    makes production behavior depend on the current working directory.  As
    with ``native_gravityd_binary``, installation never invokes Cargo.
    """
    source_root = Path(source_package).resolve().parent
    explicit = os.environ.get("HAWKING_NATIVE_BIN")
    candidates = [Path(explicit).expanduser()] if explicit else []
    candidates.extend((
        source_root / "hawking-rust",
        source_root / "workspace" / "ops" / "build" / "rust" / "release" / "hawking",
        source_root / "workspace" / "ops" / "build" / "rust" / "debug" / "hawking",
        source_root / "target" / "release" / "hawking",
        source_root / "target" / "debug" / "hawking",
    ))
    return next((path for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)


def native_process_authority_binary(source_package: Union[str, Path]) -> Optional[Path]:
    """Find the separately named Rust process protocol implementation.

    ``hawking-rust`` remains the inference CLI bundle.  Process inspection is
    a different protocol and must ship under a non-ambiguous name so an
    installed Hawking snapshot cannot route a `processes` request to model
    inference by accident.
    """
    source_root = Path(source_package).resolve().parent
    explicit = os.environ.get("HAWKING_PROCESS_AUTHORITY_BIN")
    candidates = [Path(explicit).expanduser()] if explicit else []
    candidates.extend((
        source_root / "hawking-process-authority",
        source_root / "hawking-backend",  # migration-only old build name
        source_root / "workspace" / "ops" / "build" / "rust" / "release" / "hawking-process-authority",
        source_root / "workspace" / "ops" / "build" / "rust" / "debug" / "hawking-process-authority",
        source_root / "target" / "release" / "hawking-process-authority",
        source_root / "target" / "debug" / "hawking-process-authority",
        source_root / "workspace" / "ops" / "build" / "rust" / "release" / "hawking-backend",
        source_root / "workspace" / "ops" / "build" / "rust" / "debug" / "hawking-backend",
        source_root / "target" / "release" / "hawking-backend",
        source_root / "target" / "debug" / "hawking-backend",
    ))
    return next((path for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)


def warn_if_stale() -> None:
    """One line when the running stamped copy no longer matches its source.

    `hawking` execs the deployed snapshot under ~/.local/share/hawking/current,
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
        f"[hawking] STALE: running the {stamp.get('installed', '?')} snapshot of "
        f"{src}, which has changed since. Fix: cd {src.parent} && "
        f"PYTHONPATH=. {sys.executable} -m hawking install-shims",
        file=sys.stderr,
    )


DAEMON_SHIM = "hawkingd"


def install_shims(home: Optional[str] = None) -> int:
    """Install the `hawking`/`h` client and the `hawkingd` daemon shims.

    There is no pre-existing install script in this repo; this subcommand is
    the source of the ~/.local/bin shims. The client names exec `python -m
    hawking`; `hawkingd` execs `python -m hawking.hawkingd`, so a supervisor
    or worker shows up in `ps` under the daemon's own name rather than under
    the interpreter hosting it. The daemon is deliberately not named for any
    one model -- a supervisor may hold several bodies, in parallel or in
    succession -- while `hawking` stays the client that talks to it.

    All shims share PYTHONPATH pointing at ~/.local/share/hawking/current.
    """
    home_path = Path(home).expanduser() if home else Path.home()
    src = Path(__file__).resolve().parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    share = home_path / ".local" / "share" / "hawking"
    dest_root = share / f"build-{stamp}"
    dest_pkg = dest_root / "hawking"
    share.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src,
        dest_pkg,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    native = native_gravityd_binary(src)
    native_stamp = {"status": "unavailable"}
    if native is not None:
        deployed_native = dest_root / "hawking-gravityd"
        shutil.copy2(native, deployed_native)
        deployed_native.chmod(deployed_native.stat().st_mode | 0o111)
        native_stamp = {
            "status": "bundled",
            "source": str(native),
            "digest": _file_digest(native),
            "path": deployed_native.name,
        }
    native_hawking = native_hawking_binary(src)
    native_hawking_stamp = {"status": "unavailable"}
    if native_hawking is not None:
        deployed_native_hawking = dest_root / "hawking-rust"
        shutil.copy2(native_hawking, deployed_native_hawking)
        deployed_native_hawking.chmod(deployed_native_hawking.stat().st_mode | 0o111)
        native_hawking_stamp = {
            "status": "bundled",
            "source": str(native_hawking),
            "digest": _file_digest(native_hawking),
            "path": deployed_native_hawking.name,
        }
    native_process_authority = native_process_authority_binary(src)
    native_process_authority_stamp = {"status": "unavailable"}
    if native_process_authority is not None:
        deployed_process_authority = dest_root / "hawking-process-authority"
        shutil.copy2(native_process_authority, deployed_process_authority)
        deployed_process_authority.chmod(deployed_process_authority.stat().st_mode | 0o111)
        native_process_authority_stamp = {
            "status": "bundled",
            "source": str(native_process_authority),
            "digest": _file_digest(native_process_authority),
            "path": deployed_process_authority.name,
        }
    (dest_root / INSTALL_STAMP).write_text(
        json.dumps(
            {
                "source": str(src),
                "digest": package_digest(src),
                "installed": stamp,
                "native_gravityd": native_stamp,
                "native_hawking": native_hawking_stamp,
                "native_process_authority": native_process_authority_stamp,
            },
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
    # The daemon's Python host is deliberately a native executable named
    # hawkingd. Rewriting argv with setproctitle makes ps useful, but Activity
    # Monitor reads the Mach-O process name and otherwise calls the root
    # Python. Keep the versioned alias inside the immutable deployed snapshot
    # so upgrades are atomic with the current symlink.
    daemon_python = dest_root / DAEMON_SHIM
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    standalone = sorted(
        (
            home_path
            / ".local"
            / "share"
            / "uv"
            / "python"
        ).glob(f"cpython-{version}*-macos-*-none/bin/python{version}"),
        reverse=True,
    )
    # The python.org macOS Framework launcher re-execs through Python.app,
    # losing the native hard-link name. Prefer an already-installed standalone
    # uv runtime of the same ABI for the lightweight daemon host. HAWKING itself
    # remains the stamped package in PYTHONPATH; this does not move authority
    # into Open WebUI or an MLX environment.
    daemon_host = standalone[0] if standalone else Path(python).resolve()
    match = re.fullmatch(r"python(?P<version>\d+\.\d+)", daemon_host.name)
    if match:
        library = (
            daemon_host.parent.parent
            / "lib"
            / f"libpython{match.group('version')}.dylib"
        )
        local_library = share / "lib" / library.name
        if library.is_file():
            local_library.parent.mkdir(parents=True, exist_ok=True)
            if local_library.exists() or local_library.is_symlink():
                if local_library.resolve() != library.resolve():
                    raise RuntimeError(
                        f"Refusing to replace unrelated daemon library {local_library}"
                    )
            else:
                local_library.symlink_to(library)
    try:
        os.link(daemon_host.resolve(), daemon_python)
    except OSError:
        shutil.copy2(daemon_host.resolve(), daemon_python)

    def _script(module: str) -> str:
        executable = '"$BASE/hawkingd"' if module == "hawking.hawkingd" else f'"{python}"'
        return (
            "#!/bin/sh\n"
            'BASE="$HOME/.local/share/hawking/current"\n'
            'export PYTHONPATH="$BASE${PYTHONPATH:+:$PYTHONPATH}"\n'
            f'export HAWKING_GRAVITY_REGISTRY="{src.parent / "workspace/campaign/odyssey/gravity-artifacts.json"}"\n'
            # Python normally puts the current working directory ahead of
            # PYTHONPATH. Without -P, invoking the installed shim from any
            # Hawking checkout imports that checkout's hawking package and
            # silently bypasses the stamped deployment/rollback target.
            f"exec {executable} -P -m {module} \"$@\"\n"
        )

    bin_dir = home_path / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    # `hawkingd` is a separate entry point, not an alias: resident.py's
    # __main__ routes --supervise/--worker, so `ps` reads
    # "hawkingd --supervise <state>" instead of a bare interpreter line.
    for name, module in (
        ("hawking", "hawking"),
        ("h", "hawking"),
        (DAEMON_SHIM, "hawking.hawkingd"),
    ):
        path = bin_dir / name
        path.write_text(_script(module), encoding="utf-8")
        path.chmod(0o755)
        print(f"installed {path}")
    print(f"package {dest_pkg}")
    print(f"native gravityd {native_stamp['status']}")
    print(f"native hawking {native_hawking_stamp['status']}")
    print(f"native process authority {native_process_authority_stamp['status']}")
    print(f"python {python}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    warn_if_stale()
    if not raw:
        from .web_launcher import main as web_launcher_main
        return web_launcher_main([])
    if raw and raw[0] in ("install-shims", "--install-shims"):
        home = None
        if len(raw) >= 3 and raw[1] == "--home":
            home = raw[2]
        return install_shims(home=home)
    # Bare status is the local Hawking control-plane view.  Status with a
    # mission identifier remains the delegated-task command below.
    if _is_control_status(raw):
        from .control import main as control_main

        return control_main(raw)
    if raw and raw[0] in DELEGATE_VERBS:
        from .delegate import cli_main

        return cli_main(raw)
    if raw and raw[0] == DELEGATE_EXEC_VERB:
        from .delegate import exec_main

        return exec_main(raw[1:])
    if raw and raw[0] == "census":
        from .odyssey_census import main as census_main

        return census_main(raw[1:])
    if raw and raw[0] == "campaign":
        from .campaign_report import main as campaign_main

        return campaign_main(raw[1:])
    if raw and raw[0] == "connectivity":
        from .connectivity import main as connectivity_main

        return connectivity_main(raw[1:])
    if raw and raw[0] == "flash-next":
        from .flash_next import main as flash_next_main

        return flash_next_main(raw[1:])
    if raw and raw[0] == "agentos":
        print(
            "hawking: 'agentos' is no longer a command; use a direct Hawking verb "
            "such as 'hawking tools' or 'hawking checkpoint'",
            file=sys.stderr,
        )
        return 2
    if raw and raw[0] in ("resident", "daemon"):
        from .resident import main as resident_main

        return resident_main(raw[1:])
    if raw and raw[0] in CONTROL_VERBS:
        from .control import main as control_main

        return control_main(raw)
    if raw and raw[0] == "serve":
        from .serve import main as serve_main

        return serve_main(raw[1:])
    if raw and raw[0] in {"web", "chat", "build"}:
        # One public Hawking product: bare ``h``, ``h chat`` and ``h build``
        # all enter this launcher. ``web`` survives only as a compatibility
        # spelling; it does not own a separate runtime or client.
        from .web_launcher import main as web_launcher_main

        # Build is an authority profile over the same UI. The launcher mints a
        # server-owned builder session; it never re-enters the historical
        # resident/Open WebUI bootstrap path.
        profile = ["--profile", "build"] if raw[0] == "build" else []
        return web_launcher_main([*profile, *raw[1:]])
    if raw and raw[0] == "report":
        from .report import main as report_main

        return report_main(raw[1:])
    if raw and raw[0] in ("use", "models"):
        from .use import main as use_main

        return use_main(raw[1:])
    if raw and raw[0] == "stop":
        from .web import stop_main

        return stop_main(raw[1:])

    args = parse_hawking_args(raw)
    if args.debug:
        print(f"[hawking] args={vars(args)}")
    from .app import App
    app = App(
        workspace=args.workspace or os.getcwd(),
        runtime_count=args.runtime_count,
        model=args.model,
        resolved_action=args.resolved_action,
        debug=args.debug,
    )
    return app.run(prompt=args.prompt, plain=args.plain)
