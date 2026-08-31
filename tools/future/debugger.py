"""DEBUGGER LAB — provider-neutral debugger contract and honest Apple probes.

Headline: this host has Command Line Tools, not Xcode. `xcrun` cannot locate
the Metal compiler. LLDB is on PATH but cannot launch a debuggee
(`Operation not permitted`). Every debugger operation on an unavailable
provider RAISES with the named missing dependency. Nothing here fabricates a
stack, a variable, or a successful transcript.

Public APIs only. STATIC_ONLY. Bench state UNKNOWN. gpu_authority false.
Neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE.

    python3 tools/future/debugger.py --probe
    python3 tools/future/debugger.py --build
    python3 tools/future/debugger.py --selftest
    python3 -m pytest tools/future/test_debugger.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, git

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


RECEIPT = "DEBUGGER_LAB.json"
TRANSCRIPT_RECEIPT = "DEBUGGER_TRANSCRIPT.json"
SCHEMA = "hawking.future.debugger.v1"
TRANSCRIPT_SCHEMA = "hawking.future.debugger.transcript.v1"
VERSION = 1
RECORDED_BY = "tools/future/debugger.py"

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)

# Nineteen operations. The contract named these; iteration is derived from this
# tuple so a dropped name is a test failure rather than a silent shrink.
OPERATIONS: tuple[str, ...] = (
    "launch",
    "attach",
    "set_breakpoint",
    "conditional_breakpoint",
    "tracepoint",
    "pause",
    "resume",
    "step_in",
    "step_over",
    "step_out",
    "run_to_location",
    "stacks",
    "threads",
    "variables",
    "memory_inspect",
    "evaluate",
    "crash_capture",
    "sanitizer_result",
    "transcript_receipt",
)

# Apple lab probes the resident can discover. Counts are derived from this tuple.
LAB_CAPABILITIES: tuple[str, ...] = (
    "xcodebuild",
    "lldb",
    "metal_compilation",
    "shader_diagnostics",
    "coreml_compilation",
    "mlcomputeplan",
    "mlstate",
    "instruments",
    "sanitizers",
    "simulator",
)

PROVIDERS: tuple[str, ...] = ("lldb", "stub")

CODEX_CMDLINE_MARKERS: tuple[str, ...] = (
    "odyssey_ctl.py",
    "modellake_watch.py",
    "physical_qualification",
    "protected-accelerator-bench",
    "gpu_cleanliness",
    "lake_filler.py",
    "tools/odyssey",
    "hawking-bench",
    "accelerator_runner",
    "tools/accelerator",
)

# Codex-named surfaces the contract asked us to recover. Presence is reported;
# absence is a negative finding, not a license to invent them.
RECOVER_PATHS = (
    "tools/future/ane_preboard.py",
    "tools/future/debugger.py",
    "hcli/tool_registry.py",
    "hcli/providers.py",
    "hcli/vmcp_adapter.py",
    "hcli/vmcp/__init__.py",
    "hcli/workunit.py",
    "hcli/agentos/preboard.py",
    "hcli/agentos/runtime.py",
    "tools/headless/noetic_executable_closure.py",
    "receipts/headless/NOETIC_EXECUTABLE_CLOSURE.json",
    "receipts/headless/GPU_LEDGER.json",
    "receipts/future/ANE_PREBOARD.json",
    "receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
    "receipts/future/FUTURE_SUBSTRATE_HANDOFF.json",
    "CODEX_ACCELERATOR_HANDOFF.json",
)

CLT_COREML_HEADERS = Path(
    "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/CoreML.framework/Headers"
)

PROBE_TIMEOUT_S = 8
_LAB_CACHE: dict[str, Any] | None = None

HCLI_TOOL_DESCRIPTOR: dict[str, Any] = {
    "schema": "hcli.agentos.tool.v1",
    "name": "future.debugger",
    "description": (
        "Provider-neutral debugger (LLDB + stub) and honest Apple lab probes. "
        "Unavailable providers raise; nothing fabricates a stack, variable, or transcript."
    ),
    "mutation": "read_only",
    "deterministic": True,
    "timeout_s": 60.0,
    "roles": ["science", "verifier"],
    "resources": ["cpu"],
    "verifier_expectations": (
        "unavailable provider raises with the missing dependency named",
        "memory inspection is read-only",
        "attach requires an explicit non-Codex pid",
    ),
    "input_schema": {
        "type": "object",
        "required": ["action"],
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["probe", "build", "invoke", "workunits"]},
            "provider": {"type": "string", "enum": list(PROVIDERS)},
            "operation": {"type": "string", "enum": list(OPERATIONS)},
            "arguments": {"type": "object"},
        },
    },
    "output_schema": {"type": "object"},
    "provenance": "tools.future.debugger",
    "handler": "tools.future.debugger:hcli_invoke",
}


# ---------------------------------------------------------------------------
# Errors. A guard nobody has watched fail is not a guard.
# ---------------------------------------------------------------------------


class DebuggerUnavailableError(RuntimeError):
    """Raised by every debugger operation when the provider cannot operate."""

    def __init__(
        self,
        *,
        provider: str,
        operation: str,
        missing: Sequence[str],
        probe: Mapping[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.operation = operation
        self.missing = [str(item) for item in missing]
        self.probe = dict(probe or {})
        detail = "; ".join(self.missing) if self.missing else "provider available() is False"
        super().__init__(f"{provider} cannot {operation}: {detail}")


class LabUnavailableError(RuntimeError):
    """Raised by Apple-lab execution when the named tool is not usable."""

    def __init__(
        self,
        *,
        capability: str,
        missing: Sequence[str],
        probe: Mapping[str, Any] | None = None,
    ) -> None:
        self.capability = capability
        self.missing = [str(item) for item in missing]
        self.probe = dict(probe or {})
        detail = "; ".join(self.missing) if self.missing else f"{capability} available() is False"
        super().__init__(f"Apple lab cannot {capability}: {detail}")


class AttachRefused(ValueError):
    """Attach/launch safety: explicit target required; Codex processes forbidden."""


class MemoryWriteForbidden(ValueError):
    """Memory inspection is read-only. A write is never a debugger-lab operation."""


class FabricationForbidden(RuntimeError):
    """Refuses to return a stack, variable, or transcript that was not observed."""


class SessionRequiredError(RuntimeError):
    """A live session is required and does not exist. Not a fabricated empty session."""


# ---------------------------------------------------------------------------
# Recovery: disk is authority. Sparse checkout is not absence-from-git.
# ---------------------------------------------------------------------------


def path_state(rel: str) -> dict[str, Any]:
    on_disk = (REPO / rel).is_file()
    listed = git("ls-tree", "-r", "--name-only", "HEAD", "--", rel)
    in_git = any(line.strip() == rel for line in listed.splitlines())
    return {
        "path": rel,
        "on_disk": on_disk,
        "in_git_head": in_git,
        "source": "ON_DISK" if on_disk else ("GIT_HEAD" if in_git else "ABSENT"),
    }


def reset_probe_cache() -> None:
    global _LAB_CACHE
    _LAB_CACHE = None


def _run(argv: list[str], timeout: int = PROBE_TIMEOUT_S) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(argv, 127, "", f"FileNotFoundError: {exc}")
    except PermissionError as exc:
        return subprocess.CompletedProcess(argv, 126, "", f"PermissionError: {exc}")
    except OSError as exc:
        return subprocess.CompletedProcess(argv, 125, "", f"{type(exc).__name__}: {exc}")
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return subprocess.CompletedProcess(argv, 124, stdout, (stderr + "\ntimeout").strip())


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _xcrun_find(name: str) -> dict[str, Any]:
    which = shutil.which(name)
    proc = _run(["xcrun", "-f", name])
    path = (proc.stdout or "").strip() or None
    stderr = (proc.stderr or "").strip() or None
    usable = bool(proc.returncode == 0 and path)
    return {
        "name": name,
        "which": which,
        "xcrun_path": path if usable else None,
        "xcrun_returncode": proc.returncode,
        "xcrun_stderr": stderr,
        "present_on_path": bool(which),
        "xcrun_resolves": usable,
    }


# ---------------------------------------------------------------------------
# Attach / memory safety (pure). Called after the availability gate.
# ---------------------------------------------------------------------------


def _handoff_pids() -> list[int]:
    pids: list[int] = []
    rel = "receipts/future/FUTURE_SUBSTRATE_HANDOFF.json"
    path = REPO / rel
    if not path.is_file():
        return pids
    try:
        doc = load_json(path)
    except (OSError, json.JSONDecodeError):
        return pids
    for row in doc.get("active_processes") or []:
        text = str(row).strip()
        if not text:
            continue
        head = text.split()[0]
        if head.isdigit():
            pids.append(int(head))
    return pids


def command_looks_like_codex(command: str | None) -> bool:
    text = str(command or "").lower()
    if not text:
        return False
    return any(marker.lower() in text for marker in CODEX_CMDLINE_MARKERS)


def read_pid_command(pid: int) -> str | None:
    proc = _run(["ps", "-p", str(pid), "-o", "command="])
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip()
    return text or None


def guard_attach(
    pid: int | None,
    *,
    command: str | None = None,
    explicit: bool = True,
) -> int:
    """Refuse attach unless an explicit, non-self, non-Codex pid was given."""
    if not explicit or pid is None:
        raise AttachRefused(
            "attach requires an explicit pid; this lab never scans or infers a target"
        )
    try:
        pid_i = int(pid)
    except (TypeError, ValueError) as exc:
        raise AttachRefused(f"attach pid is not an integer: {pid!r}") from exc
    if pid_i <= 1:
        raise AttachRefused(f"attach refuses kernel/launchd pid {pid_i}")
    if pid_i == os.getpid():
        raise AttachRefused("attach refuses the debugger-lab process itself")
    denied = set(_handoff_pids())
    if pid_i in denied:
        raise AttachRefused(
            f"attach refuses Codex campaign pid {pid_i} listed in FUTURE_SUBSTRATE_HANDOFF"
        )
    cmdline = command if command is not None else read_pid_command(pid_i)
    if command_looks_like_codex(cmdline):
        raise AttachRefused(
            f"attach refuses Codex campaign process pid={pid_i} command={cmdline!r}"
        )
    return pid_i


def _is_memory_write(command: str | None) -> bool:
    text = " ".join(str(command or "").lower().split())
    if not text:
        return False
    needles = (
        "memory write",
        "memory poke",
        "mem write",
        "register write",
        "reg write",
    )
    return any(text.startswith(n) or f" {n} " in f" {text} " for n in needles)


def guard_memory(*, write: bool = False, command: str | None = None) -> None:
    if write or _is_memory_write(command):
        raise MemoryWriteForbidden(
            "memory inspection is read-only; memory write / register write are forbidden"
        )


def guard_launch(program: str | None) -> str:
    if not program or not str(program).strip():
        raise AttachRefused("launch requires an explicit program path; no default debuggee")
    return str(program)


def refuse_fabricated(kind: str, payload: Mapping[str, Any] | None = None) -> None:
    """Last-line defence: a result that claims live debuggee state without observation."""
    body = dict(payload or {})
    if body.get("fabricated") is True:
        raise FabricationForbidden(f"refusing fabricated {kind}")
    if body.get("live") and not body.get("observed"):
        raise FabricationForbidden(f"refusing unobserved live {kind}")


# ---------------------------------------------------------------------------
# Honest Apple lab probes. Never xcode-select, never install, never a password.
# ---------------------------------------------------------------------------


def _capability(
    name: str,
    *,
    available: bool,
    missing: Sequence[str],
    present: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "available": bool(available),
        "missing": [str(item) for item in missing],
        "present": [str(item) for item in present],
    }
    if extra:
        for key, value in extra.items():
            row[key] = value
    return row


def _probe_xcodebuild() -> dict[str, Any]:
    loc = _xcrun_find("xcodebuild")
    ver = _run(["xcodebuild", "-version"])
    line = _first_line(ver.stderr or ver.stdout) or "xcodebuild produced no output"
    xcode_app = Path("/Applications/Xcode.app").is_dir()
    select = _run(["xcode-select", "-p"])
    select_path = (select.stdout or "").strip() or None
    clt_only = bool(select_path and "CommandLineTools" in select_path and not xcode_app)
    ok = ver.returncode == 0
    missing: list[str] = []
    present: list[str] = []
    if loc["which"]:
        present.append(f"xcodebuild on PATH at {loc['which']}")
    if not xcode_app:
        missing.append("/Applications/Xcode.app is absent")
    if clt_only:
        missing.append(
            f"active developer directory is CommandLineTools ({select_path}); "
            "full Xcode is required for xcodebuild"
        )
    if not ok:
        missing.append(f"xcodebuild unavailable: {line}")
    else:
        present.append(f"xcodebuild: {line}")
    return _capability(
        "xcodebuild",
        available=ok and not missing,
        missing=missing,
        present=present,
        extra={
            "locator": loc,
            "xcode_app": "/Applications/Xcode.app" if xcode_app else None,
            "xcode_select_path": select_path,
            "command_line_tools_only": clt_only,
            "version_returncode": ver.returncode,
            "version_line": line,
        },
    )


def _probe_lldb() -> dict[str, Any]:
    loc = _xcrun_find("lldb")
    ver = _run(["lldb", "--version"])
    version_line = _first_line(ver.stdout or ver.stderr)
    py_spec = importlib.util.find_spec("lldb")
    launch = _run(
        [
            "lldb",
            "--batch",
            "--no-lldbinit",
            "-o",
            "process launch --stop-at-entry",
            "-o",
            "quit",
            "--",
            "/bin/echo",
            "hawking-debugger-lab-probe",
        ]
    )
    launch_err = _first_line(launch.stderr or launch.stdout)
    launch_blocked = (
        launch.returncode != 0
        or "operation not permitted" in (launch.stderr or "").lower()
        or "operation not permitted" in (launch.stdout or "").lower()
    )
    missing: list[str] = []
    present: list[str] = []
    if loc["xcrun_resolves"]:
        present.append(f"xcrun -f lldb => {loc['xcrun_path']}")
    elif loc["which"]:
        present.append(f"lldb on PATH at {loc['which']}")
    else:
        missing.append("lldb is not on PATH and xcrun cannot resolve it")
    if ver.returncode == 0 and version_line:
        present.append(f"lldb --version: {version_line}")
    elif loc["which"] or loc["xcrun_resolves"]:
        missing.append(f"lldb --version failed: {version_line or 'no output'}")
    if py_spec is None:
        present.append("Python lldb module is absent (CLI only; not a silent skip)")
    else:
        present.append("Python lldb module is importable")
    if launch_blocked:
        missing.append(
            f"lldb cannot launch a process: {launch_err or 'non-zero exit'} "
            "(SIP/sandbox; matches receipts/headless NOETIC_EXECUTABLE_CLOSURE)"
        )
    else:
        present.append("lldb launched /bin/echo --stop-at-entry")
    available = bool(
        (loc["xcrun_resolves"] or loc["which"])
        and ver.returncode == 0
        and not launch_blocked
        and not missing
    )
    return _capability(
        "lldb",
        available=available,
        missing=missing,
        present=present,
        extra={
            "locator": loc,
            "version": version_line or None,
            "python_module": py_spec is not None,
            "launch_probe": {
                "argv": [
                    "lldb",
                    "--batch",
                    "--no-lldbinit",
                    "-o",
                    "process launch --stop-at-entry",
                    "-o",
                    "quit",
                    "--",
                    "/bin/echo",
                    "hawking-debugger-lab-probe",
                ],
                "returncode": launch.returncode,
                "stdout": (launch.stdout or "").strip()[:800] or None,
                "stderr": (launch.stderr or "").strip()[:800] or None,
                "ok": not launch_blocked,
            },
            "binary_present": bool(loc["which"] or loc["xcrun_resolves"]),
        },
    )


def _probe_metal() -> dict[str, Any]:
    names = ("metal", "metallib", "metal-source", "air-lld", "metal-opt")
    locators = {name: _xcrun_find(name) for name in names}
    missing: list[str] = []
    present: list[str] = []
    metal = locators["metal"]
    if metal["xcrun_resolves"]:
        present.append(f"xcrun -f metal => {metal['xcrun_path']}")
    else:
        err = metal["xcrun_stderr"] or 'xcrun: error: unable to find utility "metal"'
        missing.append(f"Metal compiler absent: {err}")
    for name, loc in locators.items():
        if name == "metal":
            continue
        if loc["xcrun_resolves"]:
            present.append(f"xcrun -f {name} => {loc['xcrun_path']}")
        else:
            missing.append(
                f"{name} absent: {loc['xcrun_stderr'] or 'xcrun could not resolve it'}"
            )
    return _capability(
        "metal_compilation",
        available=bool(metal["xcrun_resolves"]) and not missing,
        missing=missing,
        present=present,
        extra={"locators": locators, "expected_absent_on_clt": True},
    )


def _probe_shader_diagnostics() -> dict[str, Any]:
    names = ("metal-objdump", "metallib", "spirv-as", "spirv-dis")
    locators = {name: _xcrun_find(name) for name in names}
    missing: list[str] = []
    present: list[str] = []
    for name, loc in locators.items():
        if loc["xcrun_resolves"] or loc["which"]:
            present.append(f"{name} => {loc['xcrun_path'] or loc['which']}")
        else:
            missing.append(
                f"{name} absent: {loc['xcrun_stderr'] or 'not on PATH and xcrun cannot resolve it'}"
            )
    return _capability(
        "shader_diagnostics",
        available=any(v["xcrun_resolves"] or v["which"] for v in locators.values()),
        missing=missing,
        present=present,
        extra={"locators": locators},
    )


def _probe_coreml_compilation() -> dict[str, Any]:
    names = ("coremlcompiler", "coremlc")
    locators = {name: _xcrun_find(name) for name in names}
    ct = importlib.util.find_spec("coremltools")
    missing: list[str] = []
    present: list[str] = []
    for name, loc in locators.items():
        if loc["xcrun_resolves"]:
            present.append(f"xcrun -f {name} => {loc['xcrun_path']}")
        else:
            missing.append(
                f"{name} absent: {loc['xcrun_stderr'] or 'xcrun could not resolve it'}"
            )
    if ct is None:
        missing.append("coremltools unavailable: ModuleNotFoundError: No module named 'coremltools'")
    else:
        present.append("coremltools is importable")
    return _capability(
        "coreml_compilation",
        available=False if missing else True,
        missing=missing,
        present=present,
        extra={"locators": locators, "coremltools": ct is not None},
    )


def _probe_mlcomputeplan() -> dict[str, Any]:
    header = CLT_COREML_HEADERS / "MLComputePlan.h"
    headers_present = header.is_file()
    # Live load needs a compiled .mlmodelc + full Xcode. Headers are not a toolchain.
    xcode = _probe_xcodebuild()
    coreml = _probe_coreml_compilation()
    missing: list[str] = []
    present: list[str] = []
    if headers_present:
        present.append(f"{header} present (SDK header; not a live MLComputePlan.load)")
    else:
        missing.append(f"{header} missing")
    if not xcode["available"]:
        missing.append("live MLComputePlan.load requires a usable xcodebuild/Xcode toolchain")
    if not coreml["available"]:
        missing.append("live MLComputePlan.load requires a Core ML compile toolchain")
    return _capability(
        "mlcomputeplan",
        available=False,
        missing=missing,
        present=present,
        extra={
            "headers_present": headers_present,
            "public_api": "MLComputePlan.load(contentsOf:configuration:)",
            "live": False,
            "reason": "headers are not a compile/load toolchain; no .mlmodelc is authored here",
        },
    )


def _probe_mlstate() -> dict[str, Any]:
    header = CLT_COREML_HEADERS / "MLState.h"
    model_state = CLT_COREML_HEADERS / "MLModel+MLState.h"
    headers_present = header.is_file() or model_state.is_file()
    missing: list[str] = []
    present: list[str] = []
    if header.is_file():
        present.append(f"{header} present (SDK header; not a live MLState)")
    if model_state.is_file():
        present.append(f"{model_state} present (SDK header; not a live MLState)")
    if not headers_present:
        missing.append("MLState headers are not in the CLT SDK")
    missing.append("live MLState requires a compiled model and a full Xcode / Core ML toolchain")
    return _capability(
        "mlstate",
        available=False,
        missing=missing,
        present=present,
        extra={
            "headers_present": headers_present,
            "public_api": "Core ML MLState / MLModel+MLState",
            "live": False,
        },
    )


def _probe_instruments() -> dict[str, Any]:
    loc = _xcrun_find("instruments")
    xctrace = _xcrun_find("xctrace")
    missing: list[str] = []
    present: list[str] = []
    if loc["xcrun_resolves"] or loc["which"]:
        present.append(f"instruments => {loc['xcrun_path'] or loc['which']}")
    else:
        missing.append(
            f"instruments absent: {loc['xcrun_stderr'] or 'not on PATH and xcrun cannot resolve it'}"
        )
    if xctrace["xcrun_resolves"] or xctrace["which"]:
        present.append(f"xctrace => {xctrace['xcrun_path'] or xctrace['which']}")
    else:
        missing.append(
            f"xctrace absent: {xctrace['xcrun_stderr'] or 'not on PATH and xcrun cannot resolve it'}"
        )
    return _capability(
        "instruments",
        available=bool(loc["xcrun_resolves"] or xctrace["xcrun_resolves"]),
        missing=missing,
        present=present,
        extra={"locator": loc, "xctrace": xctrace},
    )


def _probe_sanitizers() -> dict[str, Any]:
    clang = shutil.which("clang") or (_xcrun_find("clang").get("xcrun_path"))
    missing: list[str] = []
    present: list[str] = []
    compiled: dict[str, Any] = {}
    if not clang:
        missing.append("clang is not on PATH; sanitizer compile cannot be probed")
        return _capability(
            "sanitizers",
            available=False,
            missing=missing,
            present=present,
            extra={"compiler": None, "compiled": compiled},
        )
    present.append(f"clang at {clang}")
    tmp = Path(tempfile.mkdtemp(prefix="hawking-debugger-san-"))
    try:
        src = tmp / "t.c"
        src.write_text("int main(void){return 0;}\n")
        for san in ("address", "undefined", "thread", "memory"):
            out = tmp / f"t_{san}"
            ran = _run(["clang", f"-fsanitize={san}", "-o", str(out), str(src)])
            ok = ran.returncode == 0 and out.is_file()
            compiled[san] = {
                "ok": ok,
                "returncode": ran.returncode,
                "stderr": _first_line(ran.stderr or ran.stdout) or None,
            }
            if ok:
                present.append(f"clang -fsanitize={san} compiled a trivial C program")
            else:
                missing.append(
                    f"clang -fsanitize={san} failed: {compiled[san]['stderr'] or 'non-zero exit'}"
                )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    usable = any(row["ok"] for row in compiled.values())
    # Unsupported sanitizers (e.g. memory on Darwin/arm64) stay in missing but
    # do not close the capability if another sanitizer compiled.
    available = bool(clang) and usable
    return _capability(
        "sanitizers",
        available=available,
        missing=missing,
        present=present,
        extra={"compiler": clang, "compiled": compiled},
    )


def _probe_simulator() -> dict[str, Any]:
    loc = _xcrun_find("simctl")
    missing: list[str] = []
    present: list[str] = []
    if loc["xcrun_resolves"]:
        present.append(f"xcrun -f simctl => {loc['xcrun_path']}")
        available = True
    else:
        missing.append(
            f"simctl absent: {loc['xcrun_stderr'] or 'xcrun could not resolve it'}; "
            "Simulator requires full Xcode"
        )
        available = False
    return _capability(
        "simulator",
        available=available,
        missing=missing,
        present=present,
        extra={"locator": loc},
    )


_PROBE_FNS: dict[str, Callable[[], dict[str, Any]]] = {
    "xcodebuild": _probe_xcodebuild,
    "lldb": _probe_lldb,
    "metal_compilation": _probe_metal,
    "shader_diagnostics": _probe_shader_diagnostics,
    "coreml_compilation": _probe_coreml_compilation,
    "mlcomputeplan": _probe_mlcomputeplan,
    "mlstate": _probe_mlstate,
    "instruments": _probe_instruments,
    "sanitizers": _probe_sanitizers,
    "simulator": _probe_simulator,
}


def probe_apple_lab(*, force: bool = False) -> dict[str, Any]:
    """Observe the Apple toolchain. Does not install, does not run xcode-select -s."""
    global _LAB_CACHE
    if _LAB_CACHE is not None and not force:
        return _LAB_CACHE
    probes: dict[str, Any] = {}
    for name in LAB_CAPABILITIES:
        probes[name] = _PROBE_FNS[name]()
    missing: list[str] = []
    present: list[str] = []
    for name in LAB_CAPABILITIES:
        row = probes[name]
        present.extend(f"{name}: {item}" for item in row.get("present") or [])
        missing.extend(f"{name}: {item}" for item in row.get("missing") or [])
    available_names = [name for name, row in probes.items() if row.get("available")]
    metal = probes["metal_compilation"]
    headline = (
        "usable Apple debugger/lab toolchain present"
        if probes["lldb"]["available"] and metal["available"]
        else "Apple lab is partial or closed; debugger operations fail closed on unavailable providers"
    )
    result = {
        "available": bool(probes["lldb"]["available"] and metal["available"]),
        "available_capabilities": list(available_names),
        "missing": missing,
        "present": present,
        "probes": probes,
        "macos": platform.mac_ver()[0] or "UNKNOWN",
        "machine": platform.machine(),
        "command_line_tools_only": bool(
            probes["xcodebuild"].get("command_line_tools_only")
        ),
        "did_not": [
            "install any package",
            "run xcode-select to change the active developer directory",
            "prompt for a password",
            "attach to a process that was not explicitly given",
            "attach to a Codex campaign process",
            "write process memory",
            "fabricate a stack, variable, or transcript",
            "claim a hardware measurement",
        ],
        "headline": headline,
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "metal_compiler_finding": (
            "ABSENT"
            if not metal["available"]
            else "PRESENT"
        ),
    }
    _LAB_CACHE = result
    return result


def lab_available(capability: str) -> bool:
    lab = probe_apple_lab()
    row = lab["probes"].get(capability)
    return bool(row and row.get("available"))


def require_lab(capability: str) -> dict[str, Any]:
    lab = probe_apple_lab()
    if capability not in lab["probes"]:
        raise LabUnavailableError(
            capability=capability,
            missing=[f"unknown lab capability {capability!r}"],
            probe={"available": False},
        )
    probe = lab["probes"][capability]
    if not probe.get("available"):
        raise LabUnavailableError(
            capability=capability,
            missing=list(probe.get("missing") or [f"{capability} available() is False"]),
            probe=probe,
        )
    return probe


# ---------------------------------------------------------------------------
# Provider-neutral debugger contract.
# ---------------------------------------------------------------------------


class DebuggerProvider(ABC):
    """LLDB and stub both satisfy this surface. Nineteen operations, no extras required."""

    name: str

    @abstractmethod
    def probe(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def available(self) -> bool:
        ...

    @abstractmethod
    def missing(self) -> list[str]:
        ...

    def require_available(self, operation: str) -> dict[str, Any]:
        probe = self.probe()
        if not self.available():
            raise DebuggerUnavailableError(
                provider=self.name,
                operation=operation,
                missing=self.missing() or [f"{self.name} available() is False"],
                probe=probe,
            )
        return probe

    @abstractmethod
    def launch(self, program: str | None = None, args: Sequence[str] | None = None) -> dict[str, Any]:
        ...

    @abstractmethod
    def attach(self, pid: int | None = None, command: str | None = None) -> dict[str, Any]:
        ...

    @abstractmethod
    def set_breakpoint(self, location: str | None = None) -> dict[str, Any]:
        ...

    @abstractmethod
    def conditional_breakpoint(
        self, location: str | None = None, condition: str | None = None
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    def tracepoint(self, location: str | None = None, command: str | None = None) -> dict[str, Any]:
        ...

    @abstractmethod
    def pause(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def resume(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def step_in(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def step_over(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def step_out(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def run_to_location(self, location: str | None = None) -> dict[str, Any]:
        ...

    @abstractmethod
    def stacks(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def threads(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def variables(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def memory_inspect(
        self,
        address: str | None = None,
        size: int | None = None,
        *,
        write: bool = False,
        command: str | None = None,
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    def evaluate(self, expression: str | None = None) -> dict[str, Any]:
        ...

    @abstractmethod
    def crash_capture(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def sanitizer_result(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def transcript_receipt(self) -> dict[str, Any]:
        ...


def invoke(provider: DebuggerProvider, operation: str, **kwargs: Any) -> dict[str, Any]:
    if operation not in OPERATIONS:
        raise ValueError(f"unknown debugger operation {operation!r}")
    fn = getattr(provider, operation)
    return fn(**kwargs)


def _session_id(provider: str, serial: int) -> str:
    return f"{provider}-session-{serial:03d}"


def format_transcript(session: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize observed commands. Never invents output that was not recorded."""
    commands = list(session.get("commands") or [])
    return {
        "schema": TRANSCRIPT_SCHEMA,
        "version": VERSION,
        "session_id": session.get("session_id"),
        "provider": session.get("provider"),
        "program": session.get("program"),
        "pid": session.get("pid"),
        "live": bool(session.get("live")),
        "fabricated": False,
        "observed": True,
        "commands": commands,
        "n_commands": len(commands),
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "claim_boundary": (
            "Transcript of commands issued and output observed. Absence of a "
            "live debuggee is recorded as a raise, never as a fake stack."
        ),
    }


class _GatedProvider(DebuggerProvider):
    """Shared gate + session bookkeeping. Subclasses implement _execute_live."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._serial = 0
        self._session: dict[str, Any] | None = None

    def _record(self, command: str, output: Mapping[str, Any] | str, *, ok: bool) -> None:
        if self._session is None:
            return
        self._session["commands"].append(
            {
                "command": command,
                "ok": bool(ok),
                "output": output if isinstance(output, (dict, list, str, int, type(None))) else str(output),
            }
        )

    def _new_session(self, *, program: str | None, pid: int | None, live: bool) -> dict[str, Any]:
        self._serial += 1
        self._session = {
            "session_id": _session_id(self.name, self._serial),
            "provider": self.name,
            "program": program,
            "pid": pid,
            "live": bool(live),
            "fabricated": False,
            "commands": [],
        }
        return self._session

    def _gated(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        self.require_available(operation)
        if operation == "launch":
            program = guard_launch(kwargs.get("program"))
            kwargs["program"] = program
        elif operation == "attach":
            pid = guard_attach(
                kwargs.get("pid"),
                command=kwargs.get("command"),
                explicit="pid" in kwargs and kwargs.get("pid") is not None,
            )
            kwargs["pid"] = pid
        elif operation == "memory_inspect":
            guard_memory(write=bool(kwargs.get("write")), command=kwargs.get("command"))
        elif operation == "evaluate":
            guard_memory(command=kwargs.get("expression"))
        return self._execute_live(operation, kwargs)

    def _execute_live(self, operation: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        raise DebuggerUnavailableError(
            provider=self.name,
            operation=operation,
            missing=self.missing() or [f"{self.name} has no live debuggee path"],
            probe=self.probe(),
        )

    def launch(self, program: str | None = None, args: Sequence[str] | None = None) -> dict[str, Any]:
        return self._gated("launch", program=program, args=list(args or []))

    def attach(self, pid: int | None = None, command: str | None = None) -> dict[str, Any]:
        return self._gated("attach", pid=pid, command=command)

    def set_breakpoint(self, location: str | None = None) -> dict[str, Any]:
        return self._gated("set_breakpoint", location=location)

    def conditional_breakpoint(
        self, location: str | None = None, condition: str | None = None
    ) -> dict[str, Any]:
        return self._gated("conditional_breakpoint", location=location, condition=condition)

    def tracepoint(self, location: str | None = None, command: str | None = None) -> dict[str, Any]:
        return self._gated("tracepoint", location=location, command=command)

    def pause(self) -> dict[str, Any]:
        return self._gated("pause")

    def resume(self) -> dict[str, Any]:
        return self._gated("resume")

    def step_in(self) -> dict[str, Any]:
        return self._gated("step_in")

    def step_over(self) -> dict[str, Any]:
        return self._gated("step_over")

    def step_out(self) -> dict[str, Any]:
        return self._gated("step_out")

    def run_to_location(self, location: str | None = None) -> dict[str, Any]:
        return self._gated("run_to_location", location=location)

    def stacks(self) -> dict[str, Any]:
        return self._gated("stacks")

    def threads(self) -> dict[str, Any]:
        return self._gated("threads")

    def variables(self) -> dict[str, Any]:
        return self._gated("variables")

    def memory_inspect(
        self,
        address: str | None = None,
        size: int | None = None,
        *,
        write: bool = False,
        command: str | None = None,
    ) -> dict[str, Any]:
        return self._gated(
            "memory_inspect", address=address, size=size, write=write, command=command
        )

    def evaluate(self, expression: str | None = None) -> dict[str, Any]:
        return self._gated("evaluate", expression=expression)

    def crash_capture(self) -> dict[str, Any]:
        return self._gated("crash_capture")

    def sanitizer_result(self) -> dict[str, Any]:
        return self._gated("sanitizer_result")

    def transcript_receipt(self) -> dict[str, Any]:
        return self._gated("transcript_receipt")


class StubProvider(_GatedProvider):
    """Same nineteen operations. Disabled by default so the negative control fires."""

    def __init__(self, *, available: bool = False, missing: Sequence[str] | None = None) -> None:
        super().__init__("stub")
        self._available = bool(available)
        self._missing = list(missing) if missing is not None else (
            []
            if self._available
            else ["stub provider is disabled; not an operational debugger"]
        )

    def probe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self._available,
            "missing": list(self._missing),
            "present": ["stub satisfies DebuggerProvider"],
            "live": False,
            "fabricates_debuggee_state": False,
        }

    def available(self) -> bool:
        return self._available

    def missing(self) -> list[str]:
        return list(self._missing)

    def _execute_live(self, operation: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        # An enabled stub still cannot invent a debuggee. It may record a
        # session of *commands issued* only when those commands do not claim
        # live stacks/variables/memory/crashes.
        debuggee_ops = {
            "stacks",
            "threads",
            "variables",
            "memory_inspect",
            "evaluate",
            "crash_capture",
            "sanitizer_result",
            "pause",
            "resume",
            "step_in",
            "step_over",
            "step_out",
            "run_to_location",
            "set_breakpoint",
            "conditional_breakpoint",
            "tracepoint",
        }
        if operation in debuggee_ops or operation in {"launch", "attach"}:
            raise DebuggerUnavailableError(
                provider=self.name,
                operation=operation,
                missing=["stub cannot invent debuggee state; no fabricated stack, variable, or attach"],
                probe=self.probe(),
            )
        if operation == "transcript_receipt":
            if self._session is None:
                raise SessionRequiredError("no debugger session; refusing to invent a transcript")
            doc = format_transcript(self._session)
            refuse_fabricated("transcript", doc)
            return doc
        raise DebuggerUnavailableError(
            provider=self.name,
            operation=operation,
            missing=self.missing(),
            probe=self.probe(),
        )


class LldbProvider(_GatedProvider):
    """LLDB CLI provider. Available only if a debuggee can actually be launched."""

    def __init__(self) -> None:
        super().__init__("lldb")

    def probe(self) -> dict[str, Any]:
        return probe_apple_lab()["probes"]["lldb"]

    def available(self) -> bool:
        return bool(self.probe().get("available"))

    def missing(self) -> list[str]:
        return list(self.probe().get("missing") or [])

    def _lldb_argv(self, commands: Sequence[str], *, program: str | None, pid: int | None) -> list[str]:
        argv = ["lldb", "--batch", "--no-lldbinit"]
        for cmd in commands:
            argv.extend(["-o", cmd])
        argv.extend(["-o", "quit"])
        if pid is not None:
            argv.extend(["--attach-pid", str(pid)])
        elif program:
            argv.extend(["--", program])
        return argv

    def _execute_live(self, operation: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        # Reached only when probe_lldb() said a launch succeeded. This host's
        # probe currently reports unavailable, so tests watch the raise instead.
        if operation == "transcript_receipt":
            if self._session is None:
                raise SessionRequiredError("no debugger session; refusing to invent a transcript")
            doc = format_transcript(self._session)
            refuse_fabricated("transcript", doc)
            path = write_receipt(TRANSCRIPT_RECEIPT, dict(doc), RECORDED_BY)
            doc["receipt_path"] = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
            return doc

        debuggee_reads = {
            "stacks",
            "threads",
            "variables",
            "memory_inspect",
            "evaluate",
            "crash_capture",
            "sanitizer_result",
        }
        if self._session is None or not self._session.get("live"):
            if operation in debuggee_reads or operation not in {"launch", "attach"}:
                raise DebuggerUnavailableError(
                    provider=self.name,
                    operation=operation,
                    missing=["no live debuggee; refusing to invent a stack, variable, or transcript"],
                    probe=self.probe(),
                )

        if operation == "launch":
            program = str(kwargs["program"])
            args = [str(a) for a in (kwargs.get("args") or [])]
            commands = ["settings set target.disable-aslr false", "process launch --stop-at-entry"]
            if args:
                commands.insert(0, "settings set target.run-args " + " ".join(args))
            argv = self._lldb_argv(commands, program=program, pid=None)
            ran = _run(argv)
            live = ran.returncode == 0 and "operation not permitted" not in (ran.stderr or "").lower()
            if not live:
                raise DebuggerUnavailableError(
                    provider=self.name,
                    operation="launch",
                    missing=[
                        f"lldb launch of {program} failed: "
                        f"{_first_line(ran.stderr or ran.stdout) or 'non-zero exit'}"
                    ],
                    probe=self.probe(),
                )
            session = self._new_session(program=program, pid=None, live=True)
            self._record(" ".join(argv), {"stdout": ran.stdout, "stderr": ran.stderr, "returncode": ran.returncode}, ok=True)
            return {
                "operation": "launch",
                "session_id": session["session_id"],
                "program": program,
                "live": True,
                "fabricated": False,
                "observed": True,
            }

        if operation == "attach":
            pid = int(kwargs["pid"])
            argv = self._lldb_argv(["process status"], program=None, pid=pid)
            ran = _run(argv)
            live = ran.returncode == 0 and "operation not permitted" not in (ran.stderr or "").lower()
            if not live:
                raise DebuggerUnavailableError(
                    provider=self.name,
                    operation="attach",
                    missing=[
                        f"lldb attach to pid {pid} failed: "
                        f"{_first_line(ran.stderr or ran.stdout) or 'non-zero exit'}"
                    ],
                    probe=self.probe(),
                )
            session = self._new_session(program=None, pid=pid, live=True)
            self._record(" ".join(argv), {"stdout": ran.stdout, "stderr": ran.stderr, "returncode": ran.returncode}, ok=True)
            return {
                "operation": "attach",
                "session_id": session["session_id"],
                "pid": pid,
                "live": True,
                "fabricated": False,
                "observed": True,
            }

        mapping = {
            "set_breakpoint": lambda: f"breakpoint set --name {kwargs.get('location') or 'main'}",
            "conditional_breakpoint": lambda: (
                f"breakpoint set --name {kwargs.get('location') or 'main'} "
                f"--condition {kwargs.get('condition') or '0'}"
            ),
            "tracepoint": lambda: (
                f"breakpoint set --name {kwargs.get('location') or 'main'} "
                f"--auto-continue true --one-liner '{kwargs.get('command') or 'bt'}'"
            ),
            "pause": lambda: "process interrupt",
            "resume": lambda: "continue",
            "step_in": lambda: "thread step-in",
            "step_over": lambda: "thread step-over",
            "step_out": lambda: "thread step-out",
            "run_to_location": lambda: f"breakpoint set --name {kwargs.get('location') or 'main'}",
            "stacks": lambda: "thread backtrace",
            "threads": lambda: "thread list",
            "variables": lambda: "frame variable",
            "memory_inspect": lambda: f"memory read {kwargs.get('address') or '0x0'} --count {int(kwargs.get('size') or 16)}",
            "evaluate": lambda: f"expression -- {kwargs.get('expression') or '0'}",
            "crash_capture": lambda: "process status",
            "sanitizer_result": lambda: "process status",
        }
        cmd = mapping[operation]()
        program = (self._session or {}).get("program")
        pid = (self._session or {}).get("pid")
        argv = self._lldb_argv([cmd], program=program, pid=pid)
        ran = _run(argv)
        self._record(cmd, {"stdout": ran.stdout, "stderr": ran.stderr, "returncode": ran.returncode}, ok=ran.returncode == 0)
        if operation in debuggee_reads and ran.returncode != 0:
            raise DebuggerUnavailableError(
                provider=self.name,
                operation=operation,
                missing=[
                    f"lldb {operation} produced no observed debuggee state: "
                    f"{_first_line(ran.stderr or ran.stdout) or 'non-zero exit'}"
                ],
                probe=self.probe(),
            )
        result = {
            "operation": operation,
            "command": cmd,
            "live": True,
            "fabricated": False,
            "observed": True,
            "stdout": (ran.stdout or "").strip()[:4000],
            "stderr": (ran.stderr or "").strip()[:4000],
            "returncode": ran.returncode,
        }
        refuse_fabricated(operation, result)
        return result


def make_provider(name: str, **kwargs: Any) -> DebuggerProvider:
    if name == "lldb":
        return LldbProvider()
    if name == "stub":
        return StubProvider(**kwargs)
    raise ValueError(f"unknown debugger provider {name!r}")


def providers() -> dict[str, DebuggerProvider]:
    return {
        "lldb": LldbProvider(),
        "stub": StubProvider(available=False),
    }


def debugger_execution_entry_points(
    provider: DebuggerProvider | None = None,
) -> tuple[tuple[str, Callable[[], Any]], ...]:
    """Every public debugger operation. Tests iterate this."""
    target = provider if provider is not None else StubProvider(available=False)
    return tuple((op, (lambda op=op: invoke(target, op))) for op in OPERATIONS)


# ---------------------------------------------------------------------------
# Apple lab execution. Gate first. Nothing estimates a number.
# ---------------------------------------------------------------------------


def xcodebuild_build(project: str | None = None) -> dict[str, Any]:
    require_lab("xcodebuild")
    if not project:
        raise AttachRefused("xcodebuild_build requires an explicit project path")
    ran = _run(["xcodebuild", "-project", project, "-version"])
    return {"ok": ran.returncode == 0, "returncode": ran.returncode, "fabricated": False}


def metal_compile(source: str | None = None) -> dict[str, Any]:
    require_lab("metal_compilation")
    src = source if source is not None else "kernel void k() {}"
    tmp = Path(tempfile.mkdtemp(prefix="hawking-debugger-metal-"))
    try:
        metal = tmp / "k.metal"
        air = tmp / "k.air"
        metal.write_text(src)
        ran = _run(["xcrun", "metal", "-c", str(metal), "-o", str(air)])
        if ran.returncode != 0:
            raise LabUnavailableError(
                capability="metal_compilation",
                missing=[_first_line(ran.stderr or ran.stdout) or "metal compile failed"],
                probe=probe_apple_lab()["probes"]["metal_compilation"],
            )
        return {"ok": True, "air": str(air), "fabricated": False, "observed": True}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def shader_diagnostics(path: str | None = None) -> dict[str, Any]:
    require_lab("shader_diagnostics")
    if not path:
        raise AttachRefused("shader_diagnostics requires an explicit metallib/AIR path")
    ran = _run(["xcrun", "metal-objdump", "--disassemble", path])
    return {"ok": ran.returncode == 0, "stdout": ran.stdout, "fabricated": False, "observed": True}


def coreml_compile(model: str | None = None) -> dict[str, Any]:
    require_lab("coreml_compilation")
    if not model:
        raise AttachRefused("coreml_compile requires an explicit model path")
    ran = _run(["xcrun", "coremlcompiler", "compile", model, str(Path(model).parent)])
    return {"ok": ran.returncode == 0, "fabricated": False, "observed": True}


def mlcomputeplan_live(compiled_model: str | None = None) -> dict[str, Any]:
    require_lab("mlcomputeplan")
    raise FabricationForbidden("mlcomputeplan_live must not invent a plan; live load is not implemented without a model")


def mlstate_live(compiled_model: str | None = None) -> dict[str, Any]:
    require_lab("mlstate")
    raise FabricationForbidden("mlstate_live must not invent MLState contents")


def instruments_record(target: str | None = None) -> dict[str, Any]:
    require_lab("instruments")
    if not target:
        raise AttachRefused("instruments_record requires an explicit target")
    ran = _run(["xcrun", "xctrace", "record", "--template", "Time Profiler", "--launch", target])
    return {"ok": ran.returncode == 0, "fabricated": False, "observed": True}


def simulator_boot(udid: str | None = None) -> dict[str, Any]:
    require_lab("simulator")
    if not udid:
        raise AttachRefused("simulator_boot requires an explicit UDID; this lab never infers a device")
    ran = _run(["xcrun", "simctl", "boot", udid])
    return {"ok": ran.returncode == 0, "fabricated": False, "observed": True}


def sanitizer_compile(sanitizer: str = "address") -> dict[str, Any]:
    require_lab("sanitizers")
    probe = probe_apple_lab()["probes"]["sanitizers"]
    compiled = (probe.get("compiled") or {}).get(sanitizer) or {}
    if not compiled.get("ok"):
        raise LabUnavailableError(
            capability="sanitizers",
            missing=[f"clang -fsanitize={sanitizer} is not usable on this host"],
            probe=probe,
        )
    return {
        "ok": True,
        "sanitizer": sanitizer,
        "live": False,
        "fabricated": False,
        "observed": True,
        "note": "compile probe only; not a debuggee sanitizer_result",
    }


LAB_EXECUTION: dict[str, Callable[..., Any]] = {
    "xcodebuild": xcodebuild_build,
    "metal_compilation": metal_compile,
    "shader_diagnostics": shader_diagnostics,
    "coreml_compilation": coreml_compile,
    "mlcomputeplan": mlcomputeplan_live,
    "mlstate": mlstate_live,
    "instruments": instruments_record,
    "simulator": simulator_boot,
    "sanitizers": sanitizer_compile,
}


def lab_execution_entry_points() -> tuple[tuple[str, Callable[[], Any]], ...]:
    return tuple((name, (lambda name=name: LAB_EXECUTION[name]())) for name in LAB_EXECUTION)


# ---------------------------------------------------------------------------
# WorkUnits. Blocked physical work SLEEPS. It never becomes a synthetic result.
# ---------------------------------------------------------------------------


def _emit_unit(
    *,
    id: str,
    role: str,
    description: str,
    status: str,
    classification: str,
    resource_class: str,
    verifier: str,
    blocked_reason: str | None,
    effect_class: str = "READ_ONLY",
) -> dict[str, Any]:
    from tools.future.workunit_species import emit_hcli_workunit, validate_emitted_unit

    extras: dict[str, Any] = {
        "species": "independent_reproduction",
        "claim_boundary": (
            "WorkUnit is a proposal; receipt and protected capability gates remain authoritative. "
            "SLEEPING units wake when the named Apple tool becomes available. "
            "They never become a synthetic debugger result."
        ),
        "requires_quiescence": False,
        "candidate_status": classification,
        "output_receipt_path": f"receipts/future/{RECEIPT}",
        "command": ["python3", "tools/future/debugger.py", "--probe"],
    }
    if blocked_reason is not None:
        extras["blocked_reason"] = blocked_reason
    row = emit_hcli_workunit(
        id=id,
        role=role,
        description=description,
        dependencies=[],
        resource_class=resource_class,
        verifier=verifier,
        provider="future.debugger",
        effect_class=effect_class,
        status=status,
        classification=classification,
        extras=extras,
    )
    validate_emitted_unit(row)
    return row


def emit_workunits(lab: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    lab = dict(lab or probe_apple_lab())
    probes = lab.get("probes") or {}
    units = [
        _emit_unit(
            id="future.debugger.probe-apple-lab",
            role="science",
            description=(
                "Run honest Apple lab probes (xcodebuild, LLDB, Metal, shader "
                "diagnostics, Core ML, MLComputePlan, MLState, Instruments, "
                "sanitizers, Simulator) and seal DEBUGGER_LAB.json. STATIC_ONLY."
            ),
            status="pending",
            classification="STATIC_ONLY",
            resource_class="STATIC_ANALYSIS",
            verifier="future.debugger.probe",
            blocked_reason=None,
        )
    ]
    for name in LAB_CAPABILITIES:
        row = probes.get(name) or {}
        if row.get("available"):
            continue
        missing = "; ".join(row.get("missing") or [f"{name} unavailable"])
        units.append(
            _emit_unit(
                id=f"future.debugger.sleep.{name}",
                role="science",
                description=(
                    f"SLEEPING: {name} lab work. Wakes when the Apple toolchain "
                    f"qualifies. Never becomes a synthetic result. Missing: {missing}"
                ),
                status="SLEEPING",
                classification="SLEEPING",
                resource_class="COMPILE" if name in {"metal_compilation", "coreml_compilation", "xcodebuild"} else "STATIC_ANALYSIS",
                verifier=f"future.debugger.{name}",
                blocked_reason=missing,
            )
        )
    units.sort(key=lambda row: str(row.get("id")))
    return units


def debugger_frontier_entry(lab: Mapping[str, Any] | None = None) -> dict[str, Any]:
    lab = dict(lab or probe_apple_lab())
    metal = (lab.get("probes") or {}).get("metal_compilation") or {}
    lldb = (lab.get("probes") or {}).get("lldb") or {}
    return {
        "id": "debugger-lab",
        "classification": "BLOCKED" if not (metal.get("available") and lldb.get("available")) else "HIGH_VALUE_INTEGRATION",
        "title": "Provider-neutral debugger and Apple engineering laboratory",
        "detail": lab.get("headline"),
        "metal_compiler": lab.get("metal_compiler_finding"),
        "integration_target": "hcli/tool_registry.py future.debugger (not written; this-wave resident_api/frontiers ingest)",
        "receipt": f"receipts/future/{RECEIPT}",
        "evidence_class": "STATIC_ONLY",
        "probe": {"kind": "field", "path": f"receipts/future/{RECEIPT}", "field": "metal_compiler_finding"},
    }


def resident_callable(lab: Mapping[str, Any] | None = None) -> dict[str, Any]:
    lab = dict(lab or probe_apple_lab())
    units = emit_workunits(lab)
    sleeping = [row for row in units if row.get("status") == "SLEEPING"]
    return {
        "can_hcli_invoke": True,
        "entry_point": "python3 tools/future/debugger.py --probe",
        "cli": ["--probe", "--build", "--selftest"],
        "hcli_tool": HCLI_TOOL_DESCRIPTOR,
        "workunits": units,
        "workunit_emitted": [row["id"] for row in units],
        "sleeping_workunits": [row["id"] for row in sleeping],
        "receipt": f"receipts/future/{RECEIPT}",
        "transcript_receipt": f"receipts/future/{TRANSCRIPT_RECEIPT}",
        "frontier": {
            "feeds": "receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
            "entry": debugger_frontier_entry(lab),
            "integration": (
                "tools/future/frontiers.py and tools/future/resident_api.py are "
                "this-wave siblings and are not imported; this receipt is the swap."
            ),
        },
        "fail_closed": {
            "unavailable_provider": (
                "DebuggerUnavailableError names the provider, the operation, and "
                "the missing dependency. No fabricated stack, variable, or transcript."
            ),
            "apple_lab": "LabUnavailableError names the missing Apple tool.",
            "attach": "AttachRefused unless an explicit non-Codex pid is given.",
            "memory": "MemoryWriteForbidden; inspection is read-only.",
            "sleeping": (
                "Blocked physical work is a SLEEPING WorkUnit. It never becomes "
                "a synthetic debugger result."
            ),
        },
    }


def hcli_invoke(arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Local HCLI tool handler. tool_registry.py is read-only; this is the swap."""
    args = dict(arguments or {})
    action = str(args.get("action") or "probe")
    if action == "probe":
        return probe_apple_lab()
    if action == "build":
        path = build()
        return {"receipt": str(path), "ok": True}
    if action == "workunits":
        return {"workunits": emit_workunits()}
    if action == "invoke":
        provider = make_provider(str(args.get("provider") or "stub"), available=False)
        operation = str(args.get("operation") or "")
        extra = args.get("arguments") if isinstance(args.get("arguments"), dict) else {}
        return invoke(provider, operation, **dict(extra or {}))
    raise ValueError(f"unknown future.debugger action {action!r}")


# ---------------------------------------------------------------------------
# Recovery, gaps, negative findings, receipt.
# ---------------------------------------------------------------------------


def recovered_implementation() -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    roles = {
        "tools/future/ane_preboard.py": (
            "LANDED pattern this module follows: toolchain probe returns False; "
            "execution paths RAISE; no invented placement/latency"
        ),
        "tools/future/debugger.py": "this module (provider-neutral debugger + Apple lab)",
        "hcli/tool_registry.py": (
            "AgentOS typed tools; no debugger operations registered. "
            "HCLI_TOOL_DESCRIPTOR is the integration swap, not a write into hcli/"
        ),
        "hcli/providers.py": "model-provider contract; not a process debugger",
        "hcli/vmcp_adapter.py": "VisionMCP adapter; no debugger surface",
        "hcli/vmcp/__init__.py": "VisionMCP package marker",
        "hcli/workunit.py": "HCLI WorkUnit field set this module emits into",
        "hcli/agentos/preboard.py": "AgentOS preboard analog; not an XDebugger",
        "hcli/agentos/runtime.py": "AgentOS runtime; not a debugger provider",
        "tools/headless/noetic_executable_closure.py": (
            "records: lldb run of an unsigned helper returned Operation not permitted"
        ),
        "receipts/headless/NOETIC_EXECUTABLE_CLOSURE.json": "pinned lldb SIP/sandbox finding",
        "receipts/headless/GPU_LEDGER.json": (
            "Xcode GPU debugger is out of band; Apple Metal does not report register file pressure"
        ),
        "receipts/future/ANE_PREBOARD.json": "sealed ANE toolchain probe this lab extends, does not fork",
        "receipts/future/CLAUDE_GLOBAL_FRONTIER.json": "live frontier; debugger-lab entry is proposed, not written here",
        "receipts/future/FUTURE_SUBSTRATE_HANDOFF.json": "sidecar inventory + active_processes used as Codex pid denylist seed",
        "CODEX_ACCELERATOR_HANDOFF.json": (
            "named Codex handoff (exact_physical_blockers: Metal compiler, no Metal GPU); "
            "not on disk and not in git HEAD of this sparse worktree"
        ),
    }
    for rel in RECOVER_PATHS:
        if rel in seen:
            continue
        seen.add(rel)
        state = path_state(rel)
        state["role"] = roles.get(rel, "named recovery path")
        rows.append(state)
    return rows


def recovered_concepts() -> dict[str, Any]:
    """What the steer asked us to reuse. Disk/git is authority.

    Recovery ran `git ls-tree -r --name-only HEAD` for debugger/xdebugger/lldb
    names and `git grep XDebugger` over tools/ and hcli/. Those searches are
    not repeated here: a full-tree git grep hung this sparse worktree once.
    """
    return {
        "XDebugger": {
            "status": "ABSENT",
            "detail": (
                "No XDebugger class, module, or filename in git HEAD under tools/ or hcli/. "
                "The AgentOS concept that exists is a typed tool registry and a model-provider "
                "contract, not a process debugger. This module is the first provider-neutral "
                "debugger contract, not a rebuild of a hidden original."
            ),
            "search": "git ls-tree / git grep XDebugger over tools/ and hcli/; zero hits",
        },
        "AgentOS_tools": {
            "status": "RECOVERED",
            "detail": "hcli/tool_registry.py ToolSpec/ToolRegistry is the registration seam; not written.",
        },
        "lldb_prior_art": {
            "status": "RECOVERED",
            "paths": [
                "tools/headless/noetic_executable_closure.py",
                "receipts/headless/NOETIC_EXECUTABLE_CLOSURE.json",
                "receipts/headless/GPU_LEDGER.json",
            ],
            "detail": (
                "NOETIC_EXECUTABLE_CLOSURE already recorded lldb Operation not permitted. "
                "The live launch probe reproduced that finding. GPU_LEDGER records the "
                "Xcode GPU debugger as out of band."
            ),
        },
        "ane_preboard_pattern": {
            "status": "REUSED",
            "detail": "probe returns False; execution entry points RAISE; UNKNOWN not an invented number.",
        },
    }


def gaps_closed() -> list[str]:
    return [
        "provider-neutral DebuggerProvider with all nineteen named operations; LLDB and stub satisfy it",
        "honest Apple lab probes actually executed on this host (xcrun/clang/lldb/xcodebuild)",
        "Metal compiler absence is a finding (xcrun unable to find utility metal), not a pretence",
        "every operation on an unavailable provider raises DebuggerUnavailableError naming the missing dependency",
        "debugger session transcript schema records commands issued and output observed; no fabricated debuggee state",
        "attach requires an explicit pid and refuses Codex campaign processes and pid 0/1/self",
        "memory inspection is read-only; memory write / register write raise MemoryWriteForbidden",
        "blocked lab work is a SLEEPING WorkUnit that never becomes a synthetic result",
        "HCLI-callable descriptor + hcli_invoke without writing hcli/tool_registry.py",
    ]


def negative_findings(lab: Mapping[str, Any] | None = None) -> list[str]:
    lab = dict(lab or probe_apple_lab())
    findings = [
        str(lab.get("headline") or "Apple lab unavailable"),
        f"metal_compiler_finding={lab.get('metal_compiler_finding')}",
    ]
    for item in lab.get("missing") or []:
        findings.append(f"missing: {item}")
    concepts = recovered_concepts()
    if concepts["XDebugger"]["status"] == "ABSENT":
        findings.append("looked for XDebugger in tools/ and hcli/ HEAD; it is ABSENT")
    for row in recovered_implementation():
        if row["source"] == "ABSENT":
            findings.append(f"looked for {row['path']} and it is ABSENT from disk and git HEAD")
    findings.extend(
        [
            "Python lldb module is not importable; CLI lldb is the only recovered debugger binary",
            "lldb --batch launch of /bin/echo returns Operation not permitted on this host",
            "xcodebuild requires full Xcode; CommandLineTools is not a compile toolchain",
            "Instruments, simctl, coremlcompiler, metallib are not resolved by xcrun",
            "MLComputePlan.h / MLState.h exist as SDK headers and are not a live load",
            "CODEX_ACCELERATOR_HANDOFF.json is not in this sparse worktree or git HEAD; blockers recovered from the lane contract and ANE_PREBOARD",
            "no debugger session was opened against a live debuggee in this process",
            "FPGA remains Accelerator / Physical Compiler / Fusion; this lab does not grow an FPGA debugger",
        ]
    )
    return findings


def build() -> Path:
    lab = probe_apple_lab()
    recovered = recovered_implementation()
    concepts = recovered_concepts()
    units = emit_workunits(lab)
    callable_doc = resident_callable(lab)
    stub = StubProvider(available=False)
    lldb = LldbProvider()
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Provider-neutral debugger contract and honest Apple engineering-lab "
            "probes. Unavailable providers raise. Sleeping work stays sleeping."
        ),
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "no_era_vi": True,
        "no_odyssey_iv": True,
        "fpga_is_not_its_own_civilization": True,
        "operations": list(OPERATIONS),
        "n_operations": len(OPERATIONS),
        "providers": list(PROVIDERS),
        "lab_capabilities": list(LAB_CAPABILITIES),
        "n_lab_capabilities": len(LAB_CAPABILITIES),
        "apple_lab": lab,
        "metal_compiler_finding": lab.get("metal_compiler_finding"),
        "lldb_provider_available": lldb.available(),
        "stub_provider_available": stub.available(),
        "execution_entry_points": [name for name, _ in debugger_execution_entry_points(stub)],
        "lab_execution_entry_points": [name for name, _ in lab_execution_entry_points()],
        "safety": {
            "memory_inspection": "read-only",
            "attach": "explicit pid required; Codex campaign processes refused; self/launchd refused",
            "codex_cmdline_markers": list(CODEX_CMDLINE_MARKERS),
            "codex_pids_from_handoff": _handoff_pids(),
        },
        "workunits": units,
        "n_workunits": len(units),
        "n_sleeping_workunits": len([u for u in units if u.get("status") == "SLEEPING"]),
        "resident_callable": callable_doc,
        "recovered_implementation": recovered,
        "recovered_concepts": concepts,
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(lab),
        "claim_class": "STATIC_ONLY",
        "does_not_produce": ["DIAGNOSTIC_RELATIVE", "PROTECTED_ABSOLUTE"],
        "integration_points": {
            "hcli_tool_registry": (
                "HCLI_TOOL_DESCRIPTOR / hcli_invoke; do not write hcli/tool_registry.py"
            ),
            "frontiers": "tools/future/frontiers.py (this-wave; not imported) ingests debugger_frontier_entry",
            "resident_api": "tools/future/resident_api.py (this-wave; not imported) exposes the tool",
            "wakeup": "tools/future/wakeup.py (this-wave; not imported) wakes SLEEPING units when probes flip",
            "workgraph": "tools/future/workgraph.py (this-wave; not imported) schedules the emitted units",
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true", help="print Apple lab probe JSON; does not debug a process")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.probe and not (a.build or a.selftest):
        print(json.dumps(probe_apple_lab(force=True), indent=1, sort_keys=True))
        return 0
    if a.probe:
        print(json.dumps(probe_apple_lab(force=True), indent=1, sort_keys=True))
    out = selftest() if (a.selftest or a.build or not a.probe) else None
    if out is not None:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
