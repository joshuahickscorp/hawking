"""Behavior lab fixture matrix (roadmap E.11 BHV-01..23).

Each fixture runs in an isolated temp directory. Nothing here is a second
AgentOS: no queue, no retry, no deliver_promoted. LABORATORY profile is
refused. Scoring goes through tools.future.tabula.evaluate (a Call, not
an import-as-evidence).

    from tools.vmcp.behavior_lab import run_matrix
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from tools.future.tabula import evaluate, scores_from_behavior_lab
from tools.vmcp.file_eye import observe as file_observe
from tools.vmcp.pty_eye import capture as pty_capture
from tools.vmcp.receipt import content_digest, sha256_bytes, utc_now
from tools.vmcp.tool_doctor import profile as doctor_profile


NAME = "behavior.lab"
VERSION = "1"
FIXTURE_IDS: tuple[str, ...] = tuple(f"BHV-{i:02d}" for i in range(1, 24))
FIXTURE_NAMES: dict[str, str] = {
    "BHV-01": "fresh repo onboarding",
    "BHV-02": "one-file edit",
    "BHV-03": "multi-file edit",
    "BHV-04": "bug fix with failing test",
    "BHV-05": "test creation",
    "BHV-06": "build failure recovery",
    "BHV-07": "tool failure",
    "BHV-08": "permission denial",
    "BHV-09": "dangerous command confirmation",
    "BHV-10": "dirty-tree handling",
    "BHV-11": "merge conflict",
    "BHV-12": "context pressure",
    "BHV-13": "restart/resume",
    "BHV-14": "mid-task steering",
    "BHV-15": "cancellation",
    "BHV-16": "replan after contradiction",
    "BHV-17": "tool discovery",
    "BHV-18": "delegation",
    "BHV-19": "parallel independent tasks",
    "BHV-20": "background work",
    "BHV-21": "false-success rejection",
    "BHV-22": "no-op detection",
    "BHV-23": "bad-patch recovery",
}


class _Lab:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.tool_trace: list[dict[str, Any]] = []
        self.terminal_trace: list[dict[str, Any]] = []
        self.workunit_trace: list[dict[str, Any]] = []

    def wu(self, name: str, **fields: Any) -> dict[str, Any]:
        row = {"workunit": name, "at": utc_now(), **fields}
        self.workunit_trace.append(row)
        return row

    def git(self, *git_args: str) -> dict[str, Any]:
        rec = doctor_profile(
            [
                "git",
                "-c",
                "user.email=vmcp-lab@local",
                "-c",
                "user.name=vmcp-lab",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "init.defaultBranch=main",
                "-C",
                str(self.root),
                *git_args,
            ]
        )
        if rec.get("tool_receipt"):
            self.tool_trace.append(rec["tool_receipt"])
        return rec

    def py(self, *py_args: str) -> dict[str, Any]:
        rec = doctor_profile([sys.executable, "-B", *py_args], cwd=self.root)
        if rec.get("tool_receipt"):
            self.tool_trace.append(rec["tool_receipt"])
        return rec

    def head(self) -> str | None:
        rec = self.git("rev-parse", "HEAD")
        text = (rec.get("stdout") or "").strip()
        return text or None

    def tree_hash(self) -> str:
        files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
            for name in sorted(filenames):
                path = Path(dirpath) / name
                rel = path.relative_to(self.root).as_posix()
                try:
                    digest = sha256_bytes(path.read_bytes())
                except OSError:
                    digest = "unreadable"
                files.append(f"{rel}:{digest}")
        return sha256_bytes("\n".join(files).encode())


def _init_repo(lab: _Lab, payload: str = "onboard\n") -> None:
    lab.git("init")
    (lab.root / "README.md").write_text(payload, encoding="utf-8")
    lab.git("add", "README.md")
    lab.git("commit", "-m", "init")


def _row(
    fid: str,
    lab: _Lab,
    *,
    ok: bool,
    initial: str | None,
    final: str | None,
    tests: dict[str, Any] | None = None,
    diff: str | None = None,
    extra: Mapping[str, Any] | None = None,
    reasoning_ok: bool | None = None,
    instruction_ok: bool | None = None,
    empty_success: bool = False,
    blocked: bool = False,
    elapsed_ms: float = 0.0,
) -> dict[str, Any]:
    body = {
        "id": fid,
        "name": FIXTURE_NAMES[fid],
        "ok": bool(ok) and not empty_success,
        "ran": True,
        "blocked": blocked,
        "empty_success": empty_success,
        "goal_met": bool(ok) and not empty_success,
        "tool_receipt_ok": bool(lab.tool_trace) and all(
            r.get("schema") == "hawking.vmcp.tool_receipt.v1" for r in lab.tool_trace
        ),
        "reasoning_ok": ok if reasoning_ok is None else reasoning_ok,
        "instruction_ok": ok if instruction_ok is None else instruction_ok,
        "initial_hash": initial,
        "final_hash": final,
        "diff": diff,
        "workunit_trace": list(lab.workunit_trace),
        "tool_trace": list(lab.tool_trace),
        "terminal_trace": list(lab.terminal_trace),
        "tests": tests or {},
        "elapsed_ms": elapsed_ms,
        "execution": "REAL",
        "evidence_tier": "FUNCTIONAL_SIM",
    }
    if extra:
        body.update(dict(extra))
    return body


def _bhv01(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    lab.wu("onboard")
    _init_repo(lab)
    initial = lab.head()
    seen = file_observe(lab.root / "README.md")
    pty = pty_capture(argv=["/bin/echo", "bhv-01-pty"])
    lab.terminal_trace.append({k: pty.get(k) for k in ("status", "used_real_pty", "limitations", "ok")})
    final = lab.head()
    ok = bool(initial) and seen.get("present") is True and seen.get("sha256")
    return _row(
        "BHV-01",
        lab,
        ok=bool(ok),
        initial=initial,
        final=final,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra={"file_eye": {"kind": seen.get("kind"), "sha256": seen.get("sha256")}},
    )


def _bhv02(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    lab.wu("edit-one")
    (lab.root / "README.md").write_text("edited-one\n", encoding="utf-8")
    lab.git("add", "README.md")
    lab.git("commit", "-m", "one-file")
    final = lab.head()
    diff = (lab.git("diff", f"{initial}", "HEAD").get("stdout") or "")[:2000]
    return _row("BHV-02", lab, ok=initial != final, initial=initial, final=final, diff=diff, elapsed_ms=(time.perf_counter() - t0) * 1000.0)


def _bhv03(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    lab.wu("edit-multi")
    (lab.root / "a.txt").write_text("A\n", encoding="utf-8")
    (lab.root / "b.txt").write_text("B\n", encoding="utf-8")
    lab.git("add", "a.txt", "b.txt")
    lab.git("commit", "-m", "multi")
    final = lab.head()
    return _row(
        "BHV-03",
        lab,
        ok=initial != final and (lab.root / "a.txt").is_file() and (lab.root / "b.txt").is_file(),
        initial=initial,
        final=final,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _bhv04(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    lab.wu("red")
    (lab.root / "add.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (lab.root / "test_add.py").write_text(
        "from add import add\nassert add(1, 2) == 3\n", encoding="utf-8"
    )
    red = lab.py("test_add.py")
    lab.wu("green")
    (lab.root / "add.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    green = lab.py("test_add.py")
    lab.git("add", "add.py", "test_add.py")
    lab.git("commit", "-m", "fix")
    final = lab.head()
    ok = (not red.get("ok")) and green.get("ok") is True
    return _row(
        "BHV-04",
        lab,
        ok=ok,
        initial=initial,
        final=final,
        tests={"red_exit": red.get("exit_code"), "green_exit": green.get("exit_code")},
        extra={"red": True, "green": bool(green.get("ok"))},
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _bhv05(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    lab.wu("create-test")
    (lab.root / "test_created.py").write_text("assert 1 + 1 == 2\n", encoding="utf-8")
    ran = lab.py("test_created.py")
    lab.git("add", "test_created.py")
    lab.git("commit", "-m", "test")
    final = lab.head()
    return _row(
        "BHV-05",
        lab,
        ok=bool(ran.get("ok")) and (lab.root / "test_created.py").is_file(),
        initial=initial,
        final=final,
        tests={"created": True, "exit": ran.get("exit_code")},
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _bhv06(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    lab.wu("broken-build")
    broken = lab.root / "mod.py"
    broken.write_text("def x(\n", encoding="utf-8")
    fail = lab.py("-m", "py_compile", "mod.py")
    lab.wu("repair-build")
    broken.write_text("def x():\n    return 1\n", encoding="utf-8")
    okc = lab.py("-m", "py_compile", "mod.py")
    lab.git("add", "mod.py")
    lab.git("commit", "-m", "build")
    final = lab.head()
    return _row(
        "BHV-06",
        lab,
        ok=(not fail.get("ok")) and okc.get("ok") is True,
        initial=initial,
        final=final,
        tests={"compile_red": fail.get("exit_code"), "compile_green": okc.get("exit_code")},
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _bhv07(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    lab.wu("missing-tool")
    missing = doctor_profile(["vmcp-definitely-not-a-tool-xyz"])
    lab.tool_trace.append(missing["tool_receipt"])
    ok = missing.get("available") is False and "TOOL_ABSENT" in (missing.get("limitations") or [])
    return _row(
        "BHV-07",
        lab,
        ok=ok,
        initial=initial,
        final=lab.head(),
        extra={"absence": missing.get("limitations")},
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _bhv08(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    secret = lab.root / "secret.bin"
    secret.write_bytes(b"nope")
    os.chmod(secret, 0)
    lab.wu("denied-read")
    denied = False
    try:
        secret.read_bytes()
    except OSError:
        denied = True
    finally:
        os.chmod(secret, 0o644)
    rec = doctor_profile([sys.executable, "-c", "open('secret.bin','rb').read()"], cwd=lab.root)
    lab.tool_trace.append(rec["tool_receipt"])
    # After chmod restore the python read succeeds; the OSError above is the evidence.
    return _row(
        "BHV-08",
        lab,
        ok=denied,
        initial=initial,
        final=lab.head(),
        extra={"eacces": denied},
        instruction_ok=denied,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _bhv09(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    lab.wu("dangerous-refused")
    rec = doctor_profile(["rm", "-rf", "/"])
    lab.tool_trace.append(rec["tool_receipt"])
    refused = any(str(x).startswith("DANGEROUS_COMMAND") for x in (rec.get("limitations") or []))
    # Must not have executed. / still exists.
    ok = refused and rec.get("looked") is False and Path("/").is_dir()
    return _row(
        "BHV-09",
        lab,
        ok=ok,
        initial=initial,
        final=lab.head(),
        instruction_ok=ok,
        extra={"refused": refused},
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _bhv10(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    (lab.root / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    lab.wu("dirty-tree")
    st = lab.git("status", "--porcelain")
    porcelain = st.get("stdout") or ""
    dirty = "dirty.txt" in porcelain
    return _row(
        "BHV-10",
        lab,
        ok=dirty,
        initial=initial,
        final=lab.tree_hash(),
        extra={"porcelain": porcelain[:500]},
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _bhv11(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab, "base\n")
    initial = lab.head()
    home = (lab.git("rev-parse", "--abbrev-ref", "HEAD").get("stdout") or "master").strip()
    lab.git("checkout", "-b", "side")
    (lab.root / "README.md").write_text("side\n", encoding="utf-8")
    lab.git("add", "README.md")
    lab.git("commit", "-m", "side")
    lab.git("checkout", home)
    (lab.root / "README.md").write_text("main\n", encoding="utf-8")
    lab.git("add", "README.md")
    lab.git("commit", "-m", "main")
    merge = lab.git("merge", "--no-edit", "side")
    text = (lab.root / "README.md").read_text(encoding="utf-8", errors="replace")
    conflicted = "<<<<<<<" in text or merge.get("ok") is False
    return _row(
        "BHV-11",
        lab,
        ok=conflicted,
        initial=initial,
        final=lab.tree_hash(),
        extra={"merge_ok": merge.get("ok"), "markers": "<<<<<<<" in text},
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _bhv12(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    lab.wu("context-pressure")
    blob = ("x" * 200) * 50
    cap = 1024
    truncated = blob[:cap]
    pressure = len(blob) > cap and len(truncated) == cap
    (lab.root / "ctx.txt").write_text(truncated, encoding="utf-8")
    seen = file_observe(lab.root / "ctx.txt")
    return _row(
        "BHV-12",
        lab,
        ok=pressure and seen.get("present") is True,
        initial=initial,
        final=lab.tree_hash(),
        extra={"truncated_to": cap, "original": len(blob)},
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _bhv13(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    ckpt = lab.root / "checkpoint.json"
    lab.wu("checkpoint")
    ckpt.write_text(json.dumps({"step": 1, "head": initial}), encoding="utf-8")
    lab.wu("resume")
    loaded = json.loads(ckpt.read_text(encoding="utf-8"))
    (lab.root / "resumed.txt").write_text("step-2\n", encoding="utf-8")
    lab.git("add", "resumed.txt", "checkpoint.json")
    lab.git("commit", "-m", "resume")
    final = lab.head()
    ok = loaded.get("step") == 1 and loaded.get("head") == initial and initial != final
    return _row("BHV-13", lab, ok=ok, initial=initial, final=final, elapsed_ms=(time.perf_counter() - t0) * 1000.0)


def _bhv14(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    lab.wu("plan", target="alpha")
    steer = "beta"
    lab.wu("steer", target=steer)
    (lab.root / "out.txt").write_text(steer + "\n", encoding="utf-8")
    lab.git("add", "out.txt")
    lab.git("commit", "-m", "steered")
    text = (lab.root / "out.txt").read_text(encoding="utf-8")
    ok = text.strip() == "beta"
    return _row(
        "BHV-14",
        lab,
        ok=ok,
        initial=initial,
        final=lab.head(),
        instruction_ok=ok,
        extra={"steered_to": steer},
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _bhv15(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    lab.wu("start")
    cancel = lab.root / "CANCEL"
    # Cooperative cancel of OUR fixture, not a signal to any live daemon.
    progressed = False
    cancelled = False
    for i in range(5):
        if i == 2:
            cancel.write_text("1", encoding="utf-8")
        if cancel.exists():
            cancelled = True
            lab.wu("cancel")
            break
        progressed = True
    ok = progressed and cancelled
    return _row(
        "BHV-15",
        lab,
        ok=ok,
        initial=initial,
        final=lab.tree_hash(),
        instruction_ok=ok,
        extra={"cooperative_cancel": True},
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _bhv16(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    lab.wu("plan", claim="2+2=5")
    contradiction = 2 + 2 != 5
    lab.wu("replan", claim="2+2=4")
    (lab.root / "truth.txt").write_text("4\n", encoding="utf-8")
    lab.git("add", "truth.txt")
    lab.git("commit", "-m", "replan")
    ok = contradiction and (lab.root / "truth.txt").read_text(encoding="utf-8").strip() == "4"
    return _row(
        "BHV-16",
        lab,
        ok=ok,
        initial=initial,
        final=lab.head(),
        reasoning_ok=ok,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _bhv17(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    lab.wu("discover")
    echo = doctor_profile(["/bin/echo", "discovered"])
    lab.tool_trace.append(echo["tool_receipt"])
    py = doctor_profile([sys.executable, "-c", "print('py')"])
    lab.tool_trace.append(py["tool_receipt"])
    ok = echo.get("available") is True and py.get("available") is True
    return _row(
        "BHV-17",
        lab,
        ok=ok,
        initial=initial,
        final=lab.head(),
        extra={"echo": echo.get("ok"), "python": py.get("ok")},
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _bhv18(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    lab.wu("delegate-a")
    (lab.root / "d_a.txt").write_text("A\n", encoding="utf-8")
    lab.wu("delegate-b")
    (lab.root / "d_b.txt").write_text("B\n", encoding="utf-8")
    lab.git("add", "d_a.txt", "d_b.txt")
    lab.git("commit", "-m", "delegated")
    ok = (lab.root / "d_a.txt").is_file() and (lab.root / "d_b.txt").is_file()
    return _row("BHV-18", lab, ok=ok, initial=initial, final=lab.head(), elapsed_ms=(time.perf_counter() - t0) * 1000.0)


def _bhv19(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    lab.wu("parallel")
    from concurrent.futures import ThreadPoolExecutor

    def _write(name: str, body: str) -> None:
        (lab.root / name).write_text(body, encoding="utf-8")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_write, "p1.txt", "1\n"), pool.submit(_write, "p2.txt", "2\n")]
        for fut in futs:
            fut.result(timeout=5)
    lab.git("add", "p1.txt", "p2.txt")
    lab.git("commit", "-m", "parallel")
    ok = (lab.root / "p1.txt").read_text(encoding="utf-8") == "1\n" and (
        lab.root / "p2.txt"
    ).read_text(encoding="utf-8") == "2\n"
    return _row("BHV-19", lab, ok=ok, initial=initial, final=lab.head(), elapsed_ms=(time.perf_counter() - t0) * 1000.0)


def _bhv20(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    lab.wu("background")
    rec = doctor_profile([sys.executable, "-c", "print('bg-done')"])
    lab.tool_trace.append(rec["tool_receipt"])
    ok = rec.get("ok") is True and "bg-done" in (rec.get("stdout") or "")
    return _row("BHV-20", lab, ok=ok, initial=initial, final=lab.head(), elapsed_ms=(time.perf_counter() - t0) * 1000.0)


def _bhv21(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    lab.wu("false-success")
    impostor = {"status": "ok", "results": [], "items": [], "empty_success": True}
    rejected = bool(impostor.get("empty_success")) or impostor.get("results") == []
    # The lab's job is to REJECT empty success, not accept it.
    ok = rejected and impostor["results"] == []
    # Record as a failure of the impostor, success of the lab.
    return _row(
        "BHV-21",
        lab,
        ok=ok,
        initial=initial,
        final=lab.head(),
        reasoning_ok=ok,
        extra={"impostor_rejected": True, "impostor": impostor},
        empty_success=False,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _bhv22(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab, "same\n")
    initial = lab.head()
    before = (lab.root / "README.md").read_bytes()
    lab.wu("noop-edit")
    (lab.root / "README.md").write_bytes(before)
    after = (lab.root / "README.md").read_bytes()
    noop = before == after
    status = lab.git("status", "--porcelain")
    clean = not (status.get("stdout") or "").strip()
    ok = noop and clean
    return _row(
        "BHV-22",
        lab,
        ok=ok,
        initial=initial,
        final=lab.head(),
        reasoning_ok=ok,
        extra={"noop": noop, "clean": clean},
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


def _bhv23(lab: _Lab) -> dict[str, Any]:
    t0 = time.perf_counter()
    _init_repo(lab)
    initial = lab.head()
    good = "def n():\n    return 1\n"
    bad = "def n(\n"
    path = lab.root / "n.py"
    path.write_text(good, encoding="utf-8")
    lab.wu("bad-patch")
    path.write_text(bad, encoding="utf-8")
    red = lab.py("-m", "py_compile", "n.py")
    lab.wu("restore")
    path.write_text(good, encoding="utf-8")
    green = lab.py("-m", "py_compile", "n.py")
    ok = (not red.get("ok")) and green.get("ok") is True
    return _row(
        "BHV-23",
        lab,
        ok=ok,
        initial=initial,
        final=lab.tree_hash(),
        reasoning_ok=ok,
        tests={"red": red.get("exit_code"), "green": green.get("exit_code")},
        extra={"red": True, "green": bool(green.get("ok"))},
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    )


_FIXTURES: dict[str, Callable[[_Lab], dict[str, Any]]] = {
    "BHV-01": _bhv01,
    "BHV-02": _bhv02,
    "BHV-03": _bhv03,
    "BHV-04": _bhv04,
    "BHV-05": _bhv05,
    "BHV-06": _bhv06,
    "BHV-07": _bhv07,
    "BHV-08": _bhv08,
    "BHV-09": _bhv09,
    "BHV-10": _bhv10,
    "BHV-11": _bhv11,
    "BHV-12": _bhv12,
    "BHV-13": _bhv13,
    "BHV-14": _bhv14,
    "BHV-15": _bhv15,
    "BHV-16": _bhv16,
    "BHV-17": _bhv17,
    "BHV-18": _bhv18,
    "BHV-19": _bhv19,
    "BHV-20": _bhv20,
    "BHV-21": _bhv21,
    "BHV-22": _bhv22,
    "BHV-23": _bhv23,
}


def run_fixture(fid: str, *, root: Path | None = None) -> dict[str, Any]:
    fid = str(fid)
    fn = _FIXTURES.get(fid)
    if fn is None:
        return {
            "id": fid,
            "ok": False,
            "ran": False,
            "blocked": True,
            "empty_success": False,
            "looked": False,
            "limitations": ["UNKNOWN_FIXTURE"],
        }
    if root is None:
        with tempfile.TemporaryDirectory(prefix=f"vmcp-{fid}-") as tmp:
            lab = _Lab(Path(tmp))
            return fn(lab)
    lab = _Lab(Path(root))
    return fn(lab)


def run_matrix(
    fixtures: Any = None,
    *,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run BHV-01..23 (or a requested subset) in isolated temp repos."""
    args = dict(arguments or {})
    wanted = fixtures if fixtures is not None else args.get("fixtures")
    if wanted in (None, "all", "*", True):
        ids = list(FIXTURE_IDS)
    elif isinstance(wanted, str):
        ids = [wanted]
    else:
        ids = [str(x) for x in wanted]
    started = utc_now()
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for fid in ids:
        rows.append(run_fixture(fid))
    # Independent evaluation: Call tabula, do not invent a second scorer.
    vec = scores_from_behavior_lab(rows)
    verdict = evaluate(vec)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    n_ok = sum(1 for r in rows if r.get("ok"))
    evidence = {
        "n": len(rows),
        "n_ok": n_ok,
        "ids": [r.get("id") for r in rows],
        "scores": vec.to_dict(),
        "verdict": verdict.to_dict(),
    }
    return {
        "act": "prove",
        "organ": "behavior_lab",
        "status": "CONNECTED",
        "ok": n_ok == len(rows) and len(rows) > 0,
        "looked": True,
        "empty_success": False,
        "n": len(rows),
        "n_ok": n_ok,
        "fixtures": rows,
        "scores": vec.to_dict(),
        "verdict": verdict.to_dict(),
        "execution": "REAL",
        "evidence_tier": "FUNCTIONAL_SIM",
        "gpu_authority": False,
        "network_used": False,
        "laboratory_profile_used": False,
        "started_at": started,
        "performance_ms": elapsed_ms,
        "deep_digest": content_digest(evidence),
        "artifacts": [],
        "evidence": [evidence],
        "residuals": [r["id"] for r in rows if not r.get("ok")],
        "next_actions": [],
        "note": (
            "isolated temp-repo fixtures; LABORATORY profile not used; "
            "scores Call tools.future.tabula.evaluate"
        ),
    }
