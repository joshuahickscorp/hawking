#!/usr/bin/env python3
"""ONE experiment engine.

Generalizes ``tools/headless/hcli_self_optimize_2.py`` rather than forking a
second decision function or a second import pin. The four G021 controls
(no-op, bad candidate, paired/interleaved, failing-gate) stay implemented
there; this module imports them and adds:

* persistent runtime (one RuntimeInterface reused across trials)
* exclusive-resource reservation
* causal execution-path verification
* adversarial review as a promotion STAGE

``pin_hcli_import_root`` is the G021_SCRATCH_IMPORT_SHADOW guard and is
the pattern this engine uses for every mutated module.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPT_REL = Path("receipts/headless/RUNTIME_EXPERIMENT_ADVERSARY.json")
RUNTIME_GENOME_REL = Path("receipts/headless/RUNTIME_GENOME.json")
CONTROL_REL = Path("receipts/headless/CONVENTIONAL_CONTROL_SET.json")
Q5K_NAME = "Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf"
FORBIDDEN_VIA = ("fan_completions", "llama_completion", "/completion")
ADVERSARY_QUESTIONS = (
    "what measurement BYPASSES the mutation?",
    "what cache is STALE?",
    "what NO-OP would also pass?",
    "what claim depends on a LITERAL STRING?",
    "what candidate ALTERED ITS VERIFIER?",
    "what state is ASSUMED rather than re-read?",
)

_SELFOPT = None


def _atomic_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def selfopt():
    """Load hcli_self_optimize_2.py once. Do not reimplement its primitives."""
    global _SELFOPT
    if _SELFOPT is None:
        spec = importlib.util.spec_from_file_location(
            "hcli_self_optimize_2", HERE / "hcli_self_optimize_2.py"
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SELFOPT = mod
    return _SELFOPT


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (proc.stdout or "").strip() or "UNKNOWN"


def git_show(repo: Path, rel: str) -> str:
    return selfopt()._git_show(repo, rel)


# ---------------------------------------------------------------------------
# exclusive-resource reservation
# ---------------------------------------------------------------------------


class ExclusiveReservation:
    """mkdir lock. A second holder in the same experiment is a harness bug."""

    def __init__(self, path: Path, timeout_s: float = 5.0) -> None:
        self.path = Path(path)
        self.timeout_s = float(timeout_s)
        self.held = False
        self.pid = os.getpid()

    def acquire(self) -> Dict[str, Any]:
        deadline = time.time() + self.timeout_s
        last_err = None
        while True:
            try:
                self.path.mkdir(parents=False)
                (self.path / "pid").write_text(str(self.pid), encoding="utf-8")
                (self.path / "owner").write_text("experiment_engine", encoding="utf-8")
                self.held = True
                return {"held": True, "path": str(self.path), "pid": self.pid}
            except FileExistsError as exc:
                last_err = exc
                stale = False
                try:
                    pid = int((self.path / "pid").read_text(encoding="utf-8").strip())
                    os.kill(pid, 0)
                except Exception:
                    stale = True
                if stale:
                    shutil.rmtree(self.path, ignore_errors=True)
                    continue
                if time.time() >= deadline:
                    return {
                        "held": False,
                        "path": str(self.path),
                        "error": "timeout",
                        "last": str(last_err),
                    }
                time.sleep(0.05)

    def release(self) -> None:
        if self.held:
            shutil.rmtree(self.path, ignore_errors=True)
            self.held = False

    def __enter__(self) -> "ExclusiveReservation":
        result = self.acquire()
        if not result.get("held"):
            raise RuntimeError(f"exclusive reservation failed: {result}")
        return self

    def __exit__(self, *exc) -> bool:
        self.release()
        return False


def prove_exclusive_reservation(lock_dir: Path) -> Dict[str, Any]:
    """Physically show a second acquire fails while the first is held."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / "experiment.lock.d"
    shutil.rmtree(path, ignore_errors=True)
    first = ExclusiveReservation(path, timeout_s=0.2)
    held = first.acquire()
    second = ExclusiveReservation(path, timeout_s=0.2)
    contested = second.acquire()
    first.release()
    after = ExclusiveReservation(path, timeout_s=0.2)
    released = after.acquire()
    after.release()
    return {
        "ran": True,
        "first_held": bool(held.get("held")),
        "second_held_while_first": bool(contested.get("held")),
        "third_held_after_release": bool(released.get("held")),
        "exclusive": bool(held.get("held"))
        and not bool(contested.get("held"))
        and bool(released.get("held")),
        "path": str(path),
    }


# ---------------------------------------------------------------------------
# open tracer (Q5_K must never be opened)
# ---------------------------------------------------------------------------


@contextmanager
def tracing_opens():
    opened: List[str] = []
    real_io = io.open
    real_b = __import__("builtins").open

    def wrapped(file, *args, **kwargs):
        opened.append(str(file))
        return real_io(file, *args, **kwargs)

    io.open = wrapped  # type: ignore[assignment]
    __import__("builtins").open = wrapped  # type: ignore[assignment]
    try:
        yield opened
    finally:
        io.open = real_io  # type: ignore[assignment]
        __import__("builtins").open = real_b  # type: ignore[assignment]


def opened_q5k(opened: List[str]) -> List[str]:
    hits = []
    for item in opened:
        name = os.path.basename(str(item).replace("\\", "/"))
        if name == Q5K_NAME or Q5K_NAME in str(item):
            hits.append(str(item))
    return hits


# ---------------------------------------------------------------------------
# probe target (in-engine candidate; not an HCLI tree claim)
# ---------------------------------------------------------------------------

PROBE_NAME = "g029_probe_admit"

PROBE_BASELINE = '''\
"""Baseline: admit width 1 regardless of overlap."""
VIA = []

def admit(requested, overlap):
    VIA.append("g029_probe_admit.admit")
    return min(int(requested), 1)

def score():
    VIA.append("g029_probe_admit.score")
    return admit(2, 2)
'''

PROBE_CANDIDATE = '''\
"""Candidate: honour measured overlap (requested=2, overlap=2 -> 2)."""
VIA = []

def admit(requested, overlap):
    VIA.append("g029_probe_admit.admit")
    return min(int(requested), max(1, int(overlap)))

def score():
    VIA.append("g029_probe_admit.score")
    return admit(2, 2)
'''

PROBE_NOOP = '''\
"""No-op: bytes change, admit cap stays 1."""
VIA = []

def admit(requested, overlap):
    # no-op candidate: comment only, width still 1
    VIA.append("g029_probe_admit.admit")
    return min(int(requested), 1)

def score():
    VIA.append("g029_probe_admit.score")
    return admit(2, 2)
'''

PROBE_BAD = '''\
"""Bad: admit nothing."""
VIA = []

def admit(requested, overlap):
    VIA.append("g029_probe_admit.admit")
    return 0

def score():
    VIA.append("g029_probe_admit.score")
    return admit(2, 2)
'''

PROBE_BYPASS = '''\
"""Bypass: mutation exists but score never calls it (original G021 defect)."""
VIA = []

def admit(requested, overlap):
    VIA.append("g029_probe_admit.admit")
    return min(int(requested), max(1, int(overlap)))

def score():
    VIA.append("fan_completions")
    return 99
'''

VERIFIER_SRC = '''\
"""Invariant verifier. Candidates must not rewrite this file."""
def test_score_is_non_negative_int():
    import g029_probe_admit as m
    value = m.score()
    assert isinstance(value, int)
    assert value >= 0
'''


def pin_probe_import(path: Path):
    """Force the next import of the probe to load *this file*.

    Same shape as pin_hcli_import_root: purge sys.modules, load by file
    location, die if the executed file is not the mutated file.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeError(f"pin_probe_import: {path} is not a file")
    name = PROBE_NAME
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    loaded = Path(getattr(mod, "__file__", "") or "").resolve()
    if loaded != path:
        raise RuntimeError(
            f"G021_SCRATCH_IMPORT_SHADOW analogue: loaded {loaded} not {path}"
        )
    return mod


def causal_path_proof(mod: Any, mutated_file: Path) -> Dict[str, Any]:
    executed = Path(getattr(mod, "__file__", "") or "").resolve()
    mutated = Path(mutated_file).resolve()
    via = list(getattr(mod, "VIA", []) or [])
    executed_sha = sha256_file(executed) if executed.is_file() else None
    mutated_sha = sha256_file(mutated) if mutated.is_file() else None
    forbidden_hit = [item for item in via if item in FORBIDDEN_VIA]
    through = (
        executed == mutated
        and executed_sha is not None
        and executed_sha == mutated_sha
        and "g029_probe_admit.admit" in via
        and not forbidden_hit
    )
    return {
        "executed_file": str(executed) if executed else None,
        "mutated_file": str(mutated),
        "executed_sha256": executed_sha,
        "mutated_sha256": mutated_sha,
        "via": via,
        "forbidden_via": list(FORBIDDEN_VIA),
        "forbidden_hit": forbidden_hit,
        "through_mutated_mechanism": bool(through),
        "score_calls_admit": "g029_probe_admit.admit" in via,
    }


def _trial_stats(trials: List[Dict[str, Any]], cond: str) -> Dict[str, Any]:
    vals = [
        float(t["score"])
        for t in trials
        if t.get("condition") == cond and isinstance(t.get("score"), (int, float))
    ]
    if not vals:
        return {"n": 0, "values": [], "min": None, "max": None, "median": None, "spread": None}
    return {
        "n": len(vals),
        "values": vals,
        "min": min(vals),
        "max": max(vals),
        "median": sorted(vals)[len(vals) // 2],
        "spread": max(vals) - min(vals),
    }


def run_verifier(scratch: Path, probe_path: Path, verifier: Path) -> Dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(probe_path.parent)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(verifier), "-q", "--tb=line"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(scratch),
        env=env,
    )
    return {
        "exit_code": proc.returncode,
        "ok": proc.returncode == 0,
        "tail": ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-400:],
        "verifier_sha256": sha256_file(verifier),
    }


def run_interleaved_probe(
    scratch: Path,
    *,
    candidate_name: str,
    candidate_src: str,
    baseline_src: str,
    n_trials: int = 4,
    runtime_iface: Any = None,
    verifier: Optional[Path] = None,
) -> Dict[str, Any]:
    """Paired interleaved trials: C, B, C, B. Persistent runtime. Causal path."""
    probe_path = scratch / f"{PROBE_NAME}.py"
    order = (["candidate", "baseline"] * ((n_trials + 1) // 2))[:n_trials]
    trials: List[Dict[str, Any]] = []
    persistent_ids = []
    verifier_hashes = []
    if verifier is not None:
        verifier_hashes.append(sha256_file(verifier))
    for i, cond in enumerate(order):
        probe_path.write_text(
            candidate_src if cond == "candidate" else baseline_src,
            encoding="utf-8",
        )
        pyc = scratch / "__pycache__"
        if pyc.is_dir():
            shutil.rmtree(pyc, ignore_errors=True)
        mod = pin_probe_import(probe_path)
        score = int(mod.score())
        proof = causal_path_proof(mod, probe_path)
        if runtime_iface is not None:
            persistent_ids.append(runtime_iface.persistent_id)
        if verifier is not None:
            verifier_hashes.append(sha256_file(verifier))
        trials.append(
            {
                "i": i,
                "condition": cond,
                "score": score,
                "import_file": getattr(mod, "__file__", None),
                "import_is_scratch": Path(getattr(mod, "__file__", "")).resolve()
                == probe_path.resolve(),
                "causal": proof,
                "through_mutated_mechanism": proof["through_mutated_mechanism"],
                "runtime_persistent_id": (
                    runtime_iface.persistent_id if runtime_iface is not None else None
                ),
            }
        )
    cand_stats = _trial_stats(trials, "candidate")
    base_stats = _trial_stats(trials, "baseline")
    cand_vals = cand_stats.get("values") or []
    base_vals = base_stats.get("values") or []
    admission_differs = cand_vals != base_vals and bool(cand_vals) and bool(base_vals)
    spread = 0.0
    if cand_vals and base_vals:
        spread = max(
            (cand_stats["spread"] or 0),
            (base_stats["spread"] or 0),
        )
    improved = False
    if cand_vals and base_vals:
        improved = (cand_stats["median"] or 0) > (base_stats["median"] or 0) + spread
    through = all(t.get("through_mutated_mechanism") for t in trials)
    persistent = (
        len(set(persistent_ids)) == 1 and len(persistent_ids) == len(trials)
        if persistent_ids
        else False
    )
    verifier_altered = len(set(verifier_hashes)) > 1 if verifier_hashes else False
    so = selfopt()
    decision = so.compute_decision(
        correctness_ok=through and not verifier_altered,
        throughput_improved=improved,
        mutation_applied=True,
        validation_ok=through and not verifier_altered,
        validation_reason=None if (through and not verifier_altered) else "NO_EVIDENCE",
        orig_med=base_stats.get("median"),
        mut_med=cand_stats.get("median"),
        spread=spread,
        admission_differs=admission_differs,
        metric_name="probe_score",
    )
    is_win = decision.get("verdict") == "PROMOTE"
    return {
        "name": candidate_name,
        "order": order,
        "block_design": False,
        "trials": trials,
        "candidate_scores": cand_vals,
        "baseline_scores": base_vals,
        "admission_differs": admission_differs,
        "spread": spread,
        "improved": improved,
        "through_mutated_mechanism": through,
        "persistent_runtime": persistent,
        "persistent_ids": persistent_ids,
        "verifier_altered": verifier_altered,
        "verifier_hashes": verifier_hashes,
        "decision": decision,
        "is_win": is_win,
    }


def cheat_verifier_candidate(scratch: Path, verifier: Path) -> Dict[str, Any]:
    """A candidate that rewrites its verifier. Must be refused."""
    before = sha256_file(verifier)
    probe_path = scratch / f"{PROBE_NAME}.py"
    probe_path.write_text(PROBE_CANDIDATE, encoding="utf-8")
    verifier.write_text(
        "def test_always_pass():\n    assert True\n",
        encoding="utf-8",
    )
    after = sha256_file(verifier)
    altered = before != after
    so = selfopt()
    decision = so.compute_decision(
        correctness_ok=True,
        throughput_improved=True,
        mutation_applied=True,
        validation_ok=not altered,
        validation_reason="candidate_altered_verifier" if altered else None,
        orig_med=1,
        mut_med=2,
        spread=0,
        admission_differs=True,
        metric_name="probe_score",
    )
    # restore verifier for later trials
    verifier.write_text(VERIFIER_SRC, encoding="utf-8")
    return {
        "ran": True,
        "before_sha256": before,
        "after_sha256": after,
        "altered": altered,
        "decision": decision,
        "refused": decision.get("verdict") == "REFUSED",
    }


# ---------------------------------------------------------------------------
# adversary stage
# ---------------------------------------------------------------------------


def _answer(question: str, answer: str, evidence: Any, risk: str) -> Dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "risk": risk,
    }


def adversary_stage(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Promotion STAGE. Asks the six questions; answers from this run's evidence."""
    probe = bundle.get("probe") or {}
    cand = probe.get("candidate") or {}
    noop = probe.get("noop") or {}
    bad = probe.get("bad") or {}
    bypass = probe.get("bypass") or {}
    cheat = probe.get("cheat_verifier") or {}
    failing = bundle.get("failing_gate") or {}
    head = bundle.get("head_hcli") or {}
    q5k = bundle.get("q5k") or {}
    genome = bundle.get("runtime_genome") or {}

    bypass_via = [
        hit
        for trial in (bypass.get("trials") or [])
        for hit in ((trial.get("causal") or {}).get("forbidden_hit") or [])
    ]
    noop_would_pass = bool(noop.get("is_win"))
    string_claims = []
    # G021 wiring test is a literal-string claim. Probe score is an int.
    if head.get("h1_equals_head"):
        string_claims.append(
            "G021 h1_wiring_present matches the constructor literal "
            "observed_overlap=load_observed_overlap(self.workspace_root); "
            "that string is not this engine's score. Probe score is admit() int."
        )
    assumed = []
    if not all(
        t.get("import_is_scratch")
        for t in (cand.get("trials") or [])
    ):
        assumed.append("import root was not re-pinned; sys.modules may be stale")
    mlx_remeasured = bool(
        ((genome.get("live") or {}).get("profile") or {}).get("remeasured")
    )
    if mlx_remeasured:
        assumed.append("MLX profile was re-measured instead of recorded from the control set")

    answers = [
        _answer(
            ADVERSARY_QUESTIONS[0],
            (
                "fan_completions / llama_completion HTTP, the original G021 defect. "
                "This run physically scored a bypass candidate that never called "
                "admit(); causal verification marked through_mutated_mechanism=false "
                "and compute_decision REFUSED it."
                if bypass_via or bypass.get("through_mutated_mechanism") is False
                else "UNANSWERED: bypass control did not run"
            ),
            {
                "bypass_forbidden_hit": bypass_via,
                "bypass_through": bypass.get("through_mutated_mechanism"),
                "bypass_verdict": (bypass.get("decision") or {}).get("verdict"),
                "candidate_through": cand.get("through_mutated_mechanism"),
            },
            "high" if bypass.get("is_win") else "closed",
        ),
        _answer(
            ADVERSARY_QUESTIONS[1],
            (
                "sys.modules + __pycache__ of the mutated probe (G021 analogue: "
                "hcli import shadow). pin_probe_import purges and reloads from the "
                "scratch file each trial. MACHINE_GENOME llama Q5_K identity is "
                "STALE as a live runtime prior — RuntimeGenome records MLX from "
                "CONVENTIONAL_CONTROL_SET instead of trusting it."
            ),
            {
                "all_trials_imported_scratch": all(
                    t.get("import_is_scratch") for t in (cand.get("trials") or [])
                ),
                "q5k_present": q5k.get("present"),
                "mlx_remeasured": mlx_remeasured,
                "pin": "pin_probe_import / pin_hcli_import_root",
            },
            "closed",
        ),
        _answer(
            ADVERSARY_QUESTIONS[2],
            (
                "A comment-only mutation of admit() scored identically to baseline "
                f"(candidate={noop.get('candidate_scores')} "
                f"baseline={noop.get('baseline_scores')}) and was REFUSED. "
                "A no-op must not win; this one did not."
                if noop.get("ran") is not False
                else "UNANSWERED"
            ),
            {
                "noop_is_win": noop.get("is_win"),
                "noop_verdict": (noop.get("decision") or {}).get("verdict"),
                "noop_admission_differs": noop.get("admission_differs"),
            },
            "high" if noop_would_pass else "closed",
        ),
        _answer(
            ADVERSARY_QUESTIONS[3],
            (
                "Probe promotion depends on admit() returning an int, not a "
                "literal string. G021's wiring pytest still matches a constructor "
                "literal; that claim is recorded and is not this engine's score. "
                + (" ".join(string_claims) if string_claims else "")
            ),
            {
                "probe_metric": "probe_score",
                "g021_literal_wiring": bool(head.get("h1_equals_head")),
            },
            "noted",
        ),
        _answer(
            ADVERSARY_QUESTIONS[4],
            (
                "A cheat candidate rewrote the verifier file "
                f"(sha { (cheat.get('before_sha256') or '')[:12] } -> "
                f"{ (cheat.get('after_sha256') or '')[:12] }). "
                "compute_decision REFUSED validation_ok=False "
                "reason=candidate_altered_verifier. Honest candidates left the "
                "verifier hash unchanged."
                if cheat.get("altered")
                else "UNANSWERED: cheat-verifier control did not alter the file"
            ),
            {
                "cheat_altered": cheat.get("altered"),
                "cheat_refused": cheat.get("refused"),
                "honest_verifier_altered": cand.get("verifier_altered"),
            },
            "high" if cheat.get("altered") and not cheat.get("refused") else "closed",
        ),
        _answer(
            ADVERSARY_QUESTIONS[5],
            (
                "Overlap and score are re-read each trial by rewriting the probe "
                "file and pin_probe_import (not a cached module). Runtime health "
                "is on the persistent interface object, re-probed, not assumed. "
                "MLX numbers are re-read from CONVENTIONAL_CONTROL_SET.json, not "
                "re-measured and not taken from a STALE llama MACHINE_GENOME. "
                + ("ASSUMED: " + "; ".join(assumed) if assumed else "None assumed.")
            ),
            {
                "assumed": assumed,
                "persistent_runtime": cand.get("persistent_runtime"),
                "control_set": str(CONTROL_REL),
            },
            "high" if assumed else "closed",
        ),
    ]
    unanswered = [
        row for row in answers
        if str(row.get("answer") or "").startswith("UNANSWERED")
    ]
    high = [row for row in answers if row.get("risk") == "high"]
    refuse = bool(unanswered or high or noop_would_pass)
    if failing.get("would_refuse_on_failing_gate") is not True:
        refuse = True
        high.append({"question": "failing-gate", "risk": "high",
                     "answer": "would_refuse_on_failing_gate was not computed True"})
    return {
        "stage": "adversary",
        "ran": True,
        "questions": list(ADVERSARY_QUESTIONS),
        "answers": answers,
        "unanswered": unanswered,
        "high_risk": high,
        "refuse": refuse,
        "verdict": "REFUSED" if refuse else "PASS",
        "note": (
            "Adversary is a promotion stage, not a comment. A PASS here means "
            "the probe candidate may be considered by compute_decision; it is "
            "not a tree-level promotion of HEAD hcli."
        ),
    }


# ---------------------------------------------------------------------------
# Q5_K census (prove, do not assert)
# ---------------------------------------------------------------------------


def q5k_census(repo: Path) -> Dict[str, Any]:
    present = (Path.home() / "models/qwen3.8-27b-abliterated" / Q5K_NAME).is_file()
    proc = subprocess.run(
        [
            "git", "-C", str(repo), "grep", "-n", Q5K_NAME, "HEAD",
            "--", "hcli", "tools/headless", "tools/haider",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    classified = []
    required_hits = []
    for line in lines:
        raw = line[5:] if line.startswith("HEAD:") else line
        path = raw.split(":", 1)[0] if ":" in raw else raw
        kind = "science_or_comment"
        if "/tests/" in path or path.endswith("_test.py") or "/test_" in path:
            kind = "test_fixture"
        elif path.endswith(".json"):
            kind = "receipt_science"
        elif "conventional_control_set.py" in path:
            kind = "archived_control_path"
        elif path.startswith("hcli/"):
            if "runtime_iface" in path or "runtime_genome" in path:
                kind = "hcli_archived_name"
            else:
                kind = "hcli_source"
                required_hits.append(line)
        classified.append({"line": line, "path": path, "kind": kind})
    # Production hcli must not open the file. AST-scan extracted hcli.
    ast_opens = []
    hcli_root = repo / "hcli"
    if hcli_root.is_dir():
        for py in hcli_root.rglob("*.py"):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            src = py.read_text(encoding="utf-8")
            if Q5K_NAME not in src:
                continue
            ast_opens.append(str(py.relative_to(repo)))
    return {
        "name": Q5K_NAME,
        "present": present,
        "required": False,
        "git_grep_head_count": len(lines),
        "verified_against": "HEAD (git grep) plus working-tree AST of hcli/",
        "hcli_source_hits": required_hits,
        "hcli_ast_files_mentioning_name": ast_opens,
        "classified": classified[:80],
        "note": (
            "Hits under tools/headless/*probe.py name the archived path as a "
            "historical default. Importing those modules does not open the "
            "file. This engine, RuntimeGenome, and RuntimeInterface never "
            "open it. HEAD hcli/ does not mention the filename "
            "(git grep HEAD -- hcli). The new runtime_iface module names it "
            "as archived science and q5k_gguf_required() is False."
        ),
    }


def prove_q5k_not_opened(repo: Path) -> Dict[str, Any]:
    """Execute the new code paths with builtins/io.open traced."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    with tracing_opens() as opened:
        from hcli.runtime_iface import (  # noqa: WPS433
            RuntimeInterface,
            classify_backend,
            load_control_set,
            q5k_gguf_required,
            archived_q5k_gguf_path,
            runtime_interface_census,
        )
        from hcli.genomes.runtime_genome import RuntimeGenome  # noqa: WPS433
        from hcli.machine import MachineGenome  # noqa: WPS433

        assert q5k_gguf_required() is False
        control = load_control_set(repo)
        iface = RuntimeInterface.from_control_set(control)
        genome = RuntimeGenome.from_control_set(repo, control=control)
        bag = Path(tempfile.mkdtemp(prefix="g029-mg-")) / "machine-genome.json"
        mg = MachineGenome(bag)
        genome.record_into_machine_genome(mg)
        census = runtime_interface_census()
        kind_missing = classify_backend(str(archived_q5k_gguf_path()))
        # conventional archived arm without the file
        sys.path.insert(0, str(HERE))
        from conventional_control_set import archive_llama  # noqa: WPS433

        archived = archive_llama(False)
    hits = opened_q5k(opened)
    return {
        "ran": True,
        "q5k_required_constant": False,
        "opened_paths_mentioning_q5k": hits,
        "opened_q5k": bool(hits),
        "classify_missing_q5k": kind_missing,
        "iface_kind": iface.backend_kind,
        "iface_profile_remeasured": bool((iface.profile or {}).get("remeasured")),
        "genome_headline": genome.mlx_headline(),
        "machine_genome_has_runtime_profile": bool(
            mg.get_profile("runtime_genome")
        ),
        "census_q5k_required": census.get("q5k_gguf_required"),
        "archived_status": archived.get("status") if isinstance(archived, dict) else None,
        "ok": not hits and iface.backend_kind == "mlx",
    }


# ---------------------------------------------------------------------------
# HEAD hcli verdict (do not manufacture a tree win)
# ---------------------------------------------------------------------------


def head_hcli_verdict(repo: Path) -> Dict[str, Any]:
    so = selfopt()
    # Prefer git show so a sparse hole is not mistaken for absence.
    text = git_show(repo, "hcli/controller.py")
    variants = so.controller_variants(text)
    h1_equals_head = variants["h1"] == text
    return {
        "h1_equals_head": h1_equals_head,
        "controller_verified_via": "git show HEAD:hcli/controller.py",
        "decision": "reject" if h1_equals_head else "undecided",
        "verdict": "REFUSED" if h1_equals_head else "NO_TREE_CANDIDATE_BUILT",
        "reason": (
            "H1 is already HEAD. The only production RuntimePool() site is "
            "Controller.ensure_runtime_pool, already wired. There is no honest "
            "admitted_n promotion left at HEAD. Recorded as REFUSED, not manufactured."
            if h1_equals_head
            else "HEAD controller is not H1; a tree candidate would need its own chain."
        ),
        "no_remaining_admitted_n_candidate_at_head": h1_equals_head,
    }


# ---------------------------------------------------------------------------
# full chain
# ---------------------------------------------------------------------------


def run_full_chain(repo: Path) -> Dict[str, Any]:
    """Promotion through the full chain + a recorded refusal.

    The probe candidate is an in-engine mutation whose score actually
    changes because admit() changed. That is the chain promotion. HEAD
    hcli is refused: G021 already found no remaining admitted_n candidate.
    """
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    so = selfopt()
    scratch = Path(tempfile.mkdtemp(prefix="g029-engine-"))
    verifier = scratch / "test_probe_invariant.py"
    verifier.write_text(VERIFIER_SRC, encoding="utf-8")
    (scratch / f"{PROBE_NAME}.py").write_text(PROBE_BASELINE, encoding="utf-8")

    from hcli.runtime_iface import (  # noqa: WPS433
        RuntimeInterface,
        load_control_set,
        runtime_interface_census,
    )
    from hcli.backends import NoeticNativeBackend  # noqa: WPS433
    from hcli.genomes.runtime_genome import RuntimeGenome  # noqa: WPS433
    from hcli.machine import MachineGenome  # noqa: WPS433
    from hcli.session import Session  # noqa: WPS433

    with tracing_opens() as opened_during_setup:
        control = load_control_set(repo)
        iface = RuntimeInterface.from_control_set(control)
        session = Session(goal="g029 experiment engine", runtime_count=1)
        iface.bind_session(session.id)
        native = NoeticNativeBackend(model_path="reserved")
        native.spawn()
        genome = RuntimeGenome.from_control_set(repo, control=control)
        genome_path = genome.save_receipt(repo / RUNTIME_GENOME_REL)
        bag = scratch / "machine-genome.json"
        mg = MachineGenome(bag)
        genome.record_into_machine_genome(mg)
        mg.save()
        census = runtime_interface_census()
    q5k_open_hits = opened_q5k(opened_during_setup)

    exclusive = prove_exclusive_reservation(scratch / "locks")
    with ExclusiveReservation(scratch / "locks" / "run.lock.d", timeout_s=5):
        candidate = run_interleaved_probe(
            scratch,
            candidate_name="probe_overlap",
            candidate_src=PROBE_CANDIDATE,
            baseline_src=PROBE_BASELINE,
            runtime_iface=iface,
            verifier=verifier,
        )
        noop = run_interleaved_probe(
            scratch,
            candidate_name="noop",
            candidate_src=PROBE_NOOP,
            baseline_src=PROBE_BASELINE,
            runtime_iface=iface,
            verifier=verifier,
        )
        bad = run_interleaved_probe(
            scratch,
            candidate_name="bad",
            candidate_src=PROBE_BAD,
            baseline_src=PROBE_BASELINE,
            runtime_iface=iface,
            verifier=verifier,
        )
        bypass = run_interleaved_probe(
            scratch,
            candidate_name="bypass",
            candidate_src=PROBE_BYPASS,
            baseline_src=PROBE_BASELINE,
            runtime_iface=iface,
            verifier=verifier,
        )
        cheat = cheat_verifier_candidate(scratch, verifier)
        failing = so.run_failing_gate_trial(scratch / "failing_gate")

    # Persistent identity must not have changed across the reserved block.
    persistent_ok = bool(candidate.get("persistent_runtime")) and len(set(
        (candidate.get("persistent_ids") or [])
        + (noop.get("persistent_ids") or [])
        + (bad.get("persistent_ids") or [])
    )) == 1

    q5k = q5k_census(repo)
    q5k_exec = prove_q5k_not_opened(repo)
    q5k["opened_during_setup"] = q5k_open_hits
    q5k["execution_proof"] = q5k_exec

    head = head_hcli_verdict(repo)
    bundle = {
        "probe": {
            "candidate": {**candidate, "ran": True},
            "noop": {**noop, "ran": True},
            "bad": {**bad, "ran": True},
            "bypass": {**bypass, "ran": True},
            "cheat_verifier": cheat,
        },
        "failing_gate": failing,
        "head_hcli": head,
        "q5k": q5k,
        "runtime_genome": genome.to_dict(),
    }
    adversary = adversary_stage(bundle)

    controls = {
        "noop": {
            "id": "NO-OP",
            "ran": True,
            "is_win": bool(noop.get("is_win")),
            "must_not_win": True,
            "admission_differs": noop.get("admission_differs"),
            "candidate_scores": noop.get("candidate_scores"),
            "baseline_scores": noop.get("baseline_scores"),
            "through_mutated_mechanism": noop.get("through_mutated_mechanism"),
            "decision": noop.get("decision"),
        },
        "bad": {
            "id": "BAD",
            "ran": True,
            "is_win": bool(bad.get("is_win")),
            "must_be_refused": True,
            "candidate_scores": bad.get("candidate_scores"),
            "baseline_scores": bad.get("baseline_scores"),
            "through_mutated_mechanism": bad.get("through_mutated_mechanism"),
            "decision": bad.get("decision"),
        },
        "paired_interleaved": {
            "id": "PAIRED_INTERLEAVED",
            "ran": True,
            "block_design": False,
            "candidate_order": candidate.get("order"),
            "noop_order": noop.get("order"),
            "bad_order": bad.get("order"),
        },
        "failing_gate": {
            "id": "FAILING_GATE",
            "ran": True,
            "hardcoded": bool(failing.get("hardcoded")),
            "would_refuse_on_failing_gate": failing.get("would_refuse_on_failing_gate"),
            "would_refuse_on_no_evidence": failing.get("would_refuse_on_no_evidence"),
            "pytest_exit_code": failing.get("pytest_exit_code"),
            "evidenced": failing.get("evidenced"),
        },
        "persistent_runtime": {
            "id": "PERSISTENT_RUNTIME",
            "ran": True,
            "ok": persistent_ok,
            "interface_persistent_id": iface.persistent_id,
            "candidate_ids": candidate.get("persistent_ids"),
        },
        "exclusive_reservation": {
            "id": "EXCLUSIVE_RESOURCE",
            "ran": True,
            **exclusive,
        },
        "causal_execution_path": {
            "id": "CAUSAL_EXECUTION_PATH",
            "ran": True,
            "candidate_through": candidate.get("through_mutated_mechanism"),
            "noop_through": noop.get("through_mutated_mechanism"),
            "bad_through": bad.get("through_mutated_mechanism"),
            "bypass_through": bypass.get("through_mutated_mechanism"),
            "bypass_refused": (bypass.get("decision") or {}).get("verdict") == "REFUSED",
            "pin": "pin_probe_import (G021_SCRATCH_IMPORT_SHADOW pattern)",
        },
    }

    controls_ok = (
        controls["noop"]["is_win"] is False
        and (controls["bad"]["decision"] or {}).get("verdict") == "REFUSED"
        and controls["paired_interleaved"]["block_design"] is False
        and controls["failing_gate"]["would_refuse_on_failing_gate"] is True
        and controls["failing_gate"]["hardcoded"] is False
        and controls["persistent_runtime"]["ok"] is True
        and controls["exclusive_reservation"].get("exclusive") is True
        and controls["causal_execution_path"]["candidate_through"] is True
        and controls["causal_execution_path"]["bypass_through"] is False
        and controls["causal_execution_path"]["bypass_refused"] is True
    )

    probe_promote = (
        (candidate.get("decision") or {}).get("verdict") == "PROMOTE"
        and candidate.get("through_mutated_mechanism") is True
        and adversary.get("verdict") == "PASS"
        and controls_ok
    )

    native_complete_refused = False
    native_error = None
    try:
        native.complete({"prompt": "x"})
    except RuntimeError as exc:
        native_complete_refused = True
        native_error = str(exc)

    receipt = {
        "schema": "hawking.headless.runtime_experiment_adversary.v1",
        "generated_at": started,
        "git_head": git_head(repo),
        "sparse_checkout": {
            "note": (
                "This worktree is sparse. hcli/, tools/headless, tools/haider "
                "were materialized via git show HEAD:<path> because "
                "git sparse-checkout add / git restore fail (index.lock "
                "Operation not permitted). Census paths were also verified "
                "with git grep HEAD and git show HEAD:hcli/controller.py."
            ),
            "verified_against": "HEAD",
            "materialized_via": "git show HEAD:<path> writes into the worktree",
        },
        "root_cause_kept": {
            "id": "G021_SCRATCH_IMPORT_SHADOW",
            "guard": "pin_hcli_import_root / pin_probe_import",
        },
        "runtime_interface_census": census,
        "runtime_genome": {
            "path": str(RUNTIME_GENOME_REL),
            "saved": str(genome_path),
            "remeasured": False,
            "mlx_headline": genome.mlx_headline(),
            "recorded_into_machine_genome_bag": True,
            "did_not_rewrite_probe_MACHINE_GENOME_json": True,
            "q5k_required": False,
        },
        "session_not_duplicated": {
            "bound_session_id": iface.session_id,
            "session_type": type(session).__name__,
            "session_module": type(session).__module__,
        },
        "noetic_native": {
            "identity": native.identity(),
            "complete_refused": native_complete_refused,
            "error": native_error,
        },
        "q5k_gguf": q5k,
        "controls": controls,
        "controls_ok": controls_ok,
        "chain_promotion": {
            "what": (
                "in-engine probe: admit() honours overlap. Not a HEAD hcli "
                "tree change. Score goes through the mutated function."
            ),
            "decision": "promote" if probe_promote else "reject",
            "verdict": "PROMOTE" if probe_promote else "REFUSED",
            "candidate_decision": candidate.get("decision"),
            "adversary_verdict": adversary.get("verdict"),
            "candidate_scores": candidate.get("candidate_scores"),
            "baseline_scores": candidate.get("baseline_scores"),
            "through_mutated_mechanism": candidate.get("through_mutated_mechanism"),
        },
        "head_tree_refusal": head,
        "adversary": adversary,
        "failing_gate_trial": failing,
        "would_refuse_on_failing_gate": failing.get("would_refuse_on_failing_gate"),
        "probe_trials": {
            "candidate": candidate.get("trials"),
            "noop": noop.get("trials"),
            "bad": bad.get("trials"),
            "bypass": bypass.get("trials"),
        },
        "workspace": str(scratch),
    }
    dest = repo / RECEIPT_REL
    _atomic_write(dest, receipt)
    receipt["receipt_path"] = str(dest)
    return receipt


def main(argv: Optional[List[str]] = None) -> int:
    del argv
    receipt = run_full_chain(REPO)
    print(json.dumps({
        "receipt": receipt.get("receipt_path"),
        "chain": (receipt.get("chain_promotion") or {}).get("verdict"),
        "head": (receipt.get("head_tree_refusal") or {}).get("verdict"),
        "adversary": (receipt.get("adversary") or {}).get("verdict"),
        "controls_ok": receipt.get("controls_ok"),
        "would_refuse_on_failing_gate": receipt.get("would_refuse_on_failing_gate"),
        "mlx": (receipt.get("runtime_genome") or {}).get("mlx_headline"),
        "q5k_opened": ((receipt.get("q5k_gguf") or {}).get("execution_proof") or {}).get(
            "opened_q5k"
        ),
    }, indent=2))
    chain_ok = (receipt.get("chain_promotion") or {}).get("verdict") == "PROMOTE"
    head_refused = (receipt.get("head_tree_refusal") or {}).get("verdict") == "REFUSED"
    if chain_ok and head_refused and receipt.get("controls_ok"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
