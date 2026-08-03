"""Adjudicate a proof capsule against the Tier 3 requirements, and refuse otherwise.

`odyssey/verifiers/VERIFICATION_LATTICE.json` makes Tier 3 exact:

    lean_proof_compiles, no_sorry, no_undeclared_axioms, pinned_mathlib,
    clean_container_reproduction

This module is the thing that checks those five and, only if all five hold,
mints the `machine_check` VerifierEvent that `evidence.promote` will accept for
Tier 3. It is not a proof search. Nothing here writes Lean.

The design point is the refusal. Two of the five checks cannot be done without
actually running the pinned container, so when the container is unavailable
this returns UNAVAILABLE rather than passing the other three and calling it
proven. A prover that degrades to "I checked what I could" is precisely how a
claim ends up cited at a tier nobody granted it, and this repository has three
receipts that outlived the tool that justified them.

`machine_check_event` therefore has one job: it cannot construct a Tier 3
license unless every requirement is MET and a real container hash exists.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ramanujan.evidence import VerifierEvent

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "ramanujan/container/pins.json"
REPLAY = ROOT / "ramanujan/container/replay_capsule.sh"

TIER3_REQUIREMENTS = (
    "lean_proof_compiles",
    "no_sorry",
    "no_undeclared_axioms",
    "pinned_mathlib",
    "clean_container_reproduction",
)

# `sorry` and `admit` close a goal without proving it. Matched on word
# boundaries so `sorryAx` in a comment about sorries does not trip it, and
# checked against the proof text with comments stripped.
_HOLE = re.compile(r"\b(sorry|admit)\b")
_AXIOM = re.compile(r"^\s*axiom\s+(\w+)", re.M)
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/-.*?-/", re.S)


class Status(str, Enum):
    MET = "MET"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"  # cannot be decided here; never counts as MET


@dataclass(frozen=True)
class Finding:
    requirement: str
    status: Status
    detail: str


def _strip_comments(src: str) -> str:
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", src))


def check_no_sorry(capsule: dict[str, Any]) -> Finding:
    src = _strip_comments(capsule.get("proof_lean", ""))
    hit = _HOLE.search(src)
    if hit:
        return Finding("no_sorry", Status.FAILED, f"proof contains {hit.group(1)!r}, which closes a goal without proving it")
    return Finding("no_sorry", Status.MET, "no sorry or admit in the proof text")


def check_no_undeclared_axioms(capsule: dict[str, Any]) -> Finding:
    src = _strip_comments(capsule.get("proof_lean", ""))
    declared = _AXIOM.findall(src)
    if declared:
        return Finding("no_undeclared_axioms", Status.FAILED,
                       f"proof introduces axiom(s) {declared}; an axiom is an assumption, not a proof")
    return Finding("no_undeclared_axioms", Status.MET, "proof declares no axioms of its own")


def check_pinned_mathlib(capsule: dict[str, Any]) -> Finding:
    """The capsule's pins must equal the environment lock's, exactly."""
    if not PINS.is_file():
        return Finding("pinned_mathlib", Status.UNAVAILABLE, f"{PINS.relative_to(ROOT)} is absent, so nothing is pinned")
    lock = json.loads(PINS.read_text())
    want = capsule.get("pins") or {}
    if not want:
        return Finding("pinned_mathlib", Status.FAILED, "capsule declares no pins")
    got = {
        "mathlib_commit": (lock.get("mathlib") or {}).get("commit"),
        "lean_toolchain": (lock.get("lean_for_mathlib_checks") or {}).get("toolchain"),
        "lean_commit": (lock.get("lean_for_mathlib_checks") or {}).get("commit"),
    }
    diff = {k: (v, got.get(k)) for k, v in want.items() if k in got and got.get(k) != v}
    if diff:
        return Finding("pinned_mathlib", Status.FAILED,
                       "capsule pins disagree with the environment lock: "
                       + "; ".join(f"{k}: capsule {c!r} vs lock {l!r}" for k, (c, l) in diff.items()))
    return Finding("pinned_mathlib", Status.MET, f"pins match the lock on {sorted(want)}")


def _container_available() -> tuple[bool, str]:
    if not REPLAY.is_file():
        return False, f"{REPLAY.relative_to(ROOT)} is absent"
    if shutil.which("docker") is None:
        return False, "docker is not installed"
    probe = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        return False, "docker daemon is not reachable"
    return True, probe.stdout.strip()


def check_container_reproduction(
    capsule: dict[str, Any], *, run: bool = True, capsule_path: Path | None = None
) -> tuple[Finding, str | None]:
    """Returns the finding and, only on success, the container hash Tier 3 needs.

    `replay_capsule.sh` takes the capsule *path*, not its id.
    """
    ok, why = _container_available()
    if not ok:
        return Finding("clean_container_reproduction", Status.UNAVAILABLE,
                       f"cannot replay: {why}. Not a failure of the proof; the check did not run."), None
    if not run:
        return Finding("clean_container_reproduction", Status.UNAVAILABLE, "replay not requested"), None
    target = capsule_path or (ROOT / f"ramanujan/container/capsules/{capsule.get('id','')}.capsule.json")
    if not Path(target).is_file():
        return Finding("clean_container_reproduction", Status.UNAVAILABLE,
                       f"capsule file not found at {target}"), None
    proc = subprocess.run([str(REPLAY), str(target)], capture_output=True, text=True, cwd=ROOT, timeout=1800)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        msg = tail[-1][:200] if tail else f"replay exited {proc.returncode}"
        # An unbuilt image means the check did not run. Reporting that as FAILED
        # would say the proof is wrong, which is a different and much stronger
        # claim than "the environment is not ready".
        if "not present" in msg or "Build first" in msg:
            return Finding("clean_container_reproduction", Status.UNAVAILABLE,
                           f"{msg} -- environment not built; the proof was not judged"), None
        return Finding("clean_container_reproduction", Status.FAILED, msg), None
    digest = None
    for line in (proc.stdout or "").splitlines():
        if "sha256:" in line:
            digest = line.split("sha256:")[1].split()[0][:64]
            break
    if not digest:
        return Finding("clean_container_reproduction", Status.FAILED,
                       "replay succeeded but reported no image digest, so it is not reproducible"), None
    return Finding("clean_container_reproduction", Status.MET, f"replayed in sha256:{digest[:16]}"), digest


def check_compiles(capsule: dict[str, Any], container_finding: Finding) -> Finding:
    """Compilation is decided by the container run, not separately.

    Checking it on the host would be checking a different environment than the
    one the tier requires, which is worse than not checking it.
    """
    if container_finding.status is Status.MET:
        return Finding("lean_proof_compiles", Status.MET, "compiled inside the pinned container")
    return Finding("lean_proof_compiles", container_finding.status,
                   f"decided by the container run, which is {container_finding.status.value}")


def adjudicate(capsule: dict[str, Any], *, run_container: bool = True,
               capsule_path: Path | None = None) -> dict[str, Any]:
    """Check all five requirements. MET only when every one is MET."""
    cont, digest = check_container_reproduction(capsule, run=run_container, capsule_path=capsule_path)
    findings = [
        check_compiles(capsule, cont),
        check_no_sorry(capsule),
        check_no_undeclared_axioms(capsule),
        check_pinned_mathlib(capsule),
        cont,
    ]
    by_req = {f.requirement: f for f in findings}
    assert set(by_req) == set(TIER3_REQUIREMENTS), "adjudication must cover exactly the tier-3 set"
    met = all(f.status is Status.MET for f in findings)
    return {
        "capsule_id": capsule.get("id"),
        "tier3_met": met,
        "container_hash": digest if met else None,
        "findings": [{"requirement": f.requirement, "status": f.status.value, "detail": f.detail} for f in findings],
        "verdict": "PROVEN" if met else (
            "REFUSED_UNAVAILABLE" if any(f.status is Status.UNAVAILABLE for f in findings) else "REFUSED_FAILED"),
    }


def machine_check_event(capsule: dict[str, Any], actor: str, *, run_container: bool = True) -> VerifierEvent:
    """Mint the Tier 3 license, or raise. There is no partial-credit path."""
    result = adjudicate(capsule, run_container=run_container)
    if not result["tier3_met"]:
        unmet = [f"{f['requirement']}={f['status']}" for f in result["findings"] if f["status"] != "MET"]
        raise PermissionError(
            f"refusing to mint a machine_check event for {capsule.get('id')!r}: " + ", ".join(unmet)
        )
    return VerifierEvent(
        kind="machine_check",
        actor=actor,
        container_hash=result["container_hash"],
        independent_of_author=True,
        detail={"capsule_id": result["capsule_id"], "requirements": TIER3_REQUIREMENTS},
    )


def demo() -> None:
    path = ROOT / "ramanujan/container/capsules/two_plus_two.capsule.json"
    r = adjudicate(json.loads(path.read_text()), run_container=True, capsule_path=path)
    cap = json.loads(path.read_text())
    print(f"  {cap['id']}: {r['verdict']}")
    for f in r["findings"]:
        print(f"    {f['requirement']:<30} {f['status']:<12} {f['detail'][:76]}")
    assert not r["tier3_met"] or r["container_hash"], "MET without a container hash is impossible"


if __name__ == "__main__":
    demo()
