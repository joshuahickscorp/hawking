"""Lean verifier interface — fails closed when the formal toolchain is incomplete.

Rules (hard):
* Never invent a container hash.
* Never treat static text heuristics alone as Tier-3 PROVEN.
* If lean/lake/container/mathlib cannot actually check the claim, return
  UNAVAILABLE (or REJECTED only when a real run proved the proof wrong).

Host may have ``~/.elan/bin/lean`` without a pinned clean-container image.
That is GATED, not REAL machine-check capacity.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ramanujan.verifier.base import (
    BackendAvailability,
    BackendStatus,
    VerificationRequest,
    VerificationResult,
    Verdict,
)

BACKEND_ID = "lean"

_HOLE = re.compile(r"\b(sorry|admit)\b")
_AXIOM = re.compile(r"^\s*axiom\s+(\w+)", re.M)
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/-.*?-/", re.S)

_ELAN_BIN = Path.home() / ".elan" / "bin"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_PINS = _REPO_ROOT / "ramanujan" / "container" / "pins.json"
_REPLAY = _REPO_ROOT / "ramanujan" / "container" / "replay_capsule.sh"


def _which(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    cand = _ELAN_BIN / name
    if cand.is_file():
        return str(cand)
    return None


def _strip_comments(src: str) -> str:
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", src))


def static_refuse_holes(proof_lean: str) -> str | None:
    """Return a refusal reason if the proof text is immediately invalid."""
    src = _strip_comments(proof_lean or "")
    hit = _HOLE.search(src)
    if hit:
        return f"proof contains {hit.group(1)!r}, which closes a goal without proving it"
    axioms = _AXIOM.findall(src)
    if axioms:
        return f"proof introduces axiom(s) {axioms}; axioms are not proofs"
    if not src.strip():
        return "empty Lean proof"
    return None


class LeanBackend:
    """Lean/Mathlib machine-check surface with fail-closed host probing."""

    backend_id = BACKEND_ID

    def availability(self) -> BackendAvailability:
        lean = _which("lean")
        lake = _which("lake")
        docker = shutil.which("docker")
        pins_ok = _PINS.is_file()
        replay_ok = _REPLAY.is_file()

        if not lean:
            return BackendAvailability(
                backend_id=self.backend_id,
                status=BackendStatus.ABSENT,
                detail="lean binary not found on PATH or ~/.elan/bin — fail closed",
                capabilities=("static_hole_refuse",),
            )

        version = None
        try:
            proc = subprocess.run(
                [lean, "--version"], capture_output=True, text=True, timeout=60
            )
            version = (proc.stdout or proc.stderr or "").strip().splitlines()[0] or None
        except (OSError, subprocess.TimeoutExpired) as exc:
            return BackendAvailability(
                backend_id=self.backend_id,
                status=BackendStatus.GATED,
                detail=f"lean present but unusable: {exc}",
                version=None,
                capabilities=("static_hole_refuse",),
            )

        # Full Tier-3 machine check needs pinned container replay.
        container_ready = False
        container_detail = "docker/container replay not verified"
        if docker and replay_ok and pins_ok:
            try:
                info = subprocess.run(
                    ["docker", "info", "--format", "{{.ServerVersion}}"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if info.returncode == 0 and (info.stdout or "").strip():
                    container_ready = True
                    container_detail = f"docker {info.stdout.strip()}; replay script present"
            except (OSError, subprocess.TimeoutExpired):
                container_detail = "docker present but daemon probe failed"

        if container_ready and lake:
            return BackendAvailability(
                backend_id=self.backend_id,
                status=BackendStatus.GATED,
                detail=(
                    f"lean {version}; lake present; {container_detail}. "
                    "Tier-3 machine_check still requires a built replay image and "
                    "capsule path — interface is ready, full proof check is gated."
                ),
                version=version,
                capabilities=(
                    "static_hole_refuse",
                    "pinned_container_replay_interface",
                    "no_fake_proofs",
                ),
            )

        missing = []
        if not lake:
            missing.append("lake")
        if not pins_ok:
            missing.append("pins.json")
        if not replay_ok:
            missing.append("replay_capsule.sh")
        if not docker:
            missing.append("docker")
        return BackendAvailability(
            backend_id=self.backend_id,
            status=BackendStatus.GATED,
            detail=(
                f"lean host binary present ({version}); full formal check gated "
                f"(missing/unready: {', '.join(missing) or 'container image'}). "
                "Fail closed — will not claim PROVEN."
            ),
            version=version,
            capabilities=("static_hole_refuse", "no_fake_proofs"),
        )

    def supports(self, request: VerificationRequest) -> bool:
        return request.kind in {"lean", "lean_capsule", "lean_proof"}

    def verify(self, request: VerificationRequest) -> VerificationResult:
        """Check a Lean capsule / proof claim without faking machine-check success.

        Stages:
        1. Static refuse of sorry/admit/axiom (can REJECT).
        2. If a real container replay is not ready, return UNAVAILABLE rather than
           partial credit that looks like a proof.
        """
        proof = str(
            request.payload.get("proof_lean")
            or request.claimed_answer
            or ""
        )
        static = static_refuse_holes(proof)
        if static:
            return VerificationResult(
                backend_id=self.backend_id,
                verdict=Verdict.REJECTED,
                detail=static,
                evidence={"static_check": "failed"},
                outcome_kind="negative",
            )

        avail = self.availability()
        # Optional: only run container replay when explicitly requested AND ready.
        run_container = bool(request.payload.get("run_container"))
        capsule_path = request.payload.get("capsule_path")
        if run_container and capsule_path and _REPLAY.is_file() and shutil.which("docker"):
            return self._replay_capsule(Path(str(capsule_path)))

        return VerificationResult(
            backend_id=self.backend_id,
            verdict=Verdict.UNAVAILABLE,
            detail=(
                "Lean static checks passed (no sorry/admit/axiom), but Tier-3 "
                f"machine-check is not available on this host: {avail.detail}. "
                "Refusing to accept as verified — no faked proofs."
            ),
            evidence={
                "availability": avail.as_dict(),
                "static_check": "passed",
                "machine_check": "not_run",
            },
            outcome_kind="unavailable",
        )

    def _replay_capsule(self, capsule_path: Path) -> VerificationResult:
        if not capsule_path.is_file():
            return VerificationResult(
                backend_id=self.backend_id,
                verdict=Verdict.UNAVAILABLE,
                detail=f"capsule not found: {capsule_path}",
                evidence={"machine_check": "not_run"},
            )
        try:
            proc = subprocess.run(
                [str(_REPLAY), str(capsule_path)],
                capture_output=True,
                text=True,
                cwd=str(_REPO_ROOT),
                timeout=1800,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return VerificationResult(
                backend_id=self.backend_id,
                verdict=Verdict.UNAVAILABLE,
                detail=f"container replay could not run: {exc}",
                evidence={"machine_check": "not_run"},
            )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            msg = tail[-1][:240] if tail else f"replay exited {proc.returncode}"
            if "not present" in msg or "Build first" in msg:
                return VerificationResult(
                    backend_id=self.backend_id,
                    verdict=Verdict.UNAVAILABLE,
                    detail=f"environment not built: {msg}",
                    evidence={"machine_check": "not_run"},
                )
            return VerificationResult(
                backend_id=self.backend_id,
                verdict=Verdict.REJECTED,
                detail=msg,
                evidence={"machine_check": "failed"},
                outcome_kind="negative",
            )
        digest = None
        for line in (proc.stdout or "").splitlines():
            if "sha256:" in line:
                digest = line.split("sha256:")[1].split()[0][:64]
                break
        if not digest:
            return VerificationResult(
                backend_id=self.backend_id,
                verdict=Verdict.REJECTED,
                detail="replay succeeded but reported no image digest",
                evidence={"machine_check": "failed"},
                outcome_kind="negative",
            )
        return VerificationResult(
            backend_id=self.backend_id,
            verdict=Verdict.ACCEPTED,
            detail=f"pinned container replay succeeded (sha256:{digest[:16]})",
            evidence={
                "container_hash": digest,
                "machine_check": "passed",
                "arithmetic": "formal",
            },
            outcome_kind="lean_replay",
        )
