"""Pinned-Lean compile checks for repair predictions.

Reuses the Mathlib pin and lake-env-lean pattern from D4 extraction.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ramanujan.data.paths import MATHLIB_ROOT


def _elan_env() -> dict[str, str]:
    env = os.environ.copy()
    elan = Path.home() / ".elan" / "bin"
    env["PATH"] = f"{elan}:{env.get('PATH', '')}"
    return env


def _by_block(proof: str) -> str:
    proof = (proof or "").strip()
    if proof.startswith("by"):
        proof = proof[2:].lstrip()
    if "\n" not in proof and not re.match(
        r"^(rw|simp|exact|apply|intro|rfl|refine|cases|have|let)\b", proof
    ):
        if proof and not proof.startswith("exact"):
            proof = f"exact ({proof})" if not proof.startswith("@") else f"exact {proof}"
    body = textwrap.indent(proof, "  ")
    return "by\n" + body if body.strip() else "by\n  skip"


def check_proof(
    *,
    import_mod: str,
    signature: str,
    proof: str,
    mathlib: Path | None = None,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Return {ok, returncode, error_excerpt} for example signature := proof under Mathlib."""
    root = mathlib or MATHLIB_ROOT
    opens = "open Nat List Function"
    lean_src = f"""import {import_mod}
{opens}
set_option maxHeartbeats 400000
example {signature} :=
{_by_block(proof)}
"""
    env = _elan_env()
    with tempfile.TemporaryDirectory(prefix="ramanujan_repair_eval_") as td:
        src_path = Path(td) / "Check.lean"
        src_path.write_text(lean_src, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["lake", "env", "lean", str(src_path)],
                cwd=str(root),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "returncode": -1, "error_excerpt": "timeout"}
        except OSError as e:
            return {"ok": False, "returncode": -2, "error_excerpt": str(e)}
        err = ((proc.stderr or "") + (proc.stdout or "")).strip()
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "error_excerpt": "\n".join(err.splitlines()[:12]) if err else "",
        }


def check_many(
    jobs: list[dict[str, str]],
    *,
    workers: int = 4,
    timeout: float = 90.0,
) -> list[dict[str, Any]]:
    """Parallel lean checks. Each job: import, signature, proof, id?."""
    workers = max(1, min(int(workers), 8))
    results: list[dict[str, Any] | None] = [None] * len(jobs)

    def _one(i: int, job: dict[str, str]) -> tuple[int, dict[str, Any]]:
        r = check_proof(
            import_mod=job["import"],
            signature=job["signature"],
            proof=job["proof"],
            timeout=timeout,
        )
        r["id"] = job.get("id")
        return i, r

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, i, j) for i, j in enumerate(jobs)]
        for fut in as_completed(futs):
            i, r = fut.result()
            results[i] = r
    return [r if r is not None else {"ok": False, "error_excerpt": "missing"} for r in results]
