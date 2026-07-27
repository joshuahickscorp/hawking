#!/usr/bin/env python3
"""Durably finish the amended pre-Odyssey campaign after PASS3 succeeds.

The expensive PASS3 worker is deliberately independent from this watcher.  This
process waits for PASS3's verified receipt, audits the immutable facts needed by
terminal gate A04, rebuilds and validates the Odyssey package against the
content-addressed Math-Preserve artifact, proves the launch fence refuses T0,
and only then seals the six-gate Motherload endpoint.

It never starts Odyssey and never writes ``ODYSSEY_LAUNCH_AUTHORIZED=true``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "GLM52_H0_98_MATH_PRESERVE_RECEIPT.json"
ALLOCATION = ROOT / "PROMETHEUS_MATH_ALLOCATION_MANIFEST.json"
ARTIFACT = (
    Path.home()
    / "Library/Application Support/Hawking/Models/GLM-5.2"
    / "b4734de4facf877f85769a911abafc5283eab3d9"
    / "GLM-5.2-H0.98-Math-Preserve.gravity"
)
ODYSSEY = ROOT / "odyssey"
FENCE = ODYSSEY / "launch/ODYSSEY_LAUNCH_AUTHORIZED"
AUDIT = ROOT / "HAWKING_ODYSSEY_READY_AUDIT.json"
POLL_SECONDS = int(os.environ.get("HAWKING_ODYSSEY_READY_POLL_SECONDS", "60"))


def _now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def audit_pass3(
    receipt_path: Path = RECEIPT,
    allocation_path: Path = ALLOCATION,
    artifact: Path = ARTIFACT,
) -> dict[str, Any]:
    """Independently reconcile PASS3's receipt with its durable artifact."""
    receipt = _read_json(receipt_path)
    coverage = receipt["coverage"]
    verification = receipt["verification"]
    byte_ledger = receipt["byte_ledger"]
    actual_bpw = float(receipt["actual_complete_bpw"])
    # The target rate is carried as an EXACT rational ({"num": 49, "den": 50}), because the
    # campaign's rate law is exact-rational and a float target would let a rounding artefact
    # decide a gate. float() on that dict is what crashed this watcher on the first real
    # receipt it ever saw -- the path had never run end to end before.
    target = receipt["target_complete_bpw"]
    target_bpw = (
        float(target["num"]) / float(target["den"])
        if isinstance(target, dict)
        else float(target)
    )

    _require(Path(receipt["artifact"]) == artifact, "receipt names the wrong artifact")
    _require(receipt.get("odyssey_input_ready") is True, "receipt does not admit Odyssey input")
    _require(coverage.get("complete") is True, "official coverage is incomplete")
    _require(coverage.get("official_tensors") == 59_585, "official tensor census changed")
    dispositions = coverage.get("dispositions", {})
    _require(
        isinstance(dispositions, dict) and sum(dispositions.values()) == 59_585,
        "not every tensor has a disposition",
    )
    _require(verification.get("shards_verified") == 282, "not all shards were verified")
    _require(verification.get("all_shards_ok") is True, "a shard failed verification")
    _require(
        verification.get("frozen_tensor_decisions_verified") == 59_585,
        "not every frozen tensor decision was verified",
    )
    _require(verification.get("decision_mismatches") == 0, "frozen allocation mismatch")
    _require(byte_ledger.get("itemization_reconciles") is True, "byte ledger does not reconcile")
    _require(actual_bpw <= target_bpw <= 0.98, "actual artifact exceeds H0.98")
    _require(actual_bpw <= 1.0, "actual artifact violates the one-bit law")
    _require(int(receipt["headroom_bytes"]) >= 0, "negative H0.98 headroom")

    index_path = artifact / "model.gravity.index.json"
    artifact_allocation = artifact / allocation_path.name
    _require(index_path.is_file(), "artifact model index is missing")
    _require(artifact_allocation.is_file(), "artifact allocation manifest is missing")
    _require(
        _sha256(index_path) == receipt["artifact_index_sha256"],
        "artifact content address differs from the PASS3 receipt",
    )
    _require(
        _sha256(allocation_path) == receipt["allocation_manifest_sha256"],
        "repository allocation manifest differs from the PASS3 receipt",
    )
    _require(
        _sha256(artifact_allocation) == receipt["allocation_manifest_sha256"],
        "artifact allocation manifest differs from the PASS3 receipt",
    )

    index = _read_json(index_path)
    _require(index.get("shard_count") == 282, "model index does not name 282 shards")
    _require(index.get("tensor_count") == 59_585, "model index tensor count changed")
    named_shards = index.get("shards", [])
    _require(len(named_shards) == 282, "model index shard list is incomplete")
    _require(len(set(named_shards)) == 282, "model index repeats a shard")
    _require(
        all((artifact / name).is_file() for name in named_shards),
        "a model-index shard is absent",
    )
    observed_bytes = _tree_bytes(artifact)
    _require(
        observed_bytes == int(byte_ledger["actual_package_bytes"]),
        "artifact tree bytes differ from the complete byte ledger",
    )

    return {
        "receipt": str(receipt_path),
        "artifact": str(artifact),
        "artifact_index_sha256": receipt["artifact_index_sha256"],
        "allocation_manifest_sha256": receipt["allocation_manifest_sha256"],
        "shards_verified": 282,
        "tensor_decisions_verified": 59_585,
        "actual_package_bytes": observed_bytes,
        "actual_complete_bpw": actual_bpw,
        "target_complete_bpw": target_bpw,
        "headroom_bytes": int(receipt["headroom_bytes"]),
        "one_bit_law": verification["one_bit_law"],
    }


def selftest() -> None:
    """Exercise the strict receipt/artifact reconciliation on a tiny fake tree."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        artifact = root / "math.gravity"
        artifact.mkdir()
        allocation = root / ALLOCATION.name
        allocation.write_text('{"frozen":true}\n')
        (artifact / allocation.name).write_bytes(allocation.read_bytes())

        shards = [f"model-{i:05d}-of-00282.gravity" for i in range(1, 283)]
        for i, name in enumerate(shards, 1):
            (artifact / name).write_bytes(bytes([i % 251]))
        index = {
            "shard_count": 282,
            "tensor_count": 59_585,
            "shards": shards,
        }
        index_path = artifact / "model.gravity.index.json"
        index_path.write_text(json.dumps(index, sort_keys=True) + "\n")
        package_bytes = _tree_bytes(artifact)
        receipt = root / RECEIPT.name
        receipt_obj = {
            "artifact": str(artifact),
            "odyssey_input_ready": True,
            "coverage": {
                "complete": True,
                "official_tensors": 59_585,
                "dispositions": {
                    "COMPACT_PAYLOAD": 59_000,
                    "PROTECTED_NATIVE_PAYLOAD": 585,
                },
            },
            "verification": {
                "shards_verified": 282,
                "all_shards_ok": True,
                "frozen_tensor_decisions_verified": 59_585,
                "decision_mismatches": 0,
                "one_bit_law": {"verdict": "PASS"},
            },
            "byte_ledger": {
                "itemization_reconciles": True,
                "actual_package_bytes": package_bytes,
            },
            "actual_complete_bpw": 0.97,
            "target_complete_bpw": 0.98,
            "headroom_bytes": 1,
            "artifact_index_sha256": _sha256(index_path),
            "allocation_manifest_sha256": _sha256(allocation),
        }
        receipt.write_text(json.dumps(receipt_obj) + "\n")

        got = audit_pass3(receipt, allocation, artifact)
        assert got["shards_verified"] == 282
        assert got["tensor_decisions_verified"] == 59_585
        receipt_obj["verification"]["decision_mismatches"] = 1
        receipt.write_text(json.dumps(receipt_obj) + "\n")
        try:
            audit_pass3(receipt, allocation, artifact)
        except RuntimeError as exc:
            assert "allocation mismatch" in str(exc)
        else:
            raise AssertionError("a frozen allocation mismatch passed the audit")
    print("odyssey_ready_finalize selftest PASS")


def _run(*args: str, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{' '.join(args)} failed ({proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def finish() -> dict[str, Any]:
    _require(RECEIPT.is_file(), "PASS3 receipt is not present")
    pass3 = audit_pass3()

    _run("tools/campaign/odyssey_package.py", "build")
    validation_proc = _run("tools/campaign/odyssey_package.py", "validate")
    validation = json.loads(validation_proc.stdout)
    _require(validation["verdict"] == "DRY_RUN_PASS", "Odyssey dry-run did not pass")
    _require(validation["checks_failed"] == 0, "Odyssey dry-run has failed checks")
    _require(FENCE.read_text().strip().lower() == "false", "Odyssey fence is not false")

    runner = subprocess.run(
        [sys.executable, str(ODYSSEY / "training/run.py"), "T0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    _require(
        runner.returncode == 1 and "refusing to start" in runner.stderr,
        "Odyssey runner did not enforce the false launch fence",
    )

    _run("tools/campaign/motherload.py", "selftest")
    note = (
        "PASS3 finalized 282/282 shards and verified 59,585/59,585 frozen tensor "
        f"decisions with zero mismatches; actual complete BPW={pass3['actual_complete_bpw']:.9f} "
        f"under H0.98; artifact index sha256={pass3['artifact_index_sha256']}; "
        "Odyssey package rebound and dry-run validated with launch fence false."
    )
    status_proc = _run(
        "tools/campaign/motherload.py", "gate", "A04", "GREEN", note
    )
    status = json.loads(status_proc.stdout)
    _require(status["gates_closed"] == 6, "not all six amended gates are closed")
    _require(status["gates_total"] == 6, "terminal gate set is not the six-gate amendment")
    _require(status["odyssey_launch_authorized"] is False, "launch fence became authorized")
    _require(status["endpoint"] == "HAWKING_ODYSSEY_READY", "READY endpoint was not reached")

    result = {
        "schema": "hawking.odyssey_ready.audit.v1",
        "at": _now(),
        "verdict": "HAWKING_ODYSSEY_READY",
        "pass3": pass3,
        "odyssey_validation": validation,
        "runner_refused_t0": True,
        "motherload": {
            "terminal_gate_set": status["terminal_gate_set"],
            "gates_closed": status["gates_closed"],
            "gates_total": status["gates_total"],
            "odyssey_launch_authorized": status["odyssey_launch_authorized"],
            "endpoint": status["endpoint"],
        },
    }
    tmp = AUDIT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, AUDIT)
    return result


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "once"
    if command == "status":
        print(json.dumps({
            "receipt_present": RECEIPT.is_file(),
            "audit_present": AUDIT.is_file(),
            "artifact": str(ARTIFACT),
        }, indent=1))
        return 0
    if command == "selftest":
        selftest()
        return 0
    if command == "once":
        if not RECEIPT.is_file():
            print("PASS3 receipt not present", file=sys.stderr)
            return 75
        print(json.dumps(finish(), indent=1, sort_keys=True))
        return 0
    if command != "watch":
        raise SystemExit(__doc__)

    while not RECEIPT.is_file():
        print(f"{_now()} waiting for {RECEIPT}", flush=True)
        time.sleep(POLL_SECONDS)
    print(json.dumps(finish(), indent=1, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
