"""Receipt for the first production caller migration onto the canonical fs door.

The compatibility aliases remain registered and tested.  This small audit
measures the actual registry path, so a vocabulary reduction cannot be claimed
from a source edit alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from hcli.tool_registry import default_tool_registry


ROOT = Path(__file__).resolve().parents[2]
VMCP_GATE = ROOT / "hcli" / "agentos" / "vmcp_gate.py"
SCHEMA = "hawking.gravity.fs_canonical_migration.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measure(registry, name: str, args: dict[str, Any], rounds: int) -> dict[str, Any]:
    samples: list[float] = []
    last = None
    for _ in range(rounds):
        started = time.perf_counter_ns()
        last = registry.invoke(name, args)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    samples.sort()
    return {
        "tool": name,
        "rounds": rounds,
        "median_ms": samples[len(samples) // 2],
        "p95_ms": samples[min(len(samples) - 1, int(len(samples) * 0.95))],
        "ok": bool(last and last.ok),
        "sha256": (last.value or {}).get("sha256") if last and last.ok else None,
        "mutation": last.mutation if last else None,
    }


def build(*, rounds: int = 200) -> dict[str, Any]:
    registry = default_tool_registry(ROOT, repo_root=ROOT)
    args = {"path": "hcli/__init__.py", "max_bytes": 4096}
    canonical = _measure(registry, "fs", {"op": "read", **args}, rounds)
    alias = _measure(registry, "filesystem.read", args, rounds)
    canonical_spec = registry.get("fs")
    alias_spec = registry.get("filesystem.read")
    source = VMCP_GATE.read_text()
    source_checks = {
        "production_call_uses_canonical_name": 'call("fs", {"op": "read"' in source,
        "receipt_row_accepts_canonical_name": 'name in {"fs", "fs.read", "filesystem.read"}' in source,
        "compatibility_alias_remains_callable": bool(alias_spec and alias_spec.alias_of == "fs.read"),
    }
    parity = {
        "same_output_sha256": canonical["sha256"] == alias["sha256"],
        "both_ok": canonical["ok"] and alias["ok"],
        "same_mutation": canonical["mutation"] == alias["mutation"] == "read_only",
        "canonical_is_primary": bool(canonical_spec and canonical_spec.alias_of is None),
    }
    return {
        "schema": SCHEMA,
        "status": "MIGRATION_PARITY_PASSED" if all(source_checks.values()) and all(parity.values()) else "MIGRATION_PARITY_FAILED",
        "canonical_owner": "fs (op=read)",
        "migrated_production_caller": "hcli.agentos.vmcp_gate.run_vmcp_gate",
        "compatibility_alias": "filesystem.read",
        "source": {"path": str(VMCP_GATE), "sha256": _sha256(VMCP_GATE)},
        "source_checks": source_checks,
        "parity": parity,
        "measurements": {"canonical": canonical, "alias": alias},
        "authority": {
            "external_writes": False,
            "weights_written": False,
            "aliases_removed": False,
            "folders_moved": False,
        },
        "rollback": "restore the vmcp_gate call literal to filesystem.read; alias remains callable",
        "claim_boundary": (
            "One production caller now names the canonical fs family. The old "
            "alias remains callable; this receipt claims parity and vocabulary "
            "migration only, not alias retirement or broad repository cleanup."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "receipts" / "future" / "GRAVITY_FS_CANONICAL_MIGRATION_20260910.json",
    )
    args = parser.parse_args()
    result = build(rounds=max(20, int(args.rounds)))
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": result["status"], "parity": result["parity"]}, indent=2))
    return 0 if result["status"] == "MIGRATION_PARITY_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
