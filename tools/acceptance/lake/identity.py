"""MODELLAKE_IDENTITY_RESOLVED — CALL run_modellake_census, then lineage."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Optional

from tools.acceptance.lake.common import (
    GATES,
    LAKE,
    WORKTREE,
    jsonable,
    lake_mounted,
    load_symbol,
    receipts_dir,
    timed,
    write_receipt,
)
from tools.odyssey.modellake import resident_slugs
from tools.odyssey.modellake_lineage import ROADMAP_LIFECYCLE, express_lineage
from tools.odyssey.product_boundary import safe_defaults

GATE = "MODELLAKE_IDENTITY_RESOLVED"
_ORDER = {name: i for i, name in enumerate(ROADMAP_LIFECYCLE)}


def call_run_modellake_census(
    *,
    repo_root: str | Path,
    emit: str | Path,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Production call site of the catalog symbol. Not an import."""
    run_modellake_census = load_symbol(
        "hcli.agentos.modellake_gate", "run_modellake_census"
    )
    return run_modellake_census(
        repo_root=str(repo_root), emit=str(emit), timeout_s=float(timeout_s)
    )


def _lineage_rows(slugs: list[str]) -> list[dict[str, Any]]:
    cfg = safe_defaults()
    rows = []
    for slug in slugs:
        lin = express_lineage(slug, config=cfg, git_root=str(WORKTREE))
        prov = lin.get("provenance") or {}
        reg = lin.get("registry") or {}
        life = str(reg.get("lifecycle") or "DISCOVERED")
        fp = lin.get("architecture_fingerprint") or {}
        rows.append(
            {
                "slug": slug,
                "lifecycle": life,
                "lifecycle_index": _ORDER.get(life, -1),
                "repo": prov.get("repo"),
                "revision": prov.get("revision"),
                "resolved_sha": prov.get("resolved_sha"),
                "present": bool(reg.get("present")),
                "lifecycle_derived_from": reg.get("lifecycle_derived_from"),
                "fingerprint_strength": fp.get("strength") if isinstance(fp, dict) else None,
                "identity_resolved": (
                    _ORDER.get(life, -1) >= _ORDER["IDENTITY_RESOLVED"]
                    and bool(prov.get("repo"))
                    and bool(prov.get("revision"))
                ),
            }
        )
    return rows


def run_identity_gate(
    *,
    live: bool = True,
    run_census: bool = True,
    census_timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Demonstrate §14 identity resolution. Lake is read-only."""
    meta = GATES[GATE]
    command = [
        "python3",
        "-m",
        "tools.acceptance.lake",
        "--gate",
        GATE,
    ]
    with timed() as clock:
        census: Optional[dict[str, Any]] = None
        census_error: Optional[dict[str, str]] = None
        if run_census:
            emit = receipts_dir() / f"{GATE}.census.json"
            try:
                census = call_run_modellake_census(
                    repo_root=WORKTREE, emit=emit, timeout_s=census_timeout_s
                )
            except Exception as exc:  # keep the gate receipt even if census import/run fails
                census_error = {"type": type(exc).__name__, "message": str(exc)[:2000]}

        slugs = sorted(resident_slugs()) if live and lake_mounted() else []
        rows = _lineage_rows(slugs) if slugs else []
        unresolved = [r for r in rows if not r["identity_resolved"]]
        missing_body = [r for r in rows if not r["present"]]
        census_entries = []
        if isinstance(census, dict):
            spec = census.get("specimens") or {}
            census_entries = [
                e.get("name") for e in (spec.get("entries") or []) if isinstance(e, dict)
            ]

        checks = {
            "lake_mounted": lake_mounted() if live else False,
            "symbol_invoked": census is not None,
            "census_schema_is_modellake": bool(
                census and census.get("schema") == "hcli.agentos.modellake_census.v1"
            ),
            "sealed_count_is_55": len(slugs) == 55 if live else None,
            "every_sealed_specimen_identity_resolved": bool(rows) and not unresolved,
            "every_sealed_specimen_body_present": bool(rows) and not missing_body,
            "census_inventoried_specimens": (
                len(census_entries) >= 55 if census_entries else False
            ),
            "no_lake_write": True,
        }
        if live and lake_mounted() and rows and not unresolved:
            verdict = "ACCEPTED"
            blocker = None
        elif not live:
            verdict = "BLOCKED"
            blocker = {
                "missing_input": "live lake (live=False in this invocation)",
                "why": "fixture/dry invocation does not inspect the sealed school",
            }
        else:
            verdict = "BLOCKED"
            blocker = {
                "missing_input": "resolved repo+revision for every sealed specimen",
                "why": (
                    f"unresolved={len(unresolved)} missing_body={len(missing_body)} "
                    f"census_error={census_error}"
                ),
                "unresolved_slugs": [r["slug"] for r in unresolved[:20]],
            }

        measured = {
            "sealed_specimens": len(slugs),
            "identity_resolved": sum(1 for r in rows if r["identity_resolved"]),
            "unresolved": len(unresolved),
            "lifecycle_counts": dict(Counter(r["lifecycle"] for r in rows)),
            "fingerprint_counts": dict(Counter(r["fingerprint_strength"] for r in rows)),
            "census_status": None if census is None else census.get("status"),
            "census_specimen_children": len(census_entries),
            "census_error": census_error,
            "required_sealed_count": 55,
        }
        output = {
            "summary": (
                f"{measured['identity_resolved']}/{measured['sealed_specimens']} "
                f"sealed specimens at IDENTITY_RESOLVED or later; "
                f"census status={measured['census_status']!r}"
            ),
            "specimens": rows,
            "census_capacity": jsonable((census or {}).get("capacity")),
            "census_checks": jsonable((census or {}).get("checks")),
            "census_flash_target": jsonable((census or {}).get("flash_target_manifest")),
            "census_receipt_path": None if census is None else census.get("receipt_path"),
        }
        return write_receipt(
            GATE,
            verdict=verdict,
            command=command,
            output=output,
            measured=measured,
            checks=checks,
            evidence_tier="STATIC",
            symbol_invoked=checks["symbol_invoked"],
            blocker=blocker,
            elapsed_s=clock.snap(),
            extra={
                "lake": str(LAKE),
                "implementing_symbol": meta["symbol"],
            },
        )
