"""FPGA_HWIR acceptance. No board, no synthesis, no hardware number.

The criterion is NOT taken from APPENDIX O: that span (9000-9061) is APPENDIX K,
a schema dump shared identically by all four FPGA gates, and nothing in it can be
satisfied or refused. It is taken from civilization/GATE_CRITERIA_SUPPLEMENT.json,
which was committed BEFORE this runner existed and before the defect it names was
fixed, and which records clause (e) as known-false at authoring.

Five clauses over one corpus: conservation, discrimination, totality, invariance,
refusal. Each is a property of the IR, never "the tests pass".

    python3 -m tools.acceptance.fpga.run_gates
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.future import hwir

REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / "receipts" / "acceptance" / "FPGA_HWIR.json"
SUPPLEMENT = REPO / "civilization" / "GATE_CRITERIA_SUPPLEMENT.json"
SCHEMA = "hawking.acceptance.gate.v1"
GATE = "FPGA_HWIR"
EVIDENCE_TIER = "FUNCTIONAL_SIM"

# Every negative control and the code its own defect must produce.
CONTROLS = {
    "graph_dangling_edge": "DANGLING_EDGE",
    "graph_dense_source_rematerialization": "SOURCE_TENSOR_IDENTITY",
    "graph_over_budget": "RESOURCE_OVER_BUDGET",
    "graph_state_without_owner": "STATE_NO_OWNER",
    "graph_type_mismatch": "TYPE_MISMATCH",
}

# A serialised IR that carries wall-clock time is not content-addressed.
_TIMESTAMP_KEYS = ("generated_at", "timestamp", "created_at", "recorded_at", "measured_at")
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def declared_validator_codes() -> set[str]:
    """The codes the validator declares, read from its own _error call sites.

    Hardcoding twelve strings here would let the module grow a thirteenth defect
    that this acceptance never notices.
    """
    src = (REPO / "tools" / "future" / "hwir.py").read_text(encoding="utf-8")
    return set(re.findall(r"_error\(\s*[\"']([A-Z_]+)[\"']", src))


def legal_graphs() -> list[tuple[str, Any]]:
    return [
        ("from_qgemv", hwir.from_qgemv()),
        ("organ_map:FLASH", hwir.from_organ_map(hwir.FLASH_ORGAN_MAP)),
        ("organ_map:QWEN", hwir.from_organ_map(hwir.QWEN_ORGAN_MAP)),
    ]


def control_graphs() -> list[tuple[str, Any, str]]:
    return [(name, getattr(hwir, name)(), code) for name, code in sorted(CONTROLS.items())]


def probe_graph(primitive: str) -> Any:
    """One single-node graph per catalog primitive, built the way the module's own
    lowering-honesty test builds it."""
    from tools.future.test_hwir_lowering import _probe_graph
    return _probe_graph(primitive)


def _check(name: str, ok: bool, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def clause_a_conservation() -> list[dict[str, Any]]:
    """Round trip conserves bytes, fingerprint and freedom from wall-clock time."""
    out = []
    corpus = legal_graphs() + [(n, g) for n, g, _ in control_graphs()]
    corpus += [(f"probe:{p}", probe_graph(p)) for p in sorted(hwir.HARDWARE_PRIMITIVE_CATALOG)]
    for name, g in corpus:
        blob = g.to_json()
        again = hwir.from_json(blob).to_json()
        out.append(_check(f"a.bytes[{name}]", blob == again,
                          None if blob == again else "to_json is not a fixed point"))
        rt_fp = hwir.from_json(blob).fingerprint()
        out.append(_check(f"a.fingerprint[{name}]", rt_fp == g.fingerprint(),
                          None if rt_fp == g.fingerprint() else f"{rt_fp} != {g.fingerprint()}"))
        doc = json.loads(blob)
        stamped = sorted(k for k in _walk_keys(doc) if k in _TIMESTAMP_KEYS)
        iso = bool(_ISO.search(blob))
        out.append(_check(f"a.timeless[{name}]", not stamped and not iso,
                          {"keys": stamped, "iso8601_in_body": iso} if (stamped or iso) else None))
    return out


def _walk_keys(node: Any):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_keys(v)


def clause_b_discrimination() -> list[dict[str, Any]]:
    """The validator separates five distinct defects, each by its own code."""
    out = []
    declared = declared_validator_codes()
    observed: set[str] = set()
    for name, g in legal_graphs():
        r = hwir.validate(g)
        out.append(_check(f"b.legal_ok[{name}]", r.ok, None if r.ok else r.codes()))
        observed |= set(r.codes())
    sets: dict[str, frozenset[str]] = {}
    for name, g, expected in control_graphs():
        r = hwir.validate(g)
        codes = set(r.codes())
        observed |= codes
        out.append(_check(f"b.control_refused[{name}]", not r.ok))
        out.append(_check(f"b.control_own_code[{name}]", expected in codes,
                          None if expected in codes else {"expected": expected, "got": sorted(codes)}))
        sets[name] = frozenset(codes)
    distinct = len(set(sets.values())) == len(sets)
    out.append(_check("b.controls_are_distinguishable", distinct,
                      None if distinct else {n: sorted(s) for n, s in sets.items()}))
    unknown = sorted(observed - declared)
    out.append(_check("b.codes_are_declared", not unknown,
                      {"undeclared": unknown, "declared_n": len(declared)} if unknown else
                      {"declared_n": len(declared), "observed_n": len(observed)}))
    return out


def clause_c_totality() -> list[dict[str, Any]]:
    """Each target partitions the primitive catalog exhaustively and admits its holes."""
    out = []
    catalog = set(hwir.HARDWARE_PRIMITIVE_CATALOG)
    for tid in hwir.list_lowering_targets():
        t = hwir.get_lowering_target(tid)
        sup, cannot, hand = set(t.supported_primitives()), set(t.cannot_express()), set(t.handwritten_hdl())
        out.append(_check(f"c.disjoint[{tid}]", not (sup & cannot), sorted(sup & cannot) or None))
        undeclared = sorted(catalog - sup - cannot)
        out.append(_check(f"c.covers_catalog[{tid}]", not undeclared, undeclared or None))
        out.append(_check(f"c.handwritten_is_a_hole[{tid}]",
                          bool(hand) and hand <= cannot, sorted(hand - cannot) or None))
        for p in sorted(catalog):
            doc = t.lower(probe_graph(p))
            emitted = set(hwir.lowering_emitted_primitives(doc))
            holes = set(hwir.lowering_hole_primitives(doc))
            out.append(_check(f"c.emit_iff_supported[{tid}:{p}]", (p in emitted) == (p in sup),
                              {"emitted": p in emitted, "supported": p in sup}
                              if (p in emitted) != (p in sup) else None))
            out.append(_check(f"c.hole_iff_cannot[{tid}:{p}]", (p in holes) == (p in cannot),
                              {"hole": p in holes, "cannot_express": p in cannot}
                              if (p in holes) != (p in cannot) else None))
    return out


def clause_d_invariance() -> list[dict[str, Any]]:
    """Lowering never perturbs graph identity and no target is a ranked default."""
    out = []
    targets = list(hwir.list_lowering_targets())
    out.append(_check("d.no_preferred_target", hwir.PREFERRED_LOWERING_TARGET is None,
                      hwir.PREFERRED_LOWERING_TARGET))
    for name, g in legal_graphs():
        keysets = {}
        for tid in targets:
            doc = hwir.lower_hwir(g, tid)
            same = doc.get("graph_fingerprint") == g.fingerprint()
            out.append(_check(f"d.fingerprint_preserved[{name}:{tid}]", same,
                              None if same else
                              {"doc": doc.get("graph_fingerprint"), "graph": g.fingerprint()}))
            out.append(_check(f"d.not_preferred[{name}:{tid}]",
                              doc.get("preferred") is False and doc.get("toolchain_choice") is None,
                              {"preferred": doc.get("preferred"),
                               "toolchain_choice": doc.get("toolchain_choice")}))
            keysets[tid] = frozenset(doc.keys())
        identical = len(set(keysets.values())) == 1
        out.append(_check(f"d.identical_result_shape[{name}]", identical,
                          None if identical else {t: sorted(k) for t, k in keysets.items()}))
    for tid, man in sorted(hwir.lowering_target_manifests().items()):
        out.append(_check(f"d.manifest_unranked[{tid}]",
                          man.get("preferred") is False and man.get("toolchain_choice") is None,
                          {"preferred": man.get("preferred"),
                           "toolchain_choice": man.get("toolchain_choice")}))
    return out


def clause_e_refusal() -> list[dict[str, Any]]:
    """An IR that emits a plausible artifact for a graph its validator rejects has
    published a lie in a form synthesis will not catch."""
    out = []
    for tid in hwir.list_lowering_targets():
        for name, g, code in control_graphs():
            try:
                doc = hwir.lower_hwir(g, tid)
            except Exception as exc:            # the required outcome
                out.append(_check(f"e.refuses[{tid}:{name}]", True,
                                  {"raised": type(exc).__name__, "code": code}))
                continue
            out.append(_check(f"e.refuses[{tid}:{name}]", False, {
                "lowered_an_invalid_graph": True,
                "validator_codes": hwir.validate(g).codes(),
                "artifacts": len(doc.get("artifacts") or []),
                "doc_validate_ok": (doc.get("validate") or {}).get("ok"),
            }))
    return out


CLAUSES = {
    "a_conservation": clause_a_conservation,
    "b_discrimination": clause_b_discrimination,
    "c_totality": clause_c_totality,
    "d_invariance": clause_d_invariance,
    "e_refusal": clause_e_refusal,
}


def run() -> dict[str, Any]:
    supplement = json.loads(SUPPLEMENT.read_text(encoding="utf-8"))
    entry = supplement["gates"][GATE]
    started = time.perf_counter()

    checks: list[dict[str, Any]] = []
    per_clause: dict[str, Any] = {}
    for key, fn in CLAUSES.items():
        rows = fn()
        checks.extend(rows)
        failed = [r["name"] for r in rows if not r["ok"]]
        per_clause[key] = {"checks": len(rows), "failed": len(failed), "failing": failed[:12]}

    failed = [c for c in checks if not c["ok"]]
    verdict = "ACCEPTED" if not failed else "BLOCKED"
    blocker = None
    if failed:
        clauses = sorted({c["name"].split(".", 1)[0] for c in failed})
        blocker = {
            "missing": f"clause(s) {', '.join(clauses)} do not hold",
            "first_failure": failed[0],
            "failed_checks": len(failed),
            "total_checks": len(checks),
        }

    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    return {
        "schema": SCHEMA,
        "gate": GATE,
        "verdict": verdict,
        "criterion_quoted": entry["criterion"],
        "criterion_source": {
            "file": "civilization/GATE_CRITERIA_SUPPLEMENT.json",
            "pointer": f"gates.{GATE}.criterion",
            "authored_at_commit": entry["authored_at_commit"],
            "why_not_appendix_o": entry["span_states"],
        },
        "criterion_altered": False,
        "criterion_weakened": False,
        "command": ["python3", "-m", "tools.acceptance.fpga.run_gates"],
        "evidence_tier": EVIDENCE_TIER,
        "gpu_authority": False,
        "hardware_measured": False,
        "u50_present": False,
        "git_head": head,
        "clauses": per_clause,
        "checks": checks,
        "measured": {
            "corpus": {
                "legal_graphs": len(legal_graphs()),
                "negative_controls": len(CONTROLS),
                "catalog_primitives": len(hwir.HARDWARE_PRIMITIVE_CATALOG),
                "lowering_targets": list(hwir.list_lowering_targets()),
            },
            "checks_total": len(checks),
            "checks_failed": len(failed),
        },
        "blocker": blocker,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "recorded_by": "tools/acceptance/fpga/run_gates.py",
        "claim_boundary": (
            "FUNCTIONAL_SIM acceptance of the hardware IR itself. No board, no "
            "synthesis, no bitstream, no clock. Says nothing about FPGA throughput, "
            "HBM bandwidth, Fmax or resource utilisation; U50_PRESENT is false. A "
            "total, refusing IR is necessary for a physical compiler, not sufficient."
        ),
    }


def main() -> int:
    doc = run()
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{GATE} {doc['verdict']} "
          f"({doc['measured']['checks_failed']}/{doc['measured']['checks_total']} checks failed)")
    for key, row in doc["clauses"].items():
        mark = "ok " if not row["failed"] else "FAIL"
        print(f"  {mark} {key:18s} {row['checks']:4d} checks, {row['failed']} failed")
        for name in row["failing"]:
            print(f"        {name}")
    print(f"wrote {RECEIPT.relative_to(REPO)}")
    return 0 if doc["verdict"] == "ACCEPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
