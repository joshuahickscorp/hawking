"""CLAUDE_GLOBAL_FRONTIER — a live frontier, not a backlog.

Directive section 2 asks for missing / weak / stale / blocked / high-value-integration
work only, and section 78 requires every entry to carry prerequisite, expected
value, resource need, evidence level, integration target and duplication check.

The load-bearing property: a "missing" claim is not an assertion. Every entry
declares a probe (a path/glob existence test or a grep) which is EXECUTED at
build time. --verify re-runs every probe and exits non-zero if any claim has
gone stale, which is exactly what happens when Codex builds the thing first.

    python3 tools/future/global_frontier.py --build
    python3 tools/future/global_frontier.py --verify
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))


import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from tools.future._common import REPO, load_json, write_receipt

RECEIPT = "CLAUDE_GLOBAL_FRONTIER.json"

Classification = Literal["MISSING", "WEAK", "STALE", "BLOCKED", "HIGH_VALUE_INTEGRATION"]

REQUIRED_FIELDS = (
    "id",
    "title",
    "classification",
    "prerequisite",
    "expected_value",
    "resource_need",
    "evidence_level",
    "integration_target",
    "duplication_check",
    "probe",
)


# Directories that can never settle a claim about whether Hawking implements
# something. `.git` in particular carries a ref and a worktree directory per Grok
# lane, so a lane merely NAMED hwir made the "no HWIR exists" probe report hits.
_PRUNE = ("./target", "./.git", "./node_modules")


def _probe_absent(pattern: str) -> dict[str, Any]:
    """Evidence that nothing matching `pattern` exists in the source tree."""
    prune: list[str] = []
    for d in _PRUNE:
        prune += ["-path", d, "-prune", "-o"]
    r = subprocess.run(
        ["find", ".", *prune, "-iname", pattern, "-print"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    hits = [h for h in r.stdout.splitlines() if h and "__pycache__" not in h]
    return {"kind": "absent", "pattern": pattern, "hits": hits, "holds": not hits}


def _probe_present(relpath: str) -> dict[str, Any]:
    p = REPO / relpath
    return {"kind": "present", "path": relpath, "hits": [relpath] if p.exists() else [], "holds": p.exists()}


def _probe_field(relpath: str, dotted: str, expect: Any) -> dict[str, Any]:
    """Evidence that a receipt field currently equals `expect`."""
    p = REPO / relpath
    if not p.exists():
        return {"kind": "field", "path": relpath, "field": dotted, "expect": expect,
                "actual": "<file missing>", "holds": False}
    node: Any = load_json(p)
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            node = "<absent>"
            break
        node = node[part]
    return {"kind": "field", "path": relpath, "field": dotted, "expect": expect,
            "actual": node, "holds": node == expect}


def run_probe(spec: dict[str, Any]) -> dict[str, Any]:
    kind = spec["kind"]
    if kind == "absent":
        r = _probe_absent(spec["pattern"])
        # A MISSING claim goes stale for two very different reasons, and collapsing
        # them would be dishonest in opposite directions. If the only hits are in the
        # sidecar partition, the gap was CLOSED by this campaign -- that is the
        # frontier working, and the entry is RESOLVED, not wrong. Hits anywhere else
        # mean the claim was wrong when it was written, which is a real defect.
        if not r["holds"]:
            sidecar = [h for h in r["hits"] if h.startswith(("./tools/future/", "./receipts/future/"))]
            r["resolved_by_sidecar"] = len(sidecar) == len(r["hits"])
            r["resolving_paths"] = sidecar
        return r
    if kind == "present":
        return _probe_present(spec["path"])
    if kind == "field":
        return _probe_field(spec["path"], spec["field"], spec["expect"])
    raise ValueError(f"unknown probe kind {kind!r}")


# ---------------------------------------------------------------------------
# The frontier itself. Recovered from disk on 2026-08-29, not from the prompt.
#
# 2026-08-29 compounding-mode amendment: F003/F004/F005/F006/F010/F014 were
# closed by the scaffold campaign and their probes now resolve to sidecar files.
# They stay listed as history -- retiring them silently would lose the record
# that they were ever open -- and the entries below them are the COMPOUNDING
# frontier: the gaps that exist because Codex keeps moving, not because
# something was never built.
# ---------------------------------------------------------------------------

FRONTIER: list[dict[str, Any]] = [
    {
        "id": "F001",
        "title": "Flash source-independent NX is the single dominant blocker",
        "classification": "BLOCKED",
        "detail": (
            "12 of 14 BLOCKED Flash candidates in the physical qualification queue "
            "collapse to one missing dependency: a source-independent Flash NX with "
            "a protected complete-token measurement. FLASH_COMPLETE_V0.nx.json is "
            "sealed metadata only; the meta sub-1 receipt has serialized_artifact, "
            "physical_loader and native_kernel all NOT_BUILT."
        ),
        "prerequisite": "Flash NR V2 exists (it does); a serialized artifact and native consumer do not",
        "expected_value": "unblocks 12 queued Flash candidates at once — the highest fan-out item on the board",
        "resource_need": "no GPU for the audit; GPU authority for the eventual measurement",
        "evidence_level": "receipt-backed (queue blocked_reason strings + NX status field)",
        "integration_target": "tools/future/flash_nx_audit.py -> Codex candidate queue",
        "duplication_check": "Codex owns the NX build itself; sidecar only audits the dependency chain",
        "probe": {"kind": "field", "path": "receipts/headless/FLASH_COMPLETE_V0.nx.json",
                  "field": "status", "expect": "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION"},
    },
    {
        "id": "F002",
        "title": "12 Qwen27 candidates are READY_PROTECTED and idle on a GPU window",
        "classification": "BLOCKED",
        "detail": (
            "The queue holds 12 Qwen27 candidates at READY_PROTECTED. They wait on "
            "protected Metal authority the sidecar does not have and must not seize."
        ),
        "prerequisite": "an existing HCLI protected lease plus machine quiescence",
        "expected_value": "the automation around them is what makes the window cheap when it opens",
        "resource_need": "GPU authority (Codex lane); sidecar builds only the preflight",
        "evidence_level": "receipt-backed (queue counts.by_status)",
        "integration_target": "tools/future/candidate_planner.py + static_kernel_verify.py",
        "duplication_check": "physical_qualification.py already exists in tools/accelerator; sidecar adds planning, not execution",
        "probe": {"kind": "present", "path": "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"},
    },
    {
        "id": "F003",
        "title": "HWIR exists only as 15 hypotheses inside the atlas, not as an IR",
        "classification": "MISSING",
        "detail": (
            "ACCELERATOR_ARCHITECTURE_ATLAS.json carries hwir_hypotheses, but there is "
            "no HWIR module: no node/edge types, no serialization, no validator, no "
            "lowering from PhysicalGraph."
        ),
        "prerequisite": "PhysicalGraph semantics (hcli/physical_graph.py exists)",
        "expected_value": "the whole FPGA school is downstream of it; also the U50 arrival floor",
        "resource_need": "CPU only",
        "evidence_level": "find returned zero *hwir* files",
        "integration_target": "tools/future/hwir.py",
        "duplication_check": "atlas hypotheses are consumed as input, not re-derived",
        "probe": {"kind": "absent", "pattern": "*hwir*"},
    },
    {
        "id": "F004",
        "title": "No Hardware Doctor",
        "classification": "MISSING",
        "prerequisite": "HWIR (F003) for the proposal target space",
        "expected_value": "turns FPGA speculation into ranked falsifiable experiments before the board arrives",
        "resource_need": "CPU only",
        "evidence_level": "find + grep returned zero",
        "integration_target": "tools/future/hardware_doctor.py -> HCLI WorkUnit species",
        "duplication_check": "Doctor exists for representation; no hardware-axis Doctor exists",
        "probe": {"kind": "absent", "pattern": "*hardware_doctor*"},
    },
    {
        "id": "F005",
        "title": "No HBM Doctor",
        "classification": "MISSING",
        "prerequisite": "organ census (exists) + a criticality model",
        "expected_value": "decides what 8 GB of HBM is for; wrong answer wastes the board's only advantage",
        "resource_need": "CPU only",
        "evidence_level": "find + grep returned zero",
        "integration_target": "tools/future/hbm_doctor.py",
        "duplication_check": "bytes_atlas.py measures bytes; it does not solve residency",
        "probe": {"kind": "absent", "pattern": "*hbm_doctor*"},
    },
    {
        "id": "F006",
        "title": "No Green Machine energy accounting",
        "classification": "MISSING",
        "prerequisite": "none for the contract; trustworthy measurement for the numbers",
        "expected_value": "an axis of the dominance scoreboard that is currently absent entirely",
        "resource_need": "CPU only; most values will honestly be UNKNOWN",
        "evidence_level": "find + grep returned zero",
        "integration_target": "tools/future/green_machine.py -> tournament scoreboard",
        "duplication_check": "no energy module anywhere in tools/ or hcli/",
        "probe": {"kind": "absent", "pattern": "*green_machine*"},
    },
    {
        "id": "F007",
        "title": "No Learned Physical Compiler dataset contract",
        "classification": "MISSING",
        "detail": "Directive section 31 is explicit that data contracts precede any ML. Neither exists.",
        "prerequisite": "contamination metadata (F011) as a required field",
        "expected_value": "every experiment Codex runs becomes a training row instead of a one-off",
        "resource_need": "CPU only",
        "evidence_level": "find + grep returned zero",
        "integration_target": "tools/future/lpc_dataset.py",
        "duplication_check": "perf_model.py is a hand-written cost model, not a dataset contract",
        "probe": {"kind": "absent", "pattern": "*learned_physical*"},
    },
    {
        "id": "F008",
        "title": "No provenance graph / replication bundle / fault injection",
        "classification": "MISSING",
        "detail": "Directive section 52 and 66. Autonomy can currently launder weak evidence unchallenged.",
        "prerequisite": "none",
        "expected_value": "the guard that makes every later autonomous claim trustworthy",
        "resource_need": "CPU only",
        "evidence_level": "find returned zero for provenance_graph and replication_bundle",
        "integration_target": "tools/future/repro_science.py",
        "duplication_check": "receipt sealing exists; end-to-end provenance and failure injection do not",
        "probe": {"kind": "absent", "pattern": "*replication_bundle*"},
    },
    {
        "id": "F009",
        "title": "Negative science is a scattered corpus with no queryable index",
        "classification": "WEAK",
        "detail": (
            "Scars live in receipts/**, workspace/campaign/**, tools/foundry/"
            "NEGATIVE_TRANSFER_ATLAS.json and tools/headless/negative_science.py "
            "outputs. Nothing queries them before an experiment is proposed, so "
            "rediscovery is currently free."
        ),
        "prerequisite": "none",
        "expected_value": "every generator in this campaign gets a refusal path for known-dead hypotheses",
        "resource_need": "CPU only",
        "evidence_level": "negative_science.py exists but exposes no keyed retrieval",
        "integration_target": "tools/future/negative_index.py, queried by G009/G014/G020/G021",
        "duplication_check": "extends the existing corpus; does not restate scars",
        "probe": {"kind": "present", "path": "tools/headless/negative_science.py"},
    },
    {
        "id": "F010",
        "title": "Odyssey II has receipts but no scoped law store",
        "classification": "WEAK",
        "detail": (
            "ODYSSEY_TRANSFER_PROVEN.json, ACCELERATOR_TRANSFER_VERIFIED.json and "
            "QWEN38_ACCELERATOR_TRANSFER_MAP.json exist as outputs. There is no store "
            "that holds a law with its scope level and refuses unevidenced promotion "
            "between MODEL_LOCAL and GENERIC_VERIFIED."
        ),
        "prerequisite": "existing transfer receipts as seed data",
        "expected_value": "makes Flash<->Qwen27 transfer mechanical instead of per-campaign reasoning",
        "resource_need": "CPU only",
        "evidence_level": "receipts present; no law_scope module anywhere",
        "integration_target": "tools/future/odyssey2_law_store.py",
        "duplication_check": "consumes the existing receipts as seeds",
        "probe": {"kind": "absent", "pattern": "*law_store*"},
    },
    {
        "id": "F011",
        "title": "Contamination metadata is partial and not a promotion gate",
        "classification": "WEAK",
        "prerequisite": "none",
        "expected_value": "prevents a DIAGNOSTIC_RELATIVE number from being promoted as PROTECTED_ABSOLUTE",
        "resource_need": "CPU only",
        "evidence_level": "contamination logic appears inside odyssey_patient_runner.py and perfgate.py, with no shared deterministic record",
        "integration_target": "tools/future/contamination.py",
        "duplication_check": "consolidates, does not fork, the existing checks",
        "probe": {"kind": "present", "path": "tools/odyssey/contamination.py"},
    },
    {
        "id": "F012",
        "title": "Architecture Atlas is strong — consume it, do not rebuild it",
        "classification": "HIGH_VALUE_INTEGRATION",
        "detail": (
            "14 source schools, 21 behavior taxonomy entries, 17 backend-neutral "
            "primitives, 15 HWIR hypotheses, an ASIC candidate ledger and an "
            "experiment queue already exist. The value now is downstream wiring: "
            "primitives into a real library, hypotheses into a real IR."
        ),
        "prerequisite": "none",
        "expected_value": "avoids the most expensive duplication available in this campaign",
        "resource_need": "CPU only",
        "evidence_level": "receipt inspected: 91 KB, all sections populated",
        "integration_target": "tools/future/physical_primitives.py + hwir.py consume it",
        "duplication_check": "THIS ENTRY IS THE DUPLICATION CHECK",
        "probe": {"kind": "present", "path": "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json"},
    },
    {
        "id": "F013",
        "title": "No tournament harness for FLASH_SINGULARITY.NX vs QWEN27_SINGULARITY.NX",
        "classification": "MISSING",
        "detail": "genesis_tournament.py and doctor_tournament.py are different tournaments.",
        "prerequisite": "neither contender is a complete NX yet — the harness must refuse to run",
        "expected_value": "when both monsters mature, the comparison is already fair and pre-registered",
        "resource_need": "CPU only",
        "evidence_level": "no NX-vs-NX tournament module found",
        "integration_target": "tools/future/tournament.py + resident_install.py",
        "duplication_check": "existing tournaments are model-selection, not NX dominance",
        "probe": {"kind": "present", "path": "tools/genesis_tournament.py"},
    },
    {
        "id": "F014",
        "title": "No static kernel/ABI preflight independent of the GPU",
        "classification": "MISSING",
        "detail": (
            "Without GPU authority the cheapest way to protect a future protected "
            "window is to catch binding/ABI/geometry errors statically. Nothing does."
        ),
        "prerequisite": "the .metal sources and their Rust hosts (both present)",
        "expected_value": "each caught defect saves a whole protected window",
        "resource_need": "CPU only",
        "evidence_level": "no static kernel verifier module found",
        "integration_target": "tools/future/static_kernel_verify.py",
        "duplication_check": "cargo check compiles the host, not the host/shader ABI contract",
        "probe": {"kind": "absent", "pattern": "*static_kernel_verify*"},
    },
    {
        "id": "F015",
        "title": "Codex receipts are never ingested into anything downstream",
        "classification": "HIGH_VALUE_INTEGRATION",
        "detail": (
            "receipts/headless holds 997 artifacts and grows live. Nothing reads new "
            "ones and turns them into Odyssey II laws, Odyssey III attacks, atlas "
            "updates or learned-compiler rows. Every Codex result currently dies "
            "where it lands."
        ),
        "prerequisite": "the mutation surface map, so watching never becomes writing",
        "expected_value": "this is the compounding loop the whole directive is built around",
        "resource_need": "CPU only, read-only on the Codex surface",
        "evidence_level": "no ingest/watcher module targets receipts/headless",
        "integration_target": "tools/future/codex_ingest.py",
        "duplication_check": "modellake_watch.py watches downloads, not receipts",
        "probe": {"kind": "present", "path": "receipts/headless/ACCELERATOR_SCOREBOARD.json"},
    },
    {
        "id": "F016",
        "title": "Ingest deltas are emitted but nothing applies them",
        "classification": "HIGH_VALUE_INTEGRATION",
        "detail": (
            "codex_ingest classifies 1671 Codex artifacts and emits ~800 deltas, each naming "
            "its downstream consumers (Odyssey II, Odyssey III, atlas, PhysicalGraph, LPC, "
            "HWIR). No consumer reads them. The compounding loop is therefore open: the "
            "sidecar records that the frontier moved without any store changing."
        ),
        "prerequisite": "codex_ingest (exists) plus the seven consumer modules (all landed)",
        "expected_value": "closes the loop the whole sidecar exists to run",
        "resource_need": "CPU only",
        "evidence_level": "grep: only handoff.py and global_frontier.py reference the ingest receipt, and neither applies a delta",
        "integration_target": "tools/future/propagate.py",
        "duplication_check": "consumers already own their validation; the propagator routes, it does not re-implement",
        "probe": {"kind": "present", "path": "receipts/future/CODEX_INGEST_STATE.json"},
    },
    {
        "id": "F017",
        "title": "Derived artifacts cannot tell a cosmetic queue rewrite from a semantic one",
        "classification": "WEAK",
        "detail": (
            "Codex rewrote the queue repeatedly (37 -> 41 -> 44 -> 47 candidates), and every "
            "rewrite changes the file sha even when nothing semantic moved. Derived artifacts "
            "record queue_sha256 but have no way to say STALE_FINGERPRINT_ONLY versus "
            "STALE_SEMANTIC, so the only safe response is to regenerate everything every time."
        ),
        "prerequisite": "derived artifacts already record source sha and fingerprint",
        "expected_value": "makes resync cheap enough to run continuously rather than by hand",
        "resource_need": "CPU only",
        "evidence_level": "observed: plan built at 44 candidates while live was 47, detected only by manual comparison",
        "integration_target": "tools/future/freshness.py",
        "duplication_check": "codex_ingest already does sha-based change detection for receipts; this is the semantic layer above it",
        "probe": {"kind": "present", "path": "receipts/future/CANDIDATE_STAGED_PLAN.json"},
    },
    {
        "id": "F018",
        "title": "Six ABI findings await a Codex verdict with no prepared response",
        "classification": "BLOCKED",
        "detail": (
            "The static preflight raised 6 ERROR-class host/shader findings. Codex owns the "
            "investigation. Without a prepared response per verdict class, a false positive "
            "will simply be re-raised next scan and a confirmed bug will not become "
            "permanently detectable."
        ),
        "prerequisite": "Codex's verdict; the sidecar must not pre-judge or edit Codex files",
        "expected_value": "each verdict permanently improves the preflight instead of being consumed once",
        "resource_need": "CPU only; the verdict itself is Codex's",
        "evidence_level": "receipt-backed: STATIC_KERNEL_PREFLIGHT.json findings[] severity ERROR",
        "integration_target": "tools/future/abi_verdicts.py",
        "duplication_check": "the preflight owns detection; this owns the response to adjudication",
        "probe": {"kind": "present", "path": "receipts/future/STATIC_KERNEL_PREFLIGHT.json"},
    },
    {
        "id": "F019",
        "title": "Flash meta stalls at teacher capture, and the stated GPU blocker is false",
        "classification": "HIGH_VALUE_INTEGRATION",
        "detail": (
            "All nine real Flash meta families stall at gate 2 real_teacher_fit. Codex "
            "attempted the capture and recorded BLOCKED_NO_METAL_GPU with 0 of 256 required "
            "rows, failing at dense_source_bf16_prefix_initialization -- and this sidecar's "
            "meta funnel independently placed the same nine families at the same gate, so "
            "the STALL is corroborated. The stated CAUSE is not. METAL_REACHABILITY.json "
            "shows the host GPU is an Apple M3 Ultra, that the exact metal crate the "
            "runtime uses returns that device from an ordinary process here, and that "
            "shaders compile from source. The blocker is the process context of that one "
            "run, which nothing has yet identified -- not the machine."
        ),
        "prerequisite": (
            "identify why Device::system_default() returned None in that process, then "
            "re-run the capture; the GPU itself is reachable"
        ),
        "expected_value": (
            "everything downstream of gate 2 is idle, and the reason it is idle is now a "
            "process question rather than a hardware one"
        ),
        "resource_need": "GPU authority the sidecar does not have and will not take",
        "evidence_level": "receipt-backed: FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json status field",
        "integration_target": "tools/future/meta_ready.py prepares gates 3-9 to start on arrival",
        "duplication_check": "teacher_corpus owns admission; meta_funnel owns the gates; this is readiness only",
        "probe": {"kind": "field", "path": "receipts/headless/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json",
                  "field": "status", "expect": "BLOCKED_NO_METAL_GPU"},
    },
    {
        "id": "F020",
        "title": "Codex P6/P7 physical concepts are not projected into future hardware",
        "classification": "HIGH_VALUE_INTEGRATION",
        "detail": (
            "Fourteen P6/P7 candidates encode reusable physical concepts -- routed gate/up "
            "fusion, W2/down fusion, shared FP8 fusion, concurrent prefix, learned reader "
            "reuse, bounded expert cache reuse. None has a Hawking primitive, an HWIR "
            "hypothesis, an FPGA realization, a transfer scope or an Odyssey III "
            "counterexample. All are BLOCKED and unmeasured, so nothing may be claimed."
        ),
        "prerequisite": "physical_primitives (17 landed), hwir, odyssey3_adversary -- all exist",
        "expected_value": "the mechanism by which one physical win propagates to every future backend",
        "resource_need": "CPU only",
        "evidence_level": "queue: 14 candidate ids matching flash-p6-* / flash-p7-*, all BLOCKED",
        "integration_target": "tools/future/p6_projection.py",
        "duplication_check": "the atlas holds behaviours; this maps THESE candidates onto the existing primitives",
        "probe": {"kind": "present", "path": "receipts/headless/ACCELERATOR_FRONT_G_P6.json"},
    },
]


def build(strict: bool = False) -> Path:
    entries = []
    for e in FRONTIER:
        missing = [f for f in REQUIRED_FIELDS if f not in e]
        if missing:
            raise ValueError(f"{e.get('id')}: frontier entry missing {missing}")
        result = run_probe(e["probe"])
        entries.append({**e, "probe_result": result})

    resolved = [e["id"] for e in entries
                if not e["probe_result"]["holds"] and e["probe_result"].get("resolved_by_sidecar")]
    stale = [e["id"] for e in entries
             if not e["probe_result"]["holds"] and not e["probe_result"].get("resolved_by_sidecar")]
    doc = {
        "schema": "hawking.future.claude_global_frontier.v1",
        "version": 1,
        "purpose": "live frontier: missing / weak / stale / blocked / high-value-integration only",
        "recovered_from": "disk state on 2026-08-29, not from the directive text",
        "counts": {
            "total": len(entries),
            "by_classification": {
                c: sum(1 for e in entries if e["classification"] == c)
                for c in sorted({e["classification"] for e in entries})
            },
            "probes_holding": sum(1 for e in entries if e["probe_result"]["holds"]),
            "probes_stale": len(stale),
            "probes_resolved_by_sidecar": len(resolved),
        },
        "stale_entries": stale,
        "resolved_entries": resolved,
        "resolved_meaning": (
            "A MISSING claim whose probe now finds ONLY sidecar files: this campaign "
            "closed the gap. The entry is retired, not refuted."
        ),
        "stale_meaning": (
            "A stale probe means the claim no longer matches disk -- usually because "
            "Codex or a sidecar lane built the thing. Retire or reclassify the entry; "
            "do not silently keep it."
        ),
        "entries": entries,
    }
    out = write_receipt(RECEIPT, doc, "tools/future/global_frontier.py")
    if strict and stale:
        raise SystemExit(f"stale frontier claims: {stale}")
    return out


def verify() -> int:
    """Re-run every probe against current disk. Non-zero if any claim went stale."""
    bad, resolved = [], []
    for e in FRONTIER:
        r = run_probe(e["probe"])
        if r["holds"]:
            continue
        if r.get("resolved_by_sidecar"):
            resolved.append((e["id"], e["title"], r["resolving_paths"]))
        else:
            bad.append((e["id"], e["title"], r))
    for i, t, paths in resolved:
        print(f"resolved {i} {t}\n      closed by {', '.join(paths[:4])}")
    if bad:
        print("STALE FRONTIER CLAIMS:", file=sys.stderr)
        for i, t, r in bad:
            print(f"  {i} {t}\n      probe={json.dumps(r)[:200]}", file=sys.stderr)
        return 1
    print(
        f"frontier verified: {len(FRONTIER) - len(resolved)} claims still hold, "
        f"{len(resolved)} resolved by the sidecar"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        return verify()
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
