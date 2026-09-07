"""VMCP disposition — what of the All-Seeing Eye exists, and a compact invoke.

Roadmap §8 / APPENDIX E: VMCP is Hawking's perception + visual / 3D /
application-generation organ. Nine Acts: SEE HOLD OPEN KNOW MAKE CHECK FIX
KEEP PROVE. It is not a prose wrapper.

What actually exists
--------------------
VisionMCP is a foreign package (visionmcp 0.8.0a2, ~1770 Python modules, 15
core MCP tools). It is NOT a tree in this repository's HEAD. Hawking-side
callers live in hcli/agentos/vmcp/ (FileEye + CaptureBus integration, capability
probe, lattice disposition, forgery canary, unavailable gate) and
hcli/vmcp_adapter.py (read-only; this lane does not edit hcli/).

This sidecar's compact E.14 surface stays nine verbs. Organs that can run
on this host without visionmcp and without the network are implemented under
hcli/agentos/vmcp/ and CALLED from see/check/prove (an import is not a call site):

  see  -> hcli.agentos.vmcp.file_eye.observe  (real magic/header classification)
  see  -> hcli.agentos.vmcp.pty_eye.capture   (real PTY if openpty works; else PARKED)
  check-> hcli.agentos.vmcp.tool_doctor.profile (real local ToolReceipt)
  prove-> hcli.agentos.vmcp.behavior_lab.run_matrix (BHV-01..23 in isolated temp repos)

OPEN / MAKE / FIX / KEEP stay PARKED: those still live in the foreign package.
WEB / SPATIAL / visual-proof / LABORATORY stay PARKED with explicit blockers.

    python3 hcli/agentos/vmcp/disposition.py --build
    python3 hcli/agentos/vmcp/disposition.py --selftest
    python3 hcli/agentos/vmcp/disposition.py --build
    python3 -m pytest hcli/agentos/vmcp/test_behavior_lab.py hcli/agentos/vmcp/test_file_eye.py -q -o addopts=""
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, REPO, git
from hcli.agentos.vmcp.behavior_lab import run_matrix as bhv_run_matrix
from hcli.agentos.vmcp.file_eye import observe as file_observe
from hcli.agentos.vmcp.pty_eye import capture as pty_capture
from hcli.agentos.vmcp.pty_eye import probe as pty_probe
from hcli.agentos.vmcp.tool_doctor import profile as doctor_profile
from hcli.agentos.vmcp.tool_doctor import report as doctor_report

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


RECEIPT = "VMCP_DISPOSITION.json"
SCHEMA = "hawking.future.vmcp.v1"
VERSION = 1
RECORDED_BY = "hcli/agentos/vmcp/disposition.py"
DISPOSITION_SCHEMA = "hawking.audit.subsystem_disposition.v1"
WAKE_SCHEMA = "hawking.audit.wake_condition.v1"
WAKE_REQUIRED_KIND = "call"

NINE_ACTS: tuple[str, ...] = (
    "see",
    "hold",
    "open",
    "know",
    "make",
    "check",
    "fix",
    "keep",
    "prove",
)
CONNECTED_ACTS: frozenset[str] = frozenset({"see", "hold", "know", "check", "prove"})
PARKED_ACTS: frozenset[str] = frozenset({"open", "make", "fix", "keep"})

CORE_MCP_TOOLS: tuple[str, ...] = (
    "vision.capabilities",
    "vision.observe",
    "vision.query",
    "vision.explain_region",
    "vision.compare",
    "vision.verify",
    "vision.progress",
    "vision.review_queue",
    "vision.open_project",
    "vision.close_project",
    "vision.list_artifacts",
    "vision.get_artifact",
    "system.doctor",
    "project.create",
    "project.status",
)

# Roadmap §8.2 / E.2. Already disposed against visionmcp in
# receipts/headless/VMCP_LATTICE_DISPOSITION.json — CONSOLIDATE, do not rebuild.
LATTICE_NAMES: tuple[str, ...] = (
    "DEEP_DIGEST",
    "ASSET_LATTICE",
    "DECODE_LATTICE",
    "ENTITY_GENOME",
    "RENDER_GENOME",
    "SPATIAL_GENOME",
    "REPAIR_VECTOR",
    "DIRECTOR_STATE",
    "PERFORMANCE_LEDGER",
    "TRUTH_LEDGER",
)

CENSUS_REL = "receipts/headless/VMCP_CENSUS.json"
LATTICE_REL = "receipts/headless/VMCP_LATTICE_DISPOSITION.json"
INTEGRATION_REL = "receipts/headless/VMCP_AGENTOS_INTEGRATION.json"

CLAIM_BOUNDARY = (
    "Local-file classification, local ToolReceipts, and isolated BHV fixtures "
    "on this host. No browser, no Blender, no Ghidra, no GPU, no claim that "
    "VisionMCP ran. PTY is attempted via openpty; EPERM on the slave is a "
    "blocker, never a pipe stand-in. An empty collection is not evidence of "
    "absence: unavailable tools must not look like a negative finding."
)
VERB_COUNT = len(NINE_ACTS)


def _head_names() -> set[str]:
    blob = git("ls-tree", "-r", "--name-only", "HEAD")
    return {line for line in blob.splitlines() if line}


def _read_text(rel: str) -> tuple[str | None, str]:
    path = REPO / rel
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace"), "worktree"
    blob = git("show", f"HEAD:{rel}")
    if blob:
        return blob, "git:HEAD"
    return None, "unlocated"


def _load_optional(rel: str) -> tuple[dict[str, Any] | None, str]:
    text, taken = _read_text(rel)
    if text is None:
        return None, taken
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None, taken
    return (value if isinstance(value, dict) else None), taken


def locate_visionmcp() -> dict[str, Any]:
    """Record where the foreign package is, without treating sparse as absence.

    Candidates match hcli/agentos/vmcp/capability_probe.py. This sidecar
    never imports visionmcp: an import is not a call of the compact surface,
    and the heavy stack is out of scope here.
    """
    env = os.environ.get("VISIONMCP_SRC")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            REPO / "visionmcp" / "src",
            Path("/Users/scammermike/Downloads/hawking/visionmcp/src"),
            Path("/Users/scammermike/Downloads/hawking-copy/visionmcp/src"),
            Path.home() / ".searcher-donors" / "visionmcp" / "src",
        ]
    )
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    found: Path | None = None
    for src in candidates:
        key = str(src)
        if key in seen:
            continue
        seen.add(key)
        init = src / "visionmcp" / "__init__.py"
        bus = src / "visionmcp" / "perception" / "bus.py"
        on_disk = init.is_file()
        rows.append(
            {
                "path": key,
                "on_disk": on_disk,
                "has_bus": bus.is_file(),
            }
        )
        if found is None and on_disk:
            found = src
    in_head = "visionmcp" in {p.split("/")[0] for p in _head_names() if p}
    return {
        "found": found is not None,
        "path": str(found) if found is not None else None,
        "in_this_repo_HEAD": bool(in_head),
        "candidates": rows,
        "note": (
            "visionmcp is a foreign package, not a hawking tree. "
            "git cat-file HEAD:visionmcp is missing. Sparse checkout is not "
            "evidence the package does not exist elsewhere."
        ),
        "evidence_tier": "STATIC",
    }


def _wake(
    *,
    kind: str,
    required_symbol: str,
    predicate: str,
    blocker: str,
    missing_dependency: str,
) -> dict[str, Any]:
    return {
        "schema": WAKE_SCHEMA,
        "kind": kind,
        "required_kind": WAKE_REQUIRED_KIND,
        "required_symbol": required_symbol,
        "required_caller_prefix": "hcli/",
        "predicate": predicate,
        "blocker": blocker,
        "missing_dependency": missing_dependency,
        "satisfy_by": [
            {
                "channel": "visionmcp",
                "where": "VISIONMCP_SRC / visionmcp/src",
                "required_kind": WAKE_REQUIRED_KIND,
                "required_symbol": required_symbol,
                "how": (
                    "materialize the foreign visionmcp 0.8.0a2 checkout, then a "
                    "production AST Call of the named symbol (an import is not a "
                    "call site)"
                ),
            },
            {
                "channel": "adapter",
                "where": "receipts/future/REACHABILITY_TRIAGE.json",
                "required_kind": WAKE_REQUIRED_KIND,
                "required_symbol": required_symbol,
                "how": (
                    "add a WIRED handler that AST-Calls the named symbol; an "
                    "import is not enough"
                ),
            },
        ],
        "evidence_tier": "STATIC",
    }


def _parked_act_wake(act: str) -> dict[str, Any]:
    table = {
        "open": (
            "VISIONMCP_OPENER",
            "visionmcp.perception.bus.CaptureBus.observe",
            "production AST Call of CaptureBus.observe (or an authorized opener) "
            "on a compressed / encoded / minified / binary target from an HCLI "
            "entry after VISIONMCP_SRC is importable",
            "no opener in this repo; decoding lives in the foreign package",
            "VISIONMCP_SRC pointing at visionmcp 0.8.0a2 src/ (parent of the "
            "visionmcp package) plus the format-specific opener",
        ),
        "make": (
            "VISIONMCP_COMPILER",
            "visionmcp.mcp.factory.create_server",
            "production AST Call of visionmcp compiler surface (vision.compile / "
            "VisualProgramIR emit) from an HCLI entry with profile='compiler'",
            "application / visual-program generation is aspiration on the "
            "Hawking side; the compiler profile is the foreign package",
            "VISIONMCP_SRC with the compiler extra; do not build a parallel "
            "VisualProgramIR in hawking/",
        ),
        "fix": (
            "VISIONMCP_REPAIR",
            "visionmcp.repair.ledger.RepairEvidenceLedger",
            "production AST Call of the visionmcp repair ledger / residual loop "
            "from an HCLI entry after a CHECK residual exists",
            "repair loop is in the foreign package; Hawking must not grow a "
            "second residual director",
            "VISIONMCP_SRC importable and a residual from CHECK",
        ),
        "keep": (
            "VISIONMCP_PROJECT_STORE",
            "visionmcp.projects.store.ProjectStore.create",
            "production AST Call of ProjectStore.create / CaptureBus.observe "
            "that persists CAS artifacts from an HCLI entry",
            "sensory KEEP is ProjectStore + ArtifactStore in the foreign "
            "package; this sidecar only writes a STATIC disposition receipt",
            "VISIONMCP_SRC importable; profile='core' only (LABORATORY is a "
            "second AgentOS and is refused)",
        ),
    }
    kind, symbol, predicate, blocker, dep = table[act]
    return _wake(
        kind=kind,
        required_symbol=symbol,
        predicate=predicate,
        blocker=blocker,
        missing_dependency=dep,
    )


def _organ_table(located: Mapping[str, Any]) -> list[dict[str, Any]]:
    located_note = (
        f"foreign package located at {located['path']}"
        if located.get("found")
        else "foreign package not on any candidate path in this process"
    )
    web_wake = _wake(
        kind="VISIONMCP_WEB_PROFILE",
        required_symbol="visionmcp.mcp.factory.create_server",
        predicate=(
            "production AST Call of create_server(..., profile='web') plus a "
            "Chromium-backed observe from an HCLI entry"
        ),
        blocker="web eye needs the visionmcp web extra and host Chrome",
        missing_dependency="VISIONMCP_SRC with .[web] and a host Chrome binary",
    )
    spatial_wake = _wake(
        kind="VISIONMCP_3D_PROFILE",
        required_symbol="visionmcp.mcp.factory.create_server",
        predicate=(
            "production AST Call of create_server(..., profile='3d') plus a "
            "Blender/GLTF observe from an HCLI entry"
        ),
        blocker="spatial eye needs the visionmcp 3d extra and Blender; FPGA/U50 "
        "is unrelated and absent hardware is never a measurement",
        missing_dependency="VISIONMCP_SRC with Blender CLI on PATH",
    )
    probed = pty_probe()
    if probed.get("ok"):
        pty_wake = None
    else:
        pty_wake = _wake(
            kind="PTY_OPEN_DENIED",
            required_symbol="hcli.agentos.vmcp.pty_eye.capture",
            predicate=(
                "production AST Call of hcli.agentos.vmcp.pty_eye.capture after "
                "openpty succeeds (unsandboxed / gate profile). A pipe "
                "capture is not a PTY session. An import is not a call site."
            ),
            blocker=str(probed.get("blocker") or "openpty EPERM"),
            missing_dependency="unsandboxed process that can openpty (gate profile)",
        )
    bhv_wake = None
    doctor_wake = None
    lab_wake = _wake(
        kind="DO_NOT_USE_LABORATORY_PROFILE",
        required_symbol="visionmcp.mcp.factory.create_server",
        predicate=(
            "never: LABORATORY is a second AgentOS (queue / retry / "
            "deliver_promoted). Wake is an explicit refusal to integrate "
            "against that profile."
        ),
        blocker=(
            "receipts/headless/VMCP_CENSUS.json: the laboratory server is the "
            "second AgentOS; integrate against profile='core' only"
        ),
        missing_dependency="none — this is a permanent restriction, not a missing library",
    )
    capture_wake = _wake(
        kind="VISIONMCP_CAPTURE_BUS",
        required_symbol="visionmcp.perception.bus.CaptureBus.observe",
        predicate=(
            "production AST Call of CaptureBus.observe from an HCLI entry "
            "after VISIONMCP_SRC is importable (tools/headless/"
            "hcli_integration.py already contains that Call, but it is "
            "not reachable until the foreign package is on sys.path)"
        ),
        blocker=(
            "CaptureBus is real in visionmcp and called from "
            "hcli/agentos/vmcp/hcli_integration.py:observe_file; this "
            "worktree cannot import visionmcp from HEAD"
        ),
        missing_dependency="VISIONMCP_SRC / a materialized visionmcp/src in the checkout",
    )
    return [
        {
            "id": "vmcp.file_eye",
            "organ": "EYES / generic file metadata eye (roadmap VMCP-002)",
            "disposition": "CONNECTED",
            "exists_today": (
                "hcli.agentos.vmcp.file_eye.observe: magic/header classification of a "
                "real local file (Mach-O/ELF/PE/PNG/ZIP/WASM/...) plus hashlib "
                "identity fields matching FileEye (present, path, size, mode, "
                "sha256). TARGET_ABSENT is a classification, never an empty success."
            ),
            "aspiration": "CaptureBus-backed observe with CAS reuse in visionmcp",
            "symbol": "hcli.agentos.vmcp.file_eye.observe",
            "call_sites": [
                {
                    "file": "hcli/agentos/vmcp/disposition.py",
                    "symbol": "hcli.agentos.vmcp.file_eye.observe",
                    "kind": "call",
                    "via": "see()",
                },
                {
                "file": "receipts/future/REACHABILITY_TRIAGE.json",
                    "symbol": "hcli.agentos.vmcp.disposition.compact_surface",
                    "kind": "call",
                    "via": "WIRED vmcp.disposition (compatibility capability id)",
                },
            ],
            "evidence_tier": "FUNCTIONAL_SIM",
            "execution": "REAL",
            "wake": None,
        },
        {
            "id": "vmcp.capture_bus",
            "organ": "CORE / CaptureBus.observe (the acquire seam)",
            "disposition": "PARKED",
            "exists_today": (
            "hcli/agentos/vmcp/hcli_integration.py:observe_file CALLS "
                "CaptureBus.observe; FileEye implements SensorAdapter. Proven "
                "when visionmcp is on sys.path. Not in hawking HEAD."
            ),
            "aspiration": "HCLI tool_registry verb that calls observe_file",
            "symbol": "hcli.agentos.vmcp.hcli_integration.observe_file",
            "call_sites": [],
            "evidence_tier": "STATIC",
            "wake": capture_wake,
            "located": located_note,
        },
        {
            "id": "vmcp.web_eye",
            "organ": "WEB EYE",
            "disposition": "PARKED",
            "exists_today": "visionmcp web profile (foreign); nothing in hawking calls it",
            "aspiration": "roadmap E.7 controlled page capture / layout / canaries",
            "evidence_tier": "STATIC",
            "wake": web_wake,
        },
        {
            "id": "vmcp.spatial_eye",
            "organ": "SPATIAL EYE",
            "disposition": "PARKED",
            "exists_today": "visionmcp 3d profile + Blender demo (foreign)",
            "aspiration": "roadmap E.9 mesh / material / multi-view proof",
            "evidence_tier": "STATIC",
            "wake": spatial_wake,
        },
        {
            "id": "vmcp.visual_proof",
            "organ": "VISUAL PROOF",
            "disposition": "PARKED",
            "exists_today": "visionmcp compiler residual / receipt verification (foreign)",
            "aspiration": "roadmap E.8 pixel / layout / region diffs as a Hawking organ",
            "evidence_tier": "STATIC",
            "wake": _parked_act_wake("make"),
        },
        {
            "id": "vmcp.pty_eye",
            "organ": "PTY EYE",
            "disposition": "CONNECTED" if probed.get("ok") else "PARKED",
            "exists_today": (
                "hcli.agentos.vmcp.pty_eye.capture attempts a real posix PTY pair. "
                "This sandbox has been measured to deny the slave (openpty "
                "EPERM); that is a blocker, not a pipe stand-in."
            ),
            "aspiration": "roadmap E.10 terminal screen / scrollback / ANSI",
            "symbol": "hcli.agentos.vmcp.pty_eye.capture",
            "call_sites": [
                {
                    "file": "hcli/agentos/vmcp/disposition.py",
                    "symbol": "hcli.agentos.vmcp.pty_eye.capture",
                    "kind": "call",
                    "via": "see(organ=pty)",
                }
            ],
            "evidence_tier": "FUNCTIONAL_SIM" if probed.get("ok") else "STATIC",
            "execution": "REAL" if probed.get("ok") else "BLOCKED",
            "used_real_pty": bool(probed.get("ok")),
            "wake": pty_wake,
        },
        {
            "id": "vmcp.behavior_lab",
            "organ": "BEHAVIOR LAB",
            "disposition": "CONNECTED",
            "exists_today": (
                "hcli.agentos.vmcp.behavior_lab.run_matrix runs BHV-01..23 in isolated "
                "temp git repos and CALLs tools.future.tabula.evaluate. "
                "LABORATORY profile is not used."
            ),
            "aspiration": "roadmap E.11 fixture matrix",
            "symbol": "hcli.agentos.vmcp.behavior_lab.run_matrix",
            "call_sites": [
                {
                    "file": "hcli/agentos/vmcp/disposition.py",
                    "symbol": "hcli.agentos.vmcp.behavior_lab.run_matrix",
                    "kind": "call",
                    "via": "prove(organ=behavior_lab)",
                }
            ],
            "evidence_tier": "FUNCTIONAL_SIM",
            "execution": "REAL",
            "wake": bhv_wake,
        },
        {
            "id": "vmcp.tool_doctor",
            "organ": "TOOL DOCTOR",
            "disposition": "CONNECTED",
            "exists_today": (
                "hcli.agentos.vmcp.tool_doctor.profile runs a real local argv into an "
                "E.4 ToolReceipt (availability, version, permissions, health, "
                "limits, refusal). visionmcp system.doctor is a different "
                "symbol and is not called."
            ),
            "aspiration": "roadmap E.3 required classes hosted by HCLI",
            "symbol": "hcli.agentos.vmcp.tool_doctor.profile",
            "call_sites": [
                {
                    "file": "hcli/agentos/vmcp/disposition.py",
                    "symbol": "hcli.agentos.vmcp.tool_doctor.profile",
                    "kind": "call",
                    "via": "check(organ=tool_doctor)",
                }
            ],
            "evidence_tier": "FUNCTIONAL_SIM",
            "execution": "REAL",
            "wake": doctor_wake,
        },
        {
            "id": "vmcp.laboratory_profile",
            "organ": "LABORATORY profile (do not use)",
            "disposition": "PARKED",
            "exists_today": (
                "visionmcp LABORATORY contains a real queue, retry and "
                "deliver_promoted — a second AgentOS"
            ),
            "aspiration": "none; integrating against it is forbidden",
            "evidence_tier": "STATIC",
            "wake": lab_wake,
        },
        {
            "id": "vmcp.app_generation",
            "organ": "MAKE / VisualProgramIR / application generation",
            "disposition": "PARKED",
            "exists_today": "visionmcp compiler profile (foreign); not in hawking HEAD",
            "aspiration": "roadmap §8 MAKE: emit real artifacts, not narration",
            "evidence_tier": "STATIC",
            "wake": _parked_act_wake("make"),
        },
    ]


def disposition() -> dict[str, Any]:
    """CONNECTED file-eye vs PARKED organs. Neither is absent by accident."""
    located = locate_visionmcp()
    census, census_taken = _load_optional(CENSUS_REL)
    lattice, lattice_taken = _load_optional(LATTICE_REL)
    integration, integration_taken = _load_optional(INTEGRATION_REL)
    acts = []
    for act in NINE_ACTS:
        if act in CONNECTED_ACTS:
            acts.append(
                {
                    "act": act,
                    "disposition": "CONNECTED",
            "symbol": f"hcli.agentos.vmcp.disposition.{act}",
                    "evidence_tier": "FUNCTIONAL_SIM",
                    "wake": None,
                    "empty_success": False,
                }
            )
        else:
            wake = _parked_act_wake(act)
            acts.append(
                {
                    "act": act,
                    "disposition": "PARKED",
                    "symbol": None,
                    "evidence_tier": "STATIC",
                    "wake": wake,
                    "empty_success": False,
                    "looked": False,
                }
            )
    lattice_rows = []
    for name in LATTICE_NAMES:
        lattice_rows.append(
            {
                "name": name,
                "disposition": "PARKED",
                "reason": (
                    "ABSENT as a named type in visionmcp; VMCP_LATTICE_DISPOSITION "
                    "says CONSOLIDATE onto WorldIR / EvidenceGraph / content_digest. "
                    "Do not build a parallel lattice in hawking/."
                ),
                "wake": _wake(
                    kind="DO_NOT_BUILD_NAMED_LATTICE",
                    required_symbol="visionmcp.worldir.canonical.content_digest",
                    predicate=(
                        "do not add this type; wake is an HCLI Call of the "
                        "existing visionmcp consolidating symbol after "
                        "VISIONMCP_SRC is importable"
                    ),
                    blocker=f"{name} is a roadmap name; building it would split the address space",
                    missing_dependency="VISIONMCP_SRC; reuse WorldIR / EvidenceGraph",
                ),
                "evidence_tier": "STATIC",
            }
        )
    return {
        "schema": DISPOSITION_SCHEMA,
        "subsystem": "vmcp",
        "what_it_is": (
            "Hawking's perception + visual / 3D / application-generation organ "
            "as specified in H-ROADMAP.md §8 and APPENDIX E. The live "
            "implementation is the foreign visionmcp package, not a tree in "
            "this repo. This sidecar is the compact E.14 surface bound to "
            "what can run here, plus an honest PARKED table for the rest."
        ),
        "not_this": (
            "crates/hawking-perception is a document-pipeline stub "
            "(document.inspect etc.) and is not VMCP. tools/theia is a "
            "different bounty domain (APPENDIX F)."
        ),
        "roadmap": {
            "section": "§8 VMCP / All-Seeing Eye",
            "appendix": "APPENDIX E",
            "compact_surface": list(NINE_ACTS),
        },
        "foreign_package": located,
        "receipts": {
            "census": {"path": CENSUS_REL, "path_taken": census_taken, "present": census is not None},
            "lattice": {"path": LATTICE_REL, "path_taken": lattice_taken, "present": lattice is not None},
            "integration": {
                "path": INTEGRATION_REL,
                "path_taken": integration_taken,
                "present": integration is not None,
            },
        },
        "core_mcp_tools": list(CORE_MCP_TOOLS),
        "acts": acts,
        "organs": _organ_table(located),
        "lattice": lattice_rows,
        "claim_boundary": CLAIM_BOUNDARY,
        "empty_success_rule": (
            "A PARKED act returns looked=false and a wake. It must not return "
            "an empty results/items list a verifier could read as 'I looked "
            "and there was nothing'."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


def see(
    path: str | os.PathLike[str] | None = None,
    *,
    max_bytes: int = 8_000_000,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """E.14 SEE. Default organ is the file eye; organ=pty selects the PTY eye.

    Both organs are CALLED (an import of hcli.agentos.vmcp is not a call site).
    """
    args = dict(arguments or {})
    if path is not None and "path" not in args:
        args["path"] = str(path)
    args["max_bytes"] = int(args.get("max_bytes") or max_bytes)
    organ = str(args.get("organ") or "").strip().lower()
    if organ in {"pty", "terminal"}:
        return pty_capture(arguments=args)
    return file_observe(path, max_bytes=int(args["max_bytes"]), arguments=args)


def hold(
    path: str | os.PathLike[str] | None = None,
    *,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Acquire + bind identity. Identity is the content digest, not the path."""
    observed = see(path, arguments=arguments)
    observed["act"] = "hold"
    if observed.get("present"):
        observed["asset_id"] = f"sha256:{observed['sha256']}"
        observed["bound"] = True
    else:
        observed["asset_id"] = None
        observed["bound"] = False
    return observed


def know(
    path: str | os.PathLike[str] | None = None,
    *,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Identity, not similarity. Two files with the same bytes are still two files."""
    args = dict(arguments or {})
    observed = see(path, arguments=args)
    observed["act"] = "know"
    observed["identity_kind"] = "content_sha256"
    other = args.get("other_path")
    if other:
        twin = see(other, arguments={"path": other, "max_bytes": args.get("max_bytes")})
        same_bytes = bool(
            observed.get("present")
            and twin.get("present")
            and observed.get("sha256")
            and observed["sha256"] == twin.get("sha256")
        )
        same_path = False
        if observed.get("path") and twin.get("path"):
            same_path = Path(str(observed["path"])).resolve() == Path(str(twin["path"])).resolve()
        observed["other"] = {
            "path": twin.get("path"),
            "sha256": twin.get("sha256"),
            "present": twin.get("present"),
        }
        observed["same_bytes"] = same_bytes
        observed["same_subject"] = same_path
        # Content identity is not subject identity — the replay canary.
        observed["subject_identity"] = "path"
        observed["content_identity"] = "sha256"
    else:
        observed["subject_identity"] = "path"
        observed["content_identity"] = "sha256"
    return observed


def check(
    path: str | os.PathLike[str] | None = None,
    *,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """E.14 CHECK. organ=tool_doctor profiles a real argv; otherwise re-hash."""
    args = dict(arguments or {})
    organ = str(args.get("organ") or "").strip().lower()
    if organ in {"tool_doctor", "doctor"}:
        if args.get("report"):
            return doctor_report(arguments=args)
        return doctor_profile(arguments=args)
    if (args.get("argv") or args.get("command") or args.get("tool")) and organ != "file":
        return doctor_profile(arguments=args)
    observed = see(path, arguments=args)
    claimed = args.get("expected_sha256") or args.get("sha256")
    observed["act"] = "check"
    if not observed.get("present"):
        observed["ok"] = False
        observed["reason"] = "target not present"
        return observed
    if not claimed:
        observed["ok"] = False
        observed["reason"] = "expected_sha256 is required; a missing claim is not a pass"
        return observed
    match = str(claimed) == str(observed.get("sha256"))
    observed["ok"] = match
    observed["expected_sha256"] = str(claimed)
    observed["reason"] = "match" if match else "EVIDENCE_STALE"
    return observed


def prove(
    path: str | os.PathLike[str] | None = None,
    *,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """E.14 PROVE. organ=behavior_lab runs BHV-01..23; otherwise the file canary.

    Mutation must change the digest (RED). Restoration must restore it (GREEN).
    A canary that cannot go RED is not a canary. Runs on a temp copy so the
    caller's file is not the mutation surface.
    """
    args = dict(arguments or {})
    organ = str(args.get("organ") or "").strip().lower()
    if organ in {"behavior_lab", "bhv"} or args.get("fixtures") is not None:
        return bhv_run_matrix(arguments=args)
    raw = path if path is not None else args.get("path")
    payload = b"vmcp-file-eye-canary\n"
    if raw:
        src = Path(str(raw))
        if src.is_file():
            payload = src.read_bytes()
    with tempfile.TemporaryDirectory(prefix="vmcp-prove-") as tmp:
        subject = Path(tmp) / "subject.bin"
        subject.write_bytes(payload)
        baseline = see(subject)
        subject.write_bytes(payload + b"\x00")
        mutated = see(subject)
        red = bool(
            baseline.get("sha256")
            and mutated.get("sha256")
            and baseline["sha256"] != mutated["sha256"]
        )
        subject.write_bytes(payload)
        restored = see(subject)
        green = bool(restored.get("sha256") == baseline.get("sha256") and red)
        return {
            "act": "prove",
            "status": "CONNECTED",
            "ok": bool(red and green),
            "red": red,
            "green": green,
            "baseline_sha256": baseline.get("sha256"),
            "mutated_sha256": mutated.get("sha256"),
            "restored_sha256": restored.get("sha256"),
            "empty_success": False,
            "looked": True,
            "evidence_tier": "FUNCTIONAL_SIM",
            "note": (
                "controlled mutation of a temp copy. RED is a digest change; "
                "GREEN is restoration. A always-GREEN canary is a failure."
            ),
        }


def _parked_response(act: str) -> dict[str, Any]:
    wake = _parked_act_wake(act)
    return {
        "act": act,
        "status": "PARKED",
        "ok": False,
        "looked": False,
        "empty_success": False,
        "results": None,
        "items": None,
        "wake": wake,
        "wake_condition": wake["predicate"],
        "missing_dependency": wake["missing_dependency"],
        "evidence_tier": "STATIC",
        "note": (
            f"{act} is not implemented in this repo. Returning PARKED with a "
            "wake rather than an empty collection a verifier could accept."
        ),
    }


def compact_surface(
    act: str,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """E.14 compact VMCP surface. CONNECTED acts run; PARKED acts wake.

    The former reachability-triage adapter recorded this symbol; current VMCP
    execution is owned directly by this compact surface.
    """
    name = str(act or "").strip().lower()
    args = arguments or {}
    if name in {"disposition", "status", ""}:
        doc = disposition()
        return {
            "act": "disposition",
            "status": "CONNECTED",
            "ok": True,
            "empty_success": False,
            "looked": True,
            "evidence_tier": "STATIC",
            "result": doc,
        }
    if name == "see":
        return see(arguments=args)
    if name == "hold":
        return hold(arguments=args)
    if name == "know":
        return know(arguments=args)
    if name == "check":
        return check(arguments=args)
    if name == "prove":
        return prove(arguments=args)
    if name in PARKED_ACTS:
        return _parked_response(name)
    return {
        "act": name or None,
        "status": "UNKNOWN_ACT",
        "ok": False,
        "looked": False,
        "empty_success": False,
        "known_acts": list(NINE_ACTS) + ["disposition"],
        "evidence_tier": "STATIC",
    }


def selftest() -> dict[str, Any]:
    doc = disposition()
    parked = [a for a in doc["acts"] if a["disposition"] == "PARKED"]
    connected = [a for a in doc["acts"] if a["disposition"] == "CONNECTED"]
    if {a["act"] for a in doc["acts"]} != set(NINE_ACTS):
        raise AssertionError("disposition dropped a Nine Act")
    for row in parked:
        wake = row.get("wake") or {}
        if not wake.get("predicate") or not wake.get("missing_dependency"):
            raise AssertionError(f"{row['act']} PARKED without a wake")
        if wake.get("required_kind") != WAKE_REQUIRED_KIND:
            raise AssertionError(f"{row['act']} wake required_kind is not call")
    if {a["act"] for a in connected} != CONNECTED_ACTS:
        raise AssertionError("connected acts drifted")
    with tempfile.TemporaryDirectory(prefix="vmcp-selftest-") as tmp:
        p = Path(tmp) / "x.txt"
        p.write_text("hello-vmcp\n", encoding="utf-8")
        seen = see(p)
        if not seen.get("present") or not seen.get("sha256"):
            raise AssertionError(f"see failed on a real file: {seen}")
        missing = see(Path(tmp) / "nope.txt")
        if missing.get("present") or "TARGET_ABSENT" not in (missing.get("limitations") or []):
            raise AssertionError(f"absent file was not TARGET_ABSENT: {missing}")
        if missing.get("empty_success"):
            raise AssertionError("absent file looked like empty success")
        held = hold(p)
        if held.get("asset_id") != f"sha256:{seen['sha256']}":
            raise AssertionError("hold did not bind the digest")
        checked = check(p, arguments={"expected_sha256": seen["sha256"]})
        if not checked.get("ok"):
            raise AssertionError("check refused a matching digest")
        stale = check(p, arguments={"expected_sha256": "0" * 64})
        if stale.get("ok"):
            raise AssertionError("check accepted a stale digest")
        proof = prove(p)
        if not (proof.get("red") and proof.get("green") and proof.get("ok")):
            raise AssertionError(f"prove canary did not go RED then GREEN: {proof}")
        opened = compact_surface("open", {})
        if opened.get("status") != "PARKED" or opened.get("looked") or opened.get("empty_success"):
            raise AssertionError(f"open must PARK without looking: {opened}")
        if not (opened.get("wake") or {}).get("predicate"):
            raise AssertionError("open PARKED without a wake predicate")
        classified = file_observe(p)
        if classified.get("kind") not in {"text", "script"}:
            raise AssertionError(f"file eye did not classify a text file: {classified}")
        echo = doctor_profile(["/bin/echo", "vmcp-doctor"])
        if echo.get("exit_code") != 0 or "vmcp-doctor" not in (echo.get("stdout") or ""):
            raise AssertionError(f"tool doctor did not profile /bin/echo: {echo}")
        if (echo.get("tool_receipt") or {}).get("schema") != "hawking.vmcp.tool_receipt.v1":
            raise AssertionError(f"tool doctor missing E.4 receipt: {echo}")
        pty = pty_capture(argv=["/bin/echo", "vmcp-pty"])
        if pty.get("empty_success"):
            raise AssertionError(f"pty empty success: {pty}")
        if pty.get("used_real_pty") and not pty.get("ok"):
            raise AssertionError(f"real pty ran but failed: {pty}")
        if not pty.get("used_real_pty"):
            if pty.get("status") != "PARKED" or "PTY_OPEN_DENIED" not in (pty.get("limitations") or []):
                raise AssertionError(f"blocked pty must PARK with PTY_OPEN_DENIED: {pty}")
        matrix = bhv_run_matrix()
        if matrix.get("n") != 23:
            raise AssertionError(f"behavior lab did not run 23 fixtures: {matrix.get('n')}")
        if matrix.get("n_ok") != 23:
            raise AssertionError(
                "behavior lab residuals: " + ",".join(matrix.get("residuals") or [])
            )
        if (matrix.get("verdict") or {}).get("outcome") != "PASS":
            raise AssertionError(f"tabula verdict not PASS: {matrix.get('verdict')}")
    organs = {row["id"]: row for row in doc["organs"]}
    for oid in ("vmcp.file_eye", "vmcp.tool_doctor", "vmcp.behavior_lab"):
        if organs[oid]["disposition"] != "CONNECTED":
            raise AssertionError(f"{oid} not CONNECTED")
        if organs[oid].get("execution") != "REAL":
            raise AssertionError(f"{oid} not REAL execution")
    return {
        "ok": True,
        "n_connected_acts": len(connected),
        "n_parked_acts": len(parked),
        "n_organs": len(doc["organs"]),
        "verb_count": len(NINE_ACTS),
        "file_eye_kind": classified.get("kind"),
        "tool_doctor_exit": echo.get("exit_code"),
        "pty_used_real": bool(pty.get("used_real_pty")),
        "behavior_lab_n_ok": matrix.get("n_ok"),
        "behavior_lab_verdict": (matrix.get("verdict") or {}).get("outcome"),
    }


def build() -> Path:
    proof = selftest()
    doc = disposition()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "BUILT_NOT_PROMOTED",
        "promoted": False,
        "purpose": (
            "Dispose VMCP against source and roadmap. Promote the local file "
            "classifier, tool doctor and behavior-lab fixture matrix to REAL "
            "execution behind the same nine-act compact surface. Park PTY if "
            "openpty is denied, and park organs that need visionmcp or "
            "absent hardware."
        ),
        "disposition": doc,
        "selftest": proof,
        "resident_callable": {
            "entry_point": "python3 hcli/agentos/vmcp/disposition.py --build",
            "invoke": (
                "python3 hcli/agentos/vmcp/disposition.py --build "
                "--args '{\"act\":\"see\",\"path\":\"...\"}'"
            ),
            "symbol": "hcli.agentos.vmcp.disposition.compact_surface",
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "weights_modified": False,
        "verb_count_before": 9,
        "verb_count_after": len(NINE_ACTS),
        "promoted_organs": [
            "vmcp.file_eye",
            "vmcp.tool_doctor",
            "vmcp.behavior_lab",
        ],
    }
    out = write_receipt(RECEIPT, payload, RECORDED_BY)
    echo_path = "/bin/echo"
    file_run = file_observe(echo_path)
    doctor_run = doctor_profile([echo_path, "perception-depth"])
    pty_run = pty_capture(argv=[echo_path, "perception-depth-pty"])
    bhv_run = bhv_run_matrix(fixtures=["BHV-02", "BHV-09", "BHV-21"])
    before = {
        "vmcp.file_eye": {"disposition": "CONNECTED", "evidence_tier": "FUNCTIONAL_SIM", "execution": "hash-only"},
        "vmcp.pty_eye": {"disposition": "PARKED", "evidence_tier": "STATIC"},
        "vmcp.behavior_lab": {"disposition": "PARKED", "evidence_tier": "STATIC"},
        "vmcp.tool_doctor": {"disposition": "PARKED", "evidence_tier": "STATIC"},
    }
    depth = {
        "schema": "hawking.future.vmcp_perception_depth.v1",
        "version": 1,
        "status": "BUILT_NOT_PROMOTED",
        "promoted": False,
        "purpose": "Per-organ before/after plus the real command and output of each promoted organ.",
        "verb_count_before": 9,
        "verb_count_after": len(NINE_ACTS),
        "compact_surface": list(NINE_ACTS),
        "selftest": proof,
        "before": before,
        "after": {
            row["id"]: {
                "disposition": row.get("disposition"),
                "evidence_tier": row.get("evidence_tier"),
                "execution": row.get("execution"),
                "symbol": row.get("symbol"),
                "wake": row.get("wake"),
            }
            for row in doc["organs"]
            if row["id"] in before
        },
        "runs": {
            "file_eye": {
                "command": f"python3 hcli/agentos/vmcp/disposition.py --act see --path {echo_path}",
                "invoke": (
                "python3 hcli/agentos/vmcp/disposition.py --build "
                    f"--args '{{\"act\":\"see\",\"path\":\"{echo_path}\"}}'"
                ),
                "kind": file_run.get("kind"),
                "container": file_run.get("container"),
                "sha256": file_run.get("sha256"),
                "present": file_run.get("present"),
                "execution": file_run.get("execution"),
                "cpu": (file_run.get("classification") or {}).get("cpu")
                or (file_run.get("classification") or {}).get("slices"),
            },
            "tool_doctor": {
                "command": (
                    "python3 hcli/agentos/vmcp/disposition.py --act check --organ tool_doctor "
                    "--args '{\"argv\":[\"/bin/echo\",\"perception-depth\"]}'"
                ),
                "invoke": (
                "python3 hcli/agentos/vmcp/disposition.py --build "
                    "--args '{\"act\":\"check\",\"organ\":\"tool_doctor\","
                    "\"argv\":[\"/bin/echo\",\"perception-depth\"]}'"
                ),
                "exit_code": doctor_run.get("exit_code"),
                "stdout": (doctor_run.get("stdout") or "").strip(),
                "elapsed_ms": doctor_run.get("performance_ms"),
                "receipt_schema": (doctor_run.get("tool_receipt") or {}).get("schema"),
                "execution": doctor_run.get("execution"),
            },
            "behavior_lab": {
                "command": (
                    "python3 hcli/agentos/vmcp/disposition.py --act prove --organ behavior_lab "
                    "--args '{\"fixtures\":[\"BHV-02\",\"BHV-09\",\"BHV-21\"]}'"
                ),
                "invoke": (
                "python3 hcli/agentos/vmcp/disposition.py --build "
                    "--args '{\"act\":\"prove\",\"organ\":\"behavior_lab\","
                    "\"fixtures\":[\"BHV-02\",\"BHV-09\",\"BHV-21\"]}'"
                ),
                "n": bhv_run.get("n"),
                "n_ok": bhv_run.get("n_ok"),
                "verdict": (bhv_run.get("verdict") or {}).get("outcome"),
                "scores": bhv_run.get("scores"),
                "ids": [r.get("id") for r in (bhv_run.get("fixtures") or [])],
                "execution": bhv_run.get("execution"),
                "laboratory_profile_used": bhv_run.get("laboratory_profile_used"),
            },
            "pty_eye": {
                "command": (
                    "python3 hcli/agentos/vmcp/disposition.py --act see --organ pty "
                    "--args '{\"argv\":[\"/bin/echo\",\"perception-depth-pty\"]}'"
                ),
                "status": pty_run.get("status"),
                "used_real_pty": pty_run.get("used_real_pty"),
                "limitations": pty_run.get("limitations"),
                "blocker": (pty_run.get("wake") or {}).get("blocker"),
                "promoted": False,
            },
        },
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "weights_modified": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_receipt("VMCP_PERCEPTION_DEPTH.json", depth, RECORDED_BY)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--disposition", action="store_true")
    ap.add_argument("--act")
    ap.add_argument("--path")
    ap.add_argument("--organ")
    ap.add_argument("--args", default="", help="JSON object merged into compact_surface arguments")
    args = ap.parse_args()
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        return 0
    if args.act:
        payload: dict[str, Any] = {}
        if args.args:
            loaded = json.loads(args.args)
            if isinstance(loaded, dict):
                payload.update(loaded)
        if args.path:
            payload["path"] = args.path
        if args.organ:
            payload["organ"] = args.organ
        print(json.dumps(compact_surface(args.act, payload), indent=2, sort_keys=True))
        return 0
    if args.disposition:
        print(json.dumps(disposition(), indent=2, sort_keys=True))
        return 0
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
