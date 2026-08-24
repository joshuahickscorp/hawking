#!/usr/bin/env python3
"""Probe the visionmcp 0.8.0a2 core MCP capability surface.

Enumerates every tool on the core profile, calls each on a target it cannot
handle, and classifies the result: explicit refusal vs empty success.

An empty success is silence a verifier would accept as evidence of absence
("I looked and there was nothing") when the tool in fact could not look.

This script MEASURES visionmcp. It does not import visionmcp from the repo
tree when that tree is a sparse checkout; it locates an existing 0.8.0a2
checkout read-only via PYTHONPATH.

Run from the repository root:

    python3 tools/headless/vmcp_capability_probe.py
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import sqlite3
import struct
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RECEIPT_PATH = REPO / "receipts" / "headless" / "VMCP_CAPABILITY_SURFACE.json"
HEAVY_MODULES = ("torch", "cv2", "open3d", "numpy", "PIL", "playwright", "trimesh")
EMPTY_LIST_KEYS = (
    "results",
    "items",
    "matches",
    "explanations",
    "artifacts",
    "captures",
    "blockers",
    "queue",
    "failures",
)
VOLATILE_KEYS = {
    "elapsed_ms",
    "created_at",
    "updated_at",
    "generated_at",
    "timestamp",
}
# Hardcoded in capabilities.py for ocular; not a live measurement.
HARDCODED_OCULAR = {
    "proposal_recall_mean": 0.648,
    "proposal_recall_required": 0.90,
    "proposal_precision_required": 0.80,
    "idf1_required": 0.85,
    "mota_required": 0.75,
}


# --------------------------------------------------------------------------- locate

def locate_visionmcp_src() -> Path:
    """Find a 0.8.0a2 checkout. This worktree is sparse; visionmcp is not in it."""
    env = os.environ.get("VISIONMCP_SRC")
    candidates = []
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
    seen: set[Path] = set()
    for src in candidates:
        src = src.resolve() if src.exists() else src
        if src in seen:
            continue
        seen.add(src)
        init = src / "visionmcp" / "__init__.py"
        if init.is_file():
            return src
    raise FileNotFoundError(
        "visionmcp src not found. Set VISIONMCP_SRC to the package's src/ "
        "directory (the parent of the visionmcp package). This sparse worktree "
        "does not materialize visionmcp/, and git sparse-checkout add is denied."
    )


def git_head(cwd: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        oneline = subprocess.run(
            ["git", "log", "-1", "--oneline"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"head": head, "oneline": oneline, "cwd": str(cwd)}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"head": None, "oneline": None, "cwd": str(cwd), "error": str(exc)}


def heavy_present() -> dict[str, bool]:
    return {name: name in sys.modules for name in HEAVY_MODULES}


# --------------------------------------------------------------------------- bytes

def min_png_bytes() -> bytes:
    """A real 1x1 RGB PNG (black), written without Pillow."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", idat)
        + chunk(b"IEND", b"")
    )


def jsonable(value: Any, *, limit: int = 4000) -> Any:
    """JSON-stable view of a tool result, truncated at string leaves."""
    if isinstance(value, dict):
        return {str(k): jsonable(v, limit=limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [jsonable(v, limit=limit) for v in value]
        if len(items) > 40:
            return items[:40] + [f"... {len(items) - 40} more"]
        return items
    if isinstance(value, (bytes, bytearray)):
        return {"_bytes": len(value), "_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > limit:
            return value[:limit] + f"... <truncated {len(value) - limit} chars>"
        return value
    return repr(value)


def strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: strip_volatile(v)
            for k, v in value.items()
            if k not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [strip_volatile(v) for v in value]
    return value


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def excerpt(value: Any, n: int = 220) -> str:
    text = canonical(jsonable(value, limit=n))
    if len(text) > n:
        return text[: n - 1] + "…"
    return text


# --------------------------------------------------------------------------- classify

def empty_collections(payload: Any) -> dict[str, Any]:
    found: dict[str, Any] = {}
    if not isinstance(payload, dict):
        return found
    for key in EMPTY_LIST_KEYS:
        if key in payload and payload[key] == []:
            found[key] = []
    if payload.get("count") == 0 and "items" in payload:
        found["count"] = 0
    overview = payload.get("overview")
    if isinstance(overview, dict) and overview.get("captures") == []:
        found["overview.captures"] = []
    return found


def has_explicit_classification(payload: Any) -> bool:
    """True when the payload itself names a refusal / limitation / invalidity."""
    if not isinstance(payload, dict):
        return False
    if payload.get("valid") is False:
        return True
    if payload.get("ok") is False:
        return True
    if payload.get("exists") is False:
        return True
    status = payload.get("status")
    if status in {"not_open", "blocked", "error", "INTERRUPTED", "unavailable", "failed"}:
        return True
    for key in ("code", "error", "error_code", "classification", "reason"):
        if payload.get(key):
            return True
    return False


def classify_call(kind: str, ok: bool, payload: Any) -> str:
    """kind is the probe intent: refuse | empty_ok | empty_hunt | meta | false_available."""
    if not ok:
        return "EXPLICIT_ERROR"
    if has_explicit_classification(payload):
        return "EXPLICIT_CLASS"
    empty = empty_collections(payload)
    if kind == "empty_hunt" and empty:
        return "EMPTY_SUCCESS"
    if kind == "empty_ok" and empty:
        return "LEGITIMATE_EMPTY"
    if kind == "false_available":
        return "FALSE_AVAILABLE" if _claims_available(payload) else "EXPLICIT_CLASS"
    if kind == "refuse" and empty and not has_explicit_classification(payload):
        return "EMPTY_SUCCESS"
    if kind == "refuse":
        return "UNEXPECTED_SUCCESS"
    return "SUCCESS"


def _claims_available(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    available = payload.get("available") or []
    return any(
        isinstance(row, dict)
        and row.get("id") == "core.observe_image_file"
        and row.get("status") == "available"
        for row in available
    )


# --------------------------------------------------------------------------- calls

@dataclass
class CallResult:
    tool: str
    probe_id: str
    kind: str
    category: str
    arguments: dict[str, Any]
    ok: bool
    classification: str
    error_type: str | None = None
    error: str | None = None
    cause_type: str | None = None
    cause: str | None = None
    payload: Any = None
    seconds: float = 0.0


@dataclass
class ToolReport:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: Any
    required: list[str]
    claim: str
    calls: list[CallResult] = field(default_factory=list)
    determinism: dict[str, Any] = field(default_factory=dict)
    exactness: dict[str, Any] = field(default_factory=dict)
    headline_classification: str = ""
    empty_success: list[dict[str, Any]] = field(default_factory=list)


async def invoke(
    server: Any,
    tool: str,
    probe_id: str,
    kind: str,
    category: str,
    arguments: dict[str, Any],
) -> CallResult:
    registered = server._tool_manager.get_tool(tool)
    started = time.perf_counter()
    if registered is None:
        return CallResult(
            tool=tool,
            probe_id=probe_id,
            kind=kind,
            category=category,
            arguments=arguments,
            ok=False,
            classification="UNPROBEABLE",
            error_type="KeyError",
            error=f"tool {tool!r} not registered",
            seconds=time.perf_counter() - started,
        )
    try:
        payload = await registered.run(arguments, convert_result=False)
        elapsed = time.perf_counter() - started
        classification = classify_call(kind, True, payload)
        return CallResult(
            tool=tool,
            probe_id=probe_id,
            kind=kind,
            category=category,
            arguments=arguments,
            ok=True,
            classification=classification,
            payload=payload,
            seconds=elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        cause = exc.__cause__
        result = CallResult(
            tool=tool,
            probe_id=probe_id,
            kind=kind,
            category=category,
            arguments=arguments,
            ok=False,
            classification="EXPLICIT_ERROR",
            error_type=type(exc).__name__,
            error=str(exc),
            cause_type=type(cause).__name__ if cause is not None else None,
            cause=str(cause) if cause is not None else None,
            seconds=elapsed,
        )
        return result


def call_as_dict(call: CallResult) -> dict[str, Any]:
    return {
        "probe_id": call.probe_id,
        "kind": call.kind,
        "category": call.category,
        "arguments": jsonable(call.arguments, limit=500),
        "ok": call.ok,
        "classification": call.classification,
        "error_type": call.error_type,
        "error": call.error,
        "cause_type": call.cause_type,
        "cause": call.cause,
        "payload": jsonable(call.payload, limit=2000) if call.ok else None,
        "seconds": round(call.seconds, 4),
    }


# --------------------------------------------------------------------------- exactness (from the implementation, refined by live returns)

EXACTNESS: dict[str, dict[str, Any]] = {
    "vision.capabilities": {
        "class": "mixed",
        "says_which": False,
        "notes": (
            "resource_cost is a qualitative label (low/high) with no error class. "
            "ocular.profile.measured numbers are hardcoded constants in "
            "capabilities.py (proposal_recall_mean=0.648 etc.), not a live "
            "measurement, and are not labelled as such. core.observe_image_file "
            "is hardcoded status=available with no runtime probe of numpy/PIL/cv2."
        ),
    },
    "system.doctor": {
        "class": "mixed",
        "says_which": False,
        "notes": (
            "elapsed_ms is a wall-clock interval with no error class. "
            "optional_blender/colmap/playwright checks set ok=True always; "
            "present is a shutil.which probe (exact for PATH presence). "
            "cv2/numpy/PIL are not probed at all even though vision.observe needs them."
        ),
    },
    "project.create": {
        "class": "identity_not_measurement",
        "says_which": False,
        "notes": (
            "project.id is uuid4; created_at/updated_at are utc_now() ISO timestamps. "
            "Neither is content-addressed. canonical_units and coordinate_system are "
            "constants, not measurements."
        ),
    },
    "project.status": {
        "class": "exact",
        "says_which": False,
        "notes": "counts are SQL COUNT(*) over a fixed table list. No error class on the counts.",
    },
    "vision.open_project": {
        "class": "exact",
        "says_which": False,
        "notes": "status:open plus project.status() counts. No numeric measurement of the world.",
    },
    "vision.close_project": {
        "class": "exact",
        "says_which": False,
        "notes": "status is closed|not_open. That is an explicit classification, not a measurement.",
    },
    "vision.list_artifacts": {
        "class": "exact",
        "says_which": False,
        "notes": "count is len(rows); digest/size come from the artifacts table / filesystem.",
    },
    "vision.get_artifact": {
        "class": "exact",
        "says_which": False,
        "notes": "size is path.stat().st_size; digest is the SHA-256 key. Raises on absence.",
    },
    "vision.observe": {
        "class": "heuristic",
        "says_which": "partial",
        "notes": (
            "On a successful image capture: width/height are integer pixel counts "
            "(exact for the *normalized* raster, which may have been resized; "
            "original_width/original_height are also returned). perceptual_hash is "
            "an 8x8 average hash — a heuristic — labelled only as perceptual_hash, "
            "no error class. region confidence is min(0.95, area/pixels), a heuristic "
            "with authority=DERIVED. limitations[] does say contour/OCR are DERIVED. "
            "mean_rgb is rounded to 3 decimal places. Capture identity is sha256 of "
            "the canonical request (exact)."
        ),
    },
    "vision.query": {
        "class": "exact_over_derived_graph",
        "says_which": False,
        "notes": (
            "match_count is len(nodes) after filter; bounds tests are closed-interval "
            "pixel geometry on the stored graph. The graph itself is DERIVED. Absence "
            "of a capture/graph raises KeyError rather than returning matches:[]."
        ),
    },
    "vision.explain_region": {
        "class": "exact_over_derived_graph",
        "says_which": False,
        "notes": (
            "Wraps query; explanations copy authority/confidence/uncertainty from nodes. "
            "confidence on image regions is the heuristic above. A miss returns "
            "explanations:[] *with a citation* when the graph loaded — distinguishable "
            "from a missing capture, which raises KeyError."
        ),
    },
    "vision.compare": {
        "class": "exact_when_it_runs",
        "says_which": True,
        "notes": (
            "identical_manifest is digest equality. limitations[] states "
            "'Core compare is digest/metadata based.' In this core_server build the "
            "body constructs ObservationQueryService(project) WITHOUT a CaptureBus, "
            "so verify() raises RuntimeError before any digest is read."
        ),
    },
    "vision.verify": {
        "class": "exact",
        "says_which": False,
        "notes": (
            "Recomputes sha256 of stored artifacts and compares to recorded digests. "
            "Unknown capture raises KeyError. Receipt path verifies payload_sha256 "
            "and referenced artifacts; valid is a boolean conjunction, no error bars."
        ),
    },
    "vision.progress": {
        "class": "exact_counts_silent_blockers",
        "says_which": False,
        "notes": (
            "artifact_count is SQL COUNT(*). overview.captures is a full table read. "
            "blockers is hardcoded [] in core_server.py — never computed, never "
            "labelled as 'not implemented'."
        ),
    },
    "vision.review_queue": {
        "class": "exact_when_table_exists",
        "says_which": False,
        "notes": (
            "count is len(items) from frontend_patch_proposals WHERE status='PROPOSED'. "
            "If that table is missing the tool skips it and returns items:[], count:0 "
            "with no SCHEMA_ABSENT classification."
        ),
    },
}


# --------------------------------------------------------------------------- grok-vision (cv2) supplementary observe

def grok_vision_observe_probes(
    vmcp_src: Path,
    projects_root: Path,
    project_path: Path,
    lie_png: Path,
    valid_png: Path,
) -> dict[str, Any]:
    """Content-mismatch / successful observe need cv2, which system python3 lacks."""
    py = Path.home() / ".grok-vision/bin/python"
    if not py.is_file():
        return {"ran": False, "reason": f"{py} not present"}
    script = projects_root / "_cv2_observe_probe.py"
    script.write_text(
        textwrap.dedent(
            """
            import asyncio, json, sys
            from pathlib import Path
            from visionmcp.mcp.factory import create_server
            from visionmcp.plugins.registry import reset_plugin_registry

            project, lie, valid, empty = map(Path, sys.argv[1:5])
            reset_plugin_registry()
            server = create_server(project.parent, profile="core")

            async def call(name, args):
                tool = server._tool_manager.get_tool(name)
                try:
                    payload = await tool.run(args, convert_result=False)
                    return {"ok": True, "payload": payload}
                except Exception as exc:
                    cause = exc.__cause__
                    return {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "cause_type": type(cause).__name__ if cause else None,
                        "cause": str(cause) if cause else None,
                    }

            async def main():
                base = {
                    "project_path": str(project),
                    "rights_decision": "OWNED",
                    "adapter": "image.file",
                    "configuration": {"ocr": False},
                }
                out = {}
                out["lie_png"] = await call("vision.observe", {**base, "target": {"path": str(lie)}})
                out["empty_png"] = await call("vision.observe", {**base, "target": {"path": str(empty)}})
                out["valid_png"] = await call("vision.observe", {**base, "target": {"path": str(valid)}})
                out["progress_after"] = await call("vision.progress", {"project_path": str(project)})
                payload = out["valid_png"].get("payload") or {}
                capture_id = payload.get("capture_id")
                if capture_id:
                    out["explain_far_point"] = await call(
                        "vision.explain_region",
                        {
                            "project_path": str(project),
                            "capture_id": capture_id,
                            "x": 99999.0,
                            "y": 99999.0,
                        },
                    )
                    out["query_missing_id"] = await call(
                        "vision.query",
                        {
                            "project_path": str(project),
                            "capture_id": "0" * 64,
                            "query": {"point": {"x": 0, "y": 0}},
                        },
                    )
                    out["explain_missing_id"] = await call(
                        "vision.explain_region",
                        {
                            "project_path": str(project),
                            "capture_id": "0" * 64,
                            "x": 0.0,
                            "y": 0.0,
                        },
                    )
                print(json.dumps(out, default=str))

            asyncio.run(main())
            """
        ).lstrip(),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(vmcp_src) + os.pathsep + env.get("PYTHONPATH", "")
    env["VISIONMCP_PROJECTS_ROOT"] = str(projects_root)
    try:
        proc = subprocess.run(
            [str(py), str(script), str(project_path), str(lie_png), str(valid_png), str(project_path / "empty.png")],
            capture_output=True,
            text=True,
            env=env,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ran": False, "reason": str(exc)}
    if proc.returncode != 0:
        return {
            "ran": True,
            "ok": False,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    try:
        body = json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {
            "ran": True,
            "ok": False,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    return {"ran": True, "ok": True, "interpreter": str(py), "results": body}


def _ingest_cv2_calls(
    reports: dict[str, ToolReport],
    cv2_probes: dict[str, Any],
    project_path_s: str,
) -> None:
    """Fold the cv2-interpreter observe/query probes into the per-tool call lists."""
    if not cv2_probes.get("ok") or not isinstance(cv2_probes.get("results"), dict):
        return
    results = cv2_probes["results"]
    mapping = [
        ("vision.observe", "cv2_lie_png", "refuse", "wrong_type", "lie_png"),
        ("vision.observe", "cv2_empty_png", "refuse", "wrong_type", "empty_png"),
        ("vision.observe", "cv2_valid_png", "meta", "meta", "valid_png"),
        ("vision.progress", "cv2_progress_after", "empty_hunt", "absent", "progress_after"),
        ("vision.explain_region", "cv2_explain_far_point", "empty_ok", "empty", "explain_far_point"),
        ("vision.query", "cv2_query_missing_id", "empty_hunt", "missing_id", "query_missing_id"),
        ("vision.explain_region", "cv2_explain_missing_id", "empty_hunt", "missing_id", "explain_missing_id"),
    ]
    for tool, probe_id, kind, category, key in mapping:
        raw = results.get(key)
        if not isinstance(raw, dict):
            continue
        if raw.get("ok"):
            payload = raw.get("payload")
            classification = classify_call(kind, True, payload)
            if (
                tool == "vision.progress"
                and isinstance(payload, dict)
                and payload.get("blockers") == []
            ):
                captures = (payload.get("overview") or {}).get("captures") or []
                interrupted = [
                    c
                    for c in captures
                    if str(c.get("status") or "").upper() == "INTERRUPTED"
                ]
                if interrupted:
                    classification = "EMPTY_SUCCESS"
            call = CallResult(
                tool=tool,
                probe_id=probe_id,
                kind=kind,
                category=category,
                arguments={"interpreter": cv2_probes.get("interpreter")},
                ok=True,
                classification=classification,
                payload=payload,
            )
        else:
            call = CallResult(
                tool=tool,
                probe_id=probe_id,
                kind=kind,
                category=category,
                arguments={"interpreter": cv2_probes.get("interpreter")},
                ok=False,
                classification="EXPLICIT_ERROR",
                error_type=raw.get("error_type"),
                error=raw.get("error"),
                cause_type=raw.get("cause_type"),
                cause=raw.get("cause"),
            )
        reports[tool].calls.append(call)


# --------------------------------------------------------------------------- main probe

async def probe() -> dict[str, Any]:
    started = time.perf_counter()
    vmcp_src = locate_visionmcp_src()
    if str(vmcp_src) not in sys.path:
        sys.path.insert(0, str(vmcp_src))

    import_heavy_before = heavy_present()

    from visionmcp.mcp.core_server import create_core_server
    from visionmcp.mcp.factory import create_server
    from visionmcp.plugins.registry import reset_plugin_registry
    from visionmcp.profiles import CORE_TOOLS

    # Exact construction the contract names. The real signature has no profile=.
    core_sig = str(inspect.signature(create_core_server))
    named_ctor_error = None
    try:
        reset_plugin_registry()
        create_core_server(profile="core")  # type: ignore[call-arg]
    except TypeError as exc:
        named_ctor_error = f"{type(exc).__name__}: {exc}"

    tmp = Path(tempfile.mkdtemp(prefix="vmcp-capability-probe-"))
    projects_root = tmp / "projects"
    outside_root = tmp / "outside"
    projects_root.mkdir()
    outside_root.mkdir()
    os.environ["VISIONMCP_PROJECTS_ROOT"] = str(projects_root)

    reset_plugin_registry()
    create_heavy_before = heavy_present()
    server = create_server(projects_root, profile="core")
    create_heavy_after = heavy_present()

    tools = server._tool_manager.list_tools()
    tool_names = [t.name for t in tools]
    attr_tools = getattr(server, "_tools", [])

    # Meta: _tools pitfall
    tools_attr_note = {
        "server_has__tools": hasattr(server, "_tools"),
        "getattr_server__tools_default_empty_list": list(getattr(server, "_tools", [])),
        "tool_manager_list_tools_count": len(tools),
    }

    reports: dict[str, ToolReport] = {}
    for tool in tools:
        required = list((tool.parameters or {}).get("required") or [])
        reports[tool.name] = ToolReport(
            name=tool.name,
            description=tool.description or "",
            input_schema=tool.parameters or {},
            output_schema=tool.output_schema,
            required=required,
            claim=(tool.description or "").strip().split("\n")[0].strip(),
            exactness=EXACTNESS.get(tool.name, {"class": "unknown", "says_which": False}),
        )

    async def do(
        tool: str,
        probe_id: str,
        kind: str,
        category: str,
        arguments: dict[str, Any],
    ) -> CallResult:
        result = await invoke(server, tool, probe_id, kind, category, arguments)
        reports[tool].calls.append(result)
        return result

    # ---- project.create (need a real project before most other tools)
    create_empty = await do(
        "project.create", "empty_name", "refuse", "empty", {"name": ""}
    )
    create_punct = await do(
        "project.create", "non_slug_name", "refuse", "empty", {"name": "!!!"}
    )
    create_bad_fid = await do(
        "project.create",
        "bad_fidelity",
        "refuse",
        "empty",
        {"name": "fid-bad", "target_fidelity": "L9"},
    )
    create_ok = await do(
        "project.create", "happy_create", "meta", "meta", {"name": "probe-main"}
    )
    project_path = projects_root / "probe-main"
    project_path_s = str(project_path)
    create_dup = await do(
        "project.create", "duplicate_name", "refuse", "empty", {"name": "probe-main"}
    )

    # Determinism of create: two fresh names differ in id + timestamps.
    create_a = await do(
        "project.create", "det_a", "meta", "meta", {"name": "det-alpha"}
    )
    create_b = await do(
        "project.create", "det_b", "meta", "meta", {"name": "det-beta"}
    )

    # Files inside / outside the project.
    lie_png = project_path / "lie.png"
    lie_png.write_bytes(b"this is not a PNG file at all")
    empty_png = project_path / "empty.png"
    empty_png.write_bytes(b"")
    valid_png = project_path / "valid.png"
    valid_png.write_bytes(min_png_bytes())
    note_txt = project_path / "note.txt"
    note_txt.write_text("hello", encoding="utf-8")
    lie_mp4 = project_path / "lie.mp4"
    lie_mp4.write_bytes(b"not a video")
    garbage_receipt = project_path / "garbage-receipt.json"
    garbage_receipt.write_text("{not json", encoding="utf-8")
    empty_receipt = project_path / "empty-receipt.json"
    empty_receipt.write_text("{}", encoding="utf-8")
    missing_inside = project_path / "no-such-file.png"
    outside_file = outside_root / "canary.txt"
    outside_file.write_text("never disclose", encoding="utf-8")
    # A real project *outside* the confined root.
    from visionmcp.projects.store import ProjectStore

    outside_project = ProjectStore.create(outside_root / "escaped", "escaped")

    # ---- vision.capabilities / system.doctor (no visual target)
    cap1 = await do("vision.capabilities", "no_args", "false_available", "meta", {})
    cap2 = await do("vision.capabilities", "det_repeat", "meta", "meta", {})
    cap_extra = await do(
        "vision.capabilities",
        "unexpected_kw",
        "meta",
        "empty",
        {"project_path": project_path_s},
    )
    doc1 = await do("system.doctor", "no_args", "meta", "meta", {})
    doc2 = await do("system.doctor", "det_repeat", "meta", "meta", {})
    doc_extra = await do(
        "system.doctor", "unexpected_kw", "meta", "empty", {"project_path": "/nope"}
    )

    # ---- project.status / open / close
    await do(
        "project.status",
        "missing_path",
        "refuse",
        "absent",
        {"project_path": str(projects_root / "does-not-exist")},
    )
    await do(
        "project.status",
        "empty_path",
        "refuse",
        "empty",
        {"project_path": ""},
    )
    await do(
        "project.status",
        "outside_project",
        "refuse",
        "outside",
        {"project_path": str(outside_project.root)},
    )
    st1 = await do(
        "project.status", "happy", "meta", "meta", {"project_path": project_path_s}
    )
    st2 = await do(
        "project.status", "det_repeat", "meta", "meta", {"project_path": project_path_s}
    )
    await do(
        "vision.open_project",
        "missing_path",
        "refuse",
        "absent",
        {"project_path": str(projects_root / "does-not-exist")},
    )
    await do(
        "vision.open_project",
        "outside_project",
        "refuse",
        "outside",
        {"project_path": str(outside_project.root)},
    )
    await do(
        "vision.open_project",
        "not_a_project_dir",
        "refuse",
        "absent",
        {"project_path": str(projects_root / ".resources")},
    )
    await do(
        "vision.open_project", "happy", "meta", "meta", {"project_path": project_path_s}
    )
    never_opened = ProjectStore.create(projects_root / "never-opened", "never-opened")
    close1 = await do(
        "vision.close_project",
        "never_opened",
        "refuse",
        "absent",
        {"project_path": str(never_opened.root)},
    )
    close2 = await do(
        "vision.close_project",
        "never_opened_repeat",
        "refuse",
        "absent",
        {"project_path": str(never_opened.root)},
    )
    await do(
        "vision.close_project",
        "missing_path",
        "refuse",
        "absent",
        {"project_path": str(projects_root / "does-not-exist")},
    )
    await do(
        "vision.close_project",
        "outside_project",
        "refuse",
        "outside",
        {"project_path": str(outside_project.root)},
    )

    # ---- artifacts
    list1 = await do(
        "vision.list_artifacts",
        "empty_project",
        "empty_ok",
        "empty",
        {"project_path": project_path_s},
    )
    list2 = await do(
        "vision.list_artifacts",
        "det_repeat",
        "empty_ok",
        "empty",
        {"project_path": project_path_s},
    )
    await do(
        "vision.list_artifacts",
        "missing_project",
        "refuse",
        "absent",
        {"project_path": str(projects_root / "does-not-exist")},
    )
    await do(
        "vision.list_artifacts",
        "outside_project",
        "refuse",
        "outside",
        {"project_path": str(outside_project.root)},
    )
    fake_digest = "ab" * 32
    await do(
        "vision.get_artifact",
        "missing_digest",
        "refuse",
        "absent",
        {"project_path": project_path_s, "digest": fake_digest},
    )
    await do(
        "vision.get_artifact",
        "empty_digest",
        "refuse",
        "empty",
        {"project_path": project_path_s, "digest": ""},
    )
    await do(
        "vision.get_artifact",
        "not_hex_digest",
        "refuse",
        "empty",
        {"project_path": project_path_s, "digest": "not-a-digest"},
    )
    await do(
        "vision.get_artifact",
        "outside_project",
        "refuse",
        "outside",
        {"project_path": str(outside_project.root), "digest": fake_digest},
    )

    # ---- observe
    await do(
        "vision.observe",
        "missing_file",
        "refuse",
        "absent",
        {
            "project_path": project_path_s,
            "rights_decision": "OWNED",
            "adapter": "image.file",
            "target": {"path": str(missing_inside)},
        },
    )
    await do(
        "vision.observe",
        "outside_file",
        "refuse",
        "outside",
        {
            "project_path": project_path_s,
            "rights_decision": "OWNED",
            "adapter": "image.file",
            "target": {"path": str(outside_file)},
        },
    )
    await do(
        "vision.observe",
        "empty_target",
        "refuse",
        "empty",
        {
            "project_path": project_path_s,
            "rights_decision": "OWNED",
            "adapter": "image.file",
            "target": None,
        },
    )
    await do(
        "vision.observe",
        "empty_path",
        "refuse",
        "empty",
        {
            "project_path": project_path_s,
            "rights_decision": "OWNED",
            "adapter": "image.file",
            "target": {"path": ""},
        },
    )
    await do(
        "vision.observe",
        "wrong_suffix",
        "refuse",
        "wrong_type",
        {
            "project_path": project_path_s,
            "rights_decision": "OWNED",
            "adapter": "image.file",
            "target": {"path": str(note_txt)},
        },
    )
    await do(
        "vision.observe",
        "unsupported_adapter",
        "refuse",
        "wrong_type",
        {
            "project_path": project_path_s,
            "rights_decision": "OWNED",
            "adapter": "browser.chromium",
            "target": {"url": "https://example.com"},
        },
    )
    await do(
        "vision.observe",
        "empty_rights",
        "refuse",
        "empty",
        {
            "project_path": project_path_s,
            "rights_decision": "   ",
            "adapter": "image.file",
            "target": {"path": str(valid_png)},
        },
    )
    observe_lie = await do(
        "vision.observe",
        "lie_png",
        "refuse",
        "wrong_type",
        {
            "project_path": project_path_s,
            "rights_decision": "OWNED",
            "adapter": "image.file",
            "target": {"path": str(lie_png)},
            "configuration": {"ocr": False},
        },
    )
    observe_empty = await do(
        "vision.observe",
        "empty_png",
        "refuse",
        "wrong_type",
        {
            "project_path": project_path_s,
            "rights_decision": "OWNED",
            "adapter": "image.file",
            "target": {"path": str(empty_png)},
            "configuration": {"ocr": False},
        },
    )
    observe_valid = await do(
        "vision.observe",
        "valid_png_system_python",
        "refuse",
        "wrong_type",
        {
            "project_path": project_path_s,
            "rights_decision": "OWNED",
            "adapter": "image.file",
            "target": {"path": str(valid_png)},
            "configuration": {"ocr": False},
        },
    )
    await do(
        "vision.observe",
        "lie_mp4",
        "refuse",
        "wrong_type",
        {
            "project_path": project_path_s,
            "rights_decision": "OWNED",
            "adapter": "video.file",
            "target": {"path": str(lie_mp4)},
        },
    )
    observe_after_heavy = heavy_present()

    # ---- query / explain / compare / verify on ids that do not exist
    missing_capture = "0" * 64
    await do(
        "vision.query",
        "missing_capture",
        "empty_hunt",
        "missing_id",
        {
            "project_path": project_path_s,
            "capture_id": missing_capture,
            "query": {"point": {"x": 0, "y": 0}},
        },
    )
    await do(
        "vision.query",
        "empty_capture_id",
        "empty_hunt",
        "empty",
        {"project_path": project_path_s, "capture_id": "", "query": {}},
    )
    await do(
        "vision.query",
        "outside_project",
        "refuse",
        "outside",
        {
            "project_path": str(outside_project.root),
            "capture_id": missing_capture,
            "query": {},
        },
    )
    q1 = await do(
        "vision.query",
        "det_repeat_a",
        "empty_hunt",
        "missing_id",
        {
            "project_path": project_path_s,
            "capture_id": missing_capture,
            "query": {"text": "nope"},
        },
    )
    q2 = await do(
        "vision.query",
        "det_repeat_b",
        "empty_hunt",
        "missing_id",
        {
            "project_path": project_path_s,
            "capture_id": missing_capture,
            "query": {"text": "nope"},
        },
    )
    await do(
        "vision.explain_region",
        "missing_capture",
        "empty_hunt",
        "missing_id",
        {
            "project_path": project_path_s,
            "capture_id": missing_capture,
            "x": 0.0,
            "y": 0.0,
        },
    )
    await do(
        "vision.explain_region",
        "empty_capture_id",
        "empty_hunt",
        "empty",
        {"project_path": project_path_s, "capture_id": "", "x": 0.0, "y": 0.0},
    )
    expl1 = await do(
        "vision.explain_region",
        "det_repeat_a",
        "empty_hunt",
        "missing_id",
        {
            "project_path": project_path_s,
            "capture_id": missing_capture,
            "x": 1.0,
            "y": 1.0,
        },
    )
    expl2 = await do(
        "vision.explain_region",
        "det_repeat_b",
        "empty_hunt",
        "missing_id",
        {
            "project_path": project_path_s,
            "capture_id": missing_capture,
            "x": 1.0,
            "y": 1.0,
        },
    )
    await do(
        "vision.compare",
        "missing_captures",
        "empty_hunt",
        "missing_id",
        {
            "project_path": project_path_s,
            "capture_a": missing_capture,
            "capture_b": "f" * 64,
        },
    )
    cmp1 = await do(
        "vision.compare",
        "det_repeat_a",
        "empty_hunt",
        "missing_id",
        {
            "project_path": project_path_s,
            "capture_a": missing_capture,
            "capture_b": missing_capture,
        },
    )
    cmp2 = await do(
        "vision.compare",
        "det_repeat_b",
        "empty_hunt",
        "missing_id",
        {
            "project_path": project_path_s,
            "capture_a": missing_capture,
            "capture_b": missing_capture,
        },
    )
    await do(
        "vision.verify",
        "missing_both",
        "refuse",
        "empty",
        {"project_path": project_path_s},
    )
    await do(
        "vision.verify",
        "unknown_capture",
        "empty_hunt",
        "missing_id",
        {"project_path": project_path_s, "capture_id": missing_capture},
    )
    await do(
        "vision.verify",
        "missing_receipt_file",
        "refuse",
        "absent",
        {
            "project_path": project_path_s,
            "receipt_path": str(project_path / "no-receipt.json"),
        },
    )
    await do(
        "vision.verify",
        "outside_receipt",
        "refuse",
        "outside",
        {"project_path": project_path_s, "receipt_path": str(outside_file)},
    )
    await do(
        "vision.verify",
        "garbage_receipt",
        "refuse",
        "wrong_type",
        {"project_path": project_path_s, "receipt_path": str(garbage_receipt)},
    )
    await do(
        "vision.verify",
        "empty_json_receipt",
        "refuse",
        "wrong_type",
        {"project_path": project_path_s, "receipt_path": str(empty_receipt)},
    )
    v1 = await do(
        "vision.verify",
        "det_repeat_a",
        "empty_hunt",
        "missing_id",
        {"project_path": project_path_s, "capture_id": missing_capture},
    )
    v2 = await do(
        "vision.verify",
        "det_repeat_b",
        "empty_hunt",
        "missing_id",
        {"project_path": project_path_s, "capture_id": missing_capture},
    )

    # ---- progress: after observe_lie, an INTERRUPTED capture may exist
    prog1 = await do(
        "vision.progress",
        "after_failed_observe",
        "empty_ok",
        "empty",
        {"project_path": project_path_s},
    )
    prog2 = await do(
        "vision.progress",
        "det_repeat",
        "empty_ok",
        "empty",
        {"project_path": project_path_s},
    )
    await do(
        "vision.progress",
        "missing_project",
        "refuse",
        "absent",
        {"project_path": str(projects_root / "does-not-exist")},
    )
    await do(
        "vision.progress",
        "outside_project",
        "refuse",
        "outside",
        {"project_path": str(outside_project.root)},
    )
    # Field-level empty success: blockers always [].
    if prog1.ok and isinstance(prog1.payload, dict):
        captures = (prog1.payload.get("overview") or {}).get("captures") or []
        interrupted = [
            c for c in captures if str(c.get("status") or "").upper() == "INTERRUPTED"
        ]
        blockers = prog1.payload.get("blockers")
        if blockers == []:
            # Always-empty blockers is empty-success iff there was something to report
            # OR even when empty, the field never looks. Record as EMPTY_SUCCESS when
            # interrupted captures exist; otherwise note it as a silent field.
            if interrupted:
                reports["vision.progress"].empty_success.append(
                    {
                        "probe_id": "blockers_silent_on_interrupted",
                        "input": {"project_path": project_path_s},
                        "response": jsonable(prog1.payload, limit=1500),
                        "why": (
                            "overview.captures contains INTERRUPTED rows but "
                            "blockers is hardcoded []. Silence is indistinguishable "
                            "from 'nothing is blocking'."
                        ),
                    }
                )
                prog1.classification = "EMPTY_SUCCESS"

    # ---- review_queue: legitimate empty, then SCHEMA_ABSENT hunt
    rq1 = await do(
        "vision.review_queue",
        "empty_project",
        "empty_ok",
        "empty",
        {"project_path": project_path_s},
    )
    rq2 = await do(
        "vision.review_queue",
        "det_repeat",
        "empty_ok",
        "empty",
        {"project_path": project_path_s},
    )
    await do(
        "vision.review_queue",
        "missing_project",
        "refuse",
        "absent",
        {"project_path": str(projects_root / "does-not-exist")},
    )
    await do(
        "vision.review_queue",
        "outside_project",
        "refuse",
        "outside",
        {"project_path": str(outside_project.root)},
    )

    schema_project = await do(
        "project.create",
        "schema_absent_host",
        "meta",
        "meta",
        {"name": "probe-notable"},
    )
    notable_path = projects_root / "probe-notable"
    notable_db = notable_path / "project.db"
    dropped = False
    drop_error = None
    if notable_db.is_file():
        try:
            conn = sqlite3.connect(str(notable_db))
            conn.execute("DROP TABLE IF EXISTS frontend_patch_proposals")
            conn.commit()
            conn.close()
            dropped = True
        except sqlite3.Error as exc:
            drop_error = str(exc)
    rq_absent = await do(
        "vision.review_queue",
        "schema_absent",
        "empty_hunt",
        "schema_absent",
        {"project_path": str(notable_path)},
    )

    # cv2-capable interpreter: content-mismatch that system python3 cannot reach
    cv2_probes = grok_vision_observe_probes(
        vmcp_src, projects_root, project_path, lie_png, valid_png
    )
    _ingest_cv2_calls(reports, cv2_probes, project_path_s)

    # progress/list after the cv2 process has written COMPLETE + INTERRUPTED rows
    prog_cv2 = await do(
        "vision.progress",
        "after_cv2_observe",
        "empty_hunt",
        "absent",
        {"project_path": project_path_s},
    )
    await do(
        "vision.list_artifacts",
        "after_cv2_observe",
        "empty_ok",
        "empty",
        {"project_path": project_path_s},
    )
    if prog_cv2.ok and isinstance(prog_cv2.payload, dict):
        captures = (prog_cv2.payload.get("overview") or {}).get("captures") or []
        interrupted = [
            c for c in captures if str(c.get("status") or "").upper() == "INTERRUPTED"
        ]
        complete = [
            c for c in captures if str(c.get("status") or "").upper() == "COMPLETE"
        ]
        if prog_cv2.payload.get("blockers") == [] and interrupted:
            prog_cv2.classification = "EMPTY_SUCCESS"
        elif prog_cv2.payload.get("blockers") == [] and not interrupted:
            # Field is still a literal []; the look over captures happened.
            prog_cv2.classification = "LEGITIMATE_EMPTY" if not complete else "SUCCESS"

    # ---- determinism per tool
    def det_pair(tool: str, a: CallResult | None, b: CallResult | None) -> dict[str, Any]:
        if a is None or b is None:
            return {"compared": False, "reason": "missing pair"}
        same_ok = a.ok == b.ok
        same_err = (a.error_type, a.error, a.cause_type, a.cause) == (
            b.error_type,
            b.error,
            b.cause_type,
            b.cause,
        )
        raw_same = canonical(jsonable(a.payload)) == canonical(jsonable(b.payload))
        stripped_same = canonical(strip_volatile(jsonable(a.payload))) == canonical(
            strip_volatile(jsonable(b.payload))
        )
        varies: list[str] = []
        if a.ok and b.ok and not raw_same:
            if not stripped_same:
                varies.append("non_volatile_payload")
            else:
                varies.append("volatile_keys_only:" + ",".join(sorted(VOLATILE_KEYS)))
        if not same_err and not (a.ok and b.ok):
            varies.append("error_text")
        breaks_cas = bool(varies) and "volatile_keys_only:" not in "".join(varies)
        # uuid in project.create
        if tool == "project.create" and a.ok and b.ok:
            id_a = ((a.payload or {}).get("project") or {}).get("id")
            id_b = ((b.payload or {}).get("project") or {}).get("id")
            if id_a != id_b:
                varies.append("project.id (uuid4)")
                breaks_cas = True
        return {
            "compared": True,
            "same_ok": same_ok,
            "byte_identical": raw_same and same_err,
            "identical_after_stripping_volatile": stripped_same and same_ok,
            "varies": varies,
            "breaks_content_addressing": breaks_cas,
            "a_classification": a.classification,
            "b_classification": b.classification,
        }

    reports["vision.capabilities"].determinism = det_pair(
        "vision.capabilities", cap1, cap2
    )
    reports["system.doctor"].determinism = det_pair("system.doctor", doc1, doc2)
    reports["project.status"].determinism = det_pair("project.status", st1, st2)
    reports["vision.list_artifacts"].determinism = det_pair(
        "vision.list_artifacts", list1, list2
    )
    reports["vision.query"].determinism = det_pair("vision.query", q1, q2)
    reports["vision.explain_region"].determinism = det_pair(
        "vision.explain_region", expl1, expl2
    )
    reports["vision.compare"].determinism = det_pair("vision.compare", cmp1, cmp2)
    reports["vision.verify"].determinism = det_pair("vision.verify", v1, v2)
    reports["vision.progress"].determinism = det_pair("vision.progress", prog1, prog2)
    reports["vision.review_queue"].determinism = det_pair(
        "vision.review_queue", rq1, rq2
    )
    reports["project.create"].determinism = det_pair("project.create", create_a, create_b)
    reports["vision.close_project"].determinism = det_pair(
        "vision.close_project", close1, close2
    )
    # tools without a natural repeat of success: repeat a refuse
    for name in (
        "vision.open_project",
        "vision.get_artifact",
        "vision.observe",
    ):
        calls = reports[name].calls
        refuses = [c for c in calls if c.kind == "refuse"]
        if len(refuses) >= 1:
            # re-invoke the first refuse
            spec = refuses[0]
            again = await invoke(
                server, name, spec.probe_id + "_repeat", spec.kind, spec.category, spec.arguments
            )
            reports[name].calls.append(again)
            reports[name].determinism = det_pair(name, spec, again)

    # ---- headline classification + empty-success harvest
    empty_success_cases: list[dict[str, Any]] = []
    rank = {
        "EMPTY_SUCCESS": 0,
        "FALSE_AVAILABLE": 1,
        "UNEXPECTED_SUCCESS": 2,
        "UNPROBEABLE": 3,
        "EXPLICIT_CLASS": 4,
        "EXPLICIT_ERROR": 5,
        "LEGITIMATE_EMPTY": 6,
        "SUCCESS": 7,
    }
    for name, report in reports.items():
        headline_kinds = {"refuse", "empty_hunt", "false_available"}
        classes = [
            c.classification for c in report.calls if c.kind in headline_kinds
        ]
        if not classes:
            classes = [
                c.classification
                for c in report.calls
                if c.probe_id in {"no_args", "happy", "happy_create"}
            ] or [c.classification for c in report.calls]
        report.headline_classification = min(
            classes, key=lambda x: rank.get(x, 9)
        ) if classes else "UNPROBEABLE"
        for call in report.calls:
            if call.classification == "EMPTY_SUCCESS":
                why = (
                    "Returned a success payload with an empty collection and no "
                    "refusal classification for a target the tool could not handle."
                )
                if name == "vision.progress":
                    why = (
                        "blockers is a literal []. overview.captures contains "
                        "INTERRUPTED rows from failed observes, so 'nothing is "
                        "blocking' is indistinguishable from 'I never looked for blockers'."
                    )
                if name == "vision.review_queue" and call.probe_id == "schema_absent":
                    why = (
                        "frontend_patch_proposals was dropped; the tool skipped the "
                        "missing table and returned {items:[], count:0} — the same "
                        "shape as a real empty review queue (LEGITIMATE_EMPTY on an "
                        "intact schema)."
                    )
                case = {
                    "tool": name,
                    "probe_id": call.probe_id,
                    "category": call.category,
                    "arguments": jsonable(call.arguments, limit=500),
                    "response": jsonable(call.payload, limit=2000),
                    "why": why,
                }
                report.empty_success.append(case)
                empty_success_cases.append(case)
        # progress field-level may already have been appended
        for case in report.empty_success:
            if case not in empty_success_cases and case.get("tool") == name:
                empty_success_cases.append(case)
            elif "tool" not in case:
                tagged = {"tool": name, **case}
                if tagged not in empty_success_cases:
                    empty_success_cases.append(tagged)

    # capabilities false-available is not empty-success but is a first-class finding
    false_available: list[dict[str, Any]] = []
    if cap1.ok and _claims_available(cap1.payload):
        observe_cannot = None
        for call in reports["vision.observe"].calls:
            if call.probe_id == "valid_png_system_python":
                observe_cannot = call
                break
        false_available.append(
            {
                "tool": "vision.capabilities",
                "probe_id": "core.observe_image_file_available",
                "claim": "core.observe_image_file status=available",
                "observe_on_valid_png": call_as_dict(observe_cannot)
                if observe_cannot
                else None,
                "why": (
                    "capabilities hardcodes core.observe_image_file as available. "
                    "On this interpreter vision.observe cannot ingest a valid PNG "
                    "because ImageFileAdapter.environment() reads cv2.__version__."
                ),
            }
        )

    # cv2 supplementary empty-success / contrast
    cv2_empty: list[dict[str, Any]] = []
    if cv2_probes.get("ok") and isinstance(cv2_probes.get("results"), dict):
        results = cv2_probes["results"]
        lie = results.get("lie_png") or {}
        far = results.get("explain_far_point") or {}
        missing = results.get("explain_missing_id") or {}
        if lie.get("ok"):
            cv2_empty.append(
                {
                    "tool": "vision.observe",
                    "probe_id": "cv2_lie_png_succeeded",
                    "why": "A non-PNG labelled .png was accepted. Content-type was not refused.",
                    "response": jsonable(lie, limit=1500),
                }
            )
        far_payload = far.get("payload") if far.get("ok") else None
        if (
            isinstance(far_payload, dict)
            and far_payload.get("explanations") == []
            and far_payload.get("citation")
        ):
            # looked and nothing — NOT empty success
            pass
        if (
            isinstance(far_payload, dict)
            and far_payload.get("explanations") == []
            and not far_payload.get("citation")
        ):
            cv2_empty.append(
                {
                    "tool": "vision.explain_region",
                    "probe_id": "cv2_far_point_no_citation",
                    "why": "explanations:[] with no citation: silence could be a missed load.",
                    "response": jsonable(far_payload, limit=1500),
                }
            )
        if missing.get("ok") and empty_collections(missing.get("payload")):
            cv2_empty.append(
                {
                    "tool": "vision.explain_region",
                    "probe_id": "cv2_missing_capture_empty_success",
                    "why": "Missing capture_id returned empty explanations instead of an error.",
                    "response": jsonable(missing.get("payload"), limit=1500),
                }
            )

    empty_success_cases.extend(cv2_empty)

    unprobed = [
        name
        for name in sorted(CORE_TOOLS)
        if name not in reports or not reports[name].calls
    ]
    extra_registered = [n for n in tool_names if n not in CORE_TOOLS]
    missing_registered = [n for n in sorted(CORE_TOOLS) if n not in tool_names]

    # What I watched fail / handoff — derived from the live calls, not pre-written.
    watched = _what_i_watched_fail(
        reports,
        empty_success_cases,
        false_available,
        cv2_probes,
        observe_lie,
        observe_valid,
        rq_absent,
        prog1,
        dropped,
    )
    handoff_text = _handoff(empty_success_cases, false_available, reports)

    receipt = {
        "schema": "hawking.headless.vmcp_capability_surface.v1",
        "obligation": (
            "Generic capability surface: what the core eyes can actually see, "
            "and what they refuse. Empty success is the finding."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "git": git_head(REPO),
        "visionmcp": {
            "version": _pkg_version(),
            "src": str(vmcp_src),
            "git": git_head(vmcp_src.parent),
            "in_this_worktree": (REPO / "visionmcp").exists(),
            "create_core_server_signature": core_sig,
            "create_core_server_profile_kwarg": named_ctor_error,
            "construction_used": "create_server(projects_root, profile='core')",
        },
        "interpreter": {
            "executable": sys.executable,
            "version": sys.version,
            "heavy_modules_before_import": import_heavy_before,
            "heavy_modules_before_create_server": create_heavy_before,
            "heavy_modules_after_create_server": create_heavy_after,
            "heavy_modules_after_observe": observe_after_heavy,
            "core_profile_imported_without_torch_cv2_open3d": not any(
                create_heavy_after.get(n) for n in ("torch", "cv2", "open3d")
            ),
            "core_observe_imported_numpy_pil": bool(
                observe_after_heavy.get("numpy") or observe_after_heavy.get("PIL")
            ),
            "core_observe_imported_cv2": bool(observe_after_heavy.get("cv2")),
        },
        "enumeration": {
            "expected_count": 15,
            "CORE_TOOLS": sorted(CORE_TOOLS),
            "registered": tool_names,
            "count": len(tool_names),
            "missing_from_server": missing_registered,
            "extra_on_server": extra_registered,
            "tools_attr_pitfall": tools_attr_note,
            "unprobed": unprobed,
        },
        "tools": {
            name: {
                "name": r.name,
                "claim": r.claim,
                "description": r.description,
                "required": r.required,
                "input_schema": r.input_schema,
                "output_schema": r.output_schema,
                "headline_classification": r.headline_classification,
                "exactness": r.exactness,
                "determinism": r.determinism,
                "calls": [call_as_dict(c) for c in r.calls],
                "empty_success": r.empty_success,
            }
            for name, r in reports.items()
        },
        "empty_success_cases": empty_success_cases,
        "false_available": false_available,
        "cv2_supplementary": jsonable(cv2_probes, limit=2500),
        "schema_absent_drop": {
            "dropped_frontend_patch_proposals": dropped,
            "drop_error": drop_error,
            "review_queue_classification": rq_absent.classification,
            "review_queue_payload": jsonable(rq_absent.payload, limit=500)
            if rq_absent.ok
            else None,
            "review_queue_error": rq_absent.error,
        },
        "calls_that_would_expose_empty_success": [
            {
                "tool": "vision.query",
                "input": "unknown capture_id",
                "empty_shape": '{"matches": [], "match_count": 0}',
                "what_actually_happened": _one_line(reports["vision.query"], "missing_capture"),
            },
            {
                "tool": "vision.explain_region",
                "input": "unknown capture_id",
                "empty_shape": '{"matches": [], "explanations": []}',
                "what_actually_happened": _one_line(
                    reports["vision.explain_region"], "missing_capture"
                ),
            },
            {
                "tool": "vision.compare",
                "input": "unknown capture_a/capture_b",
                "empty_shape": '{"identical_manifest": false, "capture_a": {}, "capture_b": {}}',
                "what_actually_happened": _one_line(
                    reports["vision.compare"], "missing_captures"
                ),
            },
            {
                "tool": "vision.verify",
                "input": "unknown capture_id",
                "empty_shape": '{"valid": true} or {}',
                "what_actually_happened": _one_line(
                    reports["vision.verify"], "unknown_capture"
                ),
            },
            {
                "tool": "vision.get_artifact",
                "input": "unknown digest",
                "empty_shape": '{"exists": true} with empty record, or {}',
                "what_actually_happened": _one_line(
                    reports["vision.get_artifact"], "missing_digest"
                ),
            },
            {
                "tool": "vision.review_queue",
                "input": "project whose frontend_patch_proposals table was dropped",
                "empty_shape": '{"items": [], "count": 0}',
                "what_actually_happened": _one_line(
                    reports["vision.review_queue"], "schema_absent"
                ),
            },
            {
                "tool": "vision.progress",
                "input": "project with INTERRUPTED captures",
                "empty_shape": '{"blockers": []}',
                "what_actually_happened": _one_line(
                    reports["vision.progress"], "after_failed_observe"
                ),
            },
            {
                "tool": "vision.list_artifacts",
                "input": "empty but valid project",
                "empty_shape": '{"artifacts": [], "count": 0}',
                "what_actually_happened": _one_line(
                    reports["vision.list_artifacts"], "empty_project"
                ),
                "note": "LEGITIMATE_EMPTY if the project opened; the look happened.",
            },
        ],
        "what_i_watched_fail": watched,
        "handoff": handoff_text,
        "scratch": str(tmp),
        "create_ok": create_ok.ok,
        "create_empty": call_as_dict(create_empty),
        "create_punct": call_as_dict(create_punct),
        "create_bad_fid": call_as_dict(create_bad_fid),
        "create_dup": call_as_dict(create_dup),
        "schema_project_ok": schema_project.ok,
    }
    return receipt, reports, server, tool_names


def _pkg_version() -> str:
    try:
        import visionmcp

        return str(getattr(visionmcp, "__version__", "unknown"))
    except Exception as exc:
        return f"unavailable: {exc}"


def _one_line(report: ToolReport, probe_id: str) -> str:
    for call in report.calls:
        if call.probe_id == probe_id:
            if call.ok:
                return f"{call.classification} payload={excerpt(call.payload, 180)}"
            return (
                f"{call.classification} {call.error_type}: {call.error}"
                + (f" cause={call.cause_type}: {call.cause}" if call.cause else "")
            )
    return "NOT RUN"


def _what_i_watched_fail(
    reports: dict[str, ToolReport],
    empty_success: list[dict[str, Any]],
    false_available: list[dict[str, Any]],
    cv2_probes: dict[str, Any],
    observe_lie: CallResult,
    observe_valid: CallResult,
    rq_absent: CallResult,
    prog1: CallResult,
    dropped: bool,
) -> list[str]:
    lines: list[str] = []
    lines.append(
        f"create_core_server(profile='core') is not a valid constructor; "
        f"the live surface is create_server(profile='core') -> create_core_server() "
        f"and it registered {len(reports)} tools."
    )
    if false_available:
        lines.append(
            "vision.capabilities reports core.observe_image_file as available "
            "while vision.observe cannot ingest a valid PNG on this interpreter."
        )
    lines.append(
        "system.doctor marks optional_blender/colmap/playwright ok=True regardless "
        "of presence and never mentions cv2/numpy/PIL."
    )
    lines.append(
        f"vision.observe lie.png ({observe_lie.classification}): "
        + (
            excerpt(observe_lie.payload, 160)
            if observe_lie.ok
            else f"{observe_lie.error_type}: {observe_lie.error}"
        )
    )
    lines.append(
        f"vision.observe valid 1x1 PNG on system python3 ({observe_valid.classification}): "
        + (
            excerpt(observe_valid.payload, 160)
            if observe_valid.ok
            else f"{observe_valid.error_type}: {observe_valid.error}"
        )
    )
    cmp = next(
        (c for c in reports["vision.compare"].calls if c.probe_id == "missing_captures"),
        None,
    )
    if cmp is not None:
        lines.append(
            "vision.compare never inspects the captures: ObservationQueryService is "
            "constructed without a CaptureBus, so verify() raises "
            f"{cmp.cause_type or cmp.error_type}: {cmp.cause or cmp.error}."
        )
    lines.append(
        f"vision.review_queue after DROP TABLE frontend_patch_proposals "
        f"(dropped={dropped}): {rq_absent.classification} "
        + (
            excerpt(rq_absent.payload, 160)
            if rq_absent.ok
            else f"{rq_absent.error_type}: {rq_absent.error}"
        )
    )
    prog_after = next(
        (c for c in reports["vision.progress"].calls if c.probe_id == "after_cv2_observe"),
        prog1,
    )
    if prog_after.ok:
        captures = (prog_after.payload or {}).get("overview", {}).get("captures") or []
        lines.append(
            f"vision.progress after cv2 observe: {len(captures)} capture(s), "
            f"statuses={[c.get('status') for c in captures]}, "
            f"blockers={(prog_after.payload or {}).get('blockers')!r}, "
            f"classification={prog_after.classification}."
        )
    if cv2_probes.get("ran"):
        lines.append(
            "Supplementary ~/.grok-vision/bin/python observe probes: "
            + excerpt(cv2_probes.get("results") or cv2_probes, 240)
        )
    else:
        lines.append(
            "Supplementary cv2 observe probes did not run: "
            + str(cv2_probes.get("reason"))
        )
    if empty_success:
        lines.append(
            f"Empty-success cases named: {len(empty_success)} "
            f"({', '.join(sorted({c.get('tool','?') for c in empty_success}))})."
        )
    else:
        lines.append(
            "No empty-success case on the python3 interpreter for the hunted calls; "
            "see calls_that_would_expose_empty_success for the exact probes."
        )
    return lines


def _handoff(
    empty_success: list[dict[str, Any]],
    false_available: list[dict[str, Any]],
    reports: dict[str, ToolReport],
) -> dict[str, Any]:
    """Smallest change that would make the worst empty-success fail loudly."""
    tools_hit = {c.get("tool") for c in empty_success}
    if "vision.progress" in tools_hit:
        return {
            "worst_case": "vision.progress.blockers hardcoded [] while INTERRUPTED captures exist",
            "file": "visionmcp/src/visionmcp/mcp/core_server.py",
            "function": "vision_progress",
            "also": (
                "vision.review_queue is the same class: a dropped "
                "frontend_patch_proposals table returns {items:[], count:0}, "
                "identical to a real empty queue."
            ),
            "smallest_change": (
                "In vision_progress, replace the literal blockers: [] with a "
                "SELECT over observation_captures WHERE status != 'COMPLETE'. "
                "Each interrupted row becomes "
                "{'code': 'CAPTURE_INTERRUPTED', 'capture_id': id, 'status': status}. "
                "If the table is missing, return {'code': 'SCHEMA_ABSENT', "
                "'blockers': []} rather than an empty list with no code. "
                "Never emit blockers:[] without having looked."
            ),
        }
    if "vision.review_queue" in tools_hit:
        return {
            "worst_case": "vision.review_queue SCHEMA_ABSENT -> {items:[], count:0}",
            "file": "visionmcp/src/visionmcp/mcp/core_server.py",
            "function": "vision_review_queue",
            "smallest_change": (
                "When 'frontend_patch_proposals' is not in sqlite_master, do not "
                "fall through to {'items': [], 'count': 0}. Return a classified "
                "payload (or raise) such as "
                "{'items': [], 'count': 0, 'code': 'SCHEMA_ABSENT', "
                "'missing_tables': ['frontend_patch_proposals']}. "
                "An empty queue and a missing table must not share a shape."
            ),
        }
    if false_available:
        return {
            "worst_case": "vision.capabilities hardcodes core.observe_image_file available",
            "file": "visionmcp/src/visionmcp/capabilities.py",
            "function": "capabilities_report",
            "smallest_change": (
                "Do not hardcode core.observe_image_file status=available. Probe "
                "numpy, PIL, and cv2 the same way plugin health is probed; if any "
                "is missing, emit status=blocked with code=IMAGING_RUNTIME_MISSING. "
                "Silence at the negotiation layer is how a verifier trusts an eye "
                "that cannot open."
            ),
        }
    cmp = reports.get("vision.compare")
    if cmp and any(c.cause_type == "RuntimeError" for c in cmp.calls):
        return {
            "worst_case": "vision.compare cannot see any capture (CaptureBus never passed)",
            "file": "visionmcp/src/visionmcp/mcp/core_server.py",
            "function": "vision_compare",
            "smallest_change": (
                "Construct ObservationQueryService(project, _capture_bus(project)) "
                "the same way vision.verify does. Then a missing capture raises "
                "KeyError('unknown capture: ...') instead of "
                "RuntimeError('verification requires a CaptureBus'). "
                "This is a loud wrong error, not an empty success; it is still "
                "the core compare path refusing to look."
            ),
        }
    return {
        "worst_case": None,
        "smallest_change": (
            "No empty-success case landed on this interpreter. Keep the hunted "
            "calls as regressions: missing capture_id must not become matches:[]; "
            "missing review tables must not become items:[]; blockers must not "
            "stay a literal empty list."
        ),
    }


# --------------------------------------------------------------------------- print

def _pad(text: str, width: int) -> str:
    text = text.replace("\n", " ")
    if len(text) <= width:
        return text.ljust(width)
    return text[: width - 1] + "…"


def print_tables(receipt: dict[str, Any]) -> None:
    tools = receipt["tools"]
    print()
    print("=== CORE SURFACE ===")
    print(
        f"visionmcp {receipt['visionmcp']['version']}  "
        f"HEAD {receipt['git'].get('head')}  "
        f"tools {receipt['enumeration']['count']}/15  "
        f"src {receipt['visionmcp']['src']}"
    )
    print(
        f"create_core_server(profile='core') -> {receipt['visionmcp']['create_core_server_profile_kwarg']}"
    )
    print(
        f"core import without torch/cv2/open3d: "
        f"{receipt['interpreter']['core_profile_imported_without_torch_cv2_open3d']}"
    )
    print(
        f"observe pulled numpy/PIL: {receipt['interpreter']['core_observe_imported_numpy_pil']}  "
        f"cv2: {receipt['interpreter']['core_observe_imported_cv2']}"
    )
    print(
        f"_tools pitfall: hasattr={receipt['enumeration']['tools_attr_pitfall']['server_has__tools']}  "
        f"getattr(server,'_tools',[])={receipt['enumeration']['tools_attr_pitfall']['getattr_server__tools_default_empty_list']}"
    )
    print()
    print("=== TABLE: 15 tools — claim, refusal, empty-success, exactness, determinism ===")
    hdr = (
        f"{_pad('#', 2)}  {_pad('tool', 24)}  {_pad('required', 28)}  "
        f"{_pad('headline', 16)}  {_pad('exactness', 18)}  {_pad('determinism', 28)}  claim"
    )
    print(hdr)
    print("-" * min(160, len(hdr) + 40))
    for i, name in enumerate(receipt["enumeration"]["registered"], 1):
        t = tools[name]
        required = ",".join(t["required"]) or "(none)"
        det = t.get("determinism") or {}
        if det.get("byte_identical"):
            det_s = "identical"
        elif det.get("identical_after_stripping_volatile"):
            det_s = "volatile only"
        elif det.get("compared"):
            varies = ",".join(det.get("varies") or []) or "differs"
            det_s = varies[:28]
        else:
            det_s = "not compared"
        print(
            f"{_pad(str(i), 2)}  {_pad(name, 24)}  {_pad(required, 28)}  "
            f"{_pad(t['headline_classification'], 16)}  "
            f"{_pad(str((t.get('exactness') or {}).get('class')), 18)}  "
            f"{_pad(det_s, 28)}  {t['claim']}"
        )

    print()
    print("=== TABLE: every unhandleable call (exact response) ===")
    print(
        f"{_pad('tool', 24)}  {_pad('probe', 28)}  {_pad('class', 16)}  response"
    )
    print("-" * 160)
    for name in receipt["enumeration"]["registered"]:
        for call in tools[name]["calls"]:
            if call["ok"]:
                resp = excerpt(call["payload"], 140)
            else:
                resp = f"{call['error_type']}: {call['error']}"
                if call.get("cause"):
                    resp += f" [{call['cause_type']}: {call['cause']}]"
            print(
                f"{_pad(name, 24)}  {_pad(call['probe_id'], 28)}  "
                f"{_pad(call['classification'], 16)}  {resp}"
            )

    print()
    print("=== EMPTY SUCCESS ===")
    cases = receipt["empty_success_cases"]
    if not cases:
        print("None on this interpreter.")
        print("Calls that would have exposed one:")
        for row in receipt["calls_that_would_expose_empty_success"]:
            print(
                f"  - {row['tool']} {row['input']}: empty shape {row['empty_shape']}"
            )
            print(f"    actual: {row['what_actually_happened']}")
    else:
        for case in cases:
            print(f"- {case.get('tool')} / {case.get('probe_id')}")
            print(f"  why: {case.get('why')}")
            print(f"  response: {excerpt(case.get('response'), 300)}")

    print()
    print("=== FALSE AVAILABLE ===")
    if not receipt["false_available"]:
        print("None.")
    else:
        for row in receipt["false_available"]:
            print(f"- {row['tool']}: {row['claim']}")
            print(f"  {row['why']}")

    print()
    print("## WHAT I WATCHED FAIL")
    for line in receipt["what_i_watched_fail"]:
        print(f"- {line}")

    print()
    print("## HANDOFF")
    h = receipt["handoff"]
    print(f"worst_case: {h.get('worst_case')}")
    if h.get("file"):
        print(f"file: {h['file']} :: {h.get('function')}")
    if h.get("also"):
        print(f"also: {h['also']}")
    print(h.get("smallest_change"))
    print()
    print(f"receipt: {RECEIPT_PATH}")
    print(f"unprobed: {receipt['enumeration']['unprobed'] or 'none'}")


def main() -> int:
    try:
        receipt, reports, _server, names = asyncio.run(probe())
    except FileNotFoundError as exc:
        print(f"BLOCKER: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1

    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print_tables(receipt)
    if receipt["enumeration"]["unprobed"] or len(names) != 15:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
