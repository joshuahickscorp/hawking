#!/usr/bin/env python3
"""Gate: unavailable visionmcp tools must not look like a negative finding.

A WorkUnit's acceptance can rest on VMCP evidence. The single most dangerous
shape for a verifier is an empty or observed-class return that reads as
"I looked and there was nothing" (or "I looked and this is what I saw") when
the tool in fact could not look.

This gate runs against the real visionmcp substrate. It is expected to FAIL
today against the cited auto-degrade sites. A first-run PASS means the gate
is measuring the wrong thing.

    python3 tools/headless/vmcp_unavailable_gate.py

visionmcp/ is not materialized in this sparse worktree. The gate locates a
0.8.0a2 checkout read-only (VISIONMCP_SRC or well-known donor trees). It
does not modify visionmcp.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RECEIPT_PATH = REPO / "receipts" / "headless" / "VMCP_UNAVAILABLE_GATE.json"

# Tokens a caller holding only the return can treat as unavailability.
# Prose limitations and a fallback_reason sitting next to a stored solution
# do not count — that is the defect this gate exists to catch.
UNAVAIL_STATUS = {
    "UNAVAILABLE",
    "BLOCKED",
    "INTERRUPTED",
    "unavailable",
    "blocked",
}
UNAVAIL_TOKENS = {
    "UNAVAILABLE",
    "TOOL_ABSENT",
    "BACKEND_UNAVAILABLE",
    "OCR_UNAVAILABLE",
    "COLMAP_UNAVAILABLE",
    "BLENDER_UNAVAILABLE",
    "OCC_RUNTIME_UNAVAILABLE",
    "TARGET_ABSENT",
}
OBSERVED_CLASSES = {
    "MULTI_VIEW_OBSERVED",
    "SINGLE_VIEW_OBSERVED",
    "MEASURED",
    "TEARDOWN_OBSERVED",
    "INFERRED_HIGH_CONFIDENCE",
    "INFERRED_LOW_CONFIDENCE",
}
FALLBACK_BACKENDS = {
    "silhouette",
    "turntable_fallback",
    "heuristic-pinhole",
    "step_fixture_dialect",
}


# --------------------------------------------------------------------------- locate / identity


def locate_visionmcp_src() -> Path:
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
    seen: set[Path] = set()
    for src in candidates:
        resolved = src.resolve() if src.exists() else src
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "visionmcp" / "__init__.py").is_file():
            return resolved
    raise FileNotFoundError(
        "visionmcp src not found. Set VISIONMCP_SRC to the package's src/ "
        "directory. This sparse worktree does not materialize visionmcp/, "
        "and git sparse-checkout add is denied."
    )


def git_head(cwd: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        oneline = subprocess.run(
            ["git", "-C", str(cwd), "log", "-1", "--oneline"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"head": head, "oneline": oneline, "cwd": str(cwd)}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"head": None, "oneline": None, "cwd": str(cwd), "error": str(exc)}


def maybe_reexec_for_cv2() -> None:
    """OCR goes through ImageFileAdapter, which needs OpenCV.

    The operator command is `python3 tools/headless/vmcp_unavailable_gate.py`.
    System python3 has Pillow but not cv2. ~/.grok-vision/bin/python has cv2.
    Re-exec is an implementation detail of that command, not a different gate.
    """
    if os.environ.get("VMCP_UNAVAILABLE_GATE_REEXEC") == "1":
        return
    try:
        import cv2  # noqa: F401
        return
    except ImportError:
        pass
    vision = Path.home() / ".grok-vision" / "bin" / "python"
    if not vision.is_file():
        return
    os.environ["VMCP_UNAVAILABLE_GATE_REEXEC"] = "1"
    os.execv(str(vision), [str(vision), *sys.argv])


def jsonable(value: Any, *, limit: int = 800) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v, limit=limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [jsonable(v, limit=limit) for v in value]
        if len(items) > 40:
            return items[:40] + [f"... {len(items) - 40} more"]
        return items
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > limit:
            return value[:limit] + f"... <truncated {len(value) - limit} chars>"
        return value
    if hasattr(value, "value") and not callable(value):
        try:
            return jsonable(value.value, limit=limit)
        except Exception:
            pass
    return repr(value)


# --------------------------------------------------------------------------- classification
#
# A fallback_reason next to a stored observed-class solution does NOT count.
# The contract lists fallback_reason as a valid classification, but a gate
# that treated cameras.auto as classified-unavailable would PASS today —
# and a first-run PASS against these sites means the gate is wrong.
# Classification must be visible as unavailability *instead of* a usable
# observation, not buried beside one.


def _limitations_tokens(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("limitations")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    return [item for item in raw if isinstance(item, str)]


def classified_unavailable(payload: Any, *, error: BaseException | None = None) -> dict[str, Any]:
    if error is not None:
        name = type(error).__name__
        if name == "BackendUnavailable":
            return {"ok": True, "via": f"exception:{name}", "error": str(error)}
        return {"ok": False, "via": f"exception:{name}", "error": str(error)}
    if isinstance(payload, tuple) and payload:
        if payload[0] is False:
            return {"ok": True, "via": "available() False", "reason": payload[1] if len(payload) > 1 else None}
        return {"ok": False, "via": "available() True"}
    if not isinstance(payload, dict):
        return {"ok": False, "via": f"non-dict:{type(payload).__name__}"}
    status = payload.get("status")
    if isinstance(status, str) and status.upper() in {s.upper() for s in UNAVAIL_STATUS}:
        return {"ok": True, "via": f"status={status}"}
    evidence_class = payload.get("evidence_class")
    if evidence_class == "UNAVAILABLE":
        return {"ok": True, "via": "evidence_class=UNAVAILABLE"}
    if payload.get("available") is False:
        return {"ok": True, "via": "available=False"}
    for item in _limitations_tokens(payload):
        token = item.strip()
        if token in UNAVAIL_TOKENS or token.split(":")[0] in UNAVAIL_TOKENS:
            return {"ok": True, "via": f"limitations:{token}"}
        if token.upper() in UNAVAIL_TOKENS:
            return {"ok": True, "via": f"limitations:{token}"}
    # fallback_reason only counts when the payload is not also a usable
    # observation (handled by the verdict, which checks usable_as_observation).
    return {"ok": False, "via": "none"}


def usable_as_observation(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"ok": False, "via": None}
    evidence_class = payload.get("evidence_class")
    if evidence_class in OBSERVED_CLASSES:
        return {"ok": True, "via": f"evidence_class={evidence_class}"}
    cameras = payload.get("cameras")
    n_cameras = payload.get("n_cameras")
    if isinstance(cameras, list) and cameras:
        return {"ok": True, "via": f"cameras[{len(cameras)}]"}
    if isinstance(n_cameras, int) and n_cameras > 0:
        return {"ok": True, "via": f"n_cameras={n_cameras}"}
    backend = payload.get("backend")
    if backend in FALLBACK_BACKENDS and payload.get("status") not in UNAVAIL_STATUS:
        if payload.get("n_masks") or payload.get("mask_artifacts"):
            return {"ok": True, "via": f"backend={backend} with masks"}
        if payload.get("available") is True:
            return {"ok": True, "via": f"backend={backend} available=True"}
    return {"ok": False, "via": None}


def empty_negative_shape(payload: Any) -> dict[str, Any]:
    """Shapes a verifier would read as 'looked, found nothing'."""
    if not isinstance(payload, dict):
        return {"ok": False, "via": None}
    if payload.get("ocr") == []:
        return {"ok": True, "via": "ocr=[]"}
    summary = payload.get("summary")
    if isinstance(summary, dict) and summary.get("text_symbol_count") == 0:
        return {"ok": True, "via": "summary.text_symbol_count=0"}
    if payload.get("text_symbol_count") == 0:
        return {"ok": True, "via": "text_symbol_count=0"}
    tools = payload.get("available_tools")
    if isinstance(tools, dict) and tools.get("blender") is False:
        return {"ok": True, "via": "available_tools.blender=false"}
    if payload.get("blender") is False:
        return {"ok": True, "via": "blender=false"}
    if payload.get("comparisons") == []:
        return {"ok": True, "via": "comparisons=[]"}
    return {"ok": False, "via": None}


def verdict_absent(*, payload: Any, error: BaseException | None = None, extra: dict[str, Any] | None = None) -> str:
    extra = extra or {}
    classified = classified_unavailable(payload, error=error)
    usable = usable_as_observation(payload)
    empty = empty_negative_shape(payload)
    if classified["ok"] and not usable["ok"]:
        return "PASS_CLASSIFIED"
    if classified["ok"] and usable["ok"]:
        return "FAIL_DEGRADED_TO_OBSERVED"
    if extra.get("tool_actually_installed") and empty["ok"]:
        return "FAIL_FALSE_ABSENT"
    if extra.get("real_backend_failed") and payload and payload.get("available") is True:
        return "FAIL_FALSE_AVAILABLE"
    if empty["ok"]:
        return "FAIL_EMPTY_LOOKS_NEGATIVE"
    if usable["ok"]:
        return "FAIL_DEGRADED_TO_OBSERVED"
    return "FAIL_UNCLASSIFIED"


def verdict_true_negative(*, payload: Any, error: BaseException | None = None) -> str:
    classified = classified_unavailable(payload, error=error)
    if classified["ok"]:
        # A capability that always reports UNAVAILABLE would fail here.
        return "FAIL_ALWAYS_UNAVAILABLE"
    return "PASS_TRUE_NEGATIVE"


# --------------------------------------------------------------------------- hiding tools without uninstalling


@contextmanager
def hide_tools(*names: str):
    """Make named binaries invisible to shutil.which without uninstalling them.

    Technique: monkeypatch shutil.which to return None for those names.
    PATH-stripping the homebrew prefix would also hide sibling tools
    (colmap and tesseract both live in /opt/homebrew/bin), so the which
    patch is the narrower faithful injection: every cited site decides
    with shutil.which("colmap"|"tesseract"|"blender"), looked up at the
    call, not cached.
    """
    orig = shutil.which
    hidden = set(names)

    def wrapped(cmd: str, mode: int = os.F_OK, path: str | None = None) -> str | None:
        if os.path.basename(str(cmd)) in hidden:
            return None
        return orig(cmd, mode=mode, path=path)

    shutil.which = wrapped  # type: ignore[assignment]
    try:
        yield {
            "technique": "monkeypatch shutil.which",
            "hidden": list(names),
            "real_paths": {name: orig(name) for name in names},
        }
    finally:
        shutil.which = orig


def make_images(root: Path) -> dict[str, Path]:
    from PIL import Image, ImageDraw

    root.mkdir(parents=True, exist_ok=True)
    view_a = root / "view_a.png"
    view_b = root / "view_b.png"
    blank_a = root / "blank_a.png"
    blank_b = root / "blank_b.png"
    text = root / "hello.png"
    Image.new("RGB", (128, 128), (255, 255, 255)).save(blank_a)
    Image.new("RGB", (128, 128), (255, 255, 255)).save(blank_b)
    a = Image.new("RGB", (128, 128), (255, 255, 255))
    ImageDraw.Draw(a).rectangle((20, 20, 90, 90), fill=(200, 20, 20))
    a.save(view_a)
    b = Image.new("RGB", (128, 128), (255, 255, 255))
    ImageDraw.Draw(b).ellipse((30, 30, 100, 100), fill=(20, 20, 200))
    b.save(view_b)
    hello = Image.new("RGB", (320, 80), (255, 255, 255))
    ImageDraw.Draw(hello).text((10, 30), "HELLO WORLD", fill=(0, 0, 0))
    hello.save(text)
    return {
        "view_a": view_a,
        "view_b": view_b,
        "blank_a": blank_a,
        "blank_b": blank_b,
        "text": text,
    }


def temp_project(name: str, root: Path):
    from visionmcp.projects.store import ProjectStore

    path = root / name
    return ProjectStore.create(path, name=name)


def ingest_images(project: Any, *paths: Path) -> None:
    from visionmcp.evidence.references import ReferenceIngestor

    ingestor = ReferenceIngestor(project)
    for path in paths:
        ingestor.import_file(path, rights_state="OWNED")


def summarize_geometry(record: dict[str, Any]) -> dict[str, Any]:
    evidence = record.get("evidence") or {}
    diagnostics = evidence.get("diagnostics") or {}
    configuration = record.get("configuration") or {}
    return {
        "id": record.get("id"),
        "backend": record.get("backend"),
        "evidence_class": record.get("evidence_class"),
        "configuration": configuration,
        "colmap_fallback_reason": diagnostics.get("colmap_fallback_reason")
        or configuration.get("colmap_fallback_reason"),
        "method": diagnostics.get("method"),
        "uncertainty": evidence.get("uncertainty"),
        "n_masks": len(evidence.get("mask_artifacts") or []),
        "mask_occupied": [m.get("occupied_fraction") for m in (diagnostics.get("masks") or [])],
        "status": record.get("status"),
        "limitations": record.get("limitations"),
    }


def summarize_cameras(record: dict[str, Any]) -> dict[str, Any]:
    cameras = []
    for camera in record.get("cameras") or []:
        cameras.append(
            {
                "confidence": camera.get("confidence"),
                "evidence_class": camera.get("evidence_class"),
                "registration_class": camera.get("registration_class"),
                "solve_method": camera.get("solve_method"),
            }
        )
    diagnostics = record.get("diagnostics") or {}
    return {
        "id": record.get("id"),
        "backend": record.get("backend"),
        "n_cameras": len(record.get("cameras") or []),
        "cameras": cameras,
        "colmap_fallback_reason": diagnostics.get("colmap_fallback_reason"),
        "diagnostics_warning": diagnostics.get("warning"),
        "registered_images": diagnostics.get("registered_images"),
        "status": record.get("status"),
        "limitations": record.get("limitations"),
        "evidence_class": (cameras[0]["evidence_class"] if cameras else None),
    }


def summarize_ocr_capture(record: dict[str, Any], *, analysis: dict[str, Any] | None = None) -> dict[str, Any]:
    environment = record.get("environment") or {}
    if isinstance(environment, str):
        environment = json.loads(environment)
    summary = record.get("summary") or {}
    if isinstance(summary, str):
        summary = json.loads(summary)
    limitations = record.get("limitations") or []
    if isinstance(limitations, str):
        limitations = json.loads(limitations)
    ocr = None if analysis is None else analysis.get("ocr")
    return {
        "status": record.get("status"),
        "authority": record.get("authority"),
        "summary": summary,
        "text_symbol_count": summary.get("text_symbol_count"),
        "limitations": limitations,
        "environment_tesseract": environment.get("tesseract"),
        "ocr": ocr if ocr is None else jsonable(ocr, limit=120),
    }


def probe_record(
    *,
    name: str,
    payload: Any,
    error: BaseException | None = None,
    kind: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    classified = classified_unavailable(payload, error=error)
    usable = usable_as_observation(payload)
    empty = empty_negative_shape(payload)
    if kind == "tool_absent":
        verdict = verdict_absent(payload=payload, error=error, extra=extra)
    elif kind == "true_negative":
        verdict = verdict_true_negative(payload=payload, error=error)
    elif kind == "present_control":
        verdict = "PASS_PRESENT_CONTROL" if extra and extra.get("found") else "FAIL_PRESENT_CONTROL"
    elif kind == "not_probed":
        verdict = "NOT_PROBED"
    else:
        verdict = "UNKNOWN"
    row = {
        "name": name,
        "kind": kind,
        "verdict": verdict,
        "classified_unavailable": classified,
        "usable_as_observation": usable,
        "empty_negative_shape": empty,
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else None,
        "return": jsonable(payload),
    }
    if extra:
        row["extra"] = jsonable(extra)
    return row


def print_probe(site_id: str, row: dict[str, Any]) -> None:
    mark = "FAIL" if row["verdict"].startswith("FAIL") else (
        "skip" if row["verdict"] == "NOT_PROBED" else "ok  "
    )
    print(f"  {mark} {site_id}/{row['name']}: {row['verdict']}")
    classified = row["classified_unavailable"]
    usable = row["usable_as_observation"]
    empty = row["empty_negative_shape"]
    print(f"      classified_unavailable={classified['ok']} via={classified.get('via')}")
    print(f"      usable_as_observation={usable['ok']} via={usable.get('via')}")
    print(f"      empty_negative_shape={empty['ok']} via={empty.get('via')}")
    if row.get("error_type"):
        print(f"      error={row['error_type']}: {row['error']}")
    returned = row.get("return")
    excerpt = json.dumps(returned, default=str)
    if len(excerpt) > 420:
        excerpt = excerpt[:419] + "…"
    print(f"      return={excerpt}")


# --------------------------------------------------------------------------- sites


def site_geometry(images: dict[str, Path], work: Path) -> dict[str, Any]:
    from visionmcp.vision.pipeline import GeometryPipeline

    site: dict[str, Any] = {
        "id": "geometry.pipeline.auto_colmap",
        "cited": "visionmcp/src/visionmcp/vision/pipeline.py:39-46, :240-248",
        "technique": (
            "monkeypatch shutil.which so 'colmap' returns None. The cited auto "
            "branch is `if shutil.which('colmap') and len(references) >= 2`."
        ),
        "technique_why_faithful": (
            "GeometryPipeline.run('auto') consults shutil.which at call time "
            "and only sets colmap_fallback_reason on an exception from "
            "_run_colmap, not when the binary is simply absent. Hiding the "
            "name from which() is exactly the absent-binary path, without "
            "uninstalling COLMAP."
        ),
        "probes": [],
    }
    # tool absent
    project = temp_project("geom-absent", work)
    ingest_images(project, images["view_a"], images["view_b"])
    with hide_tools("colmap"):
        record = GeometryPipeline(project).run("auto")
    summary = summarize_geometry(record)
    site["probes"].append(
        probe_record(name="tool_absent_auto", payload=summary, kind="tool_absent")
    )
    # true negative: tool present, explicit silhouette on blank views, nothing occupied
    project_tn = temp_project("geom-tn", work)
    ingest_images(project_tn, images["blank_a"], images["blank_b"])
    tn = GeometryPipeline(project_tn).run(
        "silhouette", configuration={"solve_cameras": False}
    )
    tn_summary = summarize_geometry(tn)
    site["probes"].append(
        probe_record(name="true_negative_blank_silhouettes", payload=tn_summary, kind="true_negative")
    )
    site["probes"].append(
        probe_record(
            name="auto_colmap_present_empty_scene",
            payload={"status": "NOT_PROBED"},
            kind="not_probed",
            extra={
                "reason": (
                    "A true-negative of COLMAP reconstruction (tool present, "
                    "genuinely no 3D points) requires running the COLMAP mapper "
                    "on a no-feature scene. This gate does not start that job. "
                    "The constructible true negative is explicit silhouette "
                    "backend on blank images with colmap still on PATH."
                )
            },
        )
    )
    return site


def site_cameras(images: dict[str, Path], work: Path) -> dict[str, Any]:
    from visionmcp.cameras.solver import CameraSolver
    from visionmcp.core.errors import VisionMCPError

    site: dict[str, Any] = {
        "id": "cameras.solver.auto_colmap",
        "cited": "visionmcp/src/visionmcp/cameras/solver.py:251-258, :1568-1609",
        "technique": (
            "monkeypatch shutil.which so 'colmap' returns None. _solve_colmap "
            "raises BackendUnavailable('COLMAP is not installed') when which() "
            "is None; solve('auto') catches every Exception including that."
        ),
        "technique_why_faithful": (
            "The cited auto branch is try/_solve_colmap/except Exception, and "
            "_solve_colmap's first instruction is shutil.which('colmap'). "
            "Hiding the binary is the live BackendUnavailable path, without "
            "uninstalling COLMAP. The stored turntable at confidence=0.2 is "
            "the return a caller holding only the solution document sees."
        ),
        "probes": [],
    }
    project = temp_project("cam-absent", work)
    ingest_images(project, images["view_a"], images["view_b"])
    with hide_tools("colmap"):
        record = CameraSolver(project).solve("auto")
    summary = summarize_cameras(record)
    site["probes"].append(
        probe_record(name="tool_absent_auto", payload=summary, kind="tool_absent")
    )

    # true negative: colmap present, request that cannot produce cameras, raises
    project_tn = temp_project("cam-tn", work)
    ingest_images(project_tn, images["view_a"])
    tn_error: BaseException | None = None
    tn_payload: Any = None
    try:
        tn_payload = summarize_cameras(CameraSolver(project_tn).solve("colmap"))
    except VisionMCPError as exc:
        tn_error = exc
        tn_payload = {"raised": type(exc).__name__, "message": str(exc)}
    site["probes"].append(
        probe_record(
            name="true_negative_colmap_one_view",
            payload=tn_payload,
            error=tn_error,
            kind="true_negative",
        )
    )
    # extra: auto with colmap present and one view still stores turntable
    project_one = temp_project("cam-one", work)
    ingest_images(project_one, images["view_a"])
    one = summarize_cameras(CameraSolver(project_one).solve("auto"))
    site["probes"].append(
        probe_record(
            name="auto_one_view_colmap_present_still_turntable",
            payload=one,
            kind="tool_absent",
        )
    )
    site["probes"].append(
        probe_record(
            name="auto_colmap_present_empty_reconstruction",
            payload={"status": "NOT_PROBED"},
            kind="not_probed",
            extra={
                "reason": (
                    "Running COLMAP on a no-feature pair is a photogrammetry "
                    "job this gate does not start. CameraSolver has no "
                    "empty-cameras success path: auto always stores a "
                    "solution or the explicit colmap backend raises."
                )
            },
        )
    )
    return site


def site_cad() -> dict[str, Any]:
    from visionmcp.compilers.spatial.interfaces import (
        FixtureCadBackend,
        resolve_cad_backend,
    )
    from visionmcp.worlds.spatial.models import SpatialIR
    import visionmcp.worlds.spatial.cad as cadmod

    site: dict[str, Any] = {
        "id": "compilers.spatial.resolve_cad_backend",
        "cited": "visionmcp/src/visionmcp/compilers/spatial/interfaces.py:326-338",
        "technique": (
            "Live import failure plus injected exception. "
            "`from visionmcp.worlds.spatial.cad import get_default_backend` "
            "fails because the symbol is not exported; the except Exception "
            "path returns FixtureCadBackend(). We also assign a raising "
            "get_default_backend to prove the catch-all."
        ),
        "technique_why_faithful": (
            "The cited function is try/import/call/except Exception/pass/"
            "return FixtureCadBackend(). Injecting the import failure is "
            "exactly that branch. Uninstalling OpenCascade is unnecessary: "
            "the Step 20 entry point is already missing."
        ),
        "probes": [],
    }

    def as_payload(backend: Any) -> dict[str, Any]:
        available = backend.available()
        return {
            "type": type(backend).__name__,
            "name": getattr(backend, "name", None),
            "backend": getattr(backend, "name", None),
            "available": bool(available[0]) if isinstance(available, tuple) else bool(available),
            "available_reason": available[1] if isinstance(available, tuple) and len(available) > 1 else None,
            "status": "EXECUTED" if (available[0] if isinstance(available, tuple) else available) else "UNAVAILABLE",
        }

    live = resolve_cad_backend()
    live_payload = as_payload(live)
    site["probes"].append(
        probe_record(
            name="tool_absent_live_import",
            payload=live_payload,
            kind="tool_absent",
            extra={"real_backend_failed": True},
        )
    )

    previous = getattr(cadmod, "get_default_backend", None)

    def boom() -> Any:
        raise RuntimeError("injected Step 20 failure")

    cadmod.get_default_backend = boom  # type: ignore[attr-defined]
    try:
        injected = resolve_cad_backend()
        site["probes"].append(
            probe_record(
                name="tool_absent_injected_exception",
                payload=as_payload(injected),
                kind="tool_absent",
                extra={"real_backend_failed": True},
            )
        )
    finally:
        if previous is None:
            try:
                delattr(cadmod, "get_default_backend")
            except AttributeError:
                pass
        else:
            cadmod.get_default_backend = previous

    # control: when Step 20 returns an unavailable backend, resolve forwards it
    class UnavailableCad:
        name = "occ"

        def available(self) -> tuple[bool, str]:
            return False, "OCC_RUNTIME_UNAVAILABLE"

    cadmod.get_default_backend = lambda: UnavailableCad()  # type: ignore[attr-defined]
    try:
        forwarded = resolve_cad_backend()
        forwarded_payload = as_payload(forwarded)
        classified = classified_unavailable(forwarded_payload)
        site["probes"].append(
            {
                **probe_record(
                    name="present_control_unavailable_backend_forwarded",
                    payload=forwarded_payload,
                    kind="present_control",
                    extra={"found": classified["ok"] and forwarded_payload.get("name") == "occ"},
                ),
            }
        )
    finally:
        if previous is None:
            try:
                delattr(cadmod, "get_default_backend")
            except AttributeError:
                pass
        else:
            cadmod.get_default_backend = previous

    tn = FixtureCadBackend().surface_distance(
        SpatialIR(scene_id="empty-a"), SpatialIR(scene_id="empty-b")
    )
    site["probes"].append(
        probe_record(name="true_negative_empty_surface_distance", payload=tn, kind="true_negative")
    )
    return site


def site_ocr(images: dict[str, Path], work: Path) -> dict[str, Any]:
    from visionmcp.perception.bus import AdapterRegistry, CaptureBus
    from visionmcp.perception.media import ImageFileAdapter, analyze_image

    site: dict[str, Any] = {
        "id": "perception.media.ocr_tesseract",
        "cited": "visionmcp/src/visionmcp/perception/media.py:666-670, :698-715",
        "technique": (
            "monkeypatch shutil.which so 'tesseract' returns None. analyze_image "
            "then takes the `else []` branch; ImageFileAdapter.capture stores "
            "text_symbol_count=0 with the same limitations prose as a blank image."
        ),
        "technique_why_faithful": (
            "The cited condition is `if config.get('ocr') and shutil.which('tesseract')`. "
            "Hiding the binary is the absent-tesseract path, without uninstalling "
            "Tesseract 5.5.2. The return a caller holding only the capture sees "
            "is summary.text_symbol_count and analysis['ocr']."
        ),
        "probes": [],
    }

    def observe(project: Any, path: Path) -> dict[str, Any]:
        registry = AdapterRegistry()
        registry.register(ImageFileAdapter())
        bus = CaptureBus(project, registry)
        return bus.observe(
            "image.file",
            {"path": str(path)},
            {"ocr": True, "maximum_dimension": 512},
            rights_decision="owned",
        )

    config = {"maximum_dimension": 512, "ocr": True, "maximum_regions": 128, "depth": None}

    project = temp_project("ocr-absent", work)
    with hide_tools("tesseract"):
        analysis, _png = analyze_image(images["text"], config)
        capture = observe(project, images["text"])
    payload = summarize_ocr_capture(capture, analysis=analysis)
    site["probes"].append(
        probe_record(name="tool_absent_text_image", payload=payload, kind="tool_absent")
    )

    project_tn = temp_project("ocr-tn", work)
    analysis_tn, _png = analyze_image(images["blank_a"], config)
    capture_tn = observe(project_tn, images["blank_a"])
    payload_tn = summarize_ocr_capture(capture_tn, analysis=analysis_tn)
    site["probes"].append(
        probe_record(name="true_negative_blank_image", payload=payload_tn, kind="true_negative")
    )

    # present control: tesseract on PATH actually reads HELLO WORLD
    project_pos = temp_project("ocr-pos", work)
    analysis_pos, _png = analyze_image(images["text"], config)
    capture_pos = observe(project_pos, images["text"])
    payload_pos = summarize_ocr_capture(capture_pos, analysis=analysis_pos)
    found = int(payload_pos.get("text_symbol_count") or 0) > 0
    site["probes"].append(
        probe_record(
            name="present_control_text_image",
            payload=payload_pos,
            kind="present_control",
            extra={"found": found},
        )
    )
    return site


def site_blender() -> dict[str, Any]:
    from visionmcp.core.config import discover_blender
    from visionmcp.orchestration.resources import discover_resources

    site: dict[str, Any] = {
        "id": "orchestration.resources.discover_blender",
        "cited": "visionmcp/src/visionmcp/orchestration/resources.py:73-88",
        "technique": (
            "No injection for the live probe: shutil.which('blender') is already "
            "None on this machine while Blender 4.2.1 is at "
            "/Applications/Blender.app/Contents/MacOS/Blender. Present control "
            "sets BLENDER_PATH to that binary (the only extra channel "
            "discover_resources consults)."
        ),
        "technique_why_faithful": (
            "discover_resources uses only shutil.which('blender') or "
            "os.environ['BLENDER_PATH']. discover_blender() in core.config "
            "already knows the macOS app path. The disagreement is the live "
            "defect, not a synthetic one."
        ),
        "probes": [],
    }

    app = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    which_blender = shutil.which("blender")
    doctor = discover_blender()
    resources = discover_resources()
    live_payload = {
        "available_tools": resources.get("available_tools"),
        "blender": (resources.get("available_tools") or {}).get("blender"),
        "degradation_policy": resources.get("degradation_policy"),
        "which_blender": which_blender,
        "app_path": str(app),
        "app_exists": app.is_file(),
        "discover_blender": {
            "available": bool(getattr(doctor, "available", False)),
            "path": getattr(doctor, "path", None),
            "version": getattr(doctor, "version", None),
        },
        "limitations": resources.get("limitations"),
        "status": resources.get("status"),
    }
    site["probes"].append(
        probe_record(
            name="live_which_miss_app_present",
            payload=live_payload,
            # This probe was labelled `tool_absent`, but its premise is the
            # opposite: `which` misses AND the app bundle is present, so the
            # tool IS installed and the site must report it FOUND. Under the
            # absent label the verdict function looked for a classification of
            # unavailability and returned UNCLASSIFIED for a correct answer.
            # It is a present-control, and as one it still fails on the old
            # `shutil.which("blender")`-only code.
            kind="present_control",
            extra={
                "tool_actually_installed": bool(app.is_file() and getattr(doctor, "available", False)),
                # `found` is what the present-control verdict reads. It is the
                # site's OWN answer, not the probe's knowledge that the tool
                # exists -- otherwise this would assert that Blender is
                # installed (which the probe already knows) instead of that
                # `discover_resources` NOTICED it.
                "found": bool((resources.get("available_tools") or {}).get("blender")),
            },
        )
    )

    orig = os.environ.get("BLENDER_PATH")
    os.environ["BLENDER_PATH"] = str(app)
    try:
        present = discover_resources()
        present_payload = {
            "available_tools": present.get("available_tools"),
            "blender": (present.get("available_tools") or {}).get("blender"),
        }
        site["probes"].append(
            probe_record(
                name="present_control_BLENDER_PATH",
                payload=present_payload,
                kind="present_control",
                extra={"found": bool((present.get("available_tools") or {}).get("blender"))},
            )
        )
    finally:
        if orig is None:
            os.environ.pop("BLENDER_PATH", None)
        else:
            os.environ["BLENDER_PATH"] = orig

    # True negative of the channels discover_resources actually consults:
    # which() miss and BLENDER_PATH unset. That return is `blender: false`.
    # It is the same shape as the live miss, which is the defect — live is
    # FAIL_FALSE_ABSENT because the app bundle exists; this probe has no
    # such extra, so the classifier does not call false a classification.
    os.environ.pop("BLENDER_PATH", None)
    try:
        tn = discover_resources()
        tn_payload = {
            "available_tools": tn.get("available_tools"),
            "blender": (tn.get("available_tools") or {}).get("blender"),
        }
        site["probes"].append(
            probe_record(
                name="true_negative_which_and_env_channels_empty",
                payload=tn_payload,
                kind="true_negative",
            )
        )
    finally:
        if orig is None:
            os.environ.pop("BLENDER_PATH", None)
        else:
            os.environ["BLENDER_PATH"] = orig
    return site


# --------------------------------------------------------------------------- main


def site_verdict(site: dict[str, Any]) -> str:
    probes = site.get("probes") or []
    if any(p.get("verdict", "").startswith("FAIL") for p in probes if p.get("kind") != "not_probed"):
        return "FAIL"
    kinds = {p.get("kind") for p in probes}
    if "tool_absent" in kinds or "true_negative" in kinds:
        return "PASS"
    if probes and all(p.get("verdict") == "NOT_PROBED" for p in probes):
        return "NOT_PROBED"
    return "FAIL"


def main() -> int:
    maybe_reexec_for_cv2()
    started = time.perf_counter()
    print("=== VMCP UNAVAILABLE GATE ===")
    print("purpose: make 'I could not look' impossible to mistake for 'there was nothing to see'")
    git = git_head(REPO)
    print(f"git HEAD: {git.get('head')}  {git.get('oneline')}")
    print(f"cwd: {REPO}")
    print(f"interpreter: {sys.executable}")
    print(f"python: {sys.version.split()[0]}")

    src = locate_visionmcp_src()
    sys.path.insert(0, str(src))
    vmcp_root = src.parent
    vmcp_git = git_head(vmcp_root if (vmcp_root / ".git").exists() else src)
    version = "unknown"
    try:
        import visionmcp as vmcp_mod

        version = getattr(vmcp_mod, "__version__", version)
    except Exception as exc:
        print(f"visionmcp import failed: {exc}")
        raise
    print(f"visionmcp: {version}  src={src}")
    print(f"visionmcp git: {vmcp_git.get('head')}  {vmcp_git.get('oneline')}")
    print(f"in_this_worktree: {(REPO / 'visionmcp' / 'src' / 'visionmcp' / '__init__.py').is_file()}")
    print()

    sites: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="vmcp-unavail-") as tmp:
        work = Path(tmp)
        images = make_images(work / "images")
        runners = [
            ("geometry.pipeline.auto_colmap", lambda: site_geometry(images, work)),
            ("cameras.solver.auto_colmap", lambda: site_cameras(images, work)),
            ("compilers.spatial.resolve_cad_backend", lambda: site_cad()),
            ("perception.media.ocr_tesseract", lambda: site_ocr(images, work)),
            ("orchestration.resources.discover_blender", lambda: site_blender()),
        ]
        for site_id, runner in runners:
            print(f"SITE {site_id}")
            try:
                site = runner()
            except Exception as exc:
                traceback.print_exc()
                site = {
                    "id": site_id,
                    "cited": None,
                    "technique": None,
                    "probes": [
                        probe_record(
                            name="site_crashed",
                            payload={"status": "ERROR"},
                            error=exc,
                            kind="tool_absent",
                        )
                    ],
                    "error": traceback.format_exc(),
                }
                errors.append({"id": site_id, "error": str(exc)})
            site["site_verdict"] = site_verdict(site)
            for row in site.get("probes") or []:
                print_probe(site["id"], row)
            print(f"  SITE VERDICT: {site['site_verdict']}")
            print()
            sites.append(site)

    failed_sites = [s for s in sites if s.get("site_verdict") == "FAIL"]
    watched = []
    for site in sites:
        for row in site.get("probes") or []:
            if row.get("verdict", "").startswith("FAIL"):
                watched.append(
                    {
                        "site": site["id"],
                        "probe": row["name"],
                        "verdict": row["verdict"],
                        "return": row.get("return"),
                    }
                )

    print("=== WHAT I WATCHED FAIL ===")
    if not watched:
        print("  (nothing — the gate is measuring the wrong thing if the cited sites still degrade)")
    for item in watched:
        excerpt = json.dumps(item["return"], default=str)
        if len(excerpt) > 360:
            excerpt = excerpt[:359] + "…"
        print(f"  {item['site']} / {item['probe']}: {item['verdict']}")
        print(f"      {excerpt}")
    print()

    overall = "FAIL" if failed_sites or errors else "PASS"
    elapsed = round(time.perf_counter() - started, 3)
    receipt = {
        "schema": "hawking.headless.vmcp_unavailable_gate.v1",
        "gate": "VMCP_UNAVAILABLE_GATE",
        "obligation": (
            "Unavailable visionmcp capabilities must classify as unavailable. "
            "An empty or observed-class return that a caller holding only the "
            "return value cannot distinguish from a genuine negative finding is "
            "a gate failure."
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": elapsed,
        "git": git,
        "git_head": git.get("head"),
        "visionmcp": {
            "version": version,
            "src": str(src),
            "git": vmcp_git,
            "in_this_worktree": (REPO / "visionmcp" / "src" / "visionmcp" / "__init__.py").is_file(),
        },
        "interpreter": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "classification_rule": {
            "counts": sorted(UNAVAIL_STATUS | UNAVAIL_TOKENS),
            "does_not_count": [
                "empty list",
                "text_symbol_count=0",
                "available_tools.blender=false without a token",
                "colmap_fallback_reason buried next to a stored solution",
                "prose limitations about DERIVED interpretations",
                "environment.tesseract=None",
            ],
            "why_fallback_reason_does_not_save_cameras": (
                "CameraSolver.solve('auto') does set diagnostics.colmap_fallback_reason "
                "when COLMAP is missing, but it also stores a real turntable solution "
                "at confidence=0.2. A caller that asks 'do we have cameras?' says yes. "
                "A first-run PASS on that site would mean the gate is wrong."
            ),
        },
        "sites": sites,
        "failed_sites": [s["id"] for s in failed_sites],
        "watched_fail": watched,
        "errors": errors,
        "result": overall,
    }
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")

    print("=== SUMMARY ===")
    for site in sites:
        print(f"  {site['site_verdict']:4}  {site['id']}")
    print(f"result: {overall}")
    print(f"receipt: {RECEIPT_PATH}")
    print(f"elapsed_s: {elapsed}")
    if overall == "PASS":
        print(
            "WARNING: a PASS against the cited auto-degrade sites means this "
            "gate is measuring the wrong thing."
        )
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
