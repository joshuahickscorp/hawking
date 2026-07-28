#!/usr/bin/env python3.12
"""Fail-closed release controller for the sealed five-shard GLM pilot window.

The bounded promotion-valid pilot is sealed by the measurement receipt, controller
reseal, revision-0 evidence, and final-ascent rehydration receipt. The five BF16
shard bodies under pilot_source may be evicted once every gate is green; the
content-addressed rehydration route and exact hashes must survive.

Commands (deliberately separated):

    gate      READ-ONLY. Resolve the pilot root, verify the sealed deletion set,
              full-hash the five bodies, reseal/measurement/fence bindings, process
              quiescence, and path isolation. Never deletes.
    status    READ-ONLY. Summarize pilot root residency and prior release receipt.
    release   Re-run the complete gate in-process. Refuse unless every gate is green
              AND --confirm RELEASE_EXACT_SEALED_FIVE_SHARD_PILOT. Delete only the
              five exact verified shard files (no globs, no recursive remove). Seal
              HAWKING_FINAL_ASCENT_PILOT_SOURCE_RELEASE_RECEIPT.json atomically.

This tool must never expand the deletion set beyond the five named shards in the
rehydration receipt. Teacher capsules, compact artifacts, MOP, HIDE, Odyssey, and
authorization fences are out of scope and must remain untouched.

Tests inject a temporary fake world via configure_for_tests(); production defaults
point at the real support tree but the Grok implementation task itself must not
run release against the real pilot source.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

SCHEMA = "hawking.final_ascent.pilot_source_release.v1"
RECEIPT_SCHEMA = "hawking.final_ascent.pilot_source_release_receipt.v1"
CONFIRM_PHRASE = "RELEASE_EXACT_SEALED_FIVE_SHARD_PILOT"

CONDENSE = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = CONDENSE.parents[1]
DEFAULT_SUPPORT = Path(
    os.environ.get(
        "GLM52_SUPPORT_ROOT",
        str(Path.home() / "Library/Application Support/Hawking/GLM52Gravity"),
    )
)
DEFAULT_PILOT_ROOT = DEFAULT_SUPPORT / "pilot_source"

DEFAULT_MEASUREMENT = DEFAULT_REPO_ROOT / "GLM52_BASIS_PILOT_RECEIPT.json"
DEFAULT_RESEAL = DEFAULT_REPO_ROOT / "GLM52_BASIS_PILOT_CONTROLLER_RESEAL.json"
DEFAULT_REVISION_0 = DEFAULT_REPO_ROOT / "GLM52_BASIS_PILOT_REVISION_0_EVIDENCE.json"
DEFAULT_REHYDRATION = (
    DEFAULT_REPO_ROOT / "HAWKING_FINAL_ASCENT_SOURCE_REHYDRATION_RECEIPT.json"
)
DEFAULT_STATUS = DEFAULT_REPO_ROOT / "HAWKING_FINAL_ASCENT_STATUS.json"
DEFAULT_ODYSSEY_AUTH = DEFAULT_REPO_ROOT / "odyssey" / "launch" / "ODYSSEY_LAUNCH_AUTHORIZED"
DEFAULT_RELEASE_RECEIPT = (
    DEFAULT_REPO_ROOT / "HAWKING_FINAL_ASCENT_PILOT_SOURCE_RELEASE_RECEIPT.json"
)
DEFAULT_BASIS_PILOT_PY = CONDENSE / "glm52_basis_pilot.py"
DEFAULT_PACK_PY = CONDENSE / "glm52_activation_aware_pack.py"
DEFAULT_TEST_BASIS = CONDENSE / "tests" / "test_glm52_basis_pilot.py"

RETAINED_NAMES = (
    "REHYDRATE_LEDGER.jsonl",
    "final_ascent_rehydrate.stdout.log",
    "final_ascent_rehydrate.stderr.log",
    "hf_home",
    ".cache",
)

FENCE_KEYS = (
    "ODYSSEY_LAUNCH_AUTHORIZED",
    "RAMANUJAN_RESEARCH_AUTHORIZED",
    "HIDE_KERNEL_TURN",
)


@dataclass
class Paths:
    """All absolute paths the controller may touch or refuse to touch."""

    support_root: Path
    pilot_root: Path
    repo_root: Path
    measurement_receipt: Path
    controller_reseal: Path
    revision_0_evidence: Path
    rehydration_receipt: Path
    final_ascent_status: Path
    odyssey_authorized: Path
    release_receipt: Path
    basis_pilot_py: Path
    activation_aware_pack_py: Path
    test_basis_pilot_py: Path
    capsules: Path
    compact: Path
    mop: Path


@dataclass
class Runtime:
    """Injectable runtime hooks for tests (process probes, free-byte probe)."""

    process_scan: Callable[[Path], dict[str, Any]] | None = None
    free_bytes: Callable[[Path], int] | None = None
    # When True, gate may still run process scan; tests set a custom scan.
    self_pid: int = field(default_factory=os.getpid)


_PATHS = Paths(
    support_root=DEFAULT_SUPPORT,
    pilot_root=DEFAULT_PILOT_ROOT,
    repo_root=DEFAULT_REPO_ROOT,
    measurement_receipt=DEFAULT_MEASUREMENT,
    controller_reseal=DEFAULT_RESEAL,
    revision_0_evidence=DEFAULT_REVISION_0,
    rehydration_receipt=DEFAULT_REHYDRATION,
    final_ascent_status=DEFAULT_STATUS,
    odyssey_authorized=DEFAULT_ODYSSEY_AUTH,
    release_receipt=DEFAULT_RELEASE_RECEIPT,
    basis_pilot_py=DEFAULT_BASIS_PILOT_PY,
    activation_aware_pack_py=DEFAULT_PACK_PY,
    test_basis_pilot_py=DEFAULT_TEST_BASIS,
    capsules=DEFAULT_SUPPORT / "source_fetch" / "teacher" / "capsules",
    compact=DEFAULT_SUPPORT / "compact",
    mop=Path.home() / "Downloads" / "mop",
)
_RUNTIME = Runtime()
# Test instrumentation: count complete gate() invocations in this process.
_GATE_RUNS = 0


def configure_for_tests(paths: Paths, runtime: Runtime | None = None) -> None:
    """Point the controller at a temporary fake world. Production must not call this."""
    global _PATHS, _RUNTIME, _GATE_RUNS
    _PATHS = paths
    _RUNTIME = runtime or Runtime()
    _GATE_RUNS = 0


def reset_to_defaults() -> None:
    """Restore production path defaults (used by tests in teardown)."""
    configure_for_tests(
        Paths(
            support_root=DEFAULT_SUPPORT,
            pilot_root=DEFAULT_PILOT_ROOT,
            repo_root=DEFAULT_REPO_ROOT,
            measurement_receipt=DEFAULT_MEASUREMENT,
            controller_reseal=DEFAULT_RESEAL,
            revision_0_evidence=DEFAULT_REVISION_0,
            rehydration_receipt=DEFAULT_REHYDRATION,
            final_ascent_status=DEFAULT_STATUS,
            odyssey_authorized=DEFAULT_ODYSSEY_AUTH,
            release_receipt=DEFAULT_RELEASE_RECEIPT,
            basis_pilot_py=DEFAULT_BASIS_PILOT_PY,
            activation_aware_pack_py=DEFAULT_PACK_PY,
            test_basis_pilot_py=DEFAULT_TEST_BASIS,
            capsules=DEFAULT_SUPPORT / "source_fetch" / "teacher" / "capsules",
            compact=DEFAULT_SUPPORT / "compact",
            mop=Path.home() / "Downloads" / "mop",
        ),
        Runtime(),
    )


def paths() -> Paths:
    return _PATHS


def gate_run_count() -> int:
    return _GATE_RUNS


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_seal(payload: dict[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != "seal_sha256"}
    return sha256_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _gate(status: bool, reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "green" if status else "red", "reason": reason, **extra}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_no_symlink(path: Path) -> tuple[Path | None, str | None]:
    """Return the absolute path if it exists and is not a symlink; else (None, reason)."""
    if not path.exists():
        return None, f"path does not exist: {path}"
    if path.is_symlink():
        return None, f"path is a symlink (refused): {path}"
    # Refuse if any parent component is a symlink leading outside the stated path.
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        return None, f"path does not resolve: {path}"
    # Compare absolute forms without following a final symlink (already checked).
    absolute = path if path.is_absolute() else path.absolute()
    if resolved != absolute.resolve():
        return None, f"resolved path diverges: {absolute} -> {resolved}"
    # Ensure no intermediate symlink changes identity relative to the configured path.
    if os.path.realpath(path) != str(resolved):
        return None, f"realpath diverges from resolve: {path}"
    return resolved, None


def _free_bytes(path: Path) -> int:
    if _RUNTIME.free_bytes is not None:
        return int(_RUNTIME.free_bytes(path))
    probe = path if path.exists() else path.parent
    return int(shutil.disk_usage(str(probe)).free)


# --------------------------------------------------------------------------- process scan


def default_process_scan(pilot_root: Path) -> dict[str, Any]:
    """lsof (when available) + full argv scan. Fail closed if neither can run."""
    target = str(pilot_root)
    self_pid = str(_RUNTIME.self_pid)
    findings: dict[str, Any] = {
        "lsof": {"available": False},
        "argv": {"available": False},
        "matches": [],
        "self_pid": _RUNTIME.self_pid,
    }

    lsof = shutil.which("lsof")
    if lsof:
        try:
            # +D walks the tree; may be slow on large roots — pilot has five files.
            out = subprocess.run(
                [lsof, "+D", target] if pilot_root.is_dir() else [lsof, "--", target],
                capture_output=True,
                text=True,
                timeout=120,
            )
            lines = []
            for ln in out.stdout.splitlines()[1:]:
                if not ln.strip():
                    continue
                # Exclude this controller's own PID.
                parts = ln.split()
                if parts and parts[1] == self_pid:
                    continue
                lines.append(ln)
            findings["lsof"] = {
                "available": True,
                "open_references": len(lines),
            }
            findings["matches"] += [
                {"probe": "lsof", "line": ln[:200]} for ln in lines
            ]
        except Exception as error:  # noqa: BLE001
            findings["lsof"] = {"available": False, "error": repr(error)[:200]}

    try:
        out = subprocess.run(
            ["ps", "-Axww", "-o", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=60,
        )
        hits = []
        for ln in out.stdout.splitlines():
            if target not in ln:
                continue
            stripped = ln.strip()
            if not stripped:
                continue
            pid_token = stripped.split(None, 1)[0]
            if pid_token == self_pid:
                continue
            hits.append(stripped)
        findings["argv"] = {
            "available": True,
            "referencing_processes": len(hits),
        }
        findings["matches"] += [
            {"probe": "argv", "line": ln[:200]} for ln in hits
        ]
    except Exception as error:  # noqa: BLE001
        findings["argv"] = {"available": False, "error": repr(error)[:200]}

    lsof_ok = bool(findings["lsof"].get("available"))
    argv_ok = bool(findings["argv"].get("available"))
    findings["any_probe_ran"] = lsof_ok or argv_ok
    findings["both_probes_unavailable"] = not findings["any_probe_ran"]
    findings["clean"] = (
        findings["any_probe_ran"]
        and (not lsof_ok or findings["lsof"].get("open_references", 0) == 0)
        and (not argv_ok or findings["argv"].get("referencing_processes", 0) == 0)
        and not findings["matches"]
    )
    return findings


def process_scan(pilot_root: Path) -> dict[str, Any]:
    if _RUNTIME.process_scan is not None:
        return _RUNTIME.process_scan(pilot_root)
    return default_process_scan(pilot_root)


# --------------------------------------------------------------------------- sealed set


def load_sealed_deletion_set() -> dict[str, Any]:
    """The rehydration receipt is the sole authority for the five deletion targets."""
    p = _PATHS
    if not p.rehydration_receipt.is_file() or p.rehydration_receipt.is_symlink():
        raise ValueError(f"rehydration receipt missing or symlink: {p.rehydration_receipt}")
    receipt = _load_json(p.rehydration_receipt)
    shards = receipt.get("shards")
    if not isinstance(shards, list) or len(shards) != 5:
        raise ValueError(f"rehydration receipt must list exactly five shards; got {shards!r}")
    sealed: list[dict[str, Any]] = []
    names: set[str] = set()
    for entry in shards:
        name = entry.get("name")
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if not isinstance(name, str) or not name.endswith(".safetensors"):
            raise ValueError(f"invalid shard name in rehydration receipt: {name!r}")
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid shard bytes for {name}: {size!r}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid shard sha256 for {name}: {digest!r}")
        if name in names:
            raise ValueError(f"duplicate sealed shard name: {name}")
        names.add(name)
        sealed.append({"name": name, "bytes": size, "sha256": digest, "role": entry.get("role")})
    source = receipt.get("source") or {}
    return {
        "shards": sealed,
        "repo": source.get("repo"),
        "revision": source.get("revision"),
        "destination": source.get("destination"),
        "aggregate_bytes": sum(s["bytes"] for s in sealed),
        "receipt_path": str(p.rehydration_receipt),
        "receipt_sha256": sha256_file(p.rehydration_receipt),
    }


# --------------------------------------------------------------------------- gates


def g01_pilot_root_exact() -> dict[str, Any]:
    p = _PATHS
    configured = p.pilot_root if p.pilot_root.is_absolute() else p.pilot_root.absolute()
    if configured.is_symlink():
        return _gate(False, "pilot root is a symlink (refused)", path=str(configured))
    if not configured.exists():
        return _gate(False, "pilot root does not exist", path=str(configured))
    if not configured.is_dir():
        return _gate(False, "pilot root is not a directory", path=str(configured))
    resolved, err = _resolve_no_symlink(configured)
    if err or resolved is None:
        return _gate(False, err or "pilot root resolution failed", path=str(configured))
    support = p.support_root.resolve()
    if support not in resolved.parents and resolved != support:
        return _gate(
            False,
            "pilot root is not below the GLM52Gravity support root",
            path=str(resolved),
            support=str(support),
        )
    if resolved.name != "pilot_source":
        return _gate(False, f"pilot root basename must be pilot_source, got {resolved.name}")
    if resolved != configured.resolve():
        return _gate(
            False,
            "resolved pilot root does not equal configured absolute root",
            configured=str(configured),
            resolved=str(resolved),
        )
    return _gate(
        True,
        "pilot root resolves without symlinks to the configured absolute path under support",
        path=str(resolved),
        support=str(support),
    )


def g02_rehydration_deletion_set() -> dict[str, Any]:
    try:
        sealed = load_sealed_deletion_set()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return _gate(False, f"rehydration deletion set load failed: {error}")
    return _gate(
        True,
        f"exactly five sealed shards from rehydration receipt ({sealed['aggregate_bytes']} bytes)",
        shards=[{"name": s["name"], "bytes": s["bytes"], "sha256": s["sha256"]} for s in sealed["shards"]],
        repo=sealed["repo"],
        revision=sealed["revision"],
        aggregate_bytes=sealed["aggregate_bytes"],
    )


def g03_exact_five_regular_bodies() -> dict[str, Any]:
    p = _PATHS
    try:
        sealed = load_sealed_deletion_set()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return _gate(False, f"cannot load sealed set: {error}")
    root = p.pilot_root
    if not root.is_dir() or root.is_symlink():
        return _gate(False, "pilot root missing or symlink")

    present = sorted(root.glob("model-*.safetensors"))
    present_names = {f.name for f in present}
    sealed_names = {s["name"] for s in sealed["shards"]}
    extra = sorted(present_names - sealed_names)
    missing = sorted(sealed_names - present_names)
    if extra:
        return _gate(
            False,
            f"extra model-*.safetensors bodies present; deletion set will not expand: {extra}",
            extra=extra,
        )
    if missing:
        return _gate(False, f"missing sealed shard bodies: {missing}", missing=missing)

    bad: list[str] = []
    details: list[dict[str, Any]] = []
    for s in sealed["shards"]:
        path = root / s["name"]
        if path.is_symlink():
            bad.append(f"{s['name']}: symlink")
            continue
        if not path.is_file():
            bad.append(f"{s['name']}: not a regular file")
            continue
        # Refuse if path resolves outside pilot root.
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            bad.append(f"{s['name']}: does not resolve")
            continue
        if root.resolve() not in resolved.parents and resolved.parent != root.resolve():
            bad.append(f"{s['name']}: resolves outside pilot root")
            continue
        size = path.stat().st_size
        details.append({"name": s["name"], "path": str(path), "bytes": size})
        if size != s["bytes"]:
            bad.append(f"{s['name']}: size {size} != sealed {s['bytes']}")
    if bad:
        return _gate(False, "shard body identity checks failed", problems=bad, present=details)
    return _gate(
        True,
        "exactly five regular non-symlink sealed shard bodies present; no extras",
        bodies=details,
    )


def g04_full_hash_match() -> dict[str, Any]:
    p = _PATHS
    try:
        sealed = load_sealed_deletion_set()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return _gate(False, f"cannot load sealed set: {error}")
    mismatches: list[dict[str, Any]] = []
    matched: list[dict[str, Any]] = []
    for s in sealed["shards"]:
        path = p.pilot_root / s["name"]
        if not path.is_file() or path.is_symlink():
            mismatches.append({"name": s["name"], "error": "missing or symlink"})
            continue
        live = sha256_file(path)
        entry = {
            "name": s["name"],
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": live,
            "sealed_sha256": s["sha256"],
        }
        if live != s["sha256"]:
            mismatches.append(entry)
        else:
            matched.append(entry)
    if mismatches:
        return _gate(
            False,
            f"full-hash mismatch on {len(mismatches)} shard(s); size-only is insufficient",
            mismatches=mismatches,
            matched=matched,
        )
    return _gate(
        True,
        "all five bodies full-hash match the sealed rehydration values",
        bodies=matched,
    )


def g05_controller_reseal() -> dict[str, Any]:
    p = _PATHS
    for required in (p.controller_reseal, p.measurement_receipt, p.revision_0_evidence):
        if not required.is_file() or required.is_symlink():
            return _gate(False, f"required binding artifact missing or symlink: {required}")
    try:
        reseal = _load_json(p.controller_reseal)
        measurement = _load_json(p.measurement_receipt)
        revision_0 = _load_json(p.revision_0_evidence)
    except (OSError, json.JSONDecodeError) as error:
        return _gate(False, f"failed to load reseal bindings: {error}")

    live_receipt_hash = sha256_file(p.measurement_receipt)
    bound = (reseal.get("measurement_receipt") or {}).get("sha256")
    if bound != live_receipt_hash:
        return _gate(
            False,
            "controller reseal measurement receipt hash does not match live receipt",
            bound=bound,
            live=live_receipt_hash,
        )

    # Revision-0 binding: measurement receipt points at the live revision-0 evidence identity.
    rec_rev0 = measurement.get("revision_0_evidence") or {}
    live_rev0_hash = revision_0.get("sha256")
    if not live_rev0_hash or rec_rev0.get("sha256") != live_rev0_hash:
        return _gate(
            False,
            "revision-0 hash does not match live revision-0 evidence",
            receipt_revision_0=rec_rev0.get("sha256"),
            live_revision_0=live_rev0_hash,
        )
    # Optional explicit reseal binding of the revision-0 evidence *file* hash.
    reseal_rev0 = reseal.get("revision_0_evidence") or reseal.get("revision_0") or {}
    if isinstance(reseal_rev0, dict) and reseal_rev0.get("sha256"):
        if reseal_rev0["sha256"] not in (live_rev0_hash, sha256_file(p.revision_0_evidence)):
            return _gate(
                False,
                "controller reseal revision-0 binding does not match live evidence",
                reseal_revision_0=reseal_rev0.get("sha256"),
            )

    reviewed = reseal.get("reviewed_current_code") or {}
    code_checks = {
        "glm52_basis_pilot_py_sha256": p.basis_pilot_py,
        "glm52_activation_aware_pack_py_sha256": p.activation_aware_pack_py,
        "test_glm52_basis_pilot_py_sha256": p.test_basis_pilot_py,
    }
    code_mismatches: dict[str, Any] = {}
    code_live: dict[str, str] = {}
    for key, path in code_checks.items():
        if not path.is_file():
            code_mismatches[key] = f"missing file: {path}"
            continue
        live = sha256_file(path)
        code_live[key] = live
        expected = reviewed.get(key)
        if expected != live:
            code_mismatches[key] = {"expected": expected, "live": live}
    if code_mismatches:
        return _gate(
            False,
            "reviewed current pilot code/test hashes do not match live files",
            mismatches=code_mismatches,
        )

    post = reseal.get("post_measurement_fix") or {}
    if post.get("measurement_math_changed") is not False:
        return _gate(
            False,
            "measurement_math_changed must be false",
            value=post.get("measurement_math_changed"),
        )
    sci = reseal.get("scientific_disposition") or {}
    if sci.get("full_traversal_authorized") is not False:
        return _gate(
            False,
            "full_traversal_authorized must be false on controller reseal",
            value=sci.get("full_traversal_authorized"),
        )

    return _gate(
        True,
        "controller reseal binds live measurement receipt, revision-0, and reviewed code",
        measurement_receipt_sha256=live_receipt_hash,
        revision_0_sha256=live_rev0_hash,
        revision_0_evidence_file_sha256=sha256_file(p.revision_0_evidence),
        reviewed_current_code=code_live,
        measurement_math_changed=False,
        full_traversal_authorized=False,
    )


def g06_measurement_receipt_safety() -> dict[str, Any]:
    p = _PATHS
    if not p.measurement_receipt.is_file():
        return _gate(False, "measurement receipt missing")
    try:
        measurement = _load_json(p.measurement_receipt)
    except (OSError, json.JSONDecodeError) as error:
        return _gate(False, f"measurement receipt unreadable: {error}")

    inputs = measurement.get("inputs") or {}
    verified = inputs.get("verified_shards") or []
    if len(verified) != 5:
        return _gate(False, f"expected five verified_shards, got {len(verified)}")
    if not all(isinstance(v, dict) and v.get("verified") is True and v.get("sha256") for v in verified):
        return _gate(False, "not all five source hashes are verified=true with sha256")

    try:
        sealed = load_sealed_deletion_set()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return _gate(False, f"sealed set unavailable: {error}")
    sealed_by_name = {s["name"]: s for s in sealed["shards"]}
    for v in verified:
        name = v.get("name")
        if name not in sealed_by_name:
            return _gate(False, f"verified shard not in sealed deletion set: {name}")
        if v.get("sha256") != sealed_by_name[name]["sha256"]:
            return _gate(False, f"verified shard hash diverges from rehydration seal: {name}")
        if v.get("bytes") != sealed_by_name[name]["bytes"]:
            return _gate(False, f"verified shard size diverges from rehydration seal: {name}")

    safety = measurement.get("safety") or {}
    if safety.get("gaussian_proxy_used_for_selection") is not False:
        return _gate(False, "gaussian selection must be false", safety=safety)
    if safety.get("full_parent_traversal_started") is not False:
        return _gate(False, "full_parent_traversal_started must be false", safety=safety)
    fence_bad = {k: safety.get(k) for k in FENCE_KEYS if safety.get(k) is not False}
    if fence_bad:
        return _gate(False, "measurement receipt authorization fences not all false", fences=fence_bad)

    return _gate(
        True,
        "measurement receipt: five hashes verified, no Gaussian selection, no parent traversal, fences false",
        verified_shards=[v.get("name") for v in verified],
        fences={k: False for k in FENCE_KEYS},
    )


def g07_final_ascent_fences() -> dict[str, Any]:
    p = _PATHS
    if not p.final_ascent_status.is_file():
        return _gate(False, "HAWKING_FINAL_ASCENT_STATUS.json missing")
    try:
        status = _load_json(p.final_ascent_status)
    except (OSError, json.JSONDecodeError) as error:
        return _gate(False, f"status unreadable: {error}")
    fences = status.get("fences") or {}
    bad = {k: fences.get(k) for k in FENCE_KEYS if fences.get(k) is not False}
    if bad:
        return _gate(False, "final-ascent status fences not all false", fences=bad)

    # Contract surface: ODYSSEY_LAUNCH_AUTHORIZED file must remain false when present.
    odyssey_detail: dict[str, Any] = {"path": str(p.odyssey_authorized)}
    if p.odyssey_authorized.exists():
        raw = p.odyssey_authorized.read_text(encoding="utf-8").strip().lower()
        odyssey_detail["raw"] = raw
        if raw not in {"false", "0", "no"}:
            return _gate(
                False,
                "ODYSSEY_LAUNCH_AUTHORIZED contract surface is not false",
                odyssey=odyssey_detail,
            )
    return _gate(
        True,
        "final-ascent fences remain false (Odyssey, Ramanujan, HIDE)",
        fences={k: False for k in FENCE_KEYS},
        odyssey=odyssey_detail,
    )


def g08_process_quiescence() -> dict[str, Any]:
    scan = process_scan(_PATHS.pilot_root)
    if scan.get("both_probes_unavailable") or not scan.get("any_probe_ran"):
        return _gate(
            False,
            "no process probe could establish safety; fail closed",
            scan=scan,
        )
    if not scan.get("clean"):
        return _gate(
            False,
            f"live consumer of pilot root detected ({len(scan.get('matches', []))} match(es))",
            scan=scan,
        )
    return _gate(
        True,
        "no process other than this controller opens, maps, or names the pilot root",
        scan=scan,
    )


def g09_path_isolation() -> dict[str, Any]:
    p = _PATHS
    try:
        sealed = load_sealed_deletion_set()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return _gate(False, f"sealed set unavailable: {error}")

    pilot = p.pilot_root.resolve()
    protected = [
        ("repo", p.repo_root.resolve()),
        ("teacher_capsules", p.capsules.resolve() if p.capsules.exists() else p.capsules),
        ("compact", p.compact.resolve() if p.compact.exists() else p.compact),
        ("mop", p.mop.resolve() if p.mop.exists() else p.mop),
        ("support_root", p.support_root.resolve()),
    ]
    problems: list[str] = []
    targets: list[dict[str, Any]] = []
    for s in sealed["shards"]:
        path = p.pilot_root / s["name"]
        if not path.exists():
            problems.append(f"missing target {s['name']}")
            continue
        if path.is_dir():
            problems.append(f"target is a directory (refused): {path}")
            continue
        if path.is_symlink():
            problems.append(f"target is a symlink (refused): {path}")
            continue
        resolved = path.resolve()
        targets.append({"name": s["name"], "path": str(resolved)})
        if pilot not in resolved.parents and resolved.parent != pilot:
            problems.append(f"target outside pilot root: {resolved}")
        if resolved == pilot or resolved.is_dir():
            problems.append(f"target must be a file, not pilot root/dir: {resolved}")
        for label, prot in protected:
            try:
                prot_res = prot.resolve() if prot.exists() else prot
            except OSError:
                prot_res = prot
            if resolved == prot_res:
                problems.append(f"target equals protected {label}: {resolved}")
            # Target must not contain a protected path (file cannot contain dirs, but
            # guard against accidental directory targets already handled).
            if resolved.is_dir() and (
                prot_res == resolved or (prot_res.exists() and resolved in prot_res.parents)
            ):
                problems.append(f"target directory contains/equals protected {label}")
            # Protected path must not be under the target (n/a for files) and target
            # must not be the protected path itself.
            if resolved == p.support_root.resolve():
                problems.append("target equals support root")
    if problems:
        return _gate(False, "path isolation failed", problems=problems, targets=targets)
    return _gate(
        True,
        "five resolved file targets are under pilot root and isolated from protected paths",
        targets=targets,
        protected=[label for label, _ in protected],
    )


def g10_free_and_deletion_bytes() -> dict[str, Any]:
    p = _PATHS
    try:
        sealed = load_sealed_deletion_set()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return _gate(False, f"sealed set unavailable: {error}")
    deletion_bytes = 0
    per: list[dict[str, Any]] = []
    for s in sealed["shards"]:
        path = p.pilot_root / s["name"]
        if path.is_file() and not path.is_symlink():
            size = path.stat().st_size
        else:
            size = s["bytes"]
        deletion_bytes += size
        per.append({"name": s["name"], "bytes": size})
    free_before = _free_bytes(p.pilot_root if p.pilot_root.exists() else p.support_root)
    return _gate(
        True,
        f"recorded free_before={free_before} deletion_bytes={deletion_bytes}",
        free_bytes_before=free_before,
        deletion_bytes=deletion_bytes,
        per_shard_bytes=per,
        sealed_aggregate_bytes=sealed["aggregate_bytes"],
    )


GATES: list[tuple[str, Callable[[], dict[str, Any]]]] = [
    ("g01_pilot_root_exact", g01_pilot_root_exact),
    ("g02_rehydration_deletion_set", g02_rehydration_deletion_set),
    ("g03_exact_five_regular_bodies", g03_exact_five_regular_bodies),
    ("g04_full_hash_match", g04_full_hash_match),
    ("g05_controller_reseal", g05_controller_reseal),
    ("g06_measurement_receipt_safety", g06_measurement_receipt_safety),
    ("g07_final_ascent_fences", g07_final_ascent_fences),
    ("g08_process_quiescence", g08_process_quiescence),
    ("g09_path_isolation", g09_path_isolation),
    ("g10_free_and_deletion_bytes", g10_free_and_deletion_bytes),
]


def gate() -> dict[str, Any]:
    global _GATE_RUNS
    _GATE_RUNS += 1
    results = {name: probe() for name, probe in GATES}
    greens = sum(1 for r in results.values() if r["status"] == "green")
    all_green = all(r["status"] == "green" for r in results.values())
    report = {
        "schema": SCHEMA,
        "evaluated_at": _now(),
        "pilot_root": str(_PATHS.pilot_root),
        "support_root": str(_PATHS.support_root),
        "gates": results,
        "green": greens,
        "total": len(results),
        "all_green": all_green,
        "release_authorized": all_green,
        "confirm_phrase_required": CONFIRM_PHRASE,
    }
    return report


def status() -> dict[str, Any]:
    """Read-only residency summary; never deletes and never writes."""
    p = _PATHS
    sealed: dict[str, Any] | None = None
    sealed_error = None
    try:
        sealed = load_sealed_deletion_set()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        sealed_error = str(error)

    bodies: list[dict[str, Any]] = []
    if p.pilot_root.is_dir() and not p.pilot_root.is_symlink():
        for path in sorted(p.pilot_root.glob("model-*.safetensors")):
            bodies.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size if path.is_file() else None,
                    "is_symlink": path.is_symlink(),
                    "is_file": path.is_file(),
                }
            )

    retained = {}
    for name in RETAINED_NAMES:
        candidate = p.pilot_root / name
        retained[name] = candidate.exists()

    prior = None
    if p.release_receipt.is_file():
        try:
            prior = _load_json(p.release_receipt)
        except (OSError, json.JSONDecodeError) as error:
            prior = {"error": str(error)}

    return {
        "schema": f"{SCHEMA}.status",
        "evaluated_at": _now(),
        "pilot_root": str(p.pilot_root),
        "pilot_root_exists": p.pilot_root.exists(),
        "resident_model_bodies": bodies,
        "resident_count": len(bodies),
        "sealed": sealed,
        "sealed_error": sealed_error,
        "retained_evidence": retained,
        "prior_release_receipt": {
            "path": str(p.release_receipt),
            "present": p.release_receipt.is_file(),
            "summary": (
                {
                    "released_at": prior.get("released_at") if isinstance(prior, dict) else None,
                    "seal_sha256": prior.get("seal_sha256") if isinstance(prior, dict) else None,
                    "all_green": prior.get("gate", {}).get("all_green")
                    if isinstance(prior, dict)
                    else None,
                }
                if isinstance(prior, dict) and "error" not in prior
                else prior
            ),
        },
        "read_only": True,
    }


def verify_receipt_seal(receipt: dict[str, Any]) -> bool:
    seal = receipt.get("seal_sha256")
    if not isinstance(seal, str) or len(seal) != 64:
        return False
    return seal == canonical_seal(receipt)


def release(confirm: str | None) -> dict[str, Any]:
    """Re-run the full gate; delete only the five sealed files; publish receipt."""
    p = _PATHS
    report = gate()
    if not report["all_green"]:
        red = [n for n, r in report["gates"].items() if r["status"] != "green"]
        raise SystemExit(f"release refused: gates not green: {red}")
    if confirm != CONFIRM_PHRASE:
        raise SystemExit(
            f"release refused: confirmation phrase must be exactly {CONFIRM_PHRASE!r}"
        )

    # Refuse replay if a successful sealed receipt already exists and bodies are gone.
    if p.release_receipt.is_file():
        try:
            prior = _load_json(p.release_receipt)
            if (
                verify_receipt_seal(prior)
                and prior.get("deletion", {}).get("all_deleted") is True
            ):
                still = [
                    s["name"]
                    for s in (prior.get("deletion", {}).get("targets") or [])
                    if (p.pilot_root / s["name"]).exists()
                ]
                if not still:
                    raise SystemExit(
                        "release refused: sealed pilot source release receipt already "
                        "records successful deletion; rehydration is the rollback path"
                    )
        except SystemExit:
            raise
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            pass

    sealed = load_sealed_deletion_set()
    free_before = report["gates"]["g10_free_and_deletion_bytes"]["free_bytes_before"]
    hash_bodies = report["gates"]["g04_full_hash_match"]["bodies"]
    by_name = {b["name"]: b for b in hash_bodies}

    deletion_results: list[dict[str, Any]] = []
    deleted_bytes = 0
    for s in sealed["shards"]:
        path = (p.pilot_root / s["name"]).resolve()
        # Final identity check immediately before unlink.
        if not path.is_file() or path.is_symlink():
            deletion_results.append(
                {
                    "name": s["name"],
                    "path": str(path),
                    "deleted": False,
                    "error": "not a regular file at delete time",
                }
            )
            continue
        if p.pilot_root.resolve() not in path.parents:
            deletion_results.append(
                {
                    "name": s["name"],
                    "path": str(path),
                    "deleted": False,
                    "error": "path escaped pilot root at delete time",
                }
            )
            continue
        live_hash = sha256_file(path)
        if live_hash != s["sha256"] or path.stat().st_size != s["bytes"]:
            deletion_results.append(
                {
                    "name": s["name"],
                    "path": str(path),
                    "deleted": False,
                    "error": "hash/size drift at delete time",
                    "live_sha256": live_hash,
                }
            )
            continue
        try:
            # Explicit single-path unlink — never unlink a directory, never glob.
            os.unlink(path)
            gone = not path.exists()
            deletion_results.append(
                {
                    "name": s["name"],
                    "path": str(path),
                    "deleted": gone,
                    "bytes": s["bytes"],
                    "sha256": s["sha256"],
                }
            )
            if gone:
                deleted_bytes += s["bytes"]
        except OSError as error:
            deletion_results.append(
                {
                    "name": s["name"],
                    "path": str(path),
                    "deleted": False,
                    "error": repr(error),
                }
            )

    all_deleted = all(r.get("deleted") for r in deletion_results) and len(deletion_results) == 5
    remaining = [
        s["name"]
        for s in sealed["shards"]
        if (p.pilot_root / s["name"]).exists()
    ]

    retained_after = {name: (p.pilot_root / name).exists() for name in RETAINED_NAMES}
    pilot_dir_retained = p.pilot_root.is_dir()
    free_after = _free_bytes(p.pilot_root if p.pilot_root.exists() else p.support_root)

    reseal = _load_json(p.controller_reseal)
    measurement_hash = sha256_file(p.measurement_receipt)
    reseal_hash = sha256_file(p.controller_reseal)
    rev0_file_hash = sha256_file(p.revision_0_evidence)
    code_hashes = {
        "glm52_basis_pilot_py_sha256": sha256_file(p.basis_pilot_py),
        "glm52_activation_aware_pack_py_sha256": sha256_file(p.activation_aware_pack_py),
        "test_glm52_basis_pilot_py_sha256": sha256_file(p.test_basis_pilot_py),
    }
    release_test = CONDENSE / "tests" / "test_glm52_pilot_source_release.py"
    release_controller_hashes = {
        "glm52_pilot_source_release_py_sha256": sha256_file(Path(__file__)),
        "test_glm52_pilot_source_release_py_sha256": (
            sha256_file(release_test) if release_test.is_file() else None
        ),
    }

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "released_at": _now(),
        "confirm_phrase": CONFIRM_PHRASE,
        "gate": {
            "all_green": report["all_green"],
            "green": report["green"],
            "total": report["total"],
            "evaluated_at": report["evaluated_at"],
            "gates": report["gates"],
        },
        "pilot_root": str(p.pilot_root.resolve()),
        "support_root": str(p.support_root.resolve()),
        "rehydration": {
            "repo": sealed["repo"],
            "revision": sealed["revision"],
            "receipt_path": sealed["receipt_path"],
            "receipt_sha256": sealed["receipt_sha256"],
        },
        "deletion": {
            "targets": [
                {
                    "name": s["name"],
                    "path": str((p.pilot_root / s["name"]).resolve())
                    if (p.pilot_root / s["name"]).exists()
                    else by_name.get(s["name"], {}).get("path", str(p.pilot_root / s["name"])),
                    "bytes": s["bytes"],
                    "sha256": s["sha256"],
                }
                for s in sealed["shards"]
            ],
            "results": deletion_results,
            "aggregate_bytes_sealed": sealed["aggregate_bytes"],
            "deleted_bytes": deleted_bytes,
            "all_deleted": all_deleted,
            "remaining_bodies": remaining,
            "method": "explicit os.unlink per resolved file; no glob; no rmtree",
        },
        "disk": {
            "free_bytes_before": free_before,
            "free_bytes_after": free_after,
            "free_delta_bytes": free_after - free_before,
        },
        "retained": {
            "pilot_directory": pilot_dir_retained,
            "evidence": retained_after,
            "note": "REHYDRATE_LEDGER.jsonl, rehydration logs, hf_home, and pilot dir retained",
        },
        "bindings": {
            "measurement_receipt_sha256": measurement_hash,
            "controller_reseal_sha256": reseal_hash,
            "revision_0_evidence_file_sha256": rev0_file_hash,
            "revision_0_content_sha256": _load_json(p.revision_0_evidence).get("sha256"),
            "reviewed_current_code": code_hashes,
            "release_controller_code": release_controller_hashes,
        },
        "fences": {k: False for k in FENCE_KEYS},
        "rollback": {
            "method": "rehydration by immutable content hash",
            "repo": sealed["repo"],
            "revision": sealed["revision"],
            "note": (
                "Rollback never claims to restore deleted bytes in place; "
                "re-fetch the five sealed shards and verify sha256."
            ),
        },
        "controller_reseal_status": reseal.get("status"),
        "success": all_deleted and pilot_dir_retained,
    }
    receipt["seal_sha256"] = canonical_seal(receipt)

    publish_error = None
    try:
        atomic_write_json(p.release_receipt, receipt)
        written = _load_json(p.release_receipt)
        if not verify_receipt_seal(written):
            publish_error = "written receipt seal verification failed"
            receipt["success"] = False
            receipt["publish_error"] = publish_error
    except OSError as error:
        publish_error = repr(error)
        receipt["success"] = False
        receipt["publish_error"] = publish_error

    if not all_deleted or publish_error:
        state = {
            "partial_deletion": not all_deleted,
            "remaining_bodies": remaining,
            "deletion_results": deletion_results,
            "publish_error": publish_error,
            "receipt_path": str(p.release_receipt),
            "receipt": receipt,
            "note": (
                "Partial deletion or receipt publish failure. "
                "Rollback is rehydration by immutable hash; this controller "
                "never claims to restore deleted bytes."
            ),
        }
        raise SystemExit(json.dumps(state, indent=2, sort_keys=True))

    return receipt


def _print_gate_summary(report: dict[str, Any]) -> None:
    for name, r in report["gates"].items():
        mark = "OK " if r["status"] == "green" else "RED"
        print(f"  {mark} {name}: {r['reason']}")
    print(
        f"\n{report['green']}/{report['total']} green; "
        f"release_authorized={report['release_authorized']}"
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        command = "gate"
        rest: list[str] = []
    else:
        command = argv[0]
        rest = argv[1:]

    if command == "gate":
        report = gate()
        _print_gate_summary(report)
        return 0 if report["all_green"] else 1

    if command == "status":
        print(json.dumps(status(), indent=2, sort_keys=True))
        return 0

    if command == "release":
        confirm: str | None = None
        if "--confirm" in rest:
            idx = rest.index("--confirm")
            if idx + 1 >= len(rest):
                print("release refused: --confirm requires the exact phrase", file=sys.stderr)
                return 2
            confirm = rest[idx + 1]
        try:
            receipt = release(confirm)
        except SystemExit as exit_exc:
            code = exit_exc.code
            if isinstance(code, str):
                print(code, file=sys.stderr)
                return 1
            if isinstance(code, int):
                return code
            return 1
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt.get("success") else 1

    print(f"unknown command: {command}", file=sys.stderr)
    print("usage: gate | status | release --confirm RELEASE_EXACT_SEALED_FIVE_SHARD_PILOT",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
