"""Runtime artifact identity and sealed environment identity.

No profiler should silently inspect a binary older than the instrumentation
it is about to interpret. Environment is part of experiment identity: a
benchmark whose env hash does not match the sealed config must not inherit
the incumbent label.

    python3 tools/future/artifact_identity.py --build
    python3 tools/future/artifact_identity.py --inspect BINARY --fields a,b

inspect_artifact REFUSES (raises StaleBinaryError), it does not warn, when
the binary mtime predates the source commit that introduced a field being read.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import (
    RECEIPTS,
    REPO,
    git,
    load_json,
    sha256_file,
    write_receipt,
)

RECEIPT = "ARTIFACT_IDENTITY.json"
SCHEMA = "hawking.future.artifact_identity.v1"
VERSION = 1
RECORDED_BY = "tools/future/artifact_identity.py"
UNKNOWN = "UNKNOWN"

# Production sealed-3.14 fusion env. Cited from
# tools/future/resident_token_budget.py and receipts/future/RESIDENT_TOKEN_BUDGET.json.
SEALED_FUSION_ENV: dict[str, str] = {
    "HAWKING_QWEN38_FUSE_ADD_RMSNORM": "1",
    "HAWKING_QWEN38_FUSE_GQA_QKV": "1",
    "HAWKING_QWEN38_FUSE_DN_INPROJ": "1",
    "HAWKING_QWEN38_FUSE_MLP": "swiglu",
}
FUSION_FLAG_KEYS: tuple[str, ...] = tuple(SEALED_FUSION_ENV)

SEALED_SERVING_MODE = "resident"
SEALED_MEASUREMENT_MODE = "DIAGNOSTIC_RELATIVE"
INCUMBENT_LABEL = "sealed-3.14-production"

DEFAULT_BINARY_REL = (
    "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_resident"
)
DEFAULT_SOURCE_REL = "crates/hawking-core/examples/ascension_qwen38_resident.rs"
DRIFT_RECEIPT_REL = "receipts/future/RESIDENT_BINARY_DRIFT.json"

INSTRUMENTATION_FIELDS: tuple[str, ...] = (
    "dispatches",
    "dispatches_per_generated_token",
    "active_bytes_per_token",
    "active_weight_bytes_per_generated_token",
    "actual_read_bytes_per_token",
    "actual_read_bytes_status",
    "gpu_ns_per_generated_token",
    "resident_weight_bytes",
)


class StaleBinaryError(RuntimeError):
    """The serving binary predates the commit that introduced a field being read.

    This is a refusal, not a warning. Interpreting the field would be reading
    source reality off a running artifact that does not have it.
    """

    def __init__(self, message: str, identity: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.identity = identity or {}


class ArtifactMissingError(FileNotFoundError):
    """The binary path does not exist. Missing is not a successful inspect."""


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _source_dirty(rel: str) -> bool:
    # Avoid `git status`: it can take the index lock. diff + untracked is enough.
    return bool(
        git("diff", "--name-only", "HEAD", "--", rel)
        or git("ls-files", "--others", "--exclude-standard", "--", rel)
    )


def _head_commit() -> dict[str, Any]:
    line = git("log", "-1", "--format=%H%x09%ct%x09%cI%x09%s")
    if not line:
        return {"sha": UNKNOWN, "unix": None, "iso": UNKNOWN, "subject": UNKNOWN}
    sha, unix, iso, subject = (line.split("\t", 3) + [UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN])[:4]
    try:
        unix_f: float | None = float(unix)
    except ValueError:
        unix_f = None
    return {"sha": sha, "unix": unix_f, "iso": iso, "subject": subject}


def _source_commit(rel: str | None) -> dict[str, Any]:
    if rel:
        line = git("log", "-1", "--format=%H%x09%ct%x09%cI%x09%s", "--", rel)
        if line:
            sha, unix, iso, subject = (line.split("\t", 3) + [UNKNOWN] * 4)[:4]
            try:
                unix_f: float | None = float(unix)
            except ValueError:
                unix_f = None
            return {
                "sha": sha,
                "unix": unix_f,
                "iso": iso,
                "subject": subject,
                "path": rel,
            }
    head = _head_commit()
    head["path"] = rel
    return head


def field_introduced_unix(source_rel: str | None, field: str) -> float | None:
    """Committer unix time of the first commit that added or removed `field`."""
    if not source_rel or not field:
        return None
    out = git("log", "--reverse", "-S", field, "--format=%ct", "--", source_rel)
    if not out:
        out = git("log", "--reverse", "-S", field, "--format=%ct")
    if not out:
        return None
    try:
        return float(out.splitlines()[0].strip())
    except ValueError:
        return None


def _nx_identity() -> Any:
    path = RECEIPTS / "RESIDENT_IDENTITY.json"
    if not path.is_file():
        return UNKNOWN
    try:
        doc = load_json(path)
    except (OSError, json.JSONDecodeError, UnicodeError):
        return UNKNOWN
    ident = doc.get("identity") if isinstance(doc, dict) else None
    if not isinstance(ident, dict):
        return UNKNOWN
    nx = ident.get("nx_id")
    return nx if nx not in (None, "", {}, []) else UNKNOWN


def environment_identity(
    env: Mapping[str, str] | None = None,
    *,
    serving_mode: str = "unknown",
    measurement_mode: str = UNKNOWN,
) -> dict[str, Any]:
    """Hash fusion flags, every HAWKING_* toggle, serving mode, measurement mode."""
    raw = dict(env) if env is not None else {
        k: v for k, v in os.environ.items() if k.startswith("HAWKING_")
    }
    hawking = {k: str(v) for k, v in sorted(raw.items()) if str(k).startswith("HAWKING_")}
    fusion = {k: hawking[k] for k in FUSION_FLAG_KEYS if k in hawking}
    payload = {
        "fusion_flags": fusion,
        "hawking_toggles": hawking,
        "serving_mode": str(serving_mode),
        "measurement_mode": str(measurement_mode),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        **payload,
        "env_hash": hashlib.sha256(blob).hexdigest(),
        "n_hawking_toggles": len(hawking),
        "n_fusion_flags": len(fusion),
    }


def sealed_environment() -> dict[str, Any]:
    """The incumbent production environment: sealed fusion + resident serving."""
    ident = environment_identity(
        SEALED_FUSION_ENV,
        serving_mode=SEALED_SERVING_MODE,
        measurement_mode=SEALED_MEASUREMENT_MODE,
    )
    ident["incumbent_label"] = INCUMBENT_LABEL
    ident["is_sealed"] = True
    return ident


def inherits_incumbent(
    observed: Mapping[str, Any] | str,
    sealed: Mapping[str, Any] | None = None,
) -> bool:
    sealed = sealed or sealed_environment()
    obs_hash = observed if isinstance(observed, str) else str(observed.get("env_hash") or "")
    return bool(obs_hash) and obs_hash == str(sealed.get("env_hash") or "")


def benchmark_label(
    observed: Mapping[str, Any] | str,
    sealed: Mapping[str, Any] | None = None,
    *,
    incumbent: str = INCUMBENT_LABEL,
) -> dict[str, Any]:
    """A mismatched env hash is labelled with its own environment, never the incumbent."""
    sealed = sealed or sealed_environment()
    ident = (
        observed
        if isinstance(observed, Mapping)
        else {"env_hash": str(observed)}
    )
    matched = inherits_incumbent(ident, sealed)
    obs_hash = str(ident.get("env_hash") or "")
    label = incumbent if matched else f"env:{obs_hash[:12] or 'unhashed'}"
    return {
        "label": label,
        "inherited_incumbent": matched,
        "incumbent_label": incumbent,
        "env_hash": obs_hash,
        "sealed_env_hash": sealed.get("env_hash"),
        "rule": (
            "A benchmark whose env hash does not match the sealed config "
            "must not inherit the incumbent label automatically."
        ),
    }


def _fields_in_binary(path: Path, fields: Sequence[str]) -> dict[str, bool]:
    try:
        blob = path.read_bytes()
    except OSError:
        return {f: False for f in fields}
    return {f: f.encode("utf-8") in blob for f in fields}


def inspect_artifact(
    binary_path: str | Path,
    *,
    fields: Sequence[str] = (),
    source_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    serving_mode: str = "unknown",
    measurement_mode: str = UNKNOWN,
    nx_id: Any = None,
    feature_flags: Mapping[str, Any] | None = None,
    field_introduced_unix: Mapping[str, float] | None = None,
    refuse_stale: bool = True,
) -> dict[str, Any]:
    """Record artifact identity. RAISE if the binary predates a field's introducing commit.

    Call this BEFORE interpreting any instrumentation field off the binary.
    refuse_stale=True is the default and the contract: a stale binary is a
    refusal, not a warning.
    """
    path = Path(binary_path)
    if not path.is_file():
        raise ArtifactMissingError(f"artifact absent: {path}")

    st = path.stat()
    source = Path(source_path) if source_path is not None else None
    source_rel = None
    if source is not None:
        source_rel = _rel(source) if source.exists() else str(source)
    else:
        # Default source only when inspecting the named resident binary.
        # A temp-file inspect in tests must not inherit the example's git log.
        try:
            inspecting_default = path.resolve() == (REPO / DEFAULT_BINARY_REL).resolve()
        except OSError:
            inspecting_default = False
        if inspecting_default or str(path).endswith(DEFAULT_BINARY_REL):
            source_rel = DEFAULT_SOURCE_REL

    commit = _source_commit(source_rel)
    dirty = _source_dirty(source_rel) if source_rel else False
    env_ident = environment_identity(
        env,
        serving_mode=serving_mode,
        measurement_mode=measurement_mode,
    )
    flags = dict(feature_flags) if feature_flags is not None else dict(env_ident["fusion_flags"])
    nx = nx_id if nx_id is not None else _nx_identity()
    present = _fields_in_binary(path, fields) if fields else {}

    introductions: dict[str, Any] = {}
    stale: list[dict[str, Any]] = []
    for field in fields:
        intro = None
        if field_introduced_unix and field in field_introduced_unix:
            try:
                intro = float(field_introduced_unix[field])
            except (TypeError, ValueError):
                intro = None
        if intro is None:
            intro = field_introduced_unix_lookup(source_rel, field) if source_rel else None
        introductions[field] = intro
        if intro is not None and st.st_mtime < intro:
            stale.append({
                "field": field,
                "binary_mtime_unix": st.st_mtime,
                "field_introduced_unix": intro,
                "delta_s": intro - st.st_mtime,
                "in_binary": present.get(field),
            })

    identity = {
        "binary": {
            "path": str(path if path.is_absolute() else path),
            "rel": _rel(path) if path.exists() else str(path),
            "sha256": sha256_file(path),
            "mtime_unix": st.st_mtime,
            "bytes": st.st_size,
        },
        "source": {
            "path": source_rel,
            "commit": commit.get("sha"),
            "commit_unix": commit.get("unix"),
            "commit_iso": commit.get("iso"),
            "dirty": dirty,
        },
        "feature_flags": flags,
        "environment": env_ident,
        "nx": nx,
        "resident_identity": nx,
        "serving_mode": serving_mode,
        "measurement_mode": measurement_mode,
        "interpreted_fields": list(fields),
        "fields_in_binary": present,
        "field_introduced_unix": introductions,
        "stale_fields": stale,
        "refused": False,
        "refuse_rule": (
            "RAISE when binary mtime predates the source commit that introduced "
            "a field being read. Source reality is not running-artifact reality."
        ),
    }
    label = benchmark_label(env_ident)
    identity["benchmark_label"] = label

    if refuse_stale and stale:
        identity["refused"] = True
        named = ", ".join(s["field"] for s in stale)
        raise StaleBinaryError(
            f"REFUSED: binary {path} mtime {st.st_mtime} predates the commit "
            f"that introduced field(s) {named}. Rebuild before interpreting "
            f"instrumentation.",
            identity=identity,
        )
    return identity


def field_introduced_unix_lookup(source_rel: str, field: str) -> float | None:
    return field_introduced_unix(source_rel, field)


def inspect_before_instrumentation(
    binary_path: str | Path,
    fields: Sequence[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Alias that makes the call site say what it is doing."""
    return inspect_artifact(binary_path, fields=fields, refuse_stale=True, **kwargs)


def historical_stale_finding() -> dict[str, Any]:
    """The campaign case that made this helper exist. Cited, not re-measured."""
    drift_path = RECEIPTS / "RESIDENT_BINARY_DRIFT.json"
    drift: dict[str, Any] | None = None
    if drift_path.is_file():
        try:
            loaded = load_json(drift_path)
            if isinstance(loaded, dict):
                drift = loaded
        except (OSError, json.JSONDecodeError, UnicodeError):
            drift = None
    binary = (drift or {}).get("binary") if isinstance((drift or {}).get("binary"), dict) else {}
    source = (drift or {}).get("source") if isinstance((drift or {}).get("source"), dict) else {}
    return {
        "scar_id": "SOURCE_INSTRUMENTED_RUNTIME_BINARY_STALE",
        "binary_mtime": binary.get("mtime"),
        "binary_path": binary.get("path") or DEFAULT_BINARY_REL,
        "instrumentation_landed_in": source.get("instrumentation_landed_in"),
        "missing_from_binary": (drift or {}).get("missing_from_binary") or list(INSTRUMENTATION_FIELDS),
        "law": (drift or {}).get("law") or (
            "A capability present in source is not a capability present in the "
            "running system. Probe the artifact that serves, not the file that "
            "describes it."
        ),
        "source_receipt": DRIFT_RECEIPT_REL,
    }


def build() -> Path:
    sealed = sealed_environment()
    unfused = environment_identity(
        {},
        serving_mode="probe",
        measurement_mode="DIAGNOSTIC_RELATIVE",
    )
    fused_probe = environment_identity(
        SEALED_FUSION_ENV,
        serving_mode="probe",
        measurement_mode="DIAGNOSTIC_RELATIVE",
    )
    fused_resident = environment_identity(
        SEALED_FUSION_ENV,
        serving_mode=SEALED_SERVING_MODE,
        measurement_mode=SEALED_MEASUREMENT_MODE,
    )
    labels = {
        "unfused_probe": benchmark_label(unfused, sealed),
        "fused_but_probe_serving": benchmark_label(fused_probe, sealed),
        "sealed_fusion_resident": benchmark_label(fused_resident, sealed),
    }

    live: dict[str, Any] | None = None
    live_error: str | None = None
    binary = REPO / DEFAULT_BINARY_REL
    if binary.is_file():
        try:
            live = inspect_artifact(
                binary,
                fields=INSTRUMENTATION_FIELDS,
                source_path=REPO / DEFAULT_SOURCE_REL,
                env=SEALED_FUSION_ENV,
                serving_mode=SEALED_SERVING_MODE,
                measurement_mode=SEALED_MEASUREMENT_MODE,
                refuse_stale=False,
            )
        except (OSError, StaleBinaryError, ArtifactMissingError) as exc:
            live_error = f"{type(exc).__name__}: {exc}"
    else:
        live_error = f"ABSENT:{DEFAULT_BINARY_REL}"

    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Make runtime artifact identity automatic, and make environment "
            "part of experiment identity. A profiler that does not call this "
            "before interpreting instrumentation is the defect "
            "SOURCE_INSTRUMENTED_RUNTIME_BINARY_STALE."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "refuse_rule": (
            "inspect_artifact RAISES StaleBinaryError when the binary mtime "
            "predates the source commit that introduced a field being read. "
            "It does not warn."
        ),
        "incumbent_label": INCUMBENT_LABEL,
        "sealed_environment": sealed,
        "unfused_probe_environment": unfused,
        "labels": labels,
        "unfused_does_not_inherit_incumbent": (
            not labels["unfused_probe"]["inherited_incumbent"]
        ),
        "sealed_does_inherit_incumbent": (
            labels["sealed_fusion_resident"]["inherited_incumbent"]
        ),
        "historical_stale_finding": historical_stale_finding(),
        "live_inspect_refuse_stale_false": live,
        "live_inspect_error": live_error,
        "nx": _nx_identity(),
        "default_binary": DEFAULT_BINARY_REL,
        "default_source": DEFAULT_SOURCE_REL,
        "instrumentation_fields": list(INSTRUMENTATION_FIELDS),
        "fusion_flag_keys": list(FUSION_FLAG_KEYS),
        "recovered_implementation": [
            "tools/future/resident_binary_drift.py compared strings(binary) to source after the fact",
            "tools/future/resident_token_budget.py named the unfused-vs-sealed env split",
            "tools/future/resident_identity.py persists nx_id for the incumbent",
            "nothing refused a binary older than the field it was about to read",
        ],
        "gaps_closed": [
            "artifact identity (path, sha256, mtime, commit, dirty, flags, env, nx) is a call, not a post-mortem receipt",
            "stale binary is a raise, not a warning",
            "env hash over fusion flags + HAWKING_* + serving mode + measurement mode",
            "mismatched env hash cannot inherit the incumbent label",
        ],
        "negative_findings": [
            "the serving binary that motivated this helper predated 8b6f50270 by a day",
            "the first production dispatch count of this campaign was the unfused graph",
            "a fused env with serving_mode=probe still must not inherit the incumbent (serving mode is in the hash)",
        ],
        "resident_callable": {
            "entry_point": "tools.future.artifact_identity.inspect_artifact",
            "env": "tools.future.artifact_identity.environment_identity",
            "label": "tools.future.artifact_identity.benchmark_label",
            "receipt": f"receipts/future/{RECEIPT}",
            "fails_closed": "StaleBinaryError; mismatched env keeps its own label",
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--inspect")
    ap.add_argument("--fields", default="")
    ap.add_argument("--source")
    ap.add_argument("--serving-mode", default="unknown")
    ap.add_argument("--measurement-mode", default=UNKNOWN)
    a = ap.parse_args()
    if a.inspect:
        fields = tuple(f for f in a.fields.split(",") if f.strip())
        try:
            ident = inspect_artifact(
                a.inspect,
                fields=fields,
                source_path=a.source,
                serving_mode=a.serving_mode,
                measurement_mode=a.measurement_mode,
                refuse_stale=True,
            )
        except StaleBinaryError as exc:
            print(json.dumps({"refused": True, "error": str(exc), "identity": exc.identity}, indent=1, default=str))
            return 2
        print(json.dumps(ident, indent=1, default=str))
        return 0
    out = build()
    doc = json.loads(out.read_text())
    print(out)
    print(json.dumps({
        "unfused_does_not_inherit_incumbent": doc["unfused_does_not_inherit_incumbent"],
        "sealed_does_inherit_incumbent": doc["sealed_does_inherit_incumbent"],
        "sealed_env_hash": doc["sealed_environment"]["env_hash"],
        "live_inspect_error": doc["live_inspect_error"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
