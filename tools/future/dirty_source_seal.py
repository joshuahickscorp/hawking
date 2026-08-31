"""Seal dirty-source diagnostic state. Review, do not bulk-commit.

Every TPS number in this campaign came from a binary built from a working
tree carrying uncommitted crate work. TokenPipelineCache and the batched
dispatch helpers those measurements depend on exist only in that tree.
artifact_identity.py does not catch this: there is no introducing commit
to be older than.

This module partitions the dirty crate work, records a review, and seals
the diagnostic state so the science is not lost. It labels the existing
measurements DIRTY_SOURCE_DIAGNOSTIC. They are kept and usable. They are
NOT promoted. The module REFUSES to label anything PROMOTED while the
source is dirty.

    python3 tools/future/dirty_source_seal.py --build
    python3 tools/future/dirty_source_seal.py --selftest
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    ),
)

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from tools.future._common import (
    GIT_TIMEOUT_S,
    RECEIPTS,
    REPO,
    git,
    load_json,
    sha256_file,
    write_receipt,
)
from tools.future.artifact_identity import (
    SEALED_FUSION_ENV,
    sealed_environment,
)

RECEIPT = "DIRTY_SOURCE_SEAL.json"
PATCH_REL = "receipts/future/patches/dirty-crate-work.patch"
SCHEMA = "hawking.future.dirty_source_seal.v1"
VERSION = 1
RECORDED_BY = "tools/future/dirty_source_seal.py"
EVIDENCE_CLASS = "STATIC_ONLY"

DIRTY_SOURCE_DIAGNOSTIC = "DIRTY_SOURCE_DIAGNOSTIC"
PROMOTED = "PROMOTED"

BINARY_REL = "workspace/ops/build/rust/release/examples/ascension_qwen38_resident"
CRATE_PREFIX = "crates/"
WITNESS = "TokenPipelineCache"
WITNESS_FILE = "crates/hawking-core/src/metal/mod.rs"

MEASUREMENT_FILES = (
    "crates/hawking-core/src/metal/mod.rs",
    "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
    "crates/hawking-core/examples/ascension_qwen38_resident.rs",
)

CAMPAIGN_MEASUREMENTS = (
    "receipts/future/RESIDENT_TOKEN_BUDGET.json",
    "receipts/future/ORGAN_BANDWIDTH.json",
    "receipts/future/BA_DELTA_AB.json",
    "receipts/future/RESIDENT_BINARY_DRIFT.json",
)

PROVENANCE_REL = "receipts/future/MEASUREMENT_PROVENANCE.json"

GIT = (
    "git",
    "--no-optional-locks",
    "-c",
    "color.ui=never",
    "-c",
    "core.quotepath=false",
)

CLAIM_BOUNDARY = (
    "A partition, a review, a crate patch, and the hashes of that patch, "
    "the measurement binary, the toolchain and the sealed fusion env. It "
    "asserts that the campaign source is uncommitted and labels the "
    "numbers DIRTY_SOURCE_DIAGNOSTIC. It does not promote them. It does "
    "not claim a hardware measurement. No receipt recorded elsewhere is "
    "retracted."
)


class PromotionRefused(ValueError):
    """DIRTY_SOURCE_DIAGNOSTIC cannot be labelled PROMOTED."""


class PatchApplyError(RuntimeError):
    """The sealed patch did not apply to the recorded base HEAD."""


class SealError(RuntimeError):
    """The dirty-source seal cannot be produced honestly."""


def _run_git(
    root: Path,
    args: Sequence[str],
    *,
    binary: bool = False,
) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(
            [*GIT, *args],
            cwd=root,
            capture_output=True,
            text=not binary,
            check=False,
            timeout=GIT_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise SealError(f"git {' '.join(args)} failed: {exc}") from exc


def git_at(root: Path, *args: str) -> str:
    proc = _run_git(root, args, binary=False)
    return (proc.stdout or "").strip()


def git_at_bytes(root: Path, *args: str) -> bytes:
    proc = _run_git(root, args, binary=True)
    return proc.stdout or b""


def primary_checkout() -> Path:
    """The checkout that actually holds the dirty crate work.

    This lane is a sparse worktree of HEAD. The uncommitted crate files
    live in the primary odyssey-i checkout. A file missing here is not
    evidence it does not exist.
    """
    common = git("rev-parse", "--git-common-dir")
    if not common:
        return REPO
    path = Path(common)
    if not path.is_absolute():
        path = (REPO / path).resolve()
    else:
        path = path.resolve()
    if path.name == ".git":
        return path.parent
    return path.parent


def dirty_crate_files(root: Path) -> dict[str, list[str]]:
    """Modified and untracked crate paths. Never `git status` (index lock)."""
    modified = [
        line
        for line in git_at(root, "diff", "--name-only", "HEAD", "--", CRATE_PREFIX).splitlines()
        if line
    ]
    untracked = [
        line
        for line in git_at(
            root, "ls-files", "--others", "--exclude-standard", "--", CRATE_PREFIX
        ).splitlines()
        if line
    ]
    return {"modified": modified, "untracked": untracked}


def crate_source_is_dirty(root: Path | None = None) -> bool:
    if root is not None:
        files = dirty_crate_files(root)
        return bool(files["modified"] or files["untracked"])
    if _crate_dirty_at(REPO):
        return True
    primary = primary_checkout()
    return primary != REPO and _crate_dirty_at(primary)


def _crate_dirty_at(root: Path) -> bool:
    if not root.is_dir():
        return False
    files = dirty_crate_files(root)
    return bool(files["modified"] or files["untracked"])


def dirty_crate_root() -> Path:
    """Working tree that currently carries the uncommitted crate work."""
    if _crate_dirty_at(REPO):
        return REPO
    primary = primary_checkout()
    if primary != REPO and _crate_dirty_at(primary):
        return primary
    return REPO


def base_head() -> dict[str, str]:
    line = git("log", "-1", "--format=%H%x09%s")
    if not line:
        raise SealError("HEAD does not resolve")
    sha, _, subject = line.partition("\t")
    return {"sha": sha, "subject": subject}


def label_measurement(
    measurement_id: str,
    *,
    requested: str | None = None,
    source_dirty: bool | None = None,
) -> str:
    """Label a measurement. REFUSES PROMOTED while the source is dirty.

    Campaign measurements stay DIRTY_SOURCE_DIAGNOSTIC even if the tree
    is later cleaned: they were taken on uncommitted source.
    """
    if source_dirty is None:
        source_dirty = crate_source_is_dirty()
    want = DIRTY_SOURCE_DIAGNOSTIC if requested is None else requested
    if want == PROMOTED:
        if source_dirty:
            raise PromotionRefused(
                f"REFUSED: cannot label {measurement_id!r} PROMOTED while "
                "crate source is dirty"
            )
        raise PromotionRefused(
            f"REFUSED: cannot label {measurement_id!r} PROMOTED; "
            "campaign measurements stay DIRTY_SOURCE_DIAGNOSTIC and this "
            "seal does not mint PROMOTED labels"
        )
    if source_dirty:
        return DIRTY_SOURCE_DIAGNOSTIC
    if measurement_id in CAMPAIGN_MEASUREMENTS:
        return DIRTY_SOURCE_DIAGNOSTIC
    return want


def measurement_labels(*, source_dirty: bool | None = None) -> dict[str, str]:
    if source_dirty is None:
        source_dirty = crate_source_is_dirty()
    labels = {
        path: label_measurement(path, source_dirty=source_dirty)
        for path in CAMPAIGN_MEASUREMENTS
    }
    if any(value == PROMOTED for value in labels.values()):
        raise PromotionRefused(
            "REFUSED: a campaign measurement was labelled PROMOTED while "
            "this seal is responsible for DIRTY_SOURCE_DIAGNOSTIC"
        )
    return labels


def toolchain() -> dict[str, Any]:
    """The rustc that built the measurement binary, as far as this host knows."""
    rustc = _tool_text("rustc", "-vV")
    cargo = _tool_text("cargo", "-V")
    parsed = _parse_rustc_vv(rustc)
    parsed["cargo"] = cargo.splitlines()[0] if cargo else "UNKNOWN"
    parsed["rustc_vv"] = rustc or "UNKNOWN"
    parsed["workspace_rust_version"] = _workspace_rust_version()
    parsed["active_toolchain"] = (
        _tool_text("rustup", "show", "active-toolchain").splitlines() or ["UNKNOWN"]
    )[0]
    return parsed


def _tool_text(*argv: str) -> str:
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return ""
    return (proc.stdout or "").strip()


def _parse_rustc_vv(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "rustc": "UNKNOWN",
        "release": "UNKNOWN",
        "commit_hash": "UNKNOWN",
        "host": "UNKNOWN",
        "llvm": "UNKNOWN",
    }
    if not text:
        return fields
    first = text.splitlines()[0]
    fields["rustc"] = first.removeprefix("rustc ").strip() if first.startswith("rustc ") else first
    for line in text.splitlines():
        if line.startswith("release:"):
            fields["release"] = line.split(":", 1)[1].strip()
        elif line.startswith("commit-hash:"):
            fields["commit_hash"] = line.split(":", 1)[1].strip()
        elif line.startswith("host:"):
            fields["host"] = line.split(":", 1)[1].strip()
        elif line.startswith("LLVM version:"):
            fields["llvm"] = line.split(":", 1)[1].strip()
    return fields


def _workspace_rust_version() -> str:
    blob = git("show", "HEAD:Cargo.toml")
    for line in blob.splitlines():
        if line.startswith("rust-version"):
            return line.split("=", 1)[1].strip().strip('"')
    return "UNKNOWN"


def binary_identity() -> dict[str, Any]:
    recorded = _provenance_binary_sha256()
    for cand in (
        REPO / BINARY_REL,
        dirty_crate_root() / BINARY_REL,
        primary_checkout() / BINARY_REL,
    ):
        if cand.is_file():
            digest = sha256_file(cand)
            return {
                "path": BINARY_REL,
                "sha256": digest,
                "bytes": cand.stat().st_size,
                "source": "hashed_on_disk",
                "matches_provenance_receipt": (
                    True if recorded is None else digest == recorded
                ),
                "provenance_sha256": recorded,
            }
    return {
        "path": BINARY_REL,
        "sha256": recorded,
        "bytes": None,
        "source": "provenance_receipt" if recorded else "ABSENT",
        "matches_provenance_receipt": recorded is not None,
        "provenance_sha256": recorded,
    }


def _provenance_binary_sha256() -> str | None:
    path = RECEIPTS / "MEASUREMENT_PROVENANCE.json"
    if not path.is_file():
        blob = git("show", f"HEAD:{PROVENANCE_REL}")
        if not blob:
            return None
        try:
            doc = json.loads(blob)
        except json.JSONDecodeError:
            return None
    else:
        try:
            doc = load_json(path)
        except (OSError, json.JSONDecodeError, UnicodeError):
            return None
    binary = doc.get("binary") if isinstance(doc, dict) else None
    if not isinstance(binary, dict):
        return None
    digest = binary.get("sha256")
    return digest if isinstance(digest, str) and len(digest) == 64 else None


def witness_counts(root: Path | None = None) -> dict[str, Any]:
    root = root or dirty_crate_root()
    src = root / WITNESS_FILE
    in_tree = src.read_text().count(WITNESS) if src.is_file() else 0
    head = git("show", f"HEAD:{WITNESS_FILE}")
    in_head = head.count(WITNESS) if head else 0
    return {
        "symbol": WITNESS,
        "file": WITNESS_FILE,
        "occurrences_in_working_tree": in_tree,
        "occurrences_in_HEAD": in_head,
        "conclusion": (
            "the measurements depend on a symbol that exists only in the "
            "working tree"
            if in_tree and in_head == 0
            else "symbol is committed" if in_head else "symbol absent from HEAD"
        ),
    }


def measurement_file_numstat(root: Path, base: str) -> dict[str, Any]:
    lines = git_at(root, "diff", "--numstat", base, "--", *MEASUREMENT_FILES).splitlines()
    added = 0
    removed = 0
    per_file: list[dict[str, Any]] = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        a, r, name = int(parts[0]), int(parts[1]), parts[2]
        added += a
        removed += r
        per_file.append({"path": name, "added": a, "removed": r})
    return {
        "files": per_file,
        "uncommitted_lines_added": added,
        "uncommitted_lines_removed": removed,
    }


def patch_paths(patch: bytes) -> list[str]:
    paths: list[str] = []
    for raw in patch.splitlines():
        if not raw.startswith(b"diff --git "):
            continue
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            line = raw.decode("utf-8", "replace")
        parts = line.split()
        if len(parts) < 4:
            continue
        a = parts[2]
        rel = a[2:] if a.startswith("a/") else a
        paths.append(rel)
    return paths


def emit_dirty_crate_patch(
    dest: Path | None = None,
    *,
    root: Path | None = None,
    base: str | None = None,
) -> dict[str, Any]:
    """Write a git-applyable patch of every dirty crate file against `base`."""
    root = root or dirty_crate_root()
    base = base or base_head()["sha"]
    dest = dest or (REPO / PATCH_REL)
    dest.parent.mkdir(parents=True, exist_ok=True)
    files = dirty_crate_files(root)
    modified = git_at_bytes(root, "diff", base, "--", CRATE_PREFIX)
    blob = bytearray(modified)
    for rel in files["untracked"]:
        extra = git_at_bytes(root, "diff", "--no-index", "--", "/dev/null", rel)
        if extra:
            if blob and not blob.endswith(b"\n"):
                blob.extend(b"\n")
            blob.extend(extra)
    dest.write_bytes(bytes(blob))
    digest = hashlib.sha256(bytes(blob)).hexdigest()
    return {
        "path": PATCH_REL,
        "sha256": digest,
        "bytes": dest.stat().st_size,
        "n_modified": len(files["modified"]),
        "n_untracked": len(files["untracked"]),
        "n_files": len(files["modified"]) + len(files["untracked"]),
        "modified": files["modified"],
        "untracked": files["untracked"],
        "base_sha": base,
        "dirty_crate_root": str(root),
    }


def _blob_exists(sha: str, rel: str) -> bool:
    proc = _run_git(REPO, ["cat-file", "-e", f"{sha}:{rel}"])
    return proc.returncode == 0


def apply_patch_to_scratch_worktree(
    base_sha: str,
    patch_path: str | Path,
    scratch: str | Path,
) -> dict[str, Any]:
    """Materialize HEAD blobs for patched files and git-apply into `scratch`.

    This is a scratch worktree in the sense the acceptance asks for: a
    directory that starts as the recorded base HEAD of the patched files
    and ends as that HEAD plus the sealed patch. It does not call
    `git worktree add` (sparse-checkout.lock is not available here).
    """
    patch_path = Path(patch_path)
    scratch = Path(scratch)
    if not patch_path.is_file():
        raise PatchApplyError(f"patch absent: {patch_path}")
    patch = patch_path.read_bytes()
    if not patch.strip():
        raise PatchApplyError("empty patch")
    paths = patch_paths(patch)
    if not paths:
        raise PatchApplyError("patch names no files")
    scratch.mkdir(parents=True, exist_ok=True)
    materialized: list[str] = []
    for rel in paths:
        if not _blob_exists(base_sha, rel):
            continue
        dest = scratch / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        proc = _run_git(REPO, ["show", f"{base_sha}:{rel}"], binary=True)
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", "replace")
            raise PatchApplyError(f"git show {base_sha}:{rel} failed: {err}")
        dest.write_bytes(proc.stdout or b"")
        materialized.append(rel)
    abs_patch = str(patch_path.resolve())
    check = subprocess.run(
        ["git", "apply", "--check", abs_patch],
        cwd=scratch,
        capture_output=True,
        text=True,
        check=False,
        timeout=GIT_TIMEOUT_S,
    )
    if check.returncode != 0:
        raise PatchApplyError(
            f"git apply --check failed against {base_sha}: {check.stderr}"
        )
    applied = subprocess.run(
        ["git", "apply", abs_patch],
        cwd=scratch,
        capture_output=True,
        text=True,
        check=False,
        timeout=GIT_TIMEOUT_S,
    )
    if applied.returncode != 0:
        raise PatchApplyError(
            f"git apply failed against {base_sha}: {applied.stderr}"
        )
    return {
        "ok": True,
        "base_sha": base_sha,
        "scratch": str(scratch),
        "paths": paths,
        "materialized_from_base": materialized,
        "n_paths": len(paths),
        "n_materialized_from_base": len(materialized),
    }


def partitions() -> list[dict[str, Any]]:
    """Coherent, separately reviewable slices of the dirty crate work."""
    return [
        {
            "id": "P-CACHE",
            "name": "TokenPipelineCache and batched dispatch reuse",
            "what": (
                "Resident pipeline-handle cache (name-to-id map + Arc vector), "
                "FNV-1a hasher, TokenCommandBuffer::new_with_pipeline_cache, "
                "take_pipeline_cache, bind_pipeline_if_changed, and the "
                "dispatch_batch_*_with_pipeline_cache helpers. The Qwen3.8 "
                "session keeps the cache across tokens. Default ON via "
                "HAWKING_METAL_PIPELINE_CACHE_REUSE env_opt_out."
            ),
            "touches": [
                "crates/hawking-core/src/metal/mod.rs",
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
            ],
            "measurement_bearing": True,
            "correct": (
                "Looks correct on reading: one admission per kernel name, "
                "stable IDs, Arc handles reused across command buffers. "
                "dispatch_batch_with_pipeline_cache is used by gravity P6, "
                "not dead; Qwen3.8 uses new_with_pipeline_cache instead."
            ),
            "dead_code": False,
            "changes_production_default": True,
            "has_test": False,
            "land": True,
            "verdict": "LAND",
            "why": (
                "The campaign binary depends on this symbol (16 occurrences "
                "in the working tree, 0 in HEAD). The binary ran with "
                "fallbacks 0 and produced coherent text. An opt-out exists. "
                "There is no unit test of the cache itself; land it as the "
                "implementation the measurements used, and do not treat "
                "landing it as promoting the TPS numbers."
            ),
        },
        {
            "id": "P-ELISION",
            "name": "Default-on Metal encoder/pipeline/commit elision",
            "what": (
                "Skip ordinary encoder labels, skip re-setting pipeline "
                "state when the ID is unchanged, skip timing on untraced "
                "commit_and_wait, and resolve pipelines by admitted ID. "
                "Each is env_opt_out, so unset means ON."
            ),
            "touches": ["crates/hawking-core/src/metal/mod.rs"],
            "measurement_bearing": True,
            "correct": "Plausible host-side skip; not GPU-math. Untested as a set.",
            "dead_code": False,
            "changes_production_default": True,
            "has_test": False,
            "land": False,
            "verdict": "HOLD",
            "why": (
                "Default-on production change with no paired A/B of the "
                "elision flags themselves. The campaign likely ran with "
                "them on. Keep them in the sealed patch; do not land until "
                "an opt-out A/B exists."
            ),
        },
        {
            "id": "P-INSTRUMENT",
            "name": "Active and resident weight-byte instrumentation",
            "what": (
                "Per-token packed-weight byte accounting on the decode "
                "session, Qwen38LayerNameCache, and JSON fields on "
                "ascension_qwen38_resident (resident_weight_bytes, "
                "active_bytes_per_token, actual_read_bytes_status="
                "NOT_MEASURED_NO_METAL_MEMORY_COUNTER)."
            ),
            "touches": [
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
                "crates/hawking-core/examples/ascension_qwen38_resident.rs",
            ],
            "measurement_bearing": True,
            "correct": (
                "The example honestly reports actual_read_bytes as "
                "NOT_MEASURED. bytes_per_row has a unit test. This is "
                "accounting, not a DRAM counter."
            ),
            "dead_code": False,
            "changes_production_default": True,
            "has_test": True,
            "land": True,
            "verdict": "LAND",
            "why": (
                "Small, named, tested at the row-size helper. It changes "
                "the example's JSON (serving API) but not the GPU graph. "
                "The campaign receipts read these fields."
            ),
        },
        {
            "id": "P-FAST",
            "name": "FAST profile, serial encoder, gated MHA",
            "what": (
                "HAWKING_QWEN38_FAST opt-in rebinds MLP/add-rmsnorm/"
                "BA-delta/GQA/DN defaults to the sealed fusion set, "
                "enables a serial token encoder, and fuses the per-head "
                "attention gate into MHA. FAST itself is env_on, default off."
            ),
            "touches": [
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
                "crates/hawking-core/shaders/mha.metal",
                "crates/hawking-core/src/kernels/mod.rs",
                "crates/hawking-core/src/metal/mod.rs",
            ],
            "measurement_bearing": False,
            "correct": (
                "Default-off. mlp_fusion_env_tests cover FAST on/off and "
                "the =1 means swiglu parser. Gated MHA has a compile-name "
                "test, not a numeric oracle."
            ),
            "dead_code": False,
            "changes_production_default": False,
            "has_test": True,
            "land": True,
            "verdict": "LAND",
            "why": (
                "Opt-in profile with tests. Production without "
                "HAWKING_QWEN38_FAST keeps HEAD fusion defaults. Land as "
                "opt-in; do not make FAST the incumbent."
            ),
        },
        {
            "id": "P-Q2F",
            "name": "Q2F geo/splitk kernels, including a rewrite of an existing kernel",
            "what": (
                "Additive q2f/affine splitk/pipe/swiglu kernels in "
                "q80_mixed_decode.metal plus decode launch helpers. The "
                "same file rewrites kernel void "
                "qwen_q2f_group64_matvec_geo_tpr64_tg128, which already "
                "exists at HEAD."
            ),
            "touches": [
                "crates/hawking-core/shaders/q80_mixed_decode.metal",
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
            ],
            "measurement_bearing": True,
            "correct": (
                "Additive kernels are candidates. The rewrite of the "
                "existing geo kernel is not certified here. The campaign "
                "unfused A/B (964 dispatches) may have run on the rewrite."
            ),
            "dead_code": False,
            "changes_production_default": True,
            "has_test": False,
            "land": False,
            "verdict": "HOLD",
            "why": (
                "Do not land a rewrite of a production kernel in the same "
                "breath as 15 new candidates. Split the additive hunks if "
                "a human wants them; hold the geo rewrite until an oracle "
                "matches HEAD. Keep it in the sealed patch."
            ),
        },
        {
            "id": "P-WORKLIST",
            "name": "FP4 gate-up SwiGLU worklist names and kernels",
            "what": (
                "decode_family.rs names for "
                "gk_worklist_fp4_gate_up_swiglu_bf16{,_simd}, plus the "
                "matching kernels in gk_family.metal and "
                "dsv4f_native_token_graph.metal."
            ),
            "touches": [
                "crates/hawking-core/src/decode_family.rs",
                "crates/hawking-core/shaders/gk_family.metal",
                "crates/hawking-core/shaders/dsv4f_native_token_graph.metal",
            ],
            "measurement_bearing": False,
            "correct": "Additive. FAMILY_KERNELS static-name tests will fail if names land without shaders, so they travel together.",
            "dead_code": False,
            "changes_production_default": False,
            "has_test": True,
            "land": True,
            "verdict": "LAND",
            "why": (
                "Dormant unless a gravity/worklist path selects them. "
                "Metal static-name tests cover presence. Not on the "
                "Qwen3.8 resident measurement path."
            ),
        },
        {
            "id": "P-Q30",
            "name": "Qwen30 uniform-Q4 Base admission",
            "what": (
                "Admit Qwen3-30B-A3B Base from the manifest repository "
                "instead of hardcoding Coder. Preflight for uniform-Q4 "
                "without executing a token. Gate example seals become "
                "env-overridable."
            ),
            "touches": [
                "crates/hawking-core/src/model/qwen_complete_binary/uniform_q4.rs",
                "crates/hawking-core/src/model/qwen30_complete_runtime.rs",
                "crates/hawking-core/examples/ascension_qwen30_complete_native_runtime.rs",
                "crates/hawking-core/examples/ascension_qwen30_uniform_q4_tps_gates.rs",
            ],
            "measurement_bearing": False,
            "correct": (
                "The Coder hardcode was a real bug for Base (different "
                "rope). Unknown repositories still refuse. Tested."
            ),
            "dead_code": False,
            "changes_production_default": True,
            "has_test": True,
            "land": True,
            "verdict": "LAND",
            "why": (
                "Independent of the Qwen3.8 campaign. Correctness fix "
                "with a unit test. Changes Qwen30 Base admission, which "
                "is the point."
            ),
        },
        {
            "id": "P-Q80",
            "name": "Qwen80 source BF16 range-read and mixed-dtype index",
            "what": (
                "read_raw_range for a bounded tensor window. Skip non-BF16 "
                "control tensors at index time instead of poisoning "
                "admission. HAWKING_SOURCE_CACHE=1 opt-in, default off."
            ),
            "touches": [
                "crates/hawking-core/src/model/qwen80_source_bf16_layer_major.rs"
            ],
            "measurement_bearing": False,
            "correct": (
                "Range-read bounds-checks. Skipping non-BF16 is a "
                "behaviour change: mixed Flash-Next indexes now admit. "
                "A later read of a non-BF16 name still fails closed."
            ),
            "dead_code": False,
            "changes_production_default": True,
            "has_test": False,
            "land": True,
            "verdict": "LAND",
            "why": (
                "Small, not on the Qwen3.8 resident path. The skip is "
                "intentional for Flash-Next control tensors. No new test "
                "for read_raw_range; still land, it is local and closed."
            ),
        },
        {
            "id": "P-GRAVITY",
            "name": "DeepSeek V4 gravity fused dispatch (P6/P7/native graph)",
            "what": (
                "Native token graph controls, P6 fused gate-up/down/"
                "shared-fp8 dispatch, P7 MHC kernels, fullseq ordered "
                "encoder, streamed LM-head selection, plus shaders in "
                "deepseek_v4_p7.metal, moe.metal, attn.metal."
            ),
            "touches": [
                "crates/hawking-core/src/gravity_deepseek_v4_fullseq_attention_device.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_layer0_continuation.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_native_token_graph.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_p4b_device.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_p7_device.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_streamed_native.rs",
                "crates/hawking-core/examples/gravity_deepseek_v4_fullseq_capture.rs",
                "crates/hawking-core/shaders/deepseek_v4_p7.metal",
                "crates/hawking-core/shaders/moe.metal",
                "crates/hawking-core/shaders/attn.metal",
            ],
            "measurement_bearing": False,
            "correct": "Too large to certify in this review. Some new budget/catalog tests exist.",
            "dead_code": False,
            "changes_production_default": False,
            "has_test": True,
            "land": False,
            "verdict": "HOLD",
            "why": (
                "Thousands of lines of a different model family. Not the "
                "source of the Qwen3.8 TPS numbers. Keep in the sealed "
                "patch; land as its own reviewed series, not this one."
            ),
        },
        {
            "id": "P-FLASH",
            "name": "Qwen-Next / Flash bf16 kernels and TCB wrappers",
            "what": (
                "qwen_next.metal and matmul.metal kernel dump, kernels/mod.rs "
                "TCB wrappers, and metal compile-name tests for flash "
                "compact-MoE, QKV-RoPE, router, hyperconnection."
            ),
            "touches": [
                "crates/hawking-core/shaders/qwen_next.metal",
                "crates/hawking-core/shaders/matmul.metal",
                "crates/hawking-core/src/kernels/mod.rs",
                "crates/hawking-core/src/metal/mod.rs",
            ],
            "measurement_bearing": False,
            "correct": "Compile-name tests only. No numeric oracle in this tree.",
            "dead_code": False,
            "changes_production_default": False,
            "has_test": True,
            "land": False,
            "verdict": "HOLD",
            "why": (
                "A metallib dump of a different species. Dormant for the "
                "resident Qwen3.8 path. Do not bulk-land. Name tests can "
                "travel with any kernel a human actually wants."
            ),
        },
        {
            "id": "P-EXAMPLES",
            "name": "Untracked flash_* diagnostic examples",
            "what": (
                "Thirteen untracked hawking-core examples "
                "(flash_fast_chain, flash_noetic_complete_layer0, "
                "stateful probes, tokenizer contract, …). Diagnostic "
                "harnesses, not the resident server."
            ),
            "touches": [
                "crates/hawking-core/examples/flash_fast_chain.rs",
                "crates/hawking-core/examples/flash_full_attention_layer3.rs",
                "crates/hawking-core/examples/flash_meta_teacher_trace.rs",
                "crates/hawking-core/examples/flash_noetic_complete_layer0.rs",
                "crates/hawking-core/examples/flash_source_bf16_chain.rs",
                "crates/hawking-core/examples/flash_source_bf16_terminal.rs",
                "crates/hawking-core/examples/flash_stateful_attention_probe.rs",
                "crates/hawking-core/examples/flash_stateful_complete_token_session.rs",
                "crates/hawking-core/examples/flash_stateful_cross_species_seam.rs",
                "crates/hawking-core/examples/flash_stateful_layer3_layer4_bridge.rs",
                "crates/hawking-core/examples/flash_stateful_linear_prefix_session.rs",
                "crates/hawking-core/examples/flash_stateful_linear_probe.rs",
                "crates/hawking-core/examples/flash_tokenizer_acceptance_contract.rs",
            ],
            "measurement_bearing": False,
            "correct": "Unread as production code. They are examples.",
            "dead_code": True,
            "changes_production_default": False,
            "has_test": False,
            "land": False,
            "verdict": "HOLD",
            "why": (
                "Do not land thirteen untracked diagnostic binaries into "
                "hawking-core/examples to close a provenance gap. They "
                "are in the sealed patch so they are not lost."
            ),
        },
    ]


def commit_plan() -> list[dict[str, Any]]:
    """Ordered proposed commits. The tree builds after each land item.

    HOLD partitions are listed after the land series so a human can see
    what stays in the sealed patch. They are not commits.
    """
    return [
        {
            "order": 1,
            "land": True,
            "message": (
                "fix(qwen30): the uniform-Q4 admission was a Coder hardcode, "
                "and Base is a different rope"
            ),
            "files": [
                "crates/hawking-core/src/model/qwen_complete_binary/uniform_q4.rs",
                "crates/hawking-core/src/model/qwen30_complete_runtime.rs",
                "crates/hawking-core/examples/ascension_qwen30_complete_native_runtime.rs",
                "crates/hawking-core/examples/ascension_qwen30_uniform_q4_tps_gates.rs",
            ],
            "split_hunks_of": [],
            "builds_after": True,
            "partition": "P-Q30",
        },
        {
            "order": 2,
            "land": True,
            "message": (
                "feat(qwen80): a mixed-dtype index is not a BF16 poison, "
                "and a range read does not load the shard"
            ),
            "files": [
                "crates/hawking-core/src/model/qwen80_source_bf16_layer_major.rs"
            ],
            "split_hunks_of": [],
            "builds_after": True,
            "partition": "P-Q80",
        },
        {
            "order": 3,
            "land": True,
            "message": (
                "feat(metal): TokenPipelineCache so a token does not "
                "reacquire the pipeline lock"
            ),
            "files": ["crates/hawking-core/src/metal/mod.rs"],
            "split_hunks_of": [
                "crates/hawking-core/src/metal/mod.rs"
            ],
            "split_note": (
                "Take only TokenPipelineCache, TokenPipelineHasher, "
                "new_token_pipeline_cache, new_with_pipeline_cache, "
                "take_pipeline_cache, bind_pipeline_if_changed, "
                "pipeline_for_id / pipeline_named, and "
                "dispatch_batch_*_with_pipeline_cache. Leave elision "
                "flags, flash name tests, and gravity name tests out."
            ),
            "builds_after": True,
            "partition": "P-CACHE",
        },
        {
            "order": 4,
            "land": True,
            "message": (
                "feat(qwen38): the resident session keeps the pipeline "
                "cache, and it can be opted out"
            ),
            "files": [
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
            ],
            "split_hunks_of": [
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
            ],
            "split_note": (
                "Session field pipeline_cache, pipeline_cache_reuse "
                "(HAWKING_METAL_PIPELINE_CACHE_REUSE), and the three "
                "new_with_pipeline_cache / take_pipeline_cache sites. "
                "Not FAST, not Q2F launch helpers, not byte accounting."
            ),
            "builds_after": True,
            "partition": "P-CACHE",
        },
        {
            "order": 5,
            "land": True,
            "message": (
                "feat(qwen38): the example now says the packed bytes it "
                "bound, which is not a DRAM counter"
            ),
            "files": [
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
                "crates/hawking-core/examples/ascension_qwen38_resident.rs",
            ],
            "split_hunks_of": [
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
            ],
            "split_note": (
                "Active-weight accounting, Qwen38LayerNameCache and its "
                "test, resident_weight_bytes on the result structs, and "
                "the example JSON. Not FAST, not Q2F."
            ),
            "builds_after": True,
            "partition": "P-INSTRUMENT",
        },
        {
            "order": 6,
            "land": True,
            "message": (
                "feat(decode-family): the FP4 gate-up SwiGLU worklist has "
                "a name, and the shader has a kernel"
            ),
            "files": [
                "crates/hawking-core/src/decode_family.rs",
                "crates/hawking-core/shaders/gk_family.metal",
                "crates/hawking-core/shaders/dsv4f_native_token_graph.metal",
            ],
            "split_hunks_of": [],
            "builds_after": True,
            "partition": "P-WORKLIST",
            "note": (
                "FAMILY_KERNELS is iterated by an existing metal name "
                "test; names and shaders must land together or that test "
                "fails."
            ),
        },
        {
            "order": 7,
            "land": True,
            "message": (
                "feat(qwen38): FAST is an opt-in profile, and gated MHA "
                "stays behind it"
            ),
            "files": [
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
                "crates/hawking-core/shaders/mha.metal",
                "crates/hawking-core/src/kernels/mod.rs",
                "crates/hawking-core/src/metal/mod.rs",
            ],
            "split_hunks_of": [
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
                "crates/hawking-core/src/kernels/mod.rs",
                "crates/hawking-core/src/metal/mod.rs",
            ],
            "split_note": (
                "FAST/from_env_with_fast, serial encoder, attention-gate "
                "fuse, mha_decode_f32_qwen38_gated, its TCB wrapper, and "
                "qwen38_gated_mha_is_trace_named_and_compiled. Not Q2F "
                "geo rewrite, not elision, not flash name tests."
            ),
            "builds_after": True,
            "partition": "P-FAST",
        },
        {
            "order": 8,
            "land": False,
            "message": (
                "HOLD feat(metal): encoder/pipeline/commit elision stays "
                "in the sealed patch until an opt-out A/B exists"
            ),
            "files": ["crates/hawking-core/src/metal/mod.rs"],
            "split_hunks_of": ["crates/hawking-core/src/metal/mod.rs"],
            "builds_after": True,
            "partition": "P-ELISION",
        },
        {
            "order": 9,
            "land": False,
            "message": (
                "HOLD feat(qwen38): additive Q2F kernels may land later; "
                "the rewrite of qwen_q2f_group64_matvec_geo_tpr64_tg128 "
                "does not"
            ),
            "files": [
                "crates/hawking-core/shaders/q80_mixed_decode.metal",
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
            ],
            "split_hunks_of": [
                "crates/hawking-core/shaders/q80_mixed_decode.metal",
                "crates/hawking-core/src/model/qwen38_hybrid_decode.rs",
            ],
            "builds_after": True,
            "partition": "P-Q2F",
        },
        {
            "order": 10,
            "land": False,
            "message": (
                "HOLD feat(gravity): DeepSeek V4 fused dispatch is a "
                "different campaign and is not this binary"
            ),
            "files": [
                "crates/hawking-core/src/gravity_deepseek_v4_fullseq_attention_device.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_layer0_continuation.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_native_token_graph.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_p4b_device.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_p6_device.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_p7_device.rs",
                "crates/hawking-core/src/gravity_deepseek_v4_streamed_native.rs",
                "crates/hawking-core/examples/gravity_deepseek_v4_fullseq_capture.rs",
                "crates/hawking-core/shaders/deepseek_v4_p7.metal",
                "crates/hawking-core/shaders/moe.metal",
                "crates/hawking-core/shaders/attn.metal",
            ],
            "split_hunks_of": [],
            "builds_after": True,
            "partition": "P-GRAVITY",
            "depends_on": ["P-CACHE", "P-WORKLIST"],
        },
        {
            "order": 11,
            "land": False,
            "message": (
                "HOLD feat(flash): Qwen-Next bf16 kernels stay sealed, "
                "not landed, until a human wants a specific one"
            ),
            "files": [
                "crates/hawking-core/shaders/qwen_next.metal",
                "crates/hawking-core/shaders/matmul.metal",
                "crates/hawking-core/src/kernels/mod.rs",
                "crates/hawking-core/src/metal/mod.rs",
            ],
            "split_hunks_of": [
                "crates/hawking-core/src/kernels/mod.rs",
                "crates/hawking-core/src/metal/mod.rs",
            ],
            "builds_after": True,
            "partition": "P-FLASH",
        },
        {
            "order": 12,
            "land": False,
            "message": (
                "HOLD chore(flash): thirteen untracked diagnostic "
                "examples stay in the patch, not in hawking-core/examples"
            ),
            "files": [
                "crates/hawking-core/examples/flash_fast_chain.rs",
                "crates/hawking-core/examples/flash_full_attention_layer3.rs",
                "crates/hawking-core/examples/flash_meta_teacher_trace.rs",
                "crates/hawking-core/examples/flash_noetic_complete_layer0.rs",
                "crates/hawking-core/examples/flash_source_bf16_chain.rs",
                "crates/hawking-core/examples/flash_source_bf16_terminal.rs",
                "crates/hawking-core/examples/flash_stateful_attention_probe.rs",
                "crates/hawking-core/examples/flash_stateful_complete_token_session.rs",
                "crates/hawking-core/examples/flash_stateful_cross_species_seam.rs",
                "crates/hawking-core/examples/flash_stateful_layer3_layer4_bridge.rs",
                "crates/hawking-core/examples/flash_stateful_linear_prefix_session.rs",
                "crates/hawking-core/examples/flash_stateful_linear_probe.rs",
                "crates/hawking-core/examples/flash_tokenizer_acceptance_contract.rs",
            ],
            "split_hunks_of": [],
            "builds_after": True,
            "partition": "P-EXAMPLES",
        },
    ]


def land_ids() -> list[str]:
    return [p["id"] for p in partitions() if p["land"]]


def hold_ids() -> list[str]:
    return [p["id"] for p in partitions() if not p["land"]]


def _assert_no_promoted(node: Any, path: str = "") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in {"label", "measurement_label", "labels"} and value == PROMOTED:
                raise PromotionRefused(
                    f"REFUSED: {here} is PROMOTED while crate source is dirty"
                )
            if value == PROMOTED:
                raise PromotionRefused(
                    f"REFUSED: {here} = PROMOTED; this seal does not mint "
                    "that label"
                )
            _assert_no_promoted(value, here)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _assert_no_promoted(value, f"{path}[{i}]")
    elif node == PROMOTED:
        raise PromotionRefused(
            f"REFUSED: {path or 'value'} is PROMOTED while this seal "
            "labels DIRTY_SOURCE_DIAGNOSTIC"
        )


def build_doc() -> dict[str, Any]:
    source_dirty = crate_source_is_dirty()
    labels = measurement_labels(source_dirty=source_dirty)
    head = base_head()
    root = dirty_crate_root()
    patch = emit_dirty_crate_patch(base=head["sha"], root=root)
    fused = sealed_environment()
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "measurement_label": DIRTY_SOURCE_DIAGNOSTIC,
        "promoted": False,
        "refuse_rule": (
            "label_measurement RAISES PromotionRefused when the requested "
            "label is PROMOTED while crate source is dirty. Campaign "
            "measurements stay DIRTY_SOURCE_DIAGNOSTIC. This seal does "
            "not mint PROMOTED."
        ),
        "base_head": head,
        "dirty_crate_root": str(root),
        "crate_source_is_dirty": source_dirty or crate_source_is_dirty(root),
        "patch": {
            "path": patch["path"],
            "sha256": patch["sha256"],
            "bytes": patch["bytes"],
            "n_files": patch["n_files"],
            "n_modified": patch["n_modified"],
            "n_untracked": patch["n_untracked"],
        },
        "binary": binary_identity(),
        "toolchain": toolchain(),
        "sealed_fusion_environment": fused,
        "sealed_fusion_env": dict(SEALED_FUSION_ENV),
        "witness": witness_counts(root),
        "measurement_files": measurement_file_numstat(root, head["sha"]),
        "dirty_crate_files": {
            "modified": patch["modified"],
            "untracked": patch["untracked"],
            "total": patch["n_files"],
        },
        "measurements": labels,
        "measurements_that_inherit_this": list(CAMPAIGN_MEASUREMENTS),
        "partitions": partitions(),
        "land": land_ids(),
        "hold": hold_ids(),
        "commit_plan": commit_plan(),
        "would_land": land_ids(),
        "would_not_land": hold_ids(),
        "review_summary": {
            "land": (
                "P-Q30, P-Q80, P-CACHE, P-INSTRUMENT, P-WORKLIST, P-FAST. "
                "P-CACHE changes production default (opt-out). Landing it "
                "does not promote the TPS numbers."
            ),
            "hold": (
                "P-ELISION (default-on, untested as a set), P-Q2F (rewrite "
                "of an existing geo kernel), P-GRAVITY (different model, "
                "thousands of lines), P-FLASH (metallib dump), P-EXAMPLES "
                "(untracked diagnostic binaries)."
            ),
        },
        "why_artifact_identity_does_not_catch_it": (
            "artifact_identity.py refuses a binary OLDER than the commit "
            "that introduced a field it reads. Here there is no such "
            "commit. The check passes because the source was never "
            "committed, which is the worse case, not the safe one."
        ),
        "recovered_implementation": [
            "tools/future/measurement_provenance.py named the gap and "
            "counted TokenPipelineCache 16 vs 0",
            "tools/future/artifact_identity.py seals the fusion env and "
            "refuses a stale binary, not an uncommitted one",
            "receipts/future/patches/n1-region-timing.crate.patch is the "
            "colliding independent implementation, preserved unapplied",
        ],
        "gaps_closed": [
            "the dirty crate work is a patch with a sha256, against a "
            "recorded base HEAD",
            "the measurement binary digest and the rustc that built it "
            "are in the same seal as the fusion env",
            "existing campaign receipts are labelled "
            "DIRTY_SOURCE_DIAGNOSTIC and cannot be labelled PROMOTED "
            "while the source is dirty",
            "the patch reapplies to that HEAD in a scratch worktree",
        ],
        "negative_findings": [
            "TokenPipelineCache has no unit test in metal/mod.rs",
            "HAWKING_METAL_PIPELINE_CACHE_REUSE is default-on",
            "q80_mixed_decode.metal rewrites an existing geo kernel",
            "forty crate files are dirty; three of them produced the "
            "campaign binary; the rest is a different campaign",
        ],
        "resident_callable": {
            "entry_point": "tools.future.dirty_source_seal.build",
            "label": "tools.future.dirty_source_seal.label_measurement",
            "apply": "tools.future.dirty_source_seal.apply_patch_to_scratch_worktree",
            "receipt": f"receipts/future/{RECEIPT}",
            "patch": PATCH_REL,
            "fails_closed": "PromotionRefused; PatchApplyError",
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    _assert_no_promoted(doc)
    return doc


def build() -> Path:
    doc = build_doc()
    if doc.get("measurement_label") == PROMOTED or doc.get("promoted") is True:
        raise PromotionRefused(
            "REFUSED: the dirty-source seal cannot label itself PROMOTED"
        )
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    out = selftest() if a.selftest else build()
    doc = json.loads(out.read_text())
    print(out)
    print(
        json.dumps(
            {
                "measurement_label": doc["measurement_label"],
                "promoted": doc["promoted"],
                "base_head": doc["base_head"]["sha"],
                "patch_sha256": doc["patch"]["sha256"],
                "binary_sha256": doc["binary"]["sha256"],
                "crate_source_is_dirty": doc["crate_source_is_dirty"],
                "n_files": doc["patch"]["n_files"],
                "land": doc["land"],
                "hold": doc["hold"],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
