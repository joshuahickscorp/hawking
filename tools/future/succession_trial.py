#!/usr/bin/env python3
"""SUCCESSION TRIAL — exercise succession.py on a REAL on-disk candidate.

tools/future/succession.py is the machine and is not edited here. The
synthetic SUCCESSION.json run created children with invented 1/2 scores and
placeholder lineage. This sidecar loads a real ModelLake specimen, fills the
nine-field lineage from the real manifest, clones live HCLI WorkUnits,
lets the independent judge rule, and proves Qwen27 remains restorable on
THIS run.

A 0.6B against a sealed 27B is expected to be REFUSED. That is a complete
exercise of "promote or refuse". This module will not synthesise a passer.

    python3 tools/future/succession_trial.py --selftest
    python3 tools/future/succession_trial.py --build
    python3 -m pytest tools/future/test_succession_trial.py -q

Does not fork tools/future/succession.py. Does not take a GPU lease.
Everything emitted is STATIC_ONLY, bench UNKNOWN, gpu_authority false.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from hcli.workunit import (
    DEFAULT_RETRY_BUDGET,
    MAX_REPAIR_DEPTH,
    MAX_REPAIRS_PER_ROOT,
    WorkUnit,
)
from tools.future import fallback_resident as fb
from tools.future import succession as suc
from tools.future._common import RECEIPTS, REPO, git, write_receipt
from tools.future.resident_optimizer import BoundViolation
from tools.future.specimen_verify import (
    LAKE,
    MANIFESTS,
    SPECIMENS,
    _git_blob_sha1,
    _read_metadata,
    _sha256,
    list_specimens,
    specimen_dir,
    specimen_files,
)
from tools.future.super_resident import QWEN_ID, QWEN_ROLE
from tools.future.workunit_species import emit_hcli_workunit, validate_emitted_unit


RECEIPT = "SUCCESSION_TRIAL.json"
SCHEMA = "hawking.future.succession_trial.v1"
RECORDED_BY = "tools/future/succession_trial.py"
VERSION = 1

# The only candidate this host can actually open today. Flash-Next is still
# FLASH_NX_READY=False; Coder-30B is only under ModelLake partial/.
CANDIDATE_SLUG = "Qwen--Qwen3-0.6B@c1899de289a0"
CANDIDATE_CHILD_ID = f"child.different_specimen.{CANDIDATE_SLUG}"
SYNTHETIC_RECEIPT = "SUCCESSION.json"
HCLI_WU_RECEIPT = "HCLI_FUTURE_WORKUNITS.json"
NR_NX_RECEIPT = "NR_NX_GENERIC.json"
CENSUS_REL = "receipts/future/evidence/HCLI_MODELLAKE_FLASH_CENSUS.json"
FLASH_SLUG = "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
CODER_PARTIAL_PREFIX = "Qwen--Qwen3-Coder-30B-A3B"
# Skip the 1.5GB weight file; ModelLake already recorded its sha256 count.
MAX_INDEPENDENT_HASH_BYTES = 64 << 20
# Default SuccessionBound.max_cloned_workunits is 32. Stay inside it.
MAX_SHADOW_UNITS = 32

ERAS = suc.ERAS
ODYSSEYS = suc.ODYSSEYS

CLAIM_BOUNDARY = (
    "Static sidecar artifact. Succession is exercised on a REAL on-disk "
    "ModelLake specimen. No GPU lease, no process launch, no hardware "
    "number. Physical axes stay UNKNOWN. A refusal backed by declared "
    "parameter counts is a complete verdict, not a missing trial."
)


class TrialRefused(RuntimeError):
    """The trial cannot honestly proceed. Fail closed."""


# ---------------------------------------------------------------------------
# Disk facts — ModelLake, Flash-Next, Coder, FLASH_NX_READY
# ---------------------------------------------------------------------------


def _load_receipt(name: str) -> dict[str, Any] | None:
    path = RECEIPTS / name
    if path.is_file():
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, UnicodeError):
            return None
        return doc if isinstance(doc, dict) else None
    blob = git("show", f"HEAD:receipts/future/{name}")
    if blob:
        try:
            doc = json.loads(blob)
        except json.JSONDecodeError:
            return None
        return doc if isinstance(doc, dict) else None
    return None


def flash_nx_ready() -> dict[str, Any]:
    """FLASH_NX_READY is a disk fact. Missing receipt is not a True."""
    doc = _load_receipt(NR_NX_RECEIPT)
    if not isinstance(doc, dict):
        return {
            "value": False,
            "source": f"receipts/future/{NR_NX_RECEIPT}",
            "present": False,
            "why": "NR_NX_GENERIC.json not locatable; refusing to default FLASH_NX_READY True",
        }
    return {
        "value": doc.get("FLASH_NX_READY") is True,
        "raw": doc.get("FLASH_NX_READY"),
        "source": f"receipts/future/{NR_NX_RECEIPT}",
        "present": True,
    }


def _safetensor_shards(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.glob("model-*.safetensors") if p.is_file())


def inspect_rejected_candidates() -> list[dict[str, Any]]:
    """Why Flash-Next and Coder-30B are not this trial's candidate."""
    flash_ready = flash_nx_ready()
    flash_dir = SPECIMENS / FLASH_SLUG
    flash_partial = LAKE / "partial" / FLASH_SLUG
    flash_manifest = MANIFESTS / f"{FLASH_SLUG}.json"
    shards = _safetensor_shards(flash_dir)
    rows = [
        {
            "slug": FLASH_SLUG,
            "chosen": False,
            "in_specimens": flash_dir.is_dir(),
            "in_partial": flash_partial.is_dir(),
            "manifest_present": flash_manifest.is_file(),
            "n_safetensor_shards_on_disk": len(shards),
            "FLASH_NX_READY": flash_ready["value"],
            "FLASH_NX_READY_source": flash_ready["source"],
            "why_not": (
                "FLASH_NX_READY is False; a concurrent lane is still verifying "
                "this body. An incomplete Flash-Next is not a loadable candidate "
                "for succession today."
            ),
        }
    ]
    partial_root = LAKE / "partial"
    coder_hits = []
    if partial_root.is_dir():
        coder_hits = sorted(
            p.name
            for p in partial_root.iterdir()
            if p.is_dir() and CODER_PARTIAL_PREFIX in p.name
        )
    specimens_hits = []
    if SPECIMENS.is_dir():
        specimens_hits = sorted(
            p.name
            for p in SPECIMENS.iterdir()
            if p.is_dir() and CODER_PARTIAL_PREFIX in p.name
        )
    rows.append(
        {
            "slug_prefix": CODER_PARTIAL_PREFIX,
            "chosen": False,
            "in_specimens": specimens_hits,
            "in_partial": coder_hits,
            "why_not": (
                "Qwen3-Coder-30B-A3B is downloading into ModelLake partial/; "
                "it is not a published specimens/ body this trial can load."
            ),
        }
    )
    return rows


def load_manifest(slug: str) -> dict[str, Any]:
    path = MANIFESTS / f"{slug}.json"
    if not path.is_file():
        raise TrialRefused(f"ModelLake manifest missing: {path}")
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise TrialRefused(f"ModelLake manifest unreadable: {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise TrialRefused(f"ModelLake manifest is not an object: {path}")
    required = ("repo", "resolved_sha", "path", "n_files", "n_sha256_verified", "n_size_only_verified")
    missing = [k for k in required if k not in doc]
    if missing:
        raise TrialRefused(f"ModelLake manifest missing {missing}: {path}")
    if not doc.get("resolved_sha"):
        raise TrialRefused(f"ModelLake manifest has empty resolved_sha: {path}")
    out = dict(doc)
    out["manifest_path"] = str(path)
    return out


def _safetensors_header_params(path: Path) -> dict[str, Any]:
    """Read the safetensors JSON header only. Does not load weights."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise TrialRefused(f"safetensors header truncated: {path}")
        n = int.from_bytes(raw_len, "little")
        if n <= 0 or n > 16 * 1024 * 1024:
            raise TrialRefused(f"safetensors header length {n} refused: {path}")
        header = json.loads(fh.read(n))
    if not isinstance(header, dict):
        raise TrialRefused(f"safetensors header is not an object: {path}")
    n_params = 0
    n_tensors = 0
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(spec, dict):
            continue
        shape = spec.get("shape") or []
        numel = 1
        for dim in shape:
            numel *= int(dim)
        n_params += numel
        n_tensors += 1
    return {
        "path": str(path),
        "bytes": size,
        "header_bytes": n,
        "n_tensors": n_tensors,
        "header_params": n_params,
        "lm_head_listed": "lm_head.weight" in header,
        "embed_listed": "model.embed_tokens.weight" in header,
        "weights_loaded_into_runtime": False,
    }


def inspect_specimen(slug: str) -> dict[str, Any]:
    """Open the real specimen. Hash small files. Parse the weight header.

    Does not load weights into a runtime and does not re-hash the 1.5GB
    safetensors (ModelLake already reported n_sha256_verified).
    """
    root = specimen_dir(slug)
    if not root.is_dir():
        raise TrialRefused(f"specimen not on disk: {root}")
    files = specimen_files(slug)
    if not files:
        raise TrialRefused(f"specimen has no source files: {root}")
    rows: list[dict[str, Any]] = []
    n_hashed = 0
    n_matched = 0
    n_skipped_large = 0
    for path in files:
        size = path.stat().st_size
        meta = _read_metadata(root, path.name)
        row: dict[str, Any] = {
            "file": path.name,
            "bytes": size,
            "exists": True,
            "path": str(path),
        }
        if meta is None:
            row["digest_kind"] = None
            row["published_etag"] = None
            row["independent_verdict"] = "NO_REMOTE_DIGEST"
        else:
            row["digest_kind"] = meta["digest_kind"]
            row["published_etag"] = meta["etag"]
            row["published_commit"] = meta["commit"]
            if size > MAX_INDEPENDENT_HASH_BYTES:
                n_skipped_large += 1
                row["independent_verdict"] = "SKIPPED_LARGE_NOT_RECOMPUTED"
                row["why"] = (
                    f"{size} bytes exceeds independent hash budget "
                    f"{MAX_INDEPENDENT_HASH_BYTES}; ModelLake n_sha256_verified "
                    "is the authority for the weight file"
                )
            else:
                if meta["digest_kind"] == "sha256":
                    actual = _sha256(path)
                elif meta["digest_kind"] == "git_blob_sha1":
                    actual = _git_blob_sha1(path)
                else:
                    actual = None
                    row["independent_verdict"] = "UNRECOGNIZED_DIGEST"
                if actual is not None:
                    n_hashed += 1
                    ok = actual == meta["etag"]
                    if ok:
                        n_matched += 1
                    row["actual"] = actual
                    row["independent_verdict"] = "VERIFIED" if ok else "MISMATCH"
        rows.append(row)

    config_path = root / "config.json"
    if not config_path.is_file():
        raise TrialRefused(f"specimen has no config.json (cannot load): {config_path}")
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise TrialRefused(f"specimen config.json unreadable: {exc}") from exc
    if not isinstance(config, dict):
        raise TrialRefused("specimen config.json is not an object")

    weight = root / "model.safetensors"
    if not weight.is_file():
        shards = _safetensor_shards(root)
        if not shards:
            raise TrialRefused(f"specimen has no weight file: {root}")
        raise TrialRefused(
            f"specimen uses sharded weights {shards[:3]}...; this trial "
            "expected a single model.safetensors it can header-parse"
        )
    header = _safetensors_header_params(weight)
    tokenizer = root / "tokenizer.json"
    return {
        "slug": slug,
        "path": str(root),
        "on_disk": True,
        "n_source_files": len(files),
        "files": rows,
        "independent_small_files_hashed": n_hashed,
        "independent_small_files_matched": n_matched,
        "independent_large_files_not_rehashed": n_skipped_large,
        "config": {
            "architectures": config.get("architectures"),
            "model_type": config.get("model_type"),
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "vocab_size": config.get("vocab_size"),
            "tie_word_embeddings": config.get("tie_word_embeddings"),
            "torch_dtype": config.get("torch_dtype"),
            "max_position_embeddings": config.get("max_position_embeddings"),
        },
        "tokenizer_present": tokenizer.is_file(),
        "tokenizer_bytes": tokenizer.stat().st_size if tokenizer.is_file() else None,
        "weights": header,
        "runtime_loaded": False,
        "gpu_authority": False,
    }


def census_mentions(slug: str) -> dict[str, Any]:
    """The Flash census is truncated; 0.6B lives in verified_receipts, not entries."""
    path = REPO / CENSUS_REL
    doc: dict[str, Any] | None = None
    if path.is_file():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                doc = loaded
        except (OSError, json.JSONDecodeError, UnicodeError):
            doc = None
    if doc is None:
        blob = git("show", f"HEAD:{CENSUS_REL}")
        if blob:
            try:
                loaded = json.loads(blob)
                if isinstance(loaded, dict):
                    doc = loaded
            except json.JSONDecodeError:
                doc = None
    if doc is None:
        return {
            "census_present": False,
            "in_specimens_entries": False,
            "in_verified_receipts": False,
            "note": f"{CENSUS_REL} not locatable in this checkout",
        }
    entries = (doc.get("specimens") or {}).get("entries") or []
    entry_names = [
        str(e.get("name") or "")
        for e in entries
        if isinstance(e, dict)
    ]
    receipts = (doc.get("verified_receipts") or {}).get("receipts") or []
    hit = None
    for row in receipts:
        if isinstance(row, dict) and slug in str(row.get("path") or row.get("specimen_path") or ""):
            hit = {
                "repo": row.get("repo"),
                "resolved_sha": row.get("resolved_sha"),
                "revision": row.get("revision"),
                "n_files": row.get("n_files"),
                "n_sha256_verified": row.get("n_sha256_verified"),
                "n_size_only_verified": row.get("n_size_only_verified"),
                "manifest_path": row.get("path"),
                "specimen_path": row.get("specimen_path"),
            }
            break
    return {
        "census_present": True,
        "census_path": CENSUS_REL,
        "specimens_entries_truncated": (doc.get("specimens") or {}).get("truncated"),
        "n_specimens_entries": len(entries),
        "in_specimens_entries": slug in entry_names,
        "in_verified_receipts": hit is not None,
        "verified_receipt": hit,
        "note": (
            "the Flash census specimens.entries list is a short snapshot and "
            "does not include this slug; verified_receipts names the manifest"
            if hit is not None and slug not in entry_names
            else None
        ),
    }


def choose_candidate() -> dict[str, Any]:
    """Pick the strongest REAL candidate this host can actually open today."""
    if not LAKE.is_dir() or not SPECIMENS.is_dir():
        raise TrialRefused(f"ModelLake specimens directory is not mounted at {SPECIMENS}")
    listing = list_specimens()
    rejected = inspect_rejected_candidates()
    if CANDIDATE_SLUG not in listing:
        raise TrialRefused(
            f"{CANDIDATE_SLUG} is not in the specimens listing "
            f"({len(listing)} names). A missing specimen is not a candidate."
        )
    manifest = load_manifest(CANDIDATE_SLUG)
    specimen = inspect_specimen(CANDIDATE_SLUG)
    census = census_mentions(CANDIDATE_SLUG)
    if specimen["weights"]["header_params"] <= 0:
        raise TrialRefused("safetensors header reported zero parameters; refusing to invent a count")
    if not specimen["tokenizer_present"]:
        raise TrialRefused(f"{CANDIDATE_SLUG} has no tokenizer.json; cannot load")
    return {
        "slug": CANDIDATE_SLUG,
        "chosen": True,
        "why": (
            "Qwen3-0.6B is a published ModelLake specimen with a real manifest, "
            "real resolved_sha, per-file hashing counts, source files on disk, "
            "and a parseable safetensors header. Flash-Next is FLASH_NX_READY="
            "False. Coder-30B is only in partial/. A small real body that the "
            "judge REFUSEs is the honest candidate, not a synthesised passer."
        ),
        "in_specimens_listing": True,
        "listing_n": len(listing),
        "manifest": manifest,
        "specimen": specimen,
        "census": census,
        "rejected": rejected,
        "flash_nx_ready": flash_nx_ready(),
        "synthetic": False,
        "runtime_loaded": False,
        "gpu_authority": False,
    }


# ---------------------------------------------------------------------------
# Real WorkUnits and the real incumbent
# ---------------------------------------------------------------------------


def load_real_workunits(*, limit: int = MAX_SHADOW_UNITS) -> list[dict[str, Any]]:
    """Live HCLI WorkUnits from disk, not succession.py's two seed units."""
    doc = _load_receipt(HCLI_WU_RECEIPT)
    if not isinstance(doc, dict) or not isinstance(doc.get("work_units"), list):
        raise TrialRefused(
            f"receipts/future/{HCLI_WU_RECEIPT} is missing or has no work_units; "
            "cannot clone real WorkUnits"
        )
    pending = [
        dict(u)
        for u in doc["work_units"]
        if isinstance(u, dict) and u.get("id") and u.get("status") == "pending"
    ]
    pending.sort(key=lambda r: str(r["id"]))
    chosen = pending[: max(1, int(limit))]
    if not chosen:
        raise TrialRefused("HCLI_FUTURE_WORKUNITS.json has no pending units to clone")
    for unit in chosen:
        validate_emitted_unit(unit)
        WorkUnit.from_dict(unit)
    return chosen


def declared_param_count(identity: Mapping[str, Any]) -> int:
    specimen = identity.get("specimen_identity") if isinstance(identity.get("specimen_identity"), dict) else {}
    params = specimen.get("parent_params")
    if not isinstance(params, int) or isinstance(params, bool) or params <= 0:
        raise TrialRefused(
            "Qwen27 MIX_REPORT parent_params is missing; refusing to invent a capability number"
        )
    return params


def make_qwen27_incumbent(
    *,
    identity: Mapping[str, Any],
    work_units: Sequence[Mapping[str, Any]],
    capability: int,
) -> dict[str, Any]:
    """Real sealed incumbent. Not suc.make_incumbent()'s synthetic body."""
    iid = str(identity.get("id") or QWEN_ID)
    extra = {
        "transformation": (
            "sealed Qwen27 incumbent (qwen3.8-27b-sealed-3.14); CONTROL, not a ceiling"
        ),
        "source_model_lineage": {
            "model_id": iid,
            "role": QWEN_ROLE,
            "artifact_path": identity.get("artifact_path"),
            "config_digest": identity.get("config_digest"),
            "mix_id": (identity.get("specimen_identity") or {}).get("mix_id")
            if isinstance(identity.get("specimen_identity"), dict)
            else None,
            "parent_params": capability,
            "identity_document": "hcli/hawking-native.sealed-3.14.json",
            "synthetic": False,
        },
        "data_lineage": {
            "teacher_capture": "UNKNOWN",
            "note": "incumbent identity is MIX_REPORT + HCLI seal prefixes; not a capture count",
            "config_digest": identity.get("config_digest"),
        },
        "capability_deltas": suc._capability_deltas_static(
            (f"declared-parent_params:{capability}",)
        ),
    }
    lineage = suc.lineage_for_method("adapter", parent_nx=iid, extra=extra)
    scores = suc.synthetic_scores({"capability": capability})
    return {
        "id": iid,
        "role": "incumbent",
        "status": "ACTIVE",
        "nx_id": iid,
        "control": "CONTROL_NOT_TARGET_NOT_CEILING",
        "verifier": "future.succession.incumbent",
        "canonical": True,
        "synthetic": False,
        "may_promote": False,
        "may_modify_verifier": False,
        "may_own_canonical_mission": False,
        "lineage_depth": 0,
        "lineage": lineage,
        "scores": scores,
        "score_kind": "DECLARED_PARAM_COUNT",
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "work_units": [dict(u) for u in work_units],
        "claim_boundary": CLAIM_BOUNDARY,
        "residency_status": identity.get("residency_status"),
        "fallback_identity_status": identity.get("status"),
    }


def extra_lineage_from_candidate(
    candidate: Mapping[str, Any],
    *,
    parent_id: str,
    incumbent_params: int,
) -> dict[str, Any]:
    """Nine-field extras from the real manifest — not placeholders."""
    manifest = candidate["manifest"]
    specimen = candidate["specimen"]
    config = specimen["config"]
    header_params = int(specimen["weights"]["header_params"])
    return {
        "transformation": (
            "different specimen under the Qwen3 family; incumbent remains the "
            "sealed Qwen3.8-27B. Architecture family held; weights replaced."
        ),
        "source_model_lineage": {
            "method": "different_specimen",
            "parents": [parent_id],
            "repo": manifest["repo"],
            "resolved_sha": manifest["resolved_sha"],
            "revision": manifest.get("revision") or manifest["resolved_sha"],
            "manifest_path": manifest["manifest_path"],
            "specimen_path": specimen["path"],
            "n_files": manifest["n_files"],
            "n_sha256_verified": manifest["n_sha256_verified"],
            "n_size_only_verified": manifest["n_size_only_verified"],
            "bytes": manifest.get("bytes"),
            "acquired_at": manifest.get("acquired_at"),
            "synthetic": False,
            "on_disk": True,
        },
        "representation_lineage": {
            "architecture": (config.get("architectures") or [None])[0],
            "model_type": config.get("model_type"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "vocab_size": config.get("vocab_size"),
            "header_params": header_params,
            "declared_params_readme": "0.6B",
            "tie_word_embeddings": config.get("tie_word_embeddings"),
            "note": (
                "header_params is the safetensors header numel. tie_word_embeddings "
                "is true so embed and lm_head both appear; this is not a bench."
            ),
        },
        "code_lineage": [
            "parent code path; different weight/specimen identity",
            f"specimen {candidate['slug']} @ {manifest['resolved_sha']}",
        ],
        "behavioral_changes": [
            "specimen substitution; architecture family held",
            "declared scale 0.6B vs sealed 27B incumbent; not a capability eval",
        ],
        "data_lineage": {
            "teacher_capture": "UNKNOWN",
            "note": "teacher capture is a physical/Codex concern; not estimated here",
            "manifest_hash_counts": {
                "n_files": manifest["n_files"],
                "n_sha256_verified": manifest["n_sha256_verified"],
                "n_size_only_verified": manifest["n_size_only_verified"],
                "source": "ModelLake manifest",
            },
            "independent_small_files_hashed": specimen["independent_small_files_hashed"],
            "independent_small_files_matched": specimen["independent_small_files_matched"],
            "independent_large_files_not_rehashed": specimen["independent_large_files_not_rehashed"],
            "resolved_sha": manifest["resolved_sha"],
        },
        "capability_deltas": {
            "evidence_class": "STATIC_ONLY",
            "declared": [
                f"candidate header_params={header_params}",
                f"incumbent parent_params={incumbent_params}",
            ],
            "measured": None,
            "not_a_measurement": True,
            "candidate_header_params": header_params,
            "incumbent_parent_params": incumbent_params,
        },
    }


# ---------------------------------------------------------------------------
# Live guards on the real child / real incumbent
# ---------------------------------------------------------------------------


def fire_shadow_refusals(shadow: suc.ShadowChild) -> list[dict[str, Any]]:
    """The four watched refusals, live on this ShadowChild. Not asserted."""
    trials: tuple[tuple[str, Any], ...] = (
        ("own_canonical_mission", shadow.own_canonical_mission),
        ("alter_verifier", lambda: shadow.alter_verifier("self")),
        ("widen_authority", lambda: shadow.widen_authority("self_promotion")),
        ("promote_self", shadow.promote_self),
    )
    results: list[dict[str, Any]] = []
    for name, thunk in trials:
        try:
            thunk()
        except suc.ShadowAuthorityError as exc:
            results.append({"trial": name, "refused": True, "error": str(exc), "live": True})
            continue
        raise suc.ShadowAuthorityError(f"shadow authority guard did not fire for {name} on the real child")
    got = {r["trial"] for r in results}
    if got != set(suc.SHADOW_FORBIDDEN_ACTIONS):
        raise suc.ShadowAuthorityError(f"shadow proof trials {got} != {set(suc.SHADOW_FORBIDDEN_ACTIONS)}")
    return results


def fire_incumbent_self_preference(
    incumbent: Mapping[str, Any],
    child: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Incumbent cannot promote itself — on THIS real incumbent, not the synthetic one."""
    sit = suc.Incumbent(incumbent)
    results: list[dict[str, Any]] = []
    try:
        sit.promote_self()
    except suc.SelfPreferenceError as exc:
        results.append({"trial": "promote_self", "refused": True, "error": str(exc), "live": True})
    else:
        raise suc.SelfPreferenceError("incumbent promote_self did not fire on the real incumbent")
    try:
        sit.request_self_promotion()
    except suc.SelfPreferenceError as exc:
        results.append(
            {"trial": "request_self_promotion", "refused": True, "error": str(exc), "live": True}
        )
    else:
        raise suc.SelfPreferenceError("incumbent request_self_promotion did not fire")
    try:
        sit.block_child(child)
    except suc.SelfPreferenceError as exc:
        results.append({"trial": "block_child", "refused": True, "error": str(exc), "live": True})
    else:
        raise suc.SelfPreferenceError("incumbent block_child did not fire")
    try:
        suc.SuccessionOrchestrator(incumbent, invoker="incumbent")
    except suc.SelfPreferenceError as exc:
        results.append(
            {
                "trial": "orchestrator_invoker_incumbent",
                "refused": True,
                "error": str(exc),
                "live": True,
            }
        )
    else:
        raise suc.SelfPreferenceError("orchestrator accepted invoker=incumbent")
    promote_on_incumbent = hasattr(suc.Incumbent, "promote") and callable(
        getattr(suc.Incumbent, "promote", None)
    )
    results.append(
        {
            "trial": "promote_method_absent",
            "refused": not promote_on_incumbent,
            "promote_exists_on_incumbent": promote_on_incumbent,
            "live": True,
        }
    )
    return results


def compare_vs_synthetic(run: Mapping[str, Any]) -> dict[str, Any]:
    """What a REAL candidate changed versus SUCCESSION.json's synthetic exercise."""
    synthetic = _load_receipt(SYNTHETIC_RECEIPT)
    syn_run = None
    syn_qual = None
    syn_child = None
    if isinstance(synthetic, dict):
        syn_run = (synthetic.get("succession") or {}).get("synthetic_run")
        syn_qual = (synthetic.get("qualification") or {}).get("synthetic_run")
        syn_child = None if not isinstance(syn_qual, dict) else syn_qual.get("child_id")
    lineage = (run.get("child") or {}).get("lineage") or {}
    source = lineage.get("source_model_lineage") or {}
    changed = [
        (
            "lineage.source_model_lineage carries repo, resolved_sha, manifest_path, "
            "and ModelLake hash counts rather than {method, parents}"
        ),
        (
            "shadow cloned live HCLI WorkUnits from HCLI_FUTURE_WORKUNITS.json "
            "rather than future.succession.seed-mission-a/b"
        ),
        (
            "comparable-axis scores are declared parameter counts from MIX_REPORT "
            "and the safetensors header, not synthetic 1/2"
        ),
        (
            "the independent judge REFUSEd; the synthetic run completed switch() "
            "because it invented a dominating child"
        ),
        (
            "qualification failed the incumbent floor; the synthetic child was "
            "QUALIFIED on invented scores"
        ),
        (
            "create_child still stamps synthetic=True (succession.py is unchanged); "
            "the REAL-ness lives in lineage and on-disk files"
        ),
    ]
    unchanged = [
        "physical axes remain UNKNOWN — a real specimen did not mint a hardware number without a lease",
        "PROMOTE remains unreachable without executed GPU authority",
        "the four shadow refusals still fire",
        "the incumbent still cannot promote itself",
        "Qwen27 restorable-now still holds",
        "the five-rung ladder and stop-a-bad-child machinery are the same functions",
    ]
    return {
        "synthetic_receipt": f"receipts/future/{SYNTHETIC_RECEIPT}",
        "synthetic_present": synthetic is not None,
        "synthetic_child_id": syn_child or "child.synthetic.adapter.v1",
        "synthetic_complete": None if not isinstance(syn_run, dict) else syn_run.get("complete"),
        "synthetic_active_id": None if not isinstance(syn_run, dict) else syn_run.get("active_id"),
        "real_child_id": (run.get("child") or {}).get("id"),
        "real_switched": run.get("switched"),
        "real_verdict": (run.get("verdict") or {}).get("verdict"),
        "real_reason": (run.get("verdict") or {}).get("reason"),
        "real_create_child_stamped_synthetic": (run.get("child") or {}).get("synthetic"),
        "lineage_resolved_sha": source.get("resolved_sha"),
        "lineage_repo": source.get("repo"),
        "what_changed": changed,
        "what_did_not_change": unchanged,
        "deflating_result": (
            "The protocol machinery did not grow a new physical sense by meeting "
            "a real specimen. Lineage, WorkUnit identity, scores, and the verdict "
            "changed. Physical axes, PROMOTE unreachability, and the four refusals "
            "did not. The synthetic run's completed handover was an artefact of "
            "invented 1/2 scores, not of a stronger body."
        ),
    }


# ---------------------------------------------------------------------------
# The trial
# ---------------------------------------------------------------------------


def run_real_succession_trial() -> dict[str, Any]:
    """Create, shadow, qualify, judge, stop, restore — on a real specimen."""
    candidate = choose_candidate()
    identity = fb.fallback_identity()
    if identity.get("status") != "SEALED" or not identity.get("id"):
        raise TrialRefused(
            f"Qwen27 fallback identity is not SEALED (status={identity.get('status')!r}); "
            "refusing to run succession against an unresolved incumbent"
        )
    restorable = fb.verify_restorable(identity=identity)
    restore = fb.restore_path(identity=identity, restorable=restorable)
    rollback_doc = fb.rollback_state(restorable=restorable)

    incumbent_params = declared_param_count(identity)
    child_params = int(candidate["specimen"]["weights"]["header_params"])
    units = load_real_workunits()
    incumbent = make_qwen27_incumbent(
        identity=identity, work_units=units, capability=incumbent_params
    )
    extra = extra_lineage_from_candidate(
        candidate, parent_id=incumbent["id"], incumbent_params=incumbent_params
    )
    # succession.py's own creation path. Stamps synthetic=True; we do not edit it.
    child = suc.create_child(
        method="different_specimen",
        parent=incumbent,
        child_id=CANDIDATE_CHILD_ID,
        extra_lineage=extra,
        scores={"capability": child_params},
    )
    suc.require_lineage(child["lineage"])
    source = child["lineage"]["source_model_lineage"]
    if source.get("resolved_sha") != candidate["manifest"]["resolved_sha"]:
        raise TrialRefused("create_child dropped the real resolved_sha")
    if source.get("repo") != candidate["manifest"]["repo"]:
        raise TrialRefused("create_child dropped the real repo")
    if source.get("n_sha256_verified") != candidate["manifest"]["n_sha256_verified"]:
        raise TrialRefused("create_child dropped ModelLake hash verification counts")

    shadow = suc.ShadowChild(child)
    clones = shadow.receive_cloned_workunits(units)
    if len(clones) != len(units):
        raise TrialRefused(f"shadow cloned {len(clones)} of {len(units)} real WorkUnits")
    seed_ids = {"future.succession.seed-mission-a", "future.succession.seed-mission-b"}
    for clone in clones:
        if clone.get("shadow_of") in seed_ids:
            raise TrialRefused("shadow cloned synthetic seed units; this trial requires real HCLI units")
        WorkUnit.from_dict(clone)
    proposals = list(shadow.propose_experiments(seed=0))
    classified = shadow.classify_receipt(
        {"schema": SCHEMA, "id": "succession-trial-receipt"}, "STATIC_ONLY"
    )
    refusals = fire_shadow_refusals(shadow)

    # Live bound-exceeded on the real child: the full HCLI queue is 65 > 32.
    bound_exceeded: dict[str, Any]
    tight = suc.SuccessionBound(max_cloned_workunits=1)
    tight_shadow = suc.ShadowChild(child, tight)
    try:
        tight_shadow.receive_cloned_workunits(units)
    except BoundViolation as exc:
        bound_exceeded = {"refused": True, "live": True, "error": str(exc), "n_units": len(units)}
    else:
        raise BoundViolation("over-bound clone of real WorkUnits was not refused")

    floor = {"capability": incumbent_params}
    qualified = suc.qualify_child(shadow.record() | {"scores": child["scores"]}, floor=floor)
    child_q = qualified["record"]
    q_verdict = qualified["verdict"]

    shadow_record = suc.observe_side_by_side(incumbent, child, units)
    evidence = {
        "evidence_class": "STATIC_ONLY",
        "measurement_class": "STATIC_ONLY",
        "gpu_authority": False,
        "score_kind": "DECLARED_PARAM_COUNT",
        "candidate_header_params": child_params,
        "incumbent_parent_params": incumbent_params,
        "not_a_capability_eval": True,
        "not_a_protected_measurement": True,
    }
    verdict = suc.submit_to_judge(
        incumbent,
        child,
        shadow_record,
        judge_id=suc.INDEPENDENT_JUDGE_ID,
        evidence=evidence,
    )
    if verdict.get("promoted") is True or verdict.get("verdict") == suc.VERDICT_PROMOTE:
        raise TrialRefused("this sidecar minted PROMOTE; a 0.6B cannot earn that here")

    self_pref = fire_incumbent_self_preference(incumbent, child)

    orch = suc.SuccessionOrchestrator(incumbent, invoker="lineage_gate")
    orch.checkpoint_incumbent()
    orch.seal_mission(units)
    orch.seal_rollback()
    launch_refused: dict[str, Any]
    try:
        orch.launch_child(child_q)
    except suc.SuccessionRefused as exc:
        launch_refused = {"refused": True, "live": True, "error": str(exc)}
    else:
        raise suc.SuccessionRefused("unqualified real child was launched")
    stopped = suc.stop_child(orch, child_q, reason="qualification_failed")
    if not stopped["incumbent_restored"] or orch.active_id != incumbent["id"]:
        raise TrialRefused("rollback did not restore Qwen27 on this run")

    restorable_after = fb.verify_restorable(identity=identity)
    physical_unknown = {
        name: {
            "value": None,
            "state": "UNKNOWN",
            "why": (
                "no GPU lease was taken (contract: NO GPU LEASE required; four "
                "lanes are running and this trial does not contend). A real "
                "specimen does not default a hardware number."
            ),
        }
        for name in sorted(suc.PHYSICAL_AXIS_NAMES)
    }

    snapshot = orch.snapshot()
    numbers = {
        "incumbent_parent_params": incumbent_params,
        "incumbent_parent_params_source": "MIX_REPORT.parent_params via fallback_identity",
        "candidate_header_params": child_params,
        "candidate_header_params_source": (
            f"{candidate['specimen']['path']}/model.safetensors safetensors header numel"
        ),
        "candidate_declared_params_readme": "0.6B",
        "capability_axis_direction": "higher",
        "parent_dominates_on": suc.named_dominating_dimensions(incumbent, child),
        "child_dominates_incumbent": suc.comparable_dominates(child, incumbent),
        "not_a_hardware_number": True,
        "not_a_capability_eval": True,
    }
    run = {
        "candidate": {
            "slug": candidate["slug"],
            "repo": candidate["manifest"]["repo"],
            "resolved_sha": candidate["manifest"]["resolved_sha"],
            "revision": candidate["manifest"].get("revision"),
            "manifest_path": candidate["manifest"]["manifest_path"],
            "specimen_path": candidate["specimen"]["path"],
            "on_disk": True,
            "in_specimens_listing": True,
            "n_files": candidate["manifest"]["n_files"],
            "n_sha256_verified": candidate["manifest"]["n_sha256_verified"],
            "n_size_only_verified": candidate["manifest"]["n_size_only_verified"],
            "bytes": candidate["manifest"].get("bytes"),
            "header_params": child_params,
            "declared_params_readme": "0.6B",
            "architectures": candidate["specimen"]["config"].get("architectures"),
            "independent_small_files_hashed": candidate["specimen"]["independent_small_files_hashed"],
            "independent_small_files_matched": candidate["specimen"]["independent_small_files_matched"],
            "census": candidate["census"],
            "why_chosen": candidate["why"],
            "rejected": candidate["rejected"],
            "flash_nx_ready": candidate["flash_nx_ready"],
            "synthetic": False,
            "runtime_loaded": False,
        },
        "incumbent": {
            "id": incumbent["id"],
            "role": QWEN_ROLE,
            "synthetic": False,
            "parent_params": incumbent_params,
            "fallback_status": identity.get("status"),
            "residency_status": identity.get("residency_status"),
            "config_digest": identity.get("config_digest"),
            "artifact_path": identity.get("artifact_path"),
        },
        "child": child,
        "create_child_stamped_synthetic": child.get("synthetic"),
        "cloned_workunits": {
            "n": len(clones),
            "source": f"receipts/future/{HCLI_WU_RECEIPT}",
            "source_status_filter": "pending",
            "ids": [u["id"] for u in units],
            "shadow_ids": [c["id"] for c in clones],
            "shadow_of": [c.get("shadow_of") for c in clones],
            "includes_synthetic_seeds": False,
        },
        "proposals": len(proposals),
        "classification": classified,
        "shadow_refusals": refusals,
        "bound_exceeded_on_real_units": bound_exceeded,
        "qualification": q_verdict,
        "shadow_record": {
            "n_inputs": shadow_record.get("n_inputs"),
            "same_inputs": shadow_record.get("same_inputs"),
            "parent_id": shadow_record.get("parent_id"),
            "candidate_id": shadow_record.get("candidate_id"),
            "executed_model": shadow_record.get("executed_model"),
            "gpu_authority": shadow_record.get("gpu_authority"),
            "seal_sha256": shadow_record.get("seal_sha256"),
        },
        "verdict": {
            "verdict": verdict.get("verdict"),
            "reason": verdict.get("reason"),
            "promoted": verdict.get("promoted"),
            "judge_id": verdict.get("judge_id"),
            "dominating_dimension": verdict.get("dominating_dimension"),
            "dominating_dimensions": verdict.get("dominating_dimensions"),
            "parent_id": verdict.get("parent_id"),
            "candidate_id": verdict.get("candidate_id"),
            "gpu_authority": verdict.get("gpu_authority"),
            "physical_dominance": verdict.get("physical_dominance"),
            "seal_sha256": verdict.get("seal_sha256"),
        },
        "numbers": numbers,
        "self_preference": self_pref,
        "orchestrator": {
            "completed_steps": snapshot.get("completed_steps"),
            "complete": snapshot.get("complete"),
            "switched": "switch" in (snapshot.get("completed_steps") or []),
            "active_id": snapshot.get("active_id"),
            "successor_id": snapshot.get("successor_id"),
            "rollback_parent_id": snapshot.get("rollback_parent_id"),
            "rollback_available": bool(orch.rollback_seal),
            "checkpoint_seal": None if orch.checkpoint is None else orch.checkpoint.get("seal_sha256"),
            "rollback_seal": None if orch.rollback_seal is None else orch.rollback_seal.get("seal_sha256"),
        },
        "launch_unqualified_refused": launch_refused,
        "stop": stopped,
        "switched": False,
        "qwen27_restorable": {
            "before": {
                "verdict": restorable.get("verdict"),
                "restorable": restorable.get("restorable"),
                "n_unmet": restorable.get("n_unmet"),
                "identity_id": restorable.get("identity_id"),
            },
            "after_stop": {
                "verdict": restorable_after.get("verdict"),
                "restorable": restorable_after.get("restorable"),
                "n_unmet": restorable_after.get("n_unmet"),
            },
            "restore_path_n_steps": restore.get("n_steps"),
            "restore_path_checkable": restore.get("independently_checkable"),
            "rollback_reverts": rollback_doc.get("reverts"),
            "rollback_does_not_revert": rollback_doc.get("does_not_revert"),
            "performed_restore": False,
            "started_model_process": False,
            "took_gpu_lease": False,
            "exercised_on_this_run": True,
            "not_citing_the_synthetic_run": True,
        },
        "physical_axes": physical_unknown,
        "physical_state": "UNKNOWN",
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "promoted": False,
    }
    run["vs_synthetic"] = compare_vs_synthetic(run)
    return run


# ---------------------------------------------------------------------------
# WorkUnits, receipt, CLI
# ---------------------------------------------------------------------------


def emit_trial_workunits(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    planning = (
        (
            "future.succession_trial.choose-real-candidate",
            "Choose a real on-disk ModelLake specimen; refuse Flash-Next and partial bodies.",
            "future.succession_trial.choose_candidate",
        ),
        (
            "future.succession_trial.shadow-real-workunits",
            "Clone live HCLI WorkUnits onto the real child and fire the four watched refusals.",
            "future.succession_trial.shadow",
        ),
        (
            "future.succession_trial.judge",
            "Independent judge over the real shadowed candidate. Promote or refuse with numbers.",
            "future.succession_trial.adjudicate",
        ),
        (
            "future.succession_trial.restore-qwen27",
            "Checkpoint the incumbent, stop the refused child, prove Qwen27 restorable on this run.",
            "future.succession_trial.restore",
        ),
    )
    for uid, desc, verifier in planning:
        row = emit_hcli_workunit(
            id=uid,
            role="science",
            description=desc,
            dependencies=["receipts/future/SUCCESSION.json"]
            if uid != "future.succession_trial.choose-real-candidate"
            else [],
            resource_class="STATIC_ANALYSIS",
            verifier=verifier,
            provider="future.succession_trial",
            effect_class="READ_ONLY",
            status="pending",
            classification="STATIC_ONLY",
            extras={
                "claim_boundary": CLAIM_BOUNDARY,
                "species": "resident_succession_trial",
                "may_promote": False,
                "may_modify_verifier": False,
                "output_receipt_path": f"receipts/future/{RECEIPT}",
                "command": "python3 tools/future/succession_trial.py --selftest",
                "budget": {
                    "attempts": DEFAULT_RETRY_BUDGET,
                    "max_repair_depth": MAX_REPAIR_DEPTH,
                    "max_repairs_per_root": MAX_REPAIRS_PER_ROOT,
                    "gpu_windows_held": 0,
                    "gpu_windows_requested": 0,
                    "wall_clock_s": None,
                },
            },
        )
        validate_emitted_unit(row)
        units.append(row)
    for axis in suc.QUALIFICATION_AXES:
        if axis.name not in suc.PHYSICAL_AXIS_NAMES:
            continue
        uid = f"future.succession_trial.sleeping.{axis.name}"
        row = emit_hcli_workunit(
            id=uid,
            role="science",
            description=(
                f"SLEEPING: physical axis {axis.name} is UNKNOWN until a protected "
                "hardware qualification exists. A real specimen does not default it."
            ),
            dependencies=["an existing HCLI protected lease", "machine quiescence"],
            resource_class="GPU_EXCLUSIVE",
            verifier=f"future.succession_trial.physical.{axis.name}",
            provider="future.succession_trial",
            effect_class="READ_ONLY",
            status="blocked",
            classification="SLEEPING",
            extras={
                "claim_boundary": CLAIM_BOUNDARY,
                "species": "resident_succession_trial",
                "axis": axis.name,
                "blocked_reason": (
                    f"{axis.name} is UNKNOWN: no GPU lease; four lanes running; "
                    "this trial does not contend"
                ),
                "requires_quiescence": True,
                "may_promote": False,
                "may_modify_verifier": False,
            },
        )
        validate_emitted_unit(row)
        units.append(row)
    units.sort(key=lambda r: str(r["id"]))
    return units


def recovered_implementation() -> list[dict[str, Any]]:
    return [
        {
            **suc._path_state("tools/future/succession.py"),
            "reused": True,
            "what": (
                "create_child / ShadowChild / qualify_child / IndependentJudge / "
                "SuccessionOrchestrator / stop_child. Not forked, not edited."
            ),
        },
        {
            **suc._path_state("tools/future/fallback_resident.py"),
            "reused": True,
            "what": "Qwen27 fallback_identity + verify_restorable + restore_path; exercised on this run",
        },
        {
            **suc._path_state("tools/future/specimen_verify.py"),
            "reused": True,
            "what": "list_specimens / specimen_dir / metadata readers; 1.5GB weight not re-hashed",
        },
        {
            **suc._path_state("tools/future/workunit_species.py"),
            "reused": True,
            "what": "HCLI_FUTURE_WORKUNITS.json is the real WorkUnit source",
        },
        {
            **suc._path_state("hcli/hawking-native.sealed-3.14.json"),
            "reused": True,
            "what": "sealed Qwen27 identity consumed via fallback_resident",
        },
    ]


def resident_callable(work_units: Sequence[Mapping[str, Any]], run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entry_point": "tools.future.succession_trial.run_real_succession_trial()",
        "cli": "tools/future/succession_trial.py:main",
        "invoke": [
            "python3 tools/future/succession_trial.py --selftest",
            "python3 tools/future/succession_trial.py --build",
        ],
        "callable": "tools.future.succession_trial.run_real_succession_trial",
        "workunit": (
            "one CPU_ANALYSIS unit; real ModelLake child; independent judge; "
            "Qwen27 restorable; PROMOTE unreachable"
        ),
        "frontier": "FT.CHILD_RESIDENT.install-dry-run",
        "fails_closed": (
            "TrialRefused if the specimen/manifest/resolved_sha is missing; "
            "ShadowAuthorityError on the four watched refusals; "
            "SelfPreferenceError if the incumbent promotes itself; "
            "qualification failure + judge REFUSE on declared param counts; "
            "stop_child rolls back to Qwen27"
        ),
        "work_units_emitted": [u["id"] for u in work_units],
        "receipt": f"receipts/future/{RECEIPT}",
        "hcli_can_invoke": True,
        "verdict_at_emission": (run.get("verdict") or {}).get("verdict"),
        "candidate": (run.get("candidate") or {}).get("slug"),
        "note": (
            "HCLI schedules the emitted WorkUnits. This sidecar does not start a "
            "resident process and does not take a GPU lease. succession.py is imported, not copied."
        ),
    }


def build(run: Mapping[str, Any] | None = None) -> Path:
    run_doc = dict(run) if run is not None else run_real_succession_trial()
    units = emit_trial_workunits(run_doc)
    lineage = (run_doc.get("child") or {}).get("lineage") or {}
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "REAL_CANDIDATE_REFUSED",
        "promoted": False,
        "built": True,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "purpose": (
            "Exercise tools/future/succession.py on a REAL on-disk ModelLake "
            "specimen rather than a synthetic child. Shadow, qualify, independent "
            "judge, stop-a-bad-child, Qwen27 restorable. Physical axes stay UNKNOWN."
        ),
        "head": git("rev-parse", "HEAD"),
        "vocabulary": {
            "eras": list(ERAS),
            "odysseys": list(ODYSSEYS),
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "disk_state_is_authority": True,
            "protected_absolute_not_emitted": True,
        },
        "candidate": run_doc["candidate"],
        "incumbent": run_doc["incumbent"],
        "child": {
            "id": run_doc["child"]["id"],
            "method": run_doc["child"]["method"],
            "parent_id": run_doc["child"]["parent_id"],
            "role": run_doc["child"]["role"],
            "create_child_stamped_synthetic": run_doc["create_child_stamped_synthetic"],
            "lineage_fields": list(suc.LINEAGE_FIELDS),
            "lineage": {
                field: lineage.get(field)
                for field in suc.LINEAGE_FIELDS
            },
        },
        "shadow": {
            "cloned_workunits": run_doc["cloned_workunits"],
            "proposals": run_doc["proposals"],
            "classification": run_doc["classification"],
            "refusals_proven": run_doc["shadow_refusals"],
            "forbidden_actions": list(suc.SHADOW_FORBIDDEN_ACTIONS),
            "bound_exceeded_on_real_units": run_doc["bound_exceeded_on_real_units"],
            "shadow_record": run_doc["shadow_record"],
        },
        "qualification": run_doc["qualification"],
        "judge": run_doc["verdict"],
        "numbers": run_doc["numbers"],
        "no_self_preference": run_doc["self_preference"],
        "succession": {
            "steps_completed": run_doc["orchestrator"]["completed_steps"],
            "complete": run_doc["orchestrator"]["complete"],
            "switched": run_doc["orchestrator"]["switched"],
            "active_id": run_doc["orchestrator"]["active_id"],
            "launch_unqualified_refused": run_doc["launch_unqualified_refused"],
            "stop": {
                "stopped": run_doc["stop"]["stopped"],
                "rolled_back": run_doc["stop"]["rolled_back"],
                "incumbent_restored": run_doc["stop"]["incumbent_restored"],
                "reason": run_doc["stop"]["reason"],
                "child_status": run_doc["stop"]["child_status"],
                "active_id": run_doc["stop"]["active_id"],
            },
            "checkpoint_seal": run_doc["orchestrator"]["checkpoint_seal"],
            "rollback_seal": run_doc["orchestrator"]["rollback_seal"],
        },
        "qwen27_restorable": run_doc["qwen27_restorable"],
        "physical_axes": run_doc["physical_axes"],
        "physical_state": "UNKNOWN",
        "vs_synthetic": run_doc["vs_synthetic"],
        "work_units": units,
        "counts": {
            "cloned_workunits": run_doc["cloned_workunits"]["n"],
            "shadow_refusals": len(run_doc["shadow_refusals"]),
            "qualification_axes": len(suc.QUALIFICATION_AXES),
            "physical_axes": len(suc.PHYSICAL_AXIS_NAMES),
            "work_units": len(units),
            "sleeping_work_units": sum(1 for u in units if u.get("classification") == "SLEEPING"),
        },
        "resident_callable": resident_callable(units, run_doc),
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": [
            "end-to-end succession exercised on a REAL ModelLake specimen (Qwen3-0.6B @ c1899de289a0), not a synthetic child",
            "nine-field lineage populated from the real manifest (repo, resolved_sha, manifest path, hash counts)",
            "shadow cloned live HCLI WorkUnits; four watched refusals fired live on this child",
            "qualification on the contract axes; physical stay UNKNOWN; child below incumbent declared-param floor",
            "independent judge REFUSE with the declared parameter counts that forced it",
            "incumbent cannot promote itself, proven on this real Qwen27 incumbent",
            "Qwen27 restorable-now exercised on this run (checkpoint, stop, rollback, verify_restorable)",
            "Flash-Next and Coder-30B inspected and rejected with disk reasons",
        ],
        "negative_findings": [
            "succession.py create_child still stamps synthetic=True; this trial does not edit that module",
            "physical axes (accepted_tps, token_ns, ebpw, active_bytes, resident_ram, cold/warm start, restart) stay UNKNOWN — no lease",
            "FLASH_NX_READY is False so Qwen3.8-Flash-Next was not the candidate",
            "Qwen3-Coder-30B-A3B is only in ModelLake partial/",
            "ModelLake n_sha256_verified=2 of 10; this trial hashed small files and parsed the weight header, not the 1.5GB body",
            "declared parameter counts are not a capability eval and not a protected measurement",
            "PROMOTE remains unreachable without executed GPU authority",
            "the synthetic SUCCESSION.json handover completed only because scores were invented 2 vs 1",
            "the Flash census specimens.entries snapshot does not list this slug; verified_receipts and the specimens/ directory do",
        ],
    }
    suc._refuse_hardware_numbers(doc)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    run = run_real_succession_trial()
    cand = run["candidate"]
    if cand["slug"] != CANDIDATE_SLUG:
        raise AssertionError(f"expected {CANDIDATE_SLUG}, got {cand['slug']}")
    if not cand["on_disk"] or not cand["resolved_sha"] or not cand["manifest_path"]:
        raise AssertionError("candidate is not a real on-disk specimen with a manifest")
    if Path(cand["specimen_path"]).is_dir() is False:
        raise AssertionError(f"specimen path missing: {cand['specimen_path']}")
    child = run["child"]
    for field in suc.LINEAGE_FIELDS:
        if field not in child["lineage"]:
            raise AssertionError(f"child missing lineage field {field}")
    source = child["lineage"]["source_model_lineage"]
    if source.get("resolved_sha") != cand["resolved_sha"]:
        raise AssertionError("lineage resolved_sha does not match the manifest")
    if source.get("n_sha256_verified") != cand["n_sha256_verified"]:
        raise AssertionError("lineage hash counts do not match the manifest")
    clones = run["cloned_workunits"]
    if clones["n"] < 1 or clones["includes_synthetic_seeds"]:
        raise AssertionError("shadow did not clone real HCLI WorkUnits")
    if not all(r.get("refused") and r.get("live") for r in run["shadow_refusals"]):
        raise AssertionError(f"four watched refusals were not live: {run['shadow_refusals']}")
    if len(run["shadow_refusals"]) != len(suc.SHADOW_FORBIDDEN_ACTIONS):
        raise AssertionError("expected four live shadow refusals")
    q = run["qualification"]
    if q.get("physical_state") != "UNKNOWN":
        raise AssertionError("physical_state must stay UNKNOWN")
    for name in suc.PHYSICAL_AXIS_NAMES:
        if (q.get("physical_axes") or {}).get(name) is not None:
            raise AssertionError(f"physical axis {name} was filled")
        if run["physical_axes"][name]["state"] != "UNKNOWN":
            raise AssertionError(f"{name} was not named UNKNOWN")
    if q.get("qualified") is True:
        raise AssertionError("0.6B must not qualify against the sealed 27B floor")
    v = run["verdict"]
    if v.get("verdict") not in {suc.VERDICT_REFUSE, suc.VERDICT_INSUFFICIENT}:
        raise AssertionError(f"judge must promote-or-refuse, got {v}")
    if v.get("verdict") == suc.VERDICT_PROMOTE or v.get("promoted") is True:
        raise AssertionError("PROMOTE was minted")
    if v.get("reason") != suc.REASON_DOMINATED_BY_PARENT:
        raise AssertionError(f"expected DOMINATED_BY_PARENT, got {v.get('reason')}")
    if v.get("dominating_dimension") != "capability":
        raise AssertionError(f"expected capability to dominate, got {v}")
    if run["numbers"]["candidate_header_params"] >= run["numbers"]["incumbent_parent_params"]:
        raise AssertionError("candidate param count is not below the incumbent; refuse path is a lie")
    if run["orchestrator"]["switched"] or run["switched"]:
        raise AssertionError("refused child was switched in")
    if not run["stop"]["incumbent_restored"]:
        raise AssertionError("Qwen27 was not restored on this run")
    if run["orchestrator"]["active_id"] != run["incumbent"]["id"]:
        raise AssertionError("active_id is not the Qwen27 incumbent")
    if run["qwen27_restorable"]["after_stop"]["verdict"] != fb.VERDICT_NOW:
        raise AssertionError(
            f"Qwen27 is not RESTORABLE_NOW after this run: {run['qwen27_restorable']}"
        )
    if not all(r.get("refused") for r in run["self_preference"]):
        raise AssertionError(f"incumbent self-preference proofs failed: {run['self_preference']}")
    return build(run)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        print(selftest())
        return 0
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
