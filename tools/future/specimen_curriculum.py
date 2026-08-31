"""First-wave specimen curriculum — one authority for role readiness.

`odyssey_launch` refuses on `specimen_curriculum_ready` until every role has a
live sealed specimen. This module is that check. It does not invent a specimen,
does not lower the number of roles, and does not mark a role ready when its
specimen is unpublished or unverified. ModelLake is read, never written.

Five first-wave roles. Other lake entries are recorded and deferred.

A role is READY only with a verified specimen identity (repo, resolved_sha,
manifest path, hash verification counts) earned by recomputation. A role that
cannot be filled from disk is BLOCKED with the S022 §55 fields (wake
condition, blocked reason, required resource, reevaluation trigger). Both
outcomes are progress. A criterion made to pass by editing the criterion is not.

    python3 tools/future/specimen_curriculum.py --build
    python3 -m pytest tools/future/test_specimen_curriculum.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tools.future._common import RECEIPTS, REPO, load_json, write_receipt
from tools.future import external_specimen_seal as ess
from tools.future import odyssey2_law_store as ols
from tools.future import specimen_verify as sv


RECEIPT = "SPECIMEN_CURRICULUM.json"
SCHEMA = "hawking.future.specimen_curriculum.v1"
CURRICULUM_SCHEMA = "hawking.future.odyssey_i.specimen_curriculum.v1"
VERSION = 1
RECORDED_BY = "tools/future/specimen_curriculum.py"

CURRICULUM_ROLES: tuple[tuple[str, str], ...] = (
    ("very_small_dense_procedural_speed", "very small dense for procedural speed"),
    ("small_dense_alternate_architecture_transfer", "small dense alternate architecture for transfer"),
    ("mid_size_dense_compiler", "mid-size dense for compiler"),
    ("qwen27_mature_physical", "Qwen27 for mature physical"),
    ("flash_heterogeneous_frontier", "Flash for heterogeneous frontier"),
)

# S022 §55 blocked-role fields. A blocked role is a named dependency, not a
# silent refusal: the supervisor can sleep on these and wake from them.
S022_BLOCK_FIELDS: tuple[str, ...] = (
    "blocked_reason",
    "wake_condition",
    "required_resource",
    "reevaluation_trigger",
)

GAP_ABSENT = "specimen_absent_from_modellake"
GAP_UNPUBLISHED = "present_but_unpublished"
GAP_UNVERIFIED = "published_but_not_verified"
GAP_NOT_LISTED = "verified_but_not_in_specimens_listing"
GAP_NO_CANDIDATE = "no_candidate_assigned"

QWEN27_SPECIMEN = "qwen3.8-27b-abliterated-bf16@local"
QWEN06_PARTIAL = "Qwen--Qwen3-0.6B@c1899de289a0#partial"


def _probe_json(*rels: str) -> dict[str, Any]:
    """Defer to odyssey_launch.probe_json so tests that patch it still bind."""
    from tools.future.odyssey_launch import probe_json
    return probe_json(*rels)


def _checkout_roots() -> list[Path]:
    from tools.future.odyssey_launch import _checkout_roots as _roots
    return _roots()


def _specimen_dirs_on_disk() -> set[str]:
    """Specimen directory names actually present in ModelLake. Read-only."""
    root = sv.SPECIMENS
    try:
        return {p.name for p in root.iterdir() if p.is_dir()}
    except OSError:
        return set()  # the lake may not be mounted; that is not an error here


def _manifests_on_disk() -> dict[str, dict[str, Any]]:
    """ModelLake manifests, read-only. The census is a cache of these."""
    out: dict[str, dict[str, Any]] = {}
    try:
        paths = list(sv.MANIFESTS.glob("*.json"))
    except OSError:
        return out
    for path in paths:
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, UnicodeError):
            continue
        if not isinstance(doc, dict):
            continue
        repo = doc.get("repo")
        if not isinstance(repo, str) or not repo:
            continue
        out[repo] = {
            "repo": repo,
            "revision": doc.get("revision") or doc.get("resolved_sha"),
            "resolved_sha": doc.get("resolved_sha"),
            "manifest_path": str(path),
            "specimen_path": doc.get("path"),
            "n_files": doc.get("n_files"),
            "n_sha256_verified": doc.get("n_sha256_verified"),
            "n_size_only_verified": doc.get("n_size_only_verified"),
            "whole_tree_verified": False,
            "in_specimens_listing": False,
            "source": "modellake_manifest",
        }
    return out


def _independently_verified() -> dict[str, dict[str, Any]]:
    """Specimens whose every published digest was RECOMPUTED and matched.

    ModelLake's own seals say MANIFEST_ONLY because it verified most files by
    size. That is not verification and the gate is right to refuse it. But the
    digests were never missing -- HuggingFace writes a .metadata sidecar per
    file -- so whole-tree verification can be EARNED offline, and this reads the
    receipt where it was earned.

    Strict on purpose: a row counts only if it hashed real bytes, matched every
    file, and had no file it could not check. Anything softer would turn a
    correct refusal into a false readiness, which is the exact failure this
    gate exists to prevent.
    """
    rec = _probe_json("receipts/future/SPECIMEN_VERIFICATION.json")
    doc = rec.get("doc") if isinstance(rec.get("doc"), Mapping) else None
    out: dict[str, dict[str, Any]] = {}
    for row in (doc or {}).get("results") or []:
        if not isinstance(row, Mapping):
            continue
        if row.get("status") != "WHOLE_TREE_VERIFIED":
            continue
        if not (isinstance(row.get("bytes_hashed"), int) and row["bytes_hashed"] > 0):
            continue
        if row.get("mismatched") or row.get("no_remote_digest"):
            continue
        if row.get("unrecognized_digest") or row.get("skipped_time_budget"):
            continue
        if row.get("verified") != row.get("n_files"):
            continue
        out[str(row.get("specimen") or "")] = dict(row)
    return out


def _ready(identity: Mapping[str, Any], *, require_lake_verified: bool) -> tuple[bool, str]:
    if require_lake_verified and not identity.get("whole_tree_verified"):
        if identity.get("published_as_verified") is False:
            return False, "ModelLake pin exists but the specimen is not published as verified"
        if identity.get("in_specimens_listing") and not identity.get("whole_tree_verified"):
            n_sha = identity.get("n_sha256_verified")
            n_files = identity.get("n_files")
            return False, (
                f"ModelLake manifest is partial "
                f"(n_sha256_verified={n_sha} n_files={n_files}); not a sealed specimen"
            )
        if not identity.get("in_specimens_listing"):
            if identity.get("authorized_external") and (
                identity.get("specimen_path") or identity.get("authorized_external_path")
            ):
                return False, (
                    "authorized external specimen is on disk but is not whole-tree "
                    "verified; location is not a seal"
                )
            return False, "identity known but specimen is not in the ModelLake specimens listing"
        return False, "ModelLake publication is not whole-tree verified"
    if identity.get("patient_state") == "RETIRED":
        # The odysseys are recurrent phases and the first canonical completion
        # is historical, so a patient retired from that first wave is a
        # specimen with a PROVEN role, not a disqualified one. It counts only
        # when both halves hold: the prior seal exists, and the specimen has
        # been independently whole-tree verified NOW. Retirement alone would
        # be a pass on a stale seal; verification alone would lose the prior
        # work. Recurrence is recorded so nothing downstream reads a repeat
        # phase as a first-wave result.
        if identity.get("whole_tree_verified") and identity.get("patient_seal"):
            return True, (
                "RECURRENT_PATIENT: retired from the historical first wave, prior "
                "seal intact, and whole-tree verified again now"
            )
        return False, (
            "prior Odyssey I patient is RETIRED and has not been whole-tree "
            "verified again; a stale seal is not a live first-wave specimen"
        )
    if identity.get("physical_status") == "metadata_only_weights_not_present":
        # A declared status does not outrank a measurement. This field says the
        # weights are not present; for Flash that is 335GB and 131 safetensors
        # shards, whole-tree verified 144 of 144 by recomputing every published
        # digest. The declaration was true when the law store was written and is
        # false now, and deferring to it would refuse a specimen on the strength
        # of a stale string -- the same failure as the moved Doctor parent, the
        # absent-but-present GPU, and the specimen filed under partial/.
        #
        # Measurement wins only when it is REAL: whole-tree verified AND bytes
        # actually hashed. A status flip on anything less would be exactly the
        # laundering this refuses.
        if not (identity.get("whole_tree_verified") and (identity.get("bytes_hashed") or 0) > 0):
            return False, "school identity is metadata-only; weights are not present"
    if not identity.get("revision") and not identity.get("resolved_sha") and not identity.get("patient_seal"):
        # A local directory has no repository revision. The external specimen
        # seal is that identity; lake specimens cannot take this branch.
        try:
            from tools.future.external_specimen_seal import accept_as_sealed_identity
        except ImportError:
            return False, "no sealed revision or patient seal"
        ok, why = accept_as_sealed_identity(identity)
        if ok:
            return True, why
        return False, "no sealed revision or patient seal"
    if identity.get("whole_tree_verified"):
        return True, "ModelLake whole-tree sha256 verification"
    return False, "sealed identity is not enough; live first-wave specimen is not published"


def _merge_manifest_row(dst: dict[str, Any], src: Mapping[str, Any]) -> None:
    for key in (
        "n_files",
        "n_sha256_verified",
        "n_size_only_verified",
        "manifest_path",
        "resolved_sha",
        "revision",
        "specimen_path",
        "repo",
    ):
        if dst.get(key) in (None, "", False) and src.get(key) not in (None, ""):
            dst[key] = src[key]
    if not dst.get("source"):
        dst["source"] = src.get("source") or "modellake_manifest"


def _lake_index(census: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Index ModelLake census manifests and specimen dirs by repo / slug."""
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(census, Mapping):
        census = {}
    verified = census.get("verified_receipts") if isinstance(census.get("verified_receipts"), Mapping) else {}
    for row in verified.get("receipts") or []:
        if not isinstance(row, Mapping):
            continue
        repo = str(row.get("repo") or "")
        if not repo:
            continue
        out[repo] = {
            "repo": repo,
            "revision": row.get("revision") or row.get("resolved_sha"),
            "resolved_sha": row.get("resolved_sha"),
            "manifest_path": row.get("path"),
            "specimen_path": row.get("specimen_path"),
            "n_files": row.get("n_files"),
            "n_sha256_verified": row.get("n_sha256_verified"),
            "n_size_only_verified": row.get("n_size_only_verified"),
            "whole_tree_verified": False,
            "in_specimens_listing": False,
            "source": "modellake_manifest",
        }
    # Disk manifests fill rows the census never recorded (Mistral, Flash).
    for repo, row in _manifests_on_disk().items():
        if repo not in out:
            out[repo] = dict(row)
        else:
            _merge_manifest_row(out[repo], row)
    specimens = census.get("specimens") if isinstance(census.get("specimens"), Mapping) else {}
    names = {str(e.get("name")) for e in (specimens.get("entries") or []) if isinstance(e, Mapping)}
    # DISK STATE IS AUTHORITY. The census is a cache of it, and a specimen the
    # census never recorded still exists. Reading only the census reported
    # Mistral-Small-24B as "not in the ModelLake specimens listing" while its
    # directory was sitting in that listing -- a wrong reason attached to a
    # correct refusal, which is the kind of error that sends work to the wrong
    # place. ModelLake is read, never written.
    disk_names = _specimen_dirs_on_disk()
    names |= disk_names
    for repo, row in out.items():
        slug = (row.get("specimen_path") or "").rstrip("/").split("/")[-1]
        row["in_specimens_listing"] = slug in names
        n_files = row.get("n_files")
        n_sha = row.get("n_sha256_verified")
        row["whole_tree_verified"] = bool(
            row["in_specimens_listing"]
            and isinstance(n_files, int)
            and isinstance(n_sha, int)
            and n_files > 0
            and n_sha == n_files
        )
    # A specimen present on disk but absent from the census needs a row of its
    # own, or the disk fallback above can only correct rows the cache already
    # had -- which is how Mistral-Small-24B stayed invisible.
    for name in sorted(disk_names):
        slug, _, rev = name.partition("@")
        repo = slug.replace("--", "/", 1)
        if repo in out:
            out[repo]["in_specimens_listing"] = True
            if not out[repo].get("specimen_path"):
                out[repo]["specimen_path"] = f"{sv.SPECIMENS}/{name}"
            continue
        out[repo] = {
            "repo": repo,
            "revision": rev or None,
            "resolved_sha": rev or None,
            "manifest_path": None,
            "specimen_path": str(sv.SPECIMENS / name),
            "n_files": None,
            "n_sha256_verified": None,
            "n_size_only_verified": None,
            "whole_tree_verified": False,
            "in_specimens_listing": True,
            "source": "modellake_specimens_dir",
        }

    flash = census.get("source") if isinstance(census.get("source"), Mapping) else {}
    if flash.get("repo"):
        repo = str(flash.get("repo"))
        checks = census.get("checks") if isinstance(census.get("checks"), Mapping) else {}
        manifest = census.get("flash_target_manifest") if isinstance(census.get("flash_target_manifest"), Mapping) else {}
        prior = dict(out.get(repo) or {})
        specimen_path = prior.get("specimen_path") or manifest.get("final_root")
        slug = (specimen_path or "").rstrip("/").split("/")[-1]
        on_disk = bool(prior.get("in_specimens_listing")) or slug in disk_names
        out[repo] = {
            "repo": repo,
            "revision": prior.get("revision") or flash.get("requested_revision") or flash.get("revision"),
            "resolved_sha": prior.get("resolved_sha") or flash.get("requested_revision"),
            "manifest_path": prior.get("manifest_path"),
            "specimen_path": specimen_path,
            "n_files": prior.get("n_files"),
            "n_sha256_verified": (
                prior.get("n_sha256_verified")
                if prior.get("n_sha256_verified") is not None
                else manifest.get("verified_file_count")
            ),
            "n_size_only_verified": prior.get("n_size_only_verified"),
            "whole_tree_verified": bool(prior.get("whole_tree_verified") or manifest.get("whole_tree_verified")),
            "in_specimens_listing": on_disk or bool(manifest.get("final_present")),
            "published_as_verified": (
                bool(prior.get("published_as_verified"))
                if prior.get("published_as_verified") is not None
                else (not bool(checks.get("target_not_published_as_verified", True)))
            ),
            "source": prior.get("source") or "flash_pinned_census",
            "census_qualification": census.get("qualification"),
        }
        # A stale census "final_present=false" must not un-find a directory
        # sitting in specimens/. Disk state is authority.
        if on_disk:
            out[repo]["in_specimens_listing"] = True
    # Earned verification is applied LAST, after every row exists. It used to run
    # before the pinned-Flash census branch appended its row, so Flash was the one
    # specimen that could never inherit its own whole-tree result -- 144 of 144
    # files recomputed and the role still refused. An overlay that runs before the
    # rows it overlays is a silent no-op for whatever comes after it.
    earned = _independently_verified()
    for row in out.values():
        slug = (row.get("specimen_path") or "").rstrip("/").split("/")[-1]
        hit = earned.get(slug)
        if not hit:
            continue
        row["whole_tree_verified"] = True
        row["verification_source"] = "tools/future/specimen_verify.py (offline recomputation)"
        row["bytes_hashed"] = hit.get("bytes_hashed")
        row["in_specimens_listing"] = True
        row["published_as_verified"] = True
        if hit.get("n_files") is not None:
            row["n_files_verified"] = hit.get("n_files")
            row["n_files_matched"] = hit.get("verified")

    return out


def _odyssey_i_patients() -> list[dict[str, Any]]:
    """Patient seals. Sparse checkout: search the primary tree too."""
    seen: dict[str, dict[str, Any]] = {}
    try:
        roots = _checkout_roots()
    except Exception:
        roots = [REPO]
    for root in roots:
        folder = Path(root) / "receipts" / "odyssey-i"
        if not folder.is_dir():
            continue
        for seal in sorted(folder.glob("*_PATIENT_SEAL.json")):
            try:
                doc = load_json(seal)
            except (OSError, json.JSONDecodeError):
                continue
            oxx = doc.get("oxx") or seal.name.split("_", 1)[0]
            if oxx in seen:
                continue
            ext_path = folder / f"{oxx}_EXTERNAL.json"
            weights = None
            if ext_path.is_file():
                try:
                    ext = load_json(ext_path)
                    weights = ext.get("weights_canonical")
                except (OSError, json.JSONDecodeError):
                    weights = None
            try:
                rel = str(seal.relative_to(REPO))
            except ValueError:
                rel = str(seal)
            seen[str(oxx)] = {
                "oxx": oxx,
                "status": doc.get("status"),
                "state": doc.get("state"),
                "seal": rel,
                "weights_canonical": weights,
                "sealed_mechanisms": list(doc.get("sealed_mechanisms") or []),
            }
    return [seen[k] for k in sorted(seen)]


def classify_gap(identity: Mapping[str, Any]) -> str:
    """Why a role is unready. Roles are not unready for the same reason."""
    listed = bool(identity.get("in_specimens_listing"))
    verified = bool(identity.get("whole_tree_verified"))
    published = identity.get("published_as_verified")
    has_identity = bool(
        identity.get("repo")
        or identity.get("specimen")
        or identity.get("specimen_path")
        or identity.get("revision")
        or identity.get("resolved_sha")
    )
    if not has_identity:
        return GAP_NO_CANDIDATE
    if verified and not listed:
        return GAP_NOT_LISTED
    if listed and not verified:
        if published is False:
            return GAP_UNPUBLISHED
        return GAP_UNVERIFIED
    if identity.get("authorized_external") and not listed:
        return GAP_ABSENT
    if not listed:
        return GAP_ABSENT
    if published is False:
        return GAP_UNPUBLISHED
    return GAP_UNVERIFIED


def _slug_for(identity: Mapping[str, Any]) -> str:
    specimen = identity.get("specimen")
    if isinstance(specimen, str) and specimen:
        return specimen
    path = str(identity.get("specimen_path") or identity.get("authorized_external_path") or "")
    if path:
        name = path.rstrip("/").split("/")[-1]
        if name:
            return name
    repo = str(identity.get("repo") or "")
    rev = str(identity.get("revision") or identity.get("resolved_sha") or "")[:12]
    if repo and rev:
        return f"{repo.replace('/', '--')}@{rev}"
    return repo or "unassigned"


def blocked_record(role: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    """S022 §55: wake condition, blocked reason, required resource, reevaluation trigger."""
    gap = classify_gap(identity)
    slug = _slug_for(identity)
    path = (
        identity.get("specimen_path")
        or identity.get("authorized_external_path")
        or identity.get("manifest_path")
    )
    reason = str(role.get("ready_reason") or "role is not ready")
    verify_cmd = f"python3 tools/future/specimen_verify.py --verify {slug}"
    if gap == GAP_NO_CANDIDATE:
        return {
            "blocked_reason": reason,
            "wake_condition": (
                f"a candidate specimen is assigned to role {role.get('role')!r}"
            ),
            "required_resource": f"curriculum role {role.get('role')} candidate",
            "reevaluation_trigger": (
                "python3 tools/future/specimen_curriculum.py --build after a candidate is assigned"
            ),
            "gap_class": gap,
        }
    if gap == GAP_ABSENT and identity.get("authorized_external"):
        return {
            "blocked_reason": reason,
            "wake_condition": (
                f"WHOLE_TREE_VERIFIED row for {slug} in receipts/future/SPECIMEN_VERIFICATION.json "
                "and EXTERNAL_SPECIMEN_SEAL.json status=SEALED"
            ),
            "required_resource": str(path or sv.EXTRA_SPECIMENS.get(QWEN27_SPECIMEN) or slug),
            "reevaluation_trigger": verify_cmd,
            "gap_class": gap,
        }
    if gap == GAP_ABSENT:
        return {
            "blocked_reason": reason,
            "wake_condition": (
                f"specimen for {identity.get('repo') or role.get('role')} present under "
                f"{sv.SPECIMENS}"
            ),
            "required_resource": str(path or identity.get("repo") or slug),
            "reevaluation_trigger": (
                "python3 tools/future/specimen_curriculum.py --build after the specimen "
                "lands in ModelLake specimens/"
            ),
            "gap_class": gap,
        }
    if gap == GAP_UNPUBLISHED:
        return {
            "blocked_reason": reason,
            "wake_condition": (
                f"WHOLE_TREE_VERIFIED row for {slug} in receipts/future/SPECIMEN_VERIFICATION.json "
                "(sidecar publication; ModelLake is not mutated)"
            ),
            "required_resource": str(path or slug),
            "reevaluation_trigger": verify_cmd,
            "gap_class": gap,
        }
    if gap == GAP_NOT_LISTED:
        return {
            "blocked_reason": reason,
            "wake_condition": f"{slug} present in {sv.SPECIMENS}",
            "required_resource": str(path or slug),
            "reevaluation_trigger": (
                "python3 tools/future/specimen_curriculum.py --build after the specimen "
                "is listed under ModelLake specimens/"
            ),
            "gap_class": gap,
        }
    return {
        "blocked_reason": reason,
        "wake_condition": (
            f"WHOLE_TREE_VERIFIED row for {slug} in receipts/future/SPECIMEN_VERIFICATION.json"
        ),
        "required_resource": str(path or slug),
        "reevaluation_trigger": verify_cmd,
        "gap_class": gap,
    }


def verified_specimen_record(role: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    """Traceable identity for a ready role. Nothing here is invented."""
    ml = role.get("modellake") if isinstance(role.get("modellake"), Mapping) else {}
    local = role.get("local_specimen") if isinstance(role.get("local_specimen"), Mapping) else {}
    sha = (
        ml.get("resolved_sha")
        or identity.get("resolved_sha")
        or role.get("revision")
        or local.get("resolved_sha")
    )
    identity_kind = "modellake_sha"
    manifest_path = ml.get("manifest_path") or identity.get("manifest_path")
    if not sha:
        seal = ess.load_seal() if identity.get("authorized_external") else None
        if isinstance(seal, Mapping) and seal.get("status") == "SEALED":
            sha = seal.get("tree_digest")
            identity_kind = "external_tree_digest"
            manifest_path = str(RECEIPTS / ess.RECEIPT)
    earned_n = (
        local.get("verified")
        if local.get("verified") is not None
        else ml.get("n_files_matched")
        if ml.get("n_files_matched") is not None
        else identity.get("n_files_matched")
    )
    earned_files = (
        local.get("n_files")
        if local.get("n_files") is not None
        else ml.get("n_files_verified")
        if ml.get("n_files_verified") is not None
        else identity.get("n_files_verified")
    )
    n_files = (
        earned_files
        if earned_files is not None
        else ml.get("n_files")
        if ml.get("n_files") is not None
        else identity.get("n_files")
    )
    n_sha = (
        earned_n
        if earned_n is not None
        else ml.get("n_sha256_verified")
        if ml.get("n_sha256_verified") is not None
        else identity.get("n_sha256_verified")
    )
    n_size = 0 if earned_n is not None else ml.get("n_size_only_verified")
    return {
        "repo": role.get("repo") or identity.get("repo"),
        "resolved_sha": sha,
        "revision": role.get("revision") or identity.get("revision"),
        "manifest_path": manifest_path,
        "specimen_path": (
            ml.get("specimen_path")
            or identity.get("specimen_path")
            or local.get("specimen_path")
        ),
        "n_files": n_files,
        "n_sha256_verified": n_sha,
        "n_size_only_verified": n_size,
        "bytes_hashed": (
            ml.get("bytes_hashed")
            or identity.get("bytes_hashed")
            or local.get("bytes_hashed")
        ),
        "verification_source": (
            ml.get("verification_source")
            or identity.get("verification_source")
            or ("tools/future/specimen_verify.py (offline recomputation)" if local else None)
        ),
        "whole_tree_verified": True,
        "in_specimens_listing": bool(ml.get("in_specimens_listing") or identity.get("in_specimens_listing")),
        "published_as_verified": ml.get("published_as_verified", identity.get("published_as_verified")),
        "identity_kind": identity_kind,
        "specimen_owner": identity.get("specimen_owner") or ml.get("specimen_owner") or local.get("owner"),
    }


def _annotate_role(role: dict[str, Any], identity: Mapping[str, Any]) -> None:
    if role.get("ready"):
        rec = verified_specimen_record(role, identity)
        role["verified_specimen"] = rec
        role["blocked"] = None
        role["gap_class"] = None
        return
    role["verified_specimen"] = None
    role["blocked"] = blocked_record(role, identity)
    role["gap_class"] = role["blocked"]["gap_class"]


def propose_specimen_curriculum(census_doc: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """First specimen set by curriculum role. Not 'every model in the lake'."""
    if census_doc is None:
        probe = _probe_json(
            "receipts/future/evidence/HCLI_MODELLAKE_FLASH_CENSUS.json",
            "receipts/headless/HCLI_MODELLAKE_FLASH_CENSUS.json",
            "receipts/headless/MODELLAKE_FLASH_NEXT_CENSUS.json",
        )
        census_doc = probe.get("doc")
        census_probe = probe
    else:
        census_probe = {"found": True, "path_taken": "caller", "rel": None, "resolved": None}

    lake = _lake_index(census_doc if isinstance(census_doc, Mapping) else None)
    patients = _odyssey_i_patients()
    schools = {k: dict(v) for k, v in ols.SCHOOLS.items()}
    earned = _independently_verified()

    def _patient_for(*needles: str) -> dict[str, Any] | None:
        for p in patients:
            blob = " ".join(
                str(x) for x in (p.get("oxx"), p.get("weights_canonical"), p.get("seal"))
            ).lower()
            if any(n.lower() in blob for n in needles):
                return p
        return None

    roles: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []

    q06 = dict(lake.get("Qwen/Qwen3-0.6B") or {})
    # The gate reported this specimen as absent from the ModelLake specimens
    # listing, which was true and was read as "the model is not here". It is
    # here -- complete, inside ModelLake, under partial/ rather than specimens/
    # (historically) or now promoted into specimens/. Location was the only
    # thing partial about it. ModelLake still owns it and nothing here moves
    # or writes to it.
    q06_partial = earned.get(QWEN06_PARTIAL) or {}
    if q06_partial:
        q06.update({
            "whole_tree_verified": True,
            "in_specimens_listing": True,
            "specimen_owner": "modellake_partial",
            "specimen_path": q06_partial.get("specimen_path"),
            "verification_source": "tools/future/specimen_verify.py (offline recomputation)",
            "bytes_hashed": q06_partial.get("bytes_hashed"),
        })
        q06.setdefault("revision", "c1899de289a0")
    q06_id = dict(q06)
    identities.append(q06_id)
    roles.append(
        {
            "role": CURRICULUM_ROLES[0][0],
            "purpose": CURRICULUM_ROLES[0][1],
            "repo": q06.get("repo") or "Qwen/Qwen3-0.6B",
            "revision": q06.get("revision"),
            "architecture_family": "dense_transformer",
            "identity_source": q06.get("source") or "modellake_manifest",
            "modellake": q06,
            "located_under_partial": bool(q06_partial),
            **dict(zip(("ready", "ready_reason"), _ready(q06_id, require_lake_verified=True))),
        }
    )

    falcon = lake.get("tiiuae/Falcon-H1-7B-Instruct") or {}
    p001 = _patient_for("falcon-h1", "O001")
    falcon_id = dict(falcon)
    if p001:
        falcon_id["patient_seal"] = p001.get("seal")
        falcon_id["patient_state"] = p001.get("state")
        falcon_id["patient_status"] = p001.get("status")
    identities.append(falcon_id)
    roles.append(
        {
            "role": CURRICULUM_ROLES[1][0],
            "purpose": CURRICULUM_ROLES[1][1],
            "repo": falcon.get("repo") or "tiiuae/Falcon-H1-7B-Instruct",
            "revision": falcon.get("revision"),
            "architecture_family": "falcon_h1",
            "identity_source": "modellake_manifest+odyssey_i_O001",
            "modellake": falcon,
            "prior_odyssey_i": p001,
            **dict(zip(("ready", "ready_reason"), _ready(falcon_id, require_lake_verified=True))),
        }
    )

    mistral_partial = None
    stale = []
    if isinstance(census_doc, Mapping):
        stale = list(census_doc.get("stale_partial_candidates") or [])
    for row in stale:
        path = str(row.get("path") or "")
        if "Mistral-Small" in path:
            mistral_partial = row
            break
    p004 = _patient_for("mistral-small", "O004", "24B")
    # Consult the index rather than asserting absence. This role hardcoded
    # in_specimens_listing=False and so reported "not in the ModelLake specimens
    # listing" for a specimen whose 89GB directory was sitting in that listing --
    # a wrong reason on a correct refusal, which sends the next worker to fix
    # the wrong thing.
    mistral_lake = dict(lake.get("mistralai/Mistral-Small-3.1-24B-Instruct-2503") or {})
    mistral_id = {
        **mistral_lake,
        "repo": mistral_lake.get("repo") or "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "revision": mistral_lake.get("revision")
        or mistral_lake.get("resolved_sha")
        or "68faf511d618ef198fef186659617cfd2eb8e33a",
        "stale_partial": mistral_partial,
        "patient_seal": None if not p004 else p004.get("seal"),
        "patient_state": None if not p004 else p004.get("state"),
        "source": mistral_lake.get("source") or "odyssey_i_O004+modellake_partial",
    }
    identities.append(mistral_id)
    lake_view = dict(mistral_lake)
    if mistral_partial is not None:
        lake_view["stale_partial"] = mistral_partial
    roles.append(
        {
            "role": CURRICULUM_ROLES[2][0],
            "purpose": CURRICULUM_ROLES[2][1],
            "repo": mistral_id["repo"],
            "revision": mistral_id["revision"],
            "architecture_family": "dense_transformer",
            "identity_source": mistral_id["source"],
            "modellake": lake_view,
            "prior_odyssey_i": p004,
            **dict(zip(("ready", "ready_reason"), _ready(mistral_id, require_lake_verified=True))),
        }
    )

    q27 = dict(schools.get("Qwen27") or {})
    # The Qwen27 parent is not a ModelLake specimen and never was. It is the
    # 52GB directory the Doctor and Gravity tools read, it carries the same
    # HuggingFace .metadata digests, and it is verified by exactly the same rule.
    # It is labelled local_directory so it is never mistaken for a sealed lake
    # specimen, and ModelLake's ownership of the lake is untouched.
    q27_local = earned.get(QWEN27_SPECIMEN) or {}
    extra_path = sv.EXTRA_SPECIMENS.get(QWEN27_SPECIMEN)
    extra_present = bool(extra_path and extra_path.is_dir())
    q27_id = {
        "repo": q27.get("source_model") or "Qwen3.8-27B",
        "revision": None,
        "specimen": QWEN27_SPECIMEN if extra_present or q27_local else None,
        "architecture_family": q27.get("architecture_family"),
        "in_specimens_listing": (
            "Qwen3.8-27B" in lake or "Qwen/Qwen3.8-27B" in lake
        ),
        "whole_tree_verified": bool(q27_local),
        "bytes_hashed": q27_local.get("bytes_hashed") or 0,
        "specimen_owner": "local_directory" if (extra_present or q27_local) else "modellake",
        "specimen_path": (
            q27_local.get("specimen_path")
            or (str(extra_path) if extra_present else None)
        ),
        "authorized_external": extra_present or bool(q27_local),
        "physical_status": q27.get("physical_status"),
        "source": "odyssey2_law_store.SCHOOLS.Qwen27",
        "n_files": q27_local.get("n_files"),
        "n_sha256_verified": q27_local.get("verified"),
    }
    identities.append(q27_id)
    roles.append(
        {
            "role": CURRICULUM_ROLES[3][0],
            "purpose": CURRICULUM_ROLES[3][1],
            "repo": q27_id["repo"],
            "revision": q27_id["revision"],
            "architecture_family": q27_id["architecture_family"],
            "identity_source": q27_id["source"],
            "school": q27,
            "modellake": lake.get("Qwen3.8-27B") or lake.get("Qwen/Qwen3.8-27B") or {},
            "local_specimen": q27_local or None,
            **dict(zip(("ready", "ready_reason"), _ready(q27_id, require_lake_verified=True))),
        }
    )

    flash_school = dict(schools.get("Flash") or {})
    flash_lake = dict(lake.get("Qwen/Qwen3.8-Flash-Next") or {})
    flash_id = dict(flash_lake)
    flash_id["physical_status"] = flash_school.get("physical_status") or flash_id.get("physical_status")
    identities.append(flash_id)
    roles.append(
        {
            "role": CURRICULUM_ROLES[4][0],
            "purpose": CURRICULUM_ROLES[4][1],
            "repo": flash_lake.get("repo") or flash_school.get("source_model") or "Qwen/Qwen3.8-Flash-Next",
            "revision": flash_lake.get("revision") or flash_school.get("pinned_revision"),
            "architecture_family": flash_school.get("architecture_family") or "qwen4_exp",
            "identity_source": "hcli.flash_next pin + modellake census + odyssey2 school",
            "school": flash_school,
            "modellake": flash_lake,
            **dict(zip(("ready", "ready_reason"), _ready(flash_id, require_lake_verified=True))),
        }
    )

    for role, identity in zip(roles, identities):
        _annotate_role(role, identity)

    first_wave_repos = {r["repo"] for r in roles}
    lake_extras = [
        {"repo": repo, "revision": row.get("revision"), "reason": "present in ModelLake, not a first-wave curriculum role"}
        for repo, row in sorted(lake.items())
        if repo not in first_wave_repos
    ]

    n_ready = sum(1 for r in roles if r.get("ready"))
    assert len(roles) == len(CURRICULUM_ROLES), "refusing to lower n_roles to improve the ratio"
    return {
        "schema": CURRICULUM_SCHEMA,
        "n_roles": len(CURRICULUM_ROLES),
        "n_ready": n_ready,
        "ready": n_ready == len(CURRICULUM_ROLES) and len(CURRICULUM_ROLES) > 0,
        "roles": roles,
        "not_proposed": lake_extras,
        "not_proposed_rule": (
            "Do not exhaustively optimize every downloaded model. First-wave "
            "curriculum is the five roles above; other lake entries are recorded "
            "and deferred."
        ),
        "census_probe": {
            "found": bool(census_probe.get("found")),
            "path_taken": census_probe.get("path_taken"),
            "resolved": census_probe.get("resolved"),
        },
        "prior_odyssey_i_patients": patients,
        "authority": (
            "tools/future/specimen_curriculum.py owns first-wave readiness. "
            "odyssey_launch._eval_curriculum reads this; it does not re-derive the roles."
        ),
    }


def build(*, writer=None) -> dict[str, Any]:
    write = writer or write_receipt
    cur = propose_specimen_curriculum()
    blocked = [r for r in cur["roles"] if not r.get("ready")]
    ready = [r for r in cur["roles"] if r.get("ready")]
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "First-wave specimen curriculum: five roles, each either a verified "
            "specimen identity or a S022 §55 blocked dependency. A proposal is "
            "not a ready specimen set."
        ),
        "authority": cur.get("authority"),
        "measurement_class": "STATIC_ONLY",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "n_roles": cur["n_roles"],
        "n_ready": cur["n_ready"],
        "ready": cur["ready"],
        "unready": [r["role"] for r in blocked],
        "curriculum": cur,
        "roles": cur["roles"],
        "s022_section_55": {
            "rule": (
                "A blocked role records wake_condition, blocked_reason, "
                "required_resource, reevaluation_trigger. That converts an "
                "unexplained refusal into a named dependency."
            ),
            "fields": list(S022_BLOCK_FIELDS),
            "blocked_roles": [
                {"role": r["role"], **(r.get("blocked") or {})} for r in blocked
            ],
        },
        "verified_roles": [
            {"role": r["role"], **(r.get("verified_specimen") or {})} for r in ready
        ],
        "recovered_implementation": [
            "tools/future/odyssey_launch.py::propose_specimen_curriculum (moved here; re-exported)",
            "tools/future/specimen_verify.py — offline whole-tree recomputation",
            "tools/future/external_specimen_seal.py — authorized external tree digest",
            "ModelLake manifests under /Volumes/corpdrive/hawking-modellake/manifests",
            "receipts/future/evidence/HCLI_MODELLAKE_FLASH_CENSUS.json (census is a cache; disk is authority)",
        ],
        "gaps_closed": [
            "curriculum readiness had no module of its own; the gate inlined a proposal and refused it",
            "unready roles did not carry a wake condition, so the supervisor could not sleep on a named dependency",
            "disk manifests and specimen dirs outrank a stale census that omitted Mistral and denied Flash's listing",
        ],
        "negative_findings": [
            "a proposal is not a ready specimen set",
            "an unpublished or unverified specimen is not a ready role",
            "lowering n_roles to improve the ready ratio is refused",
            "inventing a specimen, or copying a hash that was not recomputed here, is refused",
            "ModelLake is never mutated; sidecar verification is the publication path",
        ],
        "resident_callable": {
            "entry_point": "tools.future.specimen_curriculum.propose_specimen_curriculum",
            "workunit": "one unit: evaluate five roles against ModelLake + SPECIMEN_VERIFICATION.json",
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.MODEL_CAPABILITY.hard-gates (specimen curriculum readiness)",
            "fails_closed": "unpublished or unverified is not ready; n_roles is a constant",
        },
    }
    path = write(RECEIPT, doc, RECORDED_BY)
    return {"path": str(path), "doc": doc, "curriculum": cur}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    out = build()
    cur = out["curriculum"]
    summary = {
        "n_roles": cur["n_roles"],
        "n_ready": cur["n_ready"],
        "ready": cur["ready"],
        "roles": [
            {
                "role": r["role"],
                "ready": r.get("ready"),
                "ready_reason": r.get("ready_reason"),
                "gap_class": r.get("gap_class"),
                "repo": r.get("repo"),
                "blocked": r.get("blocked"),
                "verified_specimen": (
                    {k: (r.get("verified_specimen") or {}).get(k) for k in
                     ("repo", "resolved_sha", "manifest_path", "n_files",
                      "n_sha256_verified", "n_size_only_verified", "bytes_hashed")}
                    if r.get("verified_specimen") else None
                ),
            }
            for r in cur["roles"]
        ],
    }
    print(json.dumps(summary, indent=1, sort_keys=True))
    print(out["path"])
    if a.dump:
        print(json.dumps(cur, indent=1, sort_keys=True, default=str)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
