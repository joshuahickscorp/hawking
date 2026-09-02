#!/usr/bin/env python3
"""Durable, incrementally-updatable ModelLake index.

Answering "what is in the lake" previously meant walking 4.35 TB. This module
stores a catalog OUTSIDE specimens/ and answers from it.

What already existed, and what this does not replace:
- tools.odyssey.modellake: acquire / atomic publish / retire / capacity / admit.
  status() still calls du(); this index is the query path that does not.
- tools.odyssey.modellake_lineage: watch-manifest registry, role_metadata,
  architecture_fingerprint, derive_lifecycle, storage_tier_for. This module
  CALLS those. registry_index() still does not scan the live volume.
- tools.future.specimen_registry: S027 lifecycle derived from a live walk.
  No durable store; every read re-lists the volume.
- tools.odyssey.model_specimen_seal: writes MODEL_LAKE_SPECIMEN_SEAL.json INTO
  the specimen tree. This module never calls it.

Index layout (never under specimens/):
    <lake>/index/catalog.json
    <lake>/index/by-slug/<slug>.json

A query reads one JSON file. An update after one new specimen walks that
specimen's tree and rewrites that slug's JSON plus the catalog totals.

    python3 tools/odyssey/modellake.py index
    python3 tools/odyssey/modellake.py query --slug Qwen--Qwen3-0.6B@c1899de289a0
    python3 tools/odyssey/modellake.py index-update --slug <slug>
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from tools.odyssey import modellake_lineage as lin

SCHEMA_CATALOG = "hawking.modellake.index.catalog.v1"
SCHEMA_SPECIMEN = "hawking.modellake.index.specimen.v1"
SCHEMA_RECEIPT = "hawking.modellake.index.receipt.v1"
EVIDENCE_TIER = "STATIC"
INDEX_DIRNAME = "index"
CATALOG_NAME = "catalog.json"
BY_SLUG_DIRNAME = "by-slug"

# Match tools.future.specimen_registry.WEIGHT_SUFFIXES — evo2 ships .pt, boltz
# .ckpt, mamba3/musicgen .bin, Wan .pth. Counting only .safetensors would
# mis-classify complete non-HF-shard specimens.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")

# Layout roles from H-ROADMAP.md §14.1. The lake already uses these directories;
# the index names them so a query does not have to infer from a walk.
LAYOUT = {
    "specimens": {
        "path": "specimens/",
        "storage_role": "TIER2_COLD",
        "meaning": lin.STORAGE_ROLES["TIER2_COLD"],
        "writable_by_index": False,
        "law": (
            "sealed verified source bodies; the index never writes, moves, "
            "deletes or compresses anything here"
        ),
    },
    "manifests": {
        "path": "manifests/",
        "storage_role": "GIT_METADATA",
        "meaning": (
            "acquisition manifests beside the bodies they describe; "
            "reacquisition recipe lives here. One per body: without a manifest "
            "modellake.retire() refuses to relegate a specimen, so a gap here is "
            "a body the lake cannot free. manifests/retired/ keeps the recipe for "
            "a body deliberately removed, and is not a live manifest."
        ),
        "writable_by_index": False,
    },
    "index": {
        "path": "index/",
        "storage_role": "GIT_METADATA",
        "meaning": (
            "durable incrementally-updatable catalog. Query this; "
            "do not walk 4.35 TB."
        ),
        "writable_by_index": True,
    },
    "logs": {
        "path": "logs/",
        "storage_role": "GIT_METADATA",
        "meaning": (
            "everything the lake writes about itself: acquisition worker logs, "
            "acquisition-state.json, and compressed archives of retired campaign logs"
        ),
        "writable_by_index": False,
    },
    # Transient: these exist only while an acquisition is in flight. The worker
    # mkdirs them and they are absent on an idle lake -- their absence is the
    # normal state, not a missing role.
    "partial": {
        "path": "partial/",
        "storage_role": "PARTIAL",
        "meaning": lin.STORAGE_ROLES["PARTIAL"],
        "writable_by_index": False,
        "transient": True,
        "law": "quarantined until verified and atomically renamed into specimens/",
    },
    "claims": {
        "path": "claims/",
        "storage_role": "GIT_METADATA",
        "meaning": "filler worker claims (pid/slug/taken_at); not a specimen body",
        "writable_by_index": False,
        "transient": True,
    },
}


class IndexError(ValueError):
    """Index path, slug, or lake layout is not usable."""


def _tier2_budget() -> int:
    """CALL the lake's own budget constant. Do not copy the number."""
    from tools.odyssey.modellake import TIER2_BUDGET
    return TIER2_BUDGET


def _default_lake() -> Path:
    from tools.odyssey.modellake import LAKE
    return Path(LAKE)


def _ssd_stage() -> Path:
    from tools.odyssey.modellake import SSD_STAGE
    return Path(SSD_STAGE)


def repo_index_dir() -> Path:
    """Git-backed fallback, still outside specimens/. Used when the volume refuses writes."""
    return lin._module_git_root() / "receipts" / "future" / "modellake-index"


def _can_hold_index(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def resolve_index_dir(
    lake: str | Path | None = None,
    index_dir: str | Path | None = None,
    *,
    create: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Prefer <lake>/index. If the volume is not writable, fall back to receipts/future.

    The catalog must live OUTSIDE specimens/. A sandbox or ACL that refuses
    /Volumes/corpdrive is not a reason to skip indexing the lake; it is a
    reason to store the catalog where this process can actually write.

    create=False (query/load) never mkdirs; it returns an existing catalog
    location, lake first, then the repo fallback.
    """
    lake_p = Path(lake) if lake else _default_lake()
    preferred = lake_p / INDEX_DIRNAME
    fallback = repo_index_dir()
    env = os.environ.get("HAWKING_MODELLAKE_INDEX")

    def meta(chosen: Path, source: str, **extra: Any) -> dict[str, Any]:
        out = {
            "source": source,
            "path": str(chosen),
            "preferred": str(preferred),
            **extra,
        }
        return out

    if index_dir:
        chosen = Path(index_dir)
        if create and not _can_hold_index(chosen):
            raise PermissionError(f"index dir is not writable: {chosen}")
        return chosen, meta(chosen, "explicit")
    if env:
        chosen = Path(env)
        if create and not _can_hold_index(chosen):
            raise PermissionError(f"index dir is not writable: {chosen}")
        return chosen, meta(chosen, "env:HAWKING_MODELLAKE_INDEX")

    lake_cat = preferred / CATALOG_NAME
    repo_cat = fallback / CATALOG_NAME
    if lake_cat.is_file():
        return preferred, meta(preferred, "lake", preferred_writable=True)
    if repo_cat.is_file() and not create:
        return fallback, meta(
            fallback, "repo_fallback", preferred_writable=False,
            reason=f"{preferred} has no catalog; using receipts/future/modellake-index",
        )
    if not create:
        return preferred, meta(preferred, "lake_missing", present=False)

    if _can_hold_index(preferred):
        return preferred, meta(preferred, "lake", preferred_writable=True)
    if _can_hold_index(fallback):
        return fallback, meta(
            fallback, "repo_fallback", preferred_writable=False,
            reason=(
                f"could not create {preferred}; storing the catalog outside "
                "specimens/ at receipts/future/modellake-index"
            ),
        )
    raise PermissionError(f"no writable index dir (tried {preferred} and {fallback})")


def default_index_dir(lake: str | Path | None = None) -> Path:
    chosen, _meta = resolve_index_dir(lake=lake, create=False)
    return chosen


def layout(*, lake: str | Path | None = None) -> dict[str, Any]:
    root = str(Path(lake) if lake else _default_lake())
    rows = {}
    for key, spec in LAYOUT.items():
        rows[key] = {**spec, "absolute": str(Path(root) / spec["path"])}
    return {
        "schema": "hawking.modellake.layout.v1",
        "evidence_tier": EVIDENCE_TIER,
        "roadmap": {"section": "14.1", "title": "Storage roles"},
        "lake_root": root,
        "roles": rows,
        "index_never_writes_specimens": True,
    }


def _safe_slug(slug: str) -> str:
    if not slug or slug.startswith(".") or "/" in slug or "\\" in slug or ".." in slug:
        raise IndexError(f"refusing slug: {slug!r}")
    return slug


def _specimens_root(lake: Path) -> Path:
    return (Path(lake) / "specimens").resolve()


def refuse_specimens_write(path: str | Path, lake: str | Path) -> Path:
    """Refuse any write whose resolved path is the sealed tree or under it."""
    path = Path(path).resolve()
    spec = _specimens_root(lake)
    try:
        path.relative_to(spec)
    except ValueError:
        return path
    raise PermissionError(f"refusing to write into specimens/: {path}")


def _atomic_write(path: Path, doc: dict[str, Any], lake: Path) -> int:
    refuse_specimens_write(path, lake)
    refuse_specimens_write(path.parent, lake)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    refuse_specimens_write(tmp, lake)
    blob = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    tmp.write_text(blob, encoding="utf-8")
    os.replace(tmp, path)
    return len(blob.encode("utf-8"))


def catalog_path(index_dir: Path) -> Path:
    return Path(index_dir) / CATALOG_NAME


def write_layout_readme(index_dir: Path, lake: Path) -> Path:
    """Render LAYOUT as prose beside the catalog.

    Someone who opens the volume in Finder should be able to tell what each
    directory is without parsing a 96 KB JSON. Generated from LAYOUT on every
    build so it cannot drift away from the roles the index actually enforces --
    a hand-written README at the lake root is exactly the file that goes stale.
    """
    lines = [
        "# ModelLake layout",
        "",
        f"Lake root: `{lake}`",
        "",
        "Generated by `tools/odyssey/modellake_index.py`. Do not hand-edit: it is",
        "rewritten from LAYOUT on every index build.",
        "",
    ]
    for name, role in LAYOUT.items():
        note = " *(transient: exists only during an acquisition)*" if role.get("transient") else ""
        lines.append(f"## `{role['path']}`{note}")
        lines.append("")
        lines.append(role["meaning"])
        if role.get("law"):
            lines.append("")
            lines.append(f"Law: {role['law']}")
        lines.append("")
    dest = Path(index_dir) / "LAYOUT.md"
    refuse_specimens_write(dest, lake)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def slug_path(index_dir: Path, slug: str) -> Path:
    return Path(index_dir) / BY_SLUG_DIRNAME / f"{_safe_slug(slug)}.json"


def walk_specimen_files(root: str | Path) -> dict[str, Any]:
    """Metadata walk of one specimen tree. Does not read file contents."""
    root = Path(root)
    files: list[dict[str, Any]] = []
    total = 0
    if not root.is_dir():
        return {"files": [], "bytes": 0, "n_files": 0, "dir_mtime_ns": None, "path": str(root)}
    try:
        dir_mtime_ns = root.stat().st_mtime_ns
    except OSError:
        dir_mtime_ns = None
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d != ".cache" and not d.startswith(".git")]
        for name in filenames:
            p = Path(dirpath) / name
            try:
                st = p.lstat()
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                continue
            rel = p.relative_to(root).as_posix()
            files.append({
                "path": rel,
                "bytes": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            })
            total += st.st_size
    files.sort(key=lambda r: r["path"])
    return {
        "files": files,
        "bytes": total,
        "n_files": len(files),
        "dir_mtime_ns": dir_mtime_ns,
        "path": str(root),
    }


def _is_weight(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1].lower()
    return name.endswith(WEIGHT_SUFFIXES)


def _shard_info(root: Path, files: list[dict[str, Any]]) -> dict[str, Any]:
    present = sum(1 for f in files if _is_weight(f["path"]))
    expected = None
    idx = root / "model.safetensors.index.json"
    if idx.is_file():
        doc = lin.load_json_file(idx)
        wm = (doc or {}).get("weight_map") or {}
        if isinstance(wm, dict) and wm:
            expected = len(set(wm.values()))
    complete = (
        (expected is not None and present == expected)
        or (expected is None and present > 0)
    )
    return {
        "present": present,
        "expected": expected,
        "complete": complete,
        "check_strength": "INDEX_MATCHED" if expected is not None else "WEIGHTS_PRESENT_ONLY",
        "n_shards": expected if expected is not None else present,
    }


def _normalize_lake_manifest(doc: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not doc:
        return None
    if doc.get("bytes") is not None and doc.get("repo"):
        return doc
    return {
        "repo": doc.get("repo"),
        "revision": doc.get("revision"),
        "resolved_sha": doc.get("resolved_sha"),
        "bytes": doc.get("expected") if doc.get("expected") is not None else doc.get("bytes"),
        "n_files": len(doc.get("files") or []) or doc.get("n_files"),
        "reacquisition": doc.get("reacquisition"),
        "acquired_at": doc.get("acquired_at"),
        "files": doc.get("files"),
        "sizes": doc.get("sizes"),
        "_shaped_from": "watch_or_claim_manifest",
    }


def _file_seals(
    files: list[dict[str, Any]],
    watch_sizes: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str]:
    sizes = {str(k): int(v) for k, v in (watch_sizes or {}).items()
             if isinstance(v, (int, float))}
    sealed_n = 0
    stale_n = 0
    out = []
    by_path = {f["path"]: f for f in files}
    declared = set(sizes)
    present = set(by_path)
    for f in files:
        exp = sizes.get(f["path"])
        match = (exp == f["bytes"]) if exp is not None else None
        if match is True:
            sealed_n += 1
        elif match is False:
            stale_n += 1
        out.append({
            "path": f["path"],
            "bytes": f["bytes"],
            "mtime_ns": f["mtime_ns"],
            "seal": {
                "kind": "watch_size" if exp is not None else "disk_stat",
                "watch_bytes": exp,
                "size_match": match,
            },
        })
    missing = sorted(declared - present)
    if missing:
        stale_n += len(missing)
        for rel in missing:
            out.append({
                "path": rel,
                "bytes": None,
                "mtime_ns": None,
                "seal": {
                    "kind": "watch_size",
                    "watch_bytes": sizes[rel],
                    "size_match": False,
                    "missing": True,
                },
            })
    if not sizes:
        status = "UNSEALED"
    elif stale_n:
        status = "STALE"
    elif declared <= present:
        status = "SEALED"
    else:
        status = "UNSEALED"
    out.sort(key=lambda r: r["path"])
    return out, status


def _identity_from_slug(slug: str) -> tuple[str, str | None]:
    repo_part, _, rev = slug.partition("@")
    return repo_part.replace("--", "/"), (rev or None)


def list_dir_slugs(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    names = []
    try:
        for entry in os.scandir(root):
            if entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."):
                names.append(entry.name)
    except OSError:
        return []
    return sorted(names)


def list_json_stems(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(p.stem for p in Path(root).glob("*.json") if p.is_file())


def list_watch_slugs(
    *,
    manifest_dir: str | Path | None = None,
    git_root: str | Path | None = None,
) -> list[str]:
    if manifest_dir and Path(manifest_dir).is_dir():
        return list_json_stems(Path(manifest_dir))
    root = Path(git_root) if git_root else lin._module_git_root()
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "ls-tree", "-r", "--name-only", "HEAD",
             "--", lin.WATCH_MANIFEST_REL],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if r.returncode != 0:
        return []
    slugs = []
    for line in r.stdout.splitlines():
        p = Path(line)
        # ls-tree -r is recursive. Only direct children are the live queue:
        # watch-manifests/retired/ holds entries deliberately taken out of it,
        # and counting those would report every retirement as an orphan forever.
        if p.parent != Path(lin.WATCH_MANIFEST_REL):
            continue
        if p.name.endswith(".json"):
            slugs.append(p.name[:-5])
    return sorted(slugs)


def _dir_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def scan_specimen(
    slug: str,
    body: Path,
    *,
    location: str,
    lake: Path,
    watch: Optional[dict[str, Any]] = None,
    lake_man: Optional[dict[str, Any]] = None,
    manifest_dir: str | Path | None = None,
    git_root: str | Path | None = None,
) -> dict[str, Any]:
    """Walk one specimen. CALLS lineage role/fingerprint/lifecycle/tier."""
    slug = _safe_slug(slug)
    walked = walk_specimen_files(body)
    watch = watch if watch is not None else lin.load_watch_manifest(
        slug, manifest_dir=manifest_dir, git_root=git_root,
    )
    if lake_man is None:
        lake_man = _normalize_lake_manifest(
            lin.load_json_file(Path(lake) / "manifests" / f"{slug}.json")
        )
    repo, rev_from_slug = _identity_from_slug(slug)
    repo = (watch or {}).get("repo") or (lake_man or {}).get("repo") or repo
    rev = (watch or {}).get("revision") or (lake_man or {}).get("revision") or rev_from_slug
    cfg = lin.load_json_file(body / "config.json") if body.is_dir() else None
    names = lin.tensor_names_from_specimen(body) if body.is_dir() else []
    fp = lin.architecture_fingerprint(cfg, names, repo=str(repo), rev=str(rev or ""))
    file_names = [f["path"] for f in walked["files"]]
    role = lin.role_metadata(cfg, file_names, repo=str(repo), slug=slug)
    shards = _shard_info(body, walked["files"])
    sealed_files, seal_status = _file_seals(walked["files"], (watch or {}).get("sizes"))
    if location == "partial":
        seal_status = "PARTIAL"
        source_dir = None
        partial_dir = body
    else:
        source_dir = body
        partial_dir = None
    # Only consult the live SSD stage when indexing the live lake. A tmp lake
    # must not inherit residency from ~/noetic/stage.
    staged_dir = None
    try:
        if Path(lake).resolve() == _default_lake().resolve():
            staged_candidate = _ssd_stage() / slug
            if staged_candidate.is_dir():
                staged_dir = staged_candidate
    except OSError:
        staged_dir = None
    life, why = lin.derive_lifecycle(
        watch=watch, lake_man=lake_man, source_dir=source_dir,
        partial_dir=partial_dir, staged_dir=staged_dir,
        fingerprinted=fp["strength"] == "ORGAN_FINGERPRINT",
        nr_present=False,
    )
    tier = lin.storage_tier_for(
        slug, config={}, source_dir=source_dir, staged_dir=staged_dir,
        partial_dir=partial_dir, watch_only=watch is not None and source_dir is None,
    )
    family, family_source = family_from_config(cfg, fp)
    return {
        "schema": SCHEMA_SPECIMEN,
        "evidence_tier": EVIDENCE_TIER,
        "slug": slug,
        "repo": repo,
        "revision": rev,
        "resolved_sha": (watch or {}).get("resolved_sha") or (lake_man or {}).get("resolved_sha"),
        "bytes": walked["bytes"],
        "n_files": walked["n_files"],
        "n_shards": shards["n_shards"],
        "shards": shards,
        "architecture_family": family,
        "architecture_family_source": family_source,
        "role": {
            "primary": role.get("primary"),
            "roles": role.get("roles"),
            "evidence_tier": role.get("evidence_tier"),
        },
        "architecture_fingerprint": {
            "model_type": fp.get("model_type"),
            "architectures": fp.get("architectures"),
            "hidden_size": fp.get("hidden_size"),
            "num_hidden_layers": fp.get("num_hidden_layers"),
            "num_attention_heads": fp.get("num_attention_heads"),
            "num_key_value_heads": fp.get("num_key_value_heads"),
            "vocab_size": fp.get("vocab_size"),
            "organs": fp.get("organs") or [],
            "strength": fp.get("strength"),
            "n_tensors": fp.get("n_tensors"),
            "loaded_weights": False,
        },
        "storage_role": tier.get("role"),
        "lifecycle": life,
        "lifecycle_derived_from": why,
        "seal_status": seal_status,
        "files": sealed_files,
        "dir_mtime_ns": walked["dir_mtime_ns"],
        "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "location": location,
        "path": str(body),
        "manifest_bytes": (lake_man or {}).get("bytes"),
        "watch_expected_bytes": (watch or {}).get("expected"),
        "has_watch_manifest": watch is not None,
        "has_lake_manifest": lake_man is not None,
        "loaded_weights": False,
        "wrote_specimen": False,
    }


def _summary_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "slug": record["slug"],
        "repo": record.get("repo"),
        "revision": record.get("revision"),
        "bytes": record.get("bytes"),
        "n_files": record.get("n_files"),
        "n_shards": record.get("n_shards"),
        "architecture_family": record.get("architecture_family"),
        "role_primary": (record.get("role") or {}).get("primary"),
        "roles": (record.get("role") or {}).get("roles"),
        "storage_role": record.get("storage_role"),
        "lifecycle": record.get("lifecycle"),
        "seal_status": record.get("seal_status"),
        "location": record.get("location"),
        "dir_mtime_ns": record.get("dir_mtime_ns"),
        "path": record.get("path"),
        "has_watch_manifest": record.get("has_watch_manifest"),
        "has_lake_manifest": record.get("has_lake_manifest"),
        "manifest_bytes": record.get("manifest_bytes"),
        "watch_expected_bytes": record.get("watch_expected_bytes"),
    }


def family_from_config(cfg: Optional[Mapping[str, Any]],
                       fp: Mapping[str, Any]) -> tuple[str, str]:
    """(architecture family, where it came from).

    `model_type` is the HuggingFace convention and stays first. It is not the
    only convention on this lake, and reading only it filed four bodies as
    UNKNOWN -- which the retention ranking then treated as four unique
    architecture classes rather than two families of two:

      evo2_40b / evo2_7b   config carries `architecture: ["StripedHyena2"]`,
                           singular, list-valued. 96 GB read as unique.
      mamba3 siso / mimo   architecture lives at `ssm_cfg.layer: "Mamba3"`,
                           the mamba-ssm convention.

    The source is returned with the family because a family read off
    `ssm_cfg.layer` is a weaker claim than one declared in `model_type`, and
    §14.2.4 deletes bytes on the strength of family membership. A body with no
    architecture evidence at all stays UNKNOWN: Wan2.2 ships only
    `configuration.json` with framework/task, boltz-2 ships no config, and
    inventing a family for either would be worse than admitting there is none.
    """
    cfg = cfg or {}
    if cfg.get("model_type"):
        return str(cfg["model_type"]), "config.model_type"
    if fp.get("model_type"):
        return str(fp["model_type"]), "fingerprint.model_type"

    def first(value: Any) -> Optional[str]:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (list, tuple)) and value:
            return first(value[0])
        return None

    for key, source in (("architectures", "config.architectures"),
                        ("architecture", "config.architecture")):
        name = first(cfg.get(key))
        if name:
            return name.lower(), source
    ssm = cfg.get("ssm_cfg")
    if isinstance(ssm, Mapping):
        name = first(ssm.get("layer"))
        if name:
            return name.lower(), "config.ssm_cfg.layer"
    return "UNKNOWN", "no architecture evidence in the body"


def _anomalies(
    rows: list[dict[str, Any]],
    *,
    lake: Path,
    watch_slugs: Iterable[str],
) -> dict[str, Any]:
    bodies = {r["slug"] for r in rows if r.get("location") == "specimens"}
    partials = {r["slug"] for r in rows if r.get("location") == "partial"}
    watch = set(watch_slugs)
    manifests = set(list_json_stems(Path(lake) / "manifests"))
    claims = set(list_json_stems(Path(lake) / "claims"))
    stale = [r["slug"] for r in rows if r.get("seal_status") == "STALE"]
    stale_manifest_bytes = []
    for r in rows:
        mb = r.get("manifest_bytes")
        disk = r.get("bytes")
        if mb is not None and disk is not None and int(mb) != int(disk):
            stale_manifest_bytes.append({
                "slug": r["slug"],
                "manifest_bytes": mb,
                "disk_bytes": disk,
                "delta": int(disk) - int(mb),
            })
    return {
        "partial": sorted(partials),
        "orphaned_bodies": sorted(bodies - watch - manifests),
        "orphaned_manifests": sorted(manifests - bodies - partials),
        "orphaned_watch": sorted(watch - bodies - partials),
        "orphaned_claims": sorted(claims - bodies - partials),
        "stale_seals": stale,
        "stale_manifest_bytes": stale_manifest_bytes,
        "stale_manifest_bytes_note": (
            "manifest `bytes` and the catalog now use one definition -- the sum of "
            "st_size over regular files outside .cache -- so any entry here is a "
            "real disagreement between a manifest and the body it describes, not "
            "the du()-vs-st_size difference this list used to be full of. "
            "acquire() keeps the allocated figure separately as bytes_allocated."
        ),
        "bodies_without_lake_manifest": sorted(bodies - manifests),
        "n_bodies": len(bodies),
        "n_watch": len(watch),
        "n_lake_manifests": len(manifests),
        "n_claims": len(claims),
        "n_partial": len(partials),
    }


def retention_recommendation(
    rows: list[dict[str, Any]],
    *,
    budget: int,
    used: int,
) -> dict[str, Any]:
    """Rank specimens a future retention decision could consider.

    Does not retire, delete, move or compress anything. §14.2.3: if headroom
    is unsafe, stop acquisition; do not delete canonical sources automatically.
    §14.2.4: prefer architecture diversity over redundant bulk.
    """
    overage = used - budget
    fam_n = Counter(r.get("architecture_family") or "UNKNOWN" for r in rows)
    role_n = Counter((r.get("role_primary") or (r.get("role") or {}).get("primary") or "unknown")
                     for r in rows)
    ranked = []
    for r in rows:
        if r.get("location") != "specimens":
            continue
        fam = r.get("architecture_family") or "UNKNOWN"
        primary = r.get("role_primary") or (r.get("role") or {}).get("primary") or "unknown"
        n_fam = fam_n[fam]
        # UNKNOWN is a gap label, not a family. Clustering those bodies as
        # redundant bulk would treat Wan, evo2, Hunyuan and mamba3 as substitutes.
        unique_family = n_fam == 1 or fam == "UNKNOWN"
        redundancy = 0.0 if unique_family else 1.0 - (1.0 / n_fam)
        bytes_ = int(r.get("bytes") or 0)
        ranked.append({
            "slug": r["slug"],
            "bytes": bytes_,
            "architecture_family": fam,
            "family_n": n_fam,
            "primary_role": primary,
            "role_n": role_n[primary],
            "unique_family": unique_family,
            "redundancy": round(redundancy, 6),
            "score": int(bytes_ * redundancy),
            "why": (
                f"unique {fam} family — retiring this drops a diversity class"
                if unique_family else
                f"{fam} has {n_fam} residents; this is redundant bulk "
                f"under §14.2.4 ({bytes_} bytes)"
            ),
        })
    ranked.sort(key=lambda x: (-x["score"], -x["bytes"], x["slug"]))
    redundant = [x for x in ranked if not x["unique_family"]]
    unique = [x for x in ranked if x["unique_family"]]
    # Keep one (the smallest) of each multi-member family so a covering set
    # does not drop a diversity class. Surplus = the rest.
    by_fam: dict[str, list[dict[str, Any]]] = {}
    for x in redundant:
        by_fam.setdefault(x["architecture_family"], []).append(x)
    kept_as_family_rep = []
    surplus: list[dict[str, Any]] = []
    for fam, members in by_fam.items():
        members = sorted(members, key=lambda m: (m["bytes"], m["slug"]))
        keep = dict(members[0])
        keep["why"] = (
            f"keep as the {fam} representative ({keep['bytes']} bytes); "
            "not a retention candidate under §14.2.4"
        )
        kept_as_family_rep.append(keep)
        for m in members[1:]:
            row = dict(m)
            row["kept_sibling"] = keep["slug"]
            row["why"] = (
                f"{fam} already represented by {keep['slug']} "
                f"({keep['bytes']} bytes); this is surplus bulk"
            )
            surplus.append(row)
    surplus.sort(key=lambda x: (-x["bytes"], x["slug"]))
    covering: list[dict[str, Any]] = []
    acc = 0
    if overage > 0:
        for x in surplus:
            covering.append(x)
            acc += x["bytes"]
            if acc >= overage:
                break
    unique_bytes = sum(x["bytes"] for x in unique)
    redundant_bytes = sum(x["bytes"] for x in redundant)
    surplus_bytes = sum(x["bytes"] for x in surplus)
    covers = acc >= overage if overage > 0 else True
    return {
        "operator_decision_only": True,
        "does_not_retire": True,
        "roadmap": {
            "14.2.3": "If headroom becomes unsafe, stop new acquisition; do not delete canonical sources automatically.",
            "14.2.4": "Prefer architecture diversity over redundant bulk.",
            "14.2.5": "Every specimen needs a reason: what law can it teach or falsify?",
        },
        "tier2_budget": budget,
        "tier2_used_bytes": used,
        "tier2_overage_bytes": max(0, overage),
        "tier2_used_minus_budget": overage,
        "redundant_bulk_bytes": redundant_bytes,
        "surplus_bulk_bytes": surplus_bytes,
        "unique_family_bytes": unique_bytes,
        "redundant_bulk_covers_overage": covers,
        "smallest_redundant_set_that_covers_overage": covering if overage > 0 else [],
        "covering_set_bytes": acc,
        "kept_as_family_representative": kept_as_family_rep,
        "ranked_surplus_bulk": surplus,
        "ranked_redundant_bulk": redundant,
        "unique_families_not_recommended": unique,
        "reading": (
            "Redundant bulk (same architecture family as another resident) is "
            "the only class this ranking treats as a plausible retention "
            "candidate. Unique families are listed so the operator can see "
            "the cost of dropping a diversity class; they are not recommended."
            + (
                "" if covers or overage <= 0 else
                f" Surplus bulk (duplicates beyond one kept representative "
                f"per family) totalling {surplus_bytes} bytes does not cover "
                f"the {overage} byte overage; closing the gap would require "
                "considering unique-family specimens, which §14.2.4 weighs "
                "against."
            )
        ),
    }


def load_catalog(
    *,
    index_dir: str | Path | None = None,
    lake: str | Path | None = None,
) -> Optional[dict[str, Any]]:
    lake_p = Path(lake) if lake else _default_lake()
    idx, _meta = resolve_index_dir(lake=lake_p, index_dir=index_dir, create=False)
    path = catalog_path(idx)
    if not path.is_file():
        return None
    doc = lin.load_json_file(path)
    return doc if isinstance(doc, dict) else None


def load_specimen_record(
    slug: str,
    *,
    index_dir: str | Path | None = None,
    lake: str | Path | None = None,
) -> Optional[dict[str, Any]]:
    lake_p = Path(lake) if lake else _default_lake()
    idx, _meta = resolve_index_dir(lake=lake_p, index_dir=index_dir, create=False)
    path = slug_path(idx, slug)
    if not path.is_file():
        return None
    doc = lin.load_json_file(path)
    return doc if isinstance(doc, dict) else None


def query_specimen(
    slug: str,
    *,
    index_dir: str | Path | None = None,
    lake: str | Path | None = None,
) -> dict[str, Any]:
    """Read one per-slug JSON. Does not walk the lake."""
    rec = load_specimen_record(slug, index_dir=index_dir, lake=lake)
    if rec is None:
        raise IndexError(f"no index record for {slug!r}; run index or index-update")
    return rec


def _empty_catalog(lake: Path, index_dir: Path, budget: int) -> dict[str, Any]:
    return {
        "schema": SCHEMA_CATALOG,
        "evidence_tier": EVIDENCE_TIER,
        "roadmap": {
            "section": "14",
            "title": "MODELLAKE — SPECIMEN SCHOOL AND STORAGE LIFECYCLE",
        },
        "lake_root": str(lake),
        "index_dir": str(index_dir),
        "layout": layout(lake=lake),
        "tier2_budget": budget,
        "tier2_used_bytes": 0,
        "tier2_overage_bytes": 0,
        "tier2_used_minus_budget": -budget,
        "over_budget": False,
        "bytes_are": (
            "sum of st_size over regular files in specimens/ and partial/; "
            "directory metadata is not included. This is not `du` allocated blocks."
        ),
        "n_specimens": 0,
        "specimens": [],
        "scanned_slugs": [],
        "skipped_slugs": [],
        "anomalies": {},
        "retention_recommendation": {},
        "built_at": None,
        "loaded_weights": False,
        "wrote_specimens": False,
    }


def _finalize_catalog(
    catalog: dict[str, Any],
    *,
    lake: Path,
    watch_slugs: list[str],
    budget: int,
    scanned: list[str],
    skipped: list[str],
    written_paths: list[str],
    written_bytes: int,
) -> dict[str, Any]:
    rows = list(catalog.get("specimens") or [])
    rows.sort(key=lambda r: r["slug"])
    spec_bytes = sum(int(r.get("bytes") or 0) for r in rows if r.get("location") == "specimens")
    part_bytes = sum(int(r.get("bytes") or 0) for r in rows if r.get("location") == "partial")
    used = spec_bytes + part_bytes
    overage = used - budget
    catalog["specimens"] = rows
    catalog["n_specimens"] = sum(1 for r in rows if r.get("location") == "specimens")
    catalog["n_partial"] = sum(1 for r in rows if r.get("location") == "partial")
    catalog["specimens_bytes"] = spec_bytes
    catalog["partial_bytes"] = part_bytes
    catalog["tier2_budget"] = budget
    catalog["tier2_used_bytes"] = used
    catalog["tier2_overage_bytes"] = max(0, overage)
    catalog["tier2_used_minus_budget"] = overage
    catalog["over_budget"] = used > budget
    catalog["tier2_used_gb"] = round(used / 1e9, 3)
    catalog["tier2_budget_gb"] = round(budget / 1e9, 3)
    catalog["tier2_overage_gb"] = round(max(0, overage) / 1e9, 3)
    catalog["anomalies"] = _anomalies(rows, lake=lake, watch_slugs=watch_slugs)
    catalog["retention_recommendation"] = retention_recommendation(
        rows, budget=budget, used=used,
    )
    catalog["scanned_slugs"] = scanned
    catalog["skipped_slugs"] = skipped
    catalog["n_scanned"] = len(scanned)
    catalog["n_skipped"] = len(skipped)
    catalog["written_index_paths"] = written_paths
    catalog["written_index_bytes"] = written_bytes
    catalog["wrote_specimens"] = False
    catalog["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    families: dict[str, list[str]] = {}
    for r in rows:
        families.setdefault(r.get("architecture_family") or "UNKNOWN", []).append(r["slug"])
    catalog["architecture_families"] = {k: sorted(v) for k, v in sorted(families.items())}
    catalog["n_families"] = len(families)
    return catalog


def _upsert_row(catalog: dict[str, Any], summary: dict[str, Any]) -> None:
    rows = [r for r in (catalog.get("specimens") or []) if r.get("slug") != summary["slug"]]
    rows.append(summary)
    catalog["specimens"] = rows


def _body_for(slug: str, lake: Path, *, body: Path | None = None) -> tuple[Path, str]:
    if body is not None:
        p = Path(body)
        loc = "partial" if p.parent.name == "partial" else "specimens"
        return p, loc
    spec = Path(lake) / "specimens" / slug
    part = Path(lake) / "partial" / slug
    if spec.is_dir():
        return spec, "specimens"
    if part.is_dir():
        return part, "partial"
    raise IndexError(f"no body for {slug!r} under {lake}")


def _unchanged(existing: Optional[dict[str, Any]], slug: str, body: Path) -> bool:
    if not existing:
        return False
    row = next((r for r in (existing.get("specimens") or []) if r.get("slug") == slug), None)
    if not row:
        return False
    mtime = _dir_mtime_ns(body)
    return mtime is not None and row.get("dir_mtime_ns") == mtime


def update_specimen(
    slug: str,
    *,
    lake: str | Path | None = None,
    index_dir: str | Path | None = None,
    body: str | Path | None = None,
    manifest_dir: str | Path | None = None,
    git_root: str | Path | None = None,
    budget: int | None = None,
) -> dict[str, Any]:
    """Rescan one specimen. Does not walk the rest of the lake."""
    lake_p = Path(lake) if lake else _default_lake()
    idx, placement = resolve_index_dir(lake=lake_p, index_dir=index_dir, create=True)
    refuse_specimens_write(idx, lake_p)
    slug = _safe_slug(slug)
    body_p, location = _body_for(slug, lake_p, body=Path(body) if body else None)
    budget_n = _tier2_budget() if budget is None else int(budget)
    catalog = load_catalog(index_dir=idx, lake=lake_p) or _empty_catalog(lake_p, idx, budget_n)
    catalog["index_placement"] = placement
    record = scan_specimen(
        slug, body_p, location=location, lake=lake_p,
        manifest_dir=manifest_dir, git_root=git_root,
    )
    written = []
    n_bytes = 0
    n_bytes += _atomic_write(slug_path(idx, slug), record, lake_p)
    written.append(str(slug_path(idx, slug)))
    _upsert_row(catalog, _summary_row(record))
    watch_slugs = list_watch_slugs(manifest_dir=manifest_dir, git_root=git_root)
    catalog = _finalize_catalog(
        catalog, lake=lake_p, watch_slugs=watch_slugs, budget=budget_n,
        scanned=[slug], skipped=[], written_paths=written, written_bytes=n_bytes,
    )
    n_bytes += _atomic_write(catalog_path(idx), catalog, lake_p)
    written.append(str(catalog_path(idx)))
    written.append(str(write_layout_readme(idx, lake_p)))
    catalog["written_index_paths"] = written
    catalog["written_index_bytes"] = n_bytes
    return {
        "updated": True,
        "slug": slug,
        "scanned_slugs": [slug],
        "n_scanned": 1,
        "skipped_slugs": [],
        "written_index_paths": written,
        "record": record,
        "catalog": catalog,
        "walked_path": str(body_p),
        "wrote_specimens": False,
    }


def build(
    *,
    lake: str | Path | None = None,
    index_dir: str | Path | None = None,
    force: bool = False,
    manifest_dir: str | Path | None = None,
    git_root: str | Path | None = None,
    budget: int | None = None,
) -> dict[str, Any]:
    """Build or refresh the catalog. Unchanged specimen dirs are not re-walked."""
    t0 = time.perf_counter()
    lake_p = Path(lake) if lake else _default_lake()
    idx, placement = resolve_index_dir(lake=lake_p, index_dir=index_dir, create=True)
    refuse_specimens_write(idx, lake_p)
    budget_n = _tier2_budget() if budget is None else int(budget)
    existing = None if force else load_catalog(index_dir=idx, lake=lake_p)
    catalog = _empty_catalog(lake_p, idx, budget_n)
    catalog["index_placement"] = placement
    if existing and existing.get("schema") == SCHEMA_CATALOG and not force:
        # Keep skipped records as-is; scanned ones replace them.
        catalog["specimens"] = list(existing.get("specimens") or [])
    spec_slugs = list_dir_slugs(lake_p / "specimens")
    part_slugs = list_dir_slugs(lake_p / "partial")
    live = [(s, lake_p / "specimens" / s, "specimens") for s in spec_slugs]
    live += [(s, lake_p / "partial" / s, "partial") for s in part_slugs if s not in spec_slugs]
    live_slugs = {s for s, _, _ in live}
    scanned: list[str] = []
    skipped: list[str] = []
    written: list[str] = []
    n_bytes = 0
    watch_slugs = list_watch_slugs(manifest_dir=manifest_dir, git_root=git_root)
    for slug, body, location in live:
        if not force and _unchanged(existing, slug, body):
            skipped.append(slug)
            rec = load_specimen_record(slug, index_dir=idx, lake=lake_p)
            if rec:
                _upsert_row(catalog, _summary_row(rec))
            continue
        record = scan_specimen(
            slug, body, location=location, lake=lake_p,
            manifest_dir=manifest_dir, git_root=git_root,
        )
        n_bytes += _atomic_write(slug_path(idx, slug), record, lake_p)
        written.append(str(slug_path(idx, slug)))
        _upsert_row(catalog, _summary_row(record))
        scanned.append(slug)
    dropped = [r["slug"] for r in list(catalog.get("specimens") or [])
               if r["slug"] not in live_slugs]
    if dropped:
        catalog["specimens"] = [r for r in catalog["specimens"] if r["slug"] in live_slugs]
        for slug in dropped:
            p = slug_path(idx, slug)
            if p.is_file():
                refuse_specimens_write(p, lake_p)
                p.unlink()
                written.append(str(p) + "#unlinked")
    catalog = _finalize_catalog(
        catalog, lake=lake_p, watch_slugs=watch_slugs, budget=budget_n,
        scanned=scanned, skipped=skipped, written_paths=written, written_bytes=n_bytes,
    )
    catalog["dropped_slugs"] = dropped
    catalog["force"] = force
    catalog["build_wall_s"] = round(time.perf_counter() - t0, 6)
    n_bytes += _atomic_write(catalog_path(idx), catalog, lake_p)
    written.append(str(catalog_path(idx)))
    written.append(str(write_layout_readme(idx, lake_p)))
    catalog["written_index_paths"] = written
    catalog["written_index_bytes"] = n_bytes
    _atomic_write(catalog_path(idx), catalog, lake_p)
    return catalog


def measure_query(
    slug: str,
    *,
    index_dir: str | Path | None = None,
    lake: str | Path | None = None,
    n: int = 21,
) -> dict[str, Any]:
    """Host wall-clock of query_specimen. HARDWARE_MEASURED, not a GPU claim."""
    # Warm once so the number is the query, not first-import.
    query_specimen(slug, index_dir=index_dir, lake=lake)
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        query_specimen(slug, index_dir=index_dir, lake=lake)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    median = samples[len(samples) // 2]
    return {
        "slug": slug,
        "n": n,
        "min_ms": round(samples[0], 4),
        "median_ms": round(median, 4),
        "max_ms": round(samples[-1], 4),
        "samples_ms": [round(x, 4) for x in samples],
        "threshold_ms": 50,
        "pass": median < 50,
        "evidence_tier": "HARDWARE_MEASURED",
        "machine": "Apple M3 Ultra",
        "what": "wall-clock of query_specimen reading one by-slug JSON",
        "not_measured": ["FPGA/U50", "DGX", "eGPU", "ANE"],
    }


def _max_mtime(root: Path) -> float:
    newest = 0.0
    if not root.is_dir():
        return newest
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d != ".cache"]
        for name in filenames:
            p = Path(dirpath) / name
            try:
                newest = max(newest, p.lstat().st_mtime)
            except OSError:
                continue
    try:
        newest = max(newest, root.stat().st_mtime)
    except OSError:
        pass
    return newest


def measure_and_receipt(
    *,
    lake: str | Path | None = None,
    index_dir: str | Path | None = None,
    receipt_path: str | Path | None = None,
    probe_root: str | Path | None = None,
    query_slug: str | None = None,
) -> dict[str, Any]:
    """Build over the real lake, measure query + incremental, write a receipt.

    The simulated new specimen is created under probe_root, never under
    specimens/. Its index record is written to a COPY of the index, so the
    durable lake index is not polluted with a probe slug.
    """
    lake_p = Path(lake) if lake else _default_lake()
    spec_root = lake_p / "specimens"
    mtime_before = _max_mtime(spec_root)
    t0 = time.perf_counter()
    catalog = build(lake=lake_p, index_dir=index_dir, force=True)
    idx = Path(catalog["index_dir"])
    placement = catalog.get("index_placement") or {}
    build_s = time.perf_counter() - t0
    slugs = [r["slug"] for r in catalog.get("specimens") or [] if r.get("location") == "specimens"]
    if not slugs:
        raise IndexError(f"no specimens under {spec_root}")
    qslug = query_slug or slugs[0]
    q = measure_query(qslug, index_dir=idx, lake=lake_p)
    # Incremental against a copy of the just-built index.
    import shutil
    import tempfile
    probe_parent = Path(probe_root) if probe_root else Path(tempfile.mkdtemp(prefix="modellake-index-probe-"))
    probe_parent.mkdir(parents=True, exist_ok=True)
    idx_copy = probe_parent / "index-copy"
    if idx_copy.exists():
        shutil.rmtree(idx_copy)
    shutil.copytree(idx, idx_copy)
    probe_slug = "probe--index@ffffffffffffffff"
    probe_body = probe_parent / probe_slug
    probe_body.mkdir(parents=True, exist_ok=True)
    (probe_body / "config.json").write_text(json.dumps({
        "model_type": "probe",
        "architectures": ["ProbeForCausalLM"],
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "max_position_embeddings": 128,
    }), encoding="utf-8")
    (probe_body / "README.md").write_text("index probe; not a specimen\n", encoding="utf-8")
    copy_mtimes = {
        p.name: p.stat().st_mtime_ns
        for p in (idx_copy / BY_SLUG_DIRNAME).glob("*.json")
    }
    t1 = time.perf_counter()
    upd = update_specimen(
        probe_slug, lake=lake_p, index_dir=idx_copy, body=probe_body,
    )
    upd_s = time.perf_counter() - t1
    after_mtimes = {
        p.name: p.stat().st_mtime_ns
        for p in (idx_copy / BY_SLUG_DIRNAME).glob("*.json")
    }
    changed = sorted(
        name for name, ns in after_mtimes.items()
        if copy_mtimes.get(name) != ns
    )
    others_unchanged = all(
        after_mtimes.get(name) == ns
        for name, ns in copy_mtimes.items()
    )
    mtime_after = _max_mtime(spec_root)
    repo = Path(__file__).resolve().parents[2]
    receipt = {
        "schema": SCHEMA_RECEIPT,
        "produced_by": "tools/odyssey/modellake_index.py",
        "produced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": "python3 -m tools.odyssey.modellake_index measure",
        "gpu_authority": False,
        "absent_hardware_not_measured": ["FPGA/U50", "DGX", "eGPU"],
        "machine": "Apple M3 Ultra",
        "claim_boundary": (
            "Host wall-clock of index build/query/update is HARDWARE_MEASURED "
            "on this Apple M3 Ultra. Catalog contents (roles, families, sizes) "
            "are STATIC. No GPU, ANE, FPGA, DGX or eGPU measurement."
        ),
        "lake_root": str(lake_p),
        "index_dir": str(idx),
        "index_placement": placement,
        "lake_is_not_a_git_tree": not (lake_p / ".git").exists(),
        "n_specimens": catalog.get("n_specimens"),
        "n_partial": catalog.get("n_partial"),
        "n_families": catalog.get("n_families"),
        "tier2_budget": catalog.get("tier2_budget"),
        "tier2_used_bytes": catalog.get("tier2_used_bytes"),
        "tier2_overage_bytes": catalog.get("tier2_overage_bytes"),
        "tier2_used_minus_budget": catalog.get("tier2_used_minus_budget"),
        "tier2_used_gb": catalog.get("tier2_used_gb"),
        "tier2_overage_gb": catalog.get("tier2_overage_gb"),
        "over_budget": catalog.get("over_budget"),
        "bytes_are": catalog.get("bytes_are"),
        "build": {
            "wall_s": round(build_s, 6),
            "n_scanned": catalog.get("n_scanned"),
            "n_skipped": catalog.get("n_skipped"),
            "evidence_tier": "HARDWARE_MEASURED",
        },
        "query_ms": q,
        "incremental": {
            "simulated_slug": probe_slug,
            "simulated_body": str(probe_body),
            "index_copy": str(idx_copy),
            "scanned_slugs": upd["scanned_slugs"],
            "n_scanned": upd["n_scanned"],
            "walked_path": upd["walked_path"],
            "other_by_slug_json_mtimes_unchanged": others_unchanged,
            "changed_by_slug_files": changed,
            "n_other_records_at_update": len(copy_mtimes),
            "update_wall_s": round(upd_s, 6),
            "full_build_wall_s": round(build_s, 6),
            "proportional": (
                upd["scanned_slugs"] == [probe_slug]
                and others_unchanged
                and changed == [f"{probe_slug}.json"]
            ),
            "wrote_specimens": False,
            "evidence_tier": "HARDWARE_MEASURED",
        },
        "specimens_dir": {
            "path": str(spec_root),
            "max_mtime_before": mtime_before,
            "max_mtime_after": mtime_after,
            "unchanged": mtime_after == mtime_before,
            "bytes_written_under_specimens": 0 if mtime_after == mtime_before else "UNKNOWN_MTIME_DRIFT",
        },
        "anomalies": catalog.get("anomalies"),
        "retention_recommendation": catalog.get("retention_recommendation"),
        "layout": catalog.get("layout"),
        "architecture_families": {
            k: len(v) for k, v in (catalog.get("architecture_families") or {}).items()
        },
        "specimens": catalog.get("specimens"),
        "acceptance": {
            "query_median_ms_under_50": q["pass"],
            "incremental_proportional": True,  # filled below
            "overage_reported": catalog.get("tier2_overage_bytes") is not None,
            "zero_bytes_under_specimens": mtime_after == mtime_before,
        },
    }
    receipt["acceptance"]["incremental_proportional"] = receipt["incremental"]["proportional"]
    dest = Path(receipt_path) if receipt_path else repo / "receipts" / "future" / "MODELLAKE_INDEX.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_path"] = str(dest)
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("cmd", nargs="?", default="build",
                    choices=["build", "query", "update", "measure", "layout"])
    ap.add_argument("--slug")
    ap.add_argument("--lake")
    ap.add_argument("--index-dir")
    ap.add_argument("--manifest-dir")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--emit")
    ap.add_argument("--body")
    a = ap.parse_args(argv)
    if a.cmd == "layout":
        out = layout(lake=a.lake)
    elif a.cmd == "measure":
        out = measure_and_receipt(lake=a.lake, index_dir=a.index_dir)
    elif a.cmd == "query":
        if a.slug:
            out = query_specimen(a.slug, index_dir=a.index_dir, lake=a.lake)
        else:
            out = load_catalog(index_dir=a.index_dir, lake=a.lake) or {
                "present": False, "error": "index missing; run build",
            }
    elif a.cmd == "update":
        if not a.slug:
            print("update requires --slug", flush=True)
            return 2
        out = update_specimen(
            a.slug, lake=a.lake, index_dir=a.index_dir, body=a.body,
            manifest_dir=a.manifest_dir,
        )
    else:
        out = build(
            lake=a.lake, index_dir=a.index_dir, force=a.force,
            manifest_dir=a.manifest_dir,
        )
    text = json.dumps(out, indent=1, sort_keys=True)
    print(text)
    if a.emit:
        Path(a.emit).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
