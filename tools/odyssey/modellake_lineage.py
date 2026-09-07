"""Roadmap-layer ModelLake lineage: registry, provenance, roles, fingerprints.

H-ROADMAP.md §14 (specimen school + storage lifecycle) names a lineage the
lake did not previously expose as one object. Acquisition, promotion, the
S027 disk registry (tools.future.specimen_registry) and the architecture
recognizer already exist; this module joins them.

A capability does not exist until something CALLS it. express_lineage()
calls load_watch_manifest, role_metadata, architecture_fingerprint (which
calls arch_recognizer.recognize), artifact_lineage and storage_tier_for.

The live-volume catalog is tools.odyssey.modellake_index. build_lake_index,
query_lake_specimen, update_lake_specimen and lake_index CALL that module;
registry_index still does not scan the volume.

Does not write /Volumes/corpdrive from lineage itself. Does not restart an
acquisition worker. The index writes only under <lake>/index/, never under
specimens/.

    python3 tools/odyssey/modellake.py lineage --slug Qwen--Qwen3-0.6B@c1899de289a0
    python3 tools/odyssey/modellake.py index
    python3 tools/odyssey/modellake.py query --slug Qwen--Qwen3-0.6B@c1899de289a0
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

# Named real specimen: watch-manifest in git AND sealed body on the live lake.
CANONICAL_SPECIMEN = "Qwen--Qwen3-0.6B@c1899de289a0"
CANONICAL_REPO = "Qwen/Qwen3-0.6B"
CANONICAL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
WATCH_MANIFEST_REL = "workspace/campaign/odyssey/watch-manifests"

# H-ROADMAP.md §14. Distinct from tools.future.specimen_registry.LIFECYCLE (S027).
ROADMAP_LIFECYCLE = (
    "DISCOVERED",
    "IDENTITY_RESOLVED",
    "MANIFEST_READY",
    "DOWNLOADING",
    "VERIFYING",
    "READY_COLD",
    "CENSUSED",
    "TRANSFER_READY",
    "SSD_STAGED",
    "ACTIVE",
    "RETIRED",
)

ARTIFACT_STAGES = ("SOURCE", "NR", "NX")

# §14.1 storage roles.
STORAGE_ROLES = {
    "TIER2_COLD": "External HDD: cold/rolling specimen school and verified source bodies.",
    "TIER1_HOT": "Fast local SSD/UMA: active hot artifacts and current executable candidates.",
    "GIT_METADATA": "Git repo: code, manifests, receipts, small metadata — not multi-hundred-GB models.",
    "PARTIAL": "Partials: quarantined lifecycle objects until verified and atomically promoted.",
}

# §14.3 candidate specimen diversity. A specimen's role is one of these.
DIVERSITY_ROLES = (
    "dense decoder",
    "MoE",
    "hybrid recurrent/state-space",
    "long-context",
    "multimodal",
    "extreme expert count",
    "alternative tokenizer",
    "native MTP/speculation",
    "very-low-bit published checkpoint",
    "structured sparsity",
    "codebook/additive quantization",
    "state-heavy architecture",
    "new Apple-friendly runtime specimen",
)

EVIDENCE_TIER = "STATIC"
HEADER_CAP = 32 * 1024 * 1024

# Execution-class SIZE tiering. Distinct axis from STORAGE_ROLES above:
# that is *where* a specimen sits (SSD vs HDD); this is *how big* it is on
# disk, independent of location. Thresholds are on-disk bytes, GiB.
GIB = 1024 ** 3
SIZE_TIERS = ("A_TINY", "B_MID", "C_LARGE", "D_GIANT")
_SIZE_TIER_MAX_BYTES = {"A_TINY": 8 * GIB, "B_MID": 40 * GIB, "C_LARGE": 80 * GIB}

# Operator decision 2026-09-05, encoded not re-litigated: the top 3 lake
# specimens by on-disk size are DEFERRED from execution-class Odyssey work.
# In the operator's words -- they scale a lot, deferring them buys speed,
# and by the time the program reaches them a faster/lighter/smarter resident
# should be ready to tackle them. Keyed by HF repo id (revision-agnostic: a
# re-acquired revision of the same repo is still the same giant).
DEFERRED_GIANTS = {
    "moonshotai/Kimi-K3": (
        "operator 2026-09-05: largest lake specimen (~1.45 TB); deferred to "
        "buy execution-class speed now, revisit once a lighter/faster/smarter "
        "resident exists"
    ),
    "thinkingmachines/Inkling-Small": (
        "operator 2026-09-05: 2nd-largest lake specimen (~496 GB); same "
        "deferral rationale as Kimi-K3"
    ),
    "windowsxp811203/Qwen3.8-Flash-Next-Abliterated": (
        "operator 2026-09-05: 3rd-largest lake specimen (~336 GB); same "
        "deferral rationale as Kimi-K3"
    ),
}
# Deferred does NOT mean dropped: they stay in the static census and remain
# eligible for a later pass.
DEFERRED_NOTE = (
    "deferred does not mean dropped: these specimens stay in the static "
    "census and remain eligible for a later pass"
)


def size_tier_for(nbytes: int) -> str:
    """A_TINY / B_MID / C_LARGE / D_GIANT from on-disk bytes."""
    for name in SIZE_TIERS[:-1]:
        if nbytes <= _SIZE_TIER_MAX_BYTES[name]:
            return name
    return "D_GIANT"


def deferred_reason_for(repo: str) -> str | None:
    return DEFERRED_GIANTS.get(repo)


def execution_eligible_for(repo: str, nbytes: int | None = None) -> bool:
    """False only for the named DEFERRED_GIANTS; True otherwise.

    nbytes is accepted for symmetry with size_tier_for (a future caller may
    want to gate on tier); today eligibility is decided by the named set,
    not by tier alone -- other D_GIANT specimens are not automatically
    excluded.
    """
    return repo not in DEFERRED_GIANTS


class LineageError(ValueError):
    """A specimen identity cannot be formed from the evidence on hand."""


def _module_git_root() -> Path:
    """Git dir that contains this file. Used to read objects, not artifacts."""
    return Path(__file__).resolve().parents[2]


def load_watch_manifest(
    slug: str,
    *,
    manifest_dir: str | Path | None = None,
    git_root: str | Path | None = None,
) -> Optional[dict[str, Any]]:
    """Load a watch-manifest. Directory wins; git object is the sparse fallback.

    workspace/campaign/odyssey/watch-manifests/ is in git and often not on
    disk in a sparse worktree. git show HEAD:<path> is a read of the object
    store, not a claim that the file is checked out.
    """
    name = f"{slug}.json"
    if manifest_dir:
        path = Path(manifest_dir) / name
        if path.is_file():
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            if isinstance(doc, dict):
                doc["_manifest_source"] = str(path)
                return doc
    rel = f"{WATCH_MANIFEST_REL}/{name}"
    root = Path(git_root) if git_root else _module_git_root()
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{rel}"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or not r.stdout:
        return None
    try:
        doc = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    if isinstance(doc, dict):
        doc["_manifest_source"] = f"git:HEAD:{rel}"
        return doc
    return None


def load_json_file(path: str | Path) -> Optional[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def tensor_names_from_safetensors(path: str | Path) -> list[str]:
    """Bounded header parse. Does not load weights."""
    p = Path(path)
    try:
        with open(p, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return []
            hl = int.from_bytes(raw, "little")
            if hl <= 0 or hl > HEADER_CAP:
                return []
            header = json.loads(f.read(hl))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    if not isinstance(header, dict):
        return []
    return [k for k in header if k != "__metadata__"]


def tensor_names_from_specimen(root: Path) -> list[str]:
    idx = root / "model.safetensors.index.json"
    doc = load_json_file(idx)
    if doc:
        wm = doc.get("weight_map") or {}
        if isinstance(wm, dict) and wm:
            return list(wm)
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        return []
    return tensor_names_from_safetensors(shards[0])


def role_metadata(
    cfg: Optional[Mapping[str, Any]],
    files: Sequence[str] | None,
    *,
    repo: str = "",
    slug: str = "",
) -> dict[str, Any]:
    """Map a specimen onto §14.3 diversity classes. STATIC, from names/config."""
    cfg = cfg or {}
    mt = str(cfg.get("model_type") or "")
    archs = " ".join(str(a) for a in (cfg.get("architectures") or []))
    fileblob = " ".join(files or [])
    blob = f"{mt} {archs} {fileblob} {repo} {slug}".lower()
    roles: list[str] = []

    def add(role: str) -> None:
        if role in DIVERSITY_ROLES and role not in roles:
            roles.append(role)

    # Structural class first (§14.3), modifiers after. Primary is roles[0].
    if any(s in blob for s in ("vision", "visual", "vl-", "-vl", "multimodal", "sam2", "depth-anything", "wan2")):
        add("multimodal")
    if any(s in blob for s in ("moe", "a3b", "a2b", "experts", "mixtral")):
        add("MoE")
    if any(s in blob for s in ("rwkv", "mamba", "jamba", "ssm", "lfm2", "delta_net", "hybrid")):
        add("hybrid recurrent/state-space")
        add("state-heavy architecture")
    n_exp = cfg.get("num_experts") or cfg.get("n_routed_experts")
    try:
        if int(n_exp) >= 64:  # type: ignore[arg-type]
            add("extreme expert count")
    except (TypeError, ValueError):
        pass
    if not any(r in roles for r in ("MoE", "hybrid recurrent/state-space", "multimodal")):
        add("dense decoder")
    ctx = cfg.get("max_position_embeddings") or cfg.get("max_sequence_length")
    try:
        if int(ctx) >= 32768:  # type: ignore[arg-type]
            add("long-context")
    except (TypeError, ValueError):
        pass
    if any(s in blob for s in ("bitnet", "1.58", "q4_0", "gguf")):
        add("very-low-bit published checkpoint")
    if "tiktoken" in blob or "tokenizer.model" in blob:
        add("alternative tokenizer")
    return {
        "primary": roles[0],
        "roles": roles,
        "evidence_tier": EVIDENCE_TIER,
        "derived_from": "config.model_type / architectures / filenames / repo id",
        "diversity_catalog": "H-ROADMAP.md §14.3",
    }


def architecture_fingerprint(
    cfg: Optional[Mapping[str, Any]],
    names: Sequence[str] | None,
    *,
    repo: str,
    rev: str,
) -> dict[str, Any]:
    """CALL arch_recognizer.recognize when tensor names exist. Never loads weights."""
    cfg = dict(cfg or {})
    text = cfg.get("text_config") or {}

    def pick(k: str) -> Any:
        return cfg[k] if cfg.get(k) is not None else text.get(k)

    fp: dict[str, Any] = {
        "model_type": cfg.get("model_type"),
        "architectures": cfg.get("architectures"),
        "hidden_size": pick("hidden_size"),
        "num_hidden_layers": pick("num_hidden_layers"),
        "num_attention_heads": pick("num_attention_heads"),
        "num_key_value_heads": pick("num_key_value_heads"),
        "vocab_size": pick("vocab_size"),
        "organs": [],
        "strength": "CONFIG_ONLY",
        "evidence_tier": EVIDENCE_TIER,
        "loaded_weights": False,
    }
    if names:
        rec = _recognize(repo, rev, cfg, list(names))
        fp["organs"] = rec.get("organs") or []
        fp["unrecognized"] = rec.get("unrecognized") or []
        fp["novelty"] = rec.get("novelty")
        fp["n_tensors"] = rec.get("n_tensors")
        fp["strength"] = "ORGAN_FINGERPRINT"
        fp["recognizer_loaded_weights"] = rec.get("loaded_weights")
    return fp


def _recognize(repo: str, rev: str, cfg: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """Import-and-call site for arch_recognizer.recognize."""
    try:
        from tools.odyssey import arch_recognizer as ar
    except ImportError:
        import arch_recognizer as ar  # type: ignore
    return ar.recognize(repo, rev, cfg, names)


def artifact_lineage(
    slug: str,
    *,
    config: Mapping[str, Any],
    source_present: bool,
) -> list[dict[str, Any]]:
    """SOURCE -> NR -> NX. Absence is recorded, never invented."""
    try:
        from tools.odyssey.product_boundary import resolve_artifact
    except ImportError:
        from product_boundary import resolve_artifact  # type: ignore

    def stage(key: str, logical: str, kind: str, present_hint: bool | None = None) -> dict[str, Any]:
        hit = resolve_artifact(f"{key}:{slug}", config)
        present = hit["present"] if present_hint is None else present_hint
        row = {
            "stage": logical,
            "kind": kind,
            "path": hit["path"],
            "present": present,
            "root_key": hit["root_key"],
            "evidence_tier": EVIDENCE_TIER,
        }
        if not present:
            if logical == "NR":
                row["absent_because"] = "no NR index is written to the lake"
            elif logical == "NX":
                row["absent_because"] = "no NX executable is written to the lake"
            else:
                row["absent_because"] = "source body is not at the configured path"
        return row

    return [
        stage("specimens", "SOURCE", "verified source body", source_present),
        stage("nr", "NR", "noetic representation"),
        stage("nx", "NX", "noetic executable"),
    ]


def storage_tier_for(
    slug: str,
    *,
    config: Mapping[str, Any],
    source_dir: Path | None,
    staged_dir: Path | None,
    partial_dir: Path | None,
    watch_only: bool,
) -> dict[str, Any]:
    if staged_dir is not None and staged_dir.is_dir():
        role = "TIER1_HOT"
        path = staged_dir
    elif source_dir is not None and source_dir.is_dir():
        role = "TIER2_COLD"
        path = source_dir
    elif partial_dir is not None and partial_dir.is_dir():
        role = "PARTIAL"
        path = partial_dir
    elif watch_only:
        role = "GIT_METADATA"
        path = None
    else:
        role = "GIT_METADATA"
        path = None
    return {
        "role": role,
        "meaning": STORAGE_ROLES[role],
        "path": str(path) if path else None,
        "slug": slug,
        "evidence_tier": EVIDENCE_TIER,
        "catalog": "H-ROADMAP.md §14.1",
    }


def derive_lifecycle(
    *,
    watch: Optional[Mapping[str, Any]],
    lake_man: Optional[Mapping[str, Any]],
    source_dir: Path | None,
    partial_dir: Path | None,
    staged_dir: Path | None,
    fingerprinted: bool,
    nr_present: bool,
) -> tuple[str, str]:
    """Richest §14 state the evidence supports. Derived, not declared."""
    if staged_dir is not None and staged_dir.is_dir():
        return "SSD_STAGED", "specimen directory exists under the configured SSD stage"
    if nr_present:
        return "TRANSFER_READY", "an NR artifact is present beside the source body"
    if fingerprinted and source_dir is not None and source_dir.is_dir():
        return "CENSUSED", "sealed source plus an architecture fingerprint"
    if lake_man and lake_man.get("resolved_sha") and source_dir is not None and source_dir.is_dir():
        return "READY_COLD", "lake manifest records resolved_sha and the specimen directory exists"
    if lake_man and source_dir is not None and source_dir.is_dir():
        return "VERIFYING", "specimen directory exists but the lake manifest has no resolved_sha"
    if source_dir is not None and source_dir.is_dir():
        return "READY_COLD", "specimen directory exists under the configured specimens root"
    if partial_dir is not None and partial_dir.is_dir():
        return "DOWNLOADING", "sits under partial/"
    if watch and watch.get("files") and watch.get("resolved_sha"):
        return "MANIFEST_READY", "watch-manifest names files and a resolved sha"
    if watch and (watch.get("repo") or watch.get("revision")):
        return "IDENTITY_RESOLVED", "watch-manifest names repo/revision"
    return "DISCOVERED", "a slug was supplied and nothing on disk says more than that"


def express_lineage(
    slug: str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    manifest_dir: str | Path | None = None,
    git_root: str | Path | None = None,
) -> dict[str, Any]:
    """Join registry + provenance + role + fingerprint + SOURCE->NR->NX + tier."""
    if not slug:
        raise LineageError("slug is empty")
    if config is None:
        try:
            from tools.odyssey.product_boundary import safe_defaults
        except ImportError:
            from product_boundary import safe_defaults  # type: ignore
        config = safe_defaults()

    try:
        from tools.odyssey.product_boundary import resolve_artifact
    except ImportError:
        from product_boundary import resolve_artifact  # type: ignore

    watch_dir = manifest_dir or (config.get("artifact_roots") or {}).get("watch_manifests")
    watch = load_watch_manifest(slug, manifest_dir=watch_dir, git_root=git_root)
    lake_hit = resolve_artifact(f"lake_manifests:{slug}", config)
    lake_man = load_json_file(lake_hit["path"])
    source_hit = resolve_artifact(f"specimens:{slug}", config)
    partial_hit = resolve_artifact(f"partial:{slug}", config)
    stage_hit = resolve_artifact(f"stage:{slug}", config)
    nr_hit = resolve_artifact(f"nr:{slug}", config)
    source_dir = Path(source_hit["path"]) if source_hit["present"] else None
    partial_dir = Path(partial_hit["path"]) if partial_hit["present"] else None
    staged_dir = Path(stage_hit["path"]) if stage_hit["present"] else None

    repo = (watch or {}).get("repo") or (lake_man or {}).get("repo") or slug.split("@", 1)[0].replace("--", "/")
    rev = (watch or {}).get("revision") or (lake_man or {}).get("revision") or (
        slug.split("@", 1)[1] if "@" in slug else None
    )
    files = list((watch or {}).get("files") or [])
    sizes = dict((watch or {}).get("sizes") or {})

    cfg = load_json_file(Path(source_hit["path"]) / "config.json") if source_dir else None
    names: list[str] = tensor_names_from_specimen(source_dir) if source_dir else []
    fp = architecture_fingerprint(cfg, names, repo=str(repo), rev=str(rev or ""))
    role = role_metadata(cfg, files, repo=str(repo), slug=slug)
    source_present = source_dir is not None
    chain = artifact_lineage(slug, config=config, source_present=source_present)
    # artifact_lineage already CALLED resolve_artifact; honour live source_dir.
    chain[0]["present"] = source_present
    if source_present:
        chain[0].pop("absent_because", None)
    nr_present = bool(nr_hit["present"])
    life, why = derive_lifecycle(
        watch=watch, lake_man=lake_man, source_dir=source_dir,
        partial_dir=partial_dir, staged_dir=staged_dir,
        fingerprinted=fp["strength"] == "ORGAN_FINGERPRINT",
        nr_present=nr_present,
    )
    tier = storage_tier_for(
        slug, config=config, source_dir=source_dir, staged_dir=staged_dir,
        partial_dir=partial_dir, watch_only=watch is not None and source_dir is None,
    )
    provenance = {
        "repo": repo,
        "revision": rev,
        "resolved_sha": (watch or {}).get("resolved_sha") or (lake_man or {}).get("resolved_sha"),
        "n_files": (watch or {}).get("files") and len(files) or (lake_man or {}).get("n_files"),
        "bytes": (lake_man or {}).get("bytes") or (watch or {}).get("expected"),
        "sizes": sizes or None,
        "files": files or None,
        "reacquisition": (lake_man or {}).get("reacquisition") or (
            f"hf download {repo} --revision {rev} --local-dir <dest>" if repo and rev else None
        ),
        "acquired_at": (lake_man or {}).get("acquired_at"),
        "watch_manifest_source": (watch or {}).get("_manifest_source"),
        "lake_manifest_path": lake_hit["path"] if lake_hit["present"] else None,
        "evidence_tier": EVIDENCE_TIER,
    }
    return {
        "schema": "hawking.modellake.lineage.v1",
        "evidence_tier": EVIDENCE_TIER,
        "roadmap": {"section": "14", "title": "MODELLAKE — SPECIMEN SCHOOL AND STORAGE LIFECYCLE"},
        "slug": slug,
        "registry": {
            "id": slug,
            "lifecycle": life,
            "lifecycle_model": "H-ROADMAP.md §14",
            "lifecycle_derived_from": why,
            "path": source_hit["path"],
            "present": source_present,
        },
        "provenance": provenance,
        "role": role,
        "architecture_fingerprint": fp,
        "artifact_lineage": chain,
        "storage_tier": tier,
        "loaded_weights": False,
        "wrote": False,
    }


def registry_index(
    *,
    manifest_dir: str | Path | None = None,
    git_root: str | Path | None = None,
    slugs: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Registry of watch-manifest identities. Does not scan the live volume."""
    rows = []
    if slugs is None:
        slugs = []
        if manifest_dir and Path(manifest_dir).is_dir():
            slugs = [p.stem for p in sorted(Path(manifest_dir).glob("*.json"))]
        else:
            slugs = [CANONICAL_SPECIMEN]
    for slug in slugs:
        watch = load_watch_manifest(slug, manifest_dir=manifest_dir, git_root=git_root)
        if not watch:
            continue
        rows.append({
            "id": slug,
            "repo": watch.get("repo"),
            "revision": watch.get("revision"),
            "resolved_sha": watch.get("resolved_sha"),
            "n_files": len(watch.get("files") or []),
            "expected_bytes": watch.get("expected"),
            "manifest_source": watch.get("_manifest_source"),
        })
    return {
        "schema": "hawking.modellake.registry.v1",
        "evidence_tier": EVIDENCE_TIER,
        "n": len(rows),
        "specimens": rows,
    }


def build_lake_index(**kwargs: Any) -> dict[str, Any]:
    """CALL modellake_index.build. Durable catalog outside specimens/."""
    from tools.odyssey.modellake_index import build
    return build(**kwargs)


def query_lake_specimen(slug: str, **kwargs: Any) -> dict[str, Any]:
    """CALL modellake_index.query_specimen. One JSON read; no lake walk."""
    from tools.odyssey.modellake_index import query_specimen
    return query_specimen(slug, **kwargs)


def update_lake_specimen(slug: str, **kwargs: Any) -> dict[str, Any]:
    """CALL modellake_index.update_specimen. Walks that specimen only."""
    from tools.odyssey.modellake_index import update_specimen
    return update_specimen(slug, **kwargs)


def lake_index(**kwargs: Any) -> dict[str, Any]:
    """CALL modellake_index.load_catalog. Missing index is reported, not built."""
    from tools.odyssey.modellake_index import load_catalog
    cat = load_catalog(**kwargs)
    if cat is None:
        return {
            "present": False,
            "error": "index missing; run: python3 tools/odyssey/modellake.py index",
        }
    cat["present"] = True
    return cat


def lake_layout(**kwargs: Any) -> dict[str, Any]:
    """CALL modellake_index.layout. §14.1 storage roles as a queryable object."""
    from tools.odyssey.modellake_index import layout
    return layout(**kwargs)


def lake_tiering(**kwargs: Any) -> dict[str, Any]:
    """CALL modellake_index.tiering_census. Read-only size-tier + eligibility."""
    from tools.odyssey.modellake_index import tiering_census
    return tiering_census(**kwargs)
