#!/usr/bin/env python3.12
"""Load paired GLM×DSV4F activations for layer-correspondence cartography.

Assembles ``glm_layers`` / ``dsv4f_layers`` (each a sequence of ``[N, D]`` float
arrays) for :func:`lab.operators.frankenstein_cartography.emit_layer_correspondence`.

Sources (real capture layouts)
------------------------------
GLM (teacher-forced layer-major)::

    <capture_dir>/layers/L{nn:02d}.npz   # block_output/samples [B, 3, width]
    <capture_dir>/layers/L{nn:02d}.json   # sealed shard receipt (npz_sha256, …)

DSV4F (fullseq capture)::

    receipts/dsv4f_fullseq_capture_l{0,1}_receipt.json
    receipts/dsv4f_fullseq_capture_L{0,1}/traces/*.json

    Prefer host-resident float vectors when present (keys under each position's
    layer row, or a sibling activations/L{nn:02d}.npz).  Current captures seal
    only sha256 digests of Metal buffers (``host_activation_handoff_permitted:
    false``) — the loader reports that honestly and does **not** invent vectors.

Layer ranges are arguments so re-running after deeper DSV4F layers land is cheap
(load only the requested overlap).

Honesty: never fabricates activations or correspondence numbers.  Missing NPZ /
hash-only DSV4F traces → load report with explicit blockers; caller seals PENDING.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lab.operators.frankenstein_cartography import (
    CARTOGRAPHY_DIR,
    DEFAULT_LAYER_CORRESPONDENCE_PATH,
    DEFAULT_PHASE_ALIGNMENT_PATH,
    emit_layer_correspondence,
    emit_phase_alignment,
)
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GLM_CAPTURE = (
    REPO_ROOT
    / "workspace"
    / "campaign"
    / "evidence"
    / "models"
    / "frankenstein"
    / "teacher_forced"
    / "official_L0_stream_full_20260805T200728Z"
)

DEFAULT_DSV4F_L0_RECEIPT = REPO_ROOT / "receipts" / "dsv4f_fullseq_capture_l0_receipt.json"
DEFAULT_DSV4F_L1_RECEIPT = REPO_ROOT / "receipts" / "dsv4f_fullseq_capture_l1_receipt.json"

# Site keys accepted for DSV4F per-token layer rows (float vector or sha-only).
DSV4F_SITE_VECTOR_KEYS: tuple[str, ...] = (
    "late_hidden",
    "late_hidden_vector",
    "late_hidden_bf16",
    "post_moe",
    "post_moe_vector",
    "pre_router",
    "pre_router_vector",
    "mhc_pre_post_attn",
    "embedding_row",
)
DSV4F_SITE_SHA_KEYS: tuple[str, ...] = (
    "late_hidden_child_hc_sha256",
    "post_moe_sha256",
    "pre_router_ffn_norm_sha256",
    "mhc_pre_post_attn_hc_sha256",
    "embedding_row_sha256",
)

GLM_SAMPLE_SLOTS: tuple[str, ...] = ("first", "mid", "last")
DEFAULT_GLM_SITE = "block_output"
DEFAULT_GLM_SLOT = "last"  # one row per sequence → [B, width]


class CorrespondenceLoadError(RuntimeError):
    """Paired activations could not be assembled (payload missing / corpus gap)."""


@dataclass
class LayerRange:
    """Inclusive start, exclusive end (Python range semantics)."""

    start: int
    end: int

    def __post_init__(self) -> None:
        self.start = int(self.start)
        self.end = int(self.end)
        if self.start < 0 or self.end < self.start:
            raise CorrespondenceLoadError(
                f"invalid layer range [{self.start}, {self.end})"
            )

    def indices(self) -> list[int]:
        return list(range(self.start, self.end))

    def __len__(self) -> int:
        return max(0, self.end - self.start)

    def as_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end, "n": len(self)}


def parse_layer_range(spec: str | Sequence[int] | LayerRange | None) -> LayerRange:
    """Parse ``'0-2'``, ``'0:2'``, ``'0,1'``, ``(0, 2)``, or None → empty."""

    if spec is None:
        return LayerRange(0, 0)
    if isinstance(spec, LayerRange):
        return spec
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return LayerRange(int(spec[0]), int(spec[1]))
    if isinstance(spec, str):
        s = spec.strip()
        if not s:
            return LayerRange(0, 0)
        if re.fullmatch(r"\d+", s):
            i = int(s)
            return LayerRange(i, i + 1)
        m = re.fullmatch(r"(\d+)\s*[-:]\s*(\d+)", s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            # '0-1' means layers 0 and 1 inclusive → end exclusive = 2
            # but '0:2' is half-open. Accept both: if separator is '-' treat end
            # as inclusive when end >= start; if ':' half-open.
            if "-" in s and ":" not in s:
                return LayerRange(a, b + 1)
            return LayerRange(a, b)
        if "," in s:
            parts = [int(p.strip()) for p in s.split(",") if p.strip()]
            if not parts:
                return LayerRange(0, 0)
            return LayerRange(min(parts), max(parts) + 1)
    raise CorrespondenceLoadError(f"cannot parse layer range: {spec!r}")


@dataclass
class LoadReport:
    """Structured diagnostics — sealed into claim_boundary when PENDING/OK."""

    ok: bool = False
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    glm_capture_dir: str | None = None
    dsv4f_trace_dirs: list[str] = field(default_factory=list)
    dsv4f_receipts: list[str] = field(default_factory=list)
    glm_layer_range: dict[str, int] | None = None
    dsv4f_layer_range: dict[str, int] | None = None
    glm_layers_loaded: list[int] = field(default_factory=list)
    dsv4f_layers_loaded: list[int] = field(default_factory=list)
    glm_shapes: list[list[int]] = field(default_factory=list)
    dsv4f_shapes: list[list[int]] = field(default_factory=list)
    n_sequences: int = 0
    shared_example_ids: list[str] = field(default_factory=list)
    glm_example_ids: list[str] = field(default_factory=list)
    dsv4f_example_ids: list[str] = field(default_factory=list)
    glm_npz_present: list[str] = field(default_factory=list)
    glm_npz_missing: list[str] = field(default_factory=list)
    dsv4f_vector_sites_found: list[str] = field(default_factory=list)
    dsv4f_hash_only: bool = False
    alignment: str | None = None
    fabricated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def add_blocker(self, msg: str) -> None:
        self.blockers.append(msg)
        self.ok = False


@dataclass
class PairedActivations:
    glm_layers: list[np.ndarray]
    dsv4f_layers: list[np.ndarray]
    report: LoadReport
    glm_layer_indices: list[int]
    dsv4f_layer_indices: list[int]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def resolve_repo_path(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def glm_layer_npz_path(capture_dir: Path, layer: int) -> Path:
    return capture_dir / "layers" / f"L{layer:02d}.npz"


def glm_layer_json_path(capture_dir: Path, layer: int) -> Path:
    return capture_dir / "layers" / f"L{layer:02d}.json"


def discover_glm_layers(capture_dir: Path) -> list[int]:
    layers_dir = capture_dir / "layers"
    if not layers_dir.is_dir():
        return []
    found: list[int] = []
    for p in layers_dir.iterdir():
        m = re.fullmatch(r"L(\d{2})\.(json|npz)", p.name)
        if m:
            found.append(int(m.group(1)))
    return sorted(set(found))


def _read_json(path: Path) -> dict[str, Any]:
    import stat as statmod

    try:
        node = os.lstat(path)
    except OSError as exc:
        raise CorrespondenceLoadError(f"cannot read {path}: {exc}") from exc
    if statmod.S_ISLNK(node.st_mode) or not statmod.S_ISREG(node.st_mode):
        raise CorrespondenceLoadError(f"not a regular file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise CorrespondenceLoadError(f"JSON root must be object: {path}")
    return data


# ---------------------------------------------------------------------------
# GLM side
# ---------------------------------------------------------------------------


def load_glm_corpus_example_ids(capture_dir: Path) -> list[str]:
    """Stable example order from FROZEN_CORPUS or PAIRED_TRACE_CORPUS_INDEX."""

    frozen = capture_dir / "FROZEN_CORPUS_L0.json"
    # also L1 naming
    if not frozen.exists():
        for cand in sorted(capture_dir.glob("FROZEN_CORPUS_*.json")):
            frozen = cand
            break
    if frozen.exists():
        doc = _read_json(frozen)
        seqs = doc.get("sequences")
        if isinstance(seqs, list) and seqs:
            ids: list[str] = []
            for row in seqs:
                if isinstance(row, dict):
                    eid = row.get("example_id") or row.get("id")
                    if eid is not None:
                        ids.append(str(eid))
                elif isinstance(row, str):
                    ids.append(row)
            if ids:
                return ids
    index = capture_dir / "PAIRED_TRACE_CORPUS_INDEX.json"
    if index.exists():
        doc = _read_json(index)
        rows = doc.get("rows")
        if isinstance(rows, list):
            return [
                str(r["example_id"])
                for r in rows
                if isinstance(r, dict) and r.get("example_id")
            ]
    traces = capture_dir / "paired_traces"
    if traces.is_dir():
        return sorted(p.stem for p in traces.glob("*.json"))
    return []


def _slot_index(slot: str) -> int | None:
    if slot == "all":
        return None
    try:
        return GLM_SAMPLE_SLOTS.index(slot)
    except ValueError as exc:
        raise CorrespondenceLoadError(
            f"unknown GLM sample slot {slot!r}; expected one of {GLM_SAMPLE_SLOTS + ('all',)}"
        ) from exc


def load_glm_layer_matrix(
    capture_dir: Path,
    layer: int,
    *,
    site: str = DEFAULT_GLM_SITE,
    slot: str = DEFAULT_GLM_SLOT,
    example_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one GLM layer as ``[N, D]`` from ``block_output/samples`` (or site).

    Samples layout: ``[B, 3, width]`` for slots first/mid/last.
    """

    npz_path = glm_layer_npz_path(capture_dir, layer)
    json_path = glm_layer_json_path(capture_dir, layer)
    meta: dict[str, Any] = {
        "layer": layer,
        "npz_path": str(npz_path),
        "json_path": str(json_path),
        "npz_exists": npz_path.is_file(),
        "json_exists": json_path.is_file(),
    }
    if json_path.is_file():
        receipt = _read_json(json_path)
        meta["npz_bytes_claimed"] = receipt.get("npz_bytes")
        meta["npz_sha256_claimed"] = receipt.get("npz_sha256")
        meta["array_names"] = receipt.get("array_names")
    if not npz_path.is_file():
        raise CorrespondenceLoadError(
            f"GLM layer {layer} NPZ missing at {npz_path} "
            f"(JSON receipt present={json_path.is_file()}; "
            f"*.npz is gitignored and was not preserved after capture — "
            f"receipt claims ~{meta.get('npz_bytes_claimed')} bytes)"
        )
    with np.load(npz_path) as z:
        key = f"{site}/samples"
        if key not in z.files:
            # fall back: any */samples
            sample_keys = [k for k in z.files if k.endswith("/samples")]
            if not sample_keys:
                raise CorrespondenceLoadError(
                    f"GLM layer {layer} NPZ has no samples arrays; keys={list(z.files)[:20]}"
                )
            key = f"{site}/samples" if f"{site}/samples" in sample_keys else sample_keys[0]
        samples = np.asarray(z[key], dtype=np.float64)
    if samples.ndim != 3:
        raise CorrespondenceLoadError(
            f"GLM layer {layer} samples expected [B,3,W], got shape {samples.shape}"
        )
    bsz, n_slots, width = samples.shape
    si = _slot_index(slot)
    if si is None:
        # all slots → [B*3, W]
        mat = samples.reshape(bsz * n_slots, width)
    else:
        if si >= n_slots:
            raise CorrespondenceLoadError(
                f"slot index {si} out of range for n_slots={n_slots}"
            )
        mat = samples[:, si, :]
    if example_indices is not None:
        idx = np.asarray(list(example_indices), dtype=np.int64)
        if si is None:
            # expand each example to 3 consecutive rows
            rows = []
            for ei in idx.tolist():
                base = int(ei) * n_slots
                rows.extend(range(base, base + n_slots))
            mat = mat[np.asarray(rows, dtype=np.int64)]
        else:
            mat = mat[idx]
    meta["shape"] = list(mat.shape)
    meta["site"] = site
    meta["slot"] = slot
    meta["sample_key"] = key
    return mat, meta


def load_glm_layers(
    capture_dir: Path,
    layer_range: LayerRange,
    *,
    site: str = DEFAULT_GLM_SITE,
    slot: str = DEFAULT_GLM_SLOT,
    example_indices: Sequence[int] | None = None,
    report: LoadReport | None = None,
) -> tuple[list[np.ndarray], list[int]]:
    """Load GLM layers in ``layer_range`` → list of ``[N, D]`` arrays."""

    capture_dir = resolve_repo_path(capture_dir)
    rep = report or LoadReport()
    rep.glm_capture_dir = str(capture_dir)
    rep.glm_layer_range = layer_range.as_dict()

    available = discover_glm_layers(capture_dir)
    if not available:
        rep.add_blocker(f"no GLM layer shards under {capture_dir / 'layers'}")
        return [], []

    wanted = layer_range.indices()
    if not wanted:
        # auto: all discovered
        wanted = available
        rep.glm_layer_range = LayerRange(min(available), max(available) + 1).as_dict()

    matrices: list[np.ndarray] = []
    loaded: list[int] = []
    for layer in wanted:
        npz = glm_layer_npz_path(capture_dir, layer)
        js = glm_layer_json_path(capture_dir, layer)
        if npz.is_file():
            rep.glm_npz_present.append(str(npz))
        else:
            rep.glm_npz_missing.append(str(npz))
            if js.is_file():
                rep.warnings.append(
                    f"L{layer:02d}.json present but L{layer:02d}.npz absent "
                    f"(gitignored *.npz not in worktree)"
                )
            continue
        try:
            mat, _meta = load_glm_layer_matrix(
                capture_dir,
                layer,
                site=site,
                slot=slot,
                example_indices=example_indices,
            )
        except CorrespondenceLoadError as exc:
            rep.add_blocker(str(exc))
            continue
        matrices.append(mat)
        loaded.append(layer)
        rep.glm_shapes.append(list(mat.shape))

    rep.glm_layers_loaded = loaded
    if not matrices:
        if rep.glm_npz_missing and not rep.glm_npz_present:
            rep.add_blocker(
                "All requested GLM layer NPZ payloads are missing. "
                "JSON receipts remain (capture ran; bytes_captured≈892MB claimed) but "
                f"*.npz is gitignored and not on disk under {capture_dir / 'layers'}. "
                "Restore layer NPZs (or re-run teacher-forced capture with NPZ retention) "
                "before correspondence can measure CKA/CCA/Procrustes."
            )
        elif not rep.blockers:
            rep.add_blocker("no GLM layer matrices loaded")
    return matrices, loaded


# ---------------------------------------------------------------------------
# DSV4F side
# ---------------------------------------------------------------------------


def _resolve_dsv4f_trace_dir(receipt_or_dir: Path) -> tuple[Path, dict[str, Any] | None]:
    """Return (traces_dir, receipt_doc_or_None)."""

    path = resolve_repo_path(receipt_or_dir)
    if path.is_dir():
        # either the capture dir (with traces/) or traces/ itself
        if (path / "traces").is_dir():
            receipt = path / "DSV4F_FULLSEQ_CAPTURE_RECEIPT.json"
            doc = _read_json(receipt) if receipt.is_file() else None
            return path / "traces", doc
        return path, None
    if path.is_file():
        doc = _read_json(path)
        pt = doc.get("paired_traces") or {}
        tdir = pt.get("dir")
        if not tdir:
            raise CorrespondenceLoadError(
                f"DSV4F receipt {path} has no paired_traces.dir"
            )
        tpath = resolve_repo_path(tdir)
        if not tpath.is_dir():
            # try relative to receipt parent
            alt = path.parent / tdir
            if alt.is_dir():
                tpath = alt
            else:
                # relative to repo receipts/
                alt2 = REPO_ROOT / tdir
                if alt2.is_dir():
                    tpath = alt2
        return tpath, doc
    raise CorrespondenceLoadError(f"DSV4F path not found: {path}")


def list_dsv4f_traces(traces_dir: Path) -> list[Path]:
    return sorted(traces_dir.glob("*.json"))


def dsv4f_example_ids_from_traces(traces_dir: Path) -> list[str]:
    ids: list[str] = []
    for p in list_dsv4f_traces(traces_dir):
        try:
            doc = _read_json(p)
        except CorrespondenceLoadError:
            ids.append(p.stem)
            continue
        ids.append(str(doc.get("example_id") or p.stem))
    return ids


def _coerce_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        # nested list → flatten last-axis features (mean over sequence if 2d)
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            return arr.mean(axis=0)
        if arr.ndim >= 3:
            return arr.reshape(-1, arr.shape[-1]).mean(axis=0)
        return None
    if isinstance(value, dict):
        for k in ("values", "data", "vector", "bf16", "f32"):
            if k in value:
                return _coerce_vector(value[k])
    return None


def _extract_layer_vector_from_row(
    row: Mapping[str, Any],
    *,
    preferred_site: str = "late_hidden",
) -> tuple[np.ndarray | None, str | None, bool]:
    """Return (vector_or_None, site_key, hash_only_seen)."""

    hash_only = False
    # preferred site first
    preferred_keys = [
        preferred_site,
        f"{preferred_site}_vector",
        f"{preferred_site}_bf16",
        f"{preferred_site}_f32",
    ]
    for key in preferred_keys + list(DSV4F_SITE_VECTOR_KEYS):
        if key in row:
            vec = _coerce_vector(row[key])
            if vec is not None and vec.size > 0:
                return vec, key, False
    for key in DSV4F_SITE_SHA_KEYS:
        if key in row and row[key]:
            hash_only = True
    return None, None, hash_only


def load_dsv4f_layer_matrices_from_traces(
    traces_dir: Path,
    layer_range: LayerRange,
    *,
    site: str = "late_hidden",
    pool: str = "last_pos",  # last_pos | mean_pos
    example_ids: Sequence[str] | None = None,
    report: LoadReport | None = None,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Build per-layer ``[N, D]`` matrices from fullseq traces.

    If traces only carry sha256 digests (current fullseq capture), returns empty
    matrices and records ``dsv4f_hash_only`` + blockers on ``report``.
    """

    traces_dir = resolve_repo_path(traces_dir)
    rep = report or LoadReport()
    rep.dsv4f_trace_dirs.append(str(traces_dir))

    paths = list_dsv4f_traces(traces_dir)
    if not paths:
        rep.add_blocker(f"no DSV4F traces in {traces_dir}")
        return [], [], []

    # optional NPZ sidecar dir: activations/L{nn:02d}.npz with per-example rows
    act_dir = traces_dir.parent / "activations"
    wanted = layer_range.indices()
    if not wanted:
        # discover from first trace
        sample = _read_json(paths[0])
        sides = (sample.get("sides") or {}).get("dsv4f") or {}
        lr = sides.get("layers_run") or (sample.get("meta") or {}).get("layers_run")
        if isinstance(lr, list) and lr:
            wanted = sorted(int(x) for x in lr)
        else:
            wanted = [0, 1]
        rep.dsv4f_layer_range = LayerRange(min(wanted), max(wanted) + 1).as_dict()
    else:
        rep.dsv4f_layer_range = layer_range.as_dict()

    # Try NPZ / NPY sidecars first (offline-analysis export from fullseq capture
    # when --export-host-activations is set; default capture writes hashes only).
    if act_dir.is_dir():
        matrices: list[np.ndarray] = []
        loaded: list[int] = []
        for layer in wanted:
            npz = act_dir / f"L{layer:02d}.npz"
            npy = act_dir / f"L{layer:02d}.npy"
            mat: np.ndarray | None = None
            tag = ""
            if npz.is_file():
                with np.load(npz) as z:
                    key = site if site in z.files else (
                        "late_hidden" if "late_hidden" in z.files else z.files[0]
                    )
                    mat = np.asarray(z[key], dtype=np.float64)
                    tag = f"npz:{npz.name}:{key}"
            elif npy.is_file():
                mat = np.asarray(np.load(npy), dtype=np.float64)
                tag = f"npy:{npy.name}:{site}"
            if mat is None:
                continue
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            matrices.append(mat)
            loaded.append(layer)
            rep.dsv4f_shapes.append(list(mat.shape))
            rep.dsv4f_vector_sites_found.append(tag)
        if matrices:
            rep.dsv4f_layers_loaded = loaded
            ids_path = act_dir / "example_ids.json"
            if ids_path.is_file():
                try:
                    id_doc = _read_json(ids_path)
                    raw_ids = id_doc.get("example_ids")
                    if isinstance(raw_ids, list) and raw_ids:
                        ids = [str(x) for x in raw_ids]
                    else:
                        ids = dsv4f_example_ids_from_traces(traces_dir)
                except CorrespondenceLoadError:
                    ids = dsv4f_example_ids_from_traces(traces_dir)
            else:
                ids = dsv4f_example_ids_from_traces(traces_dir)
            return matrices, loaded, ids

    # Parse traces for host-resident vectors.
    id_filter = set(example_ids) if example_ids is not None else None
    # layer -> list of vectors (one per sequence)
    per_layer: dict[int, list[np.ndarray]] = {L: [] for L in wanted}
    ordered_ids: list[str] = []
    any_hash = False
    any_vector = False

    for path in paths:
        doc = _read_json(path)
        eid = str(doc.get("example_id") or path.stem)
        if id_filter is not None and eid not in id_filter:
            continue
        sides = doc.get("sides") or {}
        dsv = sides.get("dsv4f") or {}
        if not dsv.get("present", True) and dsv.get("capture_status") not in (
            "OK",
            None,
        ):
            continue
        positions = dsv.get("positions") or []
        if not positions:
            continue

        # pool over positions
        layer_acc: dict[int, list[np.ndarray]] = {L: [] for L in wanted}
        for pos_row in positions:
            if not isinstance(pos_row, dict):
                continue
            for lrow in pos_row.get("layers") or []:
                if not isinstance(lrow, dict):
                    continue
                try:
                    L = int(lrow.get("layer"))
                except (TypeError, ValueError):
                    continue
                if L not in per_layer:
                    continue
                vec, site_key, hash_only = _extract_layer_vector_from_row(
                    lrow, preferred_site=site
                )
                if hash_only:
                    any_hash = True
                if vec is not None:
                    any_vector = True
                    layer_acc[L].append(vec)
                    if site_key and site_key not in rep.dsv4f_vector_sites_found:
                        rep.dsv4f_vector_sites_found.append(site_key)

        if not any(layer_acc[L] for L in wanted):
            # no vectors this example
            ordered_ids.append(eid)  # still track for corpus diagnostics
            continue

        ordered_ids.append(eid)
        for L in wanted:
            vecs = layer_acc[L]
            if not vecs:
                continue
            if pool == "mean_pos":
                stacked = np.stack(vecs, axis=0).mean(axis=0)
            else:
                stacked = vecs[-1]
            per_layer[L].append(np.asarray(stacked, dtype=np.float64))

    rep.dsv4f_hash_only = bool(any_hash and not any_vector)

    if not any_vector:
        if any_hash:
            rep.add_blocker(
                "DSV4F fullseq traces contain only sha256 digests of Metal activation "
                "buffers (late_hidden_child_hc_sha256 / post_moe_sha256 / …), not host "
                "float vectors. Capture note: 'full tensor retained on device only'; "
                "host_activation_handoff_permitted=false. "
                "Re-run gravity_deepseek_v4_fullseq_capture with "
                "--export-host-activations (and --corpus-mode frozen) so "
                f"activations/L{{nn}}.npy + last-pos late_hidden f32 are written. "
                "Default path keeps host_activation_handoff_permitted=false."
            )
        else:
            rep.add_blocker(
                f"DSV4F traces under {traces_dir} have no activation vectors or hashes"
            )
        return [], [], ordered_ids or dsv4f_example_ids_from_traces(traces_dir)

    matrices = []
    loaded = []
    n_ref = None
    for L in wanted:
        rows = per_layer[L]
        if not rows:
            rep.warnings.append(f"DSV4F layer {L}: no vectors for selected examples")
            continue
        mat = np.stack(rows, axis=0)
        if n_ref is None:
            n_ref = mat.shape[0]
        elif mat.shape[0] != n_ref:
            rep.add_blocker(
                f"DSV4F layer {L} has N={mat.shape[0]} but expected N={n_ref}"
            )
            continue
        matrices.append(mat)
        loaded.append(L)
        rep.dsv4f_shapes.append(list(mat.shape))

    rep.dsv4f_layers_loaded = loaded
    if not matrices and not rep.blockers:
        rep.add_blocker("DSV4F vectors found but no layer matrices assembled")
    return matrices, loaded, ordered_ids


def load_dsv4f_from_receipts(
    receipts: Sequence[Path | str],
    layer_range: LayerRange,
    *,
    site: str = "late_hidden",
    pool: str = "last_pos",
    prefer_ladder: str = "L0",
    report: LoadReport | None = None,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Load DSV4F activations from one or more fullseq receipts.

    Prefers the L0 (32-seq) ladder by default so N matches the GLM L0 capture.
    """

    rep = report or LoadReport()
    # Prefer L0 receipt for sequence count alignment with GLM L0.
    ordered: list[Path] = [resolve_repo_path(p) for p in receipts]
    if prefer_ladder.upper() == "L0":
        ordered = sorted(
            ordered,
            key=lambda p: (0 if "l0" in p.name.lower() or "L0" in str(p) else 1, str(p)),
        )

    last_err: list[str] = []
    for path in ordered:
        try:
            traces_dir, doc = _resolve_dsv4f_trace_dir(path)
        except CorrespondenceLoadError as exc:
            last_err.append(str(exc))
            continue
        rep.dsv4f_receipts.append(str(path))
        if doc is not None:
            scope = doc.get("scope") or {}
            layers_run = scope.get("layers_run")
            if layers_run and not layer_range.indices():
                layer_range = LayerRange(min(layers_run), max(layers_run) + 1)
        n_blockers_before = len(rep.blockers)
        mats, loaded, ids = load_dsv4f_layer_matrices_from_traces(
            traces_dir,
            layer_range,
            site=site,
            pool=pool,
            report=rep,
        )
        if mats:
            return mats, loaded, ids
        # Hash-only / empty traces: do not re-probe every ladder with the same
        # structural gap (avoids duplicate blockers for L0 then L1).
        if rep.dsv4f_hash_only or len(rep.blockers) > n_blockers_before:
            return [], loaded, ids
        # keep trying other receipts only if this one failed soft
    for msg in last_err:
        rep.add_blocker(msg)
    return [], [], []


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def _align_example_indices(
    glm_ids: Sequence[str],
    dsv_ids: Sequence[str],
    *,
    mode: str = "intersection",
) -> tuple[list[int], list[int], list[str], str | None]:
    """Return (glm_indices, dsv_indices, shared_ids, blocker_or_None)."""

    if mode == "index":
        n = min(len(glm_ids), len(dsv_ids))
        if n < 2:
            return [], [], [], (
                f"index alignment needs ≥2 sequences; glm={len(glm_ids)} dsv={len(dsv_ids)}"
            )
        if set(glm_ids) != set(dsv_ids) and glm_ids[:n] != dsv_ids[:n]:
            # still allow with warning via blocker=None; caller adds warning
            pass
        return list(range(n)), list(range(n)), [f"index:{i}" for i in range(n)], None

    # intersection by example_id (stable order = GLM order)
    dsv_pos = {eid: i for i, eid in enumerate(dsv_ids)}
    shared: list[str] = []
    gi: list[int] = []
    di: list[int] = []
    for i, eid in enumerate(glm_ids):
        if eid in dsv_pos:
            shared.append(eid)
            gi.append(i)
            di.append(dsv_pos[eid])
    if len(shared) < 2:
        return gi, di, shared, (
            f"shared example_id intersection size={len(shared)} (<2). "
            f"GLM ids sample={list(glm_ids)[:3]}… DSV4F ids sample={list(dsv_ids)[:3]}…. "
            "Captures currently use different corpora (GLM pfv0:* mathlib/thesis vs "
            "DSV4F v0_math_*/v0_code_*). Re-run both captures on the same frozen "
            "FROZEN_CORPUS membership, or pass align=index only if N and order match."
        )
    return gi, di, shared, None


def load_paired_activations(
    *,
    glm_capture_dir: Path | str = DEFAULT_GLM_CAPTURE,
    dsv4f_receipts: Sequence[Path | str] | None = None,
    glm_layer_range: str | Sequence[int] | LayerRange | None = None,
    dsv4f_layer_range: str | Sequence[int] | LayerRange | None = "0-1",
    glm_site: str = DEFAULT_GLM_SITE,
    glm_slot: str = DEFAULT_GLM_SLOT,
    dsv4f_site: str = "late_hidden",
    dsv4f_pool: str = "last_pos",
    align: str = "intersection",
) -> PairedActivations:
    """Load real paired activations for the requested layer ranges.

    Returns a :class:`PairedActivations` even on failure (empty lists + report
    blockers). Never fabricates float activations.
    """

    rep = LoadReport(fabricated=False)
    glm_dir = resolve_repo_path(glm_capture_dir)
    rep.glm_capture_dir = str(glm_dir)

    g_range = parse_layer_range(glm_layer_range) if glm_layer_range is not None else LayerRange(0, 0)
    d_range = parse_layer_range(dsv4f_layer_range)

    receipts = list(dsv4f_receipts) if dsv4f_receipts is not None else [
        DEFAULT_DSV4F_L0_RECEIPT,
        DEFAULT_DSV4F_L1_RECEIPT,
    ]
    # Keep only existing paths (don't fail on optional L1).
    existing_receipts = [p for p in receipts if resolve_repo_path(p).exists()]
    if not existing_receipts:
        rep.add_blocker(
            "no DSV4F fullseq receipts found; tried "
            + ", ".join(str(resolve_repo_path(p)) for p in receipts)
        )

    glm_ids = load_glm_corpus_example_ids(glm_dir)
    rep.glm_example_ids = list(glm_ids)

    # Discover DSV4F ids from first usable receipt (for alignment planning).
    dsv_ids: list[str] = []
    for path in existing_receipts:
        try:
            tdir, _ = _resolve_dsv4f_trace_dir(path)
            dsv_ids = dsv4f_example_ids_from_traces(tdir)
            if dsv_ids:
                break
        except CorrespondenceLoadError as exc:
            rep.warnings.append(str(exc))
    rep.dsv4f_example_ids = list(dsv_ids)

    gi, di, shared, align_blocker = _align_example_indices(
        glm_ids, dsv_ids, mode=align
    )
    rep.alignment = align
    rep.shared_example_ids = shared
    if align_blocker:
        rep.add_blocker(align_blocker)
        if align == "intersection" and glm_ids and dsv_ids:
            # Offer a clear secondary note; do not silently index-align different corpora.
            rep.warnings.append(
                "Refusing silent index alignment across disjoint example_id sets "
                "(would compare unrelated prompts). Pass align='index' only if you "
                "intentionally accept order-based pairing."
            )

    # Load DSV4F first (defines available student layers).
    dsv_mats, dsv_loaded, dsv_order_ids = load_dsv4f_from_receipts(
        existing_receipts,
        d_range,
        site=dsv4f_site,
        pool=dsv4f_pool,
        report=rep,
    )
    if dsv_order_ids and not rep.dsv4f_example_ids:
        rep.dsv4f_example_ids = list(dsv_order_ids)

    # If alignment by intersection failed but user asked index and lengths match,
    # allow index when explicitly requested (already handled above).
    example_indices_glm: Sequence[int] | None = None
    if rep.ok is False and align == "intersection" and not shared:
        # Still attempt to load matrices for diagnostics, but leave ok=False.
        example_indices_glm = None
    elif shared and align == "intersection":
        example_indices_glm = gi
        # Re-filter DSV matrices rows by di if we already stacked all traces in order.
        # load_dsv4f stacks in trace-directory order = dsv_ids order when no filter.
        if dsv_mats and di:
            dsv_mats = [m[np.asarray(di, dtype=np.int64)] for m in dsv_mats]
            rep.dsv4f_shapes = [list(m.shape) for m in dsv_mats]
    elif align == "index" and dsv_mats:
        n = min(
            dsv_mats[0].shape[0],
            len(glm_ids) if glm_ids else dsv_mats[0].shape[0],
        )
        example_indices_glm = list(range(n))
        dsv_mats = [m[:n] for m in dsv_mats]
        rep.dsv4f_shapes = [list(m.shape) for m in dsv_mats]
        rep.n_sequences = n
        if glm_ids and dsv_ids and glm_ids[:n] != dsv_ids[:n]:
            rep.warnings.append(
                "align=index: GLM and DSV4F example_id sequences differ; "
                "pairing is positional only"
            )

    # Auto GLM range: if empty, load all available JSON layers that have NPZ,
    # but when DSV only has a few layers we still want many-to-one over GLM —
    # default to all discovered GLM layers.
    glm_mats, glm_loaded = load_glm_layers(
        glm_dir,
        g_range,
        site=glm_site,
        slot=glm_slot,
        example_indices=example_indices_glm,
        report=rep,
    )

    # Cross-check N
    if glm_mats and dsv_mats:
        n_g = glm_mats[0].shape[0]
        n_d = dsv_mats[0].shape[0]
        if n_g != n_d:
            rep.add_blocker(
                f"sample count mismatch after alignment: GLM N={n_g} vs DSV4F N={n_d}"
            )
        else:
            rep.n_sequences = n_g

    # Final ok only if both sides non-empty and no blockers.
    if glm_mats and dsv_mats and not rep.blockers:
        rep.ok = True
    else:
        rep.ok = False
        if not glm_mats and not any("GLM" in b for b in rep.blockers):
            rep.add_blocker("GLM activations not loaded")
        if not dsv_mats and not any("DSV4F" in b for b in rep.blockers):
            rep.add_blocker("DSV4F activations not loaded")

    return PairedActivations(
        glm_layers=glm_mats if rep.ok else [],
        dsv4f_layers=dsv_mats if rep.ok else [],
        report=rep,
        glm_layer_indices=glm_loaded if rep.ok else [],
        dsv4f_layer_indices=dsv_loaded if rep.ok else [],
    )


# ---------------------------------------------------------------------------
# Run + seal
# ---------------------------------------------------------------------------


def run_layer_correspondence(
    *,
    glm_capture_dir: Path | str = DEFAULT_GLM_CAPTURE,
    dsv4f_receipts: Sequence[Path | str] | None = None,
    glm_layer_range: str | Sequence[int] | LayerRange | None = None,
    dsv4f_layer_range: str | Sequence[int] | LayerRange | None = "0-1",
    glm_site: str = DEFAULT_GLM_SITE,
    glm_slot: str = DEFAULT_GLM_SLOT,
    dsv4f_site: str = "late_hidden",
    align: str = "intersection",
    out_dir: Path | str | None = None,
    write: bool = True,
    source_ok: str = "real_paired_captures",
) -> dict[str, Any]:
    """Load real activations (if possible) and seal correspondence + phase JSON.

    On load failure: seals PENDING with ``fabricated=False`` and embeds the full
    load report under ``claim_boundary.load_report`` (never invents CKA numbers).
    """

    paired = load_paired_activations(
        glm_capture_dir=glm_capture_dir,
        dsv4f_receipts=dsv4f_receipts,
        glm_layer_range=glm_layer_range,
        dsv4f_layer_range=dsv4f_layer_range,
        glm_site=glm_site,
        glm_slot=glm_slot,
        dsv4f_site=dsv4f_site,
        align=align,
    )
    base = resolve_repo_path(out_dir) if out_dir is not None else CARTOGRAPHY_DIR
    layer_path = base / "GLM_DSV4F_LAYER_CORRESPONDENCE.json"
    phase_path = base / "GLM_DSV4F_PHASE_ALIGNMENT.json"
    report_path = base / "GLM_DSV4F_CORRESPONDENCE_LOAD_REPORT.json"

    load_doc = seal(
        {
            "schema": "hawking.frankenstein.correspondence_load_report.v1",
            "name": "GLM_DSV4F_CORRESPONDENCE_LOAD_REPORT",
            "report": paired.report.to_dict(),
            "fabricated": False,
        }
    )
    if write:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(load_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if paired.report.ok and paired.glm_layers and paired.dsv4f_layers:
        layer = emit_layer_correspondence(
            glm_layers=paired.glm_layers,
            dsv4f_layers=paired.dsv4f_layers,
            source=source_ok,
            glm_runtime_available=True,
            out_path=layer_path,
            write=write,
        )
        phase = emit_phase_alignment(
            glm_layers=paired.glm_layers,
            dsv4f_layers=paired.dsv4f_layers,
            source=source_ok,
            out_path=phase_path,
            write=write,
        )
        # Enrich claim_boundary with load provenance (rewrite sealed docs).
        if write:
            for path, doc in ((layer_path, layer), (phase_path, phase)):
                body = {k: v for k, v in doc.items() if not str(k).startswith("_")}
                cb = dict(body.get("claim_boundary") or {})
                cb["load_report_seal_sha256"] = load_doc["seal_sha256"]
                cb["glm_layer_indices"] = paired.glm_layer_indices
                cb["dsv4f_layer_indices"] = paired.dsv4f_layer_indices
                cb["n_sequences"] = paired.report.n_sequences
                body["claim_boundary"] = cb
                body.pop("seal_sha256", None)
                resealed = seal(body)
                verify(resealed, label=path.name)
                path.write_text(
                    json.dumps(resealed, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if path == layer_path:
                    layer = {**resealed, "_written_path": str(path)}
                else:
                    phase = {**resealed, "_written_path": str(path)}
        status = "OK"
    else:
        layer = emit_layer_correspondence(
            glm_layers=None,
            dsv4f_layers=None,
            source="pending_activations_load_blocked",
            out_path=layer_path,
            write=False,
        )
        phase = emit_phase_alignment(
            glm_layers=None,
            dsv4f_layers=None,
            source="pending_activations_load_blocked",
            out_path=phase_path,
            write=False,
        )
        # Inject precise blockers into PENDING seals.
        for label, doc, path in (
            ("layer", layer, layer_path),
            ("phase", phase, phase_path),
        ):
            body = {k: v for k, v in doc.items() if not str(k).startswith("_")}
            body["status"] = "PENDING_REAL_ACTIVATIONS"
            body["source"] = "pending_activations_load_blocked"
            body["fabricated"] = False
            body["executed"] = False
            body["load_blockers"] = list(paired.report.blockers)
            body["load_warnings"] = list(paired.report.warnings)
            body["note"] = (
                "emit_layer_correspondence refused: real capture receipts exist but "
                "host-resident float activations could not be assembled. "
                "Blockers: " + " | ".join(paired.report.blockers[:4])
            )
            body["missing_infra"] = "; ".join(paired.report.blockers[:6])
            cb = dict(body.get("claim_boundary") or {})
            cb["correspondence_numbers_measured"] = False
            cb["awaiting"] = (
                "GLM layer NPZ payloads + DSV4F host float vectors on a shared "
                "frozen corpus (same example_ids)"
            )
            cb["load_report_seal_sha256"] = load_doc["seal_sha256"]
            cb["load_report"] = {
                "ok": False,
                "blockers": paired.report.blockers,
                "glm_npz_missing_count": len(paired.report.glm_npz_missing),
                "glm_npz_present_count": len(paired.report.glm_npz_present),
                "dsv4f_hash_only": paired.report.dsv4f_hash_only,
                "shared_example_ids": len(paired.report.shared_example_ids),
                "glm_layers_json_discovered": discover_glm_layers(resolve_repo_path(glm_capture_dir)),
            }
            body["claim_boundary"] = cb
            body.pop("seal_sha256", None)
            resealed = seal(body)
            verify(resealed, label=f"{label} pending")
            if write:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(resealed, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                resealed = {**resealed, "_written_path": str(path)}
            if label == "layer":
                layer = resealed
            else:
                phase = resealed
        status = "PENDING_REAL_ACTIVATIONS"

    return {
        "status": status,
        "fabricated": False,
        "load_report": paired.report.to_dict(),
        "load_report_path": str(report_path) if write else None,
        "layer_correspondence": {
            "status": layer.get("status"),
            "seal_sha256": layer.get("seal_sha256"),
            "path": layer.get("_written_path", str(layer_path)),
            "fabricated": layer.get("fabricated"),
            "matrix": layer.get("matrix"),
            "matrices": layer.get("matrices"),
            "sample_pair_metrics": layer.get("sample_pair_metrics"),
        },
        "phase_alignment": {
            "status": phase.get("status"),
            "seal_sha256": phase.get("seal_sha256"),
            "path": phase.get("_written_path", str(phase_path)),
            "fabricated": phase.get("fabricated"),
            "pairs": phase.get("pairs"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Load real GLM×DSV4F activations and seal layer correspondence "
            "(CKA/CCA/Procrustes). PENDING+blockers if payloads missing."
        )
    )
    parser.add_argument(
        "--glm-capture-dir",
        type=Path,
        default=DEFAULT_GLM_CAPTURE,
        help="Teacher-forced capture directory (layers/, paired_traces/, …)",
    )
    parser.add_argument(
        "--dsv4f-receipt",
        type=Path,
        action="append",
        default=None,
        help="DSV4F fullseq receipt JSON (repeatable). Default: L0 then L1 receipts.",
    )
    parser.add_argument(
        "--glm-layers",
        type=str,
        default=None,
        help="GLM layer range, e.g. '0-77', '0:78', '0,1'. Default: all discovered.",
    )
    parser.add_argument(
        "--dsv4f-layers",
        type=str,
        default="0-1",
        help="DSV4F layer range (inclusive dash form). Default: 0-1.",
    )
    parser.add_argument(
        "--glm-site",
        type=str,
        default=DEFAULT_GLM_SITE,
        help="GLM NPZ site prefix (default block_output → block_output/samples).",
    )
    parser.add_argument(
        "--glm-slot",
        type=str,
        default=DEFAULT_GLM_SLOT,
        choices=list(GLM_SAMPLE_SLOTS) + ["all"],
    )
    parser.add_argument(
        "--dsv4f-site",
        type=str,
        default="late_hidden",
        help="Preferred DSV4F vector site key.",
    )
    parser.add_argument(
        "--align",
        type=str,
        default="intersection",
        choices=("intersection", "index"),
        help="Sequence pairing: shared example_id (default) or positional index.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=CARTOGRAPHY_DIR,
        help="Directory for sealed correspondence JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load + report only; do not write seals.",
    )
    args = parser.parse_args(argv)

    result = run_layer_correspondence(
        glm_capture_dir=args.glm_capture_dir,
        dsv4f_receipts=args.dsv4f_receipt,
        glm_layer_range=args.glm_layers,
        dsv4f_layer_range=args.dsv4f_layers,
        glm_site=args.glm_site,
        glm_slot=args.glm_slot,
        dsv4f_site=args.dsv4f_site,
        align=args.align,
        out_dir=args.out_dir,
        write=not args.dry_run,
    )
    print(json.dumps(
        {
            "status": result["status"],
            "fabricated": result["fabricated"],
            "blockers": result["load_report"].get("blockers"),
            "warnings": result["load_report"].get("warnings"),
            "glm_layers_loaded": result["load_report"].get("glm_layers_loaded"),
            "dsv4f_layers_loaded": result["load_report"].get("dsv4f_layers_loaded"),
            "n_sequences": result["load_report"].get("n_sequences"),
            "layer_path": result["layer_correspondence"].get("path"),
            "phase_path": result["phase_alignment"].get("path"),
            "layer_seal": result["layer_correspondence"].get("seal_sha256"),
            "sample_pair_metrics": result["layer_correspondence"].get(
                "sample_pair_metrics"
            ),
        },
        indent=2,
        sort_keys=True,
    ))
    return 0 if result["status"] == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
