"""MLP BYTE CENSUS — why are we reading these bytes at all?

Sealed-3.14 (Qwen3.8-27B, artifact_root NOETIC_PARENT_A) moves a measured
9,878,901,136 active weight bytes per decoded token. A historically quoted
MLP share of that total is not an input: this module re-derives the split
from the HQ38M20 catalog, tensor by tensor, and refuses if the per-organ
sum cannot be reconciled against the recorded active-byte total.

The MLP question is not "make Q4 faster". It is: what independent function
does this MLP contain, and which of these bytes exist only because of the
present affine-Q2 packing?

    python3 tools/future/mlp_byte_census.py --build
    python3 -m pytest tools/future/test_mlp_byte_census.py -q

evidence_class STATIC_ONLY. No GPU. No bench lock. Does not touch crates/.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import re
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import REPO, git, write_receipt
from tools.future.ebpw_categories import PRODUCTION, judge_dense_rematerialization


RECEIPT = "MLP_BYTE_CENSUS.json"
SCHEMA = "hawking.future.mlp_byte_census.v1"
VERSION = 1
RECORDED_BY = "tools/future/mlp_byte_census.py"

SEALED_REL = "hcli/hawking-native.sealed-3.14.json"
PROFILE_REL = "receipts/future/QWEN27_ACCELERATOR_PROFILE_SCHEMA.json"
NOETIC_RELS = (
    "receipts/future/evidence/NOETIC_NEGATIVE_SCIENCE.json",
    "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
)
RIVAL_REL = "receipts/future/RIVAL_CODEC_SCREEN.json"
QN_REL = "tools/headless/negative_science.py"
NEGATIVE_INDEX_REL = "receipts/future/NEGATIVE_SCIENCE_INDEX.json"

DEFAULT_ARTIFACT_ROOT = Path("/Users/scammermike/noetic/NOETIC_PARENT_A")
CATALOG_NAME = "catalog.hq38m20"
MAGIC = b"HQ38M20\0"
RECORD_SIZE = 128

# Published measured active-weight total (profile schema / bytes atlas).
# This is the reconciliation TARGET. The MLP split is not this constant.
RECORDED_ACTIVE_WEIGHT_BYTES_PER_TOKEN = 9_878_901_136

# Incumbent embedding packing on this artifact (Q4 group-64 row lookup).
Q4_GROUP = 64
AFFINE_CODE_BITS = 2

DIRECT_CONSUME = "DIRECT_CONSUME"
REJECTED_DENSE_REMAT = "REJECTED_DENSE_REMAT"
DEPENDS_ON_LOWERING = "DEPENDS_ON_LOWERING"

ALREADY_FALSIFIED = "ALREADY_FALSIFIED"
OPEN = "OPEN"

STATUS_LIVE = "OPEN"
STATUS_DEAD = "ALREADY_FALSIFIED"

REQUIRED_FAMILY_IDS: tuple[str, ...] = (
    "lower_bit",
    "heterogeneous_precision",
    "generated_weights",
    "shared_bases",
    "factorized_programs",
    "dictionary_programs",
    "product_codebook_programs",
    "sparse_residuals",
    "cross_layer_prediction",
    "routed_subprograms",
    "shared_input_transforms",
    "latent_accumulation",
    "function_replacement",
    "capability_sensitive_literal_islands",
)

# Fine organs the census must name. Linear-attn projections are this
# hybrid body's attention q/k/v/o analogue (48 of 64 layers).
ORGAN_ORDER: tuple[str, ...] = (
    "mlp.gate",
    "mlp.up",
    "mlp.down",
    "attention.q",
    "attention.k",
    "attention.v",
    "attention.o",
    "attention.linear_qkvz",
    "attention.linear_ba",
    "attention.linear_out",
    "attention.linear_conv1d",
    "norms.input",
    "norms.post_attn",
    "norms.final",
    "norms.q",
    "norms.k",
    "norms.linear_attn",
    "embedding",
    "lm_head",
    "state.A_log",
    "state.dt_bias",
)

ORGAN_FAMILY_OF: dict[str, str] = {
    "mlp.gate": "mlp",
    "mlp.up": "mlp",
    "mlp.down": "mlp",
    "attention.q": "attention",
    "attention.k": "attention",
    "attention.v": "attention",
    "attention.o": "attention",
    "attention.linear_qkvz": "attention",
    "attention.linear_ba": "attention",
    "attention.linear_out": "attention",
    "attention.linear_conv1d": "attention",
    "norms.input": "norms",
    "norms.post_attn": "norms",
    "norms.final": "norms",
    "norms.q": "norms",
    "norms.k": "norms",
    "norms.linear_attn": "norms",
    "embedding": "embedding",
    "lm_head": "lm_head",
    "state.A_log": "state",
    "state.dt_bias": "state",
}

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. Active bytes are "
    "catalog stored bytes of tensors a decode token reads, with the embedding "
    "table counted as one Q4 row rather than the vocab table. Activations, "
    "KV cache, and DeltaNet recurrent state are not in the HQ38M20 catalog "
    "and are not invented here. The MLP share is re-derived from the tensor "
    "census; a historically quoted percentage is not an input."
)


class CensusRefuse(ValueError):
    """The census refused rather than guessing or silently passing."""


class CatalogAbsent(CensusRefuse):
    """The HQ38M20 catalog (or specimen config) is not readable."""


class UnreconciledCensus(CensusRefuse):
    """Per-organ active-byte sum does not match the recorded total."""

    def __init__(
        self,
        active: int,
        recorded: int,
        *,
        detail: str = "",
    ) -> None:
        self.active = int(active)
        self.recorded = int(recorded)
        extra = f" ({detail})" if detail else ""
        super().__init__(
            f"REFUSED: per-organ active-byte sum {active} != "
            f"recorded active-byte total {recorded}{extra}"
        )


class UnclassifiedTensor(CensusRefuse):
    """A catalog tensor matched no organ. Hiding it would break the sum."""


# ---------------------------------------------------------------------------
# Authority loaders. Sparse checkout is not absence.
# ---------------------------------------------------------------------------


def _read_rel(rel: str) -> tuple[str | None, str]:
    path = REPO / rel
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8", errors="replace"), "disk"
        except OSError:
            pass
    blob = git("show", f"HEAD:{rel}")
    if blob:
        return blob, "git:HEAD"
    return None, "missing"


def _load_json_rel(rel: str) -> tuple[dict[str, Any] | None, str]:
    text, via = _read_rel(rel)
    if text is None:
        return None, via
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return None, f"{via}_unparseable"
    return doc if isinstance(doc, dict) else None, via


def load_sealed() -> dict[str, Any]:
    doc, via = _load_json_rel(SEALED_REL)
    if doc is None:
        raise CatalogAbsent(
            f"{SEALED_REL} unreachable on disk and git HEAD (via={via})"
        )
    doc = dict(doc)
    doc["_resolved_via"] = via
    return doc


def recorded_active_total(profile: Mapping[str, Any] | None = None) -> int:
    """The published measured active-weight total. Not the MLP split."""
    doc = profile
    if doc is None:
        doc, _via = _load_json_rel(PROFILE_REL)
    if isinstance(doc, Mapping):
        node: Any = doc.get("control_profile") if "control_profile" in doc else doc
        if isinstance(node, Mapping):
            model = node.get("active_byte_model")
            if isinstance(model, Mapping):
                raw = model.get("active_weight_bytes_per_token")
                if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
                    return raw
    return RECORDED_ACTIVE_WEIGHT_BYTES_PER_TOKEN


def resolve_artifact_root(sealed: Mapping[str, Any] | None = None) -> Path:
    sealed = sealed if sealed is not None else load_sealed()
    candidates: list[Path] = []
    raw = sealed.get("artifact_root")
    if isinstance(raw, str) and raw.strip():
        candidates.append(Path(raw).expanduser())
    candidates.append(DEFAULT_ARTIFACT_ROOT)
    for root in candidates:
        if (root / CATALOG_NAME).is_file():
            return root
    raise CatalogAbsent(
        f"{CATALOG_NAME} not readable under {candidates!r}; "
        "refusing to invent a tensor census"
    )


# ---------------------------------------------------------------------------
# Catalog parser. Same HQ38M20 layout as tools/accelerator/bytes_atlas.py;
# that path is not materialized in this sparse checkout, so the parser lives
# here rather than being imported. Byte counts come from SEGMENTS.
# ---------------------------------------------------------------------------


def parse_catalog(path: Path) -> list[tuple[str, int]]:
    """(tensor_name, stored_bytes) in catalog order. Shared segments refuse."""
    try:
        blob = path.read_bytes()
    except OSError as exc:
        raise CatalogAbsent(f"cannot read {path}: {exc}") from exc
    if blob[:8] != MAGIC:
        raise CensusRefuse(f"bad catalog magic {blob[:8]!r} in {path}")
    _ver, n_rec, n_seg, _a, name_len, _c = struct.unpack("<IIIIII", blob[8:32])
    off = 32
    segs: dict[int, int] = {}
    for _ in range(n_seg):
        sid, nlen, nbytes, _dg = struct.unpack("<HHQ32s", blob[off:off + 44])
        off += 44
        segs[sid] = int(nbytes)
        off += nlen
    tbl = blob[off:off + n_rec * RECORD_SIZE]
    names = blob[off + n_rec * RECORD_SIZE:]
    if len(names) != name_len:
        raise CensusRefuse("catalog name blob length disagrees with the header")
    out: list[tuple[str, int]] = []
    seen: set[int] = set()
    for i in range(n_rec):
        rec = tbl[i * RECORD_SIZE:(i + 1) * RECORD_SIZE]
        noff, nlen = struct.unpack("<IH", rec[0:6])
        sid = struct.unpack("<H", rec[36:38])[0]
        if sid in seen:
            raise CensusRefuse(
                f"segment {sid} referenced by more than one record; "
                "byte count is per SEGMENT so a share would double-count"
            )
        seen.add(sid)
        if sid not in segs:
            raise CensusRefuse(f"record {i} names missing segment {sid}")
        out.append((names[noff:noff + nlen].decode(), segs[sid]))
    return out


def load_geometry(root: Path) -> dict[str, Any]:
    cfg_path = root / "config.json"
    if not cfg_path.is_file():
        raise CatalogAbsent(f"config.json missing under {root}")
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogAbsent(f"config.json unreadable under {root}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise CensusRefuse("specimen config is not an object")
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    if not isinstance(text, dict):
        raise CensusRefuse("specimen text_config is not an object")

    def _posint(key: str) -> int:
        raw = text.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise CensusRefuse(f"specimen config missing positive int {key}")
        return raw

    layer_types = text.get("layer_types")
    return {
        "hidden_size": _posint("hidden_size"),
        "intermediate_size": _posint("intermediate_size"),
        "num_hidden_layers": _posint("num_hidden_layers"),
        "vocab_size": _posint("vocab_size"),
        "head_dim": text.get("head_dim"),
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": text.get("num_key_value_heads"),
        "linear_num_key_heads": text.get("linear_num_key_heads"),
        "linear_num_value_heads": text.get("linear_num_value_heads"),
        "linear_key_head_dim": text.get("linear_key_head_dim"),
        "linear_value_head_dim": text.get("linear_value_head_dim"),
        "full_attention_interval": text.get("full_attention_interval"),
        "layer_types": list(layer_types) if isinstance(layer_types, list) else None,
        "tie_word_embeddings": bool(
            text.get("tie_word_embeddings") or cfg.get("tie_word_embeddings")
        ),
        "model_type": text.get("model_type") or cfg.get("model_type"),
    }


def q4_group64_row_bytes(hidden: int, group: int = Q4_GROUP) -> int:
    """One decode embedding lookup: nibbles + one f16 scale per group."""
    if hidden <= 0 or hidden % group:
        raise CensusRefuse(
            f"hidden {hidden} is not a positive multiple of Q4 group {group}"
        )
    return hidden // 2 + (hidden // group) * 2


# ---------------------------------------------------------------------------
# Organ classification. Unclassified is a refusal, not "other".
# ---------------------------------------------------------------------------


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

_ORGAN_NEEDLES: tuple[tuple[str, str], ...] = (
    ("mlp.gate_proj", "mlp.gate"),
    ("mlp.up_proj", "mlp.up"),
    ("mlp.down_proj", "mlp.down"),
    ("self_attn.q_proj", "attention.q"),
    ("self_attn.k_proj", "attention.k"),
    ("self_attn.v_proj", "attention.v"),
    ("self_attn.o_proj", "attention.o"),
    ("self_attn.q_norm", "norms.q"),
    ("self_attn.k_norm", "norms.k"),
    ("linear_attn.in_proj_qkvz", "attention.linear_qkvz"),
    ("linear_attn.in_proj_ba", "attention.linear_ba"),
    ("linear_attn.out_proj", "attention.linear_out"),
    ("linear_attn.conv1d", "attention.linear_conv1d"),
    ("linear_attn.norm", "norms.linear_attn"),
    ("linear_attn.A_log", "state.A_log"),
    ("linear_attn.dt_bias", "state.dt_bias"),
    ("input_layernorm", "norms.input"),
    ("post_attention_layernorm", "norms.post_attn"),
)


def classify_tensor(name: str) -> tuple[int | None, str, bool]:
    """(layer or None, organ, whole_tensor_active_per_token)."""
    if "embed_tokens" in name:
        return None, "embedding", False
    if name.endswith("lm_head.weight") or ".lm_head.weight" in name:
        return None, "lm_head", True
    match = _LAYER_RE.search(name)
    if match is None:
        if name.endswith("model.norm.weight") or name.endswith(".norm.weight"):
            return None, "norms.final", True
        raise UnclassifiedTensor(name)
    layer = int(match.group(1))
    rest = name.split(f".layers.{layer}.", 1)[1]
    for needle, organ in _ORGAN_NEEDLES:
        if rest.startswith(needle) or needle in rest:
            return layer, organ, True
    raise UnclassifiedTensor(name)


def active_bytes_for(
    organ: str,
    stored: int,
    *,
    whole: bool,
    geometry: Mapping[str, Any],
) -> int:
    if whole:
        return int(stored)
    if organ != "embedding":
        raise CensusRefuse(
            f"partial-tensor active model is only defined for embedding, not {organ}"
        )
    return q4_group64_row_bytes(int(geometry["hidden_size"]))


def reconcile_active(active: int, recorded: int, *, detail: str = "") -> None:
    """Raise rather than silently pass an irreconcilable census."""
    if int(active) != int(recorded):
        raise UnreconciledCensus(int(active), int(recorded), detail=detail)


# ---------------------------------------------------------------------------
# Census.
# ---------------------------------------------------------------------------


def _share(nbytes: int, total: int) -> float:
    if total <= 0:
        raise CensusRefuse("cannot take a share of a non-positive total")
    return nbytes / total


def _empty_organ_row(organ: str) -> dict[str, Any]:
    return {
        "organ": organ,
        "family": ORGAN_FAMILY_OF[organ],
        "n_tensors": 0,
        "storage_bytes": 0,
        "active_bytes": 0,
        "share_of_active": 0.0,
        "whole_tensor_active": organ != "embedding",
    }


def census(
    *,
    root: Path | None = None,
    recorded_total: int | None = None,
    catalog_records: Sequence[tuple[str, int]] | None = None,
    geometry: Mapping[str, Any] | None = None,
    sealed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Exact per-organ active-byte census for one decoded token.

    Raises UnreconciledCensus if the per-organ sum does not match
    `recorded_total`. Raises UnclassifiedTensor if any catalog name is
    not an organ. Does not invent KV / recurrent-state bytes.
    """
    sealed_doc = dict(sealed) if sealed is not None else load_sealed()
    artifact = root if root is not None else resolve_artifact_root(sealed_doc)
    geo = dict(geometry) if geometry is not None else load_geometry(artifact)
    records = (
        list(catalog_records)
        if catalog_records is not None
        else parse_catalog(artifact / CATALOG_NAME)
    )
    if not records:
        raise CensusRefuse("catalog is empty; refusing an empty census")
    recorded = (
        int(recorded_total)
        if recorded_total is not None
        else recorded_active_total()
    )

    n_layers = int(geo["num_hidden_layers"])
    layer_types = geo.get("layer_types")
    per_layer: list[dict[str, Any]] = [
        {
            "layer": i,
            "kind": (
                layer_types[i]
                if isinstance(layer_types, list) and i < len(layer_types)
                else "unknown"
            ),
            "organs": {},
        }
        for i in range(n_layers)
    ]
    globals_organs: dict[str, dict[str, Any]] = {}
    by_organ: dict[str, dict[str, Any]] = {
        name: _empty_organ_row(name) for name in ORGAN_ORDER
    }
    tensors: list[dict[str, Any]] = []
    catalog_total = 0

    for name, stored in records:
        stored_i = int(stored)
        catalog_total += stored_i
        layer, organ, whole = classify_tensor(name)
        if organ not in by_organ:
            raise UnclassifiedTensor(name)
        active = active_bytes_for(organ, stored_i, whole=whole, geometry=geo)
        row = {
            "name": name,
            "layer": layer,
            "organ": organ,
            "family": ORGAN_FAMILY_OF[organ],
            "storage_bytes": stored_i,
            "active_bytes": active,
            "whole_tensor_active": whole,
        }
        tensors.append(row)
        acc = by_organ[organ]
        acc["n_tensors"] += 1
        acc["storage_bytes"] += stored_i
        acc["active_bytes"] += active
        if layer is None:
            globals_organs.setdefault(organ, {"organ": organ, "storage_bytes": 0, "active_bytes": 0})
            globals_organs[organ]["storage_bytes"] += stored_i
            globals_organs[organ]["active_bytes"] += active
        else:
            if layer < 0 or layer >= n_layers:
                raise CensusRefuse(
                    f"{name} names layer {layer} but config has {n_layers} layers"
                )
            bucket = per_layer[layer]["organs"]
            slot = bucket.setdefault(
                organ, {"organ": organ, "storage_bytes": 0, "active_bytes": 0, "n_tensors": 0}
            )
            slot["storage_bytes"] += stored_i
            slot["active_bytes"] += active
            slot["n_tensors"] += 1

    active_total = sum(row["active_bytes"] for row in by_organ.values())
    unread_embed = by_organ["embedding"]["storage_bytes"] - by_organ["embedding"]["active_bytes"]
    accounted_storage = active_total + unread_embed
    if accounted_storage != catalog_total:
        raise UnreconciledCensus(
            accounted_storage,
            catalog_total,
            detail=(
                "attribution (active + unread embedding table) != catalog "
                f"storage; unread_embed={unread_embed}"
            ),
        )
    reconcile_active(
        active_total,
        recorded,
        detail="per-organ active-byte sum vs recorded measured total",
    )

    for row in by_organ.values():
        row["share_of_active"] = _share(row["active_bytes"], active_total)

    family_bytes: dict[str, int] = {}
    family_storage: dict[str, int] = {}
    family_tensors: dict[str, int] = {}
    for organ in ORGAN_ORDER:
        fam = ORGAN_FAMILY_OF[organ]
        family_bytes[fam] = family_bytes.get(fam, 0) + by_organ[organ]["active_bytes"]
        family_storage[fam] = family_storage.get(fam, 0) + by_organ[organ]["storage_bytes"]
        family_tensors[fam] = family_tensors.get(fam, 0) + by_organ[organ]["n_tensors"]
    family_sum = sum(family_bytes.values())
    reconcile_active(
        family_sum,
        recorded,
        detail="organ-family rollup vs recorded measured total",
    )

    mlp_active = family_bytes["mlp"]
    mlp_elements = (
        int(geo["num_hidden_layers"])
        * 3
        * int(geo["intermediate_size"])
        * int(geo["hidden_size"])
    )
    code_bytes = (mlp_elements * AFFINE_CODE_BITS) // 8
    packing_overhead = mlp_active - code_bytes
    if packing_overhead < 0:
        raise CensusRefuse(
            f"MLP active {mlp_active} is smaller than 2-bit code bytes {code_bytes}"
        )

    layers_out: list[dict[str, Any]] = []
    for entry in per_layer:
        organs = [
            {
                "organ": organ,
                "n_tensors": slot["n_tensors"],
                "storage_bytes": slot["storage_bytes"],
                "active_bytes": slot["active_bytes"],
                "share_of_active": _share(slot["active_bytes"], active_total),
            }
            for organ, slot in sorted(entry["organs"].items())
        ]
        layer_active = sum(o["active_bytes"] for o in organs)
        layers_out.append(
            {
                "layer": entry["layer"],
                "kind": entry["kind"],
                "active_bytes": layer_active,
                "share_of_active": _share(layer_active, active_total),
                "organs": organs,
            }
        )

    by_organ_list = [by_organ[name] for name in ORGAN_ORDER if by_organ[name]["n_tensors"]]
    missing_declared = [name for name in ORGAN_ORDER if by_organ[name]["n_tensors"] == 0]
    # Linear-attn organs are absent on full-attention-only bodies; on this
    # hybrid they must be present. Absence of a declared organ that the
    # catalog actually uses would already have unclassified-refused.
    families_out = [
        {
            "family": fam,
            "n_tensors": family_tensors[fam],
            "storage_bytes": family_storage[fam],
            "active_bytes": family_bytes[fam],
            "share_of_active": _share(family_bytes[fam], active_total),
        }
        for fam in ("mlp", "attention", "norms", "embedding", "lm_head", "state")
    ]

    identity = {
        "resident_identity": sealed_doc.get("resident_identity"),
        "artifact_root": str(artifact),
        "catalog": str(artifact / CATALOG_NAME),
        "sealed_profile": SEALED_REL,
        "model_id": sealed_doc.get("model_id"),
        "physical_ebpw_recorded": sealed_doc.get("physical_ebpw"),
        "geometry": {
            "hidden_size": geo["hidden_size"],
            "intermediate_size": geo["intermediate_size"],
            "num_hidden_layers": geo["num_hidden_layers"],
            "vocab_size": geo["vocab_size"],
            "layer_types_counts": _count_kinds(geo.get("layer_types")),
            "full_attention_interval": geo.get("full_attention_interval"),
        },
    }

    return {
        "identity": identity,
        "catalog_total_bytes": catalog_total,
        "n_tensors": len(records),
        "recorded_active_weight_bytes_per_token": recorded,
        "active_weight_bytes_per_token": active_total,
        "unread_embedding_table_bytes": unread_embed,
        "reconciliation": {
            "active_plus_unread_embed_equals_catalog": accounted_storage == catalog_total,
            "active_equals_recorded": active_total == recorded,
            "organ_family_sum_equals_recorded": family_sum == recorded,
            "unclassified_tensors": 0,
            "declared_organs_with_zero_tensors": missing_declared,
        },
        "mlp": {
            "active_bytes": mlp_active,
            "share_of_active": _share(mlp_active, active_total),
            "n_tensors": family_tensors["mlp"],
            "n_parameters": mlp_elements,
            "incumbent_packing": {
                "family": "affine_q2_group64_ls",
                "kernel": "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
                "code_bits": AFFINE_CODE_BITS,
                "code_bytes": code_bytes,
                "scale_bias_and_header_bytes": packing_overhead,
                "code_share_of_mlp": _share(code_bytes, mlp_active),
                "overhead_share_of_mlp": _share(packing_overhead, mlp_active),
                "derived_bpw": 8.0 * mlp_active / mlp_elements,
                "note": (
                    "code bytes are 2 bits per SwiGLU affine parameter. "
                    "The remaining MLP bytes are group scale/bias plus per-tensor "
                    "headers of the present packing, not extra independent function."
                ),
            },
            "independent_function": {
                "form": "F_l(x) = down_l(silu(gate_l(x)) * up_l(x))",
                "domain": f"R^{geo['hidden_size']}",
                "codomain": f"R^{geo['hidden_size']}",
                "n_layers": n_layers,
                "gate_up_shape": [geo["intermediate_size"], geo["hidden_size"]],
                "down_shape": [geo["hidden_size"], geo["intermediate_size"]],
                "what_the_bytes_are": (
                    "A packing of 64 independent (gate, up, down) affine maps, "
                    "not a lower bound on the information of F. Function-space "
                    "rank is a different quantity from these stored bytes."
                ),
            },
        },
        "by_organ": by_organ_list,
        "by_organ_family": families_out,
        "per_layer": layers_out,
        "globals": [globals_organs[k] for k in sorted(globals_organs)],
        "state_not_in_catalog": {
            "kv_cache_bytes_per_token": "UNKNOWN",
            "deltanet_recurrent_state_bytes_per_token": "UNKNOWN",
            "activation_bytes_per_token": "UNKNOWN",
            "reason": (
                "HQ38M20 describes packed weights only. A_log and dt_bias are "
                "catalogued DeltaNet parameters and ARE counted under state.*. "
                "The recurrent state tensor and the KV cache are not."
            ),
        },
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "tensors": tensors,
    }


def _count_kinds(layer_types: Any) -> dict[str, int] | None:
    if not isinstance(layer_types, list):
        return None
    out: dict[str, int] = {}
    for item in layer_types:
        key = str(item)
        out[key] = out.get(key, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Representation families. Elimination, not compression-of-the-same-W.
# ---------------------------------------------------------------------------


def _remat_tag(decompresses: bool, ordinary: bool) -> str:
    verdict = judge_dense_rematerialization(
        {
            "path_kind": PRODUCTION,
            "dense_rematerialization": decompresses,
            "decompresses_to_dense_weight_tensor": decompresses,
            "runs_ordinary_kernels": ordinary,
            "consumes_representation_directly": (not decompresses),
        }
    )
    if not verdict.ok and decompresses:
        return REJECTED_DENSE_REMAT
    if decompresses:
        return REJECTED_DENSE_REMAT
    return DIRECT_CONSUME


def _load_noetic() -> tuple[dict[str, Any] | None, str]:
    for rel in NOETIC_RELS:
        doc, via = _load_json_rel(rel)
        if doc is not None:
            return doc, f"{rel} ({via})"
    return None, "missing"


def _noetic_entry(doc: Mapping[str, Any] | None, nns_id: str) -> dict[str, Any] | None:
    if not isinstance(doc, Mapping):
        return None
    for entry in doc.get("entries") or []:
        if isinstance(entry, Mapping) and str(entry.get("id") or "") == nns_id:
            return dict(entry)
    return None


def _cite(
    *,
    scar_id: str,
    source_path: str,
    claim_refuted: str,
    reopen: str,
    surface: str,
    kind: str | None = None,
) -> dict[str, Any]:
    return {
        "scar_id": scar_id,
        "source_path": source_path,
        "claim_refuted": claim_refuted,
        "reopen_condition": reopen,
        "surface": surface,
        "kind": kind,
        "this_specimen": "qwen3.8-27b sealed-3.14 dense SwiGLU MLP",
    }


def _source_mentions(rel: str, token: str) -> bool:
    text, _via = _read_rel(rel)
    return bool(text) and token in text


def _flash_rival_surface() -> dict[str, Any]:
    doc, via = _load_json_rel(RIVAL_REL)
    families: list[str] = []
    any_pass = None
    if isinstance(doc, Mapping):
        any_pass = doc.get("any_family_passed_contract")
        for row in doc.get("families") or []:
            if isinstance(row, Mapping) and row.get("family"):
                families.append(str(row["family"]))
    return {
        "receipt": RIVAL_REL,
        "resolved": via,
        "specimen": "Qwen/Qwen3.8-Flash-Next layer_4.routed_experts.gate_up_proj",
        "not_this_specimen": True,
        "any_family_passed_contract": any_pass,
        "killed_family_ids": families,
        "use": (
            "Scoped scar on Flash MoE gate_up, not a kill of the same family "
            "on this dense 27B SwiGLU. Cited so a cousin is not re-proposed "
            "as if untested on Flash; it is not ALREADY_FALSIFIED here."
        ),
    }


def _index_hits(family_slugs: Sequence[str]) -> list[dict[str, Any]]:
    try:
        from tools.future import negative_index as ni
    except Exception as exc:  # pragma: no cover - index is in this package
        return [{"index_error": f"{type(exc).__name__}: {exc}"}]
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for slug in family_slugs:
        for organ in ("mlp", "gate", "up", "down"):
            refusal = ni.refuse_if_dead(
                {
                    "model": "qwen3.8-27b",
                    "organ": organ,
                    "hypothesis_family": slug,
                }
            )
            if not refusal:
                continue
            key = str(refusal.get("scar_id") or "")
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                {
                    "scar_id": refusal.get("scar_id"),
                    "source_path": refusal.get("source_path"),
                    "hypothesis_family": refusal.get("hypothesis_family"),
                    "organ": refusal.get("organ"),
                    "verdict": refusal.get("verdict"),
                    "claim_refuted": refusal.get("claim_refuted"),
                    "reopen_condition": refusal.get("reopen_condition"),
                    "queried_slug": slug,
                    "queried_organ": organ,
                }
            )
    return hits


def representation_families(
    snap: Mapping[str, Any] | None = None,
    *,
    consult_index: bool = True,
) -> list[dict[str, Any]]:
    """Elimination families for THIS MLP. Scars on other surfaces are cousins."""
    snap = snap if snap is not None else census()
    mlp = snap["mlp"]
    b_mlp = int(mlp["active_bytes"])
    p_mlp = int(mlp["n_parameters"])
    bpw = float(mlp["incumbent_packing"]["derived_bpw"])
    hidden = int(snap["identity"]["geometry"]["hidden_size"])
    inter = int(snap["identity"]["geometry"]["intermediate_size"])
    n_layers = int(snap["identity"]["geometry"]["num_hidden_layers"])
    noetic, noetic_via = _load_noetic()
    flash = _flash_rival_surface()

    def nns(nns_id: str) -> dict[str, Any]:
        entry = _noetic_entry(noetic, nns_id) or {}
        src = (
            entry.get("id") and noetic_via.split(" (")[0]
        ) or NOETIC_RELS[0]
        scope = entry.get("scope") if isinstance(entry.get("scope"), dict) else {}
        model = str(scope.get("model") or "").strip()
        organ = str(scope.get("organ") or "").strip()
        surface = " ".join(p for p in (model, organ) if p) or "as recorded in NOETIC_NEGATIVE_SCIENCE"
        return _cite(
            scar_id=nns_id,
            source_path=str(src),
            claim_refuted=str(entry.get("claim_refuted") or entry.get("capability") or ""),
            reopen=str(entry.get("reopen_condition") or ""),
            surface=surface,
            kind=str(entry.get("kind") or ""),
        )

    def qn(qn_id: str, claim: str, reopen: str) -> dict[str, Any]:
        return _cite(
            scar_id=qn_id,
            source_path=QN_REL,
            claim_refuted=claim,
            reopen=reopen,
            surface="qwen3.8-27b mlp_gate_up+mlp_down (QN catalog; abliterated sibling of this parent)",
            kind="MODEL_SPECIFIC",
        )

    qn_present = _source_mentions(QN_REL, "QN-SHARED-BASIS-DENSITY")
    nns_present = bool(noetic)

    rows: list[dict[str, Any]] = [
        {
            "id": "lower_bit",
            "name": "lower-bit",
            "mechanism": (
                "Replace the incumbent affine-Q2 group-64 least-squares codes "
                "(2 code bits + fp16 scale/bias per group) with fewer code bits "
                "on the same W: uniform Q2, Q1, binary sign-code, ternary."
            ),
            "byte_model": (
                f"active ≈ (bits/8)*{p_mlp} + group scale/bias. Incumbent is "
                f"{b_mlp} bytes at derived {bpw:.6f} bpw. Binary ~1.25 bpw body "
                f"would be ~{int(p_mlp * 1.25 / 8)} bytes if it were legal."
            ),
            "bytes_eliminated_if_true": (
                "Code bits below 2, and possibly scale/bias if a 1-bit code "
                "drops the affine term. Not a new function for F."
            ),
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "A native low-bit GEMV consumes codes in-register (QN-BINARY "
                "kernel was competent and fast). Unpack-to-Q4-then-generic-GEMV "
                "is a different lowering and is REJECTED_DENSE_REMAT "
                "(family low_bpw_materialize_w_expand_to_q4_float_generic_gemv)."
            ),
            "status": ALREADY_FALSIFIED,
            "cheapest_falsifier": (
                "Already run: uniform Q2 MLP output rel-fro 0.578 vs q3 0.198 "
                "(NNS-029); 1.25-bpw binary body generation-incoherent, 0/4 "
                "healers (QN-BINARY-INJURY). Retry is not a new experiment. "
                "Reopen only for a codec that hits the Shannon bound AND a "
                "generate gate, not another Q2/binary sweep."
            ),
            "citations": [
                nns("NNS-029"),
                qn(
                    "QN-BINARY-INJURY",
                    "the 1.25-bpw binary body is physically fast but generation-injured; 0 of 4 healing candidates reached coherent generation",
                    "a healing scheme that restores coherent generation while the healed body stays faster than q2f_g64",
                ),
                qn(
                    "QN-COORDINATE-TRANSFORM",
                    "tested rotation families did not materially move the Qwen MLP information floor; ~2.25 bpw held under coordinate change",
                    "a transform family not in the probe that moves the measured floor below 2.25",
                ),
            ],
            "index_slugs": ["uniform_q2", "binary_quantization", "ternary"],
        },
        {
            "id": "heterogeneous_precision",
            "name": "heterogeneous precision",
            "mechanism": (
                "Non-uniform bits across gate/up/down and across depth: spend "
                "scale/bias or extra bits only where a sensitivity map says F "
                "moves, crush the rest."
            ),
            "byte_model": (
                f"sum_organs bits_o * n_params_o / 8. Incumbent already mixes "
                f"Q2 MLP ({b_mlp}) with Q4 attention/lm_head; the open question "
                "is heterogeneous bits *inside* the MLP."
            ),
            "bytes_eliminated_if_true": (
                "A proper subset of the 64*3 affine tensors could drop below "
                "2.5 bpw if they are not capability-critical. That is elimination "
                "of packing, not of F."
            ),
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "Each organ keeps a native kernel (the incumbent already runs "
                "Q2 and Q4 side by side). A lowering that unpacks crushed "
                "organs to dense W is REJECTED_DENSE_REMAT."
            ),
            "status": OPEN,
            "cheapest_falsifier": (
                "STATIC: a bit map that is uniform across gate/up/down restates "
                "QN-BINARY-HEALING's broad injury (this MLP) and, on a different "
                "parent, NNS-018's single-family Q80 expert scar. CHEAP CPU: take "
                "real post-norm X on 2–4 layers, assign extra bits to the organ "
                "with the worst held-out rel-fro, and test whether the "
                "byte-weighted error beats uniform affine-Q2. If the extra bits "
                "have to cover all three organs, the map is not heterogeneous."
            ),
            "citations": [
                qn(
                    "QN-BINARY-HEALING",
                    "the injury is broad, not localized: no small protected island cheaply restored it; 0/4 candidates reached coherent generation",
                    "a sensitivity map that localizes the injury to a region small enough that protecting it costs less than the 2.25-bpw q2f body",
                ),
            ],
            "cousin_not_this_surface": [],
            "index_slugs": ["uniform_subbit_allocation", "protected_islands"],
            "note": (
                "uniform_subbit_allocation is dead as a *uniform* plan. A "
                "measured per-organ map is a different hypothesis and is OPEN. "
                "QN-BINARY-HEALING kills *binary-body islands*, not every mixed pack. "
                "NNS-018 (single family across Q80 expert organs) is a different parent."
            ),
        },
        {
            "id": "generated_weights",
            "name": "generated weights",
            "mechanism": (
                "Store a small generator G(θ, layer, organ) that emits W at "
                "use, instead of storing W. Elimination of independent storage "
                "of the 64*3 affines."
            ),
            "byte_model": (
                f"|θ| + generator program, independent of {p_mlp}. A win "
                f"requires |θ| << {b_mlp} AND production that never writes W."
            ),
            "bytes_eliminated_if_true": (
                "Almost all of the MLP active bytes, if G is the kernel. If G "
                "materializes W then runs the incumbent GEMV, zero bytes are "
                "eliminated at execution."
            ),
            "dense_rematerialization": REJECTED_DENSE_REMAT,
            "dense_rematerialization_reason": (
                "The cheap lowering is generate-then-ordinary-GEMV. That is "
                "dense rematerialization of W and is refused as a production "
                "path. A generator that is itself the matvec (no W) is a "
                "different family (function_replacement)."
            ),
            "status": OPEN,
            "cheapest_falsifier": (
                "STATIC: any plan whose native_execution_concept is 'emit W, "
                "then qwen_affine_q2 / generic GEMV' is REJECTED_DENSE_REMAT "
                "before a fit. CHEAP CPU: a per-layer hypernetwork fit on 2 "
                "layers whose |θ| < incumbent bytes and whose held-out rel-fro "
                "beats affine-Q2; if it only wins after writing W, it is still "
                "dead as an active-byte family."
            ),
            "citations": [nns("NNS-015")],
            "index_slugs": ["generated_tied_params"],
            "note": (
                "NNS-015's reopen is a distilled *operator* matching F, which "
                "this catalog files under function_replacement. Generated W "
                "that remats is not that reopen."
            ),
        },
        {
            "id": "shared_bases",
            "name": "shared bases",
            "mechanism": (
                "One (or K) shared weight-space basis B with local coefficients "
                "C_l,o so W_{l,o} ≈ B C_{l,o} (or y = C (B x)). Unconditioned "
                "sharing across the MLP tensors."
            ),
            "byte_model": (
                f"|B| + sum |C|. Incumbent {b_mlp}. A K=2 shared basis was "
                "measured at ~0.53 local bpw plus correction (QN-SHARED-K-HYBRID) "
                "and still failed held-out activation."
            ),
            "bytes_eliminated_if_true": "Independent copies of the same basis across layers/organs.",
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "Fused shared-basis matvec is a direct consumer (QN-SHARED-BASIS "
                "kernel cut dispatches 384→192 and the byte win *did* translate "
                "to nanoseconds). The kill was held-out function, not remat."
            ),
            "status": ALREADY_FALSIFIED,
            "cheapest_falsifier": (
                "Already run on this MLP: no shared-basis K below ~2.25 bpw is "
                "coherent at held-out activation (QN-SHARED-BASIS-DENSITY, "
                "QN-SHARED-K-HYBRID). Reopen only for a point that is coherent "
                "AND beats affine-Q2 on density AND complete-token time."
            ),
            "citations": [
                qn(
                    "QN-SHARED-BASIS-DENSITY",
                    "the KERNEL is competent and the byte win does translate to nanoseconds, but no K below ~2.25 bpw composes coherently for the MLP: the local functional probe dies at held-out activation",
                    "a shared-basis point that is coherent at held-out activation AND beats q2f on both density and COMPLETE_TOKEN_NS",
                ),
                qn(
                    "QN-SHARED-K-HYBRID",
                    "shared K=2 costs 0.531 extra bpw and still fails to restore held-out activations; the hybrid remained slower and incoherent",
                    "a shared-K variant that is coherent on held-out X at a total body bpw below 2.25",
                ),
            ],
            "index_slugs": ["shared_basis", "qn_shared_k_hybrid"],
            "qn_catalog_reachable": qn_present,
        },
        {
            "id": "factorized_programs",
            "name": "factorized programs",
            "mechanism": (
                "W ≈ U V (SVD / TT / Kronecker / two skinny matvecs) per "
                "tensor or per organ. y = U (V x) with rank r << min(m, n)."
            ),
            "byte_model": (
                f"per organ, r*(m+n)*bytes_per. Gate/up are {inter}×{hidden}; "
                f"down is {hidden}×{inter}; {n_layers} layers. Weight-space "
                "99% energy needs 92–95% of ranks on this body (NNS-016), so "
                "r that preserves F is not a byte win."
            ),
            "bytes_eliminated_if_true": (
                "Near-zero: the spectrum is not low-rank at the coherent point. "
                "Function-space rank is real (NNS-014, 99% energy at 56% ranks) "
                "and still lost to q3 held-out at matched bytes."
            ),
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "Two skinny matvecs consume the factors directly. Materializing "
                "U@V into W then running the incumbent GEMV is REJECTED_DENSE_REMAT."
            ),
            "status": ALREADY_FALSIFIED,
            "cheapest_falsifier": (
                "Already run: Van Loan / SVD energy on this parent (NNS-016); "
                "activation-aware output-PCA matched-byte held-out loss to q3 "
                "(NNS-014); low-rank correction under a 1.0 bpw budget "
                "(QN-LOWRANK-HEALING, even r=256 at +1.035 bpw, rel_fro 0.4798). "
                "Cheapest new probe is not another rank sweep: it is whether a "
                "*hybrid* (low-rank prefix + exact residual) actually drops "
                "active bytes, which is sparse_residuals, not this family."
            ),
            "citations": [
                nns("NNS-014"),
                nns("NNS-016"),
                qn(
                    "QN-LOWRANK-HEALING",
                    "no distributed correction under the 1.0 bpw budget restored held-out activations on real X; even r=256 at 1.035 extra bpw pushed the body to 2.285 > 2.25 with rel_fro 0.4798",
                    "a correction whose extra bpw keeps the body under 2.25 while rel_fro on real held-out X drops below the q2f baseline",
                ),
            ],
            "cousin_not_this_surface": [flash],
            "index_slugs": ["low_rank", "kronecker", "global_dense_lowrank"],
        },
        {
            "id": "dictionary_programs",
            "name": "dictionary programs",
            "mechanism": (
                "Replace blocks of frozen W with codebook indices + codewords "
                "(VQ). Execution looks up codewords, does not store W."
            ),
            "byte_model": (
                "n_blocks * ceil(log2(K))/8 + K * block_bytes. Post-hoc on "
                f"frozen W at ~1 bpw was complete 1.0075 BPW and collapsed 6/6 "
                "(NNS-017 A1_1p0)."
            ),
            "bytes_eliminated_if_true": (
                "The 2-bit affine codes, if blocks of W are a small dictionary. "
                "NNS-017: they are not, on raw frozen weights."
            ),
            "dense_rematerialization": REJECTED_DENSE_REMAT,
            "dense_rematerialization_reason": (
                "The cheap post-hoc lowering expands indices into a dense W "
                "(or into Q4) and runs an ordinary GEMV. That is "
                "low_bpw_materialize_w_expand_to_q4_float_generic_gemv. A "
                "native codebook-in-register kernel is a different lowering "
                "and is still blocked on this surface by NNS-017's capability kill."
            ),
            "status": ALREADY_FALSIFIED,
            "cheapest_falsifier": (
                "Already run: PQ/VQ of RAW frozen weights at ~1 bpw and at "
                "0.49 bpw collapsed 6/6 prompts (NNS-017), organ inversion "
                "applied. Reopen is never on raw frozen W; a method that "
                "CHANGES the source is not this family."
            ),
            "citations": [nns("NNS-017"), nns("NNS-010")],
            "cousin_not_this_surface": [flash],
            "index_slugs": ["raw_weight_pq_vq", "post_hoc_frozen_codec", "learned_codebook"],
        },
        {
            "id": "product_codebook_programs",
            "name": "product-codebook programs",
            "mechanism": (
                "Product VQ / residual PQ: a block is a tuple of codebook "
                "indices, reconstructed as a sum of codewords. Same frozen-W "
                "object as dictionary_programs, factored across subspaces."
            ),
            "byte_model": (
                "M * (n_blocks * ceil(log2(K))/8 + K * d/M). Lloyd-optimal "
                "indices are near-uniform so entropy coding of the indices "
                "is not an active-byte lever (NNS-022)."
            ),
            "bytes_eliminated_if_true": "Same claim as dictionary_programs, split across M codebooks.",
            "dense_rematerialization": REJECTED_DENSE_REMAT,
            "dense_rematerialization_reason": (
                "Same cheap lowering as dictionary_programs: expand product "
                "codes to dense W, then ordinary GEMV."
            ),
            "status": ALREADY_FALSIFIED,
            "cheapest_falsifier": (
                "NNS-017 already killed raw-weight PQ/VQ including residual "
                "sub-half (R2_subhalf 0.4930 bpw, 6/6 collapse). NNS-022 "
                "kills rANS-on-indices as an active-byte lever on this parent. "
                "A new product-codebook on frozen W is a restatement."
            ),
            "citations": [nns("NNS-017"), nns("NNS-022")],
            "cousin_not_this_surface": [flash],
            "index_slugs": ["raw_weight_pq_vq", "entropy_coded_pq"],
        },
        {
            "id": "sparse_residuals",
            "name": "sparse residuals",
            "mechanism": (
                "Cheap backbone (incumbent affine-Q2 or cheaper) plus a sparse "
                "or low-rank residual that restores F. y = backbone(x) + R(x)."
            ),
            "byte_model": (
                f"backbone_bytes + nnz*(index+value) or r*(m+n). NNS-015: "
                "q3+rank64 correction GENERALIZES and beats q3 quality but "
                "ADDS bytes (107% of q3) — quality lever, not elimination. "
                "QN-LOWRANK-HEALING: under a 1.0 extra-bpw budget, held-out "
                "does not return."
            ),
            "bytes_eliminated_if_true": (
                "Only if backbone+residual < incumbent 2.5 bpw at matched "
                "held-out. Measured residuals add bytes or miss quality."
            ),
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "Residual applied in the same kernel is a direct consumer. "
                "Reconstructing W_backbone + W_R as dense W is REJECTED_DENSE_REMAT."
            ),
            "status": ALREADY_FALSIFIED,
            "cheapest_falsifier": (
                "Already run as q2/q3 + low-rank correction (NNS-015, "
                "QN-LOWRANK-HEALING). Cheapest new probe: a residual whose "
                "*net* active bytes are below incumbent AND held-out rel-fro "
                "beats affine-Q2. A quality-only residual is not this family."
            ),
            "citations": [nns("NNS-015"), nns("NNS-014")],
            "cousin_not_this_surface": [flash],
            "index_slugs": ["low_rank", "residual_codebook"],
        },
        {
            "id": "cross_layer_prediction",
            "name": "cross-layer prediction",
            "mechanism": (
                "W_l = P(W_{l-1}) + Δ_l (or generate layer l from layer 0). "
                "Store the predictor and the residuals, not 64 independent copies."
            ),
            "byte_model": (
                f"W_0 + sum |Δ_l| + |P|. A win requires mean ||Δ|| << ||W||. "
                f"NNS-016: layers ≥ 1 are near-full-rank on this parent; a "
                "cross-layer delta is not a free compression of a full-rank map."
            ),
            "bytes_eliminated_if_true": "Repeated depth structure in the 64 SwiGLU maps.",
            "dense_rematerialization": REJECTED_DENSE_REMAT,
            "dense_rematerialization_reason": (
                "The cheap lowering predicts W_l into a dense buffer then runs "
                "the incumbent GEMV. Direct consume would apply P in activation "
                "space (y_l = f_P(y_{l-1}) + Δ_l(x)), which is function_replacement."
            ),
            "status": OPEN,
            "cheapest_falsifier": (
                "CHEAP CPU / STATIC: cosine and relative-Frobenius of W_l vs "
                "W_{l-1} (and vs P=identity, P=scale, P=linear) on gate/up/down "
                "separately. If residual norm ≈ parent on layers ≥ 1, the family "
                "dies on this spectrum without a fit. Do not transfer Q80 "
                "cross-expert tying (different surface, already dead there)."
            ),
            "citations": [nns("NNS-016")],
            "index_slugs": ["cross_layer_weight_delta", "cross_expert_structure"],
            "note": (
                "cross_expert_structure is a MoE scar (Q80/Flash). This body "
                "has no routed experts; that scar does not prune cross-layer "
                "prediction of a dense SwiGLU. NNS-016's full-rank depth is "
                "the relevant cousin, not a completed kill of a predictor P≠id."
            ),
        },
        {
            "id": "routed_subprograms",
            "name": "routed subprograms",
            "mechanism": (
                "Token-dependent subset of the 17408 SwiGLU hidden: a router "
                "picks k columns of gate/up and the matching down rows. "
                "Turns a dense MLP into a small MoE over its own intermediate."
            ),
            "byte_model": (
                f"active ≈ (k/{inter})*{b_mlp} + router. Unstructured "
                "activation sparsity was measured at ~2× MLP max on this "
                "body (NNS-029) and is Doctor-risky over 64 layers."
            ),
            "bytes_eliminated_if_true": (
                "The unread slice of gate/up/down per token. Storage does not "
                "drop unless unused columns are also dropped from the artifact."
            ),
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "Gathered columns consumed by the existing affine kernel (or a "
                "sparse cousin). Expanding skipped columns back to dense W is "
                "not this family."
            ),
            "status": ALREADY_FALSIFIED,
            "cheapest_falsifier": (
                "NNS-029 already killed unstructured / activation-sparsity as a "
                "clean path (compounds over 64 layers; MLP is DENSE, audit D2). "
                "Reopen is a Doctor-holding skip pattern that does not compound, "
                "not another magnitude-threshold skip. A *trained* partition of "
                "the 17408 into experts is closer to function_replacement / "
                "NNS-013 (narrow m) and is not this skip family."
            ),
            "citations": [nns("NNS-029"), nns("NNS-013")],
            "index_slugs": [],
        },
        {
            "id": "shared_input_transforms",
            "name": "shared input transforms",
            "mechanism": (
                "One shared V on the MLP input, organ- or layer-local readout: "
                "y_{l,o} = W'_{l,o} (V x). Activation-space sharing, not a "
                "shared weight-space basis of W (that is shared_bases)."
            ),
            "byte_model": (
                f"|V| ({hidden}×r) + 64*3 of W' with inner dim r. Byte win "
                f"iff r << {hidden} AND the readouts stay cheaper than "
                f"incumbent {b_mlp}."
            ),
            "bytes_eliminated_if_true": (
                "Repeated input maps across depth. Distinct from QN-SHARED-BASIS "
                "(weight-space B, C on this MLP) and from Flash "
                "shared_input_latent_plus_expert_local_output_readout (MoE experts)."
            ),
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "y = W'(V x) is two matvecs. Materializing W'V into W is "
                "REJECTED_DENSE_REMAT and would also erase the sharing."
            ),
            "status": OPEN,
            "cheapest_falsifier": (
                "CHEAP CPU: PCA / ridge V on real post-norm X pooled across a "
                "few layers; reconstruct gate and up held-out. If the r that "
                "meets incumbent rel-fro is ≈ hidden, there is no shared input "
                "to store once. Do not cite the Flash L4 rival-codec screen as "
                "a kill: that surface is routed experts on Flash-Next, 0/5 "
                "ranks passed, different specimen."
            ),
            "citations": [],
            "cousin_not_this_surface": [flash],
            "index_slugs": ["shared_basis"],
            "note": (
                "Index slug shared_basis will fire QN-SHARED-BASIS on organ=down. "
                "That scar is unconditioned weight-space sharing, which "
                "expert_bank_school explicitly splits from a one-sided / "
                "activation-PCA transform. Marked OPEN, with the cousin named."
            ),
        },
        {
            "id": "latent_accumulation",
            "name": "latent accumulation",
            "mechanism": (
                "Accumulate the SwiGLU in a latent of width m << 17408, then "
                "expand once (grouped / shared / bottleneck SwiGLU). "
                "F(x) ≈ down_m(silu(gate_m(x)) * up_m(x)) with m < intermediate."
            ),
            "byte_model": (
                f"64 * (2*m*{hidden} + {hidden}*m) * incumbent_bpw/8. "
                f"NNS-013: matching q3's 0.337 held-out needs m ~ 10000–12000, "
                f"at which active bytes approach q3 — no Pareto win. Honest "
                "single-op plateaus rel 0.59 in-family, 0.95 cross-family."
            ),
            "bytes_eliminated_if_true": "The unused slice of the 17408 intermediate.",
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "A narrower SwiGLU is a direct kernel. Expanding the latent to "
                "a 17408-wide dense W each token is REJECTED_DENSE_REMAT and "
                "cancels the byte win."
            ),
            "status": ALREADY_FALSIFIED,
            "cheapest_falsifier": (
                "Already run as G3 / narrow shared grouped SwiGLU (NNS-012 "
                "method bugs, then NNS-013 property kill after those bugs were "
                "removed). Reopen is a full-width structured nonlinear "
                "(Monarch/butterfly), which is function_replacement, not a "
                "retry of m<17408."
            ),
            "citations": [nns("NNS-013"), nns("NNS-012")],
            "index_slugs": [],
            "noetic_reachable": nns_present,
        },
        {
            "id": "function_replacement",
            "name": "function replacement",
            "mechanism": (
                "Stop representing W. Represent F_l itself with a cheaper "
                "program: full-width structured nonlinear (Monarch, butterfly), "
                "a distilled small net, a kernel not equal to three affines. "
                "Narrow bottleneck replacement is latent_accumulation (dead)."
            ),
            "byte_model": (
                f"|program_l| * {n_layers}, independent of {p_mlp}. A win is "
                f"|program| << {b_mlp} at held-out F and at generate."
            ),
            "bytes_eliminated_if_true": (
                "All 5.35e9 MLP bytes that exist because F is currently three "
                "affine maps. Those bytes are a representation of F, not F."
            ),
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "The program is the kernel. A replacement that emits W and "
                "runs GEMV is generated_weights and REJECTED_DENSE_REMAT."
            ),
            "status": OPEN,
            "cheapest_falsifier": (
                "Do not retry m<17408 (NNS-013). CHEAP CPU: one full-width "
                "Monarch/butterfly (or a distilled operator) on a single layer's "
                "real post-norm X, held-out rel-fro vs affine-Q2, with a byte "
                "ledger that does not remat W. If it cannot beat affine-Q2 on "
                "that layer, the family dies cheaply. NNS-013's reopen is "
                "exactly this probe, and it has not been run."
            ),
            "citations": [nns("NNS-013"), nns("NNS-012"), nns("NNS-015")],
            "index_slugs": [],
            "note": (
                "Narrow replacement is ALREADY_FALSIFIED (see latent_accumulation). "
                "This row is the full-width / distilled-operator reopen, kept OPEN."
            ),
        },
        {
            "id": "capability_sensitive_literal_islands",
            "name": "capability-sensitive literal islands",
            "mechanism": (
                "Keep a sensitivity-selected subset of rows/layers/organs as "
                "literal / high-precision tensors; crush the rest. The island "
                "is the capability, the bulk is packing."
            ),
            "byte_model": (
                f"island_frac * full_prec + (1-island_frac) * bulk. A win "
                f"requires island_frac small enough that island+bulk < {b_mlp} "
                "and generation holds."
            ),
            "bytes_eliminated_if_true": "The non-island majority of affine-Q2 codes.",
            "dense_rematerialization": DIRECT_CONSUME,
            "dense_rematerialization_reason": (
                "Island and bulk each have a native kernel (incumbent already "
                "mixes Q2/Q4/f32). Healing that remats the binary body to "
                "dense W around the island is REJECTED_DENSE_REMAT."
            ),
            "status": ALREADY_FALSIFIED,
            "cheapest_falsifier": (
                "Already run as binary-body + high-precision islands "
                "(QN-BINARY-HEALING): injury is broad, not localized; 0/4 "
                "reached coherent generation. Reopen needs a sensitivity map "
                "that localizes injury to a region cheaper than just using the "
                "affine-Q2 body. A map that protects most of the MLP is not "
                "an island."
            ),
            "citations": [
                qn(
                    "QN-BINARY-HEALING",
                    "the injury is broad, not localized: no small protected island cheaply restored it; 0/4 candidates reached coherent generation",
                    "a sensitivity map that localizes the injury to a region small enough that protecting it costs less than the 2.25-bpw q2f body",
                ),
                qn(
                    "QN-BINARY-INJURY",
                    "the 1.25-bpw binary body is physically fast but generation-injured; 0 of 4 healing candidates reached coherent generation",
                    "a healing scheme that restores coherent generation while the healed body stays faster than q2f_g64",
                ),
            ],
            "index_slugs": ["protected_islands", "qn_binary_healing"],
        },
    ]

    # Confirm remat tags against the category validator for the REJECTED rows.
    for row in rows:
        if row["dense_rematerialization"] == REJECTED_DENSE_REMAT:
            tag = _remat_tag(True, True)
            if tag != REJECTED_DENSE_REMAT:
                raise CensusRefuse(
                    f"{row['id']}: expected REJECTED_DENSE_REMAT from "
                    f"judge_dense_rematerialization, got {tag}"
                )
        row["evidence_class"] = "STATIC_ONLY"
        row["gpu_authority"] = False
        if consult_index:
            slugs = list(row.get("index_slugs") or [])
            if slugs:
                row["index_refusals"] = _index_hits(slugs)
            else:
                row["index_refusals"] = []
        # A family marked OPEN must not silently carry an exact this-surface
        # index refusal without saying so. shared_input_transforms is the
        # documented cousin exception.
        if (
            row["status"] == OPEN
            and row["id"] != "shared_input_transforms"
            and consult_index
        ):
            exact = [
                h
                for h in row.get("index_refusals") or []
                if h.get("scar_id") and "index_error" not in h
            ]
            row["open_with_index_hits"] = exact

    have = [r["id"] for r in rows]
    if have != list(REQUIRED_FAMILY_IDS):
        raise CensusRefuse(
            f"family catalog {have} != required {list(REQUIRED_FAMILY_IDS)}"
        )
    return rows


# ---------------------------------------------------------------------------
# Receipt.
# ---------------------------------------------------------------------------


def _public_census(snap: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the per-tensor dump from the receipt; keep the load-bearing tables."""
    out = {k: v for k, v in snap.items() if k != "tensors"}
    return out


def build() -> Path:
    sealed = load_sealed()
    snap = census(sealed=sealed)
    families = representation_families(snap, consult_index=True)
    n_dead = sum(1 for f in families if f["status"] == ALREADY_FALSIFIED)
    n_remat = sum(1 for f in families if f["dense_rematerialization"] == REJECTED_DENSE_REMAT)
    n_open = sum(1 for f in families if f["status"] == OPEN)
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Exact per-organ active-byte census of sealed-3.14 for one decoded "
            "token, and an elimination-family catalog for the MLP that cites "
            "existing negative science instead of re-proposing it."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "what_this_does_not_prove": [
            "physical EBPW of a different packing",
            "capability of any family",
            "actual_read_bytes_per_token (cache, contention)",
            "KV / DeltaNet recurrent traffic",
            "that function-space rank is an active-byte win (NNS-014: it was not)",
        ],
        "census": _public_census(snap),
        "mlp_share_of_active": snap["mlp"]["share_of_active"],
        "mlp_active_bytes": snap["mlp"]["active_bytes"],
        "families": families,
        "family_counts": {
            "n": len(families),
            "already_falsified": n_dead,
            "open": n_open,
            "rejected_dense_remat": n_remat,
        },
        "recovered_implementation": {
            "catalog_format": "HQ38M20 (same layout as tools/accelerator/bytes_atlas.py)",
            "sealed_profile": SEALED_REL,
            "artifact_root": snap["identity"]["artifact_root"],
            "negative_science": [NEGATIVE_INDEX_REL, *NOETIC_RELS, QN_REL, RIVAL_REL],
        },
        "gaps_closed": [
            "per-organ active bytes summed from the real catalog, not copied from a percentage",
            "reconciliation against the recorded 9,878,901,136 active-weight total, raising on mismatch",
            "embedding counted as one Q4 row, not the vocab table",
            "MLP incumbent packing split into 2-bit codes vs scale/bias headers, derived",
            "fourteen elimination families with remat tags and cheapest falsifiers",
            "this-surface scars (QN / NNS) marked ALREADY_FALSIFIED; Flash rival-codec scoped as a different specimen",
        ],
        "negative_findings": [
            "hcli/ is not materialized in this sparse checkout; sealed profile is git HEAD or disk",
            "tools/accelerator/bytes_atlas.py is not materialized; parser is duplicated here, not imported",
            "KV cache, DeltaNet recurrent state, and activations remain UNKNOWN",
            "QN catalog was measured on qwen3.8-27b-abliterated, the same parent geometry as sealed-3.14",
            "refuse_if_dead organ slugs often resolve mlp_gate_up+mlp_down to 'down'; citations therefore also name QN/NNS ids directly",
        ],
        "nomenclature": {
            "already_falsified": ALREADY_FALSIFIED,
            "rejected_dense_remat": REJECTED_DENSE_REMAT,
            "direct_consume": DIRECT_CONSUME,
            "open": OPEN,
            "static_only": "this sidecar. Models propose; protected deterministic evidence decides.",
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


selftest = build


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--census-only", action="store_true")
    args = parser.parse_args(argv_list)
    if args.census_only:
        snap = census()
        public = _public_census(snap)
        json.dump(
            {
                "active_weight_bytes_per_token": public["active_weight_bytes_per_token"],
                "mlp_share_of_active": public["mlp"]["share_of_active"],
                "mlp_active_bytes": public["mlp"]["active_bytes"],
                "by_organ_family": public["by_organ_family"],
                "reconciliation": public["reconciliation"],
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0
    if args.build or args.selftest or not argv_list:
        out = build()
        print(out)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(_sys.argv[1:]))
