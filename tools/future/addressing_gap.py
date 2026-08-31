"""ADDRESS THE WEIGHT-ADDRESSING GAP — attribute 703→337 without measuring hardware.

The silicon already streams a clean Q4 GEMV near published peak. Production
decode does not. This module rereads the receipts that measured that drop,
refuses unsourced rungs, keeps roofs named, and ranks what is removable.

It does not take a GPU lease, flock a bench lock, re-derive a GB/s figure,
or close an unattributed remainder so the arithmetic looks finished. A
ceiling with an unstated roof is the defect that produced 595.9.

    python3 tools/future/addressing_gap.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import time
from typing import Any, Mapping, Sequence

from tools.future._common import REPO, git, load_json, write_receipt

RECEIPT = "ADDRESSING_GAP.json"
SCHEMA = "hawking.future.addressing_gap.v1"
VERSION = 1
RECORDED_BY = "tools/future/addressing_gap.py"

# Lane-named paths. Absence is a refusal, never a silent substitute.
LANE_HONEST = "receipts/headless/HONEST_ROOF_WEIGHT_ADDRESSING.json"
LANE_HONEST_REDUCED = "receipts/headless/HONEST_ROOF_WEIGHT_ADDRESSING.reduced.json"
LANE_G044 = "receipts/headless/G044_ROOFLINE_KNEE.json"
ATLAS_REL = "receipts/headless/ACCELERATOR_TOKEN_BYTES_ATLAS.json"
ASCENT_REL = "receipts/headless/BANDWIDTH_ASCENT.json"
CENSUS_REL = "receipts/headless/NOETIC_ORGAN_CENSUS.json"
# Where those three actually live in HEAD (sparse-absent on disk is not absence).
ASCENT_HONEST = "receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.json"
ASCENT_HONEST_REDUCED = "receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.reduced.json"
ASCENT_G044 = "receipts/ascent-2026-08-16/G044_ROOFLINE_KNEE.json"
G072_REL = "receipts/ascent-2026-08-16/G072_MULTI_PLANE_GEMV.json"
GENOME_REL = "receipts/headless/MACHINE_GENOME.json"
LEDGER_REL = "receipts/headless/ORGAN_ROOF_LEDGER.json"
ORGAN_BW_REL = "receipts/headless/ORGAN_BANDWIDTH.json"
UNPACK_REL = "receipts/headless/ACCELERATOR_UNPACK_IS_THE_WALL.json"
CANON_REL = "docs/ultragoals/NOETIC_CANON.md"

# Lane-claimed rungs. Each is adjudicated against a sourced statistic.
LANE_CLAIMED_PUBLISHED = 819.0
LANE_CLAIMED_SINGLE_GEMV_MEDIAN = 703.5
LANE_CLAIMED_SINGLE_GEMV_MIN = 681.0
LANE_CLAIMED_SINGLE_GEMV_MAX = 749.7
LANE_CLAIMED_CATALOG_ADDR = 530.7
LANE_CLAIMED_CATALOG_DECODE_MAX = 513.0
LANE_CLAIMED_ATLAS_EFFECTIVE = 337.3
LANE_CLAIMED_ACTIVE_BYTES = 9_878_901_136
LANE_CLAIMED_ATLAS_ROOF = 589.73
LANE_CLAIMED_CENSUS_ROOF = 595.9
LANE_CLAIMED_ATLAS_CEILING_TPS = 59.7

GEMV_PAYLOAD_BYTES = 13_611_663_360
PUBLISHED_PEAK_GB_S = 819.0

STATUS_LOADED = "LOADED"
STATUS_REFUSED = "REFUSED"
STATUS_UNATTRIBUTED = "UNATTRIBUTED"
STATUS_ATTRIBUTED = "ATTRIBUTED"
STATUS_STRUCTURAL = "STRUCTURAL"
STATUS_REMOVABLE = "REMOVABLE"
STATUS_PARTLY_REMOVABLE = "PARTLY_REMOVABLE"
STATUS_HYPOTHESIS = "HYPOTHESIS_UNTESTED"

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. Every GB/s and every "
    "byte count is copied from a named prior receipt or refused. Arithmetic "
    "over those copies is DERIVED_FROM_CITED_RECEIPTS, not a new experiment. "
    "self_timing is process CPU of this reread and is not GPU, TPS, or a roof. "
    "A TPS change is a hypothesis with a falsifier, never a prediction of gain."
)


class AddressingGapError(ValueError):
    """Contract violation: unsourced number, unstated roof, or TPS-as-fact."""


class UnsourcedNumber(AddressingGapError):
    """A GB/s or byte figure arrived without a receipt path."""


class UnstatedRoof(AddressingGapError):
    """A TPS ceiling was asked for without naming the roof it rests on."""


class TpsClaimRefused(AddressingGapError):
    """A TPS change was asserted as fact, or a hypothesis had no falsifier."""


# ---------------------------------------------------------------------------
# Load. Sparse-absent is not campaign-absent; empty git show is a refusal.
# ---------------------------------------------------------------------------


def load_receipt(rel: str, injected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load one JSON receipt. injected[rel] is None means a watched absence."""
    if injected is not None and rel in injected:
        doc = injected[rel]
        if doc is None:
            return {
                "status": STATUS_REFUSED,
                "rel": rel,
                "reason": "injected_absent",
                "doc": None,
                "via": "injected",
            }
        if not isinstance(doc, dict):
            return {
                "status": STATUS_REFUSED,
                "rel": rel,
                "reason": "injected_not_object",
                "doc": None,
                "via": "injected",
            }
        return {"status": STATUS_LOADED, "rel": rel, "doc": doc, "via": "injected"}
    path = REPO / rel
    if path.is_file():
        try:
            return {
                "status": STATUS_LOADED,
                "rel": rel,
                "doc": load_json(path),
                "via": f"disk:{rel}",
            }
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return {
                "status": STATUS_REFUSED,
                "rel": rel,
                "reason": f"unreadable:{exc}",
                "doc": None,
                "via": f"disk:{rel}",
            }
    blob = git("show", f"HEAD:{rel}")
    if not blob:
        return {
            "status": STATUS_REFUSED,
            "rel": rel,
            "reason": "unseen_in_this_checkout_and_HEAD",
            "doc": None,
            "via": None,
        }
    try:
        doc = json.loads(blob)
    except json.JSONDecodeError as exc:
        return {
            "status": STATUS_REFUSED,
            "rel": rel,
            "reason": f"git_unreadable:{exc}",
            "doc": None,
            "via": f"git:HEAD:{rel}",
        }
    if not isinstance(doc, dict):
        return {
            "status": STATUS_REFUSED,
            "rel": rel,
            "reason": "git_not_object",
            "doc": None,
            "via": f"git:HEAD:{rel}",
        }
    return {
        "status": STATUS_LOADED,
        "rel": rel,
        "doc": doc,
        "via": f"git:HEAD:{rel}",
    }


def nested(node: Any, *path: str) -> Any:
    cur = node
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def cite(
    value: Any,
    *,
    source_receipt: str | None,
    json_path: str,
    statistic: str,
    unit: str = "GB/s",
) -> dict[str, Any]:
    """A number without a receipt is a refusal, not a rung."""
    if not source_receipt:
        return {
            "status": STATUS_REFUSED,
            "reason": "unsourced_number",
            "json_path": json_path,
            "statistic": statistic,
            "unit": unit,
            "value": None,
            "source_receipt": None,
        }
    if not _is_number(value):
        return {
            "status": STATUS_REFUSED,
            "reason": "absent_or_non_numeric",
            "json_path": json_path,
            "statistic": statistic,
            "unit": unit,
            "value": None,
            "source_receipt": source_receipt,
        }
    return {
        "status": STATUS_LOADED,
        "reason": None,
        "json_path": json_path,
        "statistic": statistic,
        "unit": unit,
        "value": float(value),
        "source_receipt": source_receipt,
    }


def cite_path(
    loaded: Mapping[str, Any],
    *path: str,
    statistic: str,
    unit: str = "GB/s",
) -> dict[str, Any]:
    rel = str(loaded.get("rel") or "")
    if loaded.get("status") != STATUS_LOADED or not isinstance(loaded.get("doc"), dict):
        return cite(
            None,
            source_receipt=rel or None,
            json_path=".".join(path),
            statistic=statistic,
            unit=unit,
        )
    return cite(
        nested(loaded["doc"], *path),
        source_receipt=rel,
        json_path=".".join(path),
        statistic=statistic,
        unit=unit,
    )


def matches_rounded(claimed: Any, actual: Any, ndigits: int = 1) -> bool:
    if not _is_number(claimed) or not _is_number(actual):
        return False
    return round(float(actual), ndigits) == round(float(claimed), ndigits)


def adjudicate_claimed_rung(
    *,
    claimed: Any,
    actual: Mapping[str, Any] | None,
    claimed_statistic: str,
) -> dict[str, Any]:
    """A lane number that is not the named statistic of a sourced probe is refused."""
    if actual is None or actual.get("status") != STATUS_LOADED:
        return {
            "status": STATUS_REFUSED,
            "reason": "no_sourced_actual",
            "claimed": claimed,
            "claimed_statistic": claimed_statistic,
            "actual": actual,
        }
    actual_value = actual.get("value")
    if matches_rounded(claimed, actual_value):
        return {
            "status": STATUS_LOADED,
            "reason": None,
            "claimed": claimed,
            "claimed_statistic": claimed_statistic,
            "actual_value": actual_value,
            "actual_statistic": actual.get("statistic"),
            "source_receipt": actual.get("source_receipt"),
            "json_path": actual.get("json_path"),
            "matches_claimed_statistic": actual.get("statistic") == claimed_statistic,
        }
    return {
        "status": STATUS_REFUSED,
        "reason": "claimed_does_not_match_sourced_statistic",
        "claimed": claimed,
        "claimed_statistic": claimed_statistic,
        "actual_value": actual_value,
        "actual_statistic": actual.get("statistic"),
        "source_receipt": actual.get("source_receipt"),
        "json_path": actual.get("json_path"),
        "matches_claimed_statistic": False,
    }


def pick_probe_row(
    probes: Any,
    *,
    payload_bytes: int | None = None,
    label_substr: str | None = None,
) -> dict[str, Any]:
    if probes is None:
        return {"status": STATUS_REFUSED, "reason": "probe_absent", "row": None}
    if not isinstance(probes, list):
        return {"status": STATUS_REFUSED, "reason": "probe_not_a_list", "row": None}
    for row in probes:
        if not isinstance(row, dict):
            continue
        if payload_bytes is not None and row.get("payload_bytes") != payload_bytes:
            continue
        if label_substr is not None and label_substr not in str(row.get("label") or ""):
            continue
        return {"status": STATUS_LOADED, "reason": None, "row": row}
    return {"status": STATUS_REFUSED, "reason": "no_matching_probe_row", "row": None}


def spread_cite(
    loaded: Mapping[str, Any],
    probe_key: str,
    *,
    payload_bytes: int | None = None,
    label_substr: str | None = None,
    statistic: str,
) -> dict[str, Any]:
    """Cite min/median/max GB/s off a probe list or a single probe object."""
    rel = str(loaded.get("rel") or "")
    if loaded.get("status") != STATUS_LOADED:
        return cite(None, source_receipt=rel or None, json_path=probe_key, statistic=statistic)
    node = nested(loaded["doc"], probe_key)
    row: dict[str, Any] | None
    json_path: str
    if isinstance(node, list):
        picked = pick_probe_row(node, payload_bytes=payload_bytes, label_substr=label_substr)
        if picked["status"] != STATUS_LOADED:
            return cite(
                None,
                source_receipt=rel,
                json_path=f"{probe_key}[{label_substr or payload_bytes}]",
                statistic=statistic,
            )
        row = picked["row"]
        json_path = f"{probe_key}[label={row.get('label')}].spread.{statistic}"
    elif isinstance(node, dict) and isinstance(node.get("spread"), dict):
        row = node
        json_path = f"{probe_key}.spread.{statistic}"
    else:
        return cite(
            None,
            source_receipt=rel,
            json_path=probe_key,
            statistic=statistic,
        )
    spread = row.get("spread") if isinstance(row, dict) else None
    if not isinstance(spread, dict):
        return cite(None, source_receipt=rel, json_path=json_path, statistic=statistic)
    field = {
        "median": "median_gb_s",
        "min": "min_gb_s",
        "max": "max_gb_s",
    }.get(statistic)
    if field is None:
        return cite(None, source_receipt=rel, json_path=json_path, statistic=statistic)
    out = cite(
        spread.get(field),
        source_receipt=rel,
        json_path=json_path,
        statistic=statistic,
    )
    out["kernel"] = row.get("kernel") if isinstance(row, dict) else None
    out["label"] = row.get("label") if isinstance(row, dict) else None
    out["payload_bytes"] = row.get("payload_bytes") if isinstance(row, dict) else None
    out["dispatches"] = row.get("dispatches") if isinstance(row, dict) else None
    out["topology"] = row.get("topology") if isinstance(row, dict) else None
    return out


def refuse_dram_roof(citation: Mapping[str, Any], published: Mapping[str, Any]) -> dict[str, Any]:
    """A GB/s figure above datasheet peak is cache or a bad denominator, not DRAM."""
    if citation.get("status") != STATUS_LOADED or published.get("status") != STATUS_LOADED:
        return {
            "status": STATUS_REFUSED,
            "reason": "cannot_adjudicate_roof_without_both_numbers",
            "as_dram_roof": False,
        }
    value = float(citation["value"])
    peak = float(published["value"])
    if value > peak:
        return {
            "status": STATUS_REFUSED,
            "reason": "exceeds_published_peak_not_a_dram_roof",
            "value": value,
            "published_peak": peak,
            "as_dram_roof": False,
            "source_receipt": citation.get("source_receipt"),
        }
    return {
        "status": STATUS_LOADED,
        "reason": None,
        "value": value,
        "published_peak": peak,
        "as_dram_roof": True,
        "source_receipt": citation.get("source_receipt"),
    }


def ceiling_raw_tps(
    active_bytes: Any,
    roof: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """raw TPS ceiling = roof_gb_s * 1e9 / active_bytes. Roof must be named."""
    if roof is None:
        raise UnstatedRoof("ceiling with unstated roof")
    if roof.get("status") != STATUS_LOADED or not _is_number(roof.get("value")):
        return {
            "status": STATUS_REFUSED,
            "reason": roof.get("reason") or "roof_refused",
            "roof_name": roof.get("name"),
            "roof_source": roof.get("source_receipt"),
            "raw_tps_ceiling": None,
        }
    if not roof.get("source_receipt"):
        raise UnstatedRoof("ceiling with unstated roof")
    if not _is_number(active_bytes) or float(active_bytes) <= 0:
        return {
            "status": STATUS_REFUSED,
            "reason": "active_bytes_absent",
            "roof_name": roof.get("name"),
            "roof_source": roof.get("source_receipt"),
            "raw_tps_ceiling": None,
        }
    ceiling = float(roof["value"]) * 1e9 / float(active_bytes)
    return {
        "status": STATUS_LOADED,
        "reason": None,
        "roof_name": roof.get("name"),
        "roof_kind": roof.get("kind"),
        "roof_value_gb_s": float(roof["value"]),
        "roof_source": roof.get("source_receipt"),
        "roof_json_path": roof.get("json_path"),
        "roof_statistic": roof.get("statistic"),
        "active_bytes": int(active_bytes),
        "formula": "roof_gb_s * 1e9 / active_weight_bytes_per_token",
        "raw_tps_ceiling": ceiling,
        "would_improve_tps": None,
        "claim": "arithmetic upper bound IF this roof were 100% utilised; not a reachable TPS",
    }


def tps_hypothesis(text: str, *, falsifier: str | None) -> dict[str, Any]:
    """A TPS change without a falsifier is refused. None of this is a gain claim."""
    if not falsifier or not str(falsifier).strip():
        raise TpsClaimRefused("hypothesis without falsifier is refused")
    return {
        "kind": STATUS_HYPOTHESIS,
        "text": text,
        "falsifier": str(falsifier).strip(),
        "would_improve_tps": None,
        "status_label": "HYPOTHESIS_UNTESTED",
    }


def loss_delta(
    high: Mapping[str, Any],
    low: Mapping[str, Any],
    *,
    same_genome: bool,
    same_receipt: bool,
    same_kernel: bool,
    same_bytes: bool,
) -> dict[str, Any]:
    """A drop across genomes is not a mechanism. Remainder stays UNATTRIBUTED."""
    if high.get("status") != STATUS_LOADED or low.get("status") != STATUS_LOADED:
        return {
            "status": STATUS_REFUSED,
            "reason": "missing_rung",
            "delta_gb_s": None,
            "comparable": False,
        }
    delta = float(high["value"]) - float(low["value"])
    comparable = bool(same_genome and same_bytes)
    return {
        "status": STATUS_LOADED,
        "delta_gb_s": delta,
        "from_gb_s": float(high["value"]),
        "to_gb_s": float(low["value"]),
        "from_source": high.get("source_receipt"),
        "to_source": low.get("source_receipt"),
        "same_genome": same_genome,
        "same_receipt": same_receipt,
        "same_kernel": same_kernel,
        "same_bytes": same_bytes,
        "comparable": comparable,
        "comparable_reason": (
            None
            if comparable
            else "different genome, kernel family, or byte denominator; not one mechanism"
        ),
    }


def close_loss_chain(
    total_delta: Any,
    attributed_pieces: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Remainder is UNATTRIBUTED. Never distributed to make the sum close."""
    if not _is_number(total_delta):
        return {
            "status": STATUS_REFUSED,
            "reason": "total_delta_absent",
            "attributed_gb_s": None,
            "unattributed_gb_s": None,
        }
    attributed = 0.0
    named: list[dict[str, Any]] = []
    for piece in attributed_pieces:
        if piece.get("status") != STATUS_ATTRIBUTED:
            continue
        if not _is_number(piece.get("delta_gb_s")):
            continue
        attributed += float(piece["delta_gb_s"])
        named.append(
            {
                "id": piece.get("id"),
                "delta_gb_s": float(piece["delta_gb_s"]),
                "mechanism": piece.get("mechanism"),
            }
        )
    remainder = float(total_delta) - attributed
    return {
        "status": STATUS_LOADED,
        "total_delta_gb_s": float(total_delta),
        "attributed_gb_s": attributed,
        "unattributed_gb_s": remainder,
        "unattributed_label": STATUS_UNATTRIBUTED,
        "pieces": named,
        "refused_policy": "do_not_distribute_unattributed_remainder",
    }


def refuse_moe_gather(genome: str) -> dict[str, Any]:
    """MoE route-selection gather is structural for MoE. This parent is not MoE."""
    g = (genome or "").lower()
    moe = "moe" in g or "mixtral" in g or "routed_expert" in g
    if moe:
        return {
            "status": STATUS_STRUCTURAL,
            "mechanism": "gather_for_route_selection",
            "reason": "MoE routing genuinely gathers expert weights; not removable by bind hoisting",
        }
    return {
        "status": STATUS_REFUSED,
        "mechanism": "gather_for_route_selection",
        "reason": (
            "this catalog mix is 192 MLP + 144 DeltaNet + 64 GQA + 1 lm_head; "
            "Qwen3.8-27B hybrid is not an MoE. Attributing 703→337 to expert "
            "gather is a genome error."
        ),
        "genome": genome,
    }


# ---------------------------------------------------------------------------
# Static kernel / bind-path notes. Source-read, not timed.
# ---------------------------------------------------------------------------


KERNEL_GEOMETRY = {
    "family": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "source": "crates/hawking-core/shaders/qwen_uniform_q4.metal",
    "threads_per_row": 64,
    "threadgroup": 128,
    "rows_per_threadgroup": 2,
    "inner_stride_cols": 512,
    "code_bytes_per_group_of_64": 32,
    "scale_bytes_per_group_of_64": 2,
    "addr_probe": (
        "same launch geometry; loads scales + packed codes and sinks them; "
        "no nibble unpack, no input-vector load, no FMA"
    ),
    "decode_probe": "address + dequant; still no input-vector load / FMA",
    "full": "address + dequant + input load + FMA",
    "cannot_establish": (
        "shader text does not establish occupancy, coalescing efficiency, "
        "or achieved GB/s"
    ),
}

BIND_PATH = {
    "catalog": (
        "crates/hawking-core/src/metal/mod.rs static_kernel_name is the kernel "
        "name catalog (string → &'static str). Tensor catalog addressing is "
        "the production launch of one GEMV per weight tensor."
    ),
    "default_dispatch": (
        "MetalContext::dispatch_threads creates a command buffer, a compute "
        "encoder, set_compute_pipeline_state, caller buffer binds, "
        "dispatch_threads, end_encoding, commit, wait_until_completed — "
        "per call."
    ),
    "batched": (
        "CommandBatch can fold dispatches into one sequential encoder "
        "(enable_ordered_encoder) or one serial/concurrent group. "
        "BANDWIDTH_ASCENT production decode recorded 756 dispatches, 1 CB."
    ),
    "icb": (
        "ReplayableComputeGraph encodes an IndirectCommandBuffer once and "
        "replays against stable addresses. Comment in metal/mod.rs: "
        "'intentionally not wired into decode selection yet' — CPU encoding "
        "share remains below the ICB ship gate. Declared, not executed, here."
    ),
    "argument_buffers": (
        "ReplayResourceDeclaration captures resources referenced indirectly. "
        "This sidecar did not watch argument-buffer setup on a decode token."
    ),
}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _roof(
    name: str,
    kind: str,
    citation: Mapping[str, Any],
    *,
    note: str,
    as_dram_roof: bool | None = None,
) -> dict[str, Any]:
    row = dict(citation)
    row["name"] = name
    row["kind"] = kind
    row["note"] = note
    if as_dram_roof is not None:
        row["as_dram_roof"] = as_dram_roof
    return row


def analyze(injected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    loads = {
        "lane_honest": load_receipt(LANE_HONEST, injected),
        "lane_honest_reduced": load_receipt(LANE_HONEST_REDUCED, injected),
        "lane_g044": load_receipt(LANE_G044, injected),
        "honest": load_receipt(ASCENT_HONEST, injected),
        "honest_reduced": load_receipt(ASCENT_HONEST_REDUCED, injected),
        "g044": load_receipt(ASCENT_G044, injected),
        "atlas": load_receipt(ATLAS_REL, injected),
        "ascent": load_receipt(ASCENT_REL, injected),
        "census": load_receipt(CENSUS_REL, injected),
        "g072": load_receipt(G072_REL, injected),
        "genome": load_receipt(GENOME_REL, injected),
        "ledger": load_receipt(LEDGER_REL, injected),
        "organ_bw": load_receipt(ORGAN_BW_REL, injected),
        "unpack": load_receipt(UNPACK_REL, injected),
    }

    path_adjudication = {
        "lane_named_honest_headless": {
            "rel": LANE_HONEST,
            "status": loads["lane_honest"]["status"],
            "reason": loads["lane_honest"].get("reason"),
            "via": loads["lane_honest"].get("via"),
        },
        "recovered_honest_ascent": {
            "rel": ASCENT_HONEST,
            "status": loads["honest"]["status"],
            "reason": loads["honest"].get("reason"),
            "via": loads["honest"].get("via"),
        },
        "lane_named_g044_headless": {
            "rel": LANE_G044,
            "status": loads["lane_g044"]["status"],
            "reason": loads["lane_g044"].get("reason"),
            "via": loads["lane_g044"].get("via"),
        },
        "recovered_g044_ascent": {
            "rel": ASCENT_G044,
            "status": loads["g044"]["status"],
            "reason": loads["g044"].get("reason"),
            "via": loads["g044"].get("via"),
        },
        "rule": (
            "the lane named receipts/headless/HONEST_ROOF_WEIGHT_ADDRESSING.json; "
            "HEAD has it under receipts/ascent-2026-08-16/. Substituting without "
            "recording the miss would repeat the 595.9 hop."
        ),
    }

    honest = loads["honest"]
    reduced = loads["honest_reduced"]
    published = cite_path(
        honest, "hardware", "published_peak_gb_s", statistic="datasheet", unit="GB/s"
    )
    if published["status"] != STATUS_LOADED:
        published = cite(
            PUBLISHED_PEAK_GB_S,
            source_receipt=None,
            json_path="hardware.published_peak_gb_s",
            statistic="datasheet",
        )

    single_addr_median = spread_cite(
        honest,
        "q4_single_gemv_addr_probe",
        payload_bytes=GEMV_PAYLOAD_BYTES,
        statistic="median",
    )
    single_addr_min = spread_cite(
        honest,
        "q4_single_gemv_addr_probe",
        payload_bytes=GEMV_PAYLOAD_BYTES,
        statistic="min",
    )
    single_addr_max = spread_cite(
        honest,
        "q4_single_gemv_addr_probe",
        payload_bytes=GEMV_PAYLOAD_BYTES,
        statistic="max",
    )
    single_decode_median = spread_cite(
        honest,
        "q4_single_gemv_decode_probe",
        payload_bytes=GEMV_PAYLOAD_BYTES,
        statistic="median",
    )
    single_full_median = spread_cite(
        honest,
        "q4_single_gemv_full",
        payload_bytes=GEMV_PAYLOAD_BYTES,
        statistic="median",
    )
    catalog_addr_median = spread_cite(
        honest, "q4_production_catalog_addr_probe", statistic="median"
    )
    catalog_decode_max = spread_cite(
        honest, "q4_production_catalog_decode_probe", statistic="max"
    )
    catalog_decode_median = spread_cite(
        honest, "q4_production_catalog_decode_probe", statistic="median"
    )
    catalog_full_median = spread_cite(
        honest, "q4_production_catalog_full", statistic="median"
    )
    tiled_addr = spread_cite(
        honest,
        "q4_tiled_production_organ",
        label_substr="13p612gb_tiled_gate_addr",
        statistic="median",
    )
    unique_once_13p6 = spread_cite(
        honest,
        "unique_once_sweep",
        payload_bytes=GEMV_PAYLOAD_BYTES,
        statistic="median",
    )
    reduced_catalog = spread_cite(
        reduced, "q4_production_catalog_addr_probe", statistic="median"
    )

    atlas = loads["atlas"]
    atlas_bytes = cite_path(
        atlas,
        "headline",
        "active_weight_bytes_per_token",
        statistic="catalog_sum",
        unit="bytes",
    )
    atlas_effective = cite_path(
        atlas,
        "THE_CEILING",
        "effective_weight_bandwidth_gb_s",
        statistic="derived_raw_tps_times_bytes",
    )
    atlas_roof = cite_path(
        atlas, "THE_CEILING", "measured_roof_gb_s", statistic="copied_machine_genome_median"
    )
    atlas_ceiling = cite_path(
        atlas,
        "THE_CEILING",
        "raw_tps_ceiling_at_100pct_of_roof",
        statistic="derived",
        unit="raw_tps_ceiling",
    )
    atlas_recorded_tps = cite_path(
        atlas,
        "THE_CEILING",
        "sealed_raw_tps_recorded",
        statistic="recorded_not_remeasured",
        unit="raw_tps_recorded",
    )
    atlas_need = cite_path(
        atlas,
        "THE_CEILING",
        "raw_tps_needed_for_50_accepted_at_the_30_of_43_floor",
        statistic="derived",
        unit="raw_tps_needed",
    )
    atlas_nothing_timed = False
    if atlas.get("status") == STATUS_LOADED:
        boundary = nested(atlas["doc"], "claim_boundary")
        atlas_nothing_timed = isinstance(boundary, list) and any(
            "NOTHING WAS TIMED" in str(x) for x in boundary
        )

    ascent = loads["ascent"]
    ascent_bytes = cite_path(
        ascent,
        "prior_not_rederived",
        "parent_active_bytes_per_token",
        statistic="cited",
        unit="bytes",
    )
    prod_decode_gpu = cite_path(
        ascent, "production_decode", "before", "achieved_gb_s", statistic="median_tpr64"
    )
    q2_gate_tpr64 = None
    q2_gate_addr = None
    shapes = nested(ascent.get("doc") or {}, "isolated_gemv", "shapes")
    if isinstance(shapes, list) and shapes and isinstance(shapes[0], dict):
        q2_gate_tpr64 = cite(
            nested(shapes[0], "arms", "tpr64", "weight_gb_s_median"),
            source_receipt=ASCENT_REL,
            json_path="isolated_gemv.shapes[0].arms.tpr64.weight_gb_s_median",
            statistic="median",
        )
        q2_gate_addr = cite(
            nested(shapes[0], "arms", "qmvfast_addr_probe", "weight_gb_s_median"),
            source_receipt=ASCENT_REL,
            json_path="isolated_gemv.shapes[0].arms.qmvfast_addr_probe.weight_gb_s_median",
            statistic="median",
        )

    census_roof = cite_path(
        loads["census"],
        "artifact",
        "anchors_not_rederived",
        "measured_roof_GB_s",
        statistic="anchor_not_rederived",
    )
    g072_roof = cite_path(
        loads["g072"], "measured_roof_gb_s", statistic="same_run_roofline_peak"
    )
    g072_basis = None
    if loads["g072"].get("status") == STATUS_LOADED:
        g072_basis = nested(loads["g072"]["doc"], "roof_basis")
    g044_roof = cite_path(
        loads["g044"], "bandwidth_ceiling_gb_s", statistic="roofline_sweep_f4_plateau"
    )
    genome_roof = cite_path(
        loads["genome"], "measured_bandwidth", "median_gb_s", statistic="f32_triad_median"
    )
    genome_is_roof = nested(
        loads["genome"].get("doc") or {}, "measured_bandwidth", "is_theoretical_roof"
    )
    ledger_theo = cite_path(
        loads["ledger"],
        "three_roofs",
        "DEVICE_THEORETICAL",
        "value",
        statistic="datasheet",
    )
    ledger_sust = cite_path(
        loads["ledger"],
        "three_roofs",
        "DEVICE_MEASURED_SUSTAINED",
        "value",
        statistic="n017_unique_once",
    )
    ledger_reach = cite_path(
        loads["ledger"],
        "three_roofs",
        "MODEL_REACHABLE",
        "value",
        statistic="derived_ai_occupancy_1cb",
    )

    # Active bytes: the lane said three recover receipts agree. Adjudicate.
    recover_byte_hits = []
    for label, citation in (
        ("ACCELERATOR_TOKEN_BYTES_ATLAS", atlas_bytes),
        ("BANDWIDTH_ASCENT", ascent_bytes),
    ):
        if citation.get("status") == STATUS_LOADED and int(citation["value"]) == LANE_CLAIMED_ACTIVE_BYTES:
            recover_byte_hits.append(label)
    honest_payload = cite_path(
        honest,
        "byte_count_adjudication",
        "defended_bytes",
        statistic="gemv_payload",
        unit="bytes",
    )
    active_bytes_adjudication = {
        "claimed": LANE_CLAIMED_ACTIVE_BYTES,
        "recover_receipts_that_carry_the_claimed_count": recover_byte_hits,
        "honest_roof_defended_gemv_payload_bytes": honest_payload,
        "census_is_a_different_artifact": True,
        "g044_silent": loads["g044"]["status"] == STATUS_LOADED
        and nested(loads["g044"]["doc"], "headline") is None,
        "status": (
            "TWO_OF_NAMED_RECOVER_SET"
            if len(recover_byte_hits) == 2
            else STATUS_REFUSED
        ),
        "not_three_of_the_named_recover_set": True,
        "why_honest_disagrees": (
            "HONEST_ROOF defends 13_611_663_360 geometry GEMV codes+scales on "
            "the uniform-Q4 genome, not sealed-3.14 active-weight-per-token"
        ),
    }
    active_bytes = (
        int(atlas_bytes["value"])
        if atlas_bytes.get("status") == STATUS_LOADED
        else None
    )

    claimed_703 = adjudicate_claimed_rung(
        claimed=LANE_CLAIMED_SINGLE_GEMV_MEDIAN,
        actual=single_addr_median,
        claimed_statistic="median",
    )
    claimed_703["closest_sourced_max"] = single_addr_max
    claimed_703["closest_sourced_min"] = single_addr_min
    claimed_703["lane_claimed_min"] = LANE_CLAIMED_SINGLE_GEMV_MIN
    claimed_703["lane_claimed_max"] = LANE_CLAIMED_SINGLE_GEMV_MAX
    claimed_703["min_adjudication"] = adjudicate_claimed_rung(
        claimed=LANE_CLAIMED_SINGLE_GEMV_MIN,
        actual=single_addr_min,
        claimed_statistic="min",
    )
    claimed_703["max_adjudication"] = adjudicate_claimed_rung(
        claimed=LANE_CLAIMED_SINGLE_GEMV_MAX,
        actual=single_addr_max,
        claimed_statistic="max",
    )
    claimed_703["reading"] = (
        "q4_single_gemv_addr_probe at the 13.612 GB payload (the fair "
        "comparison against the 401-GEMV catalog) has median 699.57, "
        "min 693.15, max 703.61. 703.5 is not the median. It is not the "
        "max rounded to one decimal (703.6). The 64 MiB point is 817 "
        "median and is cache-sized; it is not this roof."
    )

    claimed_530 = adjudicate_claimed_rung(
        claimed=LANE_CLAIMED_CATALOG_ADDR,
        actual=catalog_addr_median,
        claimed_statistic="median",
    )
    claimed_513 = adjudicate_claimed_rung(
        claimed=LANE_CLAIMED_CATALOG_DECODE_MAX,
        actual=catalog_decode_max,
        claimed_statistic="max",
    )
    claimed_513["median_of_same_probe"] = catalog_decode_median
    claimed_513["warning"] = (
        "the lane cited MAX. The same probe's median is lower and the "
        "spread is 294–513 GB/s on five reps (contamination_note: CPU "
        "builds + sealed supervisor). MAX is the optimistic decode rung, "
        "not a stable tax."
    )
    claimed_337 = adjudicate_claimed_rung(
        claimed=LANE_CLAIMED_ATLAS_EFFECTIVE,
        actual=atlas_effective,
        claimed_statistic="derived_raw_tps_times_bytes",
    )
    claimed_819 = adjudicate_claimed_rung(
        claimed=LANE_CLAIMED_PUBLISHED,
        actual=published,
        claimed_statistic="datasheet",
    )

    q2_addr_as_dram = (
        refuse_dram_roof(q2_gate_addr, published)
        if q2_gate_addr is not None
        else {"status": STATUS_REFUSED, "reason": "q2_addr_probe_absent", "as_dram_roof": False}
    )

    t2 = loss_delta(
        single_addr_median,
        catalog_addr_median,
        same_genome=True,
        same_receipt=True,
        same_kernel=True,
        same_bytes=True,
    )
    t2_tile = loss_delta(
        single_addr_median,
        tiled_addr,
        same_genome=True,
        same_receipt=True,
        same_kernel=True,
        same_bytes=False,
    )
    t2_mix = loss_delta(
        tiled_addr,
        catalog_addr_median,
        same_genome=True,
        same_receipt=True,
        same_kernel=True,
        same_bytes=False,
    )
    t3 = loss_delta(
        catalog_addr_median,
        catalog_decode_max,
        same_genome=True,
        same_receipt=True,
        same_kernel=False,
        same_bytes=True,
    )
    t4 = loss_delta(
        catalog_addr_median,
        atlas_effective,
        same_genome=False,
        same_receipt=False,
        same_kernel=False,
        same_bytes=False,
    )
    t4b = loss_delta(
        catalog_addr_median,
        prod_decode_gpu,
        same_genome=False,
        same_receipt=False,
        same_kernel=False,
        same_bytes=False,
    )
    t4c = loss_delta(
        prod_decode_gpu,
        atlas_effective,
        same_genome=True,
        same_receipt=False,
        same_kernel=True,
        same_bytes=True,
    )

    moe = refuse_moe_gather("qwen38-27b-hybrid-deltanet-gqa")

    alu_tax = None
    if (
        single_addr_median.get("status") == STATUS_LOADED
        and single_full_median.get("status") == STATUS_LOADED
        and float(single_addr_median["value"]) != 0
    ):
        alu_tax = 1.0 - float(single_full_median["value"]) / float(single_addr_median["value"])

    transition_703_to_530 = {
        "id": "T2_single_gemv_addr_to_catalog_addr",
        "from": "q4_single_gemv_addr_probe median at 13.612 GB",
        "to": "q4_production_catalog_addr_probe median at 13.612 GB",
        "delta": t2,
        "status": STATUS_ATTRIBUTED if t2.get("comparable") else STATUS_UNATTRIBUTED,
        "mechanism": "catalog_indirection_per_tensor + mixed-organ occupancy",
        "from_source": t2.get("from_source"),
        "named_by": (
            "adjudication.execution_headroom.dispatch_topology in "
            + ASCENT_HONEST
            + ": '401 mixed organs leave ~24% vs one GEMV. Tiny ba "
            "(96x5120) and encoder boundaries are genome, not ALU.'"
        ),
        "same_kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_addr_probe",
        "same_payload_bytes": GEMV_PAYLOAD_BYTES,
        "catalog_mix": "192 MLP + 144 DN + 64 GQA + 1 lm_head = 401 GEMVs",
        "not_this_drop": {
            "tile_threadgroup_geometry": (
                "STRUCTURAL to the kernel (64 threads/row, TG 128) and IDENTICAL "
                "on both rungs; it cannot be why 699 became 530"
            ),
            "coalescing": "same kernel, same packed-code load; not isolated here",
            "gather_for_route_selection": moe,
            "decode_and_fma": (
                f"single-GEMV full vs addr tax is {alu_tax}; catalog addr_probe "
                "does not unpack. ALU is not this drop."
            ),
        },
        "subsplit_same_receipt": {
            "single_gemv_to_287_same_shape_tiles": t2_tile,
            "same_shape_tiles_to_401_mixed": t2_mix,
            "tile_payload_bytes_not_identical": (
                "tiled 13p612gb_tiled_gate_addr payload is 13_589_381_120, "
                "catalog is 13_611_663_360. Direction holds; do not quote the "
                "subsplit as exact conservation."
            ),
            "reading": (
                "chopping one GEMV into 287 same-shape gate tiles already leaves "
                "the single-GEMV median; mixing 401 production shapes leaves "
                "more. Production cannot be one 13.6 GB GEMV (inputs differ per "
                "tensor) — that piece is STRUCTURAL. Per-dispatch bind and "
                "mixing tiny ba with huge MLP in one encoder are the removable "
                "hypotheses."
            ),
        },
        "removable": STATUS_PARTLY_REMOVABLE,
        "structural_piece": "many tensors / many inputs; cannot concatenate into one GEMV",
        "removable_piece": (
            "per-dispatch pipeline+buffer rebinding (ICB exists, not wired) and "
            "mixing occupancy-starved 96x5120 ba with streaming MLP in one encoder"
        ),
        "delta_gb_s": t2.get("delta_gb_s") if t2.get("comparable") else None,
    }
    if t2.get("comparable") and _is_number(t2.get("delta_gb_s")):
        transition_703_to_530["status"] = STATUS_ATTRIBUTED

    transition_530_to_337 = {
        "id": "T4_catalog_addr_to_atlas_effective",
        "from": "q4_production_catalog_addr_probe median (uniform-Q4, 13.612 GB, 401 GEMVs)",
        "to": "ACCELERATOR_TOKEN_BYTES_ATLAS effective_weight_bandwidth (sealed-3.14, 9.879 GB)",
        "delta": t4,
        "status": STATUS_UNATTRIBUTED,
        "mechanism": STATUS_UNATTRIBUTED,
        "why_not_one_mechanism": (
            "different artifact (uniform-Q4 13.6 GB GEMV payload vs sealed-3.14 "
            "HQ30UQ4 mixer + affine_q2 MLP 9.879 GB), different kernel families, "
            "different dispatch graph (401 vs 756/964), and 337 is raw_tps × "
            "bytes with NOTHING TIMED in the atlas. Subtracting 530−337 as if "
            "it continued the catalog-addressing experiment is the same class "
            "of error as promoting 595.9 from a family scoring reference."
        ),
        "candidates_not_distributed": [
            {
                "id": "q2_mlp_majority_bytes",
                "status": STATUS_UNATTRIBUTED,
                "note": (
                    "atlas pareto: 5.348 GB of 9.879 GB is "
                    "qwen_affine_q2_group32_matvec_geo_tpr64_tg128 (192 "
                    "dispatches). BANDWIDTH_ASCENT isolated gate_proj tpr64 is "
                    "organ-sized (~27 MB), not a 13 GB stream."
                ),
                "isolated_gate_tpr64": q2_gate_tpr64,
                "isolated_gate_addr_probe_refused_as_dram_roof": q2_addr_as_dram,
            },
            {
                "id": "non_gemv_organs",
                "status": STATUS_UNATTRIBUTED,
                "note": (
                    "atlas: 402 matvec dispatches carry 99.893% of weight bytes; "
                    "562 others carry 0.107% and price at ~0.586 ms of a 29.29 ms "
                    "token. ORGAN_BANDWIDTH ranks mlp_gate_up as the largest "
                    "share of the 356.7→778.8 gap, then deltanet, mlp_down, GQA. "
                    "That is a different roof (778.8) on the production genome."
                ),
            },
            {
                "id": "tps_denominator_vs_gpu_timestamp",
                "status": STATUS_UNATTRIBUTED,
                "note": (
                    "337 is recorded raw TPS × bytes. BANDWIDTH_ASCENT tpr64 "
                    "production decode is GPU-timestamp GB/s on the same "
                    "9.879 GB. They are not the same quantity and were not "
                    "the same run."
                ),
                "production_decode_gpu_tpr64": prod_decode_gpu,
                "atlas_effective": atlas_effective,
                "atlas_nothing_timed": atlas_nothing_timed,
                "delta_same_bytes_different_runs": t4c,
            },
            {
                "id": "synchronization_between_organs",
                "status": STATUS_UNATTRIBUTED,
                "note": (
                    "catalog addr probe is GEMV-only. Production decode has "
                    "DeltaNet, GQA, RMSNorm, SwiGLU, sampling. Sync is in that "
                    "graph; it is not isolated as a GB/s term here."
                ),
            },
        ],
        "gather_for_route_selection": moe,
        "catalog_decode_probe_is_not_the_rung": {
            "delta_addr_median_to_decode_max": t3,
            "decode_median": catalog_decode_median,
            "full_median": catalog_full_median,
            "status": STATUS_UNATTRIBUTED,
            "reason": (
                "max vs median, five-rep spread 294–513, and decode-median "
                "(454.9) sits below full-median (505.8). Do not treat 513 as "
                "a stable decode tax on the catalog topology."
            ),
        },
        "delta_vs_production_decode_gpu": t4b,
        "removable": STATUS_UNATTRIBUTED,
        "delta_gb_s": None,
    }

    attributed_for_chain = []
    if transition_703_to_530["status"] == STATUS_ATTRIBUTED and _is_number(t2.get("delta_gb_s")):
        attributed_for_chain.append(
            {
                "id": transition_703_to_530["id"],
                "status": STATUS_ATTRIBUTED,
                "delta_gb_s": t2["delta_gb_s"],
                "mechanism": transition_703_to_530["mechanism"],
            }
        )
    full_drop = None
    if (
        single_addr_median.get("status") == STATUS_LOADED
        and atlas_effective.get("status") == STATUS_LOADED
    ):
        full_drop = float(single_addr_median["value"]) - float(atlas_effective["value"])
    chain_close = close_loss_chain(full_drop, attributed_for_chain)

    roofs = [
        _roof(
            "published_peak",
            "DEVICE_THEORETICAL",
            published,
            note="datasheet. Honest roof: 15% unused is not available to this Q4 grouped-GEMV access pattern without changing the genome.",
            as_dram_roof=True,
        ),
        _roof(
            "q4_single_gemv_addr_13p6gb_median",
            "ACCESS_PATTERN_MEASURED",
            single_addr_median,
            note="HONEST_ROOF kernel roof. Conditioned on geo_tpr64_tg128, unique codes+scales, 13.612 GB, 1 dispatch. Timing_label GPU_PROTECTED_CPU_CONTENDED, clean_box false.",
            as_dram_roof=True,
        ),
        _roof(
            "q4_single_gemv_addr_13p6gb_max",
            "ACCESS_PATTERN_MAX_OF_FIVE",
            single_addr_max,
            note="not a roof; the best of five contended reps. Included so 703.5 can be compared and refused.",
            as_dram_roof=False,
        ),
        _roof(
            "n017_unique_once_sustained",
            "DEVICE_MEASURED_SUSTAINED",
            ledger_sust,
            note="ORGAN_ROOF_LEDGER copies BANDWIDTH_ROOF 778.8. HONEST_ROOF unique_once at 13.6 GB plateaus near 376. Two unique_once figures; do not collapse them.",
            as_dram_roof=True,
        ),
        _roof(
            "honest_unique_once_13p6gb",
            "UNIQUE_ONCE_CONTROL",
            unique_once_13p6,
            note="honest roof: unique_once is NOT the Q4 GEMV ceiling; production Q4 GEMV beats it at every size measured.",
            as_dram_roof=True,
        ),
        _roof(
            "model_reachable_organ_ledger",
            "MODEL_REACHABLE",
            ledger_reach,
            note="AI + occupancy + 1-CB serial structure on the production token graph. A third number. S022 §1.",
            as_dram_roof=False,
        ),
        _roof(
            "g044_roofline_sweep_f4",
            "KERNEL_FAMILY_SWEEP",
            g044_roof,
            note="hawking_roofline_sweep_f4 1 GiB float4, bandwidth plateau. Not a GEMV genome, not a decode roof.",
            as_dram_roof=True,
        ),
        _roof(
            "g072_multi_plane_scoring_reference",
            "FAMILY_SCORING_REFERENCE",
            g072_roof,
            note=(
                str(g072_basis)
                if g072_basis
                else "G072 set 595.9 as the reference these plane kernels are scored against."
            )
            + " NOETIC_CANON traces the three hops that promoted it to a machine property.",
            as_dram_roof=False,
        ),
        _roof(
            "census_anchor_595p9",
            "PROMOTED_MACHINE_PROPERTY",
            census_roof,
            note="NOETIC_ORGAN_CENSUS artifact.anchors_not_rederived.measured_roof_GB_s = 595.9. Copied, not measured in that receipt.",
            as_dram_roof=False,
        ),
        _roof(
            "machine_genome_f32_triad",
            "ONE_ACCESS_PATTERN",
            genome_roof,
            note=(
                "MACHINE_GENOME measured_bandwidth.median_gb_s = 589.73; "
                f"is_theoretical_roof={genome_is_roof}. Atlas THE_CEILING used this. "
                "The genome itself says it is not the SoC roof and not workload-reachable."
            ),
            as_dram_roof=True,
        ),
        _roof(
            "q4_catalog_addr_401",
            "PRODUCTION_SHAPED_Q4_ADDR",
            catalog_addr_median,
            note="same kernel as the 13.6 GB single GEMV, 401 mixed organs, unique synthetic Q4. Not sealed-3.14 production decode.",
            as_dram_roof=True,
        ),
        _roof(
            "production_decode_tpr64_gpu",
            "PRODUCTION_DECODE_GPU",
            prod_decode_gpu,
            note="BANDWIDTH_ASCENT tpr64 incumbent on sealed-3.14, 9.879 GB, 756 dispatches, 1 CB. GPU timestamps. Contended.",
            as_dram_roof=False,
        ),
        _roof(
            "atlas_effective_337",
            "DERIVED_EFFECTIVE",
            atlas_effective,
            note="raw_tps_recorded × active_bytes. Atlas: NOTHING WAS TIMED. Not a roof; the achieved-on-TPS figure.",
            as_dram_roof=False,
        ),
    ]
    for row in roofs:
        if row.get("status") == STATUS_LOADED and row.get("name") == "q4_single_gemv_addr_13p6gb_median":
            row["fraction_of_published_peak"] = (
                float(row["value"]) / float(published["value"])
                if published.get("status") == STATUS_LOADED
                else None
            )

    ceilings = []
    for row in roofs:
        try:
            ceilings.append(ceiling_raw_tps(active_bytes, row))
        except UnstatedRoof as exc:
            ceilings.append(
                {
                    "status": STATUS_REFUSED,
                    "reason": str(exc),
                    "roof_name": row.get("name"),
                    "raw_tps_ceiling": None,
                }
            )

    atlas_ceiling_check = None
    if atlas_ceiling.get("status") == STATUS_LOADED and active_bytes is not None:
        recomputed = ceiling_raw_tps(
            active_bytes,
            _roof("machine_genome_f32_triad", "ONE_ACCESS_PATTERN", genome_roof, note=""),
        )
        atlas_ceiling_check = {
            "atlas_recorded_ceiling": atlas_ceiling.get("value"),
            "recomputed_against_589p73": recomputed.get("raw_tps_ceiling"),
            "roof_the_atlas_used": "receipts/headless/MACHINE_GENOME.json measured_bandwidth.median_gb_s",
            "roof_the_atlas_did_not_state_in_THE_CEILING_name": (
                "THE_CEILING.measured_roof_gb_s = 589.73 is a number; identities.machine.receipt "
                "names MACHINE_GENOME. Census 595.9 and honest 699.57 were not used."
            ),
            "matches_recompute": (
                recomputed.get("status") == STATUS_LOADED
                and abs(float(atlas_ceiling["value"]) - float(recomputed["raw_tps_ceiling"])) < 1e-6
            ),
        }

    falsifier = (
        "GPU_PROTECTED A/B, same unique 13_611_663_360-byte Q4 slab, same "
        "addr_probe kernel: (A) 401 production-shaped GEMVs as "
        "q4_production_catalog_addr_probe, (B) ReplayableComputeGraph of the "
        "same 401 with binds hoisted, (C) fused gate_up / qkv reducing "
        "dispatch count at unchanged bytes. Hypothesis dies if B and C GPU "
        "timestamp GB/s ranges overlap A. This sidecar will not run it."
    )
    tallest = {
        "name": "production_catalog_mixed_organ_addressing_vs_single_clean_gemv",
        "status": transition_703_to_530["status"],
        "why_this_one": (
            "It is the only drop whose two rungs share genome, kernel, payload "
            "bytes, and receipt. The numerically larger 530→337 step is "
            "UNATTRIBUTED because those rungs do not."
        ),
        "from": single_addr_median,
        "to": catalog_addr_median,
        "delta_gb_s": t2.get("delta_gb_s") if t2.get("comparable") else None,
        "what_would_remove_it": tps_hypothesis(
            "Hoist per-dispatch catalog binds into the existing ICB substrate "
            "and stop mixing occupancy-starved 96x5120 ba with streaming MLP "
            "in one encoder; fuse same-input pairs (gate_up / qkv kernels "
            "already exist). Cannot be 'one 13.6 GB GEMV' — that topology is "
            "not production.",
            falsifier=falsifier,
        ),
        "structural_remainder": (
            "one dispatch per distinct input vector is required; concatenating "
            "unrelated tensors against a synthetic x is a microbenchmark"
        ),
        "not_a_tps_gain": True,
    }

    removable_ranking = [
        {
            "rank": 1,
            "id": "per_dispatch_rebinding_and_catalog_indirection",
            "class": STATUS_REMOVABLE,
            "class_is": STATUS_HYPOTHESIS,
            "applies_to": "T2_single_gemv_addr_to_catalog_addr",
            "evidence": BIND_PATH["icb"],
            "declared_not_executed": True,
            "falsifier": falsifier,
        },
        {
            "rank": 2,
            "id": "mixed_organ_occupancy_tiny_ba",
            "class": STATUS_PARTLY_REMOVABLE,
            "applies_to": "T2_single_gemv_addr_to_catalog_addr",
            "evidence": (
                "honest roof names tiny ba 96x5120. ORGAN_ROOF_LEDGER: occupancy-"
                "starved organs cannot reach DRAM roofs. Packing/batching is a "
                "layout change; the 96-row shape is the genome."
            ),
            "falsifier": (
                "same catalog addr_probe with ba rows padded/packed to the MLP "
                "tile vs current 96x5120; overlap ⇒ occupancy is not the loss"
            ),
        },
        {
            "rank": 3,
            "id": "many_tensors_vs_one_gemv",
            "class": STATUS_STRUCTURAL,
            "applies_to": "T2_single_gemv_addr_to_catalog_addr",
            "evidence": (
                "production GEMVs have distinct x vectors. The 13.6 GB single "
                "GEMV is diagnostic concatenation. Fusion of same-x pairs is "
                "rank 1, not this row."
            ),
        },
        {
            "rank": 4,
            "id": "q2_unpack_on_majority_bytes",
            "class": STATUS_STRUCTURAL,
            "applies_to": "T4_catalog_addr_to_atlas_effective",
            "evidence": (
                "ACCELERATOR_UNPACK_IS_THE_WALL + atlas pareto. Codec change is "
                "GRAVITY, not addressing. Direction cited, not re-measured."
            ),
            "status_on_this_drop": STATUS_UNATTRIBUTED,
        },
        {
            "rank": 5,
            "id": "tile_threadgroup_geometry",
            "class": STATUS_STRUCTURAL,
            "applies_to": "both, as a constant",
            "evidence": KERNEL_GEOMETRY,
            "note": "identical on the comparable T2 rungs; cannot explain T2",
        },
        {
            "rank": 6,
            "id": "gather_for_route_selection",
            "class": STATUS_REFUSED,
            "applies_to": "T2 and T4",
            "evidence": moe,
        },
        {
            "rank": 7,
            "id": "synchronization_between_organs",
            "class": STATUS_UNATTRIBUTED,
            "applies_to": "T4",
            "evidence": "no isolating probe on the same genome in the recover set",
        },
        {
            "rank": 8,
            "id": "descriptor_argument_buffer_setup",
            "class": STATUS_UNATTRIBUTED,
            "applies_to": "T2",
            "evidence": BIND_PATH["argument_buffers"],
        },
    ]

    pct_of_819 = None
    if (
        single_addr_median.get("status") == STATUS_LOADED
        and published.get("status") == STATUS_LOADED
    ):
        pct_of_819 = float(single_addr_median["value"]) / float(published["value"])

    next_workunits = [
        {
            "species": "STRUCTURAL_COST_COMPARE",
            "lane": "ANALYSIS",
            "work": (
                "this receipt. CPU reread. No GPU. Already the cheapest "
                "falsifier of 703.5-as-median and of 595.9-as-machine-roof."
            ),
            "state": "RUNNABLE_HERE",
        },
        {
            "species": "PROTECTED_AB",
            "lane": "GPU_PROTECTED",
            "work": falsifier,
            "state": "SLEEPING",
            "wake": "a distinct HCLI GPU_PROTECTED lane holds a proven lease; this sidecar never flocks",
        },
        {
            "species": "PROFILE_ACTIVE_BYTES",
            "lane": "GPU_PROTECTED",
            "work": (
                "same-genome probe: catalog addr_probe, catalog full, and "
                "production decode GPU timestamps on ONE artifact so T4 can "
                "leave UNATTRIBUTED"
            ),
            "state": "SLEEPING",
        },
    ]

    parse_s = time.perf_counter() - started
    negative_findings = [
        "receipts/headless/HONEST_ROOF_WEIGHT_ADDRESSING.json is unseen in HEAD; the measurements live under receipts/ascent-2026-08-16/",
        "receipts/headless/G044_ROOFLINE_KNEE.json is unseen in HEAD; G044 lives under receipts/ascent-2026-08-16/",
        "703.5 is not the median of q4_single_gemv_addr_probe at the 13.612 GB payload (median 699.57, max 703.61)",
        "lane min 681.0 / max 749.7 do not match that probe's min 693.15 / max 703.61",
        "reduced honest-roof catalog probes are null; 530 cannot be cited from the reduced receipt",
        "only two of the named recover receipts carry 9_878_901_136 (atlas, bandwidth_ascent); honest roof defends 13_611_663_360",
        "530→337 crosses genomes and is UNATTRIBUTED as a single addressing mechanism",
        "catalog decode MAX 513 is not a stable rung (median 454.9, spread 294–513)",
        "q2 organ-sized addr_probe ~968 GB/s exceeds published 819 and is refused as a DRAM roof",
        "HONEST unique_once 13.6 GB ~376 and ORGAN_ROOF_LEDGER 778.8 are both called unique-once and disagree",
        "this host is contaminated and holds no lease; absolute roofs in the cited receipts are contended (clean_box false)",
        "ICB is declared in metal/mod.rs and was not executed here; naming it is not evidence it ran",
        "no change is claimed to improve TPS; the tallest removable is a hypothesis with a falsifier",
    ]

    return {
        "path_adjudication": path_adjudication,
        "loads": {
            k: {"rel": v.get("rel"), "status": v.get("status"), "reason": v.get("reason"), "via": v.get("via")}
            for k, v in loads.items()
        },
        "active_bytes_adjudication": active_bytes_adjudication,
        "sourced_rungs": {
            "published_peak": published,
            "single_gemv_addr_median_13p6gb": single_addr_median,
            "single_gemv_addr_min_13p6gb": single_addr_min,
            "single_gemv_addr_max_13p6gb": single_addr_max,
            "single_gemv_decode_median_13p6gb": single_decode_median,
            "single_gemv_full_median_13p6gb": single_full_median,
            "catalog_addr_median": catalog_addr_median,
            "catalog_decode_max": catalog_decode_max,
            "catalog_decode_median": catalog_decode_median,
            "catalog_full_median": catalog_full_median,
            "tiled_same_shape_addr_median": tiled_addr,
            "unique_once_13p6gb": unique_once_13p6,
            "atlas_effective": atlas_effective,
            "atlas_active_bytes": atlas_bytes,
            "production_decode_gpu_tpr64": prod_decode_gpu,
            "reduced_catalog_addr": reduced_catalog,
        },
        "lane_claim_adjudication": {
            "published_819": claimed_819,
            "single_gemv_703p5_as_median": claimed_703,
            "catalog_addr_530p7": claimed_530,
            "catalog_decode_513_as_max": claimed_513,
            "atlas_effective_337p3": claimed_337,
        },
        "silicon_vs_datasheet": {
            "fraction_of_published_peak": pct_of_819,
            "reading": (
                "the 13.612 GB single-GEMV addr median over published peak. "
                "Honest roof wrote 699.6/819 = 85%. The unused 15% is not a "
                "catalog bug."
            ),
        },
        "transitions": [transition_703_to_530, transition_530_to_337],
        "loss_chain_closure": chain_close,
        "roofs": roofs,
        "ceilings": ceilings,
        "atlas_ceiling_check": atlas_ceiling_check,
        "target_raw_tps_for_50_accepted_at_30_of_43": atlas_need,
        "recorded_raw_tps": atlas_recorded_tps,
        "removable_ranking": removable_ranking,
        "tallest_removable": tallest,
        "kernel_geometry": KERNEL_GEOMETRY,
        "bind_path": BIND_PATH,
        "moe_gather": moe,
        "q2_organ_addr_probe_as_dram_roof": q2_addr_as_dram,
        "next_workunits": next_workunits,
        "self_timing": {
            "cpu_parse_s": parse_s,
            "not": (
                "a GPU measurement, a lease, a TPS, a roof, or evidence the "
                "cited probes still hold on this contaminated host"
            ),
            "class": "SELF_MEASURED_DIRTY",
        },
        "negative_findings": negative_findings,
        "contamination_of_cited_honest_roof": (
            nested(honest.get("doc") or {}, "contamination_note")
            if honest.get("status") == STATUS_LOADED
            else "honest_roof_not_loaded"
        ),
        "honest_timing_label": (
            nested(honest.get("doc") or {}, "timing_label")
            if honest.get("status") == STATUS_LOADED
            else None
        ),
        "honest_clean_box": (
            nested(honest.get("doc") or {}, "clean_box")
            if honest.get("status") == STATUS_LOADED
            else None
        ),
    }


def build() -> Any:
    analysis = analyze()
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Attribute the measured drop from a clean Q4 GEMV addressing probe "
            "to production-decode effective weight bandwidth, recompute the "
            "atlas TPS ceiling against every candidate roof with the roof named, "
            "and rank what is removable. No hardware measurement."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        **analysis,
        "recovered_implementation": [
            "tools/future/_common.py write_receipt / HARDWARE_FIELDS / git()",
            "tools/future/frontiers.py load_optional (sparse-absent is not absence)",
            "tools/future/hbm_doctor.py load_repo_json pattern and ATLAS/CENSUS paths",
            "tools/future/qwen27_profile_schema.py ATLAS_REL",
            "receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.json (the measurements)",
            "receipts/headless/ACCELERATOR_TOKEN_BYTES_ATLAS.json (337 derived, 589.73 roof, 9.879e9 bytes)",
            "receipts/headless/BANDWIDTH_ASCENT.json (356.7 production decode GPU, q2 organ probes)",
            "receipts/headless/NOETIC_ORGAN_CENSUS.json (595.9 anchor)",
            "receipts/ascent-2026-08-16/G072_MULTI_PLANE_GEMV.json (595.9 family scoring)",
            "receipts/ascent-2026-08-16/G044_ROOFLINE_KNEE.json (594.35 f4 sweep)",
            "receipts/headless/MACHINE_GENOME.json (589.73 f32 triad, not a SoC roof)",
            "receipts/headless/ORGAN_ROOF_LEDGER.json (819 / 778.8 / 729.7 never collapsed)",
            "receipts/headless/ORGAN_BANDWIDTH.json (mlp_gate_up holds the 356→778 gap share)",
            "receipts/headless/ACCELERATOR_UNPACK_IS_THE_WALL.json (q2 unpack vs Q4 addressing)",
            "crates/hawking-core/shaders/qwen_uniform_q4.metal (addr/decode/full geo_tpr64_tg128)",
            "crates/hawking-core/src/metal/mod.rs (kernel catalog, per-dispatch bind, ICB not wired)",
        ],
        "gaps_closed": [
            "no sidecar module adjudicated 703.5 against the honest-roof median",
            "no sidecar module kept T2 (same-receipt catalog topology) separate from T4 (cross-genome 530→337)",
            "atlas 59.70 raw-TPS ceiling was computed against 589.73 without a table of the other roofs",
            "lane-named headless honest-roof / G044 paths are absent in HEAD; ascent copies recovered",
            "unattributed remainder of 699→337 is labelled UNATTRIBUTED rather than spread across mechanisms",
        ],
        "resident_callable": {
            "entry_point": "tools.future.addressing_gap.analyze() / build()",
            "workunit": (
                "one CPU_ANALYSIS unit; reread sealed receipts; write "
                "ADDRESSING_GAP.json; no GPU, no flock, no TPS"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.ACTIVE_BYTES",
            "fails_closed": (
                "unsourced number → REFUSED; missing receipt → REFUSED rung; "
                "ceiling without a named sourced roof → UnstatedRoof; "
                "TPS change without falsifier → TpsClaimRefused; "
                "loss across genomes → UNATTRIBUTED; "
                "GB/s above published peak → refused as DRAM roof"
            ),
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
