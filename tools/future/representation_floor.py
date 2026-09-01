#!/usr/bin/env python3
"""HOW MUCH SMALLER CAN THE INCUMBENT GET, and is another compression campaign worth it?

The 10.554 GB sealed-3.14 payload is decomposed from MIX_REPORT + the catalog
census, billed at the measured per-stream rates, and compared with the 16.8%
matvec-byte cut that 60 TPS still needs after perfect decode-arithmetic
removal. Every hardware number is read from a receipt. Missing input REFUSES.

    python3 tools/future/representation_floor.py --build
    python3 -m pytest tools/future/test_representation_floor.py -q

This module MEASURES NOTHING. It is STATIC_ONLY arithmetic over cited receipts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402

RECORDED_BY = "tools/future/representation_floor.py"
RECEIPT_NAME = "REPRESENTATION_FLOOR.json"
SCHEMA = "hawking.future.representation_floor.v1"

MIX_REPORT = Path("/Users/scammermike/noetic/NOETIC_PARENT_A/MIX_REPORT.json")

GAP_REL = "receipts/future/GAP_LEDGER_60.json"
ECON_REL = "receipts/future/ECONOMICS_CALIBRATION.json"
CODE_REL = "receipts/future/MLP_CODE_INFORMATION.json"
AUX_REL = "receipts/future/MLP_AUXILIARY_INFORMATION.json"
CENSUS_REL = "receipts/future/MLP_BYTE_CENSUS.json"
SCREEN_REL = "receipts/future/AUX_CAPABILITY_SCREEN.json"
DN_REL = "receipts/future/DELTANET_REPRESENTATION.json"
SPARSE_REL = "receipts/future/MLP_SPARSE_RESIDUAL.json"
OPCLASS_REL = "receipts/future/OP_CLASS_ABLATION.json"
LOAD_REL = "receipts/future/SPECIMEN_LOAD_COST.json"
BITCAST_REL = "receipts/future/BITCAST_DEQUANT_AB.json"
Q4_BITCAST_REL = "receipts/future/Q4_BITCAST_AB.json"
LADDER_REL = "receipts/future/FLASH_BPW_LADDER.json"
U8_REL = "receipts/future/AUX_U8_NATIVE.json"

REQUIRED_RELS = (
    GAP_REL, ECON_REL, CODE_REL, AUX_REL, CENSUS_REL, SCREEN_REL, DN_REL,
    SPARSE_REL, OPCLASS_REL, LOAD_REL, BITCAST_REL, Q4_BITCAST_REL,
    LADDER_REL, U8_REL,
)

MEASURED = "MEASURED"
REFUTED = "REFUTED"
UNTESTED = "UNTESTED"
EVIDENCE_STATUSES = frozenset({MEASURED, REFUTED, UNTESTED})

STREAM_WEIGHT_CODES = "weight_codes"
STREAM_BROADCAST_AUX = "broadcast_aux"
STREAM_ACTIVATION = "activation"

# S025: a perfect win below this is not worth an hour. Same constant gap_ledger_60.
MATERIAL_MS = 1.0
# S025's other bar: 5-10% of important bytes. The low edge is the gate.
SIZE_MATERIAL_FRAC = 0.05

GB = 1e9
COHERENT_FLOOR_ID = "bpw_2_25"
COMPOSITE_SCAR = "COMPOSITE_MLP_SIMPLE_LINEAR_LOW_RANK_REFUTED"


class FloorRefused(RuntimeError):
    """An input is missing or self-inconsistent; guessing would be a fake floor."""


def _repo_path(rel: str) -> Path:
    return Path(rel) if Path(rel).is_absolute() else REPO / rel


def _load(rel: str, *, why: str) -> dict[str, Any]:
    p = _repo_path(rel)
    if not p.is_file():
        raise FloorRefused(
            f"{rel} is not on disk; {why}. A floor with a missing input is a guess "
            "wearing a receipt"
        )
    d = json.loads(p.read_text())
    if not isinstance(d, dict):
        raise FloorRefused(f"{rel} is not a JSON object")
    return d


def _need(d: dict[str, Any], *keys: str, source: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise FloorRefused(f"{source} is missing {'.'.join(keys)}")
        cur = cur[k]
    return cur


def _r(value: float, n: int = 6) -> float:
    out = round(float(value), n)
    return 0.0 if out == 0.0 else out


def _gb(n_bytes: int | float) -> float:
    return _r(float(n_bytes) / GB, 6)


def _bpw(n_bytes: int | float, n_params: int) -> float:
    if n_params <= 0:
        raise FloorRefused("parent_params is not positive; bpw is undefined")
    return _r(float(n_bytes) * 8.0 / float(n_params), 6)


# ---------------------------------------------------------------------------
# Inputs. Each loader REFUSES rather than defaulting.
# ---------------------------------------------------------------------------

def mix_report() -> dict[str, Any]:
    d = _load(str(MIX_REPORT), why="the incumbent artifact report is the payload authority")
    for k in ("payload_bytes", "parent_params", "mlp_elements", "affine_bytes",
              "q4_bytes", "f32_bytes", "storage_bpw"):
        if k not in d:
            raise FloorRefused(f"MIX_REPORT.json is missing {k}")
    payload = int(d["payload_bytes"])
    parts = int(d["affine_bytes"]) + int(d["q4_bytes"]) + int(d["f32_bytes"])
    if parts != payload:
        raise FloorRefused(
            f"MIX_REPORT parts {parts} != payload_bytes {payload}; the split does "
            "not add and must not be billed"
        )
    return d


def stream_rates() -> dict[str, Any]:
    d = _load(ECON_REL, why="stream-class ms/GB is the time-cost authority")
    classes = _need(d, "stream_classes", source=ECON_REL)
    out = {}
    for name in (STREAM_WEIGHT_CODES, STREAM_BROADCAST_AUX, STREAM_ACTIVATION):
        row = classes.get(name)
        if not isinstance(row, dict) or "ms_per_gb_saved" not in row:
            raise FloorRefused(f"{ECON_REL} stream_classes.{name}.ms_per_gb_saved is absent")
        out[name] = {
            "ms_per_gb": float(row["ms_per_gb_saved"]),
            "ms_per_gb_measured": float(row.get("ms_per_gb_saved_measured", row["ms_per_gb_saved"])),
            "on_critical_path": bool(row.get("on_critical_path")),
            "source": ECON_REL,
        }
    if out[STREAM_BROADCAST_AUX]["ms_per_gb"] != 0.0:
        raise FloorRefused(
            f"{ECON_REL} broadcast_aux ms/GB is {out[STREAM_BROADCAST_AUX]['ms_per_gb']}, "
            "not 0; the size-not-time reading would be a different measurement"
        )
    return out


def _family(census: dict[str, Any], name: str) -> dict[str, Any]:
    rows = _need(census, "census", "by_organ_family", source=CENSUS_REL)
    for r in rows:
        if r.get("family") == name:
            return r
    raise FloorRefused(f"{CENSUS_REL} has no by_organ_family {name}")


def _organ(census: dict[str, Any], name: str) -> dict[str, Any]:
    rows = _need(census, "census", "by_organ", source=CENSUS_REL)
    for r in rows:
        if r.get("organ") == name:
            return r
    raise FloorRefused(f"{CENSUS_REL} has no organ {name}")


def q4_layout() -> dict[str, Any]:
    """HQ30UQ4 g64 layout, measured on DeltaNet qkvz, named as the attention codec."""
    dn = _load(DN_REL, why="DeltaNet accounting is the HQ30UQ4 layout authority")
    mix = mix_report()
    attn = str(_need(mix, "recipe", "attention", source=str(MIX_REPORT)))
    if "HQ30UQ4" not in attn:
        raise FloorRefused(f"MIX_REPORT attention codec is {attn!r}, not HQ30UQ4")
    qkvz = _need(dn, "accounting", "by_organ", "attention.linear_qkvz", source=DN_REL)
    elements = int(qkvz["elements"])
    n_tensors = int(qkvz["n_tensors"])
    header_bytes = int(qkvz["header_bytes"])
    code_bytes = int(qkvz["code_bytes"])
    scale_bytes = int(qkvz["scale_bytes"])
    if elements <= 0 or n_tensors <= 0 or scale_bytes <= 0:
        raise FloorRefused(f"{DN_REL} qkvz layout is degenerate")
    header_per = header_bytes / n_tensors
    code_bits = code_bytes * 8.0 / elements
    group = int(round(2.0 * elements / scale_bytes))
    if abs(code_bits - 4.0) > 1e-9 or group != 64:
        raise FloorRefused(
            f"{DN_REL} qkvz is not 4-bit / group-64 (code_bits={code_bits}, group={group})"
        )
    if abs(header_per - round(header_per)) > 1e-9:
        raise FloorRefused(f"{DN_REL} header_bytes {header_bytes} does not divide n_tensors")
    return {
        "header_bytes_per_tensor": int(round(header_per)),
        "group": group,
        "code_bits": 4,
        "scale_bytes_per_group": 2,
        "codec_named_by": attn,
        "source": DN_REL,
    }


def split_q4_storage(storage_bytes: int, n_tensors: int, layout: dict[str, Any]) -> dict[str, int]:
    header_per = int(layout["header_bytes_per_tensor"])
    group = int(layout["group"])
    header = n_tensors * header_per
    body = int(storage_bytes) - header
    if body <= 0:
        raise FloorRefused(f"q4 storage {storage_bytes} does not cover {header} header bytes")
    coeff = 0.5 + (2.0 / group)
    elements_f = body / coeff
    elements = int(round(elements_f))
    if abs(elements_f - elements) > 1e-6:
        raise FloorRefused(
            f"q4 storage {storage_bytes} / {n_tensors} tensors is not an HQ30UQ4 g64 body"
        )
    codes = (elements * 4) // 8
    scale = (elements * 2) // group
    if codes + scale + header != int(storage_bytes):
        raise FloorRefused(
            f"q4 split {codes}+{scale}+{header} != storage {storage_bytes}"
        )
    return {
        "elements": elements,
        "code_bytes": codes,
        "scale_bytes": scale,
        "header_bytes": header,
        "storage_bytes": int(storage_bytes),
        "n_tensors": int(n_tensors),
    }


def byte_classes() -> dict[str, Any]:
    """Decompose the MIX_REPORT payload into stream classes. Nothing typed."""
    mix = mix_report()
    census_doc = _load(CENSUS_REL, why="the catalog census is the organ split")
    census = census_doc["census"]
    packing = _need(census, "mlp", "incumbent_packing", source=CENSUS_REL)
    rates = stream_rates()
    layout = q4_layout()

    payload = int(mix["payload_bytes"])
    catalog = int(census["catalog_total_bytes"])
    if catalog != payload:
        raise FloorRefused(
            f"census catalog_total_bytes {catalog} != MIX_REPORT payload {payload}"
        )
    mlp_storage = int(_family(census_doc, "mlp")["storage_bytes"])
    if mlp_storage != int(mix["affine_bytes"]):
        raise FloorRefused(
            f"census mlp storage {mlp_storage} != MIX_REPORT affine_bytes {mix['affine_bytes']}"
        )
    mlp_codes = int(packing["code_bytes"])
    mlp_aux = int(packing["scale_bias_and_header_bytes"])
    if mlp_codes + mlp_aux != mlp_storage:
        raise FloorRefused("MLP code+aux does not equal MLP storage")

    q4_organs = (
        "attention.q", "attention.k", "attention.v", "attention.o",
        "attention.linear_qkvz", "attention.linear_ba", "attention.linear_out",
        "embedding", "lm_head",
    )
    q4_codes = q4_scale = q4_header = 0
    q4_rows = []
    for name in q4_organs:
        row = _organ(census_doc, name)
        split = split_q4_storage(int(row["storage_bytes"]), int(row["n_tensors"]), layout)
        q4_codes += split["code_bytes"]
        q4_scale += split["scale_bytes"]
        q4_header += split["header_bytes"]
        q4_rows.append({"organ": name, **split, "active_bytes": int(row["active_bytes"])})
    q4_aux = q4_scale + q4_header
    q4_storage = q4_codes + q4_aux
    if q4_storage != int(mix["q4_bytes"]):
        raise FloorRefused(
            f"HQ30UQ4 split {q4_storage} != MIX_REPORT q4_bytes {mix['q4_bytes']}"
        )

    conv1d = int(_organ(census_doc, "attention.linear_conv1d")["storage_bytes"])
    norms = int(_family(census_doc, "norms")["storage_bytes"])
    state = int(_family(census_doc, "state")["storage_bytes"])
    f32 = conv1d + norms + state
    if f32 != int(mix["f32_bytes"]):
        raise FloorRefused(f"f32 organs {f32} != MIX_REPORT f32_bytes {mix['f32_bytes']}")

    unread = int(census["unread_embedding_table_bytes"])
    embed_active = int(_organ(census_doc, "embedding")["active_bytes"])

    def billed(n_bytes: int, stream: str, streamed: bool) -> dict[str, Any]:
        rate = rates[stream]["ms_per_gb"]
        return {
            "bytes": int(n_bytes),
            "gb": _gb(n_bytes),
            "stream_class": stream,
            "ms_per_gb": rate,
            "ms_per_token_if_streamed": _r(_gb(n_bytes) * rate, 6) if streamed else 0.0,
            "streamed_per_decode_token": streamed,
            "source_rate": ECON_REL,
        }

    classes = [
        {"id": "mlp_weight_codes", "part": "affine q2 codes",
         **billed(mlp_codes, STREAM_WEIGHT_CODES, True), "source": CENSUS_REL},
        {"id": "mlp_broadcast_aux", "part": "affine q2 scale/bias/header",
         **billed(mlp_aux, STREAM_BROADCAST_AUX, True), "source": CENSUS_REL},
        {"id": "q4_weight_codes", "part": "HQ30UQ4 codes (attention + DeltaNet + lm_head + embed table)",
         **billed(q4_codes, STREAM_WEIGHT_CODES, True), "source": DN_REL},
        {"id": "q4_broadcast_aux", "part": "HQ30UQ4 scale/header",
         **billed(q4_aux, STREAM_BROADCAST_AUX, True), "source": DN_REL},
        {"id": "f32_conv1d", "part": "DeltaNet conv1d f32 weights",
         **billed(conv1d, STREAM_WEIGHT_CODES, True), "source": CENSUS_REL},
        {"id": "f32_norms_and_state", "part": "norms + A_log + dt_bias",
         **billed(norms + state, STREAM_BROADCAST_AUX, True), "source": CENSUS_REL},
        {"id": "activation", "part": "activations / KV / DeltaNet recurrent state (not in catalog)",
         **billed(0, STREAM_ACTIVATION, False), "source": CENSUS_REL,
         "catalog_bytes": "UNKNOWN",
         "note": census["state_not_in_catalog"]["reason"]},
    ]
    summed = sum(int(c["bytes"]) for c in classes)
    if summed != payload:
        raise FloorRefused(f"byte classes sum {summed} != payload {payload}")

    by_stream = {STREAM_WEIGHT_CODES: 0, STREAM_BROADCAST_AUX: 0, STREAM_ACTIVATION: 0}
    for c in classes:
        by_stream[c["stream_class"]] += int(c["bytes"])

    attn_storage = int(_family(census_doc, "attention")["storage_bytes"])
    attn_q4_codes = sum(
        r["code_bytes"] for r in q4_rows
        if r["organ"].startswith("attention.")
    )
    return {
        "payload_bytes": payload,
        "payload_gb": _gb(payload),
        "parent_params": int(mix["parent_params"]),
        "storage_bpw": float(mix["storage_bpw"]),
        "mlp_elements": int(mix["mlp_elements"]),
        "classes": classes,
        "by_stream_class": {
            k: {"bytes": v, "gb": _gb(v), "ms_per_gb": rates[k]["ms_per_gb"]}
            for k, v in by_stream.items()
        },
        "mlp_code_bytes": mlp_codes,
        "mlp_aux_bytes": mlp_aux,
        "q4_code_bytes": q4_codes,
        "q4_aux_bytes": q4_aux,
        "attention_q4_code_bytes": attn_q4_codes,
        "attention_storage_bytes": attn_storage,
        "mlp_storage_bytes": mlp_storage,
        "f32_bytes": f32,
        "unread_embedding_table_bytes": unread,
        "embedding_active_bytes": embed_active,
        "q4_organs": q4_rows,
        "q4_layout": layout,
        "reconciliation": {
            "classes_sum_equals_payload": True,
            "mlp_equals_affine_bytes": True,
            "q4_split_equals_q4_bytes": True,
            "f32_organs_equal_f32_bytes": True,
        },
    }


def _ms_saved(n_bytes: int, stream: str, rates: dict[str, Any]) -> float:
    return _r(_gb(n_bytes) * rates[stream]["ms_per_gb"], 6)


def _candidate(
    *,
    ident: str,
    name: str,
    bytes_saved: int,
    stream_class: str,
    evidence_status: str,
    source: str,
    overlap_bucket: str,
    mechanism: str,
    notes: str,
    rates: dict[str, Any],
    counts_toward_measured_safe: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if evidence_status not in EVIDENCE_STATUSES:
        raise FloorRefused(f"{ident} has illegal evidence_status {evidence_status}")
    if not source.startswith("receipts/"):
        raise FloorRefused(f"{ident} source {source} is not under receipts/")
    if evidence_status == REFUTED:
        counts_toward_measured_safe = False
    row = {
        "id": ident,
        "name": name,
        "bytes_saved": int(bytes_saved),
        "gb_saved": _gb(bytes_saved),
        "ms_saved": _ms_saved(bytes_saved, stream_class, rates),
        "stream_class": stream_class,
        "evidence_status": evidence_status,
        "source": source,
        "overlap_bucket": overlap_bucket,
        "mechanism": mechanism,
        "notes": notes,
        "counts_toward_measured_safe": bool(counts_toward_measured_safe)
        and evidence_status == MEASURED
        and int(bytes_saved) > 0,
    }
    if extra:
        row.update(extra)
    return row


def coherent_floor_bpw() -> float:
    ladder = _load(LADDER_REL, why="the 2.25 coherent-class floor is a ladder rung")
    for r in ladder.get("rungs") or []:
        if r.get("id") == COHERENT_FLOOR_ID:
            bpw = float(r["target_bpw"])
            if abs(bpw - 2.25) > 1e-9:
                raise FloorRefused(f"{LADDER_REL} {COHERENT_FLOOR_ID} target_bpw is {bpw}, not 2.25")
            return bpw
    raise FloorRefused(f"{LADDER_REL} has no rung {COHERENT_FLOOR_ID}")


def candidates() -> list[dict[str, Any]]:
    rates = stream_rates()
    bc = byte_classes()
    mix = mix_report()
    codes = _load(CODE_REL, why="the 1.87-bit code entropy is the lossless floor")
    aux = _load(AUX_REL, why="the group-size curve and aux candidates live here")
    screen = _load(SCREEN_REL, why="group_size_256/1024 capability is the screen")
    dn = _load(DN_REL, why="q4 lower-bitwidth byte models live here")
    sparse = _load(SPARSE_REL, why="the composite low-rank scar is recorded here")
    bitcast = _load(BITCAST_REL, why="the q2 bitcast is a measured TIME lever, 0 bytes")
    q4bc = _load(Q4_BITCAST_REL, why="the q4 bitcast is a measured TIME lever, 0 bytes")
    u8n = _load(U8_REL, why="aux u8 native is the measured size-not-time wall")
    census_doc = _load(CENSUS_REL, why="the three 2.25-bpw measurements are cited here")

    floor = _need(codes, "floor", source=CODE_REL)
    entropy_row = next(
        (c for c in codes.get("candidates") or []
         if c.get("id") == "entropy_coded_code_stream"),
        None,
    )
    if not isinstance(entropy_row, dict):
        raise FloorRefused(f"{CODE_REL} has no entropy_coded_code_stream candidate")
    entropy_saved = int(floor["iid_redundant_bytes_rounded"])
    h_q = float(floor["iid_shannon_bits_per_code"])
    indep = float(floor["independent_fraction"])
    stored_codes = int(floor["stored_code_bytes"])
    zlib_mean = float(
        _need(entropy_row, "measured", "sample_zlib_ratio", "mean", source=CODE_REL)
    )
    nns022 = bool(_need(entropy_row, "measured", "nns022_reopen_fires", source=CODE_REL))
    if stored_codes != bc["mlp_code_bytes"]:
        raise FloorRefused(
            f"{CODE_REL} stored_code_bytes {stored_codes} != census mlp codes "
            f"{bc['mlp_code_bytes']}"
        )

    target_bpw = coherent_floor_bpw()
    mlp_elements = int(mix["mlp_elements"])
    header_bytes = int(_need(aux, "accounting", "header_bytes", source=AUX_REL))
    # 2.25 bpw is a parameter-body figure; the 192 tensor headers still exist.
    # body + headers equals the G=128 packing on the aux curve.
    bytes_at_2_25 = int(round(target_bpw * mlp_elements / 8.0)) + header_bytes
    mlp_to_floor = int(mix["affine_bytes"]) - bytes_at_2_25
    if mlp_to_floor <= 0:
        raise FloorRefused("2.25 bpw is not below the incumbent MLP packing")

    curve = {int(r["group_size"]): r for r in aux.get("group_size_curve") or []}
    for g in (128, 256, 1024):
        if g not in curve:
            raise FloorRefused(f"{AUX_REL} group_size_curve is missing G={g}")
    g128_active = int(curve[128]["mlp_active_bytes"])
    if bytes_at_2_25 != g128_active:
        raise FloorRefused(
            f"2.25 bpw + headers {bytes_at_2_25} != G=128 packing {g128_active}; "
            "the coherent-floor byte model is not the measured group-128 body"
        )

    levers = {str(r["id"]): r for r in screen.get("levers") or []}
    for ident in ("group_size_256", "group_size_1024"):
        if ident not in levers:
            raise FloorRefused(f"{SCREEN_REL} is missing lever {ident}")
        if levers[ident].get("status") != REFUTED:
            raise FloorRefused(
                f"{SCREEN_REL} {ident} status is {levers[ident].get('status')!r}, not REFUTED"
            )

    aux_by_id = {str(r["id"]): r for r in aux.get("candidates") or []}
    dn_by_id = {str(r["id"]): r for r in dn.get("candidates") or []}
    dead = sparse.get("dead_schools_not_rebuilt") or []
    if not any(COMPOSITE_SCAR in str(x) for x in dead):
        raise FloorRefused(f"{SPARSE_REL} does not record {COMPOSITE_SCAR}")

    u8_bytes = int(_need(levers["quantize_aux_u8"], "bytes_removed", source=SCREEN_REL))
    u8_proj = float(
        _need(u8n, "gpu_ab", "token_projection", "projected_mlp_delta_ms", source=U8_REL)
    )

    q3_all_q4 = bc["q4_code_bytes"] // 4
    q3_attn = bc["attention_q4_code_bytes"] // 4
    q3_qkvz = int(dn_by_id["lower_bit_uniform_qkvz"]["bytes_eliminated_if_true"])
    q3_out = int(dn_by_id["lower_bit_out_proj"]["bytes_eliminated_if_true"])
    q3_het = int(dn_by_id["heterogeneous_qkvz_bits"]["bytes_eliminated_if_true"])
    q4_g128 = int(dn_by_id["larger_q4_group"]["bytes_eliminated_if_true"])
    pack_headers = int(aux_by_id["pack_headers"]["bytes_eliminated_if_true"])

    bitcast_ms = float(_need(bitcast, "timing", "ms_saved", source=BITCAST_REL))
    q4bc_ms = float(_need(q4bc, "resident_measured", "gpu_ms_saved", source=Q4_BITCAST_REL))
    token_q2 = bool(_need(bitcast, "fp_boundary", "token_identical", source=BITCAST_REL))
    token_q4 = bool(_need(q4bc, "resident_measured", "token_identical", source=Q4_BITCAST_REL))
    bit_id_q4 = bool(_need(q4bc, "measured", "bit_identical", source=Q4_BITCAST_REL))

    lower_bit_family = None
    for fam in census_doc.get("families") or []:
        if fam.get("id") == "lower_bit":
            lower_bit_family = fam
            break
    if lower_bit_family is None:
        raise FloorRefused(f"{CENSUS_REL} has no lower_bit family (the 2.25 citations)")

    three_ways = (
        "QN-SHARED-BASIS-DENSITY: no shared-basis K below ~2.25 bpw is coherent "
        "at held-out activation; QN-COORDINATE-TRANSFORM: ~2.25 bpw held under "
        "tested rotation families; QN-BINARY-INJURY: 1.25-bpw binary is "
        "generation-incoherent, 0/4 healers, vs q2f_g64."
    )

    out = [
        _candidate(
            ident="entropy_code_mlp_codes",
            name="entropy-code the MLP 2-bit codes to the measured 1.87-bit Shannon floor",
            bytes_saved=entropy_saved,
            stream_class=STREAM_WEIGHT_CODES,
            evidence_status=MEASURED,
            source=CODE_REL,
            overlap_bucket="mlp_codes",
            mechanism=(
                f"i.i.d. H(q)={h_q:.6f} bits per 2 stored bits; independent_fraction="
                f"{indep:.6f}. {entropy_saved} of {stored_codes} code bytes are the "
                "4-level histogram bias, not sharing."
            ),
            notes=(
                "Lossless of the stored q. Fused rANS/Huffman affine2 is UNTESTED as a "
                "kernel (NNS-022 cousin: entropy coding is not a TPS lever without a "
                "native register-decodable path). Decode-to-incumbent-2-bit then bind "
                "affine2 eliminates zero active bytes. zlib on sampled tensors reaches "
                f"mean ratio {zlib_mean:.6f} of stored, same single-digit percent. "
                "ms_saved is billed at weight_codes IF a fused decode existed; "
                "rematerialized it is 0.000."
            ),
            rates=rates,
            counts_toward_measured_safe=True,
            extra={
                "H_q_bits": h_q,
                "independent_fraction": indep,
                "sample_zlib_ratio_mean": zlib_mean,
                "ms_saved_if_rematerialized": 0.0,
                "nns022_reopen_fires": nns022,
            },
        ),
        _candidate(
            ident="mlp_to_2_25_coherent_floor",
            name="MLP packing down to the measured 2.25 bpw coherent-class floor (q2f)",
            bytes_saved=mlp_to_floor,
            stream_class=STREAM_BROADCAST_AUX,
            evidence_status=UNTESTED,
            source=LADDER_REL,
            overlap_bucket="mlp_aux",
            mechanism=(
                f"incumbent affine_bytes {mix['affine_bytes']} at ~2.50 bpw; "
                f"{target_bpw} bpw × {mlp_elements} elements / 8 = {bytes_at_2_25} bytes. "
                f"{three_ways}"
            ),
            notes=(
                "The FLOOR is MEASURED (nothing composed below 2.25). The MOVE from "
                "this mix's 2.5 affine-LS packing to a 2.25 packing of THIS artifact "
                "is UNTESTED as a generate-gated complete executable "
                f"({LADDER_REL} rung {COHERENT_FLOOR_ID} is RESEARCH_TARGET / "
                "qualified_complete_physical_ebpw UNKNOWN). Broadcast_aux, so ms_saved "
                "is 0 even if the bytes disappear. drop_bias is MEASURED_NEGATIVE in "
                f"{AUX_REL}; this is not 'delete the bias field'."
            ),
            rates=rates,
            counts_toward_measured_safe=False,
            extra={
                "target_bpw": target_bpw,
                "bytes_at_target": bytes_at_2_25,
                "three_ways": three_ways,
                "floor_status": MEASURED,
                "move_status": UNTESTED,
            },
        ),
        _candidate(
            ident="larger_affine_group_128",
            name="affine group 128 (aux halved vs G=64)",
            bytes_saved=int(curve[128]["bytes_eliminated_vs_incumbent"]),
            stream_class=STREAM_BROADCAST_AUX,
            evidence_status=UNTESTED,
            source=AUX_REL,
            overlap_bucket="mlp_aux",
            mechanism="Same 2-bit codes, half as many f16 scale/bias groups.",
            notes=(
                "Capability UNMEASURED on the group-size curve. Between incumbent G=64 "
                "and G=256 which FAILED_HELDOUT, so it is the only coarser group that "
                "is not already refuted. Overlaps mlp_to_2_25 / quantize_aux_u8."
            ),
            rates=rates,
            counts_toward_measured_safe=False,
            extra={"group_size": 128, "capability_on_curve": curve[128]["capability"]},
        ),
        _candidate(
            ident="larger_affine_group_256",
            name="affine group 256",
            bytes_saved=int(levers["group_size_256"]["bytes_removed"]),
            stream_class=STREAM_BROADCAST_AUX,
            evidence_status=REFUTED,
            source=SCREEN_REL,
            overlap_bucket="mlp_aux",
            mechanism="Same 2-bit codes, 4× fewer scale/bias groups than G=64.",
            notes=(
                "Capability FAILED_HELDOUT: organ relfro_mean "
                f"{levers['group_size_256']['organ_space']['relfro_mean']:.4f} vs bar "
                f"{levers['group_size_256']['organ_space']['relfro_bar']}. Broadcast_aux, "
                "so even a pass would have saved SIZE not TIME."
            ),
            rates=rates,
            counts_toward_measured_safe=False,
            extra={
                "group_size": 256,
                "capability": levers["group_size_256"]["capability"],
            },
        ),
        _candidate(
            ident="larger_affine_group_1024",
            name="affine group 1024",
            bytes_saved=int(levers["group_size_1024"]["bytes_removed"]),
            stream_class=STREAM_BROADCAST_AUX,
            evidence_status=REFUTED,
            source=SCREEN_REL,
            overlap_bucket="mlp_aux",
            mechanism="Same 2-bit codes, 16× fewer scale/bias groups than G=64.",
            notes=(
                "Capability FAILED_HELDOUT: organ relfro_mean "
                f"{levers['group_size_1024']['organ_space']['relfro_mean']:.4f} vs bar "
                f"{levers['group_size_1024']['organ_space']['relfro_bar']}. G=512 was "
                "not screened; G=256 already dying makes G=512 a predicted fail, not "
                "an UNTESTED gift."
            ),
            rates=rates,
            counts_toward_measured_safe=False,
            extra={
                "group_size": 1024,
                "capability": levers["group_size_1024"]["capability"],
            },
        ),
        _candidate(
            ident="quantize_aux_u8",
            name="quantize MLP scale/bias from f16 to u8 (held-out fit passed)",
            bytes_saved=u8_bytes,
            stream_class=STREAM_BROADCAST_AUX,
            evidence_status=UNTESTED,
            source=SCREEN_REL,
            overlap_bucket="mlp_aux",
            mechanism="Keep 2-bit codes; requant per-group aux to u8 + endpoints.",
            notes=(
                "Held-out organ/logit screen MEASURED_ON_HELDOUT / OPEN. Generate "
                "identity UNMEASURED. Native decode was SLOWER: "
                f"{U8_REL} projected_mlp_delta_ms={u8_proj} (positive is a loss). "
                "Bytes bill at 0.000 ms/GB. Overlaps mlp_to_2_25."
            ),
            rates=rates,
            counts_toward_measured_safe=False,
            extra={
                "heldout_status": levers["quantize_aux_u8"]["status"],
                "generate_gate": levers["quantize_aux_u8"]["generate_gate"],
                "native_projected_mlp_delta_ms": u8_proj,
            },
        ),
        _candidate(
            ident="q4_uniform_q3_attention_deltanet",
            name="q4 -> uniform Q3 on attention + DeltaNet codes",
            bytes_saved=q3_attn,
            stream_class=STREAM_WEIGHT_CODES,
            evidence_status=UNTESTED,
            source=DN_REL,
            overlap_bucket="q4_codes",
            mechanism=(
                "HQ30UQ4 codes are 4 bits/element. Uniform Q3 keeps 3/4 of those "
                f"code bytes on the attention family ({bc['attention_q4_code_bytes']} "
                "code bytes). Scales stay f16 per group-64."
            ),
            notes=(
                "NNS-029 killed uniform bit-descent below q3 as a clean path on the "
                "MLP / whole artifact (cousin, not this organ). DN-only native Q3 is "
                f"OPEN in {DN_REL} (qkvz {q3_qkvz} + out {q3_out}). GQA Q3 is the "
                "same codec, same UNTESTED. Capability of a uniform Q3 of this mix "
                "is not measured."
            ),
            rates=rates,
            counts_toward_measured_safe=False,
            extra={
                "dn_qkvz_q3_bytes": q3_qkvz,
                "dn_out_q3_bytes": q3_out,
                "attention_q4_code_bytes": bc["attention_q4_code_bytes"],
            },
        ),
        _candidate(
            ident="q4_uniform_q3_all_q4_codes",
            name="q4 -> uniform Q3 on every HQ30UQ4 code (attention, DeltaNet, lm_head, embed)",
            bytes_saved=q3_all_q4,
            stream_class=STREAM_WEIGHT_CODES,
            evidence_status=UNTESTED,
            source=DN_REL,
            overlap_bucket="q4_codes",
            mechanism="Same 3/4-of-codes cut on the whole q4_bytes code body, including lm_head and the unread embed table.",
            notes=(
                "lm_head and the unread embed table are NOT in the 15.27 ms matvec "
                "streaming pool (GAP_LEDGER_60 names lm_head as outside that "
                "accounting; embed is one-row lookup). Counted for payload floor, "
                "not for the 16.8% matvec cut. Dominates heterogeneous_qkvz and "
                "lower_bit_out_proj in overlap_bucket q4_codes."
            ),
            rates=rates,
            counts_toward_measured_safe=False,
            extra={"q4_code_bytes": bc["q4_code_bytes"]},
        ),
        _candidate(
            ident="q4_heterogeneous_qkvz",
            name="heterogeneous bits on DeltaNet qkvz (Q3 on a crushed block)",
            bytes_saved=q3_het,
            stream_class=STREAM_WEIGHT_CODES,
            evidence_status=UNTESTED,
            source=DN_REL,
            overlap_bucket="q4_codes",
            mechanism=str(dn_by_id["heterogeneous_qkvz_bits"]["mechanism"]),
            notes="OPEN, STATIC_ONLY. Subset of uniform Q3 on qkvz. Weight-space Q3 rel-fro is similar across q/k/v/z (packing uniformity, not operator ranking).",
            rates=rates,
            counts_toward_measured_safe=False,
        ),
        _candidate(
            ident="larger_q4_group_128",
            name="Q4 group 128 on DeltaNet (scale aux halved)",
            bytes_saved=q4_g128,
            stream_class=STREAM_BROADCAST_AUX,
            evidence_status=UNTESTED,
            source=DN_REL,
            overlap_bucket="q4_aux",
            mechanism=str(dn_by_id["larger_q4_group"]["mechanism"]),
            notes="OPEN. Codes unchanged. Broadcast_aux-class Q4 scales: SIZE not TIME. CHEAP CPU falsifier in the DN receipt is unrun.",
            rates=rates,
            counts_toward_measured_safe=False,
        ),
        _candidate(
            ident="pack_headers",
            name="pack affine per-tensor headers",
            bytes_saved=pack_headers,
            stream_class=STREAM_BROADCAST_AUX,
            evidence_status=UNTESTED,
            source=AUX_REL,
            overlap_bucket="headers",
            mechanism="Header bytes are 58176 of the MLP aux; a tighter pack saves 52032.",
            notes="OPEN and immaterial on both bars (52 KB, 0 ms).",
            rates=rates,
            counts_toward_measured_safe=False,
        ),
        _candidate(
            ident="composite_mlp_simple_linear_low_rank",
            name="composite MLP simple linear low-rank",
            bytes_saved=0,
            stream_class=STREAM_WEIGHT_CODES,
            evidence_status=REFUTED,
            source=SPARSE_REL,
            overlap_bucket="mlp_program",
            mechanism="Replace the MLP maps with a bulk linear low-rank program.",
            notes=(
                f"{COMPOSITE_SCAR}: reused as bulk, not reswept. Residual rescue "
                "after it is a different school and is itself closed. Zero bytes "
                "may be counted toward any floor."
            ),
            rates=rates,
            counts_toward_measured_safe=False,
        ),
        _candidate(
            ident="mlp_uniform_below_q2",
            name="uniform bit-descent of the MLP below affine-Q2",
            bytes_saved=0,
            stream_class=STREAM_WEIGHT_CODES,
            evidence_status=REFUTED,
            source=CENSUS_REL,
            overlap_bucket="mlp_codes",
            mechanism="Replace the 2-bit LS codes with uniform Q1/binary/ternary.",
            notes=(
                "ALREADY_FALSIFIED (NNS-029 uniform Q2 MLP rel-fro 0.578 vs q3 "
                "0.198; QN-BINARY-INJURY 1.25-bpw body generation-incoherent). "
                "Not a reopen of entropy coding (different object)."
            ),
            rates=rates,
            counts_toward_measured_safe=False,
        ),
        _candidate(
            ident="q2_bitcast_dequant",
            name="bitcast dequant of the q2 MLP matvec (TIME, not SIZE)",
            bytes_saved=0,
            stream_class=STREAM_WEIGHT_CODES,
            evidence_status=MEASURED,
            source=BITCAST_REL,
            overlap_bucket="decode_arithmetic",
            mechanism="Place the 2-bit code in an f32 exponent field; same bytes, cheaper convert.",
            notes=(
                f"gpu_ms_saved={bitcast_ms}; token_identical={token_q2}. Payload "
                "unchanged. Already built, default-off. Not a compression campaign."
            ),
            rates=rates,
            counts_toward_measured_safe=False,
            extra={"gpu_ms_saved_measured": bitcast_ms, "token_identical": token_q2},
        ),
        _candidate(
            ident="q4_bitcast_dequant",
            name="bitcast unpack of the q4 matvec (TIME, not SIZE)",
            bytes_saved=0,
            stream_class=STREAM_WEIGHT_CODES,
            evidence_status=MEASURED,
            source=Q4_BITCAST_REL,
            overlap_bucket="decode_arithmetic",
            mechanism="Same q4 bytes, cheaper nibble convert.",
            notes=(
                f"gpu_ms_saved={q4bc_ms}; token_identical={token_q4}; "
                f"bit_identical={bit_id_q4}. Payload unchanged."
            ),
            rates=rates,
            counts_toward_measured_safe=False,
            extra={
                "gpu_ms_saved_measured": q4bc_ms,
                "token_identical": token_q4,
                "bit_identical": bit_id_q4,
            },
        ),
    ]
    ids = [c["id"] for c in out]
    if len(ids) != len(set(ids)):
        raise FloorRefused(f"duplicate candidate ids: {ids}")
    return out


def select_moves(
    cands: list[dict[str, Any]],
    *,
    allow: frozenset[str],
) -> list[dict[str, Any]]:
    """At most one move per overlap_bucket. REFUTED is never selected."""
    best: dict[str, dict[str, Any]] = {}
    for c in cands:
        if c["evidence_status"] not in allow:
            continue
        if c["evidence_status"] == REFUTED:
            continue
        if int(c["bytes_saved"]) <= 0:
            continue
        bucket = str(c["overlap_bucket"])
        prev = best.get(bucket)
        if prev is None or int(c["bytes_saved"]) > int(prev["bytes_saved"]):
            best[bucket] = c
    return [best[k] for k in sorted(best)]


def _payload_after(payload: int, moves: list[dict[str, Any]]) -> int:
    saved = sum(int(m["bytes_saved"]) for m in moves)
    after = payload - saved
    if after <= 0:
        raise FloorRefused(f"selected moves {saved} exceed payload {payload}")
    return after


def floor_from(cands: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    mix = mix_report()
    payload = int(mix["payload_bytes"])
    n_params = int(mix["parent_params"])
    rows = cands if cands is not None else candidates()
    measured_moves = select_moves(rows, allow=frozenset({MEASURED}))
    untested_moves = select_moves(rows, allow=frozenset({MEASURED, UNTESTED}))
    # A REFUTED row must never appear, even if a caller stuffed bytes_saved.
    if any(m["evidence_status"] == REFUTED for m in measured_moves + untested_moves):
        raise FloorRefused("REFUTED candidate leaked into a floor")
    safe_bytes = _payload_after(payload, measured_moves)
    hope_bytes = _payload_after(payload, untested_moves)
    return {
        "incumbent_bytes": payload,
        "incumbent_gb": _gb(payload),
        "incumbent_bpw": _bpw(payload, n_params),
        "measured_safe_bytes": safe_bytes,
        "measured_safe_gb": _gb(safe_bytes),
        "measured_safe_bpw": _bpw(safe_bytes, n_params),
        "measured_safe_moves": [m["id"] for m in measured_moves],
        "measured_safe_bytes_removed": payload - safe_bytes,
        "measured_safe_ms_saved_billed": _r(sum(float(m["ms_saved"]) for m in measured_moves), 6),
        "if_every_untested_move_worked_bytes": hope_bytes,
        "if_every_untested_move_worked_gb": _gb(hope_bytes),
        "if_every_untested_move_worked_bpw": _bpw(hope_bytes, n_params),
        "if_every_untested_move_worked_moves": [m["id"] for m in untested_moves],
        "if_every_untested_move_worked_bytes_removed": payload - hope_bytes,
        "if_every_untested_move_worked_ms_saved_billed": _r(
            sum(float(m["ms_saved"]) for m in untested_moves), 6
        ),
        "refuted_ids_excluded": [c["id"] for c in rows if c["evidence_status"] == REFUTED],
        "how_overlaps_are_resolved": (
            "one move per overlap_bucket, the largest bytes_saved among statuses "
            "the floor allows. REFUTED is never selected. mlp_aux holds 2.25 / "
            "G=128 / G=256 / G=1024 / u8; q4_codes holds uniform-Q3 variants."
        ),
    }


def sixty_tps(cands: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    gap = _load(GAP_REL, why="the 16.8% matvec-byte requirement is the 60-TPS authority")
    need = _need(gap, "bytes_required_after_arithmetic", source=GAP_REL)
    frac = float(need["fraction_of_matvec_bytes_to_remove"])
    streaming_ms = float(need["streaming_ms_at_arm_a"])
    further_ms = float(need["further_ms_needed_for_60"])
    tps_after = float(_need(gap, "arithmetic_ceiling", "tps_after", source=GAP_REL))
    still_short = float(_need(gap, "arithmetic_ceiling", "still_short_of_60_by_ms", source=GAP_REL))
    bc = byte_classes()
    rows = cands if cands is not None else candidates()

    # The 15.27 ms is q2+q4 matvecs after arithmetic removal. lm_head and the
    # unread embed table sit outside that accounting (GAP_LEDGER_60).
    matvec_storage = bc["mlp_storage_bytes"] + bc["attention_storage_bytes"]
    matvec_codes = (
        bc["mlp_code_bytes"] + bc["attention_q4_code_bytes"]
        + int(next(c["bytes"] for c in bc["classes"] if c["id"] == "f32_conv1d"))
    )
    required_from_storage = int(round(frac * matvec_storage))
    required_from_codes = int(round(frac * matvec_codes))

    measured_code_moves = [
        m for m in select_moves(rows, allow=frozenset({MEASURED}))
        if m["stream_class"] == STREAM_WEIGHT_CODES
    ]
    untested_code_moves = [
        m for m in select_moves(rows, allow=frozenset({MEASURED, UNTESTED}))
        if m["stream_class"] == STREAM_WEIGHT_CODES
    ]
    # Payload-floor uniform-Q3-all includes lm_head/embed. For the 16.8% cut,
    # only attention+DeltaNet+MLP codes count. Rebuild that by hand from known ids.
    entropy = next(c for c in rows if c["id"] == "entropy_code_mlp_codes")
    q3_attn = next(c for c in rows if c["id"] == "q4_uniform_q3_attention_deltanet")
    measured_conv_bytes = int(entropy["bytes_saved"])
    untested_conv_bytes = int(entropy["bytes_saved"]) + int(q3_attn["bytes_saved"])

    shortfall_measured = required_from_codes - measured_conv_bytes
    shortfall_untested = required_from_codes - untested_conv_bytes
    can_measured = shortfall_measured <= 0
    can_untested = shortfall_untested <= 0
    can = bool(can_measured)

    return {
        "can_conventional_compression_reach_60_tps": can,
        "shortfall_gb": _gb(max(shortfall_untested, 0) if not can_untested else max(shortfall_measured, 0)),
        "shortfall_bytes_measured_conventional": max(shortfall_measured, 0),
        "shortfall_bytes_if_every_untested_conventional_worked": max(shortfall_untested, 0),
        "arithmetic": {
            "perfect_arithmetic_removal_tps": tps_after,
            "still_short_of_60_by_ms": still_short,
            "streaming_ms_at_arm_a": streaming_ms,
            "further_ms_needed_for_60": further_ms,
            "fraction_of_matvec_bytes_to_remove": frac,
            "matvec_storage_bytes_mlp_plus_attention": matvec_storage,
            "matvec_code_bytes_mlp_plus_attention": matvec_codes,
            "bytes_required_at_fraction_of_storage": required_from_storage,
            "bytes_required_at_fraction_of_codes": required_from_codes,
            "measured_conventional_code_bytes": measured_conv_bytes,
            "untested_conventional_code_bytes": untested_conv_bytes,
            "measured_fraction_of_matvec_codes": _r(measured_conv_bytes / matvec_codes, 4),
            "untested_fraction_of_matvec_codes": _r(untested_conv_bytes / matvec_codes, 4),
            "why_aux_does_not_count_here": (
                "broadcast_aux bills at 0.000 ms/GB. After arithmetic removal the "
                f"{streaming_ms} ms is the binding stream. Shrinking scale/bias "
                "saves payload SIZE, not the 15.27 ms."
            ),
            "formula": (
                f"need {frac:.4f} × matvec_code_bytes {matvec_codes} = "
                f"{required_from_codes} bytes. MEASURED conventional "
                f"(Shannon on MLP codes) supplies {measured_conv_bytes} "
                f"({measured_conv_bytes / matvec_codes:.4%} of the codes). "
                f"UNTESTED conventional adds uniform Q3 on attention/DeltaNet "
                f"codes {int(q3_attn['bytes_saved'])} for a total "
                f"{untested_conv_bytes} ({untested_conv_bytes / matvec_codes:.4%}). "
                f"Both are short of {frac:.4%}."
            ),
            "source": GAP_REL,
        },
        "what_class_of_change_could": (
            "INFORMATION ELIMINATION — a smaller executable program for the same "
            "function — or removing work that is not a matvec (DeltaNet state "
            "update / rearrange / gated norm, ~2.0 ms; host time). Ordinary "
            "coding of the stored q has single-digit percent of the MLP codes "
            f"({entropy['gb_saved']} GB, {indep_pct(entropy)} of that body) and "
            "cannot supply 16.8% of matvec weight bytes."
        ),
        "measured_code_move_ids": [m["id"] for m in measured_code_moves],
        "untested_code_move_ids": [m["id"] for m in untested_code_moves],
    }


def indep_pct(entropy: dict[str, Any]) -> str:
    frac = float(entropy.get("independent_fraction") or 0.0)
    # redundant fraction is 1 - independent
    return f"{(1.0 - frac) * 100:.2f}%" if frac else f"{entropy.get('gb_saved')} GB"


def worth_it(cands: list[dict[str, Any]] | None = None,
             floor: dict[str, Any] | None = None,
             sixty: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = cands if cands is not None else candidates()
    fl = floor if floor is not None else floor_from(rows)
    sx = sixty if sixty is not None else sixty_tps(rows)
    entropy = next(c for c in rows if c["id"] == "entropy_code_mlp_codes")
    bc = byte_classes()
    entropy_frac_of_codes = int(entropy["bytes_saved"]) / bc["mlp_code_bytes"]
    expected = float(fl["measured_safe_ms_saved_billed"])
    untested_ms = float(fl["if_every_untested_move_worked_ms_saved_billed"])
    size_material = entropy_frac_of_codes >= SIZE_MATERIAL_FRAC
    time_material = expected >= MATERIAL_MS or untested_ms >= MATERIAL_MS
    verdict = "NO"
    if time_material or sx["can_conventional_compression_reach_60_tps"]:
        verdict = "YES"
    return {
        "verdict": verdict,
        "expected_ms_per_token": expected,
        "expected_ms_per_token_if_every_untested_worked": untested_ms,
        "materiality_threshold_ms": MATERIAL_MS,
        "size_materiality_frac": SIZE_MATERIAL_FRAC,
        "entropy_frac_of_mlp_codes": _r(entropy_frac_of_codes, 4),
        "entropy_meets_size_bar": size_material,
        "time_meets_ms_bar": time_material,
        "cost_basis": (
            f"S025: spend an hour only if a perfect win is >= {MATERIAL_MS} "
            f"ms/token, or >= 5-10% of important bytes. Shannon-coding the MLP "
            f"codes is {entropy_frac_of_codes:.2%} of that {_gb(bc['mlp_code_bytes'])} GB body "
            f"({size_material} on the size bar) and {expected} ms/token billed "
            f"at weight_codes {stream_rates()[STREAM_WEIGHT_CODES]['ms_per_gb']} "
            f"ms/GB ({time_material} on the time bar). Every UNTESTED "
            f"non-overlapping payload move still bills {untested_ms} ms/token "
            "because the large aux cuts are 0.000 ms/GB and the Q3 code cut is "
            "a fraction of a millisecond. Neither reaches 60 TPS."
        ),
        "why": (
            "Another conventional compression campaign is not worth running as "
            "a 60-TPS lever. The only MEASURED byte cut on the binding stream "
            f"is {entropy['gb_saved']} GB of MLP-code histogram bias, billed "
            f"{expected} ms/token if a fused decoder existed and 0.000 if it "
            "rematerializes. Aux cuts (2.25 / G=128 / u8) are SIZE, measured "
            f"0.000 ms/GB, and the native u8 path was slower. G=256 and G=1024 "
            "are capability-REFUTED. Uniform Q3 of attention/DeltaNet is "
            "UNTESTED and still leaves the 16.8% matvec-byte cut short. The "
            "bitcast levers are TIME not SIZE and are already measured."
        ),
    }


def load_context() -> dict[str, Any]:
    load = _load(LOAD_REL, why="cold USB rate is the library-volume cost")
    op = _load(OPCLASS_REL, why="the q2 arithmetic split is cited context")
    gap = _load(GAP_REL, why="perfect arithmetic removal is cited context")
    rates = _need(load, "rates", source=LOAD_REL)
    decomp = _need(op, "decomposition", "classes", source=OPCLASS_REL)
    return {
        "quiet_cold_MB_per_s": rates["quiet_cold_MB_per_s"],
        "volume": rates["volume"],
        "q2_arithmetic_split": {
            k: v["share_of_arithmetic"] for k, v in decomp.items()
        },
        "perfect_arithmetic_removal_tps": _need(
            gap, "arithmetic_ceiling", "tps_after", source=GAP_REL
        ),
        "still_short_of_60_by_ms": _need(
            gap, "arithmetic_ceiling", "still_short_of_60_by_ms", source=GAP_REL
        ),
        "source_load": LOAD_REL,
        "source_opclass": OPCLASS_REL,
        "source_gap": GAP_REL,
    }


def load_bearing(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Every number a reader might treat as a fact, with its disk source."""
    bc = doc["byte_classes"]
    fl = doc["floor"]
    sx = doc["sixty"]
    return [
        {"id": "payload_bytes", "value": bc["payload_bytes"],
         "source": str(MIX_REPORT), "role": "incumbent storage payload"},
        {"id": "parent_params", "value": bc["parent_params"],
         "source": str(MIX_REPORT), "role": "bpw denominator"},
        {"id": "weight_codes_ms_per_gb",
         "value": bc["by_stream_class"][STREAM_WEIGHT_CODES]["ms_per_gb"],
         "source": ECON_REL, "role": "binding-stream time cost"},
        {"id": "broadcast_aux_ms_per_gb",
         "value": bc["by_stream_class"][STREAM_BROADCAST_AUX]["ms_per_gb"],
         "source": ECON_REL, "role": "aux time cost; 0 means SIZE not TIME"},
        {"id": "activation_ms_per_gb",
         "value": bc["by_stream_class"][STREAM_ACTIVATION]["ms_per_gb"],
         "source": ECON_REL, "role": "catalog-scale activation billing"},
        {"id": "mlp_code_entropy_bits",
         "value": next(c["H_q_bits"] for c in doc["candidates"]
                       if c["id"] == "entropy_code_mlp_codes"),
         "source": CODE_REL, "role": "lossless code-body floor"},
        {"id": "fraction_of_matvec_bytes_to_remove",
         "value": sx["arithmetic"]["fraction_of_matvec_bytes_to_remove"],
         "source": GAP_REL, "role": "16.8% cut 60 TPS still needs after arm_a"},
        {"id": "streaming_ms_at_arm_a",
         "value": sx["arithmetic"]["streaming_ms_at_arm_a"],
         "source": GAP_REL, "role": "pure-streaming matvec time after arithmetic removal"},
        {"id": "measured_safe_gb", "value": fl["measured_safe_gb"],
         "source": "derived from MEASURED candidates only",
         "role": "smallest payload with MEASURED-safe moves"},
        {"id": "if_every_untested_move_worked_gb",
         "value": fl["if_every_untested_move_worked_gb"],
         "source": "derived from MEASURED+UNTESTED, REFUTED excluded",
         "role": "optimistic conventional payload"},
    ]


def build() -> dict[str, Any]:
    for rel in REQUIRED_RELS:
        _load(rel, why="every cited receipt must exist before a floor is computed")
    if not MIX_REPORT.is_file():
        raise FloorRefused(
            f"{MIX_REPORT} is not on disk; the incumbent payload is missing"
        )
    rows = candidates()
    fl = floor_from(rows)
    sx = sixty_tps(rows)
    wt = worth_it(rows, fl, sx)
    bc = byte_classes()
    ctx = load_context()
    doc = {
        "schema": SCHEMA,
        "obligation": "representation floor for the Qwen3.8 dense incumbent",
        "authority": "S025 materiality; S026 60-TPS gap; cited receipts below",
        "question": (
            "HOW MUCH SMALLER CAN THE INCUMBENT GET, and is another "
            "compression effort worth running?"
        ),
        "byte_classes": bc,
        "candidates": rows,
        "floor": {
            "measured_safe_gb": fl["measured_safe_gb"],
            "measured_safe_bpw": fl["measured_safe_bpw"],
            "if_every_untested_move_worked_gb": fl["if_every_untested_move_worked_gb"],
            "if_every_untested_move_worked_bpw": fl["if_every_untested_move_worked_bpw"],
            **{k: v for k, v in fl.items()
               if k not in {
                   "measured_safe_gb", "measured_safe_bpw",
                   "if_every_untested_move_worked_gb",
                   "if_every_untested_move_worked_bpw",
               }},
        },
        "can_conventional_compression_reach_60_tps": sx[
            "can_conventional_compression_reach_60_tps"
        ],
        "shortfall_gb": sx["shortfall_gb"],
        "sixty": sx,
        "worth_it": wt,
        "context": ctx,
        "what_is_measured": (
            "MIX_REPORT payload split; catalog census organ bytes; stream-class "
            "ms/GB; MLP i.i.d. Shannon of the 2-bit codes; group_size_256/1024 "
            "FAILED_HELDOUT; composite linear low-rank scar; q2/q4 bitcast "
            "gpu_ms_saved; arm_a 52.008 TPS still 2.56 ms short of 60; USB "
            "quiet-cold 77.7 MB/s; aux-u8 native projected_mlp_delta_ms > 0."
        ),
        "what_is_estimated": (
            "ms_saved for every payload cut is billed at the stream-class rate "
            "(not a complete-token re-measure of a new packing). HQ30UQ4 split "
            "of GQA/lm_head/embed applies the DeltaNet-measured 4-bit / group-64 "
            "/ 40-byte-header layout to census storage_bytes; MIX_REPORT names "
            "that codec on the hardlinked q4 tensors. Uniform Q3 byte counts "
            "are 1/4 of those codes, not a fitted Q3 artifact. Combining "
            "arithmetic removal with byte removal is the composition caveat "
            "already on GAP_LEDGER_60."
        ),
        "claim_boundary": (
            "Static sidecar artifact. No GPU work. No new fit. Hardware numbers "
            "are copied from sealed receipts and billed; they are not a protected "
            "reprofile of a smaller packing. evidence_status MEASURED means the "
            "cited quantity was measured, not that a fused kernel of that packing "
            "is in the resident."
        ),
    }
    doc["load_bearing"] = load_bearing(doc)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_measured_receipt(
            REPO / "receipts" / "future" / RECEIPT_NAME, doc, RECORDED_BY,
            provenance=measurement_provenance(
                lock_held=False, lane="derived", retrofit=True,
            ),
        ))
        return 0
    print(json.dumps({
        "floor": doc["floor"],
        "can_conventional_compression_reach_60_tps": doc[
            "can_conventional_compression_reach_60_tps"
        ],
        "shortfall_gb": doc["shortfall_gb"],
        "worth_it": doc["worth_it"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
