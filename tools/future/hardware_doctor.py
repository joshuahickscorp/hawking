"""HARDWARE DOCTOR — falsifiable hardware-axis experiment proposer.

The representation Doctor (tools/headless/doctor_diagnosis.py) ranks S026 §65
families per organ and emits ORGAN / DIAGNOSIS / PRESCRIPTION / AVOID. This
sidecar does the same job on the HARDWARE axis: arithmetic width, bit-serial
vs bit-parallel, DSP/LUT, tiling, HBM, banking, pipeline, composition,
persistent state, DFX, transport, overlap. FPGA here is Accelerator /
Physical Compiler / Fusion, not its own civilization.

Every proposal carries predicted_effect, uncertainty, cheapest_simulator and
falsifier. emit() RAISES if any required field is missing — it does not warn.
A proposal that restates a recorded scar is refused. Nothing here is a
measurement: STATIC_ONLY, bench UNKNOWN, neither DIAGNOSTIC_RELATIVE nor
PROTECTED_ABSOLUTE.

    python3 tools/future/hardware_doctor.py --build
    python3 tools/future/hardware_doctor.py --selftest
    python3 -m pytest tools/future/test_hardware_doctor.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, git

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

RECEIPT = "HARDWARE_DOCTOR.json"
SCHEMA = "hawking.future.hardware_doctor.v1"
RECORDED_BY = "tools/future/hardware_doctor.py"

# Five eras, three odysseys. There is no Era VI and no Odyssey IV.
ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)

AXES = (
    "arithmetic_width",
    "bit_serial_vs_bit_parallel",
    "dsp_lut_balance",
    "tiling",
    "hbm_mapping",
    "banking",
    "pipeline_depth",
    "module_composition",
    "persistent_state",
    "dfx_boundary",
    "transport_format",
    "compute_transfer_overlap",
)

# emit() refuses a record missing any of these. The four named in the contract
# (predicted_effect, uncertainty, cheapest_simulator, falsifier) are included.
REQUIRED_FIELDS = (
    "axis",
    "hypothesis",
    "target_organ",
    "predicted_effect",
    "uncertainty",
    "cheapest_simulator",
    "falsifier",
    "expected_removed_cost",
    "prerequisite",
)

MAGNITUDE_CLASSES = frozenset(
    {"UNKNOWN", "SUB_PERCENT", "SINGLE_DIGIT_FRACTION", "FACTOR"}
)
REFUTATION_CLASSES = frozenset({"HIGH", "MEDIUM", "LOW"})
REFUTATION_WEIGHT = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

# Cheapest simulator that could test a hypothesis. Availability is recovered
# from the organ maps; unavailability does not invent a physical measurement.
SIMULATORS: dict[str, dict[str, Any]] = {
    "static_hwir": {
        "cost": 1,
        "available": True,
        "note": "inspect organ-map HWIR graph, buffer lifetimes, placements, DFX cuts",
    },
    "transport_link_simulator": {
        "cost": 2,
        "available": True,
        "note": "hcli.fpga.link.v1 already on both organ maps; [S] sensitivity only",
    },
    "partition_simulation": {
        "cost": 3,
        "available": True,
        "note": "organ-map partition_simulation; serial mixed_ns scenario model, [S]",
    },
    "hbm_bank_model": {
        "cost": 3,
        "available": False,
        "note": "not built; cheapest that could test HBM banking. Current link simulator is hop/bandwidth only",
    },
    "rtl_resource_estimate": {
        "cost": 4,
        "available": False,
        "note": "rtl_hls_verifier_surface is CONTRACT_ONLY; resource bounds listed, not executed",
    },
    "cycle_accurate": {
        "cost": 5,
        "available": False,
        "note": "provider_capabilities.cycle_simulation is false",
    },
    "physical_board": {
        "cost": 6,
        "available": False,
        "note": "experiment_dag hardware_receipt is BLOCKED_NO_BOARD",
    },
}

ORGAN_MAPS = {
    "flash-next": "receipts/headless/FLASH_NEXT_FPGA_ORGAN_MAP.json",
    "qwen27": "receipts/headless/QWEN27_FPGA_ORGAN_MAP.json",
}

# Prefer a sibling-lane index if it has landed; never import that lane's module.
NEGATIVE_INDEX = "receipts/future/NEGATIVE_SCIENCE_INDEX.json"
FALLBACK_SCAR_SOURCES = (
    "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
    "receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
    "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
    "workspace/campaign/odyssey/NEGATIVE_SCIENCE.json",
)

DEAD_CLASSES = frozenset({"REFUTED", "UNREACHABLE", "DEAUTHORISED"})
_STOP = frozenset(
    "a an the of to for on in at as is be by or and vs with from this that "
    "not its it as a win wall time path token tokens".split()
)
_WS = re.compile(r"[^a-z0-9]+")


class HardwareDoctorError(ValueError):
    """Base error for the hardware-axis Doctor."""


class MissingFieldError(HardwareDoctorError):
    """emit() refused a proposal that omitted a required field."""


class ScarRefusal(HardwareDoctorError):
    """emit() refused a proposal that restates a recorded scar."""


class UnknownOrganError(HardwareDoctorError):
    """emit() refused a target_organ that is not on a recovered organ map."""


# ---------------------------------------------------------------------------
# repo IO (sparse checkout: missing on disk is not absence)
# ---------------------------------------------------------------------------


def repo_text(rel: str) -> str | None:
    """Read a repo-relative file from disk, else `git show HEAD:<rel>`."""
    path = REPO / rel
    if path.is_file():
        return path.read_text()
    blob = git("show", f"HEAD:{rel}")
    return blob or None


def repo_exists(rel: str) -> bool:
    return repo_text(rel) is not None


def repo_json(rel: str) -> dict[str, Any]:
    blob = repo_text(rel)
    if blob is None:
        raise FileNotFoundError(rel)
    return json.loads(blob)


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (dict, list, tuple)) and not value:
        return False
    return True


def _norm(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def _fidelity(simulator: Any) -> str:
    if isinstance(simulator, dict):
        fid = simulator.get("fidelity") or simulator.get("name") or ""
    else:
        fid = str(simulator or "")
    fid = fid.strip()
    if fid not in SIMULATORS:
        raise HardwareDoctorError(
            f"cheapest_simulator fidelity {fid!r} is not on the simulator ladder"
        )
    return fid


def _magnitude(effect: Any) -> dict[str, str]:
    if not isinstance(effect, dict):
        raise MissingFieldError(
            "predicted_effect must be {direction, magnitude_class}, not a scalar"
        )
    direction = effect.get("direction")
    magnitude = effect.get("magnitude_class")
    if not _present(direction) or not _present(magnitude):
        raise MissingFieldError(
            "predicted_effect requires both direction and magnitude_class"
        )
    if magnitude not in MAGNITUDE_CLASSES:
        raise HardwareDoctorError(
            f"predicted_effect.magnitude_class {magnitude!r} not in {sorted(MAGNITUDE_CLASSES)}"
        )
    for banned in ("magnitude", "value", "delta", "amount"):
        if isinstance(effect.get(banned), (int, float)):
            raise HardwareDoctorError(
                f"predicted_effect.{banned} is a fabricated number; use magnitude_class"
            )
    return {"direction": str(direction), "magnitude_class": str(magnitude)}


# ---------------------------------------------------------------------------
# organs — recovered from the real FPGA organ maps, never invented
# ---------------------------------------------------------------------------


def load_organs() -> dict[str, Any]:
    maps: dict[str, Any] = {}
    organs: list[dict[str, Any]] = []
    for model, rel in ORGAN_MAPS.items():
        doc = repo_json(rel)
        rows = []
        for item in doc.get("organs") or []:
            name = item.get("organ")
            if not name:
                continue
            row = {
                "model": model,
                "organ": name,
                "mapping": item.get("mapping"),
                "priority": item.get("priority"),
            }
            rows.append(row)
            organs.append(row)
        caps = doc.get("provider_capabilities") or {}
        device = doc.get("device_genome") or {}
        hbm = doc.get("hbm_genome") or {}
        dag = ((doc.get("experiment_dag") or {}).get("nodes")) or []
        buffers = (doc.get("hwir") or {}).get("buffers") or []
        maps[model] = {
            "path": rel,
            "schema": doc.get("schema"),
            "claim_boundary": doc.get("claim_boundary"),
            "organs": rows,
            "experiment_dag": [
                {"id": n.get("id"), "status": n.get("status")} for n in dag
            ],
            "provider_capabilities": {
                "cycle_simulation": bool(caps.get("cycle_simulation")),
                "hwir": bool(caps.get("hwir")),
                "link_simulation": bool(caps.get("link_simulation")),
                "physical_execution": bool(caps.get("physical_execution")),
            },
            "device_status": device.get("status"),
            "device_id": device.get("device_id"),
            "physical_board_present": bool(device.get("physical_board_present")),
            "hbm_status": hbm.get("status"),
            "hbm_channels_selected": bool(hbm.get("channels")),
            "buffers": [
                {
                    "id": b.get("id"),
                    "lifetime": b.get("lifetime"),
                    "per_token_transfer": b.get("per_token_transfer"),
                }
                for b in buffers
            ],
            "module_cache_status": (doc.get("module_cache") or {}).get("status"),
            "rtl_hls_verifier_status": (doc.get("rtl_hls_verifier_surface") or {}).get(
                "status"
            ),
            "transport_link_status": (doc.get("transport_link_simulator") or {}).get(
                "status"
            ),
        }
    if not organs:
        raise HardwareDoctorError("organ maps loaded but listed no organs")
    organs = sorted(organs, key=lambda r: (r["model"], r["organ"]))
    return {"maps": maps, "organs": organs}


def known_organs(bundle: dict[str, Any] | None = None) -> set[str]:
    bundle = bundle or load_organs()
    return {r["organ"] for r in bundle["organs"]}


# ---------------------------------------------------------------------------
# scar corpus — query before emit; do not import a sibling lane
# ---------------------------------------------------------------------------


def _scar_record(
    *,
    sid: str,
    source: str,
    mechanism: str,
    klass: str | None = None,
    verdict: str | None = None,
    kind: str | None = None,
    extra_needles: list[str] | None = None,
) -> dict[str, Any] | None:
    mechanism = (mechanism or "").strip()
    if not mechanism:
        return None
    verd = verdict or ""
    dead = False
    if (klass or "") in DEAD_CLASSES:
        dead = True
    if kind == "PROPERTY_OF_IDEA":
        dead = True
    if re.search(r"\bDEAD\b", verd, re.I) and not re.match(r"^\s*LIVE\b", verd, re.I):
        if "NOT closed" not in verd:
            dead = True
    needles = [mechanism]
    for extra in extra_needles or []:
        if extra:
            needles.append(extra)
    return {
        "id": sid,
        "source": source,
        "mechanism": mechanism,
        "class": klass,
        "verdict": verd or None,
        "kind": kind,
        "dead": dead,
        "needles": needles,
    }


def _from_index(doc: Any, source: str) -> list[dict[str, Any]]:
    if isinstance(doc, list):
        rows = doc
    elif isinstance(doc, dict):
        rows = (
            doc.get("entries")
            or doc.get("scars")
            or doc.get("index")
            or doc.get("negative_science")
            or []
        )
        if isinstance(rows, dict):
            rows = [{"id": k, **(v if isinstance(v, dict) else {"mechanism": str(v)})} for k, v in rows.items()]
    else:
        rows = []
    out: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        rec = _scar_record(
            sid=str(item.get("id") or item.get("scar_id") or item.get("name") or ""),
            source=source,
            mechanism=str(
                item.get("mechanism")
                or item.get("lever")
                or item.get("hypothesis")
                or item.get("claim_refuted")
                or item.get("claim")
                or ""
            ),
            klass=item.get("class") or item.get("status"),
            verdict=item.get("verdict"),
            kind=item.get("kind"),
            extra_needles=[
                str(x)
                for x in (
                    item.get("lever"),
                    item.get("seed"),
                    item.get("claim_refuted"),
                )
                if x
            ],
        )
        if rec and rec["id"]:
            out.append(rec)
    return out


def _fallback_scars() -> tuple[list[dict[str, Any]], list[str], list[str]]:
    scars: list[dict[str, Any]] = []
    consulted: list[str] = []
    missing: list[str] = []
    for rel in FALLBACK_SCAR_SOURCES:
        blob = repo_text(rel)
        if blob is None:
            missing.append(rel)
            continue
        consulted.append(rel)
        doc = json.loads(blob)
        if rel.endswith("NEGATIVE_TRANSFER_ATLAS.json"):
            entries = doc.get("entries") or {}
            if isinstance(entries, dict):
                for key, val in sorted(entries.items()):
                    if not isinstance(val, dict):
                        continue
                    rec = _scar_record(
                        sid=key,
                        source=rel,
                        mechanism=str(val.get("lever") or key),
                        verdict=str(val.get("verdict") or ""),
                        extra_needles=[str(val.get("killed_by") or "")],
                    )
                    if rec:
                        scars.append(rec)
            continue
        scars.extend(_from_index(doc, rel))
    return scars, consulted, missing


def load_scars() -> dict[str, Any]:
    """Queryable scar corpus. Sibling index wins; else atlas files via git."""
    index_path = REPO / NEGATIVE_INDEX
    if index_path.is_file():
        doc = load_json(index_path)
        scars = _from_index(doc, NEGATIVE_INDEX)
        return {
            "source_used": NEGATIVE_INDEX,
            "consulted": [NEGATIVE_INDEX],
            "missing": [],
            "scars": scars,
        }
    scars, consulted, missing = _fallback_scars()
    return {
        "source_used": "fallback_atlas_files",
        "consulted": consulted,
        "missing": missing,
        "scars": scars,
    }


def _needle_hits(hypothesis: str, needle: str) -> bool:
    h = _norm(hypothesis)
    n = _norm(needle)
    if len(n) >= 16 and n in h:
        return True
    toks = [t for t in n.split() if t not in _STOP and len(t) > 2]
    if len(toks) >= 6:
        bag = set(h.split())
        return all(t in bag or t in h for t in toks)
    return False


def scar_hits(hypothesis: str, scars: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return dead scars whose mechanism is restated by `hypothesis`."""
    if scars is None:
        scars = load_scars()["scars"]
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for scar in scars:
        if not scar.get("dead"):
            continue
        sid = scar.get("id") or ""
        for needle in scar.get("needles") or []:
            if needle and _needle_hits(hypothesis, needle):
                if sid not in seen:
                    hits.append(scar)
                    seen.add(sid)
                break
    return hits


def avoid_list(scars: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Doctor-shaped AVOID: hardware-axis scars with a cited source. Does not re-run them."""
    if scars is None:
        scars = load_scars()["scars"]
    items = []
    for scar in scars:
        if not scar.get("dead"):
            continue
        mech = scar["mechanism"].lower()
        hardwareish = any(
            k in mech
            for k in (
                "hbm",
                "dram",
                "lut",
                "dsp",
                "bitstream",
                "residency",
                "prefetch",
                "command-buffer",
                "command buffer",
                "serial",
                "pipeline",
                "bank",
                "cache decoded",
                "expert cache",
                "interleaving",
                "switching-activity",
                "gray",
                "gather vs sequential",
            )
        )
        if not hardwareish:
            continue
        items.append(
            {
                "family": "hardware_axis",
                "experiment": scar["mechanism"],
                "reason": (
                    f"{scar['id']} is recorded {scar.get('class') or scar.get('kind') or 'DEAD'} "
                    f"in {scar['source']}. Diagnosis does not re-run the dead experiment."
                ),
                "negative_science": [scar["source"]],
                "nns_ids": [scar["id"]],
                "predicts_not_certifies": True,
            }
        )
    items.sort(key=lambda r: (r["nns_ids"][0], r["experiment"]))
    return items


# ---------------------------------------------------------------------------
# emit / rank
# ---------------------------------------------------------------------------


def emit(
    proposal: dict[str, Any],
    *,
    organs: dict[str, Any] | None = None,
    scars: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Seal a proposal or RAISE. Missing required fields are errors, not warnings."""
    missing = [f for f in REQUIRED_FIELDS if not _present(proposal.get(f))]
    if missing:
        raise MissingFieldError(
            f"emit() refused proposal {proposal.get('id')!r}: missing required field(s) {missing}"
        )

    axis = proposal["axis"]
    if axis not in AXES:
        raise HardwareDoctorError(f"unknown proposal axis {axis!r}; axes are {list(AXES)}")

    effect = _magnitude(proposal["predicted_effect"])
    fidelity = _fidelity(proposal["cheapest_simulator"])

    organ_bundle = organs if organs is not None else load_organs()
    known = known_organs(organ_bundle)
    target = proposal["target_organ"]
    if target not in known:
        raise UnknownOrganError(
            f"target_organ {target!r} is not on the recovered FPGA organ maps {sorted(known)}"
        )

    scar_bundle = scars if scars is not None else load_scars()["scars"]
    hits = scar_hits(proposal["hypothesis"], scar_bundle)
    if hits:
        names = ", ".join(h["id"] for h in hits)
        raise ScarRefusal(
            f"emit() refused proposal {proposal.get('id')!r}: restates recorded scar(s) {names}"
        )

    ref = proposal.get("refutation_probability") or "MEDIUM"
    if ref not in REFUTATION_CLASSES:
        raise HardwareDoctorError(f"refutation_probability {ref!r} not in {sorted(REFUTATION_CLASSES)}")

    sim = SIMULATORS[fidelity]
    if isinstance(proposal["cheapest_simulator"], dict):
        cheapest = {
            "fidelity": fidelity,
            **{k: v for k, v in proposal["cheapest_simulator"].items() if k != "fidelity"},
            "cost": sim["cost"],
            "available": sim["available"],
            "ladder_note": sim["note"],
        }
    else:
        cheapest = {
            "fidelity": fidelity,
            "cost": sim["cost"],
            "available": sim["available"],
            "ladder_note": sim["note"],
        }

    out = {
        "id": proposal.get("id"),
        "axis": axis,
        "hypothesis": proposal["hypothesis"],
        "target_organ": target,
        "target_model": proposal.get("target_model"),
        "predicted_effect": effect,
        "uncertainty": proposal["uncertainty"],
        "cheapest_simulator": cheapest,
        "falsifier": proposal["falsifier"],
        # Atlas vocabulary: cheapest_falsifier is the observation that kills it
        # at the cheapest simulator that could produce that observation.
        "cheapest_falsifier": proposal["falsifier"],
        "expected_removed_cost": proposal["expected_removed_cost"],
        "prerequisite": proposal["prerequisite"],
        "refutation_probability": ref,
        "refutation_weight": REFUTATION_WEIGHT[ref],
        "simulator_cost": sim["cost"],
        "predicts_not_certifies": True,
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "related_scars_not_the_same": list(proposal.get("related_scars_not_the_same") or []),
        "organ_mapping": next(
            (
                r["mapping"]
                for r in organ_bundle["organs"]
                if r["organ"] == target
                and (not proposal.get("target_model") or r["model"] == proposal.get("target_model"))
            ),
            next((r["mapping"] for r in organ_bundle["organs"] if r["organ"] == target), None),
        ),
    }
    return out


def rank_queue(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank by expected information per unit cost.

    Information proxy = refutation_weight (HIGH=3, MEDIUM=2, LOW=1).
    Cost proxy = simulator ladder cost (static_hwir=1 … physical_board=6).
    Prefer the proposal whose cheapest falsifier is cheapest and whose
    refutation probability is highest. Integer key: -(ref * 60 // cost)
    is monotone with ref/cost on this range; no float is stored. Ties
    break on simulator_cost then id.
    """
    decorated = []
    for rec in records:
        cost = int(rec.get("simulator_cost") or SIMULATORS[_fidelity(rec["cheapest_simulator"])]["cost"])
        ref = int(
            rec.get("refutation_weight")
            or REFUTATION_WEIGHT[rec.get("refutation_probability") or "MEDIUM"]
        )
        decorated.append((rec, cost, ref))

    def _key(item: tuple[Any, ...]) -> tuple[Any, ...]:
        rec, cost, ref = item
        return (-(ref * 60 // cost), cost, rec.get("id") or "", rec.get("axis") or "")

    ordered = [item[0] for item in sorted(decorated, key=_key)]
    ranked = []
    for i, rec in enumerate(ordered, start=1):
        row = dict(rec)
        row["rank"] = i
        row["information_per_cost"] = {
            "refutation_weight": int(
                rec.get("refutation_weight")
                or REFUTATION_WEIGHT[rec.get("refutation_probability") or "MEDIUM"]
            ),
            "simulator_cost": int(
                rec.get("simulator_cost")
                or SIMULATORS[_fidelity(rec["cheapest_simulator"])]["cost"]
            ),
            "rule": "rank by refutation_weight / simulator_cost; never a hardware measurement",
        }
        ranked.append(row)
    return ranked


# ---------------------------------------------------------------------------
# catalog — one live proposal per axis, grounded in recovered organs
# ---------------------------------------------------------------------------


def catalog() -> list[dict[str, Any]]:
    """Live hardware-axis proposals. None of these restate a recorded scar."""
    return [
        {
            "id": "HD-001",
            "axis": "arithmetic_width",
            "target_model": "qwen27",
            "target_organ": "mlp_gate_up_down",
            "hypothesis": (
                "Packing the organ-map low-bit GEMV at the stored operand width, "
                "rather than unpacking every MAC to a wider accumulator, reduces "
                "LUT glue on mlp_gate_up_down. This is a datapath-width question, "
                "not a Gray/LUT permutation of packed codes."
            ),
            "predicted_effect": {
                "direction": "reduce_glue_resource",
                "magnitude_class": "UNKNOWN",
            },
            "uncertainty": (
                "device_genome is TARGET_UNSELECTED; DSP/LUT ratio is a property of "
                "a named part that has not been selected. No RTL has been compiled."
            ),
            "cheapest_simulator": "rtl_resource_estimate",
            "falsifier": (
                "A resource-bounds estimate at equal initiation-interval class shows "
                "the packed-width operator does not reduce LUT+DSP envelope versus "
                "unpack-then-wide-MAC, or the epilogue dominates the envelope."
            ),
            "expected_removed_cost": "unpack-to-wide-accumulator glue on the resident GEMV",
            "prerequisite": "a selected FPGA device_genome (currently unselected) plus an executable resource-bounds check (rtl_hls_verifier_surface is CONTRACT_ONLY)",
            "refutation_probability": "MEDIUM",
            "related_scars_not_the_same": ["NS-027"],
        },
        {
            "id": "HD-002",
            "axis": "bit_serial_vs_bit_parallel",
            "target_model": "qwen27",
            "target_organ": "mlp_gate_up_down",
            "hypothesis": (
                "A bit-parallel packed MAC on the resident low-bit GEMV, with any "
                "expand done at bind/load time into resident shards, is the legal "
                "default. Bind-time expand is a different mechanism from a per-token "
                "serial expand inside the matvec."
            ),
            "predicted_effect": {
                "direction": "reduce_initiation_idle",
                "magnitude_class": "UNKNOWN",
            },
            "uncertainty": (
                "No cycle-accurate model exists (cycle_simulation is false). Whether "
                "bind-time expand fits the module_cache key is untested because "
                "module_cache is SCHEMA_ONLY."
            ),
            "cheapest_simulator": "rtl_resource_estimate",
            "falsifier": (
                "Bind-time expand plus a bit-parallel resident MAC uses a larger "
                "LUT+DSP envelope, at the same initiation-interval class, than a "
                "purely parallel packed MAC with no expand stage."
            ),
            "expected_removed_cost": "per-token serial expand inside the GEMV",
            "prerequisite": "rtl_hls_verifier_surface resource bounds become executable; NS-031 stays as the AVOID for per-token serial expand",
            "refutation_probability": "MEDIUM",
            "related_scars_not_the_same": ["NS-031"],
        },
        {
            "id": "HD-003",
            "axis": "dsp_lut_balance",
            "target_model": "qwen27",
            "target_organ": "norm_add_epilogues",
            "hypothesis": (
                "Putting low-bit GEMV MACs on DSP and the fused epilogue (organ "
                "mapping: fused epilogue near producer) on LUT glue reduces DSP "
                "pressure on the P0 GEMV. This is arithmetic-versus-glue placement, "
                "not a switching-activity permutation of packed codes."
            ),
            "predicted_effect": {
                "direction": "reduce_dsp_pressure",
                "magnitude_class": "UNKNOWN",
            },
            "uncertainty": (
                "Part unselected, so DSP/LUT budgets are UNKNOWN. CONTRACT_ONLY "
                "verifier cannot currently emit a resource envelope."
            ),
            "cheapest_simulator": "rtl_resource_estimate",
            "falsifier": (
                "Forcing the epilogue onto LUT does not drop DSP count of the P0 "
                "GEMV, or LUT overflows the part envelope while DSP sits idle."
            ),
            "expected_removed_cost": "DSP spent on epilogue glue instead of the P0 GEMV",
            "prerequisite": "selected device_genome; related scar NS-027 is not this experiment",
            "refutation_probability": "MEDIUM",
            "related_scars_not_the_same": ["NS-027"],
        },
        {
            "id": "HD-004",
            "axis": "tiling",
            "target_model": "qwen27",
            "target_organ": "mlp_gate_up_down",
            "hypothesis": (
                "Tiles that honour the organ map partition_axis "
                "within_organ_tensor_parallel and the resident-shard policy "
                "resident_shards_no_weight_body_per_token_transfer move only "
                "activations and partial reductions per token, versus a tile that "
                "restreams the weight body."
            ),
            "predicted_effect": {
                "direction": "reduce_per_token_transport",
                "magnitude_class": "FACTOR",
            },
            "uncertainty": (
                "Link parameters on the organ map are scenario inputs labelled [S], "
                "not measurements. Device HBM capacity is UNKNOWN (hbm_genome TARGET_UNSELECTED)."
            ),
            "cheapest_simulator": "transport_link_simulator",
            "falsifier": (
                "On the organ-map transport_link_simulator, a resident-shard tile "
                "does not reduce per-token transport class versus a weight-body "
                "restream of the same organ."
            ),
            "expected_removed_cost": "per-token weight-body transfer",
            "prerequisite": "transport_link_simulator already present on QWEN27_FPGA_ORGAN_MAP",
            "refutation_probability": "HIGH",
        },
        {
            "id": "HD-005",
            "axis": "hbm_mapping",
            "target_model": "flash-next",
            "target_organ": "expert_bank",
            "hypothesis": (
                "HBM-resident selected expert subsets (the organ map's own P0 "
                "mapping for expert_bank) reduce per-token transport versus "
                "streaming the whole expert bank. This is an FPGA HBM subset "
                "map, not an 8 GiB host-side expert-residency arena on Q80."
            ),
            "predicted_effect": {
                "direction": "reduce_per_token_transport",
                "magnitude_class": "FACTOR",
            },
            "uncertainty": (
                "hbm_genome channels=0 and TARGET_UNSELECTED; no capacity is known. "
                "Selected-subset cardinality is a Flash routing fact not measured here."
            ),
            "cheapest_simulator": "transport_link_simulator",
            "falsifier": (
                "Selected-subset residency does not reduce per-token transport class "
                "versus streaming the expert bank on the organ-map [S] link simulator."
            ),
            "expected_removed_cost": "per-token transfer of unselected expert weight bodies",
            "prerequisite": "a selected HBM genome; NS-028 (8 GiB Q80 arena) is a different mechanism",
            "refutation_probability": "HIGH",
            "related_scars_not_the_same": ["NS-028", "large_expert_cache"],
        },
        {
            "id": "HD-006",
            "axis": "banking",
            "target_model": "flash-next",
            "target_organ": "expert_bank",
            "hypothesis": (
                "Once a device is selected, placing selected-expert shards across "
                "independent HBM channels reduces bank-conflict stalls versus packing "
                "every selected expert into one channel. This is a channel-bank map "
                "on a named part, not a DRAM-row interleave of a live file catalog."
            ),
            "predicted_effect": {
                "direction": "reduce_bank_conflicts",
                "magnitude_class": "UNKNOWN",
            },
            "uncertainty": (
                "device_genome.hbm_channels is 0 (TARGET_UNSELECTED). The current "
                "transport_link_simulator models hops and bandwidth, not banks."
            ),
            "cheapest_simulator": "hbm_bank_model",
            "falsifier": (
                "A bank-aware link model shows conflict stalls do not drop when "
                "selected-expert shards are spread across channels, or the selected "
                "part has a single effective channel."
            ),
            "expected_removed_cost": "HBM bank-conflict stalls on selected-expert GEMV",
            "prerequisite": "selected device_genome with hbm_channels > 0, plus a bank-aware simulator that does not yet exist",
            "refutation_probability": "LOW",
            "related_scars_not_the_same": ["NS-026"],
        },
        {
            "id": "HD-007",
            "axis": "pipeline_depth",
            "target_model": "qwen27",
            "target_organ": "deltanet_state_and_input_projection",
            "hypothesis": (
                "Deepening the state-machine/update pipeline so the resident-state "
                "read-modify-write overlaps the next token's input projection reduces "
                "idle on the persistent_state buffer (lifetime=sequence, "
                "per_token_transfer=false)."
            ),
            "predicted_effect": {
                "direction": "reduce_initiation_idle",
                "magnitude_class": "UNKNOWN",
            },
            "uncertainty": (
                "partition_simulation.deltanet_state exists but is a serial mixed_ns "
                "scenario, not a staged pipeline model. Cycle-accurate overlap is unavailable."
            ),
            "cheapest_simulator": "partition_simulation",
            "falsifier": (
                "The deltanet_state partition scenario still reports mixed as not "
                "beating apple-only after the serial mixed_ns is replaced by an "
                "overlapped (max of compute, transfer) formula, or HWIR dependencies "
                "forbid overlapping the state update with the next projection."
            ),
            "expected_removed_cost": "idle between DeltaNet state update and the next input projection",
            "prerequisite": "organ-map partition_simulation deltanet_state scenario (exists, [S])",
            "refutation_probability": "MEDIUM",
        },
        {
            "id": "HD-008",
            "axis": "module_composition",
            "target_model": "flash-next",
            "target_organ": "routed_plus_shared_expert",
            "hypothesis": (
                "Composing routed expert, shared expert and epilogue as one HWIR "
                "node (organ mapping: fused selected-expert execution and epilogue) "
                "removes a cross-module activation hop versus three separately "
                "placed modules. This is a fusion cut, not a collapse of "
                "command-buffer topology as a complete-token wall lever."
            ),
            "predicted_effect": {
                "direction": "reduce_cross_module_hops",
                "magnitude_class": "SINGLE_DIGIT_FRACTION",
            },
            "uncertainty": (
                "Activation/partial-reduction byte counts on the organ map are "
                "scenario inputs. Occupancy on a real part is UNKNOWN."
            ),
            "cheapest_simulator": "transport_link_simulator",
            "falsifier": (
                "Fused composition does not reduce per-token transport hops or "
                "bytes versus split modules on the organ-map [S] link simulator."
            ),
            "expected_removed_cost": "an extra activation round-trip between routed, shared and epilogue modules",
            "prerequisite": "transport_link_simulator; NS-020 is a different (command-buffer topology) mechanism",
            "refutation_probability": "HIGH",
            "related_scars_not_the_same": ["NS-020"],
        },
        {
            "id": "HD-009",
            "axis": "persistent_state",
            "target_model": "qwen27",
            "target_organ": "deltanet_state_and_input_projection",
            "hypothesis": (
                "DeltaNet state must live in the HWIR persistent_state buffer "
                "(lifetime=sequence, per_token_transfer=false). Shipping that state "
                "over the transport link each token contradicts the organ mapping "
                "'persistent state machine with resident state'."
            ),
            "predicted_effect": {
                "direction": "keep_state_resident",
                "magnitude_class": "FACTOR",
            },
            "uncertainty": (
                "State byte count is not in the organ map. The hypothesis is a "
                "consistency claim against declared buffer lifetime, not a measured win."
            ),
            "cheapest_simulator": "static_hwir",
            "falsifier": (
                "An HWIR that sets persistent_state.per_token_transfer=true still "
                "satisfies the organ mapping, or adding per-token state bytes to the "
                "link simulator does not increase transport class."
            ),
            "expected_removed_cost": "per-token transport of sequence-lifetime DeltaNet state",
            "prerequisite": "QWEN27_FPGA_ORGAN_MAP hwir.buffers (present)",
            "refutation_probability": "HIGH",
        },
        {
            "id": "HD-010",
            "axis": "dfx_boundary",
            "target_model": "qwen27",
            "target_organ": "command_buffer_graph",
            "hypothesis": (
                "A DFX boundary around P1 organs (norm_add_epilogues, "
                "lm_head_and_sampling, command_buffer_graph) lets P0 GEMV stay "
                "resident while the P1 graph swaps, matching the module_cache key "
                "algorithm sha256(HWIR + device genome + HBM genome + toolchain)."
            ),
            "predicted_effect": {
                "direction": "enable_p0_residency_across_p1_swap",
                "magnitude_class": "UNKNOWN",
            },
            "uncertainty": (
                "module_cache is SCHEMA_ONLY. No bitstream exists. DFX on an "
                "unselected part is a floorplan claim, not a timing claim."
            ),
            "cheapest_simulator": "static_hwir",
            "falsifier": (
                "The proposed DFX cut crosses a token-path dependency in "
                "hwir.dependencies, or the module_cache key would change per token."
            ),
            "expected_removed_cost": "reloading P0 GEMV whenever a P1 graph changes",
            "prerequisite": "module_cache schema (present, SCHEMA_ONLY); device still unselected",
            "refutation_probability": "HIGH",
        },
        {
            "id": "HD-011",
            "axis": "transport_format",
            "target_model": "qwen27",
            "target_organ": "gqa_qkv_and_output",
            "hypothesis": (
                "The only transport format that can beat an Apple-only path on the "
                "existing partition_simulation is the declared policy: activations "
                "and partial reductions only, never the weight body. Flash adds "
                "route metadata to that same restriction."
            ),
            "predicted_effect": {
                "direction": "reduce_per_token_transport",
                "magnitude_class": "FACTOR",
            },
            "uncertainty": (
                "partition_simulation inputs are labelled [S] scenario parameters. "
                "A real PCIe/HBM number would require a board."
            ),
            "cheapest_simulator": "transport_link_simulator",
            "falsifier": (
                "Adding weight-body bytes to the [S] transfer still leaves mixed "
                "beating apple-only, or the declared transport_policy already "
                "includes the weight body (the organ map says it does not)."
            ),
            "expected_removed_cost": "per-token weight-body transfer on gqa_qkv_and_output",
            "prerequisite": "organ-map transport_policy activations_and_partial_reductions_only (present)",
            "refutation_probability": "HIGH",
        },
        {
            "id": "HD-012",
            "axis": "compute_transfer_overlap",
            "target_model": "qwen27",
            "target_organ": "mlp_gate_up_down",
            "hypothesis": (
                "Overlapping FPGA GEMV compute with the return transfer of partial "
                "reductions hides link latency that partition_simulation currently "
                "adds in series (mixed = fpga_compute + transfer + sync). The first "
                "test is to replace that sum with max(compute, transfer) + sync on "
                "the existing within_ffn_split scenario — still [S], still not a board."
            ),
            "predicted_effect": {
                "direction": "hide_link_latency",
                "magnitude_class": "UNKNOWN",
            },
            "uncertainty": (
                "Whether HWIR dependencies allow the overlap is unread from a "
                "cycle model; only the scenario formula is being changed. The "
                "apple_compute_ns / fpga_compute_ns inputs are hypothetical."
            ),
            "cheapest_simulator": "partition_simulation",
            "falsifier": (
                "Overlapped mixed still does not beat apple-only on within_ffn_split, "
                "or hwir.dependencies require partial_reductions to wait for the "
                "full GEMV so the max() formula is illegal."
            ),
            "expected_removed_cost": "serial addition of transfer latency onto FPGA compute",
            "prerequisite": "organ-map partition_simulation within_ffn_split scenario (exists, [S])",
            "refutation_probability": "MEDIUM",
        },
    ]


# ---------------------------------------------------------------------------
# receipt
# ---------------------------------------------------------------------------


def recovered_implementation() -> list[dict[str, Any]]:
    rows = [
        (
            "tools/headless/doctor_diagnosis.py",
            "representation Doctor: FAMILIES, organ_plan, avoid_list citing NNS, ORGAN/DIAGNOSIS/PRESCRIPTION/AVOID",
        ),
        (
            "tools/headless/doctor_technique_library.py",
            "general TechniqueLibrary; KEEP/PRUNE; applicability per architecture class",
        ),
        (
            "tools/headless/doctor_technique_registry.py",
            "REQUIRED_ENTRY_FIELDS, cheapest() CPU probe, SCAR_* phrases, git-fallback load_json",
        ),
        (
            "hcli/agentos/fpga_preboard.py",
            "author of the FPGA organ maps; MockFPGAProvider; TransportLinkSimulator; partition_simulation",
        ),
        (
            "receipts/headless/FLASH_NEXT_FPGA_ORGAN_MAP.json",
            "7 Flash organs, experiment_dag, HWIR, transport_link_simulator, BLOCKED_NO_BOARD",
        ),
        (
            "receipts/headless/QWEN27_FPGA_ORGAN_MAP.json",
            "6 Qwen27 organs, same experiment_dag shape, APPLE_UMA_PLUS_FPGA_HBM_HYPOTHESIS",
        ),
        (
            "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json",
            "14 named dead levers (large_expert_cache, inter_expert_redundancy, ...)",
        ),
        (
            "receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
            "NS-001..NS-038; hardware-axis REFUTED: NS-018, NS-020, NS-026, NS-027, NS-028, NS-029, NS-031, NS-032",
        ),
        (
            "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
            "NNS-001..NNS-032 PROPERTY_OF_IDEA vs ARTIFACT_OF_METHOD",
        ),
        (
            "tools/headless/negative_science.py",
            "query API prior_failures(); not imported — Codex surface, sibling to this sidecar",
        ),
        (
            "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json",
            "frontier F012 cites experiment_queue and cheapest_falsifier; file is NOT in this HEAD",
        ),
        (
            "receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
            "F004 MISSING Hardware Doctor; this module is that integration_target",
        ),
    ]
    out = []
    for path, what in rows:
        out.append({"path": path, "present": repo_exists(path), "what": what})
    return out


def gaps_closed() -> list[str]:
    return [
        "hardware-axis Doctor with emit() hard-fail on missing required fields",
        "cheapest_simulator fidelity ladder recovered from organ-map experiment_dag / provider_capabilities",
        "ranking by refutation_weight / simulator_cost (information per unit cost)",
        "scar query before emit; sibling NEGATIVE_SCIENCE_INDEX if present, else atlas files via git show",
        "proposals grounded in FLASH_NEXT and QWEN27 FPGA organ maps, not imagined organs",
        "AVOID list for hardware-axis scars (NS-018/020/026/027/028/029/031/032 and atlas large_expert_cache)",
        "atlas field names experiment_queue and entries[].cheapest_falsifier reused even though the atlas file is absent from HEAD",
    ]


def negative_findings(organs: dict[str, Any], scar_pack: dict[str, Any]) -> list[str]:
    findings = [
        "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json is not in this worktree HEAD; experiment_queue / cheapest_falsifier reused as field names only",
        "receipts/future/NEGATIVE_SCIENCE_INDEX.json was absent at build; fell back to atlas files",
        "organ maps and negative-science receipts are not materialized in this sparse checkout; loaded via git show HEAD:<path>",
        "device_genome is TARGET_UNSELECTED and physical_board_present is false on both maps",
        "hbm_genome channels=0, capacity UNKNOWN, bandwidth UNKNOWN",
        "provider_capabilities.cycle_simulation is false; rtl_hls_verifier_surface is CONTRACT_ONLY; hardware_receipt is BLOCKED_NO_BOARD",
        "this lane produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE; every number that would need a GPU, FPGA, or joule meter is UNKNOWN",
        "FPGA is not a civilization and this module does not build an FPGA backend",
    ]
    if scar_pack.get("missing"):
        findings.append(
            "scar sources not readable even via git: " + ", ".join(scar_pack["missing"])
        )
    maps = organs.get("maps") or {}
    if not maps.get("flash-next") or not maps.get("qwen27"):
        findings.append("one or both FPGA organ maps failed to load")
    return findings


def _runtime_compile_state() -> str:
    """What the reachability probe last observed, or UNKNOWN. Never a guess."""
    path = REPO / "receipts" / "future" / "METAL_REACHABILITY.json"
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return "UNKNOWN"
    observed = doc.get("observed_runtime_binding") or {}
    return {"OK": "AVAILABLE"}.get(observed.get("runtime_source_compile"), "UNKNOWN")


def metal_state() -> dict[str, Any]:
    """What this host can actually do with Metal, measured, not quoted.

    The sidecar was repeating "no Metal-capable GPU and no Metal compiler on
    this host" from a blocker list. Half of that is false: the GPU is an M3
    Ultra and it is present. What is genuinely absent is the OFFLINE shader
    compiler -- `xcrun metal` ships with full Xcode, and this host has only the
    Command Line Tools -- so a .metallib cannot be built ahead of time here.

    The distinction matters because the two blockers have different scopes. A
    missing GPU would block every physical measurement. A missing offline
    compiler blocks precompilation, and says nothing on its own about what a
    runtime that compiles shaders from source can do -- which is why that stays
    UNKNOWN below rather than being guessed in either direction.

    This reports capability, never a measurement. No timing, no throughput.
    """
    def _run(*cmd: str) -> tuple[int, str]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return r.returncode, (r.stdout or r.stderr or "").strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            return -1, f"{type(exc).__name__}"

    chip = ""
    rc, out = _run("system_profiler", "SPHardwareDataType")
    if rc == 0:
        for line in out.splitlines():
            if line.strip().startswith("Chip:"):
                chip = line.split(":", 1)[1].strip()
                break
    rc_metal, metal_out = _run("xcrun", "-sdk", "macosx", "metal", "--version")
    rc_dev, dev_dir = _run("xcode-select", "-p")
    return {
        "chip": chip or "unknown",
        "gpu_present": bool(chip),
        "why_gpu": (
            f"{chip} reports as this host's chip; Apple silicon carries an "
            f"integrated Metal GPU"
            if chip
            else "the host chip could not be read"
        ),
        "offline_metal_compiler": rc_metal == 0,
        "offline_metal_compiler_detail": metal_out.splitlines()[0][:200] if metal_out else "",
        "developer_dir": dev_dir if rc_dev == 0 else "",
        "full_xcode_installed": Path("/Applications/Xcode.app").is_dir(),
        "runtime_source_compilation": _runtime_compile_state(),
        "why_runtime_state": (
            "compiling shader source at runtime goes through the Metal framework, "
            "not xcrun. tools/future/metal_reachability.py exercises that path with "
            "the crate the runtime uses; UNKNOWN here means it has not been run yet, "
            "not that it fails"
        ),
        "is_a_measurement": False,
    }


def build() -> Path:
    organs = load_organs()
    scar_pack = load_scars()
    scars = scar_pack["scars"]
    emitted: list[dict[str, Any]] = []
    for raw in catalog():
        emitted.append(emit(raw, organs=organs, scars=scars))
    queue = rank_queue(emitted)
    avoid = avoid_list(scars)
    dead = [s for s in scars if s.get("dead")]
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Propose falsifiable hardware-design experiments on recovered FPGA "
            "organs. Models propose; disk state decides. This receipt is STATIC_ONLY."
        ),
        "vocabulary": {
            "eras": list(ERAS),
            "odysseys": list(ODYSSEYS),
            "fpga_is": "Accelerator / Physical Compiler / Fusion — not its own civilization",
            "evidence": {
                "DIAGNOSTIC_RELATIVE": "contaminated A/B; guides; never promotes; this lane does not produce it",
                "PROTECTED_ABSOLUTE": "protected GPU lease; decides; this lane does not produce it",
                "STATIC_ONLY": "the only evidence class this lane may emit",
            },
        },
        "frontier_id": "F004",
        "axes": list(AXES),
        "required_fields": list(REQUIRED_FIELDS),
        "simulator_ladder": [
            {"fidelity": name, **meta} for name, meta in SIMULATORS.items()
        ],
        "ranking_rule": (
            "expected information per unit cost = refutation_weight / simulator_cost; "
            "prefer the proposal whose cheapest falsifier is cheapest and whose "
            "refutation probability is highest"
        ),
        "organs_recovered": organs,
        "scar_query": {
            "source_used": scar_pack["source_used"],
            "consulted": scar_pack["consulted"],
            "missing": scar_pack["missing"],
            "n_scars": len(scars),
            "n_dead": len(dead),
            "index_present": scar_pack["source_used"] == NEGATIVE_INDEX,
        },
        "avoid": avoid,
        "experiment_queue": queue,
        "entries": queue,
        "n_proposals": len(queue),
        "axes_covered": sorted({p["axis"] for p in queue}),
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(organs, scar_pack),
        "integration": {
            "emit": "emit(proposal, *, organs=None, scars=None) -> dict  # raises MissingFieldError | ScarRefusal | UnknownOrganError",
            "rank_queue": "rank_queue(records) -> list[dict]",
            "load_organs": "load_organs() -> {maps, organs}",
            "load_scars": "load_scars() -> {source_used, consulted, missing, scars}",
            "avoid_list": "avoid_list(scars=None) -> list[dict]  # Doctor AVOID vocabulary",
            "build": "build() -> Path  # receipts/future/HARDWARE_DOCTOR.json",
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
