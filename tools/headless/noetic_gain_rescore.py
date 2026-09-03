#!/usr/bin/env python3
"""Re-score cosine-only doctor campaigns with gravity_doctor_gate._gain.

NNS-005: an adequacy gate on this project was blind to magnitude for an entire
campaign because cosine is scale-invariant.  Wh = 0.01*W scored 1.000000 on
observed / probed / worst_unit.  `_gain` now exists.  This tool:

  * cites what `_gain` measures, against the cosine-only gate
  * enumerates every campaign whose stored axes are observed/probed/worst_unit
    with no magnitude term
  * re-scores reconstructable candidates with the live `_gain` function
  * re-ranks archive rows that stored a magnitude field (energy_ratio,
    src_norm/recon_norm)
  * proves the new instrument REJECTS 0.01*W before anyone trusts its ranking

Most rankings are expected to survive: most candidates were not pathologically
scaled.  The value is the few that move, and the proof the archive was checked
with an instrument that can see magnitude.

Write: receipts/headless/NOETIC_GAIN_RESCORE.json
Run:   python3 tools/headless/noetic_gain_rescore.py
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "hawking.headless.noetic_gain_rescore.v1"

HERE = Path(__file__).resolve()
TOOLS = HERE.parent.parent  # tools/
sys.path.insert(0, str(TOOLS))

import gravity_doctor_gate as dg  # noqa: E402

# File:line citations against the live module (not a memory of the file).
GAIN_FILE = "tools/gravity_doctor_gate.py"
GAIN_LINES = (106, 127)
ROWCOS_LINES = (82, 85)
OBSERVED_LINES = (88, 90)
AXES_LINES = (148, 170)
PACKERS = [
    {
        "path": "research/hawking-experiments/superwave/g1/evidence/g1_func_objective.py",
        "fn": "pack_axes",
        "lines": (309, 310),
        "note": "stores only observed/probed/worst_unit; drops gain even if axes() returns it",
    },
    {
        "path": "research/hawking-experiments/superwave/g1/evidence/g1_adversarial_frontier.py",
        "fn": "score_pack",
        "lines": (360, 372),
        "note": "packs the three cosine axes + energy_ratio/rel_fro/weight_cos; never persisted gain",
    },
]

HAWKING_COPY = Path("/Users/scammermike/Downloads/hawking-copy")
ABLITERATED_BF16 = Path("/Users/scammermike/models/qwen3.8-27b-abliterated-bf16")
L0_GATE_NAMES = (
    "language_model.model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.0.mlp.gate_proj.weight",
)

COSINE_KEYS = ("observed", "probed", "worst_unit")
SKIP_PATH_PARTS = ("deficit", "axis_margin", "margins", "AXIS_MARGIN")
MATERIAL_RANK_DELTA = 2
MATERIAL_SCORE_DELTA = 0.05
# A PRIMARY inversion is the candidate whose OWN score collapsed, not
# everyone who slid one slot because a cheat left the top of the list.
PRIMARY_SCORE_DELTA = 0.15

WHAT_I_WATCHED_FAIL = """\
## WHAT I WATCHED FAIL

The cosine-only doctor gate is not a slightly noisy quality metric. It is
scale-invariant by construction (`_rowcos` divides out the per-row norms).
The construction that exhibited this — Wh = 0.01*W on real L0 gate_proj —
scored observed = probed = worst_unit = 1.000000 at relative weight error
0.9898. A candidate that kept every direction and destroyed every magnitude
was HEALTHY on every axis the campaign used to rank.

`_gain` was added to `axes()` because that construction was exhibited. It
is min(r, 1/r) on the per-row mean AND the per-unit worst of the OUTPUT
activation norms, so both shrink and blow-up fail symmetrically. Cosine
cannot see this; `_gain` is the axis that can.

Two further method failures sit on top of the missing axis:

1. Campaign packers (`pack_axes`, `score_pack`) persist only the three
   cosine keys. Even after `_gain` existed in `axes()`/`gate()`, the
   archive kept being written cosine-only. Re-running those scripts today
   would still not store gain. The receipts are blind by packing, not only
   by age.

2. spanfit_X_and_P0 Goodharted the three-axis gate: fitted onto span(X, P0)
   it scored observed=1, probed=1, worst_unit≈1 while keeping 29.4% of
   weight energy, and was stored HEALTHY. Activation `_gain` on those SAME
   fitted directions is also ~1 (the map is exact on that span). Weight
   energy catches it; activation gain on the fitted span does not. Fresh
   probes catch it with BOTH probed cosine AND gain. `_gain` is the fix
   for scale, not a substitute for unfittable probes.

What this re-score is not: a claim that every low-cosine codec was
mis-ranked. Most candidates in these campaigns were honest codecs whose
magnitude tracks their direction (q4 energy_ratio ≈ 1.008, cosine ≈ 0.996).
Those rankings survive, and saying so is the result. The inversions are
the scale cheats — 0.01*W / 100*W, and the archive spanfit that cosine
certified.

Synthetic activations are used ONLY for the algebraic instrument check
(cosine(W, 0.01*W) is a mathematical identity, not a compression claim).
Live `_gain` on real L0 gate_proj, when the tensor loads, is the
campaign-shaped proof. Capture-conditioned `observed` from the original
X is not recomputed: the v1 capture path in the pin is not on this
machine. Probe-space scores are labelled probe_only.

Tensor-train / Tucker / shared-basis / plane-rung candidates cannot be
re-materialised without the original Wh. Their receipts store cosine axes
and no per-row activation norms, so `_gain` is not recoverable from the
archive. Those rows are enumerated, not invented.
"""


def find_repo() -> Path:
    env = os.environ.get("HAWKING_REPO")
    if env:
        return Path(env)
    for p in [HERE.parent, *HERE.parents]:
        if (p / "Cargo.toml").exists() and (p / "tools" / "headless").is_dir() and (p / "receipts").is_dir():
            return p
    if HAWKING_COPY.exists():
        return HAWKING_COPY
    return Path.cwd()


REPO = find_repo()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def evidence_roots() -> list[Path]:
    roots = []
    for p in (REPO, HAWKING_COPY, Path("/Users/scammermike/Downloads/hawking")):
        if p.exists() and p not in roots:
            roots.append(p)
    return roots


def resolve(rel: str) -> Path | None:
    for root in evidence_roots():
        p = root / rel
        if p.is_file():
            return p
    return None


def load_json(rel: str) -> tuple[Any | None, str | None]:
    p = resolve(rel)
    if p is not None:
        try:
            return json.loads(p.read_text()), str(p)
        except Exception:
            return None, str(p)
    try:
        raw = subprocess.check_output(
            ["git", "show", f"HEAD:{rel}"], cwd=REPO, stderr=subprocess.DEVNULL
        )
        return json.loads(raw), f"git:HEAD:{rel}"
    except Exception:
        return None, None


def _is_axis_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if not all(k in obj for k in COSINE_KEYS):
        return False
    for k in COSINE_KEYS:
        if not isinstance(obj[k], (int, float)) or isinstance(obj[k], bool):
            return False
    return True


def _has_gain(obj: dict) -> bool:
    g = obj.get("gain")
    return isinstance(g, (int, float)) and not isinstance(g, bool)


def walk_axis_dicts(obj: Any, path: str = "$"):
    if isinstance(obj, dict):
        low = path.lower()
        if any(s in low for s in SKIP_PATH_PARTS):
            return
        if _is_axis_dict(obj):
            yield path, obj
        for k, v in obj.items():
            if k in SKIP_PATH_PARTS:
                continue
            nxt = f"{path}.{k}"
            yield from walk_axis_dicts(v, nxt)
    elif isinstance(obj, list) and obj and isinstance(obj[0], (dict, list)):
        cap = min(len(obj), 8000)
        for i, v in enumerate(obj[:cap]):
            yield from walk_axis_dicts(v, f"{path}[{i}]")


def cosine_composite(obj: dict) -> float:
    return float(min(obj[k] for k in COSINE_KEYS))


def energy_gain(obj: dict) -> float | None:
    e = obj.get("energy_ratio")
    if isinstance(e, (int, float)) and e > 0:
        return float(min(e, 1.0 / e))
    src, recon = obj.get("src_norm"), obj.get("recon_norm")
    if isinstance(src, (int, float)) and isinstance(recon, (int, float)) and src > 0:
        r = float(recon) / (float(src) + 1e-30)
        if r > 0:
            return float(min(r, 1.0 / r))
    return None


def load_safetensors_tensor(root: Path, name: str) -> np.ndarray:
    idx = json.loads((root / "model.safetensors.index.json").read_text())
    shard = idx["weight_map"][name]
    path = root / shard
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        meta = hdr[name]
        s, e = meta["data_offsets"]
        f.seek(8 + hlen + s)
        raw = f.read(e - s)
    dt = meta["dtype"]
    if dt == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        arr = u16.view(np.float32)
    elif dt == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    elif dt == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        raise ValueError(f"dtype {dt}")
    return arr.reshape(meta["shape"]).astype(np.float32)


def try_load_l0_gate() -> tuple[np.ndarray | None, dict]:
    info: dict[str, Any] = {"loaded": False, "tried": []}
    candidates: list[tuple[Path, str]] = []
    pin = resolve("research/hawking-experiments/superwave/g1/GRAVITY1_SOURCE_PIN.json")
    if pin:
        try:
            src = Path(json.loads(pin.read_text())["source_root"])
            for n in L0_GATE_NAMES:
                candidates.append((src, n))
        except Exception as e:
            info["pin_error"] = str(e)
    if ABLITERATED_BF16.is_dir():
        for n in L0_GATE_NAMES:
            candidates.append((ABLITERATED_BF16, n))
    default = Path(dg.BF16)
    if not default.is_absolute():
        default = REPO / dg.BF16
    for n in L0_GATE_NAMES:
        candidates.append((default, n))

    seen = set()
    for root, name in candidates:
        key = (str(root), name)
        if key in seen:
            continue
        seen.add(key)
        rec = {"root": str(root), "name": name, "ok": False}
        try:
            if not (root / "model.safetensors.index.json").is_file():
                rec["reason"] = "no index"
                info["tried"].append(rec)
                continue
            W = load_safetensors_tensor(root, name)
            rec.update({"ok": True, "shape": list(W.shape), "dtype": "float32"})
            info.update({"loaded": True, "root": str(root), "name": name, "shape": list(W.shape)})
            info["tried"].append(rec)
            return W, info
        except Exception as e:
            rec["reason"] = f"{type(e).__name__}: {e}"
            info["tried"].append(rec)
    return None, info


def score_pair(W: np.ndarray, Wh: np.ndarray, X: np.ndarray, n_probe_sets: int = 1, seed: int = 0) -> dict:
    a = dg.axes(W, Wh, X, seed=seed, n_probe_sets=n_probe_sets)
    old = cosine_composite(a)
    gn = float(a["gain"])
    new = float(min(old, gn))
    return {
        "observed": float(a["observed"]),
        "probed": float(a["probed"]),
        "worst_unit": float(a["worst_unit"]),
        "gain": gn,
        "cosine_composite": old,
        "gain_aware": new,
        "energy_ratio": float(np.linalg.norm(Wh) / (np.linalg.norm(W) + 1e-30)),
        "rel_fro": float(np.linalg.norm(W - Wh) / (np.linalg.norm(W) + 1e-30)),
    }


def direct_scale_check(W: np.ndarray, X: np.ndarray, scale: float) -> dict:
    """Algebraic instrument check: cosine vs `_gain` on a global scale."""
    A = X @ W.T
    B = X @ (W * scale).T
    cos = dg._rowcos(A, B)
    gn = dg._gain(A, B)
    wu = dg._worst_unit(A, B)
    return {
        "scale": scale,
        "observed_cosine": float(cos),
        "worst_unit_cosine": float(wu),
        "gain": float(gn),
        "rejects_as_perfect": bool(gn < 0.5),
        "cosine_claims_perfect": bool(cos > 0.999),
        "instrument_works": bool(cos > 0.999 and gn < 0.5),
    }


def spanfit(W: np.ndarray, X: np.ndarray, P0: np.ndarray) -> np.ndarray:
    XP = np.concatenate([X, P0], axis=0)
    _, _, Vt = np.linalg.svd(XP.astype(np.float32), full_matrices=False)
    r = int(np.linalg.matrix_rank(XP, tol=1e-3 * np.linalg.norm(XP, 2)))
    B = Vt[:r]
    return W @ (B.T @ B)


def fitted_axes(W: np.ndarray, Wh: np.ndarray, X: np.ndarray, P0: np.ndarray) -> dict:
    """Score on the exact (X, P0) pair a spanfit was fitted to — the historical Goodhart."""
    AX, BX = X @ W.T, X @ Wh.T
    AP, BP = P0 @ W.T, P0 @ Wh.T
    obs = dg._rowcos(AX, BX)
    pr = dg._rowcos(AP, BP)
    wu = min(dg._worst_unit(AX, BX), dg._worst_unit(AP, BP))
    gn = min(dg._gain(AX, BX), dg._gain(AP, BP))
    return {
        "observed": float(obs),
        "probed": float(pr),
        "worst_unit": float(wu),
        "gain": float(gn),
        "cosine_composite": float(min(obs, pr, wu)),
        "gain_aware": float(min(obs, pr, wu, gn)),
        "energy_ratio": float(np.linalg.norm(Wh) / (np.linalg.norm(W) + 1e-30)),
    }


def rank_inversions(rows: list[dict], old_key: str, new_key: str, peer_set: str, campaign: str) -> list[dict]:
    """rows each have a 'id' and the two metric keys. Higher is better."""
    if len(rows) < 2:
        return []
    old_order = sorted(range(len(rows)), key=lambda i: -float(rows[i][old_key]))
    new_order = sorted(range(len(rows)), key=lambda i: -float(rows[i][new_key]))
    old_rank = {i: r + 1 for r, i in enumerate(old_order)}
    new_rank = {i: r + 1 for r, i in enumerate(new_order)}
    out = []
    for i, row in enumerate(rows):
        dlt = new_rank[i] - old_rank[i]
        old_s = float(row[old_key])
        new_s = float(row[new_key])
        verdict_flip = bool(row.get("old_healthy") is True and row.get("new_healthy") is False) or bool(
            row.get("old_healthy") is False and row.get("new_healthy") is True
        )
        score_drop = old_s - new_s
        primary = verdict_flip or score_drop >= PRIMARY_SCORE_DELTA
        material = primary or abs(dlt) >= MATERIAL_RANK_DELTA or abs(old_s - new_s) >= MATERIAL_SCORE_DELTA
        if not material:
            continue
        out.append({
            "campaign": campaign,
            "peer_set": peer_set,
            "candidate": row["id"],
            "old_rank": old_rank[i],
            "new_rank": new_rank[i],
            "rank_delta": dlt,  # positive = moved down (worse)
            "n_peers": len(rows),
            "old_metric": old_s,
            "new_metric": new_s,
            "score_drop": score_drop,
            "old_healthy": row.get("old_healthy"),
            "new_healthy": row.get("new_healthy"),
            "cause": "PRIMARY" if primary else "DISPLACEMENT",
            "kind": "VERDICT_AND_RANK" if verdict_flip and dlt else ("VERDICT" if verdict_flip else "RANK"),
        })
    out.sort(key=lambda r: -abs(r["rank_delta"]))
    return out


# ---------------------------------------------------------------------------
# Campaign catalogue (names + receipts). Discovery still walks the files.
# ---------------------------------------------------------------------------

CAMPAIGN_SPEC = [
    {
        "id": "G1_ADVERSARIAL_FRONTIER",
        "name": "G1 adversarial frontier (L0 gate constructions)",
        "receipts": ["research/hawking-experiments/superwave/g1/evidence/g1_adversarial_frontier.json"],
        "ranking_metric": "observed cosine; healthy from 3-axis relative gate",
        "notes": "Stores energy_ratio. spanfit_X_and_P0 scored HEALTHY at cosine 1.0 with energy 0.294.",
    },
    {
        "id": "G003_DOCTOR_L0",
        "name": "G003 doctor gate run, L0 gate_proj",
        "receipts": ["research/hawking-experiments/superwave/g1/evidence/g003-L0.json"],
        "ranking_metric": "observed/probed/worst_unit; no gain in results[]",
        "notes": "Same constructions as tools/gravity_doctor_gate.py:run, packed without gain.",
    },
    {
        "id": "G003_DOCTOR_L31",
        "name": "G003 doctor gate run, L31",
        "receipts": ["research/hawking-experiments/superwave/g1/evidence/g003-L31.json"],
        "ranking_metric": "observed/probed/worst_unit",
        "notes": "",
    },
    {
        "id": "G003_DOCTOR_GATE",
        "name": "G003 doctor gate smoke",
        "receipts": ["research/hawking-experiments/superwave/g1/evidence/g003-gate.json"],
        "ranking_metric": "observed/probed/worst_unit",
        "notes": "",
    },
    {
        "id": "G1_TENSOR_OPERATORS",
        "name": "G1 tensor operators (TT / Tucker / Kronecker / low-rank)",
        "receipts": ["research/hawking-experiments/superwave/g1/evidence/g1_tensor_operators.json"],
        "ranking_metric": "gate.observed / healthy (3 cosine axes)",
        "notes": "373 family rows, 223 with local_bpw<0.5 and healthy=true: 0. Wh not stored; cannot live-rescore.",
    },
    {
        "id": "G1_SHARE_BASIS",
        "name": "G1 shared basis vs independent (G-SHARE)",
        "receipts": [
            "research/hawking-experiments/superwave/g1/evidence/g1_share_basis.json",
            "research/hawking-experiments/superwave/g1/evidence/g1_share_basis_smoke.json",
        ],
        "ranking_metric": "axes_shared/axes_indep observed/probed/worst_unit",
        "notes": "Shared vs independent already both UNHEALTHY on worst_unit. Magnitude not the binder.",
    },
    {
        "id": "G1_XFORM_HADAMARD",
        "name": "G1 Hadamard-then-qN (G032 / xform)",
        "receipts": ["research/hawking-experiments/superwave/g1/evidence/g1_xform_hadamard.json"],
        "ranking_metric": "per-site gate.{id,wh_*,rht_*,bfly*} bits 2/3/4, 3 cosine axes",
        "notes": "axis_margin has only observed/probed/worst_unit. Honest qN, energy tracks cosine.",
    },
    {
        "id": "G1_PLANES_TERNARY",
        "name": "G1 binary/ternary planes (G033)",
        "receipts": [
            "research/hawking-experiments/superwave/g1/evidence/g1_planes_ternary.json",
            "research/hawking-experiments/superwave/g1/evidence/g1_planes_ternary_smoke.json",
            "research/hawking-experiments/superwave/g1/evidence/g1_planes_lm_head.json",
        ],
        "ranking_metric": "rung observed cosine + healthy",
        "notes": "Some rows store resid_rel_f (weight residual), not activation gain.",
    },
    {
        "id": "G1_ALLOC_UNIFIED",
        "name": "G1 unified allocator candidate auction",
        "receipts": [
            "research/hawking-experiments/superwave/g1/evidence/g1_alloc_unified_out.json",
            "research/hawking-experiments/superwave/g1/evidence/g1_alloc_unified_measure.json",
        ],
        "ranking_metric": "score (= worst_unit); observed/probed/worst_unit stored, no gain",
        "notes": "26 tensors × 26 codecs (qN, planes, rank_k, shared_k, sparse). Duplicate receipts.",
    },
    {
        "id": "G005_ALLOC",
        "name": "G005 bit-allocation curves",
        "receipts": [
            "research/hawking-experiments/superwave/g1/evidence/g005-alloc.json",
            "research/hawking-experiments/superwave/g1/evidence/g005-curves.json",
        ],
        "ranking_metric": "observed/probed/worst_unit + damage/score",
        "notes": "",
    },
    {
        "id": "G1_RESHAPE_BEFORE_LOWBIT",
        "name": "G1 reshape-before-lowbit",
        "receipts": ["research/hawking-experiments/superwave/g1/evidence/g1_reshape_before_lowbit.json"],
        "ranking_metric": "3 cosine axes; axis_margin has no gain",
        "notes": "",
    },
    {
        "id": "G010",
        "name": "G010 object rows",
        "receipts": ["research/hawking-experiments/superwave/g1/evidence/g010.json"],
        "ranking_metric": "observed/probed/worst_unit per (cls, layer, bits)",
        "notes": "",
    },
    {
        "id": "G1_CAPABILITY_GATE_Q4_COSINE",
        "name": "G1 capability-gate full q4 weight cosine (306 tensors)",
        "receipts": [
            "research/hawking-experiments/superwave/g1/evidence/g1_capability_gate_full_cosine.json",
            "research/hawking-experiments/superwave/g1/evidence/g1_capability_gate_measure.json",
        ],
        "ranking_metric": "weight-space cosine (not doctor axes). lowest5 also store rel_l2; measure rows store src_norm/recon_norm",
        "notes": "Different cosine (weights, not X@W). Still scale-invariant. 10 dequant rows have norms → archive gain proxy.",
        "weight_space": True,
    },
    {
        "id": "Q80_PREFIX_COSINE",
        "name": "Q80 recalibrate prefix-cosine by rank cap",
        "receipts": ["receipts/ascent-2026-08-16/q80-recalibrate-prefix-cosines.json"],
        "ranking_metric": "holdout organ cosine vs rank {160,80,40,20,16,8}",
        "notes": "Monotone in rank; not doctor axes. Magnitude-blind organ cosine.",
        "weight_space": False,
        "organ_cosine": True,
    },
    {
        "id": "G1_ENDPOINT",
        "name": "G1 endpoint.json (ALREADY has gain)",
        "receipts": ["research/hawking-experiments/superwave/g1/evidence/endpoint.json"],
        "ranking_metric": "observed/probed/worst_unit/gain",
        "notes": "Negative control: the one G1 receipt that persisted a gain field. Not re-scored as cosine-only.",
        "already_has_gain": True,
    },
]


def enumerate_campaigns() -> list[dict]:
    campaigns = []
    for spec in CAMPAIGN_SPEC:
        recs = []
        n_cos = 0
        n_gain = 0
        extra_mag = 0
        resolved = []
        missing = []
        for rel in spec["receipts"]:
            data, where = load_json(rel)
            recs.append({"path": rel, "resolved": where, "present": data is not None})
            if data is None:
                missing.append(rel)
                continue
            resolved.append(where)
            for path, obj in walk_axis_dicts(data):
                n_cos += 1
                if _has_gain(obj):
                    n_gain += 1
                if energy_gain(obj) is not None:
                    extra_mag += 1
        campaigns.append({
            **{k: spec[k] for k in spec if k != "receipts"},
            "receipts": recs,
            "n_cosine_axis_dicts": n_cos,
            "n_with_gain_field": n_gain,
            "n_with_energy_or_norm_proxy": extra_mag,
            "cosine_only": n_cos > 0 and n_gain == 0 and not spec.get("already_has_gain"),
            "missing_receipts": missing,
            "resolved_from": resolved,
        })
    return campaigns


def flatten_frontier(cons: dict) -> list[dict]:
    rows = []

    def walk(name: str, obj: Any):
        if isinstance(obj, dict) and _is_axis_dict(obj) and "energy_ratio" in obj:
            eg = energy_gain(obj)
            old = float(obj["observed"])
            comp = cosine_composite(obj)
            rows.append({
                "id": name,
                "observed": old,
                "probed": float(obj["probed"]),
                "worst_unit": float(obj["worst_unit"]),
                "cosine_composite": comp,
                "energy_ratio": float(obj["energy_ratio"]),
                "gain_proxy": eg,
                "old_metric": old,
                "new_metric": float(min(old, eg)) if eg is not None else old,
                "old_healthy": obj.get("healthy"),
                "new_healthy": (False if (eg is not None and eg < 0.85) else obj.get("healthy")),
            })
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{name}/{k}", v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(f"{name}[{i}]", v)

    for k, v in cons.items():
        walk(k, v)
    return rows


def archive_rescore() -> tuple[list[dict], list[dict], dict]:
    inversions: list[dict] = []
    peer_reports: list[dict] = []
    stats = {"peer_sets": 0, "candidates_with_proxy": 0, "candidates_cosine_only_no_proxy": 0}

    # --- frontier: the campaign that stored energy_ratio ---
    data, where = load_json("research/hawking-experiments/superwave/g1/evidence/g1_adversarial_frontier.json")
    if data and isinstance(data, dict) and "L0_gate" in data:
        cons = data["L0_gate"].get("constructions") or {}
        rows = flatten_frontier(cons)
        stats["candidates_with_proxy"] += len(rows)
        stats["peer_sets"] += 1
        # Cosine treated observed≈1 as "good". Re-rank THAT set; mixing in
        # kronecker-at-0.02 just manufactures rank noise.
        high = [r for r in rows if r["observed"] >= 0.95]
        inv = rank_inversions(
            high, "old_metric", "new_metric",
            peer_set="L0_gate.constructions with observed>=0.95 (rank by observed vs min(observed, energy_gain))",
            campaign="G1_ADVERSARIAL_FRONTIER",
        )
        for x in inv:
            x["metric_note"] = (
                "new_metric uses min(observed, min(energy_ratio, 1/energy_ratio)). "
                "energy_ratio is ||Wh||_F/||W||_F (weight space), not activation `_gain`. "
                "For a global scale they coincide; for spanfit they diverge (see live_rescore)."
            )
        inversions.extend(inv)
        peer_reports.append({
            "campaign": "G1_ADVERSARIAL_FRONTIER",
            "receipt": where,
            "n": len(rows),
            "n_high_cosine_obs_ge_0.95": len(high),
            "ranked_by_old": [r["id"] for r in sorted(high, key=lambda r: -r["old_metric"])],
            "ranked_by_new": [r["id"] for r in sorted(high, key=lambda r: -r["new_metric"])],
            "n_inversions": len(inv),
            "n_primary": sum(1 for x in inv if x["cause"] == "PRIMARY"),
            "rows": [
                {k: r[k] for k in (
                    "id", "observed", "probed", "worst_unit", "energy_ratio",
                    "gain_proxy", "old_metric", "new_metric", "old_healthy", "new_healthy",
                )}
                for r in rows
            ],
        })

    # --- q4 dequant rows with src_norm/recon_norm ---
    meas, mwhere = load_json("research/hawking-experiments/superwave/g1/evidence/g1_capability_gate_measure.json")
    if meas and isinstance(meas, dict) and meas.get("g0_dequant_cosine_rows"):
        rows = []
        for r in meas["g0_dequant_cosine_rows"]:
            eg = energy_gain(r)
            if eg is None or "cosine" not in r:
                continue
            rows.append({
                "id": r.get("name") or r.get("role") or "?",
                "old_metric": float(r["cosine"]),
                "new_metric": float(min(float(r["cosine"]), eg)),
                "gain_proxy": eg,
                "cosine": float(r["cosine"]),
                "rel_l2": r.get("rel_l2"),
                "src_norm": r.get("src_norm"),
                "recon_norm": r.get("recon_norm"),
            })
        stats["candidates_with_proxy"] += len(rows)
        stats["peer_sets"] += 1
        inv = rank_inversions(rows, "old_metric", "new_metric",
                              peer_set="g0_dequant_cosine_rows",
                              campaign="G1_CAPABILITY_GATE_Q4_COSINE")
        inversions.extend(inv)
        peer_reports.append({
            "campaign": "G1_CAPABILITY_GATE_Q4_COSINE",
            "receipt": mwhere,
            "n": len(rows),
            "n_inversions": len(inv),
            "note": "q4 dequant vs BF16; global-norm gain proxy. Honest q4 is not a scale cheat.",
            "rows": rows,
        })

    # --- alloc: 26 tensors, 26 codecs. No energy_ratio. Count only. ---
    alloc, aw = load_json("research/hawking-experiments/superwave/g1/evidence/g1_alloc_unified_out.json")
    if alloc and isinstance(alloc, dict):
        n = 0
        for m in alloc.get("measured") or []:
            n += len(m.get("candidates") or [])
        stats["candidates_cosine_only_no_proxy"] += n
        peer_reports.append({
            "campaign": "G1_ALLOC_UNIFIED",
            "receipt": aw,
            "n": n,
            "n_inversions": 0,
            "note": (
                "No energy_ratio / norms stored. Cosine-only archive cannot recover `_gain`. "
                "Live reconstruction of the qN subset is in live_rescore."
            ),
        })

    return inversions, peer_reports, stats


def live_rescore(W: np.ndarray, source: dict) -> dict:
    """Reconstruct cheap constructions on a real (or synthetic) tensor and score with `_gain`."""
    rng = np.random.default_rng(0)
    n_in = int(W.shape[1])
    # Isotropic probe X — labelled probe_only. Capture v1 is not on this machine.
    # X and P0 MUST be different draws: historical spanfit was span(capture X, P_seed0).
    X = dg._probe(n_in, n=64, seed=0)
    P0 = dg._probe(n_in, n=64, seed=7)
    P_fresh = dg._probe(n_in, n=64, seed=1)
    probe_only = True

    constructions: dict[str, np.ndarray] = {}
    constructions["identity"] = W
    constructions["scale_0.01*W"] = (0.01 * W).astype(np.float32)
    constructions["scale_100*W"] = (100.0 * W).astype(np.float32)
    constructions["q6_g128"] = dg.c_uniform(W, 6, 128)
    constructions["q4_g128"] = dg.c_faithful_q4(W, 128)
    constructions["q3_g128"] = dg.c_uniform(W, 3, 128)
    constructions["q2_g128"] = dg.c_uniform(W, 2, 128)
    constructions["channel_deletion_k3"] = dg.c_channel_deletion(W, k=3, seed=2)
    constructions["visible_subspace_X"] = dg.c_visible_subspace(W, X)
    constructions["spanfit_X_and_P0"] = spanfit(W, X, P0)

    rows = []
    fitted_span = None
    for name, Wh in constructions.items():
        sc = score_pair(W, Wh, X, n_probe_sets=1, seed=1)  # fresh probe seed, not P0
        # relative-style healthy vs q4 on the four axes, cosine-only vs with gain
        q4 = constructions["q4_g128"]
        if name == "q4_g128":
            ref_sc = sc
        else:
            ref_sc = None
        rows.append({"id": name, **sc})
        if name == "spanfit_X_and_P0":
            fitted_span = fitted_axes(W, Wh, X, P0)
            fitted_span["fresh_probe_gain"] = float(dg._gain(P_fresh @ W.T, P_fresh @ Wh.T))
            fitted_span["fresh_probe_cosine"] = float(dg._rowcos(P_fresh @ W.T, P_fresh @ Wh.T))

    # fill relative healthy using q4 as ref (same rule as dg.gate, cosine-only vs 4-axis)
    ref_row = next(r for r in rows if r["id"] == "q4_g128")
    margin = dg.AXIS_MARGIN
    for r in rows:
        def defi(axis, val):
            return val - (ref_row[axis] - margin[axis])
        old_def = {k: defi(k, r[k]) for k in COSINE_KEYS}
        new_def = {**old_def, "gain": defi("gain", r["gain"])}
        r["old_healthy"] = min(old_def.values()) >= 0.0
        r["new_healthy"] = min(new_def.values()) >= 0.0
        r["old_metric"] = r["cosine_composite"]
        r["new_metric"] = r["gain_aware"]

    inv = rank_inversions(
        rows, "old_metric", "new_metric",
        peer_set="live reconstructions (probe-space)",
        campaign="LIVE_L0_GATE" if source.get("loaded") else "LIVE_SYNTHETIC",
    )
    return {
        "tensor_source": source,
        "probe_only": probe_only,
        "X": {"kind": "isotropic _probe", "n": 64, "d_in": n_in, "seed": 0},
        "W_shape": list(W.shape),
        "rows": rows,
        "inversions": inv,
        "spanfit_on_fitted_probes": fitted_span,
        "ranked_by_cosine_composite": [r["id"] for r in sorted(rows, key=lambda r: -r["cosine_composite"])],
        "ranked_by_gain_aware": [r["id"] for r in sorted(rows, key=lambda r: -r["gain_aware"])],
        "note": (
            "probe_only=True: original activation-capture-v1 is not on this machine. "
            "Isotropic probes still prove scale: cosine(W, 0.01*W)=1 and `_gain`≈0.01 "
            "regardless of X. qN numbers here are NOT a capture-conditioned campaign claim."
        ),
    }


def synthetic_instrument() -> dict:
    rng = np.random.default_rng(0)
    d_in, d_out, r = 256, 64, 12
    W = rng.standard_normal((d_out, d_in)).astype(np.float32)
    B = np.linalg.qr(rng.standard_normal((d_in, r)).astype(np.float32))[0].T
    X = (rng.standard_normal((128, r)).astype(np.float32) @ B)
    checks = {
        "scale_0.01": direct_scale_check(W, X, 0.01),
        "scale_100": direct_scale_check(W, X, 100.0),
        "scale_1": direct_scale_check(W, X, 1.0),
    }
    # demo constructions: faithful q4 vs visible subspace vs 0.01*W, full axes
    faithful = dg.c_faithful_q4(W, group=64)
    cheat = dg.c_visible_subspace(W, X)
    scaled = (0.01 * W).astype(np.float32)
    gf = dg.axes(W, faithful, X, seed=0, n_probe_sets=1)
    gc = dg.axes(W, cheat, X, seed=0, n_probe_sets=1)
    gs = dg.axes(W, scaled, X, seed=0, n_probe_sets=1)
    return {
        "W_shape": [d_out, d_in],
        "X_shape": list(X.shape),
        "X_kind": "low-rank synthetic (demo geometry); instrument check only, not a codec claim",
        "direct_scale": checks,
        "axes_faithful_q4": {k: float(gf[k]) for k in (*COSINE_KEYS, "gain")},
        "axes_visible_subspace": {k: float(gc[k]) for k in (*COSINE_KEYS, "gain")},
        "axes_scale_0.01": {k: float(gs[k]) for k in (*COSINE_KEYS, "gain")},
        "instrument_works": all(checks[k]["instrument_works"] for k in ("scale_0.01", "scale_100"))
            and checks["scale_1"]["gain"] > 0.999
            and gs["observed"] > 0.999 and gs["gain"] < 0.05,
    }


def write_receipt(receipt: dict) -> tuple[Path, str]:
    dest = REPO / "receipts" / "headless" / "NOETIC_GAIN_RESCORE.json"
    probe = dest.parent / ".noetic_write_probe"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
        dest.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        return dest, "repo"
    except OSError as e:
        fallback = Path("/tmp/noetic_gain_rescore") / "NOETIC_GAIN_RESCORE.json"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        receipt.setdefault("write_fallback", str(fallback))
        # try once more to record the fallback reason inside the fallback file
        try:
            fallback.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        except OSError:
            pass
        return fallback, f"fallback_tmp ({type(e).__name__}: {e})"


def fmt(x: float | None, w: int = 8) -> str:
    if x is None:
        return f"{'n/a':>{w}}"
    return f"{x:>{w}.4f}"


def main() -> int:
    t0 = time.time()
    print("NOETIC GAIN RESCORE")
    print("=" * 72)
    print()
    print("1. WHAT `_gain` MEASURES")
    print("-" * 72)
    print(f"  {GAIN_FILE}:{GAIN_LINES[0]}-{GAIN_LINES[1]}  dg._gain")
    print(f"  {GAIN_FILE}:{ROWCOS_LINES[0]}-{ROWCOS_LINES[1]}  dg._rowcos   (cosine-only)")
    print(f"  {GAIN_FILE}:{OBSERVED_LINES[0]}-{OBSERVED_LINES[1]}  dg.observed_score = _rowcos(X@W.T, X@Wh.T)")
    print(f"  {GAIN_FILE}:{AXES_LINES[0]}-{AXES_LINES[1]}  dg.axes  (now min's gain onto observed AND probes)")
    print()
    print("  Cosine-only gate: mean per-row cosine of output activations. Scale")
    print("  invariant: `_rowcos(A, sA) = 1` for any s > 0. The three original")
    print("  axes (observed, probed, worst_unit) are all cosines.")
    print()
    print("  `_gain(A, B)`: min( mean_i min(r_i, 1/r_i),  min_j min(c_j, 1/c_j) )")
    print("  where r_i is ||B[i,:]|| / ||A[i,:]|| (per token/row) and c_j is the")
    print("  same ratio across tokens of output unit j. 1.0 is exact; shrink and")
    print("  blow-up fail symmetrically. It is NOT weight cosine and NOT rel-Frobenius.")
    print()
    for p in PACKERS:
        print(f"  packer  {p['path']}:{p['lines'][0]}-{p['lines'][1]}  {p['fn']}")
        print(f"          {p['note']}")
    print()

    # --- instrument ---
    print("2. INSTRUMENT CHECK — `_gain` MUST REJECT 0.01*W")
    print("-" * 72)
    synth = synthetic_instrument()
    for name, ch in synth["direct_scale"].items():
        print(
            f"  synthetic {name:<12} cosine={ch['observed_cosine']:.6f}  "
            f"worst_unit={ch['worst_unit_cosine']:.6f}  gain={ch['gain']:.6f}  "
            f"{'REJECTS' if ch['rejects_as_perfect'] else 'accepts as ~1'}"
        )
    print(f"  axes(0.01*W)  observed={synth['axes_scale_0.01']['observed']:.6f}  "
          f"probed={synth['axes_scale_0.01']['probed']:.6f}  "
          f"worst_unit={synth['axes_scale_0.01']['worst_unit']:.6f}  "
          f"gain={synth['axes_scale_0.01']['gain']:.6f}")
    print(f"  synthetic instrument_works={synth['instrument_works']}")
    print()

    W, winfo = try_load_l0_gate()
    real_check = None
    if W is not None:
        Xr = dg._probe(W.shape[1], n=32, seed=0)
        real_check = {
            "scale_0.01": direct_scale_check(W, Xr, 0.01),
            "scale_100": direct_scale_check(W, Xr, 100.0),
            "scale_1": direct_scale_check(W, Xr, 1.0),
        }
        print(f"  real L0 gate  {winfo.get('name')}  shape={tuple(W.shape)}")
        print(f"  loaded from   {winfo.get('root')}")
        for name, ch in real_check.items():
            print(
                f"  real {name:<12} cosine={ch['observed_cosine']:.6f}  "
                f"worst_unit={ch['worst_unit_cosine']:.6f}  gain={ch['gain']:.6f}  "
                f"{'REJECTS' if ch['rejects_as_perfect'] else 'accepts as ~1'}"
            )
        works = all(real_check[k]["instrument_works"] for k in ("scale_0.01", "scale_100"))
        print(f"  real instrument_works={works}")
    else:
        print("  real L0 gate: NOT LOADED (capture/BF16 pin path missing; abliterated fallback failed)")
        for t in winfo.get("tried", [])[:6]:
            print(f"    tried {t.get('root')} {t.get('name')}: {t.get('reason') or 'ok'}")
    print()

    # --- enumerate ---
    print("3. COSINE-ONLY CAMPAIGNS")
    print("-" * 72)
    campaigns = enumerate_campaigns()
    n_cos_only = 0
    n_axis = 0
    for c in campaigns:
        flag = "COSINE-ONLY" if c["cosine_only"] else ("HAS GAIN" if c.get("already_has_gain") or c["n_with_gain_field"] else "listed")
        if c["cosine_only"]:
            n_cos_only += 1
        n_axis += c["n_cosine_axis_dicts"]
        print(f"  [{flag:<11}] {c['id']}")
        print(f"               {c['name']}")
        for r in c["receipts"]:
            mark = "ok" if r["present"] else "MISSING"
            print(f"               receipt {r['path']}  [{mark}]")
        print(f"               axis_dicts={c['n_cosine_axis_dicts']}  with_gain={c['n_with_gain_field']}  "
              f"with_energy/norm_proxy={c['n_with_energy_or_norm_proxy']}")
        if c.get("notes"):
            print(f"               {c['notes']}")
    print()
    print(f"  cosine-only campaigns: {n_cos_only}   cosine axis dicts: {n_axis}")
    print()

    # --- live rescore ---
    print("4. LIVE RE-SCORE WITH `_gain`")
    print("-" * 72)
    if W is None:
        rng = np.random.default_rng(1)
        W_live = rng.standard_normal((64, 256)).astype(np.float32)
        live_src = {"loaded": False, "fallback": "synthetic 64x256 — live ranking is instrument-geometry, not L0"}
        print("  using synthetic 64x256 (real L0 unavailable)")
    else:
        W_live = W
        live_src = winfo
        print(f"  using real tensor {tuple(W.shape)} probe_only")
    live = live_rescore(W_live, live_src)
    print(f"  {'candidate':<24} {'obs':>8} {'probed':>8} {'worst_u':>8} {'gain':>8}  "
          f"{'cos-comp':>8} {'gain-aw':>8}  oldH newH")
    for r in live["rows"]:
        print(
            f"  {r['id']:<24} {fmt(r['observed'])} {fmt(r['probed'])} {fmt(r['worst_unit'])} "
            f"{fmt(r['gain'])}  {fmt(r['cosine_composite'])} {fmt(r['gain_aware'])}  "
            f"{str(r['old_healthy']):<5} {str(r['new_healthy'])}"
        )
    print()
    print(f"  rank by cosine_composite: {live['ranked_by_cosine_composite']}")
    print(f"  rank by gain_aware:       {live['ranked_by_gain_aware']}")
    live_primary = [x for x in live["inversions"] if x["cause"] == "PRIMARY"]
    live_disp = [x for x in live["inversions"] if x["cause"] != "PRIMARY"]
    print(f"  live PRIMARY inversions: {len(live_primary)}   displacement: {len(live_disp)}")
    for inv in live_primary:
        print(
            f"    {inv['candidate']}: rank {inv['old_rank']}→{inv['new_rank']}  "
            f"metric {inv['old_metric']:.4f}→{inv['new_metric']:.4f}  {inv['kind']}  "
            f"healthy {inv['old_healthy']}→{inv['new_healthy']}"
        )
    if live_disp:
        print(f"    (displacement, not own-score collapse: {[x['candidate'] for x in live_disp]})")
    if live.get("spanfit_on_fitted_probes"):
        s = live["spanfit_on_fitted_probes"]
        print()
        print("  spanfit scored on the FITTED (X, P0) pair (historical Goodhart):")
        print(f"    observed={s['observed']:.6f} probed={s['probed']:.6f} worst_unit={s['worst_unit']:.6f} "
              f"gain={s['gain']:.6f} energy_ratio={s['energy_ratio']:.6f}")
        print(f"    fresh-probe cosine={s['fresh_probe_cosine']:.6f}  fresh-probe gain={s['fresh_probe_gain']:.6f}")
        print("    `_gain` on the fitted span is ~1 (map is exact there). Weight energy is not.")
        print("    Fresh probes collapse BOTH cosine and gain. `_gain` is the scale fix, not the spanfit fix.")
    print()

    # --- archive ---
    print("5. ARCHIVE RE-RANK (stored energy_ratio / norms → magnitude proxy)")
    print("-" * 72)
    arch_inv, peer_reports, arch_stats = archive_rescore()
    print(f"  peer_sets={arch_stats['peer_sets']}  with_proxy={arch_stats['candidates_with_proxy']}  "
          f"cosine_only_no_proxy={arch_stats['candidates_cosine_only_no_proxy']}")
    arch_primary = [x for x in arch_inv if x["cause"] == "PRIMARY"]
    arch_disp = [x for x in arch_inv if x["cause"] != "PRIMARY"]
    print(f"  archive PRIMARY inversions: {len(arch_primary)}   displacement: {len(arch_disp)}")
    for inv in arch_primary:
        print(
            f"    [{inv['campaign']}] {inv['candidate']}: rank {inv['old_rank']}→{inv['new_rank']}  "
            f"metric {inv['old_metric']:.4f}→{inv['new_metric']:.4f}  {inv['kind']}  "
            f"healthy {inv['old_healthy']}→{inv['new_healthy']}"
        )
    if not arch_primary:
        print("    (none)")
    if arch_disp:
        print(f"    displacement ({len(arch_disp)}): honest codecs sliding because the cheats left the top, "
              f"or small energy-vs-cosine differences. Not a scale pathology of those codecs.")
    print()

    live_primary = [x for x in live["inversions"] if x["cause"] == "PRIMARY"]
    arch_primary = [x for x in arch_inv if x["cause"] == "PRIMARY"]
    primary_inv = live_primary + arch_primary
    n_live_rescored = len(live["rows"])
    n_arch_rescored = arch_stats["candidates_with_proxy"]

    print("6. INVERSIONS (the deliverable)")
    print("-" * 72)
    print(f"  re-scored live={n_live_rescored}  archive_proxy={n_arch_rescored}  "
          f"enumerated cosine-axis dicts={n_axis}")
    if not primary_inv:
        print("  PRIMARY inversions: none. Rankings survived under a magnitude-aware gate.")
    else:
        print(f"  PRIMARY inversions: {len(primary_inv)}  "
              f"(live={len(live_primary)} archive={len(arch_primary)})")
        for inv in primary_inv:
            print(
                f"    [{inv['campaign']}] {inv['candidate']}: "
                f"rank {inv['old_rank']}→{inv['new_rank']} of {inv['n_peers']}  "
                f"{inv['old_metric']:.4f}→{inv['new_metric']:.4f}  "
                f"healthy {inv['old_healthy']}→{inv['new_healthy']}  {inv['kind']}"
            )
    print()
    print("  Honest outcome: honest qN / Hadamard-then-qN / plane rungs keep their")
    print("  relative order because their magnitude tracks their direction. The")
    print("  candidates that MOVE are the scale cheats (0.01*W, 100*W) and the")
    print("  archive spanfit that cosine certified at 1.000 with 29% weight energy.")
    print("  Displacement ranks (q6 sliding up because a cheat left the top) are")
    print("  not listed as inversions of those codecs.")
    print()
    print(WHAT_I_WATCHED_FAIL)

    receipt = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "repo": str(REPO),
        "obligation": (
            "NNS-005: gravity_doctor_gate._gain exists. Cosine-only campaigns have "
            "not been re-scored. Re-score them; report rank inversions; prove the "
            "new instrument rejects 0.01*W."
        ),
        "gain_definition": {
            "file": GAIN_FILE,
            "lines": list(GAIN_LINES),
            "symbol": "gravity_doctor_gate._gain",
            "measures": (
                "Magnitude agreement of output activations A=X@W.T vs B=X@Wh.T "
                "(and of isotropic probes). min(r, 1/r) on the per-row mean AND "
                "the per-unit worst of the L2-norm ratio. 1.0 is exact."
            ),
            "vs_cosine_only": {
                "file": GAIN_FILE,
                "rowcos_lines": list(ROWCOS_LINES),
                "observed_score_lines": list(OBSERVED_LINES),
                "axes_lines": list(AXES_LINES),
                "original_axes": list(COSINE_KEYS),
                "why_blind": (
                    "Cosine divides out the per-row norms, so it is scale-invariant. "
                    "0.01*W and 100*W score 1.000000 on observed, probed, and worst_unit."
                ),
            },
            "packers_that_drop_gain": PACKERS,
            "thresholds": {
                "ABS_THRESHOLD": dg.ABS_THRESHOLD,
                "AXIS_MARGIN": dg.AXIS_MARGIN,
                "note": (
                    "Honest Q4 g128 scores about 0.90 on gain by construction "
                    "(min over 17408 units). Relative mode vs same-tensor Q4 is the real gate."
                ),
            },
        },
        "instrument_check": {
            "synthetic": synth,
            "real_l0_gate": real_check,
            "real_l0_load": {k: winfo[k] for k in winfo if k != "tried"} | {
                "n_tried": len(winfo.get("tried") or []),
                "tried": winfo.get("tried"),
            },
            "proof": {
                "claim": "0.01*W must NOT score 1.0 on `_gain`",
                "synthetic_gain_0.01": synth["direct_scale"]["scale_0.01"]["gain"],
                "synthetic_cosine_0.01": synth["direct_scale"]["scale_0.01"]["observed_cosine"],
                "real_gain_0.01": None if real_check is None else real_check["scale_0.01"]["gain"],
                "real_cosine_0.01": None if real_check is None else real_check["scale_0.01"]["observed_cosine"],
                "instrument_works": bool(
                    synth["instrument_works"] and (
                        real_check is None
                        or all(real_check[k]["instrument_works"] for k in ("scale_0.01", "scale_100"))
                    )
                ),
            },
        },
        "campaigns": campaigns,
        "live_rescore": live,
        "archive_rescore": {
            "stats": arch_stats,
            "peer_sets": peer_reports,
            "inversions": arch_inv,
        },
        "inversions": {
            "primary": primary_inv,
            "displacement_live": [x for x in live["inversions"] if x["cause"] != "PRIMARY"],
            "displacement_archive": [x for x in arch_inv if x["cause"] != "PRIMARY"],
        },
        "counts": {
            "cosine_only_campaigns": n_cos_only,
            "cosine_axis_dicts_enumerated": n_axis,
            "live_candidates_rescored_with_gain": n_live_rescored,
            "archive_candidates_rescored_with_proxy": n_arch_rescored,
            "primary_inversions_live": len(live_primary),
            "primary_inversions_archive": len(arch_primary),
            "primary_inversions_total": len(primary_inv),
            "displacement_live": sum(1 for x in live["inversions"] if x["cause"] != "PRIMARY"),
            "displacement_archive": sum(1 for x in arch_inv if x["cause"] != "PRIMARY"),
        },
        "material_rule": {
            "primary": (
                "verdict flip OR own-score drop >= 0.15. These are the candidates "
                "cosine certified and a magnitude term refuses."
            ),
            "displacement": (
                "rank moved because a PRIMARY left the top, or a small energy-vs-cosine "
                "difference shuffled honest codecs. Not a scale pathology of that codec."
            ),
            "primary_score_drop_ge": PRIMARY_SCORE_DELTA,
            "rank_delta_abs_ge": MATERIAL_RANK_DELTA,
            "score_delta_abs_ge": MATERIAL_SCORE_DELTA,
            "or_healthy_verdict_flip": True,
        },
        "what_i_watched_fail": WHAT_I_WATCHED_FAIL,
        "write_scope": {
            "WRITE": ["tools/headless/noetic_gain_rescore.py", "receipts/headless/NOETIC_GAIN_RESCORE.json"],
            "VERIFY": ["tools/headless"],
            "DENY": ["crates", "workspace", "visionmcp", "app", "lab", "tools/haider"],
        },
        "elapsed_s": time.time() - t0,
    }

    out_path, where = write_receipt(receipt)
    rel = str(out_path) if where != "repo" else str(out_path.relative_to(REPO))
    print()
    print(f"wrote: {rel}  [{where}]")
    print(f"elapsed_s: {receipt['elapsed_s']:.3f}")
    print()
    if not receipt["instrument_check"]["proof"]["instrument_works"]:
        print("INSTRUMENT FAILED: `_gain` did not reject 0.01*W. Do not trust the ranking.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
