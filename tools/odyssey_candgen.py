#!/usr/bin/env python3
"""Odyssey candidate search as data — deterministic generator / pruner.

Encodes the aggressive representation search space in
``workspace/campaign/odyssey/candidate_families.json`` and expands it into
ranked runner ``--gravity`` spec strings. No model calls.

    python3 tools/odyssey_candgen.py --self-check
    python3 tools/odyssey_candgen.py generate --class moe \\
        --census workspace/campaign/odyssey/patients/O005/census.json

``generate(patient_class, census, sensitivity, policy)`` expands the family
grids, applies native / disk / mem / source-pass constraints, and ranks by
expected info-gain/cost (families ordered by ``prior``).

``prune(candidates, results)`` drops strictly Pareto-dominated points on
complete_bpw vs Doctor delta vs active-bytes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
ODYSSEY = REPO / "workspace" / "campaign" / "odyssey"
FAMILIES_PATH = ODYSSEY / "candidate_families.json"
POLICY_PATH = ODYSSEY / "ODYSSEY_POLICY.json"
PATIENTS_DIR = ODYSSEY / "patients"

# GiB. Bible §33 machine class; also the ctl.py disk-floor companion.
_GIB = 1024 ** 3
_DEFAULT_MEM_BYTES = int(0.85 * 96 * _GIB)

# Suffix emission order is canonical; the parser accepts any order.
_SUFFIX_RE = re.compile(
    r"\+(correction|c(\d+(?:\.\d+)?)|r(\d+)|meta-(raw|shared|entropy))"
)
_CLAUSE_AND = re.compile(r"\s+AND\s+", re.I)

# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _as_mapping(obj: Any, default_path: Path) -> dict:
    if obj is None:
        return _load_json(default_path) if default_path.is_file() else {}
    if isinstance(obj, (str, Path)):
        return _load_json(Path(obj))
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"expected dict/path/None, got {type(obj).__name__}")


def load_families(path: Path | None = None) -> dict:
    return _load_json(path or FAMILIES_PATH)


def load_policy(path: Path | None = None) -> dict:
    return _as_mapping(path, POLICY_PATH)


# ---------------------------------------------------------------------------
# Patient class / organ helpers
# ---------------------------------------------------------------------------

_CLASS_ALIASES = {
    "moe": "moe",
    "dense": "dense",
    "hybrid": "hybrid",
}


def normalize_patient_class(patient_class: str, census: dict | None = None) -> str:
    raw = str(patient_class).strip()
    key = raw.lower()
    if key in _CLASS_ALIASES:
        return _CLASS_ALIASES[key]
    if "hybrid" in key or "mamba" in key or "ssm" in key:
        return "hybrid"
    if "moe" in key:
        return "moe"
    if "dense" in key:
        return "dense"
    census = census or {}
    if census.get("is_moe"):
        return "moe"
    organs = census.get("organs_params") or {}
    if (organs.get("other") or 0) > 0 and not organs.get("expert"):
        return "hybrid"
    if (organs.get("mlp_dense") or 0) > 0:
        return "dense"
    raise ValueError(f"unrecognized patient_class {patient_class!r}")


def _organs(census: dict) -> dict:
    return dict(census.get("organs_params") or census.get("organs_bytes") or {})


def _cfg(census: dict) -> dict:
    return dict(census.get("config") or {})


def _target_for(families: dict, patient_class: str) -> str | None:
    grammar = families.get("spec_grammar") or {}
    mapping = grammar.get("target_by_class") or {
        "moe": "experts",
        "hybrid": "attn-mlp",
        "dense": None,
    }
    t = mapping.get(patient_class)
    return t if t else None


# ---------------------------------------------------------------------------
# Spec grammar
# ---------------------------------------------------------------------------


def spec_pattern(families: dict | None = None) -> re.Pattern:
    if families:
        pat = (families.get("spec_grammar") or {}).get("pattern")
        if pat:
            return re.compile(pat)
    return re.compile(
        r"^(?:q\d+-g\d+(?:-(?:experts|attn-mlp))?|"
        r"mixed-q\d+q\d+(?:-(?:experts|attn-mlp))?|"
        r"tiers-(?:t\d+)+(?:-(?:experts|attn-mlp))?|"
        r"scale-joint-q\d+-g\d+(?:-(?:experts|attn-mlp))?)"
        r"(?:\+(?:correction|c\d+(?:\.\d+)?|r\d+|meta-(?:raw|shared|entropy)))*$"
    )


def spec_valid(spec: str, families: dict | None = None) -> bool:
    if not isinstance(spec, str) or not spec:
        return False
    return spec_pattern(families).match(spec) is not None


def _fmt_budget(x: float) -> str:
    x = float(x)
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s


def _parse_budget(s: str) -> float:
    return float(s)


def parse_spec(spec: str) -> dict:
    """Parse a runner --gravity spec. Raises ValueError on invalid strings."""
    if not spec_valid(spec):
        raise ValueError(f"invalid gravity spec {spec!r}")
    base, *rest = spec.split("+", 1)
    suffixes = "+" + rest[0] if rest else ""

    out: dict[str, Any] = {
        "spec": spec,
        "form": None,
        "bits": None,
        "group": None,
        "target": None,
        "mixed_lo": None,
        "mixed_hi": None,
        "tiers": None,
        "correction_budget": 0.0,
        "correction_token": False,
        "router_precision": None,
        "metadata_codec": "raw",
    }

    if base.startswith("scale-joint-"):
        m = re.fullmatch(
            r"scale-joint-q(\d+)-g(\d+)(?:-(experts|attn-mlp))?", base
        )
        if not m:
            raise ValueError(f"invalid scale-joint spec {spec!r}")
        out["form"] = "scale_joint"
        out["bits"] = int(m.group(1))
        out["group"] = int(m.group(2))
        out["target"] = m.group(3)
    elif base.startswith("mixed-"):
        m = re.fullmatch(r"mixed-q(\d+)q(\d+)(?:-(experts|attn-mlp))?", base)
        if not m:
            raise ValueError(f"invalid mixed spec {spec!r}")
        out["form"] = "mixed"
        out["mixed_lo"] = int(m.group(1))
        out["mixed_hi"] = int(m.group(2))
        out["bits"] = out["mixed_lo"]
        out["target"] = m.group(3)
    elif base.startswith("tiers-"):
        m = re.fullmatch(r"tiers-((?:t\d+)+)(?:-(experts|attn-mlp))?", base)
        if not m:
            raise ValueError(f"invalid tiers spec {spec!r}")
        out["form"] = "tiers"
        out["tiers"] = re.findall(r"t\d+", m.group(1))
        out["target"] = m.group(2)
    else:
        m = re.fullmatch(r"q(\d+)-g(\d+)(?:-(experts|attn-mlp))?", base)
        if not m:
            raise ValueError(f"invalid uniform spec {spec!r}")
        out["form"] = "uniform"
        out["bits"] = int(m.group(1))
        out["group"] = int(m.group(2))
        out["target"] = m.group(3)

    for sm in _SUFFIX_RE.finditer(suffixes):
        tok = sm.group(1)
        if tok == "correction":
            out["correction_token"] = True
            if out["correction_budget"] == 0.0:
                out["correction_budget"] = 0.02
        elif tok.startswith("c") and sm.group(2) is not None:
            out["correction_budget"] = _parse_budget(sm.group(2))
        elif tok.startswith("r") and sm.group(3) is not None:
            out["router_precision"] = int(sm.group(3))
        elif tok.startswith("meta-"):
            out["metadata_codec"] = sm.group(4) or tok.split("-", 1)[1]
    return out


def format_spec(
    *,
    form: str = "uniform",
    bits: int | None = None,
    group: int | None = None,
    target: str | None = None,
    mixed_lo: int | None = None,
    mixed_hi: int | None = None,
    tiers: list[str] | None = None,
    correction_budget: float = 0.0,
    correction_token: bool = False,
    router_precision: int | None = None,
    metadata_codec: str = "raw",
    default_router: int = 4,
) -> str:
    """Canonical spec string (suffix order: correction, router, meta)."""
    tgt = f"-{target}" if target else ""
    if form == "scale_joint":
        core = f"scale-joint-q{int(bits)}-g{int(group)}{tgt}"
    elif form == "mixed":
        core = f"mixed-q{int(mixed_lo)}q{int(mixed_hi)}{tgt}"
    elif form == "tiers":
        core = "tiers-" + "".join(tiers or []) + tgt
    else:
        core = f"q{int(bits)}-g{int(group)}{tgt}"

    parts: list[str] = []
    if correction_token:
        parts.append("correction")
    elif float(correction_budget or 0) > 0:
        parts.append("c" + _fmt_budget(correction_budget))
    if router_precision is not None and int(router_precision) != int(default_router):
        parts.append(f"r{int(router_precision)}")
    if metadata_codec and metadata_codec != "raw":
        parts.append(f"meta-{metadata_codec}")
    if not parts:
        return core
    return core + "".join("+" + p for p in parts)


# ---------------------------------------------------------------------------
# Applicability predicate (restricted; no eval)
# ---------------------------------------------------------------------------


def _lookup_num(path: str, census: dict) -> float:
    organs = _organs(census)
    if path.startswith("organs."):
        return float(organs.get(path.split(".", 1)[1], 0) or 0)
    if path.startswith("census.organs_params."):
        return float((_organs(census)).get(path.rsplit(".", 1)[1], 0) or 0)
    if path == "census.total_params" or path == "total_params":
        return float(census.get("total_params") or 0)
    raise ValueError(f"unknown numeric path {path!r}")


def _eval_clause(
    clause: str, patient_class: str, census: dict, sensitivity: Any
) -> bool:
    c = clause.strip()
    if not c or c.lower() == "true":
        return True
    m = re.fullmatch(r"patient_class\s*==\s*(\w+)", c, re.I)
    if m:
        return patient_class == m.group(1).lower()
    m = re.fullmatch(r"patient_class\s+in\s+([,\w]+)", c, re.I)
    if m:
        allowed = [x.strip().lower() for x in m.group(1).split(",") if x.strip()]
        return patient_class in allowed
    m = re.fullmatch(r"(organs\.\w+|census\.organs_params\.\w+)\s*>\s*([0-9.]+)", c)
    if m:
        return _lookup_num(m.group(1), census) > float(m.group(2))
    m = re.fullmatch(r"census\.is_moe\s*==\s*(true|false)", c, re.I)
    if m:
        want = m.group(1).lower() == "true"
        return bool(census.get("is_moe")) == want
    m = re.fullmatch(r"sensitivity\s*!=\s*null", c, re.I)
    if m:
        return bool(sensitivity)
    raise ValueError(f"unsupported applicability clause {clause!r}")


def eval_predicate(
    pred: str, patient_class: str, census: dict, sensitivity: Any
) -> bool:
    parts = _CLAUSE_AND.split(pred.strip()) if pred else ["true"]
    return all(_eval_clause(p, patient_class, census, sensitivity) for p in parts)


def family_applies(
    family: dict, patient_class: str, census: dict, sensitivity: Any
) -> bool:
    allowed = family.get("patient_classes")
    if allowed and patient_class not in allowed:
        return False
    pred = family.get("applicability")
    if isinstance(pred, dict):
        extra = pred.get("patient_classes")
        if extra and patient_class not in extra:
            return False
        pred = pred.get("predicate") or "true"
    if not pred:
        return True
    organs = _organs(census)
    # MoE without an organ census still applies if the caller declared moe.
    if (
        "organs.expert" in str(pred)
        and patient_class == "moe"
        and not organs
    ):
        census = {**census, "organs_params": {"expert": 1}}
    return eval_predicate(str(pred), patient_class, census, sensitivity)


# ---------------------------------------------------------------------------
# Native / complete-bpw estimates
# ---------------------------------------------------------------------------


def _nav_flag(table: dict, key: Any, default: bool = False) -> bool:
    if not table:
        return default
    if key in table:
        return bool(table[key])
    s = str(key)
    if s in table:
        return bool(table[s])
    if isinstance(key, float):
        s2 = _fmt_budget(key)
        if s2 in table:
            return bool(table[s2])
    return default


def is_native(
    families: dict,
    *,
    form: str,
    bits: int | None,
    group: int | None,
    metadata_codec: str = "raw",
    correction_budget: float = 0.0,
    router_precision: int | None = None,
    correction_token: bool = False,
) -> bool:
    nav = families.get("native_availability") or {}
    forms = nav.get("forms") or {}
    if form == "tiers" and not forms.get("tiers", False):
        return False
    if form == "scale_joint" and not forms.get("scale_joint", False):
        return False
    if correction_token and not forms.get("correction_token", False):
        return False
    if bits is not None and not _nav_flag(nav.get("bit_classes") or {}, int(bits), False):
        return False
    if group is not None and not _nav_flag(
        nav.get("group_sizes") or {}, int(group), True
    ):
        return False
    if not _nav_flag(nav.get("metadata_codec") or {}, metadata_codec or "raw", True):
        return False
    if float(correction_budget or 0) > 0 and not _nav_flag(
        nav.get("correction_budget") or {}, float(correction_budget), False
    ):
        return False
    if router_precision is not None and not _nav_flag(
        nav.get("router_precision") or {}, int(router_precision), True
    ):
        return False
    return True


def affine_complete_bpw(bits: float, group: int | None, metadata_codec: str = "raw") -> float:
    """Payload bits + f16 scale + f16 zp per group, adjusted for metadata codec.

    mlx affine grouped q3-g32 lands near 4.0 complete_bpw (O005 MEASURED 4.0253).
    """
    bits = float(bits)
    g = int(group or 64)
    sidecar = (16.0 / g) * 2.0  # scale + zero-point
    if metadata_codec == "shared":
        sidecar *= 0.35
    elif metadata_codec == "entropy":
        sidecar *= 0.70
    return bits + sidecar


def estimate_complete_bpw(parsed: dict) -> float:
    form = parsed["form"]
    meta = parsed.get("metadata_codec") or "raw"
    corr = float(parsed.get("correction_budget") or 0)
    if form == "mixed":
        lo = int(parsed["mixed_lo"])
        hi = int(parsed["mixed_hi"])
        # Most mass at lo; a small sensitive subset promoted to hi.
        payload = 0.85 * lo + 0.15 * hi
        group = parsed.get("group") or 64
        bpw = affine_complete_bpw(payload, group, meta)
    elif form == "tiers":
        tiers = parsed.get("tiers") or ["t0"]
        # T0 stored at 1 bit, each extra tier adds a 0.5-bit residual plane (counted).
        bpw = affine_complete_bpw(1.0, 32, meta) + 0.5 * max(0, len(tiers) - 1)
    elif form == "scale_joint":
        # Joint r-search can shave sidecar entropy; still count the affine container.
        bpw = affine_complete_bpw(int(parsed["bits"]), parsed.get("group"), meta)
        bpw *= 0.97
    else:
        bpw = affine_complete_bpw(
            int(parsed["bits"] or 4), parsed.get("group"), meta
        )
    bpw += corr * 16.0  # correction_budget is a fraction of weights at f16
    return bpw


def estimate_stored_bytes(census: dict, complete_bpw: float) -> int:
    params = int(census.get("total_params") or 0)
    if params <= 0:
        return 0
    return int(params * float(complete_bpw) / 8.0)


def estimate_active_bytes(
    census: dict, patient_class: str, complete_bpw: float
) -> int:
    src_bpw = float(census.get("stored_bpw") or 16.0) or 16.0
    scale = float(complete_bpw) / src_bpw
    if patient_class == "moe" and census.get("active_bytes_per_token"):
        return int(float(census["active_bytes_per_token"]) * scale)
    stored = estimate_stored_bytes(census, complete_bpw)
    return stored


def _doctor_risk_for(families: dict, family: dict, parsed: dict) -> float:
    table = families.get("doctor_risk_by_bits") or {}
    bits = parsed.get("bits")
    if parsed["form"] == "mixed":
        bits = parsed.get("mixed_lo")
    if parsed["form"] == "tiers":
        bits = 1
    if bits is None:
        base = float(family.get("doctor_risk") or 0.3)
    else:
        base = float(table.get(str(int(bits)), family.get("doctor_risk") or 0.3))
    corr = float(parsed.get("correction_budget") or 0)
    if corr > 0:
        base *= max(0.15, 1.0 - 4.0 * corr)
    if parsed["form"] == "mixed":
        base *= 0.72
    if parsed.get("router_precision") and int(parsed["router_precision"]) >= 8:
        base *= 0.9
    return max(0.01, min(0.95, base))


def _protected_organs(sensitivity: Any) -> list[str]:
    """Rank organs by |zero-ablation delta_hits| (more negative = more sensitive)."""
    if not isinstance(sensitivity, dict):
        return []
    block = sensitivity.get("per_organ_sensitivity") or sensitivity
    scored: list[tuple[int, str]] = []
    for name, rec in block.items():
        if name.startswith("_") or name in {"baseline", "treatments"}:
            continue
        if not isinstance(rec, dict):
            continue
        zero = rec.get("zero") if isinstance(rec.get("zero"), dict) else rec
        delta = zero.get("delta_hits") if isinstance(zero, dict) else None
        if delta is None:
            continue
        scored.append((int(delta), str(name)))
    scored.sort(key=lambda t: (t[0], t[1]))  # most-negative first, then name
    return [n for _, n in scored]


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------


def _candidate(
    *,
    spec: str,
    family: dict,
    families: dict,
    parsed: dict,
    patient_class: str,
    census: dict,
    sensitivity: Any,
    native: bool,
    extra_dims: dict | None = None,
) -> dict:
    bpw = estimate_complete_bpw(parsed)
    stored = estimate_stored_bytes(census, bpw)
    active = estimate_active_bytes(census, patient_class, bpw)
    conv = family["conventionality"]
    # Per-candidate class: q1/q2 or non-raw/correction/tiers/joint escalate.
    bits = parsed.get("bits")
    if parsed["form"] == "mixed":
        bits = parsed.get("mixed_lo")
    if conv == "CONVENTIONAL" and (
        (bits is not None and int(bits) <= 2)
        or float(parsed.get("correction_budget") or 0) > 0
        or parsed["form"] in {"tiers", "scale_joint"}
        or (parsed.get("metadata_codec") not in {None, "raw"})
    ):
        cand_conv = "AGGRESSIVE"
    else:
        cand_conv = conv
    class_map = {
        "CONVENTIONAL": "CONVENTIONAL_ANCHOR",
        "AGGRESSIVE": "AGGRESSIVE_QUANT",
        "STRUCTURAL": "STRUCTURAL_GRAVITY",
    }
    dims = {
        "form": parsed["form"],
        "bits": parsed.get("bits"),
        "group": parsed.get("group"),
        "target": parsed.get("target"),
        "mixed_lo": parsed.get("mixed_lo"),
        "mixed_hi": parsed.get("mixed_hi"),
        "tiers": parsed.get("tiers"),
        "correction_budget": parsed.get("correction_budget") or 0.0,
        "router_precision": parsed.get("router_precision"),
        "metadata_codec": parsed.get("metadata_codec") or "raw",
    }
    if extra_dims:
        dims.update(extra_dims)
    source_passes = float(family.get("source_passes") or 1.0)
    if (
        (dims["metadata_codec"] != "raw" or float(dims["correction_budget"]) > 0)
        and family.get("source_passes_suffix")
    ):
        source_passes += float(family["source_passes_suffix"])
    if parsed["form"] == "tiers" and parsed.get("tiers"):
        source_passes = max(source_passes, float(len(parsed["tiers"])))
    return {
        "spec": spec,
        "family": family["id"],
        "mechanism": family["mechanism"],
        "conventionality": cand_conv,
        "family_conventionality": conv,
        "candidate_class": class_map[cand_conv],
        "cheapest_falsifier": family["cheapest_falsifier"],
        "expected_win": family["expected_win"],
        "doctor_risk": round(_doctor_risk_for(families, family, parsed), 4),
        "native": bool(native),
        "estimated_complete_bpw": round(bpw, 4),
        "estimated_stored_bytes": stored,
        "estimated_active_bytes": active,
        "source_passes": round(source_passes, 4),
        "prior": int(family["prior"]),
        "info_gain_prior": int(family.get("info_gain_prior") or 0),
        "applicability": family.get("applicability"),
        "dimensions": dims,
        "patient_class": patient_class,
    }


def _emit(
    families: dict,
    family: dict,
    patient_class: str,
    census: dict,
    sensitivity: Any,
    **fmt,
) -> dict:
    grammar = families.get("spec_grammar") or {}
    default_router = int(grammar.get("default_router_precision") or 4)
    spec = format_spec(default_router=default_router, **fmt)
    parsed = parse_spec(spec)
    native = is_native(
        families,
        form=parsed["form"],
        bits=parsed.get("bits") if parsed["form"] != "mixed" else parsed.get("mixed_lo"),
        group=parsed.get("group"),
        metadata_codec=parsed.get("metadata_codec") or "raw",
        correction_budget=float(parsed.get("correction_budget") or 0),
        router_precision=parsed.get("router_precision"),
        correction_token=bool(parsed.get("correction_token")),
    )
    extra = {}
    if family["id"] == "sensitivity_driven_alloc":
        extra["protected_organs"] = _protected_organs(sensitivity)
    return _candidate(
        spec=spec,
        family=family,
        families=families,
        parsed=parsed,
        patient_class=patient_class,
        census=census,
        sensitivity=sensitivity,
        native=native,
        extra_dims=extra,
    )


def expand_family(
    family: dict,
    families: dict,
    patient_class: str,
    census: dict,
    sensitivity: Any,
) -> list[dict]:
    target = _target_for(families, patient_class)
    kind = family["spec_kind"]
    dims = family.get("dimensions") or {}
    out: list[dict] = []

    def add(**fmt):
        out.append(
            _emit(families, family, patient_class, census, sensitivity, **fmt)
        )

    if kind == "quant_grid":
        bits_l = list(dims.get("bit_classes") or [])
        groups = list(dims.get("group_sizes") or [])
        routers = list(dims.get("router_precision") or [None])
        if patient_class != "moe":
            routers = [None]
        corrs = list(dims.get("correction_budget") or [0])
        metas = list(dims.get("metadata_codec") or ["raw"])
        use_token = bool(family.get("correction_token"))
        token_budget = float(
            (families.get("spec_grammar") or {}).get("correction_token_budget") or 0.02
        )
        for bits, group, router, corr, meta in product(
            bits_l, groups, routers, corrs, metas
        ):
            corr_f = float(corr or 0)
            token = bool(use_token and abs(corr_f - token_budget) < 1e-9)
            add(
                form="uniform",
                bits=int(bits),
                group=int(group),
                target=target,
                correction_budget=0.0 if token else corr_f,
                correction_token=token,
                router_precision=router,
                metadata_codec=meta,
            )
    elif kind == "mixed_pair":
        pairs = list(dims.get("bit_pairs") or [])
        corrs = list(dims.get("correction_budget") or [0])
        for pair, corr in product(pairs, corrs):
            lo, hi = int(pair[0]), int(pair[1])
            if lo >= hi:
                continue
            add(
                form="mixed",
                mixed_lo=lo,
                mixed_hi=hi,
                target=target,
                correction_budget=float(corr or 0),
            )
    elif kind == "scale_joint":
        for bits, group in product(
            list(dims.get("bit_classes") or []), list(dims.get("group_sizes") or [])
        ):
            add(
                form="scale_joint",
                bits=int(bits),
                group=int(group),
                target=target,
            )
    elif kind == "tiers":
        for tset in dims.get("tier_sets") or []:
            tiers = re.findall(r"t\d+", str(tset))
            if not tiers:
                continue
            add(form="tiers", tiers=tiers, target=target)
    else:
        raise ValueError(f"unknown spec_kind {kind!r} on family {family.get('id')}")

    pinned = (family.get("pinned_specs") or {}).get(patient_class) or []
    have = {c["spec"] for c in out}
    for spec in pinned:
        if spec in have:
            continue
        if not spec_valid(spec, families):
            raise ValueError(f"pinned spec {spec!r} fails grammar")
        parsed = parse_spec(spec)
        native = is_native(
            families,
            form=parsed["form"],
            bits=parsed.get("bits") if parsed["form"] != "mixed" else parsed.get("mixed_lo"),
            group=parsed.get("group"),
            metadata_codec=parsed.get("metadata_codec") or "raw",
            correction_budget=float(parsed.get("correction_budget") or 0),
            router_precision=parsed.get("router_precision"),
            correction_token=bool(parsed.get("correction_token")),
        )
        out.append(
            _candidate(
                spec=spec,
                family=family,
                families=families,
                parsed=parsed,
                patient_class=patient_class,
                census=census,
                sensitivity=sensitivity,
                native=native,
            )
        )
        have.add(spec)
    return out


# ---------------------------------------------------------------------------
# Constraints + ranking
# ---------------------------------------------------------------------------


def _constraints(policy: dict, families: dict) -> dict:
    defaults = dict(families.get("constraint_defaults") or {})
    c = {
        "mem_budget_bytes": policy.get("mem_budget_bytes", defaults.get("mem_budget_bytes")),
        "disk_budget_bytes": policy.get("disk_budget_bytes", defaults.get("disk_budget_bytes")),
        "source_pass_budget": policy.get(
            "source_pass_budget", defaults.get("source_pass_budget", 512)
        ),
        "require_native": bool(
            policy.get("require_native", defaults.get("require_native", False))
        ),
        "max_candidates": int(
            policy.get("max_candidates", defaults.get("max_candidates", 4096))
        ),
    }
    if c["mem_budget_bytes"] is None:
        c["mem_budget_bytes"] = _DEFAULT_MEM_BYTES
    return c


def _pressure_zones(policy: dict) -> dict:
    z = policy.get("target_pressure_zones_bpw") or {}
    return {
        "reachable": float(z.get("reachable_or_explained") or 3.0),
        "pressure": float(z.get("pressure") or 2.5),
        "aggressive": float(z.get("aggressive") or 2.0),
        "structural": float(z.get("structural_correction_tier") or 1.5),
    }


def _tried_specs(census: dict, sensitivity: Any, policy: dict) -> set[str]:
    out: set[str] = set()
    for src in (census, sensitivity if isinstance(sensitivity, dict) else {}, policy):
        if not isinstance(src, dict):
            continue
        for key in ("tried_mechanisms", "tried_specs", "tried"):
            v = src.get(key)
            if isinstance(v, list):
                out.update(str(x) for x in v)
            elif isinstance(v, dict):
                out.update(str(x) for x in v.keys())
    g = census.get("gravity") if isinstance(census.get("gravity"), dict) else None
    if g and isinstance(g.get("tried_mechanisms"), list):
        out.update(str(x) for x in g["tried_mechanisms"])
    return out


def _score(
    cand: dict,
    *,
    policy: dict,
    patient_class: str,
    tried: set[str],
) -> tuple[int, int]:
    """Return (info_milli, cost_milli) — both positive ints.

    Cost is source-pass + non-native kernel work. Stored/active bytes are the
    *win* axes (Pareto in prune), not a reason to try q2 before a q3 anchor.
    """
    info = int(cand.get("info_gain_prior") or 0) * 10
    zones = _pressure_zones(policy)
    bpw = float(cand["estimated_complete_bpw"])
    fam_conv = cand.get("family_conventionality") or cand["conventionality"]
    if fam_conv == "CONVENTIONAL":
        target = zones["reachable"]
        info += int((1.0 - float(cand["doctor_risk"])) * 120)
    elif fam_conv == "STRUCTURAL":
        target = zones["structural"]
        info += int((1.0 - float(cand["doctor_risk"])) * 40)
    else:
        target = zones["aggressive"]
        info += int((1.0 - float(cand["doctor_risk"])) * 30)
    dist = abs(bpw - target)
    info += max(0, 80 - int(dist * 40))
    if cand["spec"] not in tried:
        info += 25
    else:
        info = max(1, info // 4)
    if cand["native"]:
        info += 10
    # MoE objective: lower estimated active bytes is information, not cost.
    if patient_class == "moe" and cand["estimated_active_bytes"]:
        # tiny deterministic tie-break: prefer denser active among equals
        info += max(0, 20 - int(cand["estimated_complete_bpw"]))

    cost = int(round(float(cand["source_passes"]) * 1000))
    if not cand["native"]:
        cost += 20_000
    cost = max(cost, 1)
    info = max(info, 1)
    return info, cost


def _passes_resource(cand: dict, constraints: dict) -> bool:
    if constraints["require_native"] and not cand["native"]:
        return False
    stored = int(cand["estimated_stored_bytes"] or 0)
    disk = constraints.get("disk_budget_bytes")
    if disk is not None and stored > 0 and stored > int(disk):
        return False
    mem = constraints.get("mem_budget_bytes")
    if mem is not None and stored > 0 and stored > int(mem):
        return False
    return True


def generate(
    patient_class: str,
    census: Any,
    sensitivity: Any,
    policy: Any,
    *,
    families: Any = None,
) -> list[dict]:
    """Expand + constrain + rank. Deterministic. No model calls.

    Returns a list of candidate_spec dicts (each has a runner-valid ``spec``),
    ordered by family prior then expected info-gain/cost.
    """
    if census is None:
        census_d: dict = {}
    elif isinstance(census, dict):
        census_d = census
    elif isinstance(census, (str, Path)):
        census_d = _load_json(Path(census))
    else:
        raise TypeError("census")

    if sensitivity is None:
        sens: Any = {}
    elif isinstance(sensitivity, (str, Path)):
        raw = _load_json(Path(sensitivity))
        sens = raw.get("representation", {}).get("per_organ_sensitivity") or raw
    elif isinstance(sensitivity, dict):
        if "per_organ_sensitivity" in sensitivity:
            sens = sensitivity["per_organ_sensitivity"]
        elif "representation" in sensitivity and isinstance(
            sensitivity["representation"], dict
        ):
            sens = sensitivity["representation"].get("per_organ_sensitivity") or {}
        else:
            sens = sensitivity
    else:
        raise TypeError("sensitivity")

    policy_d = _as_mapping(policy, POLICY_PATH)
    families_d = families if isinstance(families, dict) else load_families(families)
    klass = normalize_patient_class(patient_class, census_d)
    constraints = _constraints(policy_d, families_d)
    tried = _tried_specs(census_d, sens, policy_d)

    fams = sorted(
        families_d.get("families") or [],
        key=lambda f: (int(f.get("prior") or 0), str(f.get("id") or "")),
    )
    raw: list[dict] = []
    seen: set[str] = set()
    for fam in fams:
        if not family_applies(fam, klass, census_d, sens):
            continue
        for cand in expand_family(fam, families_d, klass, census_d, sens):
            spec = cand["spec"]
            if spec in seen:
                continue
            if not spec_valid(spec, families_d):
                raise ValueError(f"generator emitted invalid spec {spec!r}")
            if not _passes_resource(cand, constraints):
                continue
            seen.add(spec)
            raw.append(cand)

    for cand in raw:
        info, cost = _score(
            cand, policy=policy_d, patient_class=klass, tried=tried
        )
        cand["info_gain"] = info
        cand["cost"] = cost
        cand["score_milli"] = (info * 1000) // cost

    raw.sort(
        key=lambda c: (int(c["prior"]), -int(c["score_milli"]), c["spec"])
    )

    # Source-pass budget: keep rank order, skip once the running pass cost trips.
    budget = float(constraints["source_pass_budget"] or 0)
    kept: list[dict] = []
    spent = 0.0
    for cand in raw:
        need = float(cand["source_passes"])
        if budget > 0 and spent + need > budget and kept:
            continue
        kept.append(cand)
        spent += need
        if len(kept) >= int(constraints["max_candidates"]):
            break

    for i, cand in enumerate(kept, 1):
        cand["rank"] = i
    return kept


# ---------------------------------------------------------------------------
# Pareto prune
# ---------------------------------------------------------------------------

def _spec_of(c: Any) -> str:
    if isinstance(c, str):
        return c
    return str(c["spec"])


def _metrics_of(result: dict) -> dict[str, float] | None:
    """Extract the three Pareto axes. Missing axis => incomparable (None)."""
    bpw = result.get("complete_bpw")
    if bpw is None:
        bpw = result.get("stored_bpw")
    delta = result.get("doctor_delta")
    if delta is None:
        delta = result.get("delta_hits")
    active = result.get("active_bytes")
    if active is None:
        active = result.get("active_bytes_per_token")
    if None in (bpw, delta, active):
        return None
    return {
        "complete_bpw": float(bpw),
        "doctor_delta": float(delta),
        "active_bytes": float(active),
    }


def _index_results(results: Any) -> dict[str, dict]:
    if results is None:
        return {}
    if isinstance(results, dict):
        # Either {spec: metrics} or a single metrics dict with a spec field.
        if "spec" in results and not any(
            isinstance(v, dict) and ("complete_bpw" in v or "stored_bpw" in v)
            for v in results.values()
            if v is not results
        ):
            spec = str(results["spec"])
            return {spec: results}
        out = {}
        for k, v in results.items():
            if isinstance(v, dict):
                out[str(v.get("spec") or k)] = v
        return out
    if isinstance(results, list):
        out = {}
        for item in results:
            if isinstance(item, dict) and "spec" in item:
                out[str(item["spec"])] = item
        return out
    raise TypeError("results")


def _dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """True iff a is <= on costs, >= on doctor_delta, and strict on at least one."""
    better_or_eq = (
        a["complete_bpw"] <= b["complete_bpw"]
        and a["active_bytes"] <= b["active_bytes"]
        and a["doctor_delta"] >= b["doctor_delta"]
    )
    if not better_or_eq:
        return False
    strict = (
        a["complete_bpw"] < b["complete_bpw"]
        or a["active_bytes"] < b["active_bytes"]
        or a["doctor_delta"] > b["doctor_delta"]
    )
    return strict


def prune(candidates: list, results: Any) -> list:
    """Drop strictly Pareto-dominated candidates. Preserve input order.

    Axes (all required to dominate): complete_bpw (min), doctor_delta (max),
    active_bytes (min). Candidates lacking a complete result are kept.
    """
    by_spec = _index_results(results)
    metrics: dict[str, dict[str, float]] = {}
    for c in candidates:
        spec = _spec_of(c)
        rec = by_spec.get(spec)
        if not rec:
            continue
        m = _metrics_of(rec)
        if m:
            metrics[spec] = m

    dominated: set[str] = set()
    specs = [_spec_of(c) for c in candidates]
    for b in specs:
        mb = metrics.get(b)
        if not mb:
            continue
        for a in specs:
            if a == b or a in dominated:
                continue
            ma = metrics.get(a)
            if not ma:
                continue
            if _dominates(ma, mb):
                dominated.add(b)
                break

    survivors = [c for c in candidates if _spec_of(c) not in dominated]
    # Re-rank dict candidates if they carry rank.
    if survivors and isinstance(survivors[0], dict):
        for i, c in enumerate(survivors, 1):
            c = dict(c)
            c["rank"] = i
            c["pruned_dominated"] = False
            survivors[i - 1] = c
    return survivors


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def _synthetic_moe_census() -> dict:
    """Tiny MoE so disk/mem constraints do not fire in the default self-check."""
    experts, k, n = 128, 8, 1_000_000
    expert = 900_000
    return {
        "arch": "SyntheticMoe",
        "is_moe": True,
        "total_params": n,
        "total_bytes": n * 2,
        "stored_bpw": 16.0,
        "active_params_per_token": 100_000 + expert * k // experts,
        "active_bytes_per_token": (100_000 + expert * k // experts) * 2,
        "organs_params": {
            "embed": 20_000,
            "attn": 50_000,
            "router": 5_000,
            "expert": expert,
            "shared_expert": 0,
            "mlp_dense": 0,
            "norm": 1_000,
            "lm_head": 20_000,
            "other": 4_000,
        },
        "config": {
            "num_experts": experts,
            "num_experts_per_tok": k,
            "hidden_size": 256,
        },
    }


def _synthetic_sensitivity() -> dict:
    return {
        "baseline": {"battery": "10/12", "delta_hits": 0},
        "router": {"zero": {"delta_hits": -10}},
        "attn": {"zero": {"delta_hits": -8}},
        "expert": {"zero": {"delta_hits": -4}},
        "embed": {"zero": {"delta_hits": -1}},
        "norm": {"zero": {"delta_hits": 0}},
    }


def self_check() -> int:
    families = load_families()
    assert families.get("schema") == "hawking.odyssey.candidate_families.v1"
    ids = [f["id"] for f in families["families"]]
    required = [
        "per_expert_mixed_quant",
        "sensitivity_driven_alloc",
        "base_plus_correction",
        "matryoshka_tiers",
        "alt_group",
        "scale_codec_joint",
    ]
    missing = [r for r in required if r not in ids]
    assert not missing, missing
    for fam in families["families"]:
        for k in (
            "mechanism",
            "conventionality",
            "cheapest_falsifier",
            "expected_win",
            "doctor_risk",
            "applicability",
        ):
            assert k in fam and fam[k] not in (None, ""), (fam["id"], k)
        assert fam["conventionality"] in {
            "CONVENTIONAL",
            "AGGRESSIVE",
            "STRUCTURAL",
        }, fam["id"]

    policy = load_policy()
    assert policy.get("aggressive_ladder", {}).get("search_space_is_data") is True

    census_path = PATIENTS_DIR / "O005" / "census.json"
    packet_path = PATIENTS_DIR / "O005" / "ODYSSEY_PATIENT_O005.json"
    census = _load_json(census_path) if census_path.is_file() else _synthetic_moe_census()
    if packet_path.is_file():
        pkt = _load_json(packet_path)
        sensitivity = (pkt.get("representation") or {}).get("per_organ_sensitivity")
    else:
        sensitivity = _synthetic_sensitivity()
    if not census.get("organs_params", {}).get("expert"):
        # Keep a fallback so a truncated packet cannot empty the moe grid.
        census = {**_synthetic_moe_census(), **{k: census[k] for k in census}}

    ranked = generate("moe", census, sensitivity, policy)
    assert ranked, "generate() returned empty for moe"
    assert len(ranked) >= 100, f"expected hundreds, got {len(ranked)}"
    specs = [c["spec"] for c in ranked]
    resorted = sorted(
        ranked, key=lambda c: (int(c["prior"]), -int(c["score_milli"]), c["spec"])
    )
    assert [c["spec"] for c in resorted] == specs
    assert all(spec_valid(s, families) for s in specs), [
        s for s in specs if not spec_valid(s, families)
    ]
    assert all(c["rank"] == i for i, c in enumerate(ranked, 1))
    priors = [c["prior"] for c in ranked]
    assert priors == sorted(priors), "family prior order broken"

    must = ["q3-g32-experts", "q2-g32-experts", "mixed-q2q3-experts"]
    for s in must:
        assert s in specs, f"missing pinned spec {s}"
    assert ranked[0]["family"] == "alt_group"
    first_bits = parse_spec(specs[0]).get("bits")
    assert first_bits is not None and int(first_bits) >= 3, specs[0]
    assert any("+correction" in s or "+c0.02" in s or "+c0.05" in s for s in specs)
    assert any(s.startswith("tiers-") for s in specs)
    assert any(s.startswith("scale-joint-") for s in specs)
    assert any(s.endswith("+r8") or "+r8+" in s for s in specs)
    assert any("meta-entropy" in s or "meta-shared" in s for s in specs)

    # Round-trip a sample of specs through parse/format.
    sample = specs[:12] + specs[-8:] + must
    for s in sample:
        parsed = parse_spec(s)
        rebuilt = format_spec(
            form=parsed["form"],
            bits=parsed["bits"],
            group=parsed["group"],
            target=parsed["target"],
            mixed_lo=parsed["mixed_lo"],
            mixed_hi=parsed["mixed_hi"],
            tiers=parsed["tiers"],
            correction_budget=parsed["correction_budget"],
            correction_token=parsed["correction_token"],
            router_precision=parsed["router_precision"],
            metadata_codec=parsed["metadata_codec"],
        )
        assert rebuilt == s, (s, rebuilt)

    # Ranking stability: two calls, identical spec order.
    ranked2 = generate("moe", census, sensitivity, policy)
    assert [c["spec"] for c in ranked2] == specs

    # Dense / hybrid analogues exist and use the right target suffix.
    dense_census = {
        "is_moe": False,
        "total_params": 1_000_000,
        "total_bytes": 2_000_000,
        "stored_bpw": 16.0,
        "organs_params": {"mlp_dense": 800_000, "attn": 100_000, "expert": 0, "other": 0},
    }
    dense = generate("dense", dense_census, {}, policy)
    assert dense, "dense generate empty"
    assert any(c["spec"] == "q4-g64" for c in dense)
    assert all("-experts" not in c["spec"] for c in dense)

    hybrid_census = {
        "is_moe": False,
        "total_params": 1_000_000,
        "total_bytes": 2_000_000,
        "stored_bpw": 16.0,
        "organs_params": {"mlp_dense": 500_000, "attn": 100_000, "other": 300_000, "expert": 0},
    }
    hybrid = generate("hybrid", hybrid_census, {}, policy)
    assert hybrid, "hybrid generate empty"
    assert any(c["spec"] == "q4-g64-attn-mlp" for c in hybrid)

    # require_native drops 1-bit / correction / entropy / tiers / joint.
    native_only = generate(
        "moe", census, sensitivity, {**policy, "require_native": True}
    )
    assert native_only
    assert all(c["native"] for c in native_only)
    assert all(parse_spec(c["spec"]).get("bits") != 1 for c in native_only)

    # Tiny source-pass budget still returns a non-empty prefix.
    tight = generate(
        "moe", census, sensitivity, {**policy, "source_pass_budget": 3}
    )
    assert tight, "source-pass budget emptied the list"
    assert len(tight) < len(ranked)

    # Strictly-dominated prune.
    cands = [
        {"spec": "q2-g32-experts", "family": "alt_group"},
        {"spec": "q3-g32-experts", "family": "alt_group"},
        {"spec": "q2-g64-experts", "family": "alt_group"},
        {"spec": "mixed-q2q3-experts", "family": "sensitivity_driven_alloc"},
    ]
    results = {
        "q2-g32-experts": {
            "complete_bpw": 3.0,
            "delta_hits": 0,
            "active_bytes": 1_000_000_000,
        },
        "q3-g32-experts": {
            "complete_bpw": 4.0,
            "delta_hits": 0,
            "active_bytes": 1_200_000_000,
        },
        "q2-g64-experts": {
            "complete_bpw": 2.8,
            "delta_hits": -2,
            "active_bytes": 900_000_000,
        },
        # mixed has no result → kept
    }
    pruned = prune(cands, results)
    kept_specs = [_spec_of(c) for c in pruned]
    assert "q3-g32-experts" not in kept_specs, kept_specs
    assert "q2-g32-experts" in kept_specs
    assert "q2-g64-experts" in kept_specs  # better bpw, worse doctor — incomparable
    assert "mixed-q2q3-experts" in kept_specs  # no result → keep
    assert kept_specs == [
        s for s in (c["spec"] for c in cands) if s != "q3-g32-experts"
    ]

    # Alias keys (stored_bpw / doctor_delta / active_bytes_per_token) also dominate.
    pruned2 = prune(
        [{"spec": "a"}, {"spec": "b"}],
        [
            {
                "spec": "a",
                "stored_bpw": 2.0,
                "doctor_delta": 1,
                "active_bytes_per_token": 10,
            },
            {
                "spec": "b",
                "stored_bpw": 3.0,
                "doctor_delta": 0,
                "active_bytes_per_token": 20,
            },
        ],
    )
    assert [_spec_of(c) for c in pruned2] == ["a"]

    print(
        f"self-check ok  moe={len(ranked)} dense={len(dense)} "
        f"hybrid={len(hybrid)} native_only={len(native_only)} "
        f"first={specs[0]}"
    )
    return 0


def _cli_generate(args: argparse.Namespace) -> int:
    policy = load_policy(Path(args.policy) if args.policy else None)
    if args.require_native:
        policy = {**policy, "require_native": True}
    if args.max_candidates:
        policy = {**policy, "max_candidates": int(args.max_candidates)}
    ranked = generate(args.patient_class, args.census, args.sensitivity, policy)
    payload = [{"rank": c["rank"], "spec": c["spec"], "family": c["family"],
                "conventionality": c["conventionality"],
                "native": c["native"],
                "estimated_complete_bpw": c["estimated_complete_bpw"],
                "doctor_risk": c["doctor_risk"]} for c in ranked]
    text = json.dumps(payload, indent=2, sort_keys=False)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--self-check", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("generate", help="expand + rank candidates")
    g.add_argument("--class", dest="patient_class", required=True)
    g.add_argument("--census", type=Path, default=None)
    g.add_argument("--sensitivity", type=Path, default=None)
    g.add_argument("--policy", type=Path, default=None)
    g.add_argument("--require-native", action="store_true")
    g.add_argument("--max-candidates", type=int, default=None)
    g.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.self_check or args.cmd in {None, "self-check", "selfcheck"}:
        try:
            return self_check()
        except AssertionError as e:
            print(f"self-check FAILED: {e}", file=sys.stderr)
            return 1
    if args.cmd == "generate":
        return _cli_generate(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
