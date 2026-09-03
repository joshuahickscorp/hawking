#!/usr/bin/env python3
"""Odyssey compile-economics + detachment-metrics recorder.

S003 §7-11: per-patient COMPILE_ECONOMICS event log, normalized metrics,
TRANSFER_ACCELERATION (honest: no invented from-scratch denominator), and
ODYSSEY_COST_MODEL with UNCERTAINTY — never a point estimate as fact.
S004 §27/§71: frontier-depth fields + detachment metrics.

Deterministic. stdlib + json only. Timestamps are passed in; pure functions
do not call the wall clock. `record` may stamp with `time` if `ts` is omitted.

    python3 tools/odyssey_costmodel.py --self-check
    python3 tools/odyssey_costmodel.py --fit
    python3 tools/odyssey_costmodel.py --derive O005
    python3 tools/odyssey_costmodel.py --detachment
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
ODYSSEY = REPO / "workspace" / "campaign" / "odyssey"
ECONOMICS = ODYSSEY / "COMPILE_ECONOMICS.jsonl"
COST_MODEL = ODYSSEY / "ODYSSEY_COST_MODEL.json"
RUN_LOG = ODYSSEY / "RUN_LOG.jsonl"
POLICY = ODYSSEY / "ODYSSEY_POLICY.json"
ESCALATIONS = ODYSSEY / "OPUS_ESCALATIONS.jsonl"
TRANSFER = ODYSSEY / "TRANSFER_MATRIX.json"
RULEBASE = ODYSSEY / "GRAVITY_RULEBASE.json"
STATE = ODYSSEY / "ODYSSEY_STATE.json"
PATIENTS_DIR = ODYSSEY / "patients"

SCHEMA_EVENT = "hawking.odyssey.compile_economics.event.v1"
SCHEMA_DERIVE = "hawking.odyssey.compile_economics.derived.v1"
SCHEMA_MODEL = "hawking.odyssey.cost_model.v1"
SCHEMA_DETACH = "hawking.odyssey.detachment_metrics.v1"

GB = 1e9
BILLION = 1e9

# S003 §7 event → packet wall field
EVENT_WALL = {
    "acquisition": "acquisition_wall",
    "census": "census_wall",
    "doctor": "doctor_baseline_wall",
    "doctor_baseline": "doctor_baseline_wall",
    "doctor_fast": "doctor_fast_wall",
    "doctor_full": "doctor_full_wall",
    "gravity": "gravity_search_wall",
    "gravity_analysis": "gravity_analysis_wall",
    "gravity_search": "gravity_search_wall",
    "gravity_pack": "gravity_pack_wall",
    "gravity_verify": "gravity_verify_wall",
    "nx": "nx_lower_wall",
    "nx_lower": "nx_lower_wall",
    "first_valid_nx": "first_valid_nx_wall",
    "kernel_probe": "kernel_probe_wall",
    "gpu": "total_gpu_wall",
    "cpu": "total_cpu_wall",
    "grok": "grok_wall",
    "opus": "opus_wall",
    "retirement": "retirement_wall",
}

GRAVITY_EVENTS = frozenset({
    "gravity", "gravity_analysis", "gravity_search", "gravity_pack", "gravity_verify",
})
NX_EVENTS = frozenset({"nx", "nx_lower", "first_valid_nx", "valid_nx"})
DOCTOR_FAST_EVENTS = frozenset({"doctor_fast"})
DOCTOR_FULL_EVENTS = frozenset({"doctor", "doctor_baseline", "doctor_full"})
CANDIDATE_EVENTS = frozenset({"candidate", "gravity_search", "gravity_pack"})
ANCHOR_CLASSES = frozenset({"CONVENTIONAL_ANCHOR"})
FRONTIER_CLASSES = frozenset({
    "FRONTIER", "FINALIST", "STRUCTURAL_GRAVITY", "ACTIVE_NX",
})
REUSED_STATUSES = frozenset({"TRANSFERRED_UNCHANGED"})
RETUNED_STATUSES = frozenset({"TRANSFERRED_RETUNED"})

# S003 §10 numeric inputs that may drive an ETA (rates, never facts).
NUMERIC_FEATURES = (
    "source_bytes",
    "parameter_count",
    "active_parameter_count",
    "tensor_count",
    "expert_count",
    "representation_passes",
    "rule_reuse_pct",
    "doctor_workload",
    "native_primitives",
    "source_acquisition_size",
)
OUTPUT_WALLS = (
    "acquisition_wall",
    "gravity_wall",
    "first_nx_wall",
    "doctor_wall",
    "patient_wall",
)

DESIRED_TREND = {
    "deterministic_fraction": "UP",
    "candidate_throughput": "UP",
    "grok_novelty_quality": "UP",
    "opus_dependency": "DOWN",
}


# ---------------------------------------------------------------------------
# paths / io
# ---------------------------------------------------------------------------

class Paths:
    """Redirectable Odyssey artifact locations (self-check uses a temp root)."""

    def __init__(self, odyssey: Path | None = None):
        self.odyssey = Path(odyssey) if odyssey is not None else ODYSSEY

    @property
    def economics(self) -> Path:
        return self.odyssey / "COMPILE_ECONOMICS.jsonl"

    @property
    def cost_model(self) -> Path:
        return self.odyssey / "ODYSSEY_COST_MODEL.json"

    @property
    def run_log(self) -> Path:
        return self.odyssey / "RUN_LOG.jsonl"

    @property
    def policy(self) -> Path:
        p = self.odyssey / "ODYSSEY_POLICY.json"
        return p if p.is_file() else POLICY

    @property
    def escalations(self) -> Path:
        return self.odyssey / "OPUS_ESCALATIONS.jsonl"

    @property
    def transfer(self) -> Path:
        p = self.odyssey / "TRANSFER_MATRIX.json"
        return p if p.is_file() else TRANSFER

    @property
    def rulebase(self) -> Path:
        p = self.odyssey / "GRAVITY_RULEBASE.json"
        return p if p.is_file() else RULEBASE

    @property
    def state(self) -> Path:
        p = self.odyssey / "ODYSSEY_STATE.json"
        return p if p.is_file() else STATE

    @property
    def patients(self) -> Path:
        p = self.odyssey / "patients"
        return p if p.is_dir() else PATIENTS_DIR


DEFAULT = Paths()


def _oxx(patient) -> str:
    s = str(patient).strip()
    u = s.upper()
    if u.startswith("O") and u[1:].isdigit():
        return f"O{int(u[1:]):03d}"
    if u.isdigit():
        return f"O{int(u):03d}"
    return s


def _read_json(path: Path) -> dict | None:
    if not path or not Path(path).is_file():
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_jsonl(path: Path | None) -> list[dict]:
    if not path or not Path(path).is_file():
        return []
    rows: list[dict] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _append_jsonl(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        fh.flush()


def _write_json(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _as_float(x, default=None):
    if x is None or x == "":
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v


def _as_int(x, default=0) -> int:
    v = _as_float(x, None)
    if v is None:
        return default
    return int(v)


def _rate(num, den):
    """num/den, or None if not a finite real. Never returns inf/nan."""
    n = _as_float(num, None)
    d = _as_float(den, None)
    if n is None or d is None or d == 0.0:
        return None
    q = n / d
    return q if math.isfinite(q) else None


def _mean_std(xs: list[float]) -> tuple[float | None, float | None, int]:
    vals = [float(x) for x in xs if _finite(x)]
    n = len(vals)
    if n == 0:
        return None, None, 0
    mu = sum(vals) / n
    if n < 2:
        return mu, None, n
    var = sum((x - mu) ** 2 for x in vals) / (n - 1)
    return mu, math.sqrt(var), n


def _resolve_paths(paths: Paths | None = None, *, odyssey=None) -> Paths:
    if paths is not None:
        return paths
    if odyssey is not None:
        return Paths(Path(odyssey))
    return DEFAULT


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

def record(patient, event, wall_s, bytes_scanned=0, bytes_transformed=0,
           grok_lane=None, opus=False, extra=None, *, ts=None, path=None,
           paths: Paths | None = None):
    """Append one compile-economics event to COMPILE_ECONOMICS.jsonl.

    Returns the written record. `ts` may be passed in (seconds); if omitted
    the recorder stamps with time.time(). Extra fields (bpw, candidate_class,
    source_params, rules_*, valid_nx, cheap_kill, doctor_kind, opus_tokens,
    cycle, pass counts, gpu_s/cpu_s, …) land on the event via `extra`.
    """
    extra = dict(extra or {})
    wall = _as_float(wall_s, None)
    if wall is None:
        raise ValueError(f"wall_s must be a finite number, got {wall_s!r}")
    stamp = extra.pop("ts", None) if ts is None else ts
    if stamp is None:
        stamp = time.time()
    rec = {
        "schema": SCHEMA_EVENT,
        "patient": _oxx(patient),
        "event": str(event),
        "wall_s": float(wall),
        "bytes_scanned": _as_int(bytes_scanned, 0),
        "bytes_transformed": _as_int(bytes_transformed, 0),
        "grok_lane": grok_lane,
        "opus": bool(opus),
        "ts": _as_float(stamp, float(stamp) if _finite(stamp) else time.time()),
        # A ZERO WALL MAY NOT CLAIM TO BE MEASURED. Every call site in
        # tools/odyssey_ctl.py records a LAUNCH marker, where no duration exists
        # yet, and passes wall_s=0.0 -- so all 9,573 events in
        # COMPILE_ECONOMICS.jsonl carried "_evidence": "MEASURED" against 0.0
        # seconds, across 72 hours of timestamps. Zero is the right value for a
        # start marker; MEASURED is the wrong label for it, and the two together
        # are how a ledger reports a full dataset and contains no measurement.
        "_evidence": extra.pop("_evidence", "MEASURED" if wall > 0 else "UNRECORDED"),
    }
    for k, v in extra.items():
        if k in rec:
            continue
        rec[k] = v
    dest = Path(path) if path is not None else _resolve_paths(paths).economics
    _append_jsonl(dest, rec)
    return rec


# ---------------------------------------------------------------------------
# patient features / transfer
# ---------------------------------------------------------------------------

def _patient_features(patient, paths: Paths | None = None) -> dict:
    oxx = _oxx(patient)
    pdir = _resolve_paths(paths).patients
    census = _read_json(pdir / oxx / "census.json") or {}
    pkt = _read_json(pdir / oxx / f"ODYSSEY_PATIENT_{oxx}.json") or {}
    arch = pkt.get("architecture") if isinstance(pkt.get("architecture"), dict) else {}
    ident = pkt.get("identity") if isinstance(pkt.get("identity"), dict) else {}
    rep = pkt.get("representation") if isinstance(pkt.get("representation"), dict) else {}
    cfg = census.get("config") if isinstance(census.get("config"), dict) else {}
    source_bytes = census.get("total_bytes") or rep.get("source_bytes")
    params = census.get("total_params") or arch.get("total_params")
    active = (
        census.get("active_params_per_token")
        or arch.get("active_params")
        or arch.get("active_params_per_token")
    )
    if active is None and params is not None:
        # dense / unset: every source param is active
        active = params
    experts = cfg.get("num_experts") or cfg.get("n_routed_experts") or arch.get("experts")
    return {
        "source_bytes": _as_float(source_bytes, None),
        "parameter_count": _as_float(params, None),
        "active_parameter_count": _as_float(active, None),
        "tensor_count": _as_float(census.get("tensor_count"), None),
        "expert_count": _as_float(experts, None),
        "architecture_family": arch.get("kind") or census.get("model_type"),
        "modality": arch.get("modality") or ident.get("modality"),
        "source_acquisition_size": _as_float(source_bytes, None),
    }


def _transfer_rule_counts(patient, paths: Paths | None = None) -> dict:
    oxx = _oxx(patient)
    reused = retuned = new = 0
    transfer = _read_json(_resolve_paths(paths).transfer)
    if transfer:
        for row in transfer.get("rows") or []:
            if not isinstance(row, dict):
                continue
            cells = row.get("cells") or {}
            st = cells.get(oxx)
            if st in REUSED_STATUSES:
                reused += 1
            elif st in RETUNED_STATUSES:
                retuned += 1
    rulebase = _read_json(_resolve_paths(paths).rulebase)
    if rulebase:
        for rule in rulebase.get("rules") or []:
            if not isinstance(rule, dict):
                continue
            support = list(rule.get("supporting_patients") or [])
            if support == [oxx] or (support and support[0] == oxx and len(support) == 1):
                new += 1
    return {"rules_reused": reused, "rules_retuned": retuned, "rules_new": new}


def _events_for(patient, rows: list[dict]) -> list[dict]:
    oxx = _oxx(patient)
    ev = [r for r in rows if _oxx(r.get("patient") or r.get("oxx") or "") == oxx]
    ev.sort(key=lambda r: (
        _as_float(r.get("ts"), float("inf")),
        str(r.get("event") or ""),
    ))
    return ev


def _event_name(e: dict) -> str:
    return str(e.get("event") or "").strip()


def _bpw(e: dict):
    for k in ("complete_bpw", "bpw", "stored_bpw", "active_bpw"):
        v = _as_float(e.get(k), None)
        if v is not None:
            return v
    return None


def _is_gravity(e: dict) -> bool:
    return _event_name(e) in GRAVITY_EVENTS or str(e.get("phase") or "").startswith("gravity")


def _is_valid_nx(e: dict) -> bool:
    name = _event_name(e)
    if name in {"first_valid_nx", "valid_nx"}:
        return True
    if e.get("valid_nx") is True:
        return True
    if name in NX_EVENTS and e.get("valid_nx") is not False:
        return name == "first_valid_nx" or e.get("valid_nx") is True
    return False


def _is_anchor(e: dict) -> bool:
    name = _event_name(e)
    if name in {"conventional_anchor", "anchor"}:
        return True
    cls = str(e.get("candidate_class") or e.get("conventionality") or "")
    return cls in ANCHOR_CLASSES


def _is_frontier(e: dict) -> bool:
    name = _event_name(e)
    if name in {"frontier", "best_frontier"}:
        return True
    cls = str(e.get("candidate_class") or "")
    return cls in FRONTIER_CLASSES


def _is_sub(e: dict, thresh: float) -> bool:
    name = _event_name(e)
    tag = {3.0: ("sub3", "first_sub3"), 2.5: ("sub2_5", "first_sub2_5"),
           2.0: ("sub2", "first_sub2")}.get(thresh, ())
    if name in tag:
        return True
    bpw = _bpw(e)
    return bpw is not None and bpw < thresh


def _is_cheap_kill(e: dict) -> bool:
    return _event_name(e) == "cheap_kill" or e.get("cheap_kill") is True


def _is_doctor_fast(e: dict) -> bool:
    return _event_name(e) in DOCTOR_FAST_EVENTS or str(e.get("doctor_kind") or "") == "fast"


def _is_doctor_full(e: dict) -> bool:
    return _event_name(e) in DOCTOR_FULL_EVENTS or str(e.get("doctor_kind") or "") == "full"


def _is_candidate(e: dict) -> bool:
    return _event_name(e) in CANDIDATE_EVENTS or e.get("candidate") is True


def _is_grok(e: dict) -> bool:
    return bool(e.get("grok_lane")) or _event_name(e) == "grok" or e.get("actor") == "grok"


def _is_opus(e: dict) -> bool:
    return bool(e.get("opus")) or _event_name(e) == "opus" or e.get("actor") == "opus"


def _elapsed(events: list[dict], pred) -> float | None:
    """Seconds from first event to first match. Prefer passed-in timestamps."""
    if not events:
        return None
    use_ts = all(_finite(e.get("ts")) for e in events)
    t0 = float(events[0]["ts"]) if use_ts else 0.0
    cum = 0.0
    for e in events:
        cum += float(e.get("wall_s") or 0.0)
        if pred(e):
            if use_ts:
                return float(e["ts"]) - t0
            return cum
    return None


def _count_until(events: list[dict], pred_count, pred_stop) -> int:
    n = 0
    seen_stop = False
    for e in events:
        if pred_count(e):
            n += 1
        if pred_stop(e):
            seen_stop = True
            break
    return n if seen_stop or n else 0


def _last_numeric(events: list[dict], key: str):
    last = None
    for e in events:
        if key in e and e[key] is not None:
            v = _as_float(e[key], None)
            if v is not None:
                last = v
    return last


def _last_any(events: list[dict], key: str):
    last = None
    for e in events:
        if key in e and e[key] is not None:
            last = e[key]
    return last


# ---------------------------------------------------------------------------
# derive
# ---------------------------------------------------------------------------

def derive(patient, path=None, paths: Paths | None = None) -> dict:
    """Normalized compile-economics for one patient (S003 §8 + S004 §27).

    Pure over the event log + on-disk packet/census/transfer. No wall-clock.
    Ratios that cannot be formed are None — never inf/nan.
    """
    ps = _resolve_paths(paths)
    econ = Path(path) if path is not None else ps.economics
    rows = _read_jsonl(econ)
    events = _events_for(patient, rows)
    oxx = _oxx(patient)
    walls = {
        "acquisition_wall": 0.0,
        "census_wall": 0.0,
        "doctor_baseline_wall": 0.0,
        "doctor_fast_wall": 0.0,
        "doctor_full_wall": 0.0,
        "gravity_analysis_wall": 0.0,
        "gravity_search_wall": 0.0,
        "gravity_pack_wall": 0.0,
        "gravity_verify_wall": 0.0,
        "nx_lower_wall": 0.0,
        "kernel_probe_wall": 0.0,
        "total_gpu_wall": 0.0,
        "total_cpu_wall": 0.0,
        "grok_wall": 0.0,
        "opus_wall": 0.0,
        "total_patient_wall": 0.0,
        "first_valid_nx_wall": 0.0,
        "retirement_wall": 0.0,
    }
    bytes_scanned = 0
    bytes_transformed = 0
    grav_scan = grav_xform = 0
    grav_wall = 0.0
    gpu_s = cpu_s = 0.0
    grok_lanes: set[str] = set()
    opus_n = 0
    opus_tokens = 0.0
    n_experiments = 0
    n_valid_nx = 0
    full_passes = sampled_passes = transform_passes = verify_passes = 0
    source_bytes = None
    source_params = None
    active_params = None

    for e in events:
        w = float(e.get("wall_s") or 0.0)
        walls["total_patient_wall"] += w
        field = EVENT_WALL.get(_event_name(e))
        if field:
            walls[field] = walls.get(field, 0.0) + w
        bs = _as_int(e.get("bytes_scanned"), 0)
        bt = _as_int(e.get("bytes_transformed"), 0)
        bytes_scanned += bs
        bytes_transformed += bt
        if _is_gravity(e):
            grav_scan += bs
            grav_xform += bt
            grav_wall += w
        if _is_grok(e):
            walls["grok_wall"] += w if field != "grok_wall" else 0.0
            lane = e.get("grok_lane")
            if lane:
                grok_lanes.add(str(lane))
            elif _event_name(e) == "grok":
                grok_lanes.add(f"grok:{len(grok_lanes)}")
        if _is_opus(e):
            opus_n += 1
            if field != "opus_wall":
                walls["opus_wall"] += w
        gpu_s += _as_float(e.get("gpu_s"), 0.0) or 0.0
        cpu_s += _as_float(e.get("cpu_s"), 0.0) or 0.0
        opus_tokens += _as_float(e.get("opus_tokens"), 0.0) or 0.0
        if e.get("experiment") is True or _is_candidate(e) or _is_gravity(e):
            n_experiments += 1
        if _is_valid_nx(e):
            n_valid_nx += 1
        full_passes += _as_int(e.get("full_source_passes"), 0)
        sampled_passes += _as_int(e.get("sampled_passes"), 0)
        transform_passes += _as_int(e.get("transform_passes"), 0)
        verify_passes += _as_int(e.get("verify_passes"), 0)
        if e.get("source_bytes") is not None:
            source_bytes = _as_float(e.get("source_bytes"), source_bytes)
        if e.get("source_params") is not None:
            source_params = _as_float(e.get("source_params"), source_params)
        if e.get("parameter_count") is not None:
            source_params = _as_float(e.get("parameter_count"), source_params)
        if e.get("active_params") is not None:
            active_params = _as_float(e.get("active_params"), active_params)
        if e.get("active_parameter_count") is not None:
            active_params = _as_float(e.get("active_parameter_count"), active_params)

    if gpu_s:
        walls["total_gpu_wall"] = max(walls["total_gpu_wall"], gpu_s)
    if cpu_s:
        walls["total_cpu_wall"] = max(walls["total_cpu_wall"], cpu_s)

    feat = _patient_features(oxx, paths=ps)
    if source_bytes is None:
        source_bytes = feat.get("source_bytes")
    if source_params is None:
        source_params = feat.get("parameter_count")
    if active_params is None:
        active_params = feat.get("active_parameter_count")
    if source_bytes is None and bytes_scanned:
        source_bytes = float(bytes_scanned)

    # Gravity GB/s: gravity events if present, else whole-patient scan/transform.
    scan_b = grav_scan if grav_scan else bytes_scanned
    xform_b = grav_xform if grav_xform else bytes_transformed
    scan_t = grav_wall if grav_wall else walls["total_patient_wall"]
    gb_s_scanned = _rate(scan_b / GB, scan_t)
    gb_s_xform = _rate(xform_b / GB, scan_t)

    first_nx_wall = _elapsed(events, _is_valid_nx)
    if first_nx_wall is not None:
        walls["first_valid_nx_wall"] = first_nx_wall

    src_B = _rate(source_params, BILLION)
    act_B = _rate(active_params, BILLION)
    s_per_B_source = _rate(walls["total_patient_wall"], src_B)
    s_per_B_active = _rate(walls["total_patient_wall"], act_B)
    first_nx_s_per_B = _rate(walls["first_valid_nx_wall"], src_B)
    doctor_wall = (
        walls["doctor_baseline_wall"] + walls["doctor_fast_wall"] + walls["doctor_full_wall"]
    )
    doctor_s_per_B_active = _rate(doctor_wall, act_B)
    experiments_per_valid_nx = _rate(n_experiments, n_valid_nx) if n_valid_nx else (
        float(n_experiments) if n_experiments == 0 else None
    )
    # 0 experiments / 0 NX is 0 (finite); >0 experiments / 0 NX is undefined.
    if n_valid_nx == 0:
        experiments_per_valid_nx = 0.0 if n_experiments == 0 else None

    rules_xtra = {
        "rules_reused": _last_numeric(events, "rules_reused"),
        "rules_retuned": _last_numeric(events, "rules_retuned"),
        "rules_new": _last_numeric(events, "rules_new"),
    }
    rules_file = _transfer_rule_counts(oxx, paths=ps)
    rules_reused = rules_xtra["rules_reused"]
    rules_retuned = rules_xtra["rules_retuned"]
    rules_new = rules_xtra["rules_new"]
    if rules_reused is None:
        rules_reused = float(rules_file["rules_reused"])
    if rules_retuned is None:
        rules_retuned = float(rules_file["rules_retuned"])
    if rules_new is None:
        rules_new = float(rules_file["rules_new"])

    frontier = {
        "time_to_first_valid_nx": first_nx_wall,
        "time_to_conventional_anchor": _elapsed(events, _is_anchor),
        "time_to_first_sub3": _elapsed(events, lambda e: _is_sub(e, 3.0)),
        "time_to_first_sub2_5": _elapsed(events, lambda e: _is_sub(e, 2.5)),
        "time_to_first_sub2": _elapsed(events, lambda e: _is_sub(e, 2.0)),
        "time_to_best_frontier": _elapsed(events, _is_frontier),
        "candidates_to_frontier": _count_until(events, _is_candidate, _is_frontier),
        "cheap_kills": sum(1 for e in events if _is_cheap_kill(e)),
        "doctor_fast_count": sum(1 for e in events if _is_doctor_fast(e)),
        "full_doctor_count": sum(1 for e in events if _is_doctor_full(e)),
        "grok_calls_to_frontier": _count_until(events, _is_grok, _is_frontier),
    }

    normalized = {
        "gravity_gb_s_scanned": gb_s_scanned,
        "gravity_gb_s_transformed": gb_s_xform,
        "s_per_B_source_param": s_per_B_source,
        "s_per_B_active_param": s_per_B_active,
        "first_nx_s_per_B_param": first_nx_s_per_B,
        "doctor_s_per_B_active": doctor_s_per_B_active,
        "experiments_per_valid_nx": experiments_per_valid_nx,
        "grok_lanes_per_patient": float(len(grok_lanes)),
        "opus_per_patient": float(opus_n),
        "rules_reused": float(rules_reused or 0.0),
        "rules_retuned": float(rules_retuned or 0.0),
        "rules_new": float(rules_new or 0.0),
    }

    compile_economics = {
        **walls,
        "source_bytes": source_bytes,
        "bytes_scanned": bytes_scanned,
        "bytes_transformed": bytes_transformed,
        "grok_lane_count": len(grok_lanes),
        "opus_escalations": opus_n,
        "opus_tokens": opus_tokens if opus_tokens else None,
        "source_params": source_params,
        "active_params": active_params,
    }

    return {
        "schema": SCHEMA_DERIVE,
        "patient": oxx,
        "n_events": len(events),
        "compile_economics": compile_economics,
        "normalized": normalized,
        "frontier_depth": frontier,
        "transfer": {
            "rules_reused": normalized["rules_reused"],
            "rules_retuned": normalized["rules_retuned"],
            "new_rule_count": normalized["rules_new"],
            "actual_wall_s": walls["total_patient_wall"],
            "transfer_acceleration": None,
            "transfer_acceleration_status": "insufficient evidence",
            "_note": (
                "S003 §9: do not invent the expected-from-scratch denominator "
                "until enough patients exist. Until then report rules reused + "
                "new-rule count + actual wall."
            ),
        },
        "pass_counts": {
            "full_source_passes": full_passes,
            "sampled_passes": sampled_passes,
            "transform_passes": transform_passes,
            "verify_passes": verify_passes,
        },
        "features": feat,
        "_evidence": "DERIVED",
    }


def _patients_in_log(rows: list[dict]) -> list[str]:
    seen: list[str] = []
    for r in rows:
        p = r.get("patient") or r.get("oxx")
        if not p:
            continue
        o = _oxx(p)
        if o not in seen:
            seen.append(o)
    return seen


# ---------------------------------------------------------------------------
# detachment metrics (S004 §71)
# ---------------------------------------------------------------------------

def _classify_run_log(rec: dict) -> str:
    if rec.get("opus") or rec.get("escalation") or rec.get("actor") == "opus":
        return "opus"
    schema = str(rec.get("schema") or "")
    if "opus" in schema:
        return "opus"
    if rec.get("actor") == "grok" or rec.get("grok_lane") or "grok" in schema:
        return "grok"
    return "deterministic"


def _load_policy_trend(paths: Paths) -> dict:
    pol = _read_json(paths.policy) or {}
    raw = pol.get("desired_trend")
    trend = dict(DESIRED_TREND)
    if isinstance(raw, str):
        # "deterministic_fraction UP, candidate_throughput UP, ..."
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        for part in parts:
            bits = part.split()
            if len(bits) >= 2:
                key = bits[0].replace("-", "_")
                trend[key] = bits[-1].upper()
    elif isinstance(raw, dict):
        trend.update({str(k): str(v).upper() for k, v in raw.items()})
    return trend


def _cycle_id(paths: Paths, events: list[dict]) -> int | str:
    c = _last_any(events, "cycle")
    if c is not None:
        return c
    st = _read_json(paths.state) or {}
    for k in ("cycle", "storage_cycle", "cycle_id"):
        if st.get(k) is not None:
            return st[k]
    metrics = st.get("metrics") if isinstance(st.get("metrics"), dict) else {}
    if metrics.get("cycle") is not None:
        return metrics["cycle"]
    return 1


def detachment_metrics(paths: Paths | None = None, *,
                       economics_path=None, run_log_path=None,
                       escalations_path=None) -> dict:
    """S004 §71 detachment metrics. Pure over logs. No wall-clock.

    deterministic_decisions/total comes from RUN_LOG.jsonl when present.
    """
    ps = _resolve_paths(paths)
    econ_p = Path(economics_path) if economics_path is not None else ps.economics
    run_p = Path(run_log_path) if run_log_path is not None else ps.run_log
    esc_p = Path(escalations_path) if escalations_path is not None else ps.escalations

    run_rows = _read_jsonl(run_p)
    econ_rows = _read_jsonl(econ_p)
    esc_rows = _read_jsonl(esc_p)
    run_present = Path(run_p).is_file()

    det = grok = opus = 0
    if run_present:
        for rec in run_rows:
            kind = _classify_run_log(rec)
            if kind == "opus":
                opus += 1
            elif kind == "grok":
                grok += 1
            else:
                det += 1
    total = det + grok + opus

    # Economics / explicit escalations fill grok+opus counts even when
    # RUN_LOG is the decision universe for the fraction.
    econ_patients = _patients_in_log(econ_rows)
    run_patients = []
    for rec in run_rows:
        p = rec.get("oxx") or rec.get("patient")
        if p:
            o = _oxx(p)
            if o not in run_patients:
                run_patients.append(o)
    patients = econ_patients or run_patients
    n_patients = len(patients) or 0

    grok_esc = sum(1 for e in econ_rows if _is_grok(e))
    opus_esc = sum(1 for e in econ_rows if _is_opus(e)) + len(esc_rows)
    # unique grok lanes as an alternate escalation count
    grok_lanes = {str(e.get("grok_lane")) for e in econ_rows if e.get("grok_lane")}
    grok_esc_n = max(grok_esc, len(grok_lanes))

    patient_wall = 0.0
    opus_tokens = 0.0
    nx_candidates = 0
    for p in (patients or _patients_in_log(econ_rows)):
        d = derive(p, path=econ_p, paths=ps)
        patient_wall += float(d["compile_economics"]["total_patient_wall"] or 0.0)
        ot = d["compile_economics"].get("opus_tokens")
        opus_tokens += float(ot or 0.0)
        nx_candidates += sum(
            1 for e in _events_for(p, econ_rows)
            if _is_candidate(e) or _is_valid_nx(e) or _event_name(e) in NX_EVENTS
        )

    n_pat = float(n_patients) if n_patients else None
    cycle = _cycle_id(ps, econ_rows)
    # rules across the campaign (rulebase length)
    n_rules = 0
    rb = _read_json(ps.rulebase)
    if rb:
        n_rules = len(rb.get("rules") or [])

    grok_per_patient = _rate(grok_esc_n, n_pat)
    opus_per_patient = _rate(opus_esc, n_pat)
    # /cycle: if cycle is an int id we still report the campaign total this cycle
    opus_per_cycle = float(opus_esc) if opus_esc or cycle is not None else None
    wall_per_opus = _rate(patient_wall, opus_esc)
    wall_per_opus_token = _rate(patient_wall, opus_tokens)
    rules_per_opus = _rate(n_rules, opus_esc)
    nx_per_opus = _rate(nx_candidates, opus_esc)

    trend = _load_policy_trend(ps)
    # Flags are the *desired* direction (policy), not an observed slope.
    # Observed slope needs ≥2 snapshots; we refuse to invent one.
    flags = {
        "deterministic_fraction_up": trend.get("deterministic_fraction") == "UP",
        "candidate_throughput_up": trend.get("candidate_throughput") == "UP",
        "grok_novelty_quality_up": trend.get("grok_novelty_quality") == "UP",
        "opus_dependency_down": trend.get("opus_dependency") == "DOWN",
        "observed_trend": "UNKNOWN",
        "observed_trend_reason": "need ≥2 snapshots to compute a slope; desired flags are policy, not measurement",
    }

    det_frac = _rate(det, total) if run_present else _rate(
        sum(1 for e in econ_rows if not _is_grok(e) and not _is_opus(e)),
        len(econ_rows),
    )

    return {
        "schema": SCHEMA_DETACH,
        "run_log_present": run_present,
        "deterministic_decisions": det if run_present else sum(
            1 for e in econ_rows if not _is_grok(e) and not _is_opus(e)
        ),
        "total_decisions": total if run_present else len(econ_rows),
        "deterministic_fraction": det_frac,
        "grok_escalations": grok_esc_n,
        "opus_escalations": opus_esc,
        "n_patients": n_patients,
        "cycle": cycle,
        "grok_escalations_per_patient": grok_per_patient,
        "opus_escalations_per_patient": opus_per_patient,
        "opus_escalations_per_cycle": opus_per_cycle,
        "patient_wall_per_opus": wall_per_opus,
        "patient_wall_per_opus_token": wall_per_opus_token,
        "rules_per_opus_escalation": rules_per_opus,
        "nr_nx_candidates_per_opus_escalation": nx_per_opus,
        "desired_trend": trend,
        "desired_trend_flags": flags,
        "_evidence": "DERIVED",
        "_note": (
            "S004 §71. Fraction uses RUN_LOG.jsonl when present "
            "(controller decisions are deterministic software). "
            "patient_wall/opus_token is None until opus_tokens are recorded. "
            "Desired-trend flags are policy direction, not an observed slope."
        ),
    }


# ---------------------------------------------------------------------------
# cost model (S003 §10 / S004 §27)
# ---------------------------------------------------------------------------

def _sample_for_fit(patient: str, d: dict) -> dict:
    ce = d["compile_economics"]
    feat = dict(d.get("features") or {})
    # event-level overrides beat census
    if ce.get("source_bytes") is not None:
        feat["source_bytes"] = ce["source_bytes"]
        feat["source_acquisition_size"] = ce["source_bytes"]
    if ce.get("source_params") is not None:
        feat["parameter_count"] = ce["source_params"]
    if ce.get("active_params") is not None:
        feat["active_parameter_count"] = ce["active_params"]
    pc = d.get("pass_counts") or {}
    feat["representation_passes"] = (
        (pc.get("full_source_passes") or 0)
        + (pc.get("sampled_passes") or 0)
        + (pc.get("transform_passes") or 0)
    )
    reused = (d.get("normalized") or {}).get("rules_reused") or 0.0
    new = (d.get("normalized") or {}).get("rules_new") or 0.0
    retuned = (d.get("normalized") or {}).get("rules_retuned") or 0.0
    denom = reused + new + retuned
    feat["rule_reuse_pct"] = (100.0 * reused / denom) if denom else None
    feat["doctor_workload"] = (
        ce.get("doctor_baseline_wall", 0.0)
        + ce.get("doctor_fast_wall", 0.0)
        + ce.get("doctor_full_wall", 0.0)
    )
    feat["native_primitives"] = _as_float(
        (d.get("features") or {}).get("native_primitives"), None
    )
    grav = (
        ce.get("gravity_analysis_wall", 0.0)
        + ce.get("gravity_search_wall", 0.0)
        + ce.get("gravity_pack_wall", 0.0)
        + ce.get("gravity_verify_wall", 0.0)
    )
    return {
        "patient": patient,
        "features": feat,
        "outputs": {
            "acquisition_wall": ce.get("acquisition_wall") or 0.0,
            "gravity_wall": grav,
            "first_nx_wall": ce.get("first_valid_nx_wall") or 0.0,
            "doctor_wall": feat["doctor_workload"] or 0.0,
            "patient_wall": ce.get("total_patient_wall") or 0.0,
        },
        "frontier_depth": d.get("frontier_depth") or {},
        "normalized": d.get("normalized") or {},
    }


def _uncertainty_note(n: int) -> str:
    if n < 2:
        return "insufficient patients — do not treat any number as a fact"
    if n == 2:
        return "sample standard deviation; n=2 is HIGH UNCERTAINTY — not a fact"
    if n < 5:
        return f"sample standard deviation; n={n} is HIGH UNCERTAINTY — INFERRED, not a fact"
    return f"sample standard deviation; n={n} — INFERRED, not a measurement"


def fit_cost_model(paths: Paths | None = None, *,
                   economics_path=None, model_path=None) -> dict:
    """Fit per-input-feature rates WITH UNCERTAINTY (S003 §10).

    Writes ODYSSEY_COST_MODEL.json. If fewer than 2 patients of data exist,
    emits status=\"insufficient data\" and no point estimates presented as fact.
    Pure over the event log (no wall-clock).
    """
    ps = _resolve_paths(paths)
    econ_p = Path(economics_path) if economics_path is not None else ps.economics
    out_p = Path(model_path) if model_path is not None else ps.cost_model
    rows = _read_jsonl(econ_p)
    patients = _patients_in_log(rows)
    samples = [_sample_for_fit(p, derive(p, path=econ_p, paths=ps)) for p in patients]
    n = len(samples)

    model = {
        "schema": SCHEMA_MODEL,
        "source_steer": ["S003 §7-11", "S004 §27", "S004 §71"],
        "policy_ref": "workspace/campaign/odyssey/ODYSSEY_POLICY.json",
        "n_patients": n,
        "patients": patients,
        "status": "insufficient data" if n < 2 else "fit",
        "estimates": {},
        "per_input_feature": {},
        "frontier_depth_summary": {},
        "cycle2_prediction_then_reality": [],
        "uncertainty": _uncertainty_note(n),
        "_label": "UNKNOWN" if n < 2 else "INFERRED",
        "_not_fact": True,
        "_evidence": "UNKNOWN" if n < 2 else "INFERRED",
        "warning": (
            "Estimates carry UNCERTAINTY and must not be treated as measurements "
            "or facts (bible §18; S003 §10). A point estimate is never the answer."
        ),
    }

    if n < 2:
        model["reason"] = (
            "need ≥2 patients of compile-economics data; Cycle-1 trains the ETA "
            "model (S003 §10). Refusing to emit a point estimate."
        )
        _write_json(out_p, model)
        return model

    per_feat: dict = {}
    for feat in NUMERIC_FEATURES:
        outputs: dict = {}
        for yname in OUTPUT_WALLS:
            rates = []
            for s in samples:
                x = _as_float((s["features"] or {}).get(feat), None)
                y = _as_float((s["outputs"] or {}).get(yname), None)
                if x is None or y is None or x == 0.0:
                    continue
                rates.append(y / x)
            mu, sd, k = _mean_std(rates)
            if k == 0:
                continue
            outputs[yname] = {
                "mean_rate": mu,
                "std_rate": sd,
                "n": k,
                "unit": f"seconds_per_unit({feat})",
                "uncertainty": _uncertainty_note(k),
                "_label": "INFERRED",
                "_not_fact": True,
            }
        if outputs:
            per_feat[feat] = {
                "outputs": outputs,
                "_not_fact": True,
                "_label": "INFERRED",
            }
    model["per_input_feature"] = per_feat

    # Per-output wall summary (mean ± sd) — still INFERRED, never MEASURED.
    estimates = {}
    for yname in OUTPUT_WALLS:
        ys = [_as_float(s["outputs"].get(yname), 0.0) or 0.0 for s in samples]
        mu, sd, k = _mean_std(ys)
        estimates[yname] = {
            "mean_s": mu,
            "std_s": sd,
            "n": k,
            "uncertainty": _uncertainty_note(k),
            "_label": "INFERRED",
            "_not_fact": True,
        }
    model["estimates"] = estimates

    # Frontier-depth means across patients (S004 §27) — same uncertainty rule.
    fd_keys = (
        "time_to_first_valid_nx", "time_to_conventional_anchor",
        "time_to_first_sub3", "time_to_first_sub2_5", "time_to_first_sub2",
        "time_to_best_frontier", "candidates_to_frontier", "cheap_kills",
        "doctor_fast_count", "full_doctor_count", "grok_calls_to_frontier",
    )
    fd_sum: dict = {}
    for k in fd_keys:
        vals = []
        for s in samples:
            v = (s.get("frontier_depth") or {}).get(k)
            if _finite(v):
                vals.append(float(v))
        mu, sd, m = _mean_std(vals)
        fd_sum[k] = {
            "mean": mu,
            "std": sd,
            "n": m,
            "uncertainty": _uncertainty_note(m),
            "_label": "INFERRED" if m >= 2 else "UNKNOWN",
            "_not_fact": True,
        }
    model["frontier_depth_summary"] = fd_sum
    _write_json(out_p, model)
    return model


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def _assert_finite_normalized(d: dict) -> None:
    norm = d.get("normalized") or {}
    required = (
        "gravity_gb_s_scanned",
        "gravity_gb_s_transformed",
        "s_per_B_source_param",
        "s_per_B_active_param",
        "first_nx_s_per_B_param",
        "experiments_per_valid_nx",
        "grok_lanes_per_patient",
        "opus_per_patient",
        "rules_reused",
        "rules_retuned",
        "rules_new",
    )
    missing = [k for k in required if k not in norm]
    if missing:
        raise AssertionError(f"derive() missing normalized keys: {missing}")
    bad = {k: norm[k] for k in required if not _finite(norm[k])}
    if bad:
        raise AssertionError(f"derive() non-finite normalized metrics: {bad}")


def _self_check() -> int:
    tmp = tempfile.TemporaryDirectory(prefix="odyssey-costmodel-")
    root = Path(tmp.name)
    ps = Paths(root)
    econ = ps.economics
    model_p = ps.cost_model

    # Two synthetic events on ONE patient → derive finite, fit = insufficient.
    extra = {
        "source_params": 1_000_000_000,
        "active_params": 250_000_000,
        "source_bytes": 2_000_000_000,
        "complete_bpw": 2.4,
        "candidate_class": "CONVENTIONAL_ANCHOR",
        "valid_nx": False,
        "rules_reused": 3,
        "rules_retuned": 1,
        "rules_new": 2,
        "full_source_passes": 1,
        "transform_passes": 1,
        "ts": 1_000.0,
    }
    r1 = record(
        "SYN1", "gravity_pack", 2.0,
        bytes_scanned=2_000_000_000, bytes_transformed=500_000_000,
        extra=extra, ts=1_000.0, path=econ, paths=ps,
    )
    r2 = record(
        "SYN1", "first_valid_nx", 1.0,
        bytes_scanned=100_000_000, bytes_transformed=100_000_000,
        grok_lane="representation",
        extra={
            "source_params": 1_000_000_000,
            "active_params": 250_000_000,
            "valid_nx": True,
            "candidate_class": "ACTIVE_NX",
            "complete_bpw": 2.4,
            "rules_reused": 3,
            "rules_retuned": 1,
            "rules_new": 2,
        },
        ts=1_003.0, path=econ, paths=ps,
    )
    record(
        "SYN1", "doctor_fast", 0.5,
        extra={"doctor_kind": "fast", "source_params": 1_000_000_000,
               "active_params": 250_000_000},
        ts=1_003.5, path=econ, paths=ps,
    )
    assert r1["patient"] == "SYN1" and r2["event"] == "first_valid_nx"

    derived = derive("SYN1", path=econ, paths=ps)
    _assert_finite_normalized(derived)
    fd = derived["frontier_depth"]
    assert _finite(fd["time_to_conventional_anchor"]), fd
    assert _finite(fd["time_to_first_sub3"]), fd
    assert _finite(fd["time_to_first_sub2_5"]), fd
    assert _finite(fd["time_to_first_valid_nx"]), fd
    assert fd["doctor_fast_count"] >= 1
    assert derived["compile_economics"]["bytes_scanned"] > 0

    one = fit_cost_model(paths=ps, economics_path=econ, model_path=model_p)
    if one["status"] != "insufficient data":
        raise AssertionError(f"expected insufficient data with 1 patient, got {one['status']!r}")
    if one.get("n_patients", 0) != 1:
        raise AssertionError(f"n_patients should be 1, got {one.get('n_patients')}")
    if one.get("_not_fact") is not True:
        raise AssertionError("insufficient-data model must set _not_fact")
    written = _read_json(model_p)
    if not written or written.get("status") != "insufficient data":
        raise AssertionError("ODYSSEY_COST_MODEL.json was not written honestly")

    # Empty log → also insufficient, not a crash.
    empty_p = root / "empty.jsonl"
    empty_m = root / "empty_model.json"
    zero = fit_cost_model(paths=ps, economics_path=empty_p, model_path=empty_m)
    if zero["status"] != "insufficient data" or zero.get("n_patients") != 0:
        raise AssertionError(f"empty log should be insufficient data, got {zero}")

    # Detachment on the 1-patient temp log (no RUN_LOG) must not crash.
    det = detachment_metrics(paths=ps, economics_path=econ,
                             run_log_path=root / "missing_RUN_LOG.jsonl")
    if det.get("run_log_present") is not False:
        raise AssertionError("missing RUN_LOG must report run_log_present=false")
    if "desired_trend_flags" not in det or "deterministic_fraction" not in det:
        raise AssertionError("detachment_metrics missing required keys")
    if det["desired_trend_flags"].get("observed_trend") != "UNKNOWN":
        raise AssertionError("must not invent an observed trend from one snapshot")

    # Two-patient fit exists, but every number is labelled INFERRED / not fact.
    record(
        "SYN2", "gravity_pack", 4.0,
        bytes_scanned=4_000_000_000, bytes_transformed=800_000_000,
        extra={"source_params": 2_000_000_000, "active_params": 500_000_000,
               "source_bytes": 4_000_000_000, "complete_bpw": 2.8},
        ts=2_000.0, path=econ, paths=ps,
    )
    record(
        "SYN2", "first_valid_nx", 2.0,
        grok_lane="kernel",
        extra={"source_params": 2_000_000_000, "active_params": 500_000_000,
               "valid_nx": True},
        ts=2_010.0, path=econ, paths=ps,
    )
    two_p = root / "two_model.json"
    two = fit_cost_model(paths=ps, economics_path=econ, model_path=two_p)
    if two["status"] != "fit" or two.get("n_patients") != 2:
        raise AssertionError(f"2-patient fit failed: {two.get('status')} n={two.get('n_patients')}")
    if two.get("_not_fact") is not True or two.get("_label") != "INFERRED":
        raise AssertionError("2-patient fit must not present estimates as fact")
    # at least one feature rate, all tagged
    if not two.get("per_input_feature"):
        raise AssertionError("2-patient fit produced no per-input-feature rates")
    for feat, body in two["per_input_feature"].items():
        if body.get("_not_fact") is not True:
            raise AssertionError(f"feature {feat} missing _not_fact")
        for y, est in (body.get("outputs") or {}).items():
            if est.get("_label") == "MEASURED" or est.get("_not_fact") is not True:
                raise AssertionError(f"{feat}.{y} presented as fact: {est}")

    # Synthetic RUN_LOG: all controller lines → fraction 1.0
    run = root / "RUN_LOG.jsonl"
    _append_jsonl(run, {"schema": "hawking.odyssey.run_log.v1", "oxx": "SYN1",
                        "verdict": "DRY-RUN", "go": False})
    _append_jsonl(run, {"schema": "hawking.odyssey.acquire.v1", "oxx": "SYN2",
                        "verdict": "REFUSE"})
    det2 = detachment_metrics(paths=ps, economics_path=econ, run_log_path=run)
    if det2["run_log_present"] is not True:
        raise AssertionError("RUN_LOG present not detected")
    if det2["total_decisions"] != 2 or det2["deterministic_decisions"] != 2:
        raise AssertionError(f"RUN_LOG classification wrong: {det2}")
    if det2["deterministic_fraction"] != 1.0:
        raise AssertionError(f"expected fraction 1.0, got {det2['deterministic_fraction']}")

    # Pure functions did not need wall-clock: derive/fit/detachment used only
    # stamps we passed in. (record may use time — not invoked without ts here.)
    print("self-check OK")
    return 0


def _emit_campaign_model() -> None:
    """Write the real ODYSSEY_COST_MODEL.json from the campaign log (honest)."""
    ODYSSEY.mkdir(parents=True, exist_ok=True)
    if not ECONOMICS.is_file():
        ECONOMICS.write_text("", encoding="utf-8")
    fit_cost_model()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or "--help" in argv or "-h" in argv:
        sys.stdout.write(__doc__.strip() + "\n")
        return 0
    if "--self-check" in argv:
        rc = _self_check()
        # Plant an honest campaign model if the real log has <2 patients.
        try:
            rows = _read_jsonl(ECONOMICS)
            if len(_patients_in_log(rows)) < 2:
                _emit_campaign_model()
        except OSError:
            pass
        return rc
    if "--fit" in argv:
        model = fit_cost_model()
        print(json.dumps({
            "status": model["status"],
            "n_patients": model["n_patients"],
            "path": str(COST_MODEL),
            "_not_fact": True,
        }, indent=2))
        return 0
    if "--detachment" in argv:
        print(json.dumps(detachment_metrics(), indent=2))
        return 0
    if "--derive" in argv:
        i = argv.index("--derive")
        if i + 1 >= len(argv):
            print("usage: --derive PATIENT", file=sys.stderr)
            return 2
        print(json.dumps(derive(argv[i + 1]), indent=2))
        return 0
    print("unknown args; try --self-check / --fit / --derive / --detachment",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
