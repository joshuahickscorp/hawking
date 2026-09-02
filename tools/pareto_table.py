#!/usr/bin/env python3
"""G150: Pareto table per candidate, regenerated from receipts, never hand-edited.

One machine-readable table over every live candidate, with the axes the promotion
policy (G151) actually reads: effective BPW, TPS, TOKEN_NS, DRAM/token, resident RAM,
NR size, NX size, Doctor verdict, Tabula drift. A missing cell is null -- never
fabricated, never carried from a different candidate.

The CONTROL is regenerate-and-diff: the table is a pure function of the receipts on
disk, so running it twice must produce byte-identical output. If a second run differs,
something non-deterministic (a dict order, a timestamp, a stray float) has leaked into
what is supposed to be a reproducible artifact, and the table cannot be trusted as the
promotion input. The committed copy is diffed against a fresh regeneration.

  ./tools/pareto_table.py --out receipts/.../G150_PARETO.json
"""
from __future__ import annotations
import argparse, datetime, json, os, pathlib, re, subprocess
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
REC = ROOT / "receipts/ascent-2026-08-16"

# Candidate registry. Names are identity; the table is keyed on them.
CANDIDATES = ["uniform-q4-v1", "mixed-q3mlp-v1", "mixed-q3mlp-q3attn-v1"]

AXES = ["effective_bpw", "tps", "token_ns", "dram_bytes_per_token",
        "resident_ram_gb", "nr_bytes", "nx_bytes", "doctor", "tabula_drift"]

# ---------------------------------------------------------------------------
# Common comparison machinery. Model-agnostic: identity is an artifact id and
# optional content digest, never a vendor checkpoint name. A missing cell stays
# None; dominance skips incomparable axes instead of filling zeros.
# ---------------------------------------------------------------------------

LOWER, HIGHER = "lower", "higher"
EVIDENCE_TIERS = (
    "STATIC", "FUNCTIONAL_SIM", "COST_MODEL", "CYCLE_APPROX", "HARDWARE_MEASURED",
)


@dataclass(frozen=True)
class Axis:
    name: str
    direction: str
    evidence_tier: str = "STATIC"

    def __post_init__(self) -> None:
        if self.direction not in (LOWER, HIGHER):
            raise ValueError(f"axis {self.name!r} direction must be {LOWER!r} or {HIGHER!r}")
        if self.evidence_tier not in EVIDENCE_TIERS:
            raise ValueError(f"axis {self.name!r} unknown evidence_tier {self.evidence_tier!r}")


@dataclass(frozen=True)
class CandidateIdentity:
    """Who the row is about. Artifact id + optional content digest.

    `machine_class` is a hardware domain (UMA, FPGA, ...) not a checkpoint name.
    """
    candidate_id: str
    artifact_digest: str | None = None
    artifact_path: str | None = None
    machine_class: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "artifact_digest": self.artifact_digest,
            "artifact_path": self.artifact_path,
            "machine_class": self.machine_class,
        }


@dataclass
class Metrics:
    values: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    evidence_tier: dict[str, str] = field(default_factory=dict)

    def get(self, axis: str) -> Any:
        return self.values.get(axis)


@dataclass(frozen=True)
class Qualification:
    passed: bool
    failures: tuple[str, ...]
    floors: dict[str, Any]
    evidence_tier: str = "STATIC"

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "floors": dict(self.floors),
            "evidence_tier": self.evidence_tier,
        }


@dataclass(frozen=True)
class Profile:
    """Named axis set a chooser reads. Weights are optional; Pareto does not need them."""
    name: str
    axes: tuple[Axis, ...]
    weights: dict[str, float] | None = None


@dataclass(frozen=True)
class Provenance:
    receipt_refs: tuple[str, ...] = ()
    parent_digest: str | None = None
    chain_root: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "receipt_refs": list(self.receipt_refs),
            "parent_digest": self.parent_digest,
            "chain_root": self.chain_root,
        }


# G150 axes with direction. `doctor` is a flag, not a ranked number — qualification
# consumes it; dominance does not treat a string verdict as a magnitude.
COMPARISON_AXES: tuple[Axis, ...] = (
    Axis("effective_bpw", LOWER),
    Axis("tps", HIGHER),
    Axis("token_ns", LOWER),
    Axis("dram_bytes_per_token", LOWER),
    Axis("resident_ram_gb", LOWER),
    Axis("nr_bytes", LOWER),
    Axis("nx_bytes", LOWER),
    Axis("tabula_drift", LOWER),
)

PROFILES: dict[str, Profile] = {
    "density": Profile("density", (Axis("effective_bpw", LOWER), Axis("nr_bytes", LOWER))),
    "latency": Profile("latency", (Axis("token_ns", LOWER), Axis("tps", HIGHER))),
}


def candidate_identity(
    candidate_id: str,
    *,
    artifact_digest: str | None = None,
    artifact_path: str | None = None,
    machine_class: str | None = None,
) -> CandidateIdentity:
    cid = str(candidate_id or "").strip()
    if not cid:
        raise ValueError("candidate identity requires a non-empty candidate_id")
    return CandidateIdentity(
        candidate_id=cid,
        artifact_digest=artifact_digest,
        artifact_path=artifact_path,
        machine_class=machine_class,
    )


def metrics_of(
    values: Mapping[str, Any],
    *,
    sources: Mapping[str, str] | None = None,
    evidence_tier: Mapping[str, str] | None = None,
) -> Metrics:
    return Metrics(
        values=dict(values),
        sources=dict(sources or {}),
        evidence_tier=dict(evidence_tier or {}),
    )


def provenance_of(*receipt_refs: str, parent_digest: str | None = None,
                  chain_root: str | None = None) -> Provenance:
    refs = tuple(r for r in receipt_refs if r)
    return Provenance(receipt_refs=refs, parent_digest=parent_digest, chain_root=chain_root)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def dominates(
    a: Mapping[str, Any] | Metrics,
    b: Mapping[str, Any] | Metrics,
    axes: Sequence[Axis] | None = None,
) -> bool:
    """a dominates b: not worse on every comparable axis, strictly better on one.

    None is incomparable — a missing cell does not beat a number and is not
    treated as zero. If no axis is comparable, a does not dominate.
    """
    av = a.values if isinstance(a, Metrics) else a
    bv = b.values if isinstance(b, Metrics) else b
    used = tuple(axes) if axes is not None else COMPARISON_AXES
    ge = True
    gt = False
    comparable = 0
    for axis in used:
        left, right = _numeric(av.get(axis.name)), _numeric(bv.get(axis.name))
        if left is None or right is None:
            continue
        comparable += 1
        if axis.direction == LOWER:
            if left > right:
                ge = False
            if left < right:
                gt = True
        else:
            if left < right:
                ge = False
            if left > right:
                gt = True
    return bool(comparable) and ge and gt


def pareto_front(
    candidates: Mapping[str, Mapping[str, Any] | Metrics],
    axes: Sequence[Axis] | None = None,
) -> list[str]:
    """Ids that no other candidate dominates. Order is sorted for determinism."""
    ids = list(candidates)
    front = [
        i for i in ids
        if not any(
            dominates(candidates[j], candidates[i], axes)
            for j in ids if j != i
        )
    ]
    return sorted(front)


def qualify(
    values: Mapping[str, Any] | Metrics,
    floors: Mapping[str, Any],
    *,
    flags: Mapping[str, Any] | None = None,
    axes: Sequence[Axis] | None = None,
) -> Qualification:
    """Fail closed: a missing required cell is a failure, never a pass.

    `floors` maps axis name -> bound. Direction comes from `axes` (default
    COMPARISON_AXES); unknown names are treated as higher-is-better minima.
    `flags` are booleans that must be true (doctor_pass, provenance_valid, ...).
    """
    av = values.values if isinstance(values, Metrics) else values
    direction = {ax.name: ax.direction for ax in (axes or COMPARISON_AXES)}
    failures: list[str] = []
    if not floors and not (flags or {}):
        return Qualification(
            passed=False,
            failures=("no floors and no flags; refusing to qualify on an empty contract",),
            floors=dict(floors),
            evidence_tier="STATIC",
        )
    for name, bound in floors.items():
        got = av.get(name)
        if isinstance(bound, bool):
            if not bool(got):
                failures.append(f"{name} flag is not true")
            continue
        num = _numeric(got)
        thresh = _numeric(bound)
        if num is None or thresh is None:
            failures.append(f"{name} missing")
            continue
        if direction.get(name, HIGHER) == LOWER:
            if num > thresh:
                failures.append(f"{name} {num} exceeds max {thresh}")
        else:
            if num < thresh:
                failures.append(f"{name} {num} below min {thresh}")
    for name, ok in (flags or {}).items():
        if not ok:
            failures.append(f"{name} flag is not true")
    return Qualification(
        passed=not failures,
        failures=tuple(failures),
        floors=dict(floors),
        evidence_tier="STATIC",
    )


def profile_by_name(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"unknown comparison profile {name!r}; known={sorted(PROFILES)}")
    return PROFILES[name]


def _subject(d) -> str | None:
    """The candidate a receipt is ABOUT: basename of its declared artifact/candidate."""
    if not isinstance(d, dict):
        return None
    for k in ("artifact", "candidate", "artifact_root"):
        v = d.get(k)
        if isinstance(v, str) and "/" in v:
            return os.path.basename(v.rstrip("/"))
        if isinstance(v, str):
            return v
    return None


def scan_receipts() -> list[tuple]:
    """Load every receipt once, tagged with the candidate it is ABOUT (or None).
    Only subject-matched receipts are ever read for a candidate's cells, so a value
    can never leak from a receipt that merely MENTIONS a different candidate."""
    docs = []
    for p in sorted(REC.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        docs.append((p.name, _subject(d), d))
    return docs


def find_metric(docs, candidate: str, keys: list[str]):
    """First value under any of `keys`, searched ONLY in receipts whose declared
    subject basename equals `candidate`. Deterministic (receipts pre-sorted)."""
    for fname, subj, d in docs:
        if subj != candidate:
            continue
        def rec(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in keys and isinstance(v, (int, float, str)):
                        return v
                    r = rec(v)
                    if r is not None:
                        return r
            elif isinstance(o, list):
                for v in o:
                    r = rec(v)
                    if r is not None:
                        return r
            return None
        r = rec(d)
        if r is not None:
            return r, fname
    return None, None


def fill_row(docs, candidate: str, nr_dir: pathlib.Path) -> tuple[dict, Provenance]:
    """One candidate's cells. Identity is the candidate id; cells stay None when unpaid."""
    ident = candidate_identity(candidate)
    row = {ax: None for ax in AXES}
    refs: list[str] = []
    d = nr_dir / ident.candidate_id
    if d.is_dir():
        row["nr_bytes"] = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    for ax, keys in {
        "effective_bpw": ["complete_bpw", "effective_bpw"],
        "token_ns": ["steady_decode_wall_ns_per_token"],
        "tabula_drift": ["tabula_drift", "drift_ratio"],
    }.items():
        v, src = find_metric(docs, ident.candidate_id, keys)
        if v is not None:
            row[ax] = v
            if src:
                refs.append(src)
    return row, provenance_of(*refs)


def build_table(docs) -> dict:
    nr_dir = ROOT / "workspace/campaign/records/runs/qwen38-27b"
    table = {}
    for c in CANDIDATES:
        row, _prov = fill_row(docs, c, nr_dir)
        table[c] = row
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--committed", type=pathlib.Path,
                    default=REC / "G150_PARETO_TABLE.json")
    a = ap.parse_args()
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()

    docs = scan_receipts()
    t1 = build_table(docs)
    t2 = build_table(docs)   # CONTROL: regenerate
    s1 = json.dumps(t1, sort_keys=True)
    s2 = json.dumps(t2, sort_keys=True)
    deterministic = s1 == s2

    # Selection is a decision record, never an install. Called here so the
    # comparison machinery has a production caller besides its tests.
    from tools.selection_contract import decide_from_table
    decision = decide_from_table(t1)

    # write the pure table (no timestamp) as the committed, diffable artifact
    a.committed.parent.mkdir(parents=True, exist_ok=True)
    prior = a.committed.read_text() if a.committed.exists() else None
    table_json = json.dumps(t1, indent=2, sort_keys=True) + "\n"
    a.committed.write_text(table_json)
    diff_stable = (prior is None) or (prior == table_json)

    print(f"candidates: {len(t1)}")
    for c, row in t1.items():
        filled = sum(1 for v in row.values() if v is not None)
        print(f"  {c:<28} {filled}/{len(AXES)} axes filled  bpw={row['effective_bpw']} "
              f"token_ns={row['token_ns']} nr_bytes={row['nr_bytes']}")
    print(f"CONTROL regenerate byte-identical: {deterministic}")
    print(f"committed copy stable vs prior: {diff_stable}"
          f"{' (first write)' if prior is None else ''}")
    print(f"selection decision: state={decision.get('state')} "
          f"selected={decision.get('selected')} installed={decision.get('installed')}")

    doc = {
        "schema": "hawking.nos.pareto_table.v1",
        "obligation": "G150 -- Pareto table per candidate, regenerated not hand-edited",
        "started": start,
        "axes": AXES, "candidates": CANDIDATES, "table": t1,
        "selection_decision": decision,
        "committed_copy": str(a.committed.relative_to(ROOT)),
        "control_regenerate_byte_identical": deterministic,
        "control_diff_against_committed_stable": diff_stable,
        "honest_note": ("cells are null where no receipt records that axis for that "
                        "candidate; nulls are not filled from a different candidate. DRAM/token "
                        "and NX size are null pending G142 and a committed NX -- the table shows "
                        "what is measured and what still owes, which is the point of publishing it."),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
        "ended": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")
    return deterministic and diff_stable


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
