#!/usr/bin/env python3
"""N037 DENSITY_DESCENT_FRONTIER: two coupled frontiers, generated from receipts.

S024 §2: a candidate may lead coherence (lowest coherent complete EBPW) and lose
execution (COMPLETE_TOKEN_NS), or vice versa. A single scoreboard number hides
that. This harness reads the campaign's own receipts and places every density-
descent candidate on BOTH axes.

CPU only. Does not load a model, does not touch the GPU, does not run cargo or
Metal, does not mutate NOETIC_PARENT_A, does not re-derive the roofs.

    python3 tools/headless/density_descent_frontier.py
    python3 -m pytest tools/headless -q
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HEADLESS = REPO / "receipts" / "headless"

SCHEMA = "hawking.headless.density_descent_frontier.v1"
RECEIPT = HEADLESS / "DENSITY_DESCENT_FRONTIER.json"
GENERATOR = "tools/headless/density_descent_frontier.py"
OBLIGATION = (
    "N037 — DENSITY_DESCENT_FRONTIER (S024 §2, §3, §18). Track the two coupled "
    "frontiers: COHERENCE (lowest complete EBPW that still generates coherently) "
    "and EXECUTION (lowest COMPLETE_TOKEN_NS among coherent candidates). Density "
    "ladder 2.25..0.1 are coordinates, not pass literals."
)

MEASURED = "MEASURED"
DERIVED = "DERIVED"
ABSENT = "ABSENT"
CITED = "CITED"

# S024 §2 coordinates. Annotations, not obligations.
DENSITY_LADDER = (2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.53, 0.4, 0.25, 0.1)
LADDER_MATCH_TOL = 0.02

RUNGS = (
    "local_functional_probe",
    "held_out_activation",
    "adjacent_layers",
    "short_chain",
    "complete_organ",
    "complete_token",
    "coherent_generation",
    "capability",
)
RUNG_INDEX = {r: i for i, r in enumerate(RUNGS)}

REQUIRED_INPUTS = (
    "receipts/headless/BYTES_FRONTIER.json",
    "receipts/headless/SHARED_BASIS_KERNEL.json",
    "receipts/headless/DISPATCH_LEDGER.json",
    "receipts/headless/ORGAN_ROOF_LEDGER.json",
)

OPTIONAL_KNOWN = (
    "receipts/headless/COMPOSITION_LADDER.json",
    "receipts/headless/NATIVE_2BIT_MLP.json",
    "receipts/headless/Q2F_G64_GENERATION.json",
    "receipts/headless/FIRST_NOETIC_EXECUTABLE.json",
    "receipts/headless/ORGAN_FRONTIERS.json",
    "receipts/headless/ORGAN_LIBRARY.json",
    "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json",
    "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json",
    "receipts/headless/FRACTIONAL_BIT_CANON.json",
    "receipts/headless/C1SHAREDBASIS_DESIGN.json",
    "receipts/headless/C3LOWRANKSPARSE_DESIGN.json",
    "receipts/headless/ONEBIT_FAMILIES.json",
    "receipts/headless/ORGAN_DENSITY_FLOORS.json",
)

# N035 / N036 seal names. Also glob variants if they land under another suffix.
OPTIONAL_FRONTIER_GLOBS = (
    "SHARED_BASIS_COHERENT*.json",
    "BINARY_HEALING*.json",
)

PATHY = re.compile(
    r"^(receipts/|workspace/|docs/|crates/|tools/|lab/|hcli/|hawking-experiments/)"
)
CITE_KEYS = {
    "source",
    "receipt",
    "receipt_ref",
    "shader_path",
    "shader",
    "citations",
    "source_receipt",
}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def rel(path: Path) -> str:
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def git_exists(rel_path: str) -> bool:
    rel_path = rel_path.lstrip("./")
    r = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{rel_path}"],
        cwd=REPO,
        capture_output=True,
        timeout=20,
    )
    return r.returncode == 0


def citation_exists(rel_path: str) -> bool:
    """A sparse-missing file is not evidence it does not exist — check git."""
    rel_path = rel_path.lstrip("./")
    if (REPO / rel_path).is_file():
        return True
    return git_exists(rel_path)


def load_json(rel_path: str) -> dict[str, Any]:
    rel_path = rel_path.lstrip("./")
    p = REPO / rel_path
    if p.is_file():
        return json.loads(p.read_text())
    r = subprocess.run(
        ["git", "show", f"HEAD:{rel_path}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode != 0:
        raise FileNotFoundError(rel_path)
    return json.loads(r.stdout)


def load_json_optional(rel_path: str) -> dict[str, Any] | None:
    try:
        return load_json(rel_path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def qty(
    value: Any,
    *,
    kind: str,
    unit: str,
    command: str,
    source: str,
    absent_reason: str | None = None,
    note: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "value": value,
        "kind": kind,
        "unit": unit,
        "command": command,
        "source": source,
        "absent_reason": absent_reason,
    }
    if note is not None:
        out["note"] = note
    if extra:
        out.update(extra)
    return out


def absent(unit: str, command: str, reason: str, source: str = "") -> dict[str, Any]:
    return qty(
        None,
        kind=ABSENT,
        unit=unit,
        command=command,
        source=source,
        absent_reason=reason,
    )


def numeric(blob: Any) -> float | None:
    if isinstance(blob, dict):
        if blob.get("kind") == ABSENT:
            return None
        v = blob.get("value")
        if isinstance(v, bool) or v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return None
    if isinstance(blob, bool) or blob is None:
        return None
    if isinstance(blob, (int, float)):
        return float(blob)
    return None


def ns_to_ms(ns: float | None) -> float | None:
    if ns is None:
        return None
    return ns / 1e6


# ---------------------------------------------------------------------------
# citations
# ---------------------------------------------------------------------------


def expand_sources(raw: str | None) -> list[str]:
    """Split 'A.json + B.json' blobs used by older receipts into real paths."""
    if not raw:
        return []
    parts = [p.strip() for p in re.split(r"\s+\+\s+", str(raw)) if p.strip()]
    out: list[str] = []
    for p in parts:
        p = p.lstrip("./")
        if p.endswith(".json") and "/" not in p:
            p = f"receipts/headless/{p}"
        if PATHY.match(p):
            out.append(p)
    return out


def iter_citations(obj: Any, acc: list[str] | None = None) -> list[str]:
    acc = acc if acc is not None else []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "citations" and isinstance(v, list):
                for item in v:
                    if isinstance(item, str) and PATHY.match(item):
                        acc.append(item)
            elif k in CITE_KEYS and isinstance(v, str) and PATHY.match(v):
                acc.append(v)
            else:
                iter_citations(v, acc)
    elif isinstance(obj, list):
        for x in obj:
            iter_citations(x, acc)
    return acc


def unique_citations(obj: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in iter_citations(obj):
        c = c.lstrip("./")
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def unresolved_citations(obj: Any) -> list[str]:
    return [c for c in unique_citations(obj) if not citation_exists(c)]


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


class Sources:
    def __init__(self) -> None:
        self.required: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for p in REQUIRED_INPUTS:
            try:
                self.required[p] = load_json(p)
            except FileNotFoundError:
                missing.append(p)
        if missing:
            raise FileNotFoundError(
                "N037 required receipts missing (on disk and in git): " + ", ".join(missing)
            )
        self.bytes_frontier = self.required["receipts/headless/BYTES_FRONTIER.json"]
        self.shared_basis = self.required["receipts/headless/SHARED_BASIS_KERNEL.json"]
        self.dispatch = self.required["receipts/headless/DISPATCH_LEDGER.json"]
        self.organ_roof = self.required["receipts/headless/ORGAN_ROOF_LEDGER.json"]

        self.optional: dict[str, dict[str, Any] | None] = {}
        for p in OPTIONAL_KNOWN:
            self.optional[p] = load_json_optional(p) if citation_exists(p) else None

        self.healing: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()
        for pattern in OPTIONAL_FRONTIER_GLOBS:
            for path in sorted(HEADLESS.glob(pattern)):
                if path.name.startswith("_"):
                    continue
                rel_p = rel(path)
                if rel_p in seen:
                    continue
                seen.add(rel_p)
                try:
                    self.healing.append((rel_p, json.loads(path.read_text())))
                except (json.JSONDecodeError, OSError):
                    continue
            # git-only matches (sparse): known seal names
        for name in ("SHARED_BASIS_COHERENT.json", "BINARY_HEALING.json"):
            rel_p = f"receipts/headless/{name}"
            if rel_p in seen:
                continue
            if citation_exists(rel_p):
                doc = load_json_optional(rel_p)
                if doc is not None:
                    self.healing.append((rel_p, doc))
                    seen.add(rel_p)

    def opt(self, name: str) -> dict[str, Any] | None:
        return self.optional.get(f"receipts/headless/{name}.json")

    def bytes_rep(self, rid: str) -> dict[str, Any] | None:
        for r in self.bytes_frontier.get("representations") or []:
            if r.get("id") == rid:
                return r
        return None


def input_row(path: str, present: bool, *, required: bool, role: str) -> dict[str, Any]:
    return {
        "path": path,
        "present": present,
        "required": required,
        "role": role,
        "absent_reason": None
        if present
        else (
            f"{path} is not on disk or in git"
            if required
            else f"{path} not sealed yet; fields stay ABSENT (not cited)"
        ),
    }


# ---------------------------------------------------------------------------
# candidate construction
# ---------------------------------------------------------------------------


def composition_from_bytes_coherence(coh: dict[str, Any], fallback_src: str) -> dict[str, Any]:
    srcs = expand_sources(coh.get("source_receipt")) or [fallback_src]
    died = coh.get("died_at")
    unreached = coh.get("unreached_above")
    rung = coh.get("rung") or coh.get("highest_rung_reached")
    status = coh.get("status")
    return {
        "highest_rung_reached": rung,
        "highest_rung_index": RUNG_INDEX.get(rung, -1) if isinstance(rung, str) else -1,
        "status": status,
        "died_at": died,
        "unreached_above": unreached,
        "may_be_described_as": rung,
        "why": coh.get("why"),
        "source": srcs[0],
        "citations": srcs,
    }


def coherent_yes(comp: dict[str, Any]) -> tuple[bool, str]:
    """Coherent means the candidate reached coherent_generation without dying there.

    Capability is UNREACHED for everyone (COMPOSITION_LADDER). A death at or
    before coherent_generation is not coherent. An untested rung is not a pass.
    """
    reached = comp.get("highest_rung_reached")
    died = comp.get("died_at")
    status = comp.get("status")
    why = comp.get("why") or ""
    if died:
        return False, (
            f"died_at {died}" + (f": {why}" if why else "")
        )
    if status == "FAILED":
        return False, why or "composition status FAILED"
    if reached == "coherent_generation" and status in {
        "UNTESTED_ABOVE",
        "PASSED_ALL_TESTED",
    }:
        return True, why or "reached coherent_generation"
    return False, (
        f"highest rung is {reached!r} (status {status!r}); "
        "coherent_generation not reached"
    )


def token_ns_from_composed(
    composed_parent: dict[str, Any],
    *,
    source: str,
    command: str,
    note: Any = None,
) -> dict[str, Any]:
    ct = composed_parent.get("COMPLETE_TOKEN_NS") or {}
    composed = ct.get("composed") or {}
    value = composed.get("complete_token_ns")
    if value is None and isinstance(ct.get("median"), (int, float)):
        value = ct.get("median")
    mlp = ct.get("mlp_graph_gpu_ns") or {}
    extra: dict[str, Any] = {
        "ms": ns_to_ms(float(value)) if isinstance(value, (int, float)) else None,
        "composed_kind": composed.get("kind"),
        "mlp_graph_gpu_ns": {
            "n": mlp.get("n"),
            "min": mlp.get("min"),
            "median": mlp.get("median"),
            "max": mlp.get("max"),
        }
        if mlp
        else None,
        "non_mlp_ns": composed.get("non_mlp_ns"),
        "reps": ct.get("reps") or mlp.get("n"),
    }
    if value is None:
        return absent("ns/token", command, "source receipt has no composed complete_token_ns", source)
    return qty(
        int(value) if float(value).is_integer() else float(value),
        kind=CITED,
        unit="ns/token",
        command=command,
        source=source,
        note=note,
        extra=extra,
    )


def density_coordinate(bpw: float | None) -> float | None:
    if bpw is None:
        return None
    best = min(DENSITY_LADDER, key=lambda c: abs(c - bpw))
    if abs(best - bpw) <= LADDER_MATCH_TOL:
        return best
    return None


def place_flags(c: dict[str, Any]) -> None:
    c["on_coherence_frontier"] = False
    c["on_execution_frontier"] = False
    c["occupies_density_coordinate"] = density_coordinate(numeric(c.get("complete_ebpw")))


def candidate_shell(
    *,
    cid: str,
    name: str,
    family: str,
    complete_ebpw: dict[str, Any],
    active_bytes_per_token: dict[str, Any],
    complete_token_ns: dict[str, Any],
    composition_rung: dict[str, Any],
    source_receipt: str,
    citations: list[str],
    dram_bytes_per_token: dict[str, Any] | None = None,
    whole_model_complete_ebpw: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cites = []
    seen: set[str] = set()
    for x in list(citations) + expand_sources(source_receipt) + list(
        composition_rung.get("citations") or []
    ):
        if x and x not in seen and PATHY.match(x):
            seen.add(x)
            cites.append(x)
    coh, reason = coherent_yes(composition_rung)
    row: dict[str, Any] = {
        "id": cid,
        "name": name,
        "family": family,
        "complete_ebpw": complete_ebpw,
        "active_bytes_per_token": active_bytes_per_token,
        "dram_bytes_per_token": dram_bytes_per_token
        or absent(
            "bytes/token",
            "dram_bytes_per_token",
            "not billed on this candidate",
            source_receipt,
        ),
        "COMPLETE_TOKEN_NS": complete_token_ns,
        "composition_rung": composition_rung,
        "coherent": coh,
        "coherent_reason": reason,
        "source_receipt": source_receipt,
        "citations": cites,
        "whole_model_complete_ebpw": whole_model_complete_ebpw
        or absent(
            "bpw",
            "whole_model_complete_ebpw",
            "whole-model mix EBPW not measured for this operating point",
            source_receipt,
        ),
    }
    if extra:
        row.update(extra)
    place_flags(row)
    return row


def from_bytes_rep(
    src: Sources,
    *,
    rid: str,
    cid: str,
    family: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = "receipts/headless/BYTES_FRONTIER.json"
    rep = src.bytes_rep(rid)
    if not rep:
        shell = candidate_shell(
            cid=cid,
            name=rid,
            family=family,
            complete_ebpw=absent("bpw", f"BYTES_FRONTIER.representations[{rid}].active_bpw", f"{rid} missing from BYTES_FRONTIER", source),
            active_bytes_per_token=absent("bytes/token", f"BYTES_FRONTIER.representations[{rid}].active_bytes_per_token", f"{rid} missing from BYTES_FRONTIER", source),
            complete_token_ns=absent("ns/token", f"BYTES_FRONTIER.representations[{rid}].COMPLETE_TOKEN_NS", f"{rid} missing from BYTES_FRONTIER", source),
            composition_rung={
                "highest_rung_reached": None,
                "status": ABSENT,
                "died_at": None,
                "unreached_above": None,
                "why": f"{rid} missing from BYTES_FRONTIER",
                "source": source,
                "citations": [source],
            },
            source_receipt=source,
            citations=[source],
            extra=extra,
        )
        return shell
    bpw = rep.get("active_bpw")
    coh = rep.get("coherence") or {}
    cites = [source] + expand_sources(coh.get("source_receipt"))
    return candidate_shell(
        cid=cid,
        name=rep.get("name") or rid,
        family=family,
        complete_ebpw=qty(
            bpw,
            kind=CITED,
            unit="bpw",
            command=f"BYTES_FRONTIER.representations[{rid}].active_bpw",
            source=source,
            note=(
                "Fully-accounted body EBPW of the density-descent representation "
                "(scales counted, no hidden bits). This is the S024 §2 coordinate, "
                "not whole-model mix EBPW."
            ),
            extra={"representation_id": rid},
        ),
        active_bytes_per_token=qty(
            rep.get("active_bytes_per_token"),
            kind=CITED,
            unit="bytes/token",
            command=f"BYTES_FRONTIER.representations[{rid}].active_bytes_per_token",
            source=source,
        ),
        dram_bytes_per_token=qty(
            rep.get("dram_bytes_per_token"),
            kind=CITED,
            unit="bytes/token",
            command=f"BYTES_FRONTIER.representations[{rid}].dram_bytes_per_token",
            source=source,
        ),
        complete_token_ns=token_ns_from_composed(
            rep,
            source=source,
            command=(
                f"BYTES_FRONTIER.representations[{rid}]"
                ".COMPLETE_TOKEN_NS.composed.complete_token_ns"
            ),
            note=rep.get("notes"),
        ),
        composition_rung=composition_from_bytes_coherence(coh, source),
        source_receipt=source,
        citations=cites,
        extra={
            "bytes_frontier_id": rid,
            "lower_than_q2f_2_25": bool(rep.get("lower_than_q2f_2_25")),
            "dense_w_materialized": rep.get("dense_w_materialized", 0),
            "parity_ok": (rep.get("parity") or {}).get("ok"),
            "toward_roof_729_7": rep.get("toward_roof_729_7"),
            **(extra or {}),
        },
    )


def overlay_q2f(src: Sources, c: dict[str, Any]) -> dict[str, Any]:
    """q2f complete-token spread lives on N021; generation coherence on NATIVE + ladder."""
    native = src.opt("NATIVE_2BIT_MLP")
    q2f_gen = src.opt("Q2F_G64_GENERATION")
    ladder = src.opt("COMPOSITION_LADDER")
    cites = list(c["citations"])

    if native:
        src_n = "receipts/headless/NATIVE_2BIT_MLP.json"
        cites.append(src_n)
        arms = ((native.get("complete_token_bench") or {}).get("arms")) or []
        prod = next((a for a in arms if a.get("role") == "production"), None) or (
            arms[0] if arms else None
        )
        gpu = ((prod or {}).get("bench") or {}).get("gpu_ns") or {}
        if gpu.get("median") is not None:
            prior_ns = numeric(c.get("COMPLETE_TOKEN_NS"))
            c["COMPLETE_TOKEN_NS"] = qty(
                gpu["median"],
                kind=CITED,
                unit="ns/token",
                command=(
                    "NATIVE_2BIT_MLP.complete_token_bench.arms[production_fused_swiglu_geo]"
                    ".bench.gpu_ns.median"
                ),
                source=src_n,
                note=(
                    "N021 fused complete-token GPU median, 7 reps. BYTES_FRONTIER "
                    "reuses this as the q2f composed complete_token_ns."
                ),
                extra={
                    "ms": ns_to_ms(float(gpu["median"])),
                    "spread": {
                        "n": gpu.get("n"),
                        "min": gpu.get("min"),
                        "median": gpu.get("median"),
                        "max": gpu.get("max"),
                    },
                    "timing_label": ((prod or {}).get("bench") or {}).get("timing_label"),
                    "dispatches": ((prod or {}).get("bench") or {}).get("dispatches"),
                    "bytes_frontier_composed_agrees": prior_ns == float(gpu["median"]),
                },
            )
        dec = native.get("decode") or {}
        coh = dec.get("coherence") or {}
        if coh.get("coherent") is True:
            c["composition_rung"] = {
                "highest_rung_reached": "coherent_generation",
                "highest_rung_index": RUNG_INDEX["coherent_generation"],
                "status": "UNTESTED_ABOVE",
                "died_at": None,
                "unreached_above": "capability",
                "may_be_described_as": "coherent_generation",
                "why": (
                    (native.get("composition_ladder_rung") or {}).get("why")
                    or coh.get("reason")
                    or "native generation is non-degenerate"
                ),
                "source": src_n,
                "citations": [src_n, "receipts/headless/BYTES_FRONTIER.json"],
            }
        c["dense_w_materialized"] = native.get("dense_w_materialized", c.get("dense_w_materialized"))

    if q2f_gen:
        src_q = "receipts/headless/Q2F_G64_GENERATION.json"
        cites.append(src_q)
        for row in q2f_gen.get("comparison") or []:
            if "q2_4level" in str(row.get("codec") or ""):
                c["whole_model_complete_ebpw"] = qty(
                    row.get("complete_ebpw"),
                    kind=CITED,
                    unit="bpw",
                    command="Q2F_G64_GENERATION.comparison[q2_4level_fitted_g64].complete_ebpw",
                    source=src_q,
                    note=(
                        "Whole-model mix including Q4 attention/embed. The S024 "
                        "density coordinate is the 2.25 body EBPW, not this mix."
                    ),
                    extra={"body_bpw": row.get("body_bpw")},
                )
                break

    if ladder:
        src_l = "receipts/headless/COMPOSITION_LADDER.json"
        cites.append(src_l)
        for row in ladder.get("candidates") or []:
            if "q2_4level_fitted_g64" in str(row.get("id") or ""):
                c["composition_ladder_entry"] = {
                    "id": row.get("id"),
                    "highest_rung_reached": row.get("highest_rung_reached"),
                    "status": row.get("status"),
                    "died_at": row.get("died_at"),
                    "unreached_above": row.get("unreached_above"),
                    "source": src_l,
                }
                break

    c["citations"] = _dedupe(cites)
    coh, reason = coherent_yes(c["composition_rung"])
    c["coherent"] = coh
    c["coherent_reason"] = reason
    place_flags(c)
    return c


def overlay_binary(src: Sources, c: dict[str, Any]) -> dict[str, Any]:
    fne = src.opt("FIRST_NOETIC_EXECUTABLE")
    ladder = src.opt("COMPOSITION_LADDER")
    cites = list(c["citations"])
    if fne:
        src_f = "receipts/headless/FIRST_NOETIC_EXECUTABLE.json"
        cites.append(src_f)
        for m in fne.get("mix_scoreboard") or []:
            if m.get("mix_id") == "mix_c_all_mlp_binary_g64":
                c["whole_model_complete_ebpw"] = qty(
                    m.get("complete_ebpw"),
                    kind=CITED,
                    unit="bpw",
                    command="FIRST_NOETIC_EXECUTABLE.mix_scoreboard[mix_c_all_mlp_binary_g64].complete_ebpw",
                    source=src_f,
                    note=(
                        "Whole-model mix EBPW of mix_c. Body is 1.25 bpw; generation "
                        "degenerated (16 copies of token 271). Injured, not a coherent frontier."
                    ),
                    extra={
                        "mix_id": m.get("mix_id"),
                        "mix_coherent": bool(m.get("coherent")),
                        "tok_s": m.get("tok_s"),
                    },
                )
                break
    if ladder:
        src_l = "receipts/headless/COMPOSITION_LADDER.json"
        cites.append(src_l)
        for row in ladder.get("candidates") or []:
            if row.get("id") == "mix_c_all_mlp_binary_g64":
                c["composition_ladder_entry"] = {
                    "id": row.get("id"),
                    "highest_rung_reached": row.get("highest_rung_reached"),
                    "status": row.get("status"),
                    "died_at": row.get("died_at"),
                    "why": row.get("why"),
                    "source": src_l,
                }
                break
    c["citations"] = _dedupe(cites)
    c["s024_binary_is_injured_not_dead"] = True
    place_flags(c)
    return c


def overlay_ternary(src: Sources, c: dict[str, Any]) -> dict[str, Any]:
    cites = list(c["citations"])
    for name in (
        "NOETIC_COMPOSITION_WHOLEMODEL_TERNARY",
        "FRACTIONAL_BIT_CANON",
        "COMPOSITION_LADDER",
    ):
        if src.opt(name):
            cites.append(f"receipts/headless/{name}.json")
    c["citations"] = _dedupe(cites)
    place_flags(c)
    return c


def overlay_shared_k2(src: Sources, c: dict[str, Any]) -> dict[str, Any]:
    """N033 fused kernel supersedes the N032 two-pass COMPLETE_TOKEN_NS."""
    sbk = src.shared_basis
    source = "receipts/headless/SHARED_BASIS_KERNEL.json"
    cites = list(c["citations"]) + [source]
    after = sbk.get("after") or {}
    before = sbk.get("before") or {}
    ns = token_ns_from_composed(
        after,
        source=source,
        command="SHARED_BASIS_KERNEL.after.COMPLETE_TOKEN_NS.composed.complete_token_ns",
        note=(
            "N033 fused competent kernel (192 dispatches). N032 two-pass 384-dispatch "
            "path is the superseded kernel, not this operating point's execution."
        ),
    )
    c["COMPLETE_TOKEN_NS"] = ns
    c["active_bpw_cited"] = qty(
        sbk.get("active_bpw"),
        kind=CITED,
        unit="bpw",
        command="SHARED_BASIS_KERNEL.active_bpw",
        source=source,
    )
    if sbk.get("active_bytes_per_token") is not None:
        c["active_bytes_per_token"] = qty(
            sbk.get("active_bytes_per_token"),
            kind=CITED,
            unit="bytes/token",
            command="SHARED_BASIS_KERNEL.active_bytes_per_token",
            source=source,
        )
    if sbk.get("dram_bytes_per_token") is not None:
        c["dram_bytes_per_token"] = qty(
            sbk.get("dram_bytes_per_token"),
            kind=CITED,
            unit="bytes/token",
            command="SHARED_BASIS_KERNEL.dram_bytes_per_token",
            source=source,
        )
    ladder = sbk.get("composition_ladder") or {}
    c["composition_rung"] = {
        "highest_rung_reached": (sbk.get("composition") or {}).get("highest_rung_reached")
        or ladder.get("rung"),
        "highest_rung_index": RUNG_INDEX.get(
            (sbk.get("composition") or {}).get("highest_rung_reached") or ladder.get("rung") or "",
            -1,
        ),
        "status": ladder.get("status"),
        "died_at": ladder.get("died_at") or (sbk.get("composition") or {}).get("died_at"),
        "unreached_above": (sbk.get("composition") or {}).get("unreached_above"),
        "may_be_described_as": (sbk.get("composition") or {}).get("highest_rung_reached")
        or ladder.get("rung"),
        "why": ladder.get("why"),
        "source": source,
        "citations": [source],
    }
    c["kernel_competent"] = bool(sbk.get("competent"))
    c["byte_win_translates_to_token_ns"] = bool(sbk.get("byte_win_translates_to_token_ns"))
    c["superseded_two_pass"] = {
        "id": before.get("id"),
        "dispatches": before.get("dispatches"),
        "complete_token_ns": ((before.get("COMPLETE_TOKEN_NS") or {}).get("composed") or {}).get(
            "complete_token_ns"
        ),
        "source": source,
        "note": "Incompetent two-pass kernel (N032). Not the K2 execution number.",
    }
    c["fused_dispatches"] = after.get("dispatches")
    c["toward_roof_729_7"] = sbk.get("toward_roof_729_7")
    c["dense_w_materialized"] = sbk.get("dense_w")
    c["k"] = 2
    coh, reason = coherent_yes(c["composition_rung"])
    c["coherent"] = coh
    c["coherent_reason"] = reason
    c["source_receipt"] = source
    c["citations"] = _dedupe(cites)
    # complete EBPW stays the 0.53125 body figure from bytes/SBK.
    if numeric(c.get("complete_ebpw")) is None and sbk.get("active_bpw") is not None:
        c["complete_ebpw"] = qty(
            sbk.get("active_bpw"),
            kind=CITED,
            unit="bpw",
            command="SHARED_BASIS_KERNEL.active_bpw",
            source=source,
            note="K=2 signs amortized over 64 layers + f16 scales. Coefficient cost does not vanish.",
        )
    place_flags(c)
    return c


def shared_k8(src: Sources) -> dict[str, Any]:
    source = "receipts/headless/SHARED_BASIS_KERNEL.json"
    sbk = src.shared_basis
    curve = sbk.get("k_tradeoff_curve") or (sbk.get("composition") or {}).get("curve") or []
    row = next((r for r in curve if r.get("k") == 8), None)
    k2_bytes = sbk.get("active_bytes_per_token")
    k2_bpw = sbk.get("active_bpw")
    k2_dram = sbk.get("dram_bytes_per_token")

    if row is None:
        return candidate_shell(
            cid="shared_basis_k8",
            name="shared binary K=8",
            family="shared_basis",
            complete_ebpw=absent(
                "bpw",
                "SHARED_BASIS_KERNEL.k_tradeoff_curve[k=8]",
                "K=8 row missing from SHARED_BASIS_KERNEL k_tradeoff_curve",
                source,
            ),
            active_bytes_per_token=absent(
                "bytes/token",
                "k8 active bytes",
                "K=8 row missing",
                source,
            ),
            complete_token_ns=absent(
                "ns/token",
                "SHARED_BASIS_KERNEL K=8 COMPLETE_TOKEN_NS",
                "N033 timed K=2 only; K=8 token_ns was not measured",
                source,
            ),
            composition_rung={
                "highest_rung_reached": None,
                "status": ABSENT,
                "died_at": None,
                "why": "K=8 row missing",
                "source": source,
                "citations": [source],
            },
            source_receipt=source,
            citations=[source],
            extra={"k": 8},
        )

    # 2.125 bpw is a 64-layer counterfactual. Do not present it as a sealed complete EBPW.
    complete = absent(
        "bpw",
        "SHARED_BASIS_KERNEL.k_tradeoff_curve[k=8].counterfactual_64_layer_bpw",
        (
            "64-layer K=8 complete EBPW is untested. The curve bills 6.0 bpw over the "
            "2 layers actually fitted; 2.125 bpw is a counterfactual amortization of "
            "those same bases over 64 layers. Adding layers at fixed K is a tighter "
            "constraint (SHARED_BASIS_KERNEL.composition.reading)."
        ),
        source,
    )
    complete["counterfactual_64_layer_bpw"] = qty(
        row.get("counterfactual_64_layer_bpw"),
        kind=DERIVED,
        unit="bpw",
        command="SHARED_BASIS_KERNEL.k_tradeoff_curve[k=8].counterfactual_64_layer_bpw",
        source=source,
        note="Not a sealed 64-layer operating point.",
    )
    complete["tested_2_layer_storage_bpw"] = qty(
        row.get("storage_bpw_over_2_layers"),
        kind=CITED,
        unit="bpw",
        command="SHARED_BASIS_KERNEL.k_tradeoff_curve[k=8].storage_bpw_over_2_layers",
        source=source,
    )

    if isinstance(k2_bytes, (int, float)) and isinstance(k2_bpw, (int, float)):
        active = qty(
            float(k2_bytes) * (8.0 / 2.0),
            kind=DERIVED,
            unit="bytes/token",
            command="SHARED_BASIS_KERNEL.active_bytes_per_token * (K=8 / K=2)",
            source=source,
            note="Linear in K under the N033 fused billing (signs + f16 scales).",
        )
        dram = (
            qty(
                float(k2_dram) + float(k2_bytes) * 3.0,
                kind=DERIVED,
                unit="bytes/token",
                command="k2_dram + (k8_active - k2_active)",
                source=source,
                note="Q4 attention/f32 remainder held fixed; only the shared-basis payload scales with K.",
            )
            if isinstance(k2_dram, (int, float))
            else absent("bytes/token", "k8 dram", "K=2 dram missing", source)
        )
    else:
        active = absent("bytes/token", "k8 active", "K=2 active bytes missing; cannot scale to K=8", source)
        dram = absent("bytes/token", "k8 dram", "K=2 dram missing", source)

    healthy = bool(row.get("healthy"))
    comp = {
        "highest_rung_reached": "held_out_activation" if healthy else "local_functional_probe",
        "highest_rung_index": RUNG_INDEX["held_out_activation"] if healthy else RUNG_INDEX["local_functional_probe"],
        "status": "UNTESTED_ABOVE" if healthy else "FAILED",
        "died_at": None if healthy else "held_out_activation",
        "unreached_above": "adjacent_layers",
        "may_be_described_as": "held_out_activation" if healthy else "local_functional_probe",
        "why": (
            "First healthy K on the 2-layer gate L30/L31 curve "
            f"(rel_fro={row.get('mean_rel_fro')}, gain={row.get('mean_gain')}). "
            "64-layer composition and generation are UNREACHED."
        ),
        "source": source,
        "citations": [source],
        "n_layers_fitted": row.get("n_layers_fitted"),
        "mean_rel_fro": row.get("mean_rel_fro"),
        "mean_gain": row.get("mean_gain"),
        "healthy_2_layer": healthy,
    }
    return candidate_shell(
        cid="shared_basis_k8",
        name="shared binary K=8 (2-layer healthy; 64-layer untested)",
        family="shared_basis",
        complete_ebpw=complete,
        active_bytes_per_token=active,
        dram_bytes_per_token=dram,
        complete_token_ns=absent(
            "ns/token",
            "SHARED_BASIS_KERNEL K=8 COMPLETE_TOKEN_NS",
            "N033 timed the fused K=2 kernel only. K=8 COMPLETE_TOKEN_NS is unmeasured; not scaled from K=2.",
            source,
        ),
        composition_rung=comp,
        source_receipt=source,
        citations=[source],
        extra={
            "k": 8,
            "kernel_competent": bool(sbk.get("competent")),
            "k_tradeoff_row": {
                "k": 8,
                "n_layers_fitted": row.get("n_layers_fitted"),
                "storage_bpw_over_2_layers": row.get("storage_bpw_over_2_layers"),
                "counterfactual_64_layer_bpw": row.get("counterfactual_64_layer_bpw"),
                "mean_rel_fro": row.get("mean_rel_fro"),
                "mean_gain": row.get("mean_gain"),
                "healthy": healthy,
                "source": source,
            },
            "dense_w_materialized": sbk.get("dense_w"),
        },
    )


def overlay_csr(src: Sources, c: dict[str, Any]) -> dict[str, Any]:
    cites = list(c["citations"])
    if src.opt("C3LOWRANKSPARSE_DESIGN"):
        cites.append("receipts/headless/C3LOWRANKSPARSE_DESIGN.json")
    if src.opt("FIRST_NOETIC_EXECUTABLE"):
        cites.append("receipts/headless/FIRST_NOETIC_EXECUTABLE.json")
    c["citations"] = _dedupe(cites)
    place_flags(c)
    return c


def _qty_from_unknown(blob: Any, unit: str, command: str, source: str) -> dict[str, Any]:
    if isinstance(blob, dict) and "kind" in blob:
        out = dict(blob)
        if not out.get("source"):
            out["source"] = source
        return out
    if isinstance(blob, (int, float)) and not isinstance(blob, bool):
        return qty(blob, kind=CITED, unit=unit, command=command, source=source)
    return absent(unit, command, "field not present on optional receipt", source)


def ingest_optional_point(
    item: dict[str, Any], source: str, *, cid: str, family: str, name: str
) -> dict[str, Any]:
    ebpw_blob = (
        item.get("complete_ebpw")
        if item.get("complete_ebpw") is not None
        else item.get("active_bpw")
        if item.get("active_bpw") is not None
        else item.get("ebpw")
    )
    ns_blob = item.get("COMPLETE_TOKEN_NS")
    if ns_blob is None:
        ns_blob = item.get("complete_token_ns")
    bytes_blob = item.get("active_bytes_per_token")
    rung = item.get("composition_rung") or item.get("composition_ladder") or {}
    if not isinstance(rung, dict):
        rung = {"highest_rung_reached": rung, "source": source, "citations": [source]}
    else:
        rung = dict(rung)
        rung.setdefault("source", source)
        rung.setdefault("citations", [source])
        if "highest_rung_reached" not in rung and "rung" in rung:
            rung["highest_rung_reached"] = rung.get("rung")
    if item.get("coherent") is True and not rung.get("highest_rung_reached"):
        rung["highest_rung_reached"] = "coherent_generation"
        rung["status"] = item.get("status") or "UNTESTED_ABOVE"
        rung["died_at"] = None
    if item.get("coherent") is False and not rung.get("died_at"):
        rung.setdefault("status", "FAILED")

    if isinstance(ns_blob, dict) and ns_blob.get("composed"):
        ns = token_ns_from_composed(item if "COMPLETE_TOKEN_NS" in item else {"COMPLETE_TOKEN_NS": ns_blob}, source=source, command=f"{source} COMPLETE_TOKEN_NS")
    else:
        ns = _qty_from_unknown(ns_blob, "ns/token", f"{Path(source).name} COMPLETE_TOKEN_NS", source)
        if numeric(ns) is not None and ns.get("ms") is None:
            ns["ms"] = ns_to_ms(numeric(ns))

    extra = {}
    if item.get("COHERENCE_TAX_EBPW") is not None:
        extra["COHERENCE_TAX_EBPW"] = _qty_from_unknown(
            item.get("COHERENCE_TAX_EBPW"), "bpw", "COHERENCE_TAX_EBPW", source
        )
    if item.get("heal_kind") is not None:
        extra["heal_kind"] = item.get("heal_kind")
    return candidate_shell(
        cid=cid,
        name=str(item.get("name") or item.get("id") or name),
        family=str(item.get("family") or family),
        complete_ebpw=_qty_from_unknown(ebpw_blob, "bpw", "complete_ebpw", source),
        active_bytes_per_token=_qty_from_unknown(
            bytes_blob, "bytes/token", "active_bytes_per_token", source
        ),
        complete_token_ns=ns,
        composition_rung=rung,
        source_receipt=source,
        citations=[source],
        extra=extra,
    )


def ingest_healing_receipt(path: str, doc: dict[str, Any]) -> list[dict[str, Any]]:
    family = (
        "binary_heal"
        if "BINARY_HEALING" in path
        else "shared_basis_coherent"
        if "SHARED_BASIS_COHERENT" in path
        else "optional_frontier"
    )
    stem = Path(path).stem.lower()
    points: list[dict[str, Any]] = []
    for key in ("heals", "candidates", "points", "operating_points", "coherent_points"):
        blob = doc.get(key)
        if isinstance(blob, list) and blob:
            for i, item in enumerate(blob):
                if not isinstance(item, dict):
                    continue
                cid = str(item.get("id") or f"{stem}_{i}")
                points.append(
                    ingest_optional_point(
                        item, path, cid=cid, family=family, name=f"{stem}[{i}]"
                    )
                )
            break
    if not points:
        # Whole receipt as one point if it carries any of the frontier fields.
        if any(k in doc for k in ("complete_ebpw", "active_bpw", "COMPLETE_TOKEN_NS", "complete_token_ns")):
            points.append(
                ingest_optional_point(doc, path, cid=stem, family=family, name=stem)
            )
        else:
            points.append(
                candidate_shell(
                    cid=stem,
                    name=stem,
                    family=family,
                    complete_ebpw=absent(
                        "bpw",
                        "complete_ebpw",
                        f"{path} is present but has no complete_ebpw/active_bpw to map",
                        path,
                    ),
                    active_bytes_per_token=absent(
                        "bytes/token",
                        "active_bytes_per_token",
                        f"{path} has no active_bytes_per_token to map",
                        path,
                    ),
                    complete_token_ns=absent(
                        "ns/token",
                        "COMPLETE_TOKEN_NS",
                        f"{path} has no COMPLETE_TOKEN_NS to map",
                        path,
                    ),
                    composition_rung={
                        "highest_rung_reached": None,
                        "status": ABSENT,
                        "died_at": None,
                        "why": "optional frontier receipt present; composition rung not mapped",
                        "source": path,
                        "citations": [path],
                    },
                    source_receipt=path,
                    citations=[path],
                    extra={"receipt_present_unmapped": True},
                )
            )
    return points


def absent_optional_candidate(path: str, cid: str, family: str, why: str) -> dict[str, Any]:
    """Placeholder that does NOT cite the missing path (citation test would fail)."""
    return candidate_shell(
        cid=cid,
        name=cid,
        family=family,
        complete_ebpw=absent("bpw", "complete_ebpw", why, ""),
        active_bytes_per_token=absent("bytes/token", "active_bytes_per_token", why, ""),
        complete_token_ns=absent("ns/token", "COMPLETE_TOKEN_NS", why, ""),
        composition_rung={
            "highest_rung_reached": None,
            "status": ABSENT,
            "died_at": None,
            "unreached_above": None,
            "why": why,
            "source": "",
            "citations": [],
        },
        source_receipt="",
        citations=[],
        extra={
            "optional_receipt": path,
            "optional_receipt_present": False,
            "not_cited_because_missing": True,
        },
    )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# frontiers
# ---------------------------------------------------------------------------


def select_coherence(cands: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = []
    for c in cands:
        bpw = numeric(c.get("complete_ebpw"))
        if c.get("coherent") and bpw is not None:
            eligible.append((bpw, c["id"], c))
    if not eligible:
        return {
            "name": None,
            "candidate_id": None,
            "kind": ABSENT,
            "complete_ebpw": None,
            "absent_reason": "no coherent candidate has a numeric complete EBPW",
            "axis": "lowest_coherent_complete_ebpw",
        }
    eligible.sort(key=lambda t: (t[0], t[1]))
    bpw, cid, c = eligible[0]
    ties = [e[1] for e in eligible if e[0] == bpw]
    c["on_coherence_frontier"] = True
    return {
        "name": cid,
        "candidate_id": cid,
        "display_name": c.get("name"),
        "kind": CITED,
        "axis": "lowest_coherent_complete_ebpw",
        "complete_ebpw": c.get("complete_ebpw"),
        "COMPLETE_TOKEN_NS": c.get("COMPLETE_TOKEN_NS"),
        "active_bytes_per_token": c.get("active_bytes_per_token"),
        "composition_rung": c.get("composition_rung"),
        "coherent": True,
        "source_receipt": c.get("source_receipt"),
        "tied_ids": ties,
        "n_coherent_with_ebpw": len(eligible),
    }


def select_execution(cands: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = []
    for c in cands:
        ns = numeric(c.get("COMPLETE_TOKEN_NS"))
        if c.get("coherent") and ns is not None:
            eligible.append((ns, c["id"], c))
    if not eligible:
        return {
            "name": None,
            "candidate_id": None,
            "kind": ABSENT,
            "COMPLETE_TOKEN_NS": None,
            "absent_reason": "no coherent candidate has a numeric COMPLETE_TOKEN_NS",
            "axis": "lowest_coherent_complete_token_ns",
        }
    eligible.sort(key=lambda t: (t[0], t[1]))
    ns, cid, c = eligible[0]
    ties = [e[1] for e in eligible if e[0] == ns]
    c["on_execution_frontier"] = True
    return {
        "name": cid,
        "candidate_id": cid,
        "display_name": c.get("name"),
        "kind": CITED,
        "axis": "lowest_coherent_complete_token_ns",
        "complete_ebpw": c.get("complete_ebpw"),
        "COMPLETE_TOKEN_NS": c.get("COMPLETE_TOKEN_NS"),
        "ms": ns_to_ms(ns),
        "active_bytes_per_token": c.get("active_bytes_per_token"),
        "composition_rung": c.get("composition_rung"),
        "coherent": True,
        "source_receipt": c.get("source_receipt"),
        "tied_ids": ties,
        "n_coherent_with_token_ns": len(eligible),
    }


def density_ladder(cands: list[dict[str, Any]]) -> dict[str, Any]:
    by_coord: dict[float, list[dict[str, Any]]] = {c: [] for c in DENSITY_LADDER}
    off: list[dict[str, Any]] = []
    for c in cands:
        bpw = numeric(c.get("complete_ebpw"))
        coord = density_coordinate(bpw)
        info = {
            "id": c["id"],
            "complete_ebpw": bpw,
            "coherent": bool(c.get("coherent")),
            "COMPLETE_TOKEN_NS": numeric(c.get("COMPLETE_TOKEN_NS")),
        }
        if coord is None:
            off.append(info)
        else:
            by_coord[coord].append(info)
    coords = []
    for coord in DENSITY_LADDER:
        occ = by_coord[coord]
        if any(o["coherent"] for o in occ):
            status = "REACHED_COHERENT"
        elif occ:
            status = "OCCUPIED_INCOHERENT"
        else:
            status = "UNREACHED"
        coords.append(
            {
                "coordinate_bpw": coord,
                "not_an_obligation": True,
                "not_a_pass_literal": True,
                "status": status,
                "occupied_by": occ,
            }
        )
    return {
        "rule": (
            "S024 §2: 2.25→2.0→1.75→1.5→1.25→1.0→0.75→0.53→0.4→0.25→0.1 are "
            "COORDINATES, not obligations. Occupying a coordinate is not passing it. "
            "A coherent occupant is REACHED_COHERENT; an incoherent occupant is "
            "OCCUPIED_INCOHERENT; an empty coordinate is UNREACHED."
        ),
        "match_tolerance_bpw": LADDER_MATCH_TOL,
        "coordinates": coords,
        "off_ladder": off,
    }


def copy_roofs(src: Sources) -> dict[str, Any]:
    roof = src.organ_roof
    source = "receipts/headless/ORGAN_ROOF_LEDGER.json"
    three = roof.get("three_roofs") or {}
    ranked = roof.get("ranked_by_recoverable_token_ns") or []
    bf = src.bytes_frontier
    return {
        "did_not_rederive": True,
        "source": source,
        "three_roofs": three,
        "ranking_quantity": roof.get("ranking_quantity"),
        "largest_recoverable_organ": roof.get("largest_recoverable_organ"),
        "ranked_by_recoverable_token_ns": ranked,
        "model_reachable_tok_s_from_bytes_frontier": {
            "value": bf.get("roof_tok_s"),
            "kind": CITED,
            "unit": "tok/s",
            "command": "BYTES_FRONTIER.roof_tok_s",
            "source": "receipts/headless/BYTES_FRONTIER.json",
            "absent_reason": None,
            "note": "BYTES_FRONTIER's 729.7 annotation of ORGAN_ROOF_LEDGER MODEL_REACHABLE. Not re-derived.",
        },
        "model_reachable_ns_from_bytes_frontier": {
            "value": bf.get("roof_ns"),
            "kind": CITED,
            "unit": "ns/token",
            "command": "BYTES_FRONTIER.roof_ns",
            "source": "receipts/headless/BYTES_FRONTIER.json",
            "absent_reason": None,
        },
    }


def organ_density(src: Sources) -> dict[str, Any]:
    """S024 §18: whole-model EBPW may stay high while an organ is far denser.

    Floors and recoverable ns are copied, not re-derived.
    """
    source_roof = "receipts/headless/ORGAN_ROOF_LEDGER.json"
    source_fr = "receipts/headless/ORGAN_FRONTIERS.json"
    source_lib = "receipts/headless/ORGAN_LIBRARY.json"
    fr = src.opt("ORGAN_FRONTIERS") or {}
    lib = src.opt("ORGAN_LIBRARY") or {}
    df = src.opt("ORGAN_DENSITY_FLOORS") or {}
    df_organs = df.get("organs") or {}
    lib_by = {o.get("organ"): o for o in (lib.get("organs") or []) if isinstance(o, dict)}
    fr_organs = fr.get("organs") or {}
    mlp_lock = fr.get("mlp_not_extrapolated") or {}

    alias = {"embedding": "embed", "gqa": "gqa_attention"}
    rows = []
    for r in src.organ_roof.get("ranked_by_recoverable_token_ns") or []:
        organ = r.get("organ")
        contract = alias.get(organ, organ)
        floor = None
        fr_key = {
            "deltanet": "deltanet",
            "gqa_attention": "gqa",
            "embedding": "embedding_output",
        }.get(organ)
        if fr_key and isinstance(fr_organs.get(fr_key), dict):
            fl = (fr_organs[fr_key].get("floor") or {})
            floor = {
                "storage_bpw": fl.get("storage_bpw"),
                "active_fused_bpw": fl.get("active_fused_bpw"),
                "status": fl.get("status"),
                "source": source_fr,
            }
        lib_row = lib_by.get(contract) or {}
        df_key = {
            "deltanet": "deltanet",
            "gqa_attention": "gqa_attention",
            "embedding": "embedding_output",
        }.get(organ)
        cfl = ((df_organs.get(df_key) or {}).get("floor") or {}) if df_key else {}
        rows.append(
            {
                "organ": organ,
                "contract_organ": contract,
                "recoverable_token_ns": qty(
                    r.get("recoverable_token_ns"),
                    kind=CITED,
                    unit="ns/token",
                    command=f"ORGAN_ROOF_LEDGER.ranked_by_recoverable_token_ns[{organ}].recoverable_token_ns",
                    source=source_roof,
                ),
                "floor_storage_bpw": (
                    qty(
                        floor["storage_bpw"],
                        kind=CITED,
                        unit="bpw",
                        command=f"ORGAN_FRONTIERS.organs[{fr_key}].floor.storage_bpw",
                        source=source_fr,
                    )
                    if floor and floor.get("storage_bpw") is not None
                    else (
                        lib_row.get("best_complete_ebpw")
                        if isinstance(lib_row.get("best_complete_ebpw"), dict)
                        else absent(
                            "bpw",
                            f"organ {organ} floor",
                            "no ORGAN_FRONTIERS floor and no ORGAN_LIBRARY complete EBPW",
                            source_roof,
                        )
                    )
                ),
                "source": source_roof,
                "composition_floor_complete_ebpw": (
                    qty(
                        cfl.get("complete_ebpw"),
                        kind=CITED,
                        unit="EBPW",
                        command=f"ORGAN_DENSITY_FLOORS.organs[{df_key}].floor.complete_ebpw",
                        source="receipts/headless/ORGAN_DENSITY_FLOORS.json",
                    )
                    if cfl.get("complete_ebpw") is not None
                    else None
                ),
            }
        )
    cites = [source_roof]
    if src.opt("ORGAN_FRONTIERS"):
        cites.append(source_fr)
    if src.opt("ORGAN_LIBRARY"):
        cites.append(source_lib)
    if src.opt("ORGAN_DENSITY_FLOORS"):
        cites.append("receipts/headless/ORGAN_DENSITY_FLOORS.json")
    return {
        "s024": "§17, §18",
        "rule": (
            "Whole-model EBPW may stay >0.5 while an organ hits <<1 because a shared "
            "structure represents many nominal weights. Track per organ; do not collapse."
        ),
        "mlp_bracket_cited_not_transferred": mlp_lock or {
            "note": "ORGAN_FRONTIERS.mlp_not_extrapolated not loaded",
        },
        "organs": rows,
        "citations": cites,
    }


def dispatch_context(src: Sources) -> dict[str, Any]:
    source = "receipts/headless/DISPATCH_LEDGER.json"
    parent = src.dispatch.get("parent") or {}
    reduction = src.dispatch.get("reduction") or {}
    tok = parent.get("tok_s_sealed")
    parent_ns = (1e9 / float(tok)) if isinstance(tok, (int, float)) and tok else None
    return {
        "source": source,
        "note": (
            "Sealed parent execution context. Not a density-descent candidate and not "
            "mixed into candidate COMPLETE_TOKEN_NS (those use the N021/N032/N033 harness)."
        ),
        "parent_complete_ebpw": qty(
            parent.get("complete_ebpw"),
            kind=CITED,
            unit="bpw",
            command="DISPATCH_LEDGER.parent.complete_ebpw",
            source=source,
        ),
        "parent_dispatches": qty(
            parent.get("dispatches"),
            kind=CITED,
            unit="1",
            command="DISPATCH_LEDGER.parent.dispatches",
            source=source,
        ),
        "parent_tok_s_sealed": qty(
            tok,
            kind=CITED,
            unit="tok/s",
            command="DISPATCH_LEDGER.parent.tok_s_sealed",
            source=source,
        ),
        "parent_complete_wall_ns_from_tok_s": (
            qty(
                parent_ns,
                kind=DERIVED,
                unit="ns/token",
                command="1e9 / DISPATCH_LEDGER.parent.tok_s_sealed",
                source=source,
                note="Wall from sealed tok/s. Not the N021 27.55 ms GPU complete-token figure.",
            )
            if parent_ns is not None
            else absent("ns/token", "1e9 / tok_s_sealed", "tok_s_sealed missing", source)
        ),
        "reduction": {
            "from": reduction.get("parent_dispatches") or (src.dispatch.get("no_further_or_the_cut") or {}).get("from"),
            "to": reduction.get("candidate_dispatches") or (src.dispatch.get("no_further_or_the_cut") or {}).get("to"),
            "token_ids_unchanged": reduction.get("token_ids_unchanged"),
            "source": source,
        },
        "did_not_mutate_parent": True,
        "parent_path_present": parent.get("present"),
        "parent_immutable": parent.get("immutable"),
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def build_candidates(src: Sources) -> list[dict[str, Any]]:
    q2f = overlay_q2f(src, from_bytes_rep(src, rid="q2_4level_fitted_g64", cid="q2f_g64", family="q2_affine"))
    q2f["role"] = "coherent_baseline"
    binary = overlay_binary(src, from_bytes_rep(src, rid="binary_g64", cid="binary_g64", family="binary"))
    binary["role"] = "fast_incoherent_parent"
    ternary = overlay_ternary(
        src, from_bytes_rep(src, rid="ternary_5in8_g64", cid="ternary_5in8_g64", family="ternary")
    )
    ternary["role"] = "dead_on_complete_token"
    k2 = overlay_shared_k2(
        src, from_bytes_rep(src, rid="shared_binary_k2", cid="shared_basis_k2", family="shared_basis")
    )
    k2["role"] = "competent_kernel_dead_on_activation"
    k8 = shared_k8(src)
    k8["role"] = "first_healthy_k_on_2_layer_curve"
    csr = overlay_csr(
        src,
        from_bytes_rep(
            src, rid="binary_residual_sparse_2pct", cid="binary_csr_2pct", family="binary_sparse_residual"
        ),
    )
    csr["role"] = "index_carrying_residual"

    out = [q2f, binary, ternary, k2, k8, csr]

    present_healing = {p for p, _ in src.healing}
    for path, doc in src.healing:
        out.extend(ingest_healing_receipt(path, doc))
    for path, cid, family, why in (
        (
            "receipts/headless/SHARED_BASIS_COHERENT.json",
            "shared_basis_coherent",
            "shared_basis_coherent",
            "N035 SHARED_BASIS_COHERENT.json is not sealed. Coherent shared-basis operating point is ABSENT.",
        ),
        (
            "receipts/headless/BINARY_HEALING.json",
            "binary_healing",
            "binary_heal",
            "N036 BINARY_HEALING.json is not sealed. Healed binary points are ABSENT.",
        ),
    ):
        if path not in present_healing and not any(c.get("id") == cid for c in out):
            out.append(absent_optional_candidate(path, cid, family, why))
    return out


def faster_incoherent(cands: list[dict[str, Any]], exec_ns: float | None) -> list[dict[str, Any]]:
    if exec_ns is None:
        return []
    rows = []
    for c in cands:
        ns = numeric(c.get("COMPLETE_TOKEN_NS"))
        if ns is None:
            continue
        if (not c.get("coherent")) and ns < exec_ns:
            rows.append(
                {
                    "id": c["id"],
                    "COMPLETE_TOKEN_NS": ns,
                    "ms": ns_to_ms(ns),
                    "complete_ebpw": numeric(c.get("complete_ebpw")),
                    "why_not_execution_frontier": (
                        "faster than the coherent execution frontier but not coherent "
                        f"({c.get('coherent_reason')})"
                    ),
                    "source_receipt": c.get("source_receipt"),
                }
            )
    rows.sort(key=lambda r: r["COMPLETE_TOKEN_NS"])
    return rows


def build(src: Sources | None = None) -> dict[str, Any]:
    src = src or Sources()
    cands = build_candidates(src)
    coh_f = select_coherence(cands)
    exe_f = select_execution(cands)
    same = coh_f.get("candidate_id") is not None and coh_f.get("candidate_id") == exe_f.get(
        "candidate_id"
    )
    exe_ns = numeric(exe_f.get("COMPLETE_TOKEN_NS"))
    faster = faster_incoherent(cands, exe_ns)

    required_rows = [input_row(p, True, required=True, role="coupled-frontier source") for p in REQUIRED_INPUTS]
    optional_rows = []
    for p in OPTIONAL_KNOWN:
        optional_rows.append(
            input_row(p, src.optional.get(p) is not None, required=False, role="supporting citation")
        )
    healing_known = (
        "receipts/headless/SHARED_BASIS_COHERENT.json",
        "receipts/headless/BINARY_HEALING.json",
    )
    present_h = {p for p, _ in src.healing}
    for p in healing_known:
        optional_rows.append(
            input_row(
                p,
                p in present_h,
                required=False,
                role="optional healed/coherent point",
            )
        )
    for p, _ in src.healing:
        if p not in healing_known:
            optional_rows.append(
                input_row(p, True, required=False, role="optional healed/coherent point")
            )

    coh_name = coh_f.get("name")
    exe_name = exe_f.get("name")
    coh_bpw = numeric(coh_f.get("complete_ebpw"))
    exe_ms = exe_f.get("ms")
    one_line = (
        f"Two coupled frontiers (S024 §2): COHERENCE={coh_name} at {coh_bpw} bpw coherent; "
        f"EXECUTION={exe_name} at {None if exe_ms is None else round(exe_ms, 2)} ms. "
        "Faster incoherent bodies do not take the execution frontier."
    )

    citations = _dedupe(list(REQUIRED_INPUTS) + [c for cand in cands for c in cand.get("citations") or []])

    return {
        "schema": SCHEMA,
        "generated_at": now_iso(),
        "git_head": git_head(),
        "generated_by": GENERATOR,
        "obligation": OBLIGATION,
        "hand_authored": False,
        "did_not_load_a_model": True,
        "did_not_touch_gpu": True,
        "did_not_run_cargo_or_metal_benchmarks": True,
        "did_not_mutate_parent": True,
        "did_not_write_under_models": True,
        "did_not_rederive_roofs": True,
        "unmeasured_is_absent": True,
        "s024": ["§2", "§3", "§18"],
        "one_line": one_line,
        "question": (
            "On the campaign's own receipts, who currently holds the COHERENCE frontier "
            "(lowest complete EBPW that still generates coherently) and who holds the "
            "EXECUTION frontier (lowest COMPLETE_TOKEN_NS among coherent candidates), "
            "and where does every density-descent candidate sit on both axes?"
        ),
        "answer": one_line,
        "why_two_frontiers": (
            "S024 §2: a candidate may lead coherence and lose execution, or vice versa. "
            "binary_g64 is faster than q2f and not coherent; shared_basis_k2 fused is "
            "faster than q2f, kernel-competent, and dead on held-out activation. "
            "Collapsing both axes into one number is how an injured fast body gets "
            "thrown away, or a slow coherent body gets treated as the only story."
        ),
        "required_inputs": required_rows,
        "optional_inputs": optional_rows,
        "candidates": cands,
        "n_candidates": len(cands),
        "COHERENCE_FRONTIER": coh_f,
        "EXECUTION_FRONTIER": exe_f,
        "frontiers_are_the_same_artifact": same,
        "faster_than_execution_frontier_but_incoherent": faster,
        "density_ladder": density_ladder(cands),
        "roofs": copy_roofs(src),
        "organ_density": organ_density(src),
        "dispatch_parent_context": dispatch_context(src),
        "citations": citations,
        "finding": {
            "coherence_frontier_name": coh_name,
            "execution_frontier_name": exe_name,
            "currently_same_artifact": same,
            "faster_incoherent_bodies": [r["id"] for r in faster],
            "heal_receipts_present": sorted(present_h),
        },
    }


def write(doc: dict[str, Any] | None = None) -> Path:
    doc = doc or build()
    write_json(RECEIPT, doc)
    return RECEIPT


def main() -> int:
    doc = build()
    write(doc)
    missing = unresolved_citations(doc)
    print(f"wrote {rel(RECEIPT)} citations_unresolved={len(missing)}")
    if missing:
        for m in missing[:20]:
            print(f"  MISSING {m}")
        return 1
    print(doc["one_line"])
    print(
        "COHERENCE_FRONTIER",
        doc["COHERENCE_FRONTIER"].get("name"),
        "EXECUTION_FRONTIER",
        doc["EXECUTION_FRONTIER"].get("name"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
