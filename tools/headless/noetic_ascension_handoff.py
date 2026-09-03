#!/usr/bin/env python3
"""Assemble NOETIC_ASCENSION_HANDOFF.json from receipts + the demonstration loop.

Every field is read from a receipt (or from the loop transcript the loop wrote)
and carries a reproducing command. Nothing is restated from memory.

    python3 tools/headless/noetic_ascension_handoff.py
    python3 tools/headless/noetic_ascension_handoff.py --refuse-only
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECEIPTS = REPO / "receipts" / "headless"
HANDOFF_REL = "receipts/headless/NOETIC_ASCENSION_HANDOFF.json"
HANDOFF_PATH = REPO / HANDOFF_REL
LOOP_REL = "receipts/headless/NOETIC_ASCENSION_LOOP.json"
SCIENCE_REL = "receipts/headless/NOETIC_ASCENSION_SCIENCE_UPDATE.json"
SCHEMA = "hawking.headless.noetic_ascension_handoff.v1"

REQUIRED_SOURCE_RECEIPTS = (
    "NO_CANDIDATE_YET_BEATS_PARENT.json",
    "FIRST_NOETIC_EXECUTABLE.json",
    "NOETIC_Q3_MLP_Q4_ATTN.json",
    "AFFINE2_NATIVE_MLP.json",
    "NOETIC_COMPOSITION.json",
    "NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json",
    "NOETIC_COMPOSITION_WHOLEMODEL_Q3_G64.json",
    "NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json",
    "FRACTIONAL_BIT_CANON.json",
    "DENSE_SUBBIT_TRANSFER.json",
    "ATTENTION_FLOOR_REFIT.json",
    "DOCTOR_V2_PRESCRIPTION.json",
    "GRAVITY_COMPILER_SEARCH.json",
    "NOETIC_IR.json",
    "NOETIC_CLOSURE.json",
    "NOETIC_ZERO_PARENT.json",
    "NOETIC_NATIVE_OPERATOR.json",
    "CONVENTIONAL_CONTROL_SET.json",
    "CORE_AUTHORITIES.json",
    "CODE_ENTROPY.json",
    "NAMESPACE_MIGRATION.json",
    "ARCHITECTURE_CANON.json",
    "NOETIC_NEGATIVE_SCIENCE.json",
    "AGENTOS_GROK_WORKUNITS.json",
    "NOETIC_ORGAN_CENSUS.json",
    "NOETIC_ROUTE_LEDGER.json",
    "NOETIC_INFORMATION_ACCOUNTING.json",
    "NOETIC_KERNEL_CENSUS.json",
    "NOETIC_METRICS.json",
    "MACHINE_GENOME.json",
    "CAPABILITY_llamacpp-q5k.json",
    "CAPABILITY_mlx-4bit.json",
)


class ClaimRefused(ValueError):
    pass


@dataclass
class Claim:
    id: str
    kind: str
    statement: str
    evidence_path: str
    reproducing_command: List[str]
    derived_from: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "statement": self.statement,
            "evidence_path": self.evidence_path,
            "reproducing_command": self.reproducing_command,
            "reproducing_command_pretty": shlex.join(self.reproducing_command),
            "derived_from": self.derived_from,
        }


def admit(claim: Claim) -> Claim:
    if not claim.id or not str(claim.id).strip():
        raise ClaimRefused("refusing claim with empty id")
    if not claim.kind or not str(claim.kind).strip():
        raise ClaimRefused(f"refusing claim {claim.id!r}: missing kind")
    if not claim.statement or not str(claim.statement).strip():
        raise ClaimRefused(f"refusing claim {claim.id!r}: missing statement")
    if not claim.evidence_path or not str(claim.evidence_path).strip():
        raise ClaimRefused(f"refusing claim {claim.id!r}: missing evidence_path")
    cmd = claim.reproducing_command
    if not cmd:
        raise ClaimRefused(
            f"refusing claim {claim.id!r}: missing reproducing_command"
        )
    if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
        raise ClaimRefused(
            f"refusing claim {claim.id!r}: reproducing_command must be a list of strings"
        )
    if not any(part.strip() for part in cmd):
        raise ClaimRefused(f"refusing claim {claim.id!r}: reproducing_command is empty")
    return claim


def demonstrate_refusal() -> Tuple[int, List[str]]:
    malformed = [
        (
            "no reproducing_command",
            Claim(
                id="MALFORMED_NO_COMMAND",
                kind="fact",
                statement="a candidate beats the parent",
                evidence_path="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
                reproducing_command=[],
            ),
        ),
        (
            "no evidence_path",
            Claim(
                id="MALFORMED_NO_EVIDENCE",
                kind="fact",
                statement="throughput improved",
                evidence_path="",
                reproducing_command=["python3", "-c", "print('no')"],
            ),
        ),
        (
            "empty statement",
            Claim(
                id="MALFORMED_NO_STATEMENT",
                kind="fact",
                statement="",
                evidence_path="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
                reproducing_command=["python3", "-c", "print('no')"],
            ),
        ),
    ]
    lines: List[str] = []
    fired = 0
    for label, claim in malformed:
        try:
            admit(claim)
            msg = f"VACUOUS: {label} was admitted as {claim.id}"
            print(msg)
            lines.append(msg)
        except ClaimRefused as exc:
            msg = f"REFUSED ({label}): {exc}"
            print(msg)
            lines.append(msg)
            fired += 1
    return (0 if fired == len(malformed) else 1), lines


def git_out(args: Sequence[str]) -> str:
    r = subprocess.run(
        ["git", *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    return (r.stdout or "").rstrip("\n") if r.returncode == 0 else ""


def git_exists(rel: str) -> bool:
    r = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{rel}"],
        cwd=str(REPO),
        capture_output=True,
    )
    return r.returncode == 0


def load_receipt(name: str) -> Dict[str, Any]:
    path = RECEIPTS / name
    if not path.is_file():
        raise FileNotFoundError(f"missing receipt {path} (sparse hole? check git show HEAD:receipts/headless/{name})")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise TypeError(f"{name} is not a JSON object")
    return doc


def py_cmd(*body: str) -> List[str]:
    return ["python3", "-c", "\n".join(body)]


def py_receipt_eq(name: str, dotted: str, expected: Any) -> List[str]:
    parts = json.dumps(dotted.split("."))
    payload = json.dumps(expected)
    return py_cmd(
        "import json",
        "from pathlib import Path",
        f"d = json.loads(Path('receipts/headless/{name}').read_text())",
        "cur = d",
        f"for part in {parts}:",
        "    cur = cur[int(part)] if isinstance(cur, list) else cur[part]",
        f"exp = json.loads({json.dumps(payload)})",
        "assert cur == exp, (cur, exp)",
    )


def py_handoff_matches_receipt(
    handoff_dotted: str, receipt: str, receipt_dotted: str
) -> List[str]:
    h_parts = json.dumps(handoff_dotted.split("."))
    r_parts = json.dumps(receipt_dotted.split("."))
    return py_cmd(
        "import json",
        "from pathlib import Path",
        "h = json.loads(Path('receipts/headless/NOETIC_ASCENSION_HANDOFF.json').read_text())",
        f"r = json.loads(Path('receipts/headless/{receipt}').read_text())",
        "hv = h",
        f"for part in {h_parts}:",
        "    hv = hv[int(part)] if isinstance(hv, list) else hv[part]",
        "rv = r",
        f"for part in {r_parts}:",
        "    rv = rv[int(part)] if isinstance(rv, list) else rv[part]",
        "assert hv == rv, (hv, rv)",
    )


def dotted(doc: Any, path: str, default: Any = None) -> Any:
    cur = doc
    for part in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return default
        elif isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            return default
    return cur


def cited(
    *,
    value: Any,
    evidence_path: str,
    reproducing_command: List[str],
    source_field: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "value": value,
        "evidence_path": evidence_path,
        "reproducing_command": reproducing_command,
        "reproducing_command_pretty": shlex.join(reproducing_command),
        "source_field": source_field,
    }
    if extra:
        d.update(extra)
    return d


class IdGen:
    def __init__(self) -> None:
        self.n = 0

    def next(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}{self.n:03d}"


def _claim(
    out: List[Claim],
    ids: IdGen,
    *,
    kind: str,
    statement: str,
    evidence_path: str,
    cmd: List[str],
    how: str,
    source: str,
) -> None:
    out.append(
        admit(
            Claim(
                id=ids.next(kind[0].upper()),
                kind=kind,
                statement=statement,
                evidence_path=evidence_path,
                reproducing_command=cmd,
                derived_from={"how": how, "source": source},
            )
        )
    )


def assemble(loop_doc: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ids = IdGen()
    claims: List[Claim] = []

    def R(name: str) -> Dict[str, Any]:
        return load_receipt(name)

    parent_neg = R("NO_CANDIDATE_YET_BEATS_PARENT.json")
    first = R("FIRST_NOETIC_EXECUTABLE.json")
    q3 = R("NOETIC_Q3_MLP_Q4_ATTN.json")
    affine = R("AFFINE2_NATIVE_MLP.json")
    composition = R("NOETIC_COMPOSITION.json")
    wm_q2f = R("NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json")
    wm_q3 = R("NOETIC_COMPOSITION_WHOLEMODEL_Q3_G64.json")
    wm_tern = R("NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json")
    fractional = R("FRACTIONAL_BIT_CANON.json")
    dense = R("DENSE_SUBBIT_TRANSFER.json")
    attn = R("ATTENTION_FLOOR_REFIT.json")
    doctor = R("DOCTOR_V2_PRESCRIPTION.json")
    gravity = R("GRAVITY_COMPILER_SEARCH.json")
    ir = R("NOETIC_IR.json")
    closure = R("NOETIC_CLOSURE.json")
    zero = R("NOETIC_ZERO_PARENT.json")
    native = R("NOETIC_NATIVE_OPERATOR.json")
    conventional = R("CONVENTIONAL_CONTROL_SET.json")
    core_auth = R("CORE_AUTHORITIES.json")
    entropy = R("CODE_ENTROPY.json")
    namespace = R("NAMESPACE_MIGRATION.json")
    arch = R("ARCHITECTURE_CANON.json")
    negative = R("NOETIC_NEGATIVE_SCIENCE.json")
    grok_wu = R("AGENTOS_GROK_WORKUNITS.json")
    organs = R("NOETIC_ORGAN_CENSUS.json")
    route = R("NOETIC_ROUTE_LEDGER.json")
    accounting = R("NOETIC_INFORMATION_ACCOUNTING.json")
    kernels = R("NOETIC_KERNEL_CENSUS.json")
    metrics = R("NOETIC_METRICS.json")
    genome = R("MACHINE_GENOME.json")
    cap_llama = R("CAPABILITY_llamacpp-q5k.json")
    cap_mlx = R("CAPABILITY_mlx-4bit.json")

    if loop_doc is None:
        loop_path = REPO / LOOP_REL
        if loop_path.is_file():
            loop_doc = json.loads(loop_path.read_text(encoding="utf-8"))
        else:
            loop_doc = {}

    science_doc: Dict[str, Any] = {}
    science_path = REPO / SCIENCE_REL
    if science_path.is_file():
        science_doc = json.loads(science_path.read_text(encoding="utf-8"))

    head = git_out(["rev-parse", "HEAD"]) or "unknown"
    branch = git_out(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    sparse_on = git_out(["config", "--get", "core.sparseCheckout"]) == "true"
    sparse_list = [ln for ln in git_out(["sparse-checkout", "list"]).splitlines() if ln]

    # ---- parent ----
    incumbent = parent_neg["incumbent"]
    parent_section = cited(
        value={
            "artifact": incumbent["artifact"],
            "complete_ebpw": incumbent["complete_ebpw"],
            "parent_params": first.get("parent_params"),
            "q4_incumbent": first.get("q4_incumbent"),
            "zero_parent_verdict": zero.get("verdict"),
            "production_does_not_load_parent_weights": zero.get("verdict") == "PASS",
        },
        evidence_path="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
        reproducing_command=py_handoff_matches_receipt(
            "parent.value.complete_ebpw",
            "NO_CANDIDATE_YET_BEATS_PARENT.json",
            "incumbent.complete_ebpw",
        ),
        source_field="incumbent",
        extra={
            "also_cites": [
                "receipts/headless/FIRST_NOETIC_EXECUTABLE.json",
                "receipts/headless/NOETIC_ZERO_PARENT.json",
            ]
        },
    )
    _claim(
        claims, ids, kind="fact",
        statement=(
            f"the qualified parent is {incumbent['artifact']} at complete_ebpw="
            f"{incumbent['complete_ebpw']}"
        ),
        evidence_path="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
        cmd=py_receipt_eq(
            "NO_CANDIDATE_YET_BEATS_PARENT.json",
            "incumbent.complete_ebpw",
            incumbent["complete_ebpw"],
        ),
        how="json field incumbent.complete_ebpw",
        source="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
    )
    _claim(
        claims, ids, kind="capability",
        statement=f"production Noetic inference does not load parent weights (NOETIC_ZERO_PARENT verdict={zero.get('verdict')!r})",
        evidence_path="receipts/headless/NOETIC_ZERO_PARENT.json",
        cmd=py_receipt_eq("NOETIC_ZERO_PARENT.json", "verdict", zero.get("verdict")),
        how="json field verdict",
        source="receipts/headless/NOETIC_ZERO_PARENT.json",
    )

    # ---- strongest candidate ----
    best = parent_neg["best_coherent_candidate"]
    strongest = cited(
        value=best,
        evidence_path="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
        reproducing_command=py_handoff_matches_receipt(
            "strongest_candidate.value.id",
            "NO_CANDIDATE_YET_BEATS_PARENT.json",
            "best_coherent_candidate.id",
        ),
        source_field="best_coherent_candidate",
        extra={
            "qualification_failure": next(
                (
                    row
                    for row in parent_neg["why_each_candidate_fails_qualification"]
                    if row.get("id") == best.get("id")
                ),
                None,
            )
        },
    )
    _claim(
        claims, ids, kind="fact",
        statement=(
            f"strongest coherent candidate is {best['id']} at complete_ebpw="
            f"{best['complete_ebpw']} tok_s={best['tok_s']}"
        ),
        evidence_path="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
        cmd=py_receipt_eq(
            "NO_CANDIDATE_YET_BEATS_PARENT.json",
            "best_coherent_candidate.id",
            best["id"],
        ),
        how="json field best_coherent_candidate.id",
        source="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
    )

    # ---- closure ----
    closure_section = cited(
        value={
            "claim": closure.get("claim"),
            "n_hashed_members": closure.get("n_hashed_members"),
            "closure_sha256": closure.get("closure_sha256"),
            "io_executor_read_but_not_hashed": dotted(
                closure, "compare.io_executor.n_read_but_not_hashed"
            ),
            "live_decode_read_but_not_hashed": dotted(
                closure, "compare.live_decode.n_read_but_not_hashed"
            ),
        },
        evidence_path="receipts/headless/NOETIC_CLOSURE.json",
        reproducing_command=py_receipt_eq(
            "NOETIC_CLOSURE.json", "closure_sha256", closure.get("closure_sha256")
        ),
        source_field="closure_sha256 + compare.*.n_read_but_not_hashed",
    )
    _claim(
        claims, ids, kind="capability",
        statement=(
            f"NOETIC_CLOSURE hashed {closure.get('n_hashed_members')} members; "
            "io_executor read-but-not-hashed="
            f"{dotted(closure, 'compare.io_executor.n_read_but_not_hashed')}"
        ),
        evidence_path="receipts/headless/NOETIC_CLOSURE.json",
        cmd=py_receipt_eq(
            "NOETIC_CLOSURE.json",
            "n_hashed_members",
            closure.get("n_hashed_members"),
        ),
        how="json field n_hashed_members",
        source="receipts/headless/NOETIC_CLOSURE.json",
    )

    # ---- families tested ----
    candidates = parent_neg.get("candidates") or []
    mixes_not_run = first.get("mixes_not_run") or []
    ir_families = ir.get("executed_families") or []
    families = cited(
        value={
            "executable_mixes": [
                {
                    "id": c.get("id"),
                    "source_receipt": c.get("source_receipt"),
                    "complete_ebpw": c.get("complete_ebpw"),
                    "tok_s": c.get("tok_s"),
                    "coherent": c.get("coherent"),
                    "native_kernel_ran": c.get("native_kernel_ran"),
                }
                for c in candidates
            ],
            "ir_executed_families": ir_families,
            "mixes_not_run": mixes_not_run,
            "first_noetic_chosen": (first.get("chosen") or {}).get("mix_id"),
            "q3_chosen": (q3.get("chosen") or {}).get("mix_id"),
            "affine_chosen": (affine.get("chosen") or {}).get("mix_id"),
        },
        evidence_path="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
        reproducing_command=py_cmd(
            "import json",
            "from pathlib import Path",
            "d = json.loads(Path('receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json').read_text())",
            "ids = [c['id'] for c in d['candidates']]",
            f"exp = json.loads({json.dumps(json.dumps([c.get('id') for c in candidates]))})",
            "assert ids == exp, (ids, exp)",
        ),
        source_field="candidates[].id",
        extra={
            "also_cites": [
                "receipts/headless/FIRST_NOETIC_EXECUTABLE.json",
                "receipts/headless/NOETIC_Q3_MLP_Q4_ATTN.json",
                "receipts/headless/AFFINE2_NATIVE_MLP.json",
                "receipts/headless/NOETIC_IR.json",
            ]
        },
    )
    _claim(
        claims, ids, kind="fact",
        statement=(
            f"{len(candidates)} executable mixes were scored against the parent; "
            f"IR executed families={ir_families!r}"
        ),
        evidence_path="receipts/headless/NOETIC_IR.json",
        cmd=py_receipt_eq("NOETIC_IR.json", "executed_families", ir_families),
        how="json field executed_families",
        source="receipts/headless/NOETIC_IR.json",
    )
    if mixes_not_run:
        _claim(
            claims, ids, kind="not_done",
            statement=(
                "FIRST_NOETIC_EXECUTABLE records "
                f"{mixes_not_run[0].get('id')} as {mixes_not_run[0].get('status')}"
            ),
            evidence_path="receipts/headless/FIRST_NOETIC_EXECUTABLE.json",
            cmd=py_receipt_eq(
                "FIRST_NOETIC_EXECUTABLE.json",
                "mixes_not_run.0.status",
                mixes_not_run[0].get("status"),
            ),
            how="json field mixes_not_run[0].status",
            source="receipts/headless/FIRST_NOETIC_EXECUTABLE.json",
        )

    # ---- organ results ----
    frac_verdict = fractional.get("verdict") or {}
    doctor_organs = []
    for o in doctor.get("organs") or []:
        presc = o.get("prescription") or {}
        doctor_organs.append(
            {
                "organ_id": o.get("organ_id"),
                "kind": o.get("kind"),
                "layer": o.get("layer"),
                "physical_computation": presc.get("physical_computation"),
                "candidate_id": presc.get("candidate_id"),
            }
        )
    frac_organs = []
    for o in fractional.get("organs_out") or []:
        if not isinstance(o, dict):
            continue
        ternary = next(
            (
                c
                for c in (o.get("codecs") or [])
                if isinstance(c, dict) and c.get("codec") == "ternary_aa_g64"
            ),
            None,
        )
        frac_organs.append(
            {
                "layer": o.get("layer"),
                "organ": o.get("organ"),
                "q3_rel_fro": o.get("q3_rel_fro"),
                "ternary_aa_g64": None
                if ternary is None
                else {
                    "storage_bpw": ternary.get("storage_bpw"),
                    "local_survives": ternary.get("local_survives"),
                    "rel_fro": ternary.get("rel_fro"),
                    "health": ternary.get("health"),
                },
            }
        )
    organ_section = cited(
        value={
            "fractional_bit_canon_decision": frac_verdict.get("decision"),
            "fractional_deciding_number": frac_verdict.get("deciding_number"),
            "fractional_deciding_number_meaning": frac_verdict.get(
                "deciding_number_meaning"
            ),
            "doctor_organs_need_different_prescriptions": doctor.get(
                "organs_need_different_prescriptions"
            ),
            "doctor_organs": doctor_organs,
            "fractional_organs_out": frac_organs,
            "attention_floor_decision": (attn.get("verdict") or {}).get("decision"),
            "dense_subbit_decision": (dense.get("verdict") or {}).get("decision"),
            "organ_census_forward_status": organs.get("forward_status"),
        },
        evidence_path="receipts/headless/DOCTOR_V2_PRESCRIPTION.json",
        reproducing_command=py_receipt_eq(
            "DOCTOR_V2_PRESCRIPTION.json",
            "organs_need_different_prescriptions",
            True,
        ),
        source_field="organs_need_different_prescriptions",
        extra={
            "also_cites": [
                "receipts/headless/FRACTIONAL_BIT_CANON.json",
                "receipts/headless/ATTENTION_FLOOR_REFIT.json",
                "receipts/headless/DENSE_SUBBIT_TRANSFER.json",
                "receipts/headless/NOETIC_ORGAN_CENSUS.json",
            ]
        },
    )
    _claim(
        claims, ids, kind="capability",
        statement=(
            "Doctor v2: MLP and attention organs need different physical "
            "computations (organs_need_different_prescriptions=true)"
        ),
        evidence_path="receipts/headless/DOCTOR_V2_PRESCRIPTION.json",
        cmd=py_receipt_eq(
            "DOCTOR_V2_PRESCRIPTION.json",
            "organs_need_different_prescriptions",
            True,
        ),
        how="json field organs_need_different_prescriptions",
        source="receipts/headless/DOCTOR_V2_PRESCRIPTION.json",
    )
    _claim(
        claims, ids, kind="capability",
        statement=(
            f"FRACTIONAL_BIT_CANON decision={frac_verdict.get('decision')!r} "
            f"deciding_number={frac_verdict.get('deciding_number')!r}"
        ),
        evidence_path="receipts/headless/FRACTIONAL_BIT_CANON.json",
        cmd=py_receipt_eq(
            "FRACTIONAL_BIT_CANON.json",
            "verdict.decision",
            frac_verdict.get("decision"),
        ),
        how="json field verdict.decision",
        source="receipts/headless/FRACTIONAL_BIT_CANON.json",
    )
    _claim(
        claims, ids, kind="watched_fail",
        statement=(
            "ATTENTION_FLOOR_REFIT decision="
            f"{(attn.get('verdict') or {}).get('decision')!r}"
        ),
        evidence_path="receipts/headless/ATTENTION_FLOOR_REFIT.json",
        cmd=py_receipt_eq(
            "ATTENTION_FLOOR_REFIT.json",
            "verdict.decision",
            (attn.get("verdict") or {}).get("decision"),
        ),
        how="json field verdict.decision",
        source="receipts/headless/ATTENTION_FLOOR_REFIT.json",
    )
    _claim(
        claims, ids, kind="watched_fail",
        statement=(
            "DENSE_SUBBIT_TRANSFER decision="
            f"{(dense.get('verdict') or {}).get('decision')!r}"
        ),
        evidence_path="receipts/headless/DENSE_SUBBIT_TRANSFER.json",
        cmd=py_receipt_eq(
            "DENSE_SUBBIT_TRANSFER.json",
            "verdict.decision",
            (dense.get("verdict") or {}).get("decision"),
        ),
        how="json field verdict.decision",
        source="receipts/headless/DENSE_SUBBIT_TRANSFER.json",
    )

    # ---- route / density / accounting ----
    measured = parent_neg["blocker"]["measured"]
    dispatch_src = dotted(native, "columns.dispatch_count.SOURCE.value")
    dispatch_exe = dotted(native, "columns.dispatch_count.EXECUTABLE.value")
    route_section = cited(
        value={
            "route_decisions_per_token_discrete": dotted(
                route, "route_decisions_per_token.discrete.value"
            ),
            "route_decisions_per_token_continuous": dotted(
                route,
                "route_decisions_per_token.continuous_selector_applications.value",
            ),
            "native_tokens_per_second": dotted(
                route, "route_decisions_per_second.continuous_native_receipt.tokens_per_second"
            ),
        },
        evidence_path="receipts/headless/NOETIC_ROUTE_LEDGER.json",
        reproducing_command=py_receipt_eq(
            "NOETIC_ROUTE_LEDGER.json",
            "route_decisions_per_token.discrete.value",
            dotted(route, "route_decisions_per_token.discrete.value"),
        ),
        source_field="route_decisions_per_token.discrete.value",
    )
    density_section = cited(
        value={
            "blocker_id": parent_neg["blocker"]["id"],
            "bytes_reduction_fraction": measured["bytes_reduction_fraction"],
            "throughput_gain_fraction": measured["throughput_gain_fraction"],
            "reading": measured["reading"],
            "cheapest_ebpw": measured["cheapest_ebpw"],
            "dearest_ebpw": measured["dearest_ebpw"],
            "cheapest_tok_s": measured["cheapest_tok_s"],
            "dearest_tok_s": measured["dearest_tok_s"],
            "source_dispatch_count": dispatch_src,
            "executable_dispatch_count": dispatch_exe,
        },
        evidence_path="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
        reproducing_command=py_receipt_eq(
            "NO_CANDIDATE_YET_BEATS_PARENT.json",
            "blocker.measured.reading",
            measured["reading"],
        ),
        source_field="blocker.measured",
        extra={"also_cites": ["receipts/headless/NOETIC_NATIVE_OPERATOR.json"]},
    )
    buckets = dotted(accounting, "artifact_accounting.buckets")
    accounting_section = cited(
        value=buckets,
        evidence_path="receipts/headless/NOETIC_INFORMATION_ACCOUNTING.json",
        reproducing_command=py_receipt_eq(
            "NOETIC_INFORMATION_ACCOUNTING.json",
            "artifact_accounting.buckets",
            buckets,
        ),
        source_field="artifact_accounting.buckets",
    )
    _claim(
        claims, ids, kind="watched_fail",
        statement=measured["reading"],
        evidence_path="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
        cmd=py_receipt_eq(
            "NO_CANDIDATE_YET_BEATS_PARENT.json",
            "blocker.measured.bytes_reduction_fraction",
            measured["bytes_reduction_fraction"],
        ),
        how="json field blocker.measured.bytes_reduction_fraction",
        source="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
    )
    _claim(
        claims, ids, kind="fact",
        statement=(
            f"SOURCE and EXECUTABLE dispatch_count are {dispatch_src} and "
            f"{dispatch_exe} (NOETIC_NATIVE_OPERATOR columns.dispatch_count)"
        ),
        evidence_path="receipts/headless/NOETIC_NATIVE_OPERATOR.json",
        cmd=py_receipt_eq(
            "NOETIC_NATIVE_OPERATOR.json",
            "columns.dispatch_count.SOURCE.value",
            dispatch_src,
        ),
        how="json field columns.dispatch_count.SOURCE.value",
        source="receipts/headless/NOETIC_NATIVE_OPERATOR.json",
    )
    _claim(
        claims, ids, kind="fact",
        statement=(
            "the decode path makes no learned discrete per-token route decision "
            f"(value={dotted(route, 'route_decisions_per_token.discrete.value')})"
        ),
        evidence_path="receipts/headless/NOETIC_ROUTE_LEDGER.json",
        cmd=py_receipt_eq(
            "NOETIC_ROUTE_LEDGER.json",
            "route_decisions_per_token.discrete.value",
            0,
        ),
        how="json field route_decisions_per_token.discrete.value",
        source="receipts/headless/NOETIC_ROUTE_LEDGER.json",
    )

    # ---- kernels ----
    kernel_fams = []
    for fam in kernels.get("families") or []:
        if not isinstance(fam, dict):
            continue
        kernel_fams.append(
            {
                "id": fam.get("id"),
                "verdict": fam.get("verdict"),
                "kernel": (fam.get("kernel") or {}).get("name")
                if isinstance(fam.get("kernel"), dict)
                else fam.get("kernel"),
            }
        )
    kernel_section = cited(
        value={
            "production_dispatches_per_token": dotted(
                kernels, "production_token.production_dispatches_per_token"
            ),
            "workhorse_kernel": dotted(kernels, "production_token.workhorse_kernel"),
            "reconciliation_counts": dotted(kernels, "reconciliation.counts"),
            "families": kernel_fams,
            "gravity_scoring_without_kernel_refused": dotted(
                gravity, "demonstration_scoring_without_kernel.refused"
            ),
            "gravity_kernel_win_refused": dotted(
                gravity,
                "demonstration_kernel_win_refused_when_traffic_dominates.refused",
            ),
            "native_production_label": native.get("production_label"),
            "native_production_path": native.get("production_path"),
        },
        evidence_path="receipts/headless/NOETIC_KERNEL_CENSUS.json",
        reproducing_command=py_receipt_eq(
            "NOETIC_KERNEL_CENSUS.json",
            "production_token.production_dispatches_per_token",
            dotted(kernels, "production_token.production_dispatches_per_token"),
        ),
        source_field="production_token.production_dispatches_per_token",
        extra={
            "also_cites": [
                "receipts/headless/GRAVITY_COMPILER_SEARCH.json",
                "receipts/headless/NOETIC_NATIVE_OPERATOR.json",
            ]
        },
    )
    _claim(
        claims, ids, kind="fact",
        statement=(
            "kernel census production_dispatches_per_token="
            f"{dotted(kernels, 'production_token.production_dispatches_per_token')}"
        ),
        evidence_path="receipts/headless/NOETIC_KERNEL_CENSUS.json",
        cmd=py_receipt_eq(
            "NOETIC_KERNEL_CENSUS.json",
            "production_token.production_dispatches_per_token",
            964,
        ),
        how="json field production_token.production_dispatches_per_token",
        source="receipts/headless/NOETIC_KERNEL_CENSUS.json",
    )
    _claim(
        claims, ids, kind="watched_fail",
        statement=(
            "Gravity search refuses to score a candidate with no kernel "
            f"(refused={dotted(gravity, 'demonstration_scoring_without_kernel.refused')})"
        ),
        evidence_path="receipts/headless/GRAVITY_COMPILER_SEARCH.json",
        cmd=py_receipt_eq(
            "GRAVITY_COMPILER_SEARCH.json",
            "demonstration_scoring_without_kernel.refused",
            True,
        ),
        how="json field demonstration_scoring_without_kernel.refused",
        source="receipts/headless/GRAVITY_COMPILER_SEARCH.json",
    )

    # ---- composition ----
    def _wm_summary(doc: Dict[str, Any], name: str) -> Dict[str, Any]:
        rungs = doc.get("rungs") or []
        return {
            "receipt": name,
            "last_survive": dotted(doc, "failure_boundary.organ_degrade.last_survive.label"),
            "last_survive_bpw": dotted(
                doc, "failure_boundary.organ_degrade.last_survive.bpw"
            ),
            "first_die": dotted(doc, "failure_boundary.organ_degrade.first_die.label"),
            "first_die_bpw": dotted(doc, "failure_boundary.organ_degrade.first_die.bpw"),
            "first_fail_free_layer": dotted(doc, "error_accumulation.first_fail_free_layer"),
            "coherent_generation": next(
                (
                    r.get("status")
                    for r in rungs
                    if isinstance(r, dict) and r.get("rung") == "coherent_generation"
                ),
                None,
            ),
        }

    composition_section = cited(
        value={
            "survival_rule": composition.get("survival_rule"),
            "llama_server_note": (composition.get("notes") or [None])[0],
            "wholemodel": [
                _wm_summary(wm_q2f, "NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json"),
                _wm_summary(wm_q3, "NOETIC_COMPOSITION_WHOLEMODEL_Q3_G64.json"),
                _wm_summary(wm_tern, "NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json"),
            ],
            "sign_codes_vs_lowrank_from_no_candidate": (
                parent_neg.get("next_representation_family") or {}
            ).get("evidence_that_narrows_it"),
        },
        evidence_path="receipts/headless/NOETIC_COMPOSITION.json",
        reproducing_command=py_receipt_eq(
            "NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json",
            "failure_boundary.organ_degrade.last_survive.bpw",
            dotted(wm_q2f, "failure_boundary.organ_degrade.last_survive.bpw"),
        ),
        source_field="failure_boundary.organ_degrade",
        extra={
            "also_cites": [
                "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json",
                "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_Q3_G64.json",
                "receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json",
            ]
        },
    )
    _claim(
        claims, ids, kind="fact",
        statement=(
            "WHOLEMODEL_Q2F last_survive is "
            f"{dotted(wm_q2f, 'failure_boundary.organ_degrade.last_survive.label')!r} "
            f"at bpw={dotted(wm_q2f, 'failure_boundary.organ_degrade.last_survive.bpw')}"
        ),
        evidence_path="receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json",
        cmd=py_receipt_eq(
            "NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json",
            "failure_boundary.organ_degrade.last_survive.bpw",
            dotted(wm_q2f, "failure_boundary.organ_degrade.last_survive.bpw"),
        ),
        how="json field failure_boundary.organ_degrade.last_survive.bpw",
        source="receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_Q2F_G64.json",
    )
    _claim(
        claims, ids, kind="not_done",
        statement=(
            "WHOLEMODEL coherent_generation rungs are NOT_RUN "
            "(llama-server unreachable at the time of those receipts)"
        ),
        evidence_path="receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json",
        cmd=py_cmd(
            "import json",
            "from pathlib import Path",
            "d = json.loads(Path('receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json').read_text())",
            "r = next(x for x in d['rungs'] if x.get('rung')=='coherent_generation')",
            "assert r['status'] == 'NOT_RUN', r",
        ),
        how="rungs[rung=coherent_generation].status",
        source="receipts/headless/NOETIC_COMPOSITION_WHOLEMODEL_TERNARY.json",
    )

    # ---- capability ----
    capability_section = cited(
        value={
            "llamacpp_q5k": cap_llama.get("overall"),
            "mlx_4bit": cap_mlx.get("overall"),
            "native_production_label": native.get("production_label"),
            "gravity_capability_contract": gravity.get("capability_contract"),
        },
        evidence_path="receipts/headless/CAPABILITY_llamacpp-q5k.json",
        reproducing_command=py_receipt_eq(
            "CAPABILITY_llamacpp-q5k.json",
            "overall.passed",
            (cap_llama.get("overall") or {}).get("passed"),
        ),
        source_field="overall",
        extra={"also_cites": ["receipts/headless/CAPABILITY_mlx-4bit.json"]},
    )
    _claim(
        claims, ids, kind="capability",
        statement=(
            "llamacpp-q5k capability suite overall.passed="
            f"{(cap_llama.get('overall') or {}).get('passed')} / "
            f"{(cap_llama.get('overall') or {}).get('total')}"
        ),
        evidence_path="receipts/headless/CAPABILITY_llamacpp-q5k.json",
        cmd=py_receipt_eq(
            "CAPABILITY_llamacpp-q5k.json",
            "overall",
            cap_llama.get("overall"),
        ),
        how="json field overall",
        source="receipts/headless/CAPABILITY_llamacpp-q5k.json",
    )
    _claim(
        claims, ids, kind="capability",
        statement=(
            "mlx-4bit capability suite overall.passed="
            f"{(cap_mlx.get('overall') or {}).get('passed')} / "
            f"{(cap_mlx.get('overall') or {}).get('total')}"
        ),
        evidence_path="receipts/headless/CAPABILITY_mlx-4bit.json",
        cmd=py_receipt_eq(
            "CAPABILITY_mlx-4bit.json",
            "overall",
            cap_mlx.get("overall"),
        ),
        how="json field overall",
        source="receipts/headless/CAPABILITY_mlx-4bit.json",
    )

    # ---- production performance ----
    complete = dotted(metrics, "time.COMPLETE_DECODE_TPS") or {}
    production = cited(
        value={
            "complete_decode_tps": complete.get("value"),
            "complete_decode_bar": complete.get("bar"),
            "complete_decode_not_re_run": complete.get("not_re_run"),
            "output_collapsed": complete.get("output_collapsed"),
            "executable_bpw": dotted(metrics, "static.EXECUTABLE_BPW.value"),
            "parent_parameter_count": dotted(metrics, "static.PARENT_PARAMETER_COUNT.value"),
            "machine_genome_single_decoder_tps": genome.get("single_decoder_tps"),
            "machine_genome_best_aggregate_tps": genome.get("best_aggregate_tps"),
            "conventional_live_mlx_decode_tps": dotted(
                conventional, "comparison.live_mlx_over_archived_llama.live_mlx_decode_tps"
            ),
            "q3_tok_s": (q3.get("decode") or {}).get("tok_s")
            if isinstance(q3.get("decode"), dict)
            else None,
            "affine_tok_s": (affine.get("decode") or {}).get("tok_s")
            if isinstance(affine.get("decode"), dict)
            else None,
            "native_operator_shape_reading": dotted(
                gravity, "native_operator_shape.reading"
            ),
        },
        evidence_path="receipts/headless/NOETIC_METRICS.json",
        reproducing_command=py_receipt_eq(
            "NOETIC_METRICS.json",
            "time.COMPLETE_DECODE_TPS.value",
            complete.get("value"),
        ),
        source_field="time.COMPLETE_DECODE_TPS.value",
        extra={
            "also_cites": [
                "receipts/headless/MACHINE_GENOME.json",
                "receipts/headless/CONVENTIONAL_CONTROL_SET.json",
                "receipts/headless/GRAVITY_COMPILER_SEARCH.json",
            ]
        },
    )
    _claim(
        claims, ids, kind="fact",
        statement=(
            f"production complete-decode tok/s={complete.get('value')} "
            f"(bar={complete.get('bar')!r}; timing of a collapsed greedy loop)"
        ),
        evidence_path="receipts/headless/NOETIC_METRICS.json",
        cmd=py_receipt_eq(
            "NOETIC_METRICS.json",
            "time.COMPLETE_DECODE_TPS.value",
            complete.get("value"),
        ),
        how="json field time.COMPLETE_DECODE_TPS.value",
        source="receipts/headless/NOETIC_METRICS.json",
    )

    # ---- negative science ----
    live_opp = negative.get("live_opportunities_being_sat_on") or []
    negative_section = cited(
        value={
            "obligation": negative.get("obligation"),
            "counts": negative.get("counts"),
            "live_opportunity_ids": [
                x.get("id") for x in live_opp if isinstance(x, dict)
            ],
            "nns015": next(
                (x for x in live_opp if isinstance(x, dict) and x.get("id") == "NNS-015"),
                None,
            ),
        },
        evidence_path="receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
        reproducing_command=py_cmd(
            "import json",
            "from pathlib import Path",
            "d = json.loads(Path('receipts/headless/NOETIC_NEGATIVE_SCIENCE.json').read_text())",
            "ids = [x.get('id') for x in (d.get('live_opportunities_being_sat_on') or [])]",
            "assert 'NNS-015' in ids, ids",
        ),
        source_field="live_opportunities_being_sat_on",
    )
    _claim(
        claims, ids, kind="not_done",
        statement=(
            "NNS-015 remains a live opportunity: a distilled/generated MLP operator "
            "has not been run (NOETIC_NEGATIVE_SCIENCE)"
        ),
        evidence_path="receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
        cmd=py_cmd(
            "import json",
            "from pathlib import Path",
            "d = json.loads(Path('receipts/headless/NOETIC_NEGATIVE_SCIENCE.json').read_text())",
            "hit = next(x for x in d['live_opportunities_being_sat_on'] if x.get('id')=='NNS-015')",
            "assert 'has not been run' in (hit.get('reopen_condition') or '').lower() or 'has not been run' in (hit.get('why_today') or '').lower() or 'not been run' in json.dumps(hit).lower()",
        ),
        how="live_opportunities_being_sat_on[id=NNS-015]",
        source="receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
    )

    # ---- open hypotheses ----
    nxt = parent_neg.get("next_representation_family") or {}
    open_hyp = [
        {
            "id": "H-NON-MATVEC",
            "from": "receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
            "field": "next_representation_family",
            "value": nxt,
        },
        {
            "id": "H-GROK-THROTTLE",
            "from": "receipts/headless/AGENTOS_GROK_WORKUNITS.json",
            "field": "still_open",
            "value": grok_wu.get("still_open"),
        },
        {
            "id": "H-SHARED-BASIS-KERNEL-ABSENT",
            "from": "receipts/headless/NOETIC_KERNEL_CENSUS.json",
            "field": "families[id=shared_basis_x_coefficients].verdict",
            "value": next(
                (f.get("verdict") for f in kernel_fams if f.get("id") == "shared_basis_x_coefficients"),
                None,
            ),
        },
        {
            "id": "H-TT-NODE-ABSENT",
            "from": "receipts/headless/NOETIC_IR.json",
            "field": "cannot_express",
            "value": ir.get("cannot_express"),
        },
    ]
    open_section = cited(
        value=open_hyp,
        evidence_path="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
        reproducing_command=py_receipt_eq(
            "NO_CANDIDATE_YET_BEATS_PARENT.json",
            "next_representation_family.family",
            nxt.get("family"),
        ),
        source_field="next_representation_family.family",
    )
    _claim(
        claims, ids, kind="not_done",
        statement=(
            f"next representation family is {nxt.get('family')!r}; reopen_condition="
            f"{nxt.get('reopen_condition')!r}"
        ),
        evidence_path="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
        cmd=py_receipt_eq(
            "NO_CANDIDATE_YET_BEATS_PARENT.json",
            "next_representation_family.reopen_condition",
            nxt.get("reopen_condition"),
        ),
        how="json field next_representation_family.reopen_condition",
        source="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
    )

    # ---- grok lane ----
    grok_section = cited(
        value={
            "gate": grok_wu.get("gate"),
            "result": grok_wu.get("result"),
            "closed": grok_wu.get("closed"),
            "still_open": grok_wu.get("still_open"),
            "test_suite": grok_wu.get("test_suite"),
        },
        evidence_path="receipts/headless/AGENTOS_GROK_WORKUNITS.json",
        reproducing_command=py_receipt_eq(
            "AGENTOS_GROK_WORKUNITS.json", "result", grok_wu.get("result")
        ),
        source_field="result + closed + still_open",
    )
    _claim(
        claims, ids, kind="capability",
        statement=(
            f"AGENTOS_GROK_WORKUNITS result={grok_wu.get('result')!r} with "
            f"{len(grok_wu.get('closed') or [])} closed defects"
        ),
        evidence_path="receipts/headless/AGENTOS_GROK_WORKUNITS.json",
        cmd=py_receipt_eq(
            "AGENTOS_GROK_WORKUNITS.json", "result", grok_wu.get("result")
        ),
        how="json field result",
        source="receipts/headless/AGENTOS_GROK_WORKUNITS.json",
    )

    # ---- next AgentOS WorkUnits ----
    next_wus = [
        {
            "id": "gravity.non_matvec_operator",
            "role": "experiment",
            "description": nxt.get("reopen_condition"),
            "preferred_backend": "tool",
            "resource_class": "LIGHT_CONTROL",
            "source_receipt": "receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
            "source_field": "next_representation_family.reopen_condition",
            "verifier": (
                "not yet written: the experiment does not exist. A verifier will "
                "be a command that fails unless dispatch-count per token drops "
                "below the recorded 964 while remaining coherent."
            ),
            "status": "not_done",
        },
        {
            "id": "grok.throttle_measurement",
            "role": "observe",
            "description": (grok_wu.get("still_open") or [None])[0],
            "preferred_backend": "tool",
            "resource_class": "LIGHT_CONTROL",
            "source_receipt": "receipts/headless/AGENTOS_GROK_WORKUNITS.json",
            "source_field": "still_open[0]",
            "status": "not_done",
        },
        {
            "id": "doctor.distilled_mlp_operator",
            "role": "experiment",
            "description": (next((x for x in live_opp if x.get("id") == "NNS-015"), {}) or {}).get("reopen_condition"),
            "preferred_backend": "tool",
            "resource_class": "LIGHT_CONTROL",
            "source_receipt": "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
            "source_field": "live_opportunities_being_sat_on[id=NNS-015]",
            "status": "not_done",
        },
    ]
    next_wu_section = cited(
        value=next_wus,
        evidence_path="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
        reproducing_command=py_receipt_eq(
            "NO_CANDIDATE_YET_BEATS_PARENT.json",
            "next_representation_family.family",
            nxt.get("family"),
        ),
        source_field="next_representation_family + AGENTOS_GROK_WORKUNITS.still_open + NNS-015",
    )

    # ---- rollback ----
    never_delete = entropy.get("never_delete")
    rollback = cited(
        value={
            "command": (
                "rm -f receipts/headless/NOETIC_ASCENSION_HANDOFF.json "
                "receipts/headless/NOETIC_ASCENSION_LOOP.json "
                "receipts/headless/NOETIC_ASCENSION_SCIENCE_UPDATE.json "
                "docs/NOETIC_ASCENSION_HANDOFF.md; "
                "git checkout HEAD -- tools/headless/noetic_ascension_loop.py "
                "tools/headless/noetic_ascension_handoff.py "
                "tools/headless/test_noetic_ascension_handoff.py"
            ),
            "what_it_reverts": (
                "this lane's new files only. Historical receipts/ are never_delete."
            ),
            "never_delete": never_delete,
            "does_not_touch": [
                "hcli/",
                "receipts/ascent-2026-08-16",
                "workspace/campaign",
                "any receipt this lane did not create",
            ],
        },
        evidence_path="receipts/headless/CODE_ENTROPY.json",
        reproducing_command=py_receipt_eq(
            "CODE_ENTROPY.json",
            "never_delete",
            never_delete,
        ),
        source_field="never_delete",
    )
    _claim(
        claims, ids, kind="fact",
        statement="CODE_ENTROPY.never_delete forbids deleting receipts/",
        evidence_path="receipts/headless/CODE_ENTROPY.json",
        cmd=py_cmd(
            "import json",
            "from pathlib import Path",
            "d = json.loads(Path('receipts/headless/CODE_ENTROPY.json').read_text())",
            "nd = d.get('never_delete') or {}",
            "assert 'receipts/' in nd, nd",
        ),
        how="json field never_delete['receipts/']",
        source="receipts/headless/CODE_ENTROPY.json",
    )

    # ---- canonical launch ----
    launch_value = "python3 -m hcli"
    launch = cited(
        value=launch_value,
        evidence_path="docs/CURRENT_ARCHITECTURE.md",
        reproducing_command=py_cmd(
            "import subprocess",
            "t = subprocess.check_output(['git','show','HEAD:docs/CURRENT_ARCHITECTURE.md'], text=True)",
            "assert 'python3 -m hcli is the product entry' in t or '`python3 -m hcli` is the product entry' in t",
            "h = __import__('json').loads(__import__('pathlib').Path('receipts/headless/NOETIC_ASCENSION_HANDOFF.json').read_text())",
            "assert h['canonical_hcli_launch']['value'] == 'python3 -m hcli'",
        ),
        source_field="docs/CURRENT_ARCHITECTURE.md (via git show HEAD: ; file is not always materialized)",
        extra={
            "namespace_canonical_package": namespace.get("canonical_package_name"),
            "namespace_canonical_path": namespace.get("canonical_physical_path"),
            "architecture_canon_document": arch.get("document"),
            "core_authorities_canonical_package": dotted(
                core_auth, "namespace.canonical_package"
            ),
            "verified_against_head": True,
            "also_cites": [
                "receipts/headless/ARCHITECTURE_CANON.json",
                "receipts/headless/NAMESPACE_MIGRATION.json",
                "receipts/headless/CORE_AUTHORITIES.json",
            ],
        },
    )
    _claim(
        claims, ids, kind="fact",
        statement="canonical HCLI launch is python3 -m hcli (docs/CURRENT_ARCHITECTURE.md at HEAD)",
        evidence_path="docs/CURRENT_ARCHITECTURE.md",
        cmd=py_cmd(
            "import subprocess",
            "r = subprocess.run(['git','cat-file','-e','HEAD:docs/CURRENT_ARCHITECTURE.md'])",
            "assert r.returncode == 0",
            "t = subprocess.check_output(['git','show','HEAD:docs/CURRENT_ARCHITECTURE.md'], text=True)",
            "assert 'python3 -m hcli' in t",
        ),
        how="git show HEAD:docs/CURRENT_ARCHITECTURE.md substring",
        source="docs/CURRENT_ARCHITECTURE.md",
    )
    _claim(
        claims, ids, kind="fact",
        statement=(
            f"NAMESPACE_MIGRATION canonical_package_name={namespace.get('canonical_package_name')!r} "
            f"canonical_physical_path={namespace.get('canonical_physical_path')!r}"
        ),
        evidence_path="receipts/headless/NAMESPACE_MIGRATION.json",
        cmd=py_receipt_eq(
            "NAMESPACE_MIGRATION.json",
            "canonical_physical_path",
            namespace.get("canonical_physical_path"),
        ),
        how="json field canonical_physical_path",
        source="receipts/headless/NAMESPACE_MIGRATION.json",
    )

    # ---- demonstration / verdict ----
    gate = (loop_doc or {}).get("gate") or science_doc.get("gate") or "UNKNOWN"
    verdict = parent_neg.get("verdict")
    demo_steps = (loop_doc or {}).get("steps") or []
    _claim(
        claims, ids, kind="watched_fail",
        statement=(
            f"campaign verdict is {verdict}; the demonstration gate is {gate}. "
            "No candidate is promoted."
        ),
        evidence_path="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
        # Reproduce against the LIVE verdict, not a literal captured when this
        # builder was written. Hardcoding it meant the handoff's own command
        # failed the moment the reopen condition fired and the receipt moved to
        # REOPENED_CANDIDATE_LEADS_BUT_QUALIFICATION_NOT_RUN -- a claim that
        # contradicted its own statement one line above.
        cmd=py_receipt_eq(
            "NO_CANDIDATE_YET_BEATS_PARENT.json",
            "verdict",
            verdict,
        ),
        how="json field verdict",
        source="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
    )
    _claim(
        claims, ids, kind="capability",
        statement=(
            "the demonstration loop receipt records gate=REJECTED and a mission "
            "that HCLI drove to completion"
        ),
        evidence_path=LOOP_REL,
        cmd=py_cmd(
            "import json",
            "from pathlib import Path",
            "d = json.loads(Path('receipts/headless/NOETIC_ASCENSION_LOOP.json').read_text())",
            "assert d.get('gate') == 'REJECTED', d.get('gate')",
            "assert (d.get('mission') or {}).get('status') == 'completed', d.get('mission')",
            "assert d.get('did_not_load_second_27b') is True",
            "stages = {s.get('stage') for s in (d.get('steps') or [])}",
            "for need in ('HCLI','AgentOS','resident','Doctor','Gravity experiment','tools','verifier','accepted or rejected','updated science'):",
            "    assert need in stages, (need, stages)",
        ),
        how="NOETIC_ASCENSION_LOOP.json gate + stages",
        source=LOOP_REL,
    )
    _claim(
        claims, ids, kind="fact",
        statement=(
            "science update receipt records gate=REJECTED and did_not_promote=true"
        ),
        evidence_path=SCIENCE_REL,
        cmd=py_cmd(
            "import json",
            "from pathlib import Path",
            "d = json.loads(Path('receipts/headless/NOETIC_ASCENSION_SCIENCE_UPDATE.json').read_text())",
            "assert d.get('gate') == 'REJECTED'",
            "assert d.get('did_not_promote') is True",
            "assert d.get('verdict') == 'NO_CANDIDATE_YET_BEATS_PARENT'",
        ),
        how="NOETIC_ASCENSION_SCIENCE_UPDATE.json",
        source=SCIENCE_REL,
    )

    # g014 dependency
    _claim(
        claims, ids, kind="not_done",
        statement=parent_neg.get("g014_dependency") or "g014_dependency missing",
        evidence_path="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
        cmd=py_receipt_eq(
            "NO_CANDIDATE_YET_BEATS_PARENT.json",
            "g014_dependency",
            parent_neg.get("g014_dependency"),
        ),
        how="json field g014_dependency",
        source="receipts/headless/NO_CANDIDATE_YET_BEATS_PARENT.json",
    )

    missing_sources = [
        n for n in REQUIRED_SOURCE_RECEIPTS if not (RECEIPTS / n).is_file()
    ]
    for n in REQUIRED_SOURCE_RECEIPTS:
        if not (RECEIPTS / n).is_file() and git_exists(f"receipts/headless/{n}"):
            _claim(
                claims, ids, kind="unknown",
                statement=(
                    f"{n} is at git HEAD but not on disk in this worktree "
                    "(sparse hole — not evidence of absence)"
                ),
                evidence_path=f"receipts/headless/{n}",
                cmd=py_cmd(
                    "import subprocess",
                    f"r = subprocess.run(['git','cat-file','-e','HEAD:receipts/headless/{n}'])",
                    "assert r.returncode == 0",
                ),
                how="git cat-file -e",
                source=f"receipts/headless/{n}",
            )

    watched = [c.as_dict() for c in claims if c.kind == "watched_fail"]
    by_kind: Dict[str, int] = {}
    for c in claims:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1

    haider_copy = None
    haider_dir = REPO / ".haider"
    if haider_dir.is_dir():
        haider_copy = ".hcli-legacy/NOETIC_ASCENSION_HANDOFF.json"

    doc: Dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "tools/headless/noetic_ascension_handoff.py",
        "loop": "tools/headless/noetic_ascension_loop.py",
        "git_head": head,
        "branch": branch,
        "sparse": {
            "on": sparse_on,
            "roots": sparse_list,
            "hcli_materialized_in_worktree": (REPO / "hcli").is_dir(),
            "docs_materialized_in_worktree": (REPO / "docs").is_dir(),
            "hcli_verified_via": "git archive HEAD hcli (see NOETIC_ASCENSION_LOOP.json hcli_source)",
            "docs_verified_via": "git show HEAD:docs/CURRENT_ARCHITECTURE.md",
            "note": (
                "A missing path in this worktree is not evidence it is absent "
                "from the repository. git sparse-checkout add is blocked here."
            ),
        },
        "verdict": verdict,
        "gate": gate,
        "did_not_promote": True,
        "did_not_load_second_27b": True,
        "faking_the_shift": parent_neg.get("faking_the_shift"),
        "g014_dependency": parent_neg.get("g014_dependency"),
        "canonical_hcli_launch": launch,
        "parent": parent_section,
        "strongest_candidate": strongest,
        "closure": closure_section,
        "families_tested": families,
        "organ_results": organ_section,
        "route_metrics": route_section,
        "density_metrics": density_section,
        "accounting_metrics": accounting_section,
        "kernels": kernel_section,
        "composition": composition_section,
        "capability": capability_section,
        "production_performance": production,
        "negative_science": negative_section,
        "open_hypotheses": open_section,
        "grok_lane_results": grok_section,
        "next_agentos_workunits": next_wu_section,
        "rollback_command": rollback,
        "demonstration": {
            "receipt": LOOP_REL,
            "science_update": SCIENCE_REL,
            "gate": gate,
            "durable_reproduce": (loop_doc or {}).get("durable_reproduce")
            or ["python3", "tools/headless/noetic_ascension_loop.py"],
            "mission": (loop_doc or {}).get("mission"),
            "stages_required": (loop_doc or {}).get("stages_required"),
            "steps": [
                {
                    "stage": s.get("stage"),
                    "name": s.get("name"),
                    "command_pretty": s.get("command_pretty"),
                    "durable_reproduce": s.get("durable_reproduce"),
                    "exit_code": s.get("exit_code"),
                    "status": s.get("status"),
                    "workunit_status": s.get("workunit_status"),
                    "verification_ok": s.get("verification_ok"),
                    "reason": s.get("reason"),
                    "notes": s.get("notes"),
                }
                for s in demo_steps
            ],
        },
        "claims": [c.as_dict() for c in claims],
        "what_i_watched_fail": watched,
        "counts": {"claims": len(claims), "by_kind": by_kind},
        "source_receipts": [f"receipts/headless/{n}" for n in REQUIRED_SOURCE_RECEIPTS],
        "missing_source_receipts_on_disk": missing_sources,
        "haider_copy": haider_copy,
        "honesty": (
            "No Noetic candidate beats the parent. The blocker is measured: "
            f"{measured['reading']} The next family is "
            f"{nxt.get('family')!r}. A handoff that read as a success would "
            "misdirect whoever picks this up."
        ),
    }
    return doc


def write_handoff(doc: Dict[str, Any]) -> Path:
    HANDOFF_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = HANDOFF_PATH.with_suffix(HANDOFF_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, HANDOFF_PATH)
    haider = REPO / ".haider"
    if haider.is_dir():
        dest = haider / "NOETIC_ASCENSION_HANDOFF.json"
        dest.write_text(HANDOFF_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return HANDOFF_PATH


def assemble_and_write(loop_doc: Optional[Dict[str, Any]] = None) -> Path:
    rc, lines = demonstrate_refusal()
    if rc != 0:
        raise SystemExit("refusal demo did not fire; refusing to emit a handoff")
    doc = assemble(loop_doc=loop_doc)
    doc["refusal_demo"] = {
        "fired": True,
        "lines": lines,
        "reproduce": [
            "python3",
            "tools/headless/noetic_ascension_handoff.py",
            "--refuse-only",
        ],
    }
    return write_handoff(doc)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    rc, lines = demonstrate_refusal()
    if "--refuse-only" in args:
        return rc
    if rc != 0:
        print("refusal demo did not fire; refusing to emit a handoff", file=sys.stderr)
        return rc
    path = assemble_and_write()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
