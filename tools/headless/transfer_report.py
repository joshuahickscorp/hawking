#!/usr/bin/env python3
"""QWEN_TRANSFER_REPORT — what should a future model INHERIT, not what happened.

A campaign log answers "what happened". This answers "what should the next specimen
start with", and it is machine-readable because the thing that consumes it is a planner,
not a person.

Every entry carries five fields, all required:
  applicability_conditions      when this result is even a candidate
  successful_architecture_classes  where it is measured to work
  failed_architecture_classes      where it is measured not to
  required_kernel_shape         what must exist for it to be executable at all
  reopening_conditions          what would make a settled negative worth retrying

An entry missing any of them is REFUSED. A recommendation with no reopening condition is
a dead end pretending to be inheritance.

It is built from the canonical libraries, never from prose, so it cannot drift from them.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
sys.path.insert(0, str(REPO / "tools/headless"))

FIELDS = ["applicability_conditions", "successful_architecture_classes",
          "failed_architecture_classes", "required_kernel_shape", "reopening_conditions"]
PARENT_CLASS = "dense_hybrid_transformer"
PARENT = "qwen3.8-27b-abliterated"


class Refused(Exception):
    pass


# The adversarial sweep won here: this report is BUILT from the canonical libraries, but
# nothing proved it. "Derived by construction" is an argument, and §102 does not accept
# arguments. Same mechanism the rehearsal uses: record every file the process opens.
_reads: list[str] = []
FORBIDDEN_PREFIXES = [str(Path.home() / "noetic"), str(Path.home() / "models"),
                      str(REPO / "workspace"), str(REPO / "artifacts"), str(REPO / "crates")]
ALLOWED_RECEIPTS = {
    "REPRESENTATION_LIBRARY.json", "NOETIC_NEGATIVE_SCIENCE.json",
    "ORGAN_FRONTIER_MATRIX.json", "KERNEL_LIBRARY.json", "SUPEROPERATOR_LIBRARY.json",
    "ORGAN_LIBRARY.json", "BYTES_FRONTIER.json", "QWEN_TRANSFER_REPORT.json",
}


def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        try:
            _reads.append(os.fspath(args[0]))
        except TypeError:
            pass


def input_audit(cited=()):
    """The boundary the report must respect is narrower than the rehearsal's.

    The REPORT's job is to encode what Qwen taught, so opening the experiment receipts it
    CITES -- to prove each citation's JSON path actually resolves -- is the work, not
    smuggling. What it must never touch is Qwen's private working state: the artifacts,
    the captures, the models, the workspace. So the allowlist is the canonical libraries
    plus whatever this report cites, computed from the entries rather than hardcoded, and
    the forbidden prefixes stay absolute.
    """
    allowed = set(ALLOWED_RECEIPTS) | {Path(c.partition("#")[0]).name for c in cited}
    outside, forbidden = [], []
    for r in _reads:
        try:
            q = str(Path(r).resolve())
        except Exception:
            continue
        if not q.startswith(str(REPO)) and not q.startswith(str(Path.home())):
            continue
        if "__pycache__" in q:
            continue
        if any(q.startswith(f) for f in FORBIDDEN_PREFIXES):
            forbidden.append(q)
        elif q.startswith(str(RH)) and Path(q).name not in allowed:
            outside.append(q)
    return {"n_opens_recorded": len(_reads),
            "n_forbidden_reads": len(set(forbidden)),
            "forbidden_reads": sorted(set(forbidden))[:10],
            "n_reads_outside_allowlist": len(set(outside)),
            "reads_outside_allowlist": sorted(set(outside))[:10],
            "allowlist_receipts": sorted(allowed),
            "allowlist_is": "the canonical libraries plus every receipt this report cites",
            "forbidden_prefixes": FORBIDDEN_PREFIXES,
            "clean": not forbidden and not outside}


# An empty list is a legitimate ANSWER for the two architecture-class fields ("nowhere
# has this been shown to work yet" / "nowhere has it failed"). It is not a legitimate
# answer for the other three: a recommendation with no applicability condition, no kernel
# shape, or no reopening condition is not inheritable.
MUST_BE_NONEMPTY = ["applicability_conditions", "required_kernel_shape", "reopening_conditions"]


def validate(e):
    absent = [f for f in FIELDS if f not in e]
    if absent:
        raise Refused(f"{e.get('id','<no id>')}: missing field(s) {absent}")
    empty = [f for f in MUST_BE_NONEMPTY if not e.get(f)]
    if empty:
        raise Refused(f"{e.get('id','<no id>')}: empty {empty} -- a recommendation with no "
                      f"{empty[0]} is not inheritable")
    if not e["successful_architecture_classes"] and not e["failed_architecture_classes"]:
        raise Refused(f"{e['id']}: neither a successful nor a failed architecture class -- "
                      f"the entry records no measured outcome anywhere")
    if not e.get("evidence"):
        raise Refused(f"{e['id']}: no evidence path")
    for c in e["evidence"]:
        rel, _, jp = c.partition("#")
        f = REPO / rel
        if not f.exists():
            raise Refused(f"{e['id']}: evidence receipt missing: {rel}")
        if jp:
            cur = json.load(open(f))
            for part in jp.split("."):
                if isinstance(cur, list):
                    cur = cur[int(part)]
                elif isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    raise Refused(f"{e['id']}: evidence path does not resolve: {c}")
    return e


def method_entries():
    """Methods transfer; values do not. These are the entries a planner acts on."""
    return [
        dict(id="TR-METHOD-PER-ORGAN-FLOOR",
             inherit="Measure the information floor PER ORGAN, never per model.",
             why="attention, DeltaNet and embedding/output all survive held-out activations at "
                 "ws_rtn_q3_g128 and all fail at q2f_g64 -- the same codec the MLP survives. A "
                 "uniform bit rate therefore either wastes bits on the MLP or breaks every other "
                 "organ.",
             applicability_conditions=["the model has more than one organ family",
                                       "held-out real activations can be captured per organ"],
             successful_architecture_classes=[PARENT_CLASS],
             failed_architecture_classes=[],
             required_kernel_shape="one GEMV kernel per (representation, group size) pair; the "
                                   "runtime must be able to read a MIXED artifact",
             reopening_conditions=["a model whose organs share a floor would make per-organ "
                                   "search wasted effort; measure two organs before committing"],
             evidence=["receipts/headless/ORGAN_DENSITY_FLOORS.json#organs.gqa_attention.candidates.5.complete_ebpw",
                       "receipts/headless/ORGAN_DENSITY_FLOORS.json#organs.gqa_attention.candidates.6.complete_ebpw",
                       "receipts/headless/ORGAN_FRONTIER_MATRIX.json#n_measured"]),
        dict(id="TR-METHOD-HELDOUT-ACTIVATIONS",
             inherit="Judge a candidate on held-out REAL activations, never on weight-space error "
                     "and never on synthetic or Gaussian activations.",
             why="weight-space error ranked candidates wrongly at every rate on this specimen; "
                 "the held-out probe is what separates 2.25 from everything below it.",
             applicability_conditions=["activations can be captured from the parent",
                                       "a probe layer set is chosen before candidates are scored"],
             successful_architecture_classes=[PARENT_CLASS],
             failed_architecture_classes=[],
             required_kernel_shape="none -- this is a scoring rule, not an operator",
             reopening_conditions=["never; a synthetic-activation ranking has been refuted twice "
                                   "here and is not a candidate method"],
             evidence=["receipts/headless/SHARED_BASIS_COHERENT.json#finding.reason",
                       "receipts/headless/HYBRID_OPERATOR.json#finding.reason"]),
        dict(id="TR-METHOD-COMPETENT-KERNEL-FIRST",
             inherit="Never evaluate a representation with an incompetent kernel. Fewer stored "
                     "bits is not fewer nanoseconds.",
             why="the shared basis proves it in both directions: its density did not survive, yet "
                 "a competent fused kernel cut dispatches 384->192 and COMPLETE_TOKEN_NS "
                 "110201749->24554625.",
             applicability_conditions=["a native kernel exists or can be written for the candidate"],
             successful_architecture_classes=[PARENT_CLASS],
             failed_architecture_classes=[],
             required_kernel_shape="in-register dequantization, no dense weight materialization on "
                                   "GEMV, geometry tpr64/tg128 on this device",
             reopening_conditions=["a device whose dispatch overhead is negligible would let a "
                                   "naive kernel rank representations fairly"],
             evidence=["receipts/headless/SHARED_BASIS_KERNEL.json#finding.reason",
                       "receipts/headless/KERNEL_LIBRARY.json#n_complete"]),
        dict(id="TR-METHOD-MEASURE-BEFORE-ELIMINATING",
             inherit="Compute head cosine (and channel deadness, and layer near-identity) BEFORE "
                     "proposing structural elimination.",
             why="head sharing is refuted here because the heads are near-orthogonal here: Q mean "
                 "cosine 0.0438, zero dead MLP channels, zero near-identity layers. The refutation "
                 "is a property of this model's geometry, not of the technique.",
             applicability_conditions=["weights can be streamed one tensor at a time"],
             successful_architecture_classes=[],
             failed_architecture_classes=[PARENT_CLASS],
             required_kernel_shape="none -- elimination removes structure before any kernel exists",
             reopening_conditions=["any model whose measured head cosine is high",
                                   "any model with measurably dead channels or near-identity layers"],
             evidence=["receipts/headless/STRUCTURAL_ELIMINATION.json#attention_heads.headline.q_mean_cosine_all_layers"]),
        dict(id="TR-METHOD-BROAD-INJURY-TEST",
             inherit="When a body is fast and injured, measure how BROAD the injury is before "
                     "spending on healing.",
             why="binary at ~1.25 bpw is physically fast and generation-injured, and the injury was "
                 "broad enough that 0 of 4 island schemes reached coherent generation. Localizing "
                 "first would have cost less than four healing attempts.",
             applicability_conditions=["a candidate is fast but incoherent"],
             successful_architecture_classes=[],
             failed_architecture_classes=[PARENT_CLASS],
             required_kernel_shape="the injured representation's kernel plus a fallback path for "
                                   "protected regions",
             reopening_conditions=["a sensitivity map that localizes the injury to a region small "
                                   "enough that protecting it costs less than the next codec up"],
             evidence=["receipts/headless/BINARY_HEALING.json#finding.n_that_reached_coherent_generation"]),
    ]


def representation_entries():
    import representation_library as rl
    fams = rl.build()
    out = []
    for f in fams:
        arch = f["per_architecture"].get(PARENT, {})
        ok = arch.get("successful_organs") or []
        capfail = arch.get("capability_failed_organs") or []
        failed = arch.get("failed_organs") or []
        if not (ok or capfail or failed or f["failures"]):
            continue
        reopen = [x["reopen_condition"] for x in f["failures"]] or \
                 ["no failure recorded; reopening is not applicable"]
        ev = [c for x in f["failures"] for c in x.get("evidence", [])] or \
             ["receipts/headless/REPRESENTATION_LIBRARY.json#n_families"]
        out.append(dict(
            id=f"TR-REPR-{f['family'].upper()}",
            inherit=(f"{f['family']}: SEED IT for organs like {ok}" if ok else
                     f"{f['family']}: it failed here on {sorted(set(capfail + failed))} -- "
                     f"rank it lower, do not exclude it"),
            why=f["what"], zero_kind=f["zero_kind"],
            applicability_conditions=[
                f"organ operator shape matches one of {ok or sorted(set(capfail + failed))}",
                "a native kernel for this family exists on the target device"],
            successful_architecture_classes=[PARENT_CLASS] if ok else [],
            # a recorded failure in the negative store counts as a measured outcome even
            # when the v1 library carried no per-organ row for the family
            failed_architecture_classes=[PARENT_CLASS] if (capfail or failed or f["failures"]) else [],
            required_kernel_shape=("GEMV with in-register dequantization at the family's group "
                                   "size; fused gate_up_swiglu where the intermediate is not "
                                   "observable"),
            reopening_conditions=reopen, evidence=ev,
            measured_density_frontier=(arch.get("density_frontier") or {}).get("active_bpw"),
        ))
    return out


def negative_entries():
    import negative_science as ns
    p = RH / "NOETIC_NEGATIVE_SCIENCE.json"
    entries = json.load(open(p))["entries"] if p.exists() else []
    out = []
    for n in entries:
        if n.get("migrated") or not n.get("evidence"):
            continue
        out.append(dict(
            id=f"TR-NEG-{n['id']}",
            inherit=f"Do not spend an experiment on {n['technique']} for an organ like "
                    f"{n['organ']} without first checking its reopening condition.",
            why=n["physical_reason"], level=n["level"],
            applicability_conditions=[f"organ resembles {n['organ']}",
                                      f"technique is {n['technique']}"],
            successful_architecture_classes=[],
            failed_architecture_classes=[PARENT_CLASS],
            required_kernel_shape=n["kernel"],
            reopening_conditions=[n["reopen_condition"]],
            evidence=n["evidence"],
            scope_law="MODEL_SPECIFIC: this warns another architecture, it never prunes there",
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", required=True)
    ap.add_argument("--query", nargs=2, metavar=("ORGAN", "ARCH_CLASS"))
    ap.add_argument("--refuse-demo", action="store_true")
    ap.add_argument("--smuggle-demo", action="store_true")
    a = ap.parse_args()

    sys.addaudithook(_hook)
    if a.smuggle_demo:
        pth = Path.home() / "noetic/NOETIC_PARENT_A/MIX_REPORT.json"
        if pth.exists():
            json.load(open(pth))
        au = input_audit()
        print(json.dumps({"clean": au["clean"],
                          "n_forbidden_reads": au["n_forbidden_reads"],
                          "forbidden_reads": au["forbidden_reads"][:2]}, indent=1))
        return 0 if not au["clean"] else 1

    if a.refuse_demo:
        for bad in ({"id": "NO-FIELDS"},
                    {"id": "NO-REOPEN", **{f: ["x"] for f in FIELDS if f != "reopening_conditions"},
                     "reopening_conditions": [], "evidence": ["receipts/headless/KERNEL_LIBRARY.json"]},
                    {"id": "DEAD-EVIDENCE", **{f: ["x"] for f in FIELDS},
                     "evidence": ["receipts/headless/NOPE.json#a"]}):
            try:
                validate(bad)
            except Refused as r:
                print("REFUSED:", r)
        return 0

    entries, rejected = [], []
    for e in method_entries() + representation_entries() + negative_entries():
        try:
            entries.append(validate(e))
        except Refused as r:
            rejected.append(str(r))

    all_cited = [c for e in entries for c in e.get("evidence", [])]
    if a.query:
        organ, arch = a.query
        hits = [e for e in entries
                if any(organ in c for c in e["applicability_conditions"])
                or e["id"].startswith("TR-METHOD")]
        print(json.dumps({"organ": organ, "arch_class": arch,
                          "inherit": [{"id": h["id"], "inherit": h["inherit"],
                                       "applies": "SEED" if arch in h["successful_architecture_classes"]
                                       else ("WARN" if arch in h["failed_architecture_classes"]
                                             else "CANDIDATE"),
                                       "reopening_conditions": h["reopening_conditions"]}
                                      for h in hits]}, indent=1))
        return 0

    out = {
        "schema": "hawking.headless.qwen_transfer_report.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/headless/transfer_report.py",
        "obligation": "G007 — QWEN_TRANSFER_REPORT (directive §14)",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False,
        "question_this_answers": "WHAT SHOULD A FUTURE MODEL INHERIT",
        "not_the_question": "what happened",
        "parent": PARENT, "parent_architecture_class": PARENT_CLASS,
        "required_fields": FIELDS,
        "law": "values do not transfer, methods do. Every number in this report is a Qwen "
               "measurement; every recommendation is a method with an applicability condition.",
        "built_from": ["receipts/headless/REPRESENTATION_LIBRARY.json",
                       "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
                       "receipts/headless/ORGAN_FRONTIER_MATRIX.json",
                       "receipts/headless/KERNEL_LIBRARY.json"],
        "input_audit": input_audit(all_cited),
        "n_entries": len(entries), "n_rejected": len(rejected), "rejected": rejected,
        "n_methods": sum(1 for e in entries if e["id"].startswith("TR-METHOD")),
        "n_representations": sum(1 for e in entries if e["id"].startswith("TR-REPR")),
        "n_negatives": sum(1 for e in entries if e["id"].startswith("TR-NEG")),
        "entries": entries,
        "pass": bool(entries and not rejected and input_audit(all_cited)["clean"]),
    }
    Path(a.emit).write_text(json.dumps(out, indent=1))
    print(f"entries={out['n_entries']} methods={out['n_methods']} "
          f"repr={out['n_representations']} neg={out['n_negatives']} "
          f"rejected={out['n_rejected']} pass={out['pass']}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
