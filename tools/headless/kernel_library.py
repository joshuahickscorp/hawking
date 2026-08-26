#!/usr/bin/env python3
"""KERNEL LIBRARY — no anonymous performance wins.

A kernel that is fast but cannot say what machine it was fast on, against what source,
with what correctness contract, does not transfer. The checker below REFUSES such an
entry. It was written before the library was rebuilt, and it is shown rejecting an
incomplete entry before it is shown accepting a complete one.

Graph-collapsed native operators are a different thing from primitive kernels and get
their own library. The goal is evidence-backed operator families, not one universal
megakernel -- an 8-layer f16 fused megakernel measured 4.4x SLOWER here, and that
negative is carried so nobody re-runs it.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
KL = RH / "KERNEL_LIBRARY.json"
SO = RH / "SUPEROPERATOR_LIBRARY.json"
BUILD = REPO / "workspace/ops/build/rust/release-fast/examples"

REQUIRED = ["kernel_identity", "organ_identity", "representation_identity",
            "machine_identity", "shader", "shader_sha256", "compiled_identity",
            "specialization", "memory_layout", "competence", "measurements",
            "supported_capability_regime", "parity", "citations"]

# A field may hold an honest ABSENT with a reason. It may not be missing, and it may not
# be present-but-empty: that is how an unmeasured number turns into an implied zero.
def field_state(v):
    if v is None:
        return "MISSING"
    if isinstance(v, dict):
        k = v.get("kind")
        if k == "ABSENT":
            return "ABSENT_WITH_REASON" if v.get("absent_reason") else "ABSENT_NO_REASON"
        if "value" in v and v["value"] is None and not v.get("absent_reason"):
            return "NULL_NO_REASON"
    return "PRESENT"


class Refused(Exception):
    pass


def check(entry):
    missing = [f for f in REQUIRED if f not in entry]
    if missing:
        raise Refused(f"{entry.get('kernel_identity','<no id>')}: missing field(s) {missing}")
    bad = [f for f in REQUIRED if field_state(entry[f]) in ("ABSENT_NO_REASON", "NULL_NO_REASON")]
    if bad:
        raise Refused(f"{entry['kernel_identity']}: field(s) {bad} are empty with no "
                      f"absent_reason -- an unexplained blank reads as a measured zero")
    return entry


# Executable parity contracts. Each names a real binary in the repo build dir; a kernel
# whose contract has no runnable binary is recorded as such rather than assumed correct.
CONTRACTS = {
    "q2f_group64_matvec": "q2f_parity",
    "q2f_group64_matvec_geo_tpr64_tg128": "q2f_parity",
    "q2f_group64_matvec_gate_up_swiglu_geo_tpr64_tg128": "q2f_parity",
    "affine2_group32_matvec_geo_tpr64_tg128": "q2f_parity",
    "affine2_group64_matvec_geo_tpr64_tg128": "affine2_parity",
    "qwen_affine_q2_group32_matvec_geo_tpr64_tg128": "affine2_parity",
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128": "affine2_parity",
    "affine2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128": "affine2_parity",
}

# Graph-collapsed operators: the source nodes they fuse, and why that is semantically
# allowed. Separated from primitives so a fusion win is never confused with a codec win.
SUPEROPS = [
    {"operator": "gate_up_swiglu",
     "kernels": ["q2f_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
                 "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
                 "affine2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128"],
     "collapses": ["mlp.gate_proj matvec", "mlp.up_proj matvec", "silu", "elementwise mul"],
     "semantic_justification": "gate and up read the same activation vector and their "
                               "outputs are consumed only by SwiGLU; the intermediate "
                               "vectors are not observable outside the fused region",
     "cost_delta": {"kind": "CITED", "source": "receipts/headless/NOETIC_DISPATCH_FUSION.json"}},
    {"operator": "norm_projection",
     "kernels": ["qwen80_add_residual_rmsnorm_tg"],
     "collapses": ["residual add", "rmsnorm"],
     "semantic_justification": "the residual sum is consumed only by the norm",
     "cost_delta": {"kind": "ABSENT", "value": None,
                    "absent_reason": "no isolated fusion measurement for this operator"}},
    {"operator": "recurrent_fused_update",
     "kernels": ["qwen38_gated_delta_decode_vi_simd_ba_f4"],
     "collapses": ["delta state read", "gated update", "state write"],
     "semantic_justification": "the intermediate state is never read outside the step",
     "cost_delta": {"kind": "CITED", "source": "receipts/headless/DELTANET_ORGAN.json"}},
]
NEGATIVE_SUPEROPS = [
    {"operator": "multi_layer_megakernel",
     "verdict": "REFUTED",
     "physical_reason": "an 8-layer f16 fused megakernel measured 4.4x SLOWER than the "
                        "unfused sequence; use_resource 2.62us vs 4.5us set_buffer",
     "reopen_condition": "a dispatch model where per-dispatch overhead again dominates "
                         "the fused region's occupancy loss",
     "law": "build evidence-backed operator families, never chase one universal megakernel"},
]


def run_contract(binary):
    p = BUILD / binary
    if not p.exists():
        return {"runnable": False, "binary": str(p), "why": "not built in this repo's build dir"}
    r = subprocess.run([str(p)], capture_output=True, text=True, timeout=600)
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    return {"runnable": True, "binary": str(p), "exit_code": r.returncode,
            "status_line": tail[0], "passed": "PASS" in tail[0],
            "stderr_tail": (r.stderr or "")[-400:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", metavar="PATH")
    ap.add_argument("--emit-superops", metavar="PATH", default=str(SO))
    ap.add_argument("--reject-demo", action="store_true")
    ap.add_argument("--run-contracts", action="store_true")
    a = ap.parse_args()

    if a.reject_demo:
        for bad in ({"kernel_identity": "incomplete_kernel", "organ_identity": "mlp_down"},
                    {**{f: {"kind": "CITED", "value": 1} for f in REQUIRED},
                     "kernel_identity": "blank_field_kernel",
                     "parity": {"kind": "ABSENT", "value": None}}):
            try:
                check(bad)
            except Refused as r:
                print("REFUSED:", r)
        return 0

    src = json.load(open(KL))
    kernels = src["kernels"]
    accepted, rejected = [], []
    for k in kernels:
        try:
            accepted.append(check(k))
        except Refused as r:
            rejected.append(str(r))

    contracts, ran = {}, {}
    for k in accepted:
        b = CONTRACTS.get(k["kernel_identity"])
        contracts[k["kernel_identity"]] = b or None
        if a.run_contracts and b and b not in ran:
            ran[b] = run_contract(b)

    n_absent = sum(1 for k in accepted for f in REQUIRED
                   if field_state(k[f]) == "ABSENT_WITH_REASON")
    out = {
        "schema": "hawking.headless.kernel_library.v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/headless/kernel_library.py",
        "obligation": "G019 — KERNEL_LIBRARY + SUPEROPERATOR_LIBRARY (directive §51, §52)",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False, "unmeasured_is_absent": True,
        "required_fields": REQUIRED,
        "law": "no anonymous performance wins: a field may be ABSENT with a reason, never "
               "missing and never blank without one",
        "n_kernels": len(kernels), "n_complete": len(accepted),
        "n_rejected": len(rejected), "rejected": rejected,
        "n_absent_fields_with_reason": n_absent,
        "parity_contracts": contracts,
        "n_kernels_without_a_runnable_contract": sum(1 for v in contracts.values() if not v),
        "contract_runs": ran,
        "kernels": accepted,
        "preserved_from_v1": True,
        "pass": bool(accepted and not rejected),
    }
    if a.rebuild:
        Path(a.rebuild).write_text(json.dumps(out, indent=1))
    sops = {
        "schema": "hawking.headless.superoperator_library.v1",
        "generated_at": out["generated_at"],
        "obligation": "G019 — SUPEROPERATOR_LIBRARY (directive §52)",
        "law": "graph-collapsed operators are tracked separately from primitive kernels; "
               "a fusion win must never be read as a codec win",
        "operators": SUPEROPS, "refuted": NEGATIVE_SUPEROPS,
        "n_operators": len(SUPEROPS), "n_refuted": len(NEGATIVE_SUPEROPS),
    }
    Path(a.emit_superops).write_text(json.dumps(sops, indent=1))
    print(f"kernels={out['n_kernels']} complete={out['n_complete']} rejected={out['n_rejected']} "
          f"absent_with_reason={n_absent} no_contract={out['n_kernels_without_a_runnable_contract']} "
          f"superops={sops['n_operators']} refuted={sops['n_refuted']} pass={out['pass']}")
    for b, r in ran.items():
        print(f"  contract {b}: {r.get('status_line') or r.get('why')}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
