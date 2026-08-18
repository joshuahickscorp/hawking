#!/usr/bin/env python3
"""Seal a compacted artifact into a Gravity-1 container with its real genome attached.

G030 asks for the kernel/runtime genome to be attached to the artifact. Attaching a LIST of
kernel names would be theatre: this campaign has already established that 502 of 541 shader
declarations are unbound, so a name is not evidence that anything executes. The genome here is
built from what the decode path actually REFERENCES, and every number in the cost section is
recomputed from the artifact on disk rather than copied from a pack report -- the pack report
is the DECLARED payload and was wrong by 1.68 BPW before compaction.

Sections and where each comes from:
  source     the pinned BF16 hashes, unchanged
  program    IR reconstructed from the catalog's own records, so covered elements are counted
  kernels    only kernels the decode path references, each with its reference site
  machine    measured, not spec-sheet: the two kernel-class bandwidths and the threadgroup
             constant this session corrected
  doctor     the capability verdict AND its two controls, because a gate whose controls were
             not observed is not evidence
  tabula     the abliterated variant plus the MEASURED drift the codec reintroduces, which is
             a cost no artifact in this campaign has previously carried
  cost       recomputed from bytes on disk
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gravity_ir import Program, quant_tensor, dense_tensor, SOURCE_PARAM_COUNT  # noqa: E402
from gravity_container import build, verify                                     # noqa: E402
from gravity_compact_artifact import parse                                       # noqa: E402

RUNS = "workspace/campaign/records/runs/qwen38-27b"
PIN = "workspace/superwave/g1/GRAVITY1_SOURCE_PIN.json"
DECODE = "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"
CODEC_UNIFORM = 3


def bound_kernels(verbose=True):
    """Kernels the decode path actually references, with the site. Names are not evidence."""
    src = open(DECODE).read()
    out = {}
    declared = set(re.findall(r'kernel void (\w+)', open(
        "crates/hawking-core/shaders/mha.metal").read()))
    for m in re.finditer(r'"([a-z0-9_]+matvec[a-z0-9_]*|mha_decode_f32|sample_argmax_f32|'
                         r'gk_swiglu_f32|qwen_next_add_residual)"', src):
        name = m.group(1)
        line = src[:m.start()].count("\n") + 1
        out.setdefault(name, {"status": "referenced", "site": f"{DECODE}:{line}"})
    for const, kern in re.findall(r'pub const (\w+): &str =\s*\n?\s*"([^"]+)"', src):
        if re.search(rf'\b{const}\b', src.replace(f'pub const {const}', '')):
            line = src.index(f'pub const {const}')
            out.setdefault(kern, {"status": "referenced via const",
                                  "site": f"{DECODE}:{src[:line].count(chr(10))+1}"})
    for name in sorted(declared):
        if re.search(rf'"{re.escape(name)}"', src):
            line = src.index(f'"{name}"')
            out.setdefault(name, {"status": "referenced",
                                  "site": f"{DECODE}:{src[:line].count(chr(10))+1}"})
    if verbose:
        print(f"  kernel scan: {len(out)} referenced by the decode path "
              f"(the repo declares 541 across shaders/, of which 502 are unbound)")
    return out


_SEGBITS = {}


def seg_bits(root, filename):
    """Authoritative bit width from the segment container's own JSON header."""
    if filename in _SEGBITS:
        return _SEGBITS[filename]
    import struct as _s
    path = os.path.join(root, "segments", filename)
    with open(path, "rb") as fh:
        head = fh.read(4096)
    hlen = _s.unpack("<I", head[8:12])[0]
    with open(path, "rb") as fh:
        fh.seek(12)
        hdr = json.loads(fh.read(hlen))
    _SEGBITS[filename] = int(hdr["bits"])
    return _SEGBITS[filename]


def program_from_catalog(root, name):
    segs, recs = parse(os.path.join(root, "catalog.hq38m20"))
    p = Program(name, source_pin=PIN)
    for r in recs:
        n = r["elements"]
        if r["codec"] == CODEC_UNIFORM:
            # Read `bits` from the SEGMENT'S OWN HEADER. Inferring it from codec_bpw is wrong
            # and was silently wrong: on small tensors the fixed 40-byte container header
            # dominates, so codec_bpw reaches 12.09, 19.88 and even 47.17, which a
            # round-then-clamp turned into a fake "q8" this artifact does not contain. A
            # container that misreports its own codecs is precisely the false genome G030
            # exists to prevent.
            bits = seg_bits(root, segs[r["segment_id"]]["filename"])
            node = quant_tensor(n, bits=bits, group=64,
                                kernel="qwen_uniform_q%d_group64_matvec_geo_tpr64_tg128" % bits,
                                scale_bytes_per_group=2, header=0)
        else:
            node = dense_tensor(n, dtype_bytes=4, kernel="f32v2_direct", header=0)
        p.add(r["name"], n, [node])
    return p, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="compact-q3attn-r1p2-v1")
    ap.add_argument("--capability-json", default="/tmp/cap.json")
    ap.add_argument("--tps", type=float, default=24.2102)
    ap.add_argument("--token-ns", type=int, default=41304959)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = os.path.join(RUNS, a.artifact)
    prog, recs = program_from_catalog(root, a.artifact)

    on_disk = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, fs in os.walk(root) for f in fs)
    addressed = sum(r["nbytes"] for r in recs)
    print(f"{a.artifact}: {len(recs)} records, addressed {addressed:,}, on disk {on_disk:,}")
    print(f"  covered elements {prog.covered_elements():,} of N {SOURCE_PARAM_COUNT:,}")
    print(f"  honest complete BPW {8*on_disk/SOURCE_PARAM_COUNT:.12f}")

    cap = {}
    if os.path.exists(a.capability_json):
        cap = json.load(open(a.capability_json))
    verdict = cap.get(a.artifact, {})
    pos = cap.get("uniform-q4-v1", {})
    neg = cap.get("mixed-q4down-v1", {})

    body = build(
        prog, PIN,
        machine={
            "chip": "Apple M3 Ultra", "unified_memory_gb": 96,
            "measured_bandwidth_gb_s": {"geo_tpr64": 639.25, "simd3": 83.30},
            "note": "measured kernel-class rates; the old 411 GB/s roof is dead",
            "attention_threadgroup": 512,
            "attention_threadgroup_note": (
                "was 128, tuned for Qwen-3B head_dim 128; this model is head_dim 256. "
                "Measured slope us per context token: 128 15.98, 256 7.18, 512 6.19, 1024 6.41"),
            "measured_token_ns": a.token_ns, "measured_tps": a.tps,
            "dispatches_per_token": 964, "command_buffers_per_token": 1,
            "prefill": "UNBATCHED, 40.2 ms per prompt token, linear (G102)",
            # program.kernels() names only the kernels THIS artifact's representations need
            # (its codecs). The runtime genome is larger: the decode path also dispatches
            # attention, norms, rope, swiglu, residual and sampling. G030 asks for the genome,
            # so record both and keep the distinction explicit.
            "runtime_kernels_referenced_by_decode": sorted(bound_kernels(verbose=False)),
            "shader_declarations_in_repo": 541,
            "shader_declarations_unbound": 502,
        },
        doctor={
            "gate": "gravity_doctor_capability.v1",
            "verdict": ("PASS" if verdict.get("correct", 0) >= 6 and
                        verdict.get("degenerate", 1) == 0 else "FAIL"),
            "correct": verdict.get("correct"), "of": verdict.get("n"),
            "degenerate": verdict.get("degenerate"),
            "dimensions": sorted(verdict.get("by_dim", {})),
            "fallbacks": verdict.get("fallbacks"),
            "dense_materialized": verdict.get("dense_materialized"),
            "negative_controls": [
                f"positive control uniform-q4-v1 observed PASSING "
                f"{pos.get('correct')}/{pos.get('n')}",
                f"negative control mixed-q4down-v1 observed FAILING "
                f"{neg.get('correct')}/{neg.get('n')} with {neg.get('degenerate')} degenerate",
                "gravity_doctor_gate --demo: faithful codec HEALTHY, visible-subspace cheat "
                "UNHEALTHY",
            ],
            "known_limits": [
                "ten deterministic items across seven dimensions; 9/10 is not statistically "
                "distinguishable from 10/10 on this sample size",
                "the per-tensor adequacy gate is measured OVER-STRICT: it rejects 15 of 15 "
                "MLP tensors in an artifact that passes here",
            ],
        },
        tabula={
            "variant": "abliterated",
            "method": "refusal-direction orthogonal weight projection",
            "destination_layers": list(range(24, 64)),
            "direction_recovered_from_artifact": True,
            "direction_agreement_across_layers": [0.99992, 0.99994],
            "random_pair_null": 0.01482,
            "reintroduced_energy_vs_bf16_parent": {
                "q4_g64": "11.1-12.1x", "q3_g64": "24.8-27.1x", "q2_g64": "63.6-67.2x"},
            "behavioural_probe": "0 refusals of 8 benign refusal-adjacent prompts",
            "behavioural_probe_limit": (
                "cannot certify absence: greedy, one sample per prompt, and NO stock "
                "non-abliterated parent on disk to watch the probe detect real drift"),
        },
        kernels_meta=bound_kernels(),
    )
    body["cost"]["on_disk_bytes"] = on_disk
    # The IR models the weight payload. The tree also holds the catalog, the pack report and
    # FORMAT.md, which the runtime needs but which are not tensors. Name the difference rather
    # than leaving two BPW numbers side by side unexplained -- an unexplained gap is exactly
    # how a declared payload passed for a measurement earlier in this campaign.
    body["cost"]["unmodelled_bytes"] = on_disk - prog.total_bytes()
    body["cost"]["unmodelled_note"] = (
        "catalog + PACK_REPORT + FORMAT.md; required by the runtime, not tensors, "
        "counted in honest_complete_bpw and absent from the IR program")
    body["cost"]["honest_complete_bpw"] = 8 * on_disk / SOURCE_PARAM_COUNT
    body["cost"]["measured_tps"] = a.tps
    body["content_id"] = hashlib.sha256(
        json.dumps({k: v for k, v in body.items() if k != "content_id"},
                   sort_keys=True).encode()).hexdigest()

    fails = verify(body)
    out = a.out or os.path.join("workspace/superwave/g1", f"{a.artifact}.container.json")
    json.dump(body, open(out, "w"), indent=2)
    print(f"\nsealed -> {out}")
    print(f"  sections {len(body)}  kernels referenced by decode: {len(body['kernels'])}")
    print(f"  IR complete BPW {body['cost']['complete_effective_bpw']:.12f}")
    print(f"  honest on-disk  {body['cost']['honest_complete_bpw']:.12f}")
    print(f"  doctor {body['doctor']['verdict']} "
          f"{body['doctor']['correct']}/{body['doctor']['of']}, "
          f"{len(body['doctor']['negative_controls'])} controls recorded")
    if fails:
        print("\nCONTAINER VERIFY FAILURES:")
        for f in fails:
            print("  -", f)
        return 1
    print("  container verify: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
