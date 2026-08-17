#!/usr/bin/env python3
"""Gravity-1 container: a physical model program plus everything needed to trust and
reproduce it.

A quantized checkpoint is a bag of tensors. A Gravity-1 artifact has to answer more:
what source it came from, what program represents it, what it costs, what kernels
consume it, what machine it was built for, what Doctor checked, and what behavioural
identity it is supposed to preserve.

Sections, each one present because its absence has already cost something here:

  source        content hashes of the raw parent. A campaign once used a lossy
                intermediate as its semantic source of truth for weeks.
  program       the IR, from which complete BPW is COMPUTED rather than declared. A
                pack reported 3.6138 while carrying an uncounted 1.814 GB sidecar.
  shared_pool   content-addressed objects, counted once. Sharing is worthless if
                double-counted and free if forgotten; neither is honest.
  kernels       every representation names its consumer. No execution path, no
                representation.
  machine       the profile the cost model was calibrated against. A roof is
                conditioned on the execution genome, not on physics.
  doctor        the seal: which gate ran, its verdict, its negative controls.
  tabula        behavioural provenance. This patient is ABLITERATED; compression must
                not quietly restore suppression by regressing toward the stock manifold.

Identity binds to CONTENT HASHES, never to path strings. The current lineage binds
labelled hashes of paths, which is naming, not provenance.
"""
from __future__ import annotations
import hashlib, json, os, sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gravity_ir import Program, SOURCE_PARAM_COUNT  # noqa: E402

SCHEMA = "hawking.gravity1.container.v1"


def _sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def build(program: Program, source_pin_path, machine, doctor, tabula, kernels_meta=None):
    pin = json.load(open(source_pin_path))
    prog = json.loads(program.to_json())
    body = {
        "schema": SCHEMA,
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "pin_path": source_pin_path,
            "pin_sha256": hashlib.sha256(open(source_pin_path, "rb").read()).hexdigest(),
            "upstream": pin["upstream"],
            "bpw_denominator": pin["parameter_census"]["bpw_denominator"],
        },
        "program": prog,
        "shared_pool": program.pool.objects,
        "kernels": {k: (kernels_meta or {}).get(k, {"status": "declared"})
                    for k in program.kernels()},
        "machine": machine,
        "doctor": doctor,
        "tabula": tabula,
        "cost": {
            "total_bytes": program.total_bytes(),
            "site_bytes": program.site_bytes(),
            "shared_bytes": program.total_bytes() - program.site_bytes(),
            "complete_effective_bpw": program.complete_bpw(),
            "active_bytes_per_token": program.active_bytes_per_token(),
        },
    }
    # the seal covers everything above it, so any later edit is detectable
    body["content_id"] = _sha_json(body)
    return body


REQUIRED = ("schema", "source", "program", "shared_pool", "kernels", "machine",
            "doctor", "tabula", "cost", "content_id")


def verify(body, expect_bpw_denominator=SOURCE_PARAM_COUNT):
    """Structural + integrity check. Returns a list of failures; empty means good."""
    fails = []
    for k in REQUIRED:
        if k not in body:
            fails.append(f"missing section: {k}")
    if fails:
        return fails

    stated = body["content_id"]
    recomputed = _sha_json({k: v for k, v in body.items() if k != "content_id"})
    if stated != recomputed:
        fails.append(f"content_id mismatch: stated {stated[:16]} recomputed {recomputed[:16]}")

    if body["source"]["bpw_denominator"] != expect_bpw_denominator:
        fails.append(f"bpw_denominator {body['source']['bpw_denominator']} is not the "
                     f"original source parameter count {expect_bpw_denominator}")

    cost = body["cost"]
    bpw = 8 * cost["total_bytes"] / body["source"]["bpw_denominator"]
    if abs(bpw - cost["complete_effective_bpw"]) > 1e-9:
        fails.append(f"declared BPW {cost['complete_effective_bpw']} != recomputed {bpw}")

    # every shared object must actually be referenced, and every reference must resolve
    refs = {c for s in body["program"]["sites"] for t in s["terms"] for c in t["shared_refs"]}
    pool = set(body["shared_pool"])
    for c in refs - pool:
        fails.append(f"site references unknown shared object {c}")
    for c in pool - refs:
        fails.append(f"shared object {c} stored but never referenced (counted, unused)")

    # a shared object counted per-site instead of once would show up here
    shared = sum(body["shared_pool"][c]["nbytes"] for c in refs)
    if abs((cost["total_bytes"] - cost["site_bytes"]) - shared) > 0:
        fails.append("shared bytes are not the pooled objects counted exactly once")

    for k, v in body["kernels"].items():
        if not k or v is None:
            fails.append(f"representation with no kernel: {k}")

    if not body["tabula"].get("variant"):
        fails.append("tabula.variant absent: behavioural identity unrecorded")
    if not body["doctor"].get("negative_controls"):
        fails.append("doctor seal has no negative controls: a gate never observed failing")
    return fails


def demo():
    """Write a container, read it back, and prove each guard fires."""
    from gravity_ir import quant_tensor, shared_basis
    pin = "workspace/superwave/g1/GRAVITY1_SOURCE_PIN.json"
    p = Program("demo-shared", source_pin=pin)
    b = p.pool.put("SharedBasis", nbytes=256 * 5120 * 2, rank=256)
    e = 17408 * 5120
    for l in range(4):
        p.add(f"L{l}.mlp.down", e, [
            shared_basis(e, 2, b, "fused_basis_gemv"),
            quant_tensor(e // 100, 4, 128, "sparse_correct_accum"),
        ])

    body = build(
        p, pin,
        machine={"chip": "Apple M3 Ultra", "unified_memory_gb": 96,
                 "measured_bandwidth_gb_s": 639.25},
        doctor={"gate": "gravity_doctor_gate.v1", "verdict": "PENDING",
                "negative_controls": ["visible_subspace_only", "unseen_subspace_corruption",
                                      "critical_channel_deletion", "sparse_row_corruption",
                                      "q2_g128_honest_but_too_cheap"]},
        tabula={"variant": "abliterated",
                "method": "refusal-direction orthogonal weight projection",
                "destination_layers": list(range(24, 64))},
    )

    path = "/tmp/gravity1-container-demo.json"
    json.dump(body, open(path, "w"), indent=2)
    back = json.load(open(path))
    assert back == body, "round-trip changed the container"

    fails = verify(back)
    assert not fails, f"clean container rejected: {fails}"
    print(f"container demo: round-trip OK, {len(REQUIRED)} sections, "
          f"complete BPW {body['cost']['complete_effective_bpw']:.6f}")
    print(f"  shared bytes {body['cost']['shared_bytes']} counted once across "
          f"{len(body['program']['sites'])} sites")

    # Tamper detection: mutate WITHOUT resealing. Only the seal should catch this.
    d = json.loads(json.dumps(body))
    d["cost"]["total_bytes"] = 1
    f = verify(d)
    assert f and "content_id mismatch" in f[0], f"seal did not catch tampering: {f}"
    print(f"  seal catches tampering            -> {f[0][:64]}")

    # Semantic guards must catch a container BUILT wrong, which has a VALID seal.
    # Re-sealing after each mutation is the point: otherwise the seal fires first and
    # the specific guard is never exercised -- a check that cannot be observed failing.
    def reseal(d):
        d.pop("content_id", None)
        d["content_id"] = _sha_json(d)
        return d

    checks = [
        ("declared BPW not recomputable", lambda d: d["cost"].__setitem__("complete_effective_bpw", 0.5)),
        ("denominator is not the source count", lambda d: d["source"].__setitem__("bpw_denominator", 999)),
        ("pooled object stored but unreferenced", lambda d: d["shared_pool"].__setitem__("orphan", {"kind": "x", "nbytes": 1})),
        ("behavioural identity unrecorded", lambda d: d["tabula"].__setitem__("variant", "")),
        ("gate with no negative controls", lambda d: d["doctor"].__setitem__("negative_controls", [])),
        ("missing section", lambda d: d.pop("machine")),
    ]
    for label, mutate in checks:
        d = json.loads(json.dumps(body))
        mutate(d)
        reseal(d)
        f = verify(d)
        assert f, f"guard did not fire: {label}"
        assert "content_id mismatch" not in f[0], f"{label}: seal fired instead of the guard"
        print(f"  guard fires: {label:<38} -> {f[0][:62]}")
    print("container demo: PASS (seal + every semantic guard observed failing on its own)")


if __name__ == "__main__":
    demo()
