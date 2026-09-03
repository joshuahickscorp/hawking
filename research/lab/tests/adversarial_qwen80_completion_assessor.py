#!/usr/bin/env python3
"""Adversarial check: after making the producer legible to the assessor, prove the
assessor still REFUSES tampered evidence. Each case must refuse; a pass is a failure."""
import json, hashlib, pathlib, subprocess, sys, tempfile

G = sys.argv[1]
REPO = pathlib.Path("/Users/scammermike/Downloads/hawking")
R = REPO / "workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-runtime"
BIN = REPO / "workspace/ops/build/rust/debug/examples/ascension_qwen80_l1_full_layer_completion_assessor"

def canon(v): return json.dumps(v, sort_keys=True, separators=(",", ":"))
def jsha(v): return hashlib.sha256(canon(v).encode()).hexdigest()
def reseal(doc):
    d = {k: v for k, v in doc.items() if k != "seal_sha256"}
    doc = dict(doc); doc["seal_sha256"] = jsha(d); return doc
def wrap(doc):
    return {"document": doc, "document_sha256": jsha(doc), "document_seal_sha256": doc["seal_sha256"]}

base = json.loads((R / f"QWEN80_L1_FULL_LAYER_COMPLETION_ASSESSOR_INPUT_{G}.json").read_text())

def build(mutate):
    inp = json.loads(json.dumps(base))
    mutate(inp)
    inp.pop("seal_sha256", None)
    inp["seal_sha256"] = jsha(inp)
    return inp

def run(inp, label):
    with tempfile.TemporaryDirectory() as td:
        src = pathlib.Path(td) / "in.json"; out = pathlib.Path(td) / "out.json"
        src.write_text(json.dumps(inp, sort_keys=True, separators=(",", ":")))
        subprocess.run([str(BIN), "--input", str(src), "--out", str(out)],
                       capture_output=True, text=True)
        if not out.exists():
            print(f"  REFUSED (no report)   {label}"); return True
        d = json.loads(out.read_text())
        earned = d.get("earned_complete_l1_component_only")
        print(f"  {"EARNED" if earned else "REFUSED"}  {label}"
              f"{'' if earned else '  :: ' + json.dumps(d.get('blockers'))[:90]}")
        return not earned

def t_claim_token(i):
    doc = i["fresh_full_layer_inner"]["document"]
    doc["claim_boundary"]["token_generated"] = True
    i["fresh_full_layer_inner"] = wrap(reseal(doc))
def t_outer_fixture(i):
    doc = i["fresh_full_layer_outer"]["document"]
    doc["fixture_or_synthetic"] = True
    i["fresh_full_layer_outer"] = wrap(reseal(doc))
def t_broken_seal(i):
    doc = i["fresh_full_layer_inner"]["document"]
    doc["status"] = "CAPTURED_TOTALLY_DIFFERENT_THING"   # seal NOT recomputed
    i["fresh_full_layer_inner"]["document"] = doc
def t_wrong_provenance(i):
    doc = i["fresh_full_layer_inner"]["document"]
    doc["historical_component_provenance"]["document_seal_sha256"] = "0" * 64
    i["fresh_full_layer_inner"] = wrap(reseal(doc))
def t_nonzero_exit(i):
    doc = i["fresh_full_layer_outer"]["document"]
    doc["child_terminal"]["exit_code"] = 1
    i["fresh_full_layer_outer"] = wrap(reseal(doc))
def t_release_not_performed(i):
    doc = i["fresh_full_layer_release"]["document"]
    doc["actual_release_performed"] = False
    i["fresh_full_layer_release"] = wrap(reseal(doc))
def t_swap_lease(i):
    doc = i["fresh_full_layer_release"]["document"]
    doc["lease_id"] = "f" * 64
    i["fresh_full_layer_release"] = wrap(reseal(doc))

cases = [
    ("baseline untampered (MUST earn)", lambda i: None, True),
    ("inner claims token_generated", t_claim_token, False),
    ("outer marked fixture_or_synthetic", t_outer_fixture, False),
    ("inner content changed, seal stale", t_broken_seal, False),
    ("inner historical provenance seal wrong", t_wrong_provenance, False),
    ("outer child exit_code 1", t_nonzero_exit, False),
    ("release actual_release_performed false", t_release_not_performed, False),
    ("release lease_id swapped", t_swap_lease, False),
]
ok = True
for label, mut, expect_earned in cases:
    inp = build(mut)
    refused = run(inp, label)
    if expect_earned and refused:
        print("    ^ BASELINE DID NOT EARN"); ok = False
    if not expect_earned and not refused:
        ok = False
print("\nADVERSARIAL RESULT:", "PASS - verifier still bites" if ok else "FAIL - verifier leaks")
sys.exit(0 if ok else 1)
