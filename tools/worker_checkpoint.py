#!/usr/bin/env python3
"""G132: worker checkpoint and rebind, with the measurement-invalidation rule.

Directive S16 requires every worker to persist SIX things before eviction or
rebind: hypothesis, code changes, measurements, negative science, current
blocker, next experiment. S17 then requires that on promotion of a better
parent the workers are checkpointed, rebound to the new parent, and resume the
SAME obligations -- "No progress reset."

G131 is why this is routine rather than exceptional: each resident worker wires
15.59 GB and the measured ceiling is three, so eviction is the normal operating
condition of this machine, not a failure mode.

The part that is easy to get wrong, and the reason this is a tool rather than a
JSON schema: NOT ALL CARRIED STATE SURVIVES A REBIND.

  hypothesis        survives -- it is about a family of representations
  negative science  survives -- "q2 MLP is dead" is a statement about a codec
  code changes      survives -- a patch is parent-independent
  blocker           survives -- a missing input stays missing
  next experiment   survives -- the plan is what rebind exists to preserve

  MEASUREMENTS DO NOT SURVIVE. A token time, a BPW, a capability score is a
  statement about ONE artifact. Carrying "36.13 ms/token" across a rebind to a
  new parent produces a number that describes a model no longer resident, and
  the campaign has already been burned by exactly this: a lane result is valid
  only against the HEAD it measured on.

So rebind does not copy measurements forward. It moves them to `stale` and
stamps the parent they were taken against, which makes a stale number
unusable-by-accident but still readable-on-purpose.

The control is the discipline: a checkpoint MISSING a required field must be
REFUSED, and a rebind from an incomplete checkpoint must FAIL. A protocol that
accepts anything has not been shown to enforce anything.

  ./tools/worker_checkpoint.py demo
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
STORE = ROOT / "workspace/campaign/records/workers"

REQUIRED = ["hypothesis", "code_changes", "measurements",
            "negative_science", "blocker", "next_experiment"]

# Rebind partition. The whole point of the tool.
SURVIVES = ["hypothesis", "code_changes", "negative_science",
            "blocker", "next_experiment"]
INVALIDATED = ["measurements"]


class Incomplete(Exception):
    pass


def checkpoint(worker: str, parent: str, obligations: list[str], state: dict,
               store: pathlib.Path = STORE) -> pathlib.Path:
    """Write a checkpoint. Refuses unless all six required fields are present
    AND non-empty -- a field present but blank is the same as absent."""
    missing = [k for k in REQUIRED if not state.get(k)]
    if missing:
        raise Incomplete(f"worker {worker}: refusing to checkpoint, missing or "
                         f"empty {missing} (S16 requires all of {REQUIRED})")
    doc = {
        "schema": "hawking.nos.worker_checkpoint.v1",
        "worker": worker,
        "parent": parent,
        "obligations": obligations,
        "state": {k: state[k] for k in REQUIRED},
        "rebind_count": 0,
        "stale": {},
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    store.mkdir(parents=True, exist_ok=True)
    p = store / f"{worker}.checkpoint.json"
    p.write_text(json.dumps(doc, indent=2) + "\n")
    return p


def rebind(path: pathlib.Path, new_parent: str) -> dict:
    """Bind a checkpointed worker to a new parent.

    Obligations and parent-independent state carry. Measurements do NOT: they
    move to `stale` stamped with the parent they were taken against, so a
    resumed worker cannot cite them by accident."""
    doc = json.loads(path.read_text())
    missing = [k for k in REQUIRED if not doc.get("state", {}).get(k)]
    if missing:
        raise Incomplete(f"refusing to rebind {path.name}: incomplete checkpoint, "
                         f"missing {missing}")
    old = doc["parent"]
    if old == new_parent:
        raise Incomplete(f"refusing to rebind {doc['worker']}: parent unchanged "
                         f"({old}) -- a rebind that changes nothing hides whether "
                         "the protocol ran")

    for k in INVALIDATED:
        doc.setdefault("stale", {}).setdefault(old, {})[k] = doc["state"][k]
        doc["state"][k] = []

    doc["parent"] = new_parent
    doc["rebind_count"] += 1
    doc["rebind_note"] = (
        f"rebound {old} -> {new_parent}. {SURVIVES} carried unchanged; "
        f"{INVALIDATED} INVALIDATED and moved to stale[{old}] because a "
        "measurement is a statement about one artifact, not about a family. "
        "Obligations carried in full -- no progress reset.")
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def demo():
    """Three real campaign workers in the S12 roles, checkpointed and rebound,
    plus the two controls."""
    store = STORE
    workers = [
        ("worker-a-representation", [
            "G133", "G134"], {
            "hypothesis": "MLP is the binding organ, not attention: three artifacts at "
                          "1.29/2.09/3.18 BPW all share MLP 0.848 and all collapse.",
            "code_changes": ["tools/gravity_planes_functional.py",
                             "tools/gravity_joint_optimizer.py"],
            "measurements": [{"what": "uniform q3 MLP holds >=0.95 on 192/192",
                              "parent": "uniform-q4-v1"},
                             {"what": "complete BPW 3.6138", "parent": "mixed-q3mlp-v1"}],
            "negative_science": ["q2 MLP dead", "q2 attention fluent-but-wrong",
                                 "depth-weighted q2 allocation dead",
                                 "shared expert basis refuted, experts orthogonal cos 0.004"],
            "blocker": "no packer emits correction streams, so planes cannot be assembled",
            "next_experiment": "alpha=0.75 scaling beat absmax by 18% on both organs "
                               "tested -- take it, it is free",
        }),
        ("worker-b-kernel-nx", ["G135", "G136"], {
            "hypothesis": "the 7.8 ms non-GEMV residual is the wall, not weight movement.",
            "code_changes": ["shaders/qwen_uniform_q4.metal RK baselines",
                             "examples/ascension_qwen38_nongemv_census.rs"],
            "measurements": [{"what": "non-GEMV residual 7.949/7.837/7.772 ms, three ways",
                              "parent": "uniform-q4-v1"},
                             {"what": "isolated dispatch floor ~17 us", "parent": "uniform-q4-v1"}],
            "negative_science": ["R x K tiling did NOT spill at R=8, all 20 kernels report "
                                 "maxTotalThreadsPerThreadgroup 1024",
                                 "density alone caps at 99.11 TPS, multi-token at 93.8"],
            "blocker": "no decode binding for planes, so a planes artifact cannot be timed",
            "next_experiment": "attention path is 253x its per-element reference cost -- "
                               "profile it before touching MLP kernels",
        }),
        ("worker-c-doctor-tabula", ["G137", "G138"], {
            "hypothesis": "divergence and capability are independent axes.",
            "code_changes": ["tools/greedy_divergence.py", "tools/doctor_seal.py",
                             "tools/tabula_drift.py"],
            "measurements": [{"what": "g032 diverges 81.6% at equal capability",
                              "parent": "uniform-q4-v1"},
                             {"what": "Tabula ladder off recorded by constant 2.47-2.60x",
                              "parent": "uniform-q4-v1"}],
            "negative_science": ["greedy divergence ranks candidates OPPOSITELY to capability",
                                 "cosine gate was blind to magnitude all campaign"],
            "blocker": "behavioral probe too weak to certify absence of the removed direction",
            "next_experiment": "widen the battery past 60 items on the axis Tabula moves",
        }),
    ]
    print("CHECKPOINT (parent uniform-q4-v1)")
    paths = []
    for w, obl, st in workers:
        p = checkpoint(w, "uniform-q4-v1", obl, st, store)
        paths.append(p)
        print(f"  {w:<26} obligations {','.join(obl)}  "
              f"{len(st['measurements'])} measurements  "
              f"{len(st['negative_science'])} negatives")

    print("\nCONTROL 1: checkpoint missing a required field must REFUSE")
    bad = dict(workers[0][2])
    bad["next_experiment"] = ""
    try:
        checkpoint("worker-bad", "uniform-q4-v1", ["G999"], bad, store)
        print("  FAILED -- accepted an incomplete checkpoint, protocol enforces nothing")
        ctrl1 = False
    except Incomplete as e:
        print(f"  REFUSED: {e}")
        ctrl1 = True

    print("\nREBIND uniform-q4-v1 -> mixed-q3mlp-v1")
    docs = []
    for p in paths:
        d = rebind(p, "mixed-q3mlp-v1")
        docs.append(d)
        print(f"  {d['worker']:<26} obligations {','.join(d['obligations'])} CARRIED  "
              f"measurements {len(d['state']['measurements'])} "
              f"(was {len(d['stale']['uniform-q4-v1']['measurements'])}, now stale)  "
              f"negatives {len(d['state']['negative_science'])} CARRIED")

    print("\nCONTROL 2: rebind from an incomplete checkpoint must FAIL")
    holed = store / "worker-holed.checkpoint.json"
    h = json.loads(paths[0].read_text())
    h["state"]["blocker"] = ""
    holed.write_text(json.dumps(h, indent=2) + "\n")
    try:
        rebind(holed, "some-other-parent")
        print("  FAILED -- rebound an incomplete checkpoint")
        ctrl2 = False
    except Incomplete as e:
        print(f"  REFUSED: {e}")
        ctrl2 = True
    holed.unlink()

    carried = all(d["obligations"] == w[1] for d, w in zip(docs, workers))
    dropped = all(d["state"]["measurements"] == [] for d in docs)
    kept = all(d["state"]["negative_science"] for d in docs)
    print(f"\nno progress reset: obligations carried on all three -> {carried}")
    print(f"measurements invalidated on rebind        -> {dropped}")
    print(f"negative science survived rebind          -> {kept}")
    return {"controls_pass": ctrl1 and ctrl2, "obligations_carried": carried,
            "measurements_invalidated": dropped, "negative_science_survived": kept,
            "workers": [d["worker"] for d in docs], "docs": docs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["demo"])
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    r = demo()
    if a.out:
        doc = {
            "schema": "hawking.nos.worker_rebind_protocol.v1",
            "obligation": "G132 -- worker checkpoint / rebind protocol",
            "required_fields": REQUIRED,
            "survives_rebind": SURVIVES,
            "invalidated_on_rebind": INVALIDATED,
            "why_measurements_do_not_survive":
                "a token time, a BPW or a capability score is a statement about ONE artifact. "
                "Carrying it across a rebind describes a model that is no longer resident. The "
                "campaign has already been burned by this exact shape: a lane result is valid "
                "only against the HEAD it measured on. Rebind moves measurements to stale[] "
                "stamped with the parent they were taken against -- unusable by accident, "
                "readable on purpose.",
            "why_eviction_is_routine":
                "G131 measured 15.59 GB wired per resident worker and a hard ceiling of three, "
                "so eviction is the normal operating condition of this machine and the "
                "checkpoint path is the common path, not the failure path.",
            "controls": {
                "incomplete_checkpoint_refused": r["controls_pass"],
                "note": "a protocol that accepts anything enforces nothing, so both controls "
                        "run inside the same invocation as the successful path"},
            "results": {k: r[k] for k in
                        ("obligations_carried", "measurements_invalidated",
                         "negative_science_survived", "workers")},
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, cwd=ROOT).stdout.strip(),
        }
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")
    ok = r["controls_pass"] and r["obligations_carried"] and r["measurements_invalidated"]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
