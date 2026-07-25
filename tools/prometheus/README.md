# Prometheus pipeline spine

Capability-conditioned byte allocation over a real model's tensor graph. Deterministic,
training-free, no served model required. This is the gate-independent core of legs **L1**
(spine) and **L2** (RandomPolicy control) from
[`docs/plans/PROMETHEUS_LEG_PLAN.md`](../../docs/plans/PROMETHEUS_LEG_PLAN.md).

## What it does (runs today)

```text
P0  graph        read per-category element/byte counts from a sealed logical-weight ledger
P1  profile      load an allocation profile (profiles/prometheus/*.json)
P2  census       compressible vs control-sensitive; non-target components
L4  vocab prune  shrink embedding/lm_head to a retained-vocab target (byte win is exact)
P5  tiering      category -> T0/T1/T2/T3 per profile; routed_expert -> coalition + remainder
P6  allocation   bytes per block at exact-rational tier rates; Uniform / Conditioned / Random
    byte-decomp  main/protected/excised, two rates, reconciled to physical bytes
```

## What is GATED (needs a served runtime, marked in every output)

```text
P3  causal probe                       needs S0.8 served runtime
P4  cartography coalition membership    only coalition SIZE is used today; WHICH experts is gated
P8  capability evaluation (G1-G8)       needs served runtime + Lean-lite (leg L3)
    vocab-prune coverage safety          the byte win is exact; whether the target passes
                                         G4+G6 is gated
```

**No capability claim is made by any artifact here.** Everything is allocation arithmetic.

## Run

```bash
python3 tools/prometheus/test_prometheus.py                              # 9 invariants, must pass
python3 tools/prometheus/prometheus.py --profile math-v1                 # one plan to stdout
python3 tools/prometheus/prometheus.py --profile random-v1 --out /tmp/r.json
python3 tools/prometheus/prometheus.py --profile math-v1 --vocab-retain 100000
```

Sealed run over all profiles on real GLM-5.2 metadata:
[`reports/prometheus/GLM52_allocation_plans.json`](../../reports/prometheus/GLM52_allocation_plans.json).
Pre-registration: [`preregistrations/PROM-001-*`](../../preregistrations/).

## The one thing to understand

The routed experts are ~97.5% of the weights, so at category granularity every profile
allocates almost identically. **The real Claim-A surface is which experts form the protected
coalition.** Because experts are near-equal in size, protecting a random k-subset costs the
same bytes as protecting the math k-subset — so `random-v1` is exactly byte-matched to
`math-v1` (asserted across the whole coalition sweep). That byte-match is what makes the
control valid: any future "Math beats Random" cannot be "Math spent more bytes."

Deferred by design: the `crates/hawking-prometheus` Rust orchestrator. Per the plan's own
rule, no port to Rust until a Python stage proves worth it.
