# Hawking Gravity Runbook

Gravity is the sub-bit-first execution law, enforced in code inside the existing
single-owner successor controller. It is additive and **default-off**. It launches
nothing: model work enters only through the existing admission, queue, lease, event
log, checkpoint, watchdog, and transition boundary, and only when every activation
gate holds.

Doctrine:

> Gravity pulls every model toward the smallest complete physical representation.
> Doctor fights to preserve the model during the fall. Event Horizon marks the lowest
> point where capability survives. Escape above sub-bit is permitted only when evidence
> proves survival below it is impossible within the declared contract.

The correct success statement, today, is **not** "Hawking achieved sub-bit." It is:

> Hawking is now compelled to search the sub-bit region first, fight to restore
> capability there, and accept a higher physical rate only after producing a sealed
> scientific justification.

---

## 1. Where Gravity lives

| Concern | File |
|---|---|
| Policy, exact-rational rates, priors, coverage, conservation, EXTREME gate, default-off | `tools/condense/succ_gravity_policy.py` |
| Escape / structural-incompatibility / physical-impossibility receipts | `tools/condense/succ_gravity_receipts.py` |
| Inverted search, acquisition G(x), parent-state FSM, source-bound programs | `tools/condense/succ_gravity.py` |
| Adversarial tests (20 + synthetic + folded sim invariants) | `tools/condense/tests/test_succ_gravity.py` |
| Sealed policy manifest | `reports/condense/subbit_frontier/GRAVITY_POLICY.json` |
| Sealed parent states + source-bound programs | `reports/condense/subbit_frontier/GRAVITY_STATE.json` |
| Sealed validation record | `reports/condense/subbit_frontier/GRAVITY_VALIDATION.json` |

Additive hooks into the existing controller (byte-identical when Gravity is unused):
`succ_queue.make_row` (`gravity` field), `succ_frontier.build_row` (`gravity=` arg),
`succ_engine.next_experiment` (`gravity_bonus=` arg), `succ_telegram.EVENT_CATALOG`
(Gravity event kinds), `succ_cli` (`gravity-*` commands).

Scientific ground (read-only, do not drift): the sealed Sub-Bit Readiness Diagnostic
`reports/condense/subbit_frontier/SUBBIT_READINESS_PACKET.json`
(seal `f6c6b2d8...`) and its policy proof `subbit_inverted_search_sim.py` (11/11).

---

## 2. The invariant (enforced by code, not prose)

```
DEFAULT SEARCH DIRECTION : upward from a parent-specific sub-bit stress point
DEFAULT TARGET REGION    : physical whole-artifact BPW < 1.0
FALLBACK DIRECTION       : upward only
QUALITY CONTRACT         : unchanged
PHYSICAL ACCOUNTING      : complete whole artifact
HEAVY CONTROLLER COUNT   : exactly one
ESCAPE FROM SUB-BIT      : requires a sealed Escape Receipt
```

No parent may be finalized with an `EXTREME` result above one physical BPW unless
mandatory sub-bit coverage is satisfied **and** a valid sealed Escape Receipt is bound
(`succ_gravity_policy.can_finalize_extreme`). Doctor bytes are always counted inside the
whole-artifact BPW; a `0.50` base + `0.40` correction + `0.15` overhead artifact is
`1.05` BPW and is **not** sub-bit.

Mandatory sub-bit coverage (one of):

- **A** — two materially different representation *classes* reached F1/F2 (a scalar
  quantizer at another group size is the same class and does not count as a second).
- **B** — a sealed structural-incompatibility receipt.
- **C** — a sealed physical-impossibility receipt (all mandatory byte categories).
- **D** — a calibrated proxy with a known false-negative bound, plus a targeted
  parent-specific confirmation. Scale interpolation alone is forbidden.

A missing experiment, a scheduler deferral, a single failed scalar codec, and one weak
Doctor treatment are each **not** a justification (`escape_receipt_valid` rejects them).

---

## 3. Parent-specific gravitational starting points (priors, not constants)

From the readiness packet, recomputed against the live envelope by
`compute_stress_start`:

| parent | stress start | region | before / after successor |
|---|---|---|---|
| 72B | `4/5` (0.80) | collapse probe (not resident-forced) | oracle before; artifact only after lease free |
| 120B | `4/5` (0.80) | collapse probe; BF16 staging executable | staging oracle before; artifact after MoE loader + lease |
| 685B | `11/20` (0.55) | resident-forced (ceiling 0.80) | remote-stream oracle, after successor only |
| 1T | `1/3` (0.33) | resident-forced (ceiling 0.55) | remote-stream oracle, after successor only |
| 1.6T | `1/4` (0.25) | resident-forced, stream-only (ceiling 0.34) | remote-stream oracle, after successor only |

The stress start is the first serious low-rate diagnostic region, **not** automatically
the first full-model conversion. For expensive parents Gravity runs F0/F1/F2 at or below
the stress rate before choosing the cheapest credible full-model candidate.

---

## 4. Search direction

```
start aggressively low -> physical feasibility (F0) -> representation survival (F1/F2)
  -> diagnose damage -> reachable Doctor restoration -> ascend only when necessary
  -> descend again after a pass -> refine the boundary
```

- **Pass** -> descend to a lower rate that could still change the Event Horizon.
- **Signal degradation** -> stay; attempt a diagnosis-compatible reachable Doctor program.
- **Mixed failure** -> protect/reset collapsed organs while treating survivors; do not
  apply one global correction blindly.
- **Computation collapse** -> try a materially different representation family first; only
  when families are exhausted and Doctor cannot reach the missing causal subspace, ascend.
- **Undetermined** -> run the cheapest discriminating probe; never ascend because evidence
  is inconvenient.
- **Physically impossible** -> issue a physical-impossibility receipt and ascend.

Representation-family escalation precedes BPW escalation:
`same rate, different representation` -> `... different byte allocation` -> `... different
Doctor treatment` -> `slightly higher rate`.

---

## 5. Current live state (verify before acting)

- Active legacy parent: **72B**, cell `qwen2-5-72b__4bpw__doctor-static`, near completion.
  It holds the machine-wide heavy lease (`reports/cron/studio_heavy.lock`). Immutable.
- Sole active heavy controller: `doctor_v5_disk25_successor.py`. Gravity must never become
  a second one.
- Gravity is default-off. `GRAVITY_STATE.json` carries the materialized **source-bound 72B
  sub-bit program** at `4/5` (0.80) and its **higher-rate fallback** at `3/2` (1.5). Neither
  is launchable: `program_launchable` refuses while the lock is held and Gravity is off.

Confirm the lock owner without touching it:

```bash
ps -Ao pid,command | grep -E 'doctor_v5_disk25_successor|quantize-model-block-parallel' | grep -v grep
```

---

## 6. Operator commands

Status, policy, and inspection (all read-only, no lease):

```bash
# Gravity policy + invariant + per-parent stress starts
python3.12 tools/condense/succ_cli.py gravity-status
python3.12 tools/condense/succ_gravity.py policy

# inspect a parent's stress start and fresh Gravity state
python3.12 tools/condense/succ_cli.py gravity-inspect --parent 72B

# explain the next Gravity-prioritized probe (sub-bit-first)
python3.12 tools/condense/succ_cli.py gravity-explain-next --parent 72B

# validate the whole law (module selftests + invariant witnesses), sealed
python3.12 tools/condense/succ_cli.py gravity-validate

# materialize the source-bound program + higher-rate fallback (a plan; cannot launch)
python3.12 tools/condense/succ_cli.py gravity-materialize --parent 72B
```

Existing controller status / drain / resume (Gravity rides the same spine):

```bash
python3.12 tools/condense/succ_cli.py status         # controller + queue status
python3.12 tools/condense/succ_cli.py explain-next   # next experiment (adds G(x) when armed)
python3.12 tools/condense/succ_cli.py drain          # graceful stop request
python3.12 tools/condense/succ_cli.py resume         # split-brain-checked exact resume
```

Tests:

```bash
python3.12 -m pytest tools/condense/tests/test_succ_gravity.py -q
```

Eventual launch (blocked today; every gate must hold first):

```bash
# 1) the legacy campaign reaches its signed release boundary and quiesces
# 2) the successor is authorized and holds the heavy + singleton lease
# 3) resource admission passes (swap<=512MB, pressure normal, AC, thermal green, disk fits)
# 4) the parent's source-bound program is executable (packer wired at the target rate)
# 5) arm Gravity:
HAWKING_GRAVITY_ENABLED=1 python3.12 tools/condense/succ_cli.py start   # only then does it run
```

Until all five hold, `gravity_enabled` returns False and no Gravity program can launch.

---

## 7. Build-before-arm sequence (gates)

Gravity's policy is proven and installed; the mechanisms it needs to produce a real
sub-bit artifact are not all built. In order, each with a gate:

1. **Sub-1-bit deployable packer** (promote VTQ vector-trellis to a CLI + GPU decode).
   Gate: a real model roundtrips at a nominal sub-1-bit rate. Until then sub-bit rates
   correctly return DEFERRED, never PASS (see `test_real_parent_defers_subbit_no_false_floor`).
2. **Exact-rational rate ladder** — done (`succ_gravity_policy.RATE_LADDER`, `fractions.Fraction`).
3. **Ascend-on-collapse scheduler** — done (`succ_gravity.InvertedSearch`).
4. **Heavy-lease discipline** — modeled (`HeavyLock`, `program_launchable`); the real lane
   must acquire `studio_heavy.lock` `LOCK_EX|LOCK_NB` and abort on contention.
5. **Giant-parent MoE loader + remote shard streamer** — required for 120B / 685B+.
6. **Packed-output-size + quality oracle** — quality remains a real-run verdict.

Do not arm a sub-bit heavy lane while the 72B holds the heavy lease. Do not full-install
1.6T (stream-transcode only). Do not treat oracle output as a sub-bit success.

---

## 8. Telegram

Gravity emits its own kinds through the existing notifier (terse, deduplicated,
reboot-safe): `gravity_policy_enabled`, `gravity_start_rate`, `gravity_tournament_started`,
`gravity_feasibility_completed`, `gravity_diagnosis`, `gravity_rescue_started`,
`gravity_rescue_result`, `gravity_byte_allocation_changed`, `gravity_representation_changed`,
`gravity_descend`, `gravity_ascend`, `gravity_escape_requested`, `gravity_escape_decision`,
`gravity_first_pass`, `gravity_lower_boundary`, `gravity_event_horizon`,
`gravity_bpw_composition`, `gravity_queue_eta`, and a `gravity_daily` summary. Provisional
results carry the standard provisional footer until the signed physical release gate.

```bash
python3.12 tools/condense/succ_telegram.py --compose gravity_event_horizon \
  --ctx parent=72B rate=4/5 whole_bpw=0.83     # dry-run preview, no send
```
