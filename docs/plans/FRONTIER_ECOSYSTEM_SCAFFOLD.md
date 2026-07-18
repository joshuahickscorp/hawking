# Condenser Ecosystem Frontier: scaffold status

Additive, default-off scaffold for the successor generation described in
`docs/plans/CONDENSER_ECOSYSTEM_FRONTIER.md`. Built in the isolated worktree
`codex/condenser-ecosystem-frontier` while the 72B/campaign generation is live and
immutable. Nothing here activates until the campaign supersession boundary is signed.

House style: no em or en dashes, no middots. Proof discipline preserved end to end: the
campaign runs with `quality_claims_permitted:false`, so every quality reading below is
PROVISIONAL engineering evidence, never a sealed win.

## 1. What was built

Nine modules under `tools/condense/eco_*.py` plus eight test files under
`tools/condense/tests/test_eco_*.py`. Purely additive: zero existing tracked files were
modified. Interpreter `python3.12`, stdlib only, same idioms as the campaign tools
(schema string constants, frozen `@dataclass Config` + `default_config()`,
`*Error(RuntimeError)`, sha256 self-seal byte-identical to the campaign reporter,
argparse `main(argv)` with `selftest`, plan-unless-`--go`).

| module | role |
|---|---|
| `eco_common.py` | canonical hashing/sealing (verified byte-identical to the campaign reporter), atomic writes, safe reads, schema registry, the pinned `CAMPAIGN_PLAN_SHA256` |
| `eco_passport.py` | the one identity/receipt graph: binds the eight dimensions (artifact, Doctor treatment, physical bytes, capability contract, Context Horizon, session state, device profile, client compatibility) into one content-addressed Passport, plus the prefix/branch DAG edges and claim-separation enforcement |
| `eco_import.py` | immutable, read-only import of the live campaign into a frozen prior ledger; validates each cell's on-disk `result_sha256` / `disposition_sha256`; skips every non-terminal (running/pending/blocked) cell |
| `eco_planner.py` | the adaptive EXTREME planner: F0..F4 feasibility ladder, diagnose-first, Doctor programs from diagnosis, adaptive descent skipping resolved rates, Event-Horizon bracket, scaling prior, one recommended candidate per parent |
| `eco_pipeline.py` | the data-driven `Press -> ... -> Summon` state machine: validators, exact resume, rollback, offline hydration |
| `eco_activation.py` | fail-closed, default-off atomic activation + rollback, gated on the supersession signature |
| `eco_admission.py` | 120B+ admission plans: per-parent adapter, exact source manifest, streamed lifecycle, quality-evidence requirement, device fit |
| `eco_status.py` | Telegram status/ETA reusing the campaign notifier's hardened send primitives, its own state store, injectable sender |
| `eco_cli.py` | one CLI surface for everything, plus `materialize` to emit the whole plan-only artifact bundle |

Tests: 52 passing (`python3.12 -m pytest tools/condense/tests/test_eco_*.py`).

## 2. The one identity/receipt graph (Passport)

A Passport binds the eight identity dimensions, each as a content-hashed facet with a
declared claim layer, and self-seals into `passport_sha256`. Claim separation is
structural: the standalone `physical_bytes` facet counts every base, correction,
codebook, exception, protected-island, and routing byte, and REFUSES runtime roles
(`kv_cache`, `runtime_index`, `workspace_cache`, ...) so a failed standalone claim cannot
be laundered with context or agent bytes. Passports link via content-addressed edges
(parent identity + delta + model/profile identity + position policy + KV/state codec =
child identity), which is the prefix/branch DAG for exact reuse, forks, and rollback.

## 3. The adaptive planner (replaces the fixed bit ladder)

The old 8x10x4 = 320-cell matrix is imported as evidence and priors, not re-run. Per
parent the planner binds exact parameter count, parent baseline, byte ceiling
(device weight budget / params), and the frozen capability contract, then builds an
uncertainty-aware rate frontier over `4, 3, 2, 1, 0.8, 0.55, 0.5, 0.33, 0.25, 0.1` bpw.

- F0 byte feasibility (deterministic) -> F1 tensor/block realization proxy -> F2 shard
  proxy -> F3 full-model quality (provisional) -> F4 replicated/sealed. Proxies
  prioritize; only F4 proves. Physical bytes reach F4 via the campaign execution receipt;
  quality F4 is unproven everywhere under Pass B.
- Diagnose-first: each rate is classified `no_material_damage` / `signal_degradation` /
  `computation_collapse` / `mixed_failure` / `undetermined`.
- Doctor programs come from the diagnosis, not fixed branches. The four controls
  (zero-treatment, equal-byte codec, smaller-higher-bit, public same-byte) are always
  retained; only high-value causal treatments are promoted (signal_degradation ->
  doctor_static/conditional; mixed_failure -> capability-targeted).
- A codec-control failure that is treatable and has no terminal Doctor branch is
  UNRESOLVED (pending Doctor recovery), and that Doctor program becomes the boundary
  probe. A collapse, a Doctor branch that stayed out of contract, or an adaptive-defer
  disposition is a RESOLVED fail.
- The recommendation stops only when the lowest passing rate and the next lower failing or
  inconclusive rate are both evidenced; otherwise it lists the exact boundary probes.

### Results on the live campaign (read-only, plan-only)

187 terminal cells imported (140 complete + 47 dispositions), all seals valid, 0
unreadable, the 1 running + 89 blocked-dependency + 35 pending + 8 blocked-execution cells
correctly skipped. Six parents have evidence (0.5B..32B); 72B and 120B are awaiting.

- Under the campaign's own promotion gate (ppl relative delta <= 0.08, capability absolute
  delta >= -0.05), NO rate passes yet. At 4 bpw the best Doctor branch recovers to the
  very edge of the contract (0.5B doctor_full 0.0798, 3B doctor_full 0.256) but does not
  clear it; at 2 bpw every branch collapses (5x to 28x ppl). So the small-Qwen standalone
  Event Horizon sits ABOVE 4 bpw under this contract. This is honest and expected: these
  are aggressive control/recovery cells on the hardest (smallest) models.
- The collapse-boundary floor DESCENDS with scale: 3.0 bpw at 0.5-3B, 2.0 bpw at 7-14B,
  slope about -0.81 bpw per decade. This matches the standing "bit-floor descends with
  scale" hypothesis, now on real sealed data.
- EXTREME per parent is therefore UNPROVEN pending Doctor recovery evidence that clears the
  contract, with the concrete boundary probes emitted per rate.

## 4. 72B transition

72B has zero terminal cells: `qwen2-5-72b__4bpw__codec-control` is RUNNING (owned by the
live supervisor pid), the other 39 cells are pending or dependency-blocked. The scaffold
therefore does exactly what the directive requires and nothing more:

- it does NOT restart, reseal, relabel, or touch any 72B cell;
- it binds the running cell as immutable evidence and reports 72B as `awaiting_evidence`;
- it brackets 72B's Event Horizon from the smaller-scale sealed anchors via the scaling
  prior (predicted floor about 1.44 bpw), labeled SCHEDULING ONLY, never proof;
- once `qwen2-5-72b__4bpw__codec-control` seals, 72B becomes the planner's first
  calibration case: reconstruct its untreated and Doctor frontiers, locate the measured
  collapse boundary, and run only the diagnosis-selected boundary probes.

Each lower 72B rate's reason is emitted as "unproven: no terminal cell yet; codec-control
still running or dependency-blocked."

## 5. 120B+ admission plans

`eco_admission.py` produces one admission plan per large parent (the campaign's
gpt-oss-120b plus the FRONTIER_MODELS 235B..1.6T). Each binds the exact source manifest,
device fit via `size_frontier.analyze` (RESIDENT / MOE-PAGED / DENSE-OOC / TOO-BIG), the
streamed download -> bake -> seal -> capability-eval -> source-release lifecycle, the
adapter requirement, and the quality-evidence requirement (standalone capability vector +
native load/parity + F4 replicated seal). Findings:

- gpt-oss-120b: adapter `doctor-v5-strand-ladder-gpt-oss-moe` exists -> admissible.
- Five families still need a per-family Doctor-v5 adapter before admission: llama-dense,
  deepseek-moe, glm-moe, kimi-moe, qwen3-moe.
- 72B effects feed 120B+ only as scheduling priors (candidate rate seeded from the scaling
  prior, flagged `candidate_rate_is_prior_only`), never as proof. Each parent needs its own
  adapter, source manifest, streamed lifecycle, sealed quality, and admission receipt.

## 6. Non-interference proof

- Additive-only footprint: `git status` shows only new `eco_*` files; zero modifications
  to any existing tracked file.
- The import reads the campaign by content hash, skips every non-terminal cell (verified:
  the running 72B cell is never opened), and writes only to the separate
  `reports/condense/frontier_eco/` namespace.
- The activation gate, run against the live campaign, REFUSES: 133 cells non-terminal, no
  reporter checkpoints, 1 running cell, campaign supervisor pid alive, no operator
  signature. Activation is impossible by construction while the campaign runs.
- The scaffold launches no heavy compute; every command is planning/inspection only.

## 7. Activation and rollback

Default-off: with no manifest on disk the ecosystem layer is inactive. `activate` refuses
unless ALL five supersession conditions hold (terminal, reporter-sealed,
checkpoint-accepted, quiescent, signed) AND `--go` is set. The supersession signature is an
out-of-band operator authorization that the tool only reads and verifies; the agent does
not sign the live campaign. `rollback` restores the previous manifest or returns to
explicit default-off. Do not activate before the signed campaign release boundary.

## 8. CLI

```sh
python3.12 tools/condense/eco_cli.py selftest
python3.12 tools/condense/eco_cli.py import      --campaign-root <doctor_v5_ultra>
python3.12 tools/condense/eco_cli.py plan        --campaign-root <doctor_v5_ultra>
python3.12 tools/condense/eco_cli.py admission   --campaign-root <doctor_v5_ultra>
python3.12 tools/condense/eco_cli.py pipeline
python3.12 tools/condense/eco_cli.py activation gate   --campaign-root <doctor_v5_ultra>
python3.12 tools/condense/eco_cli.py status      --campaign-root <doctor_v5_ultra>   # dry
python3.12 tools/condense/eco_cli.py materialize --campaign-root <doctor_v5_ultra> --out-dir <dir>
```

`materialize` emits the full plan-only bundle (prior ledger, adaptive plan, pipeline spec,
admission plan, one Passport per parent-with-evidence, MANIFEST) under
`reports/condense/frontier_eco/materialized/`.

## 9. Evidence grades

- Physical bytes: SEALED via the campaign execution receipts (imported and re-validated).
- Quality: PROVISIONAL engineering evidence only (`quality_claims_permitted:false`). No
  public WIN is asserted anywhere; no rate is claimed to pass the standalone contract.
- Scaling prior and 72B/120B brackets: SCHEDULING priors, explicitly not proof.

## 10. Deferred (post-activation phases E2..E12)

The Python scaffold is the planning, summoning, identity, and presentation layer the
directive scopes for the non-interference window. Not built this wave (each is a separate,
post-signature phase): the Rust Continuum/Lens/Bridge/Capsule crates, the real context
evaluation battery (NIAH/multi-hop/long-code) on a sealed artifact, the context compiler
wiring, the HIDE reference UX, and the equal-device frontier trial. The pipeline stage
graph and Passport schema are the seams those phases attach to.
