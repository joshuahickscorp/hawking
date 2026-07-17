# M1 Ultra Run Report: the hawking maximization run

The single accumulating artifact for the M1 Ultra maximization loop (see `docs/plans/M1ULTRA_GOAL_PROMPT.md`
and `docs/plans/M1ULTRA_POTENTIAL_AUDIT.md`). Every wave appends here so a dead session loses at most one
wave. Every number is graded by evidence strength so nothing is over-claimed. House style: no em or en
dashes. Proof discipline: effective bpw only, a tie is a null, no WIN cell without an R3+ receipt.

## 0. How to read this (evidence grades)

- MEASURED-R3+: a real receipt at repro level R3 or higher (one-command same-machine-class repro), the
  ONLY grade that backs a public WIN.
- MEASURED-LAB: a real number on this box, author-rerunnable, below R3 (R0-R2). Supports a contingent or
  negative claim, never a public win.
- GATED: the build or artifact the claim needs is not yet complete; the number is honestly withheld.
- UNPROVEN: never run. Zero by rule.
- NULL: the lever tied its baseline at matched compute or a certified gate. Recorded as a citable bound.
- WALLED: a proven structural ceiling with a mechanistic reason (the honest terminal state of a category).

## 1. The scoreboard (start-of-run state, from M1ULTRA_POTENTIAL_AUDIT.md)

Proven = adversarially-verified current-today. Bounded = the audit's honest ceiling. Maximal = the ceiling
once unbounded wall-clock converts every time-wall. Update the Proven column as each wave lands a receipt.

| Category | Proven | Bounded | Maximal | State | Last receipt |
|---|--:|--:|--:|---|---|
| Decode throughput (single-stream, iso-quant gap) | 6.5 | 7.5 | 9.0 | MEASURED-LAB (31.03 tok/s, a4_clean_walltime.json) | - |
| Kernel maturity (45 gemm_q4_k perms, autotune) | 7.0 | 7.5 | 8.5 | MEASURED-LAB | - |
| Prefill + host dispatch | 5.5 | 6.5 | 7.5 | MEASURED-LAB | - |
| Effective-bpw floor + bit-floor-vs-scale LAW | 3.5 | 6.0 | 8.5 | GATED (0 floor points, need >= 2) | - |
| Sub-1-bit SUBBIT lane | 2.5 | 4.0 | 7.0 | GATED (side-info floor 0.1093 alive) | - |
| STRAND codec quality vs QTIP/QuIP#/AQLM | 3.0 | 5.5 | 8.0 | GATED (synthetic probe only) | - |
| Quality recovery / the Doctor (gradient moonshot) | 5.0 | 8.0 | 9.0 | GATED (train-free de-risked; gradient arm unrun) | - |
| Doctrine: CPU-bf16 headline vs bf16-on-MPS | 3.0 | 7.0 | 7.5 | UNPROVEN (microbench A) | - |
| Native .tq serve (build + throughput) | 6.0 | 8.5 | 9.5 | GATED (built, tps unmeasured) | - |
| RAM-cliff throughput demo (the money demo) | 0 | 9.0 | 9.5 | UNPROVEN | - |
| Flat-cost long-context serve (SSM wired) | 6.0 | 8.0 | 9.0 | MEASURED-LAB (B=1, 0.4B) | - |
| Honest recall (NIAH / Omega(N) theorem) | 1.0 | 6.0 | 8.0 | UNPROVEN (synthetic ladder only) | - |
| KV-frontier levers (int4-KV, YaRN, STKV) | 2.0 | 6.0 | 8.0 | GATED (int4-KV disabled, ppl gate unrun) | - |
| EAGLE-3 trained draft head | 1.5 | 3.5 | 7.0 | GATED (offline tau 0.877 << 2.5) | - |
| Verify path + lossless-verify exact-match gate | 0 | 4.0 | 8.0 | NULL (gate ran, FAILED 6/20) | - |
| SpecGovernor + drafter economics | 3.0 | 5.0 | 6.5 | MEASURED-LAB (n-gram 1.43 < 1.6 floor) | - |
| Honesty / receipt discipline | 8.5 | 9.5 | 9.5 | MEASURED-LAB (self-test passes) | receipts/official/qwen-05b-tq3.json |
| CI gap (honesty machinery that never runs) | 3.0 | 6.5 | 9.0 | UNPROVEN (tests/ never run in CI) | - |
| The GO conveyor (studio_run.py P0-P9) | 5.0 | 8.0 | 9.0 | CRASH FIXED off-box (P6/P8 unpack); go-plan green; e2e run needs box | - |
| Architecture coverage (the model families) | 2.5 | 6.5 | 8.5 | GATED (Qwen-dense verified; Mamba2 condense-track coverage added off-box) | - |
| Size frontier / expert-paging serve | 1.0 | 4.5 | 7.5 | RE-DERIVED off-box: 235B/405B/671B fit RESIDENT on 128GB, pager OFF critical path | - |
| HIDE M1: pass state not text (handoff) | 6.5 | 8.5 | 9.0 | GATED (kv_handoff seam, FakeCopier) | - |
| HIDE M2: free local fleets | 5.0 | 8.0 | 8.5 | GATED (fabric real, economics a predictor) | - |
| HIDE M3: own the .tq format | 5.5 | 8.5 | 9.0 | GATED (no e2e coherent-token receipt) | - |
| HIDE M4: grammar-guaranteed tool calls | 3.0 | 7.0 | 7.5 | MEASURED-LAB scaffold, adversarially verified (parser + parse/lint/dedup/dispatch loop + schema-aware jump-forward grammar + prompt-lookup, 39 owned tests; 6 review findings, 4 fixed w/ regression tests, 2 pre-existing edit.rs bugs reported); still GATED on decode-loop wiring + first-try-valid receipt | agentic_tool_system_audit_2026_07_11.md |
| HIDE ship-readiness (Tauri, executor, tests) | 7.0 | 8.5 | 9.0 | MEASURED-LAB (signed 18 MB DMG on disk) | - |
| The thesis gate (can the local model code) | 3.5 | 7.0 | 7.5 | MEASURED-LAB (R0/R1: Python 14/15 = 93.3% Wilson95 70.2-98.8%; Rust 10/12 = 83.3% Wilson95 55.2-95.3%; exec-grounded, Qwen2.5-7B-Q4_K served) | reports/eval/thesis_gate_qwen7b_q4km*.json |

North-star overall: proven 3.25 / bounded ceiling 6.33 / maximal ceiling ~8.4.

BOX CORRECTION (2026-07-16, Wave 1 on-box): the delivered box is NOT the M1 Ultra 128 GB / 8 TB this
report and the audit assume. It is a Mac Studio M3 Ultra: 28-core CPU (20 perf), 60-core GPU, 96 GB unified
memory, Metal 4, ~800 GB/s, and only ~162 GB free on a 926 GB internal SSD (no 8 TB volume). Bandwidth and
GPU meet or beat the M1 Ultra plan, but memory is 96 not 128 GB and usable disk is ~150 GB not 8 TB. Scope
consequence: the frontier-resident moonshot (gate 2) and the RAM-cliff demo are DISK-WALLED for parent
staging (a 235B/405B/671B bf16 parent does not fit in 150 GB free; only stream-bake is possible), and 671B
@84 GB will not fit RESIDENT on 96 GB (retire it here). Moonshot gate 1 (doctor recovery 7B-32B) is
download-gated not dead: 7B + 14B bf16 parents are now staged (see Wave 1). CORRECTION to the Wave 1 note:
the device constants were ALREADY re-derived for this box during the doctor-v5 work. studio_manifest.py
DEFAULT_HARDWARE = M3_ULTRA_96GB (ram 96, weight_budget 78 GB, ram_gbps 819, ssd_tb 1.0, disk_reserve 150)
and size_frontier DEFAULT_DEVICE = studio-m3ultra-96; the leftover `m1ultra` (112/8 TB) is only a
non-default historical comparison row. So weight_budget 78 GB already correctly retires 671B-resident
(84 GB > 78). Two residual gaps remain: (1) the AUDIT + GOAL-PROMPT PROSE still describe the box as
M1 Ultra 128 GB / 8 TB (doc drift, harmless to the math but misleading); (2) the manifest models a 1 TB SSD
with an 850 GB storage budget, but the box is 82% full with only ~162 GB actually FREE, so the frontier-fit
`fits_storage` math overstates what can be staged today (the real staging wall is current free space, not
SSD capacity). Net: the moonshot-2 disk wall stands for a different reason than "wrong constants" - it is
current disk occupancy, not a missing device row.

## 2. Open gates (answer these to move the scoreboard)

| gate | status | owned by |
|---|---|---|
| studio_run.py runs end-to-end (crash fixed)? | CODE-FIXED off-box (P6/P8 3-tuple unpack corrected); e2e heavy run still needs the box | SPINE-0 / Wave 0 |
| Thesis gate: can the .tq-served local model code? | OPEN (hawking-eval never run) | SPINE-1 |
| Gradient recovery clears <= +2% at sub-4-bit on 7B+? | OPEN (every +dr swap-died on 18 GB) | SPINE-2 |
| Bit-floor-vs-scale law fits (>= 2 verdict floor points)? | OPEN (0 on disk) | SPINE-2 |
| MoE per-expert sensitivity non-uniform on a real bake? | OPEN (synthetic probe non-uniform) | SPINE-3 |
| Native .tq serve tok/s > 10 on a full resident model? | OPEN (only 20 MB toy baked) | SPINE-4 |
| RAM-cliff: serve where a control box's Q4_K OOMs? | OPEN (unrun) | SPINE-4 |
| CI runs the tests/ integration targets? | OPEN (--lib --bins skips them) | SPINE-5 |
| Batched-verify bit-identical to greedy (spec admissible)? | OPEN (failed 6/20) | SPINE-5 |
| MPS-bf16 doctor beats CPU-bf16 throughput? | OPEN (microbench A) | Wave 0 |
| DeepSeek-V3 / GLM router implemented (correct logits)? | OPEN (plain softmax topk, wrong for V3/GLM) | SPINE-3 prereq |

## 3. Benchmark ledger (fill as measured on M1 Ultra)

| bench | M3 Pro proven | M1 Ultra measured | note |
|---|---|---|---|
| Qwen2.5-3B-Q4_K_M decode tok/s (single-stream) | 31.03 (a4_clean_walltime.json) | - | bandwidth-bound; projection ~150-165 until run |
| Aggregate tok/s at max continuous batch | ~48 (B=8, conservative) | - | the re-headlined metric on 128 GB |
| 7B doctor recovery, best recovered eff-bpw @ <=+2% | none (swap-died) | - | the moonshot gate 1 headline; 7B+14B bf16 now staged |
| Thesis gate pass@1 Python (15-task exec-grounded smoke) | none | 14/15 = 93.3% (Qwen2.5-7B-Q4_K, debug serve) | Wilson95 70.2-98.8%; 1 fail = quant token glitch `count_ vowels` |
| Thesis gate pass@1 Rust (12-task rustc exec-grounded) | none | 10/12 = 83.3% (same model) | Wilson95 55.2-95.3%; 2 real model fails (is_ascii_alphanumeric on &str; a syntax error); Rust harder than Python as expected |
| Native .tq decode tok/s (resident 70B) | none | - | microbench B / SPINE-4; GGUF serve path proven coherent this wave |
| RAM-cliff tok/s vs control box Q4_K | none | - | the money demo |
| Energy J/tok at the cliff | none | - | the energy moat |

## 4. Wave log

Append one paragraph per wave: what ran, verdict, category movement, next lever. No wave ends without a
committed (approved) artifact.

- Wave F5 (2026-07-17, High-Parameter Frontier Program: 685B/1T/1.6T preparation layer): built the giant-parent
  preparation layer (heavy conversion gated: sources 595-1371 GB vs 175 GB free disk; legacy 72B still running,
  untouched). SOURCE AUTHORITY bound by READ-ONLY HF metadata fetch (not from memory), with exact revisions +
  geometry, correcting the directive's hypotheses where the real config differed: DeepSeek-V3.2 685B (rev
  a7e62ac, 256 experts/8-selected/1-shared, MLA, MTP=1, ~bf16 1371GB); Kimi-K2.6 1T (rev 7eb5002,
  KimiK25ForConditionalGeneration = MULTIMODAL so text-core/full claim split is real, 384/8/1, MTP=0, INT4
  595GB); DeepSeek-V4-Pro 1.6T (rev b5968e9, 384 experts/6-selected [NOT 8], native FP4/FP8 ~0.54 B/param
  [directive's FP4 confirmed], MTP=1, 865GB). MODULES (5 new succ_*.py + 2 test files, 111 tests green):
  succ_frontier (exact geometry + physical fit with the OFFICIAL-total denominator: V3.2@0.80=68.5GB and
  Kimi@0.55=68.75GB fit the 72GB safe envelope -> RESIDENT_EXTREME; V4-Pro@0.38=76GB -> HYBRID_EXPERT_EXTREME,
  resident ceiling 0.36; 3 durable rows admitted to the controller queue with honest waiting_adapter blockers);
  succ_twin (MANDATORY synthetic geometry twins + systems battery - ALL THREE twins GREEN 8/8:
  deterministic_conversion, round_trip_integrity, bounded_rss, source_range_resume, expert_paging HOT/WARM/COLD,
  crash_recovery, flock duplicate_launch_prevention, output_layout - the acquisition gate is PASSED);
  succ_press (remote bounded-stream Press: 4 deterministic passes, every byte in the shard manifest, resume-not-
  restart, deterministic global reduction; press_plan proves PEAK DISK ~9-15 GB per giant parent vs 595-1371 GB
  source = a giant parent IS convertible on this disk-walled box via bounded streaming, the key feasibility
  result); succ_adapter_frontier (fail-closed adapter contracts for deepseek_v32/kimi_k25/deepseek_v4: run
  refuses exit 78, capabilities all-pending, claim components CORE/CORE+MTP and K2.6_TEXT_CORE/FULL_MULTIMODAL);
  succ_atlas (NON-COMPETING resource atlas: read-only 28-core/96GiB inventory + harvest-derived per-branch
  sec/billion; full CPU/GPU/storage benchmark deferred post-release since it would compete with the live worker).
  CLI: frontier/frontier-fit/frontier-admit/frontier-twin/frontier-press-plan wired. Source authority + frontier
  manifest sealed under the successor namespace. NON-INTERFERENCE held (no campaign-namespace write, nothing
  heavy launched, twins/press are offline fixtures). Honest gaps: the twin/press codecs are reversible
  stand-ins (systems path proven, not compression quality); the safetensors header prefetch for per-tensor byte
  ranges and the live disk-floor gate in run_pass are the remaining production wiring; real acquisition + heavy
  conversion run only post-release + admission. Committed to PR #23; no merge, no activation.
- Wave F4 (2026-07-17, empirical evidence sealed + retirement/ETA/Telegram wired): continuation. RECOVERED the
  three interrupted prior-session workflows (in session dc930fd4, NOT this one): doctor-boundary-optimization-map
  COMPLETED (5/5, read-only map + adversarial verify that had cross-checked a 140-row harvest 139/139 vs raw
  receipts); eta-mountain-parallelism-map PARTIAL (2/3; found the mountain_ladder.json already built);
  doctor-evidence-closure-build PARTIAL (1/3; its one completed builder produced the untracked
  tools/condense/doctor_v5_source_gc.py on main, arm/run never executed; the other 2 builders left nothing).
  Classified the uncommitted main-branch doctor_v5_* changes as prior-session LEGACY campaign tooling
  (operator-owned, campaign namespace) and left them untouched. Legacy 72B still running (doctor-static), 142
  complete, not released; non-interference held. BUILT (additive, in the frontier worktree): succ_harvest.py
  (seals the empirical evidence: 189 terminal rows with the full field set from result.json +
  execution_receipt.json - exact physical bpw, parameter/tensor geometry, quality, wall time from the receipt
  resource_observations timestamps, disk/memory-pressure/thermal, lifecycle, treatment-vs-equal-rate-control,
  dominance, and a failure CLASSIFICATION that never relabels a scheduling deferral as collapse: 113
  measured_computation_collapse vs 47 scheduling_deferral vs 2 missing_evaluation vs 27 measured_quality_failure;
  142/142 complete rows seal-valid; doctor improved over control in 54/103; 72B codec_control 5.078bpw took
  ~13.7h wall); succ_retire.py (evidence-closed retirement from the sealed harvest: replicated collapse
  boundaries 0.5B/3B/7B/14B=3.0bpw, 1.5B=2.0bpw; 229 FUTURE successor experiments retired with sealed receipts
  that preserve evidence + reopening criteria, ground cannot_change_frontier_conservative bound by the sealed
  harvest - honest for single-seed Pass-B, no false replication claim; additive, never mutates a legacy cell).
  WIRED into the controller: succ_engine.next_experiment now consults the retirement ledger and SKIPS
  evidence-closed (model,rate) probes (proven by test); ETA fit from 142 real wall-time observations into
  per-(branch,full_cell) segments (never one global constant). Telegram VERIFIED with a REAL send (message_id
  5827, delivery receipt stored). CLI: harvest/retire-plan/eta wired. 98 tests green (+6). Committed to PR #23;
  no merge, no activation.
- Wave F3 (2026-07-17, successor continuation: 72B seal + arming): re-invoked the same master goal. Re-audit
  found the LEGACY 72B codec_control cell SEALED (now complete; 142 complete total; the campaign moved on to
  qwen2-5-72b__4bpw__doctor-static, now running and untouched); report checkpoints still None so the campaign
  is NOT released (State B holds, transition still correctly blocked). Used the new 72B evidence: the successor
  now imports 189 terminal cells and 72B is a parent WITH evidence (its 4bpw codec_control physical bytes
  sealed at all_in_model_payload_bpw=5.078, quality DEFERRED as the 72B resident eval is disk/RAM-gated).
  BUILT (additive): succ_calibrate.py (precompiles the 72B post-release calibration program per section 9:
  untreated frontier from sealed evidence, the deferred full-model quality eval as the first experiment, the
  boundary probes, lower-rate reasons, release-bound, launches nothing) and succ_watch.py (the detached
  release watcher: singleton-lease tick that heartbeats + re-checks the gate in WAIT_OLD_RELEASE and fires the
  one-use transition only when the gate passes; plus the arming artifacts: an UNSIGNED intent template bound to
  the exact live identities (legacy_plan_sha256, successor_commit 467e8885, expected_terminal_count=320), and a
  launchd plist written to a file, NOT installed). Wired calibrate/watch/arm-template/watch-plist into the CLI.
  Also closed the last cheap adversarial-review item (transition one-use TOCTOU via an O_EXCL atomic claim).
  92 tests green (+6). VERIFIED end to end on live data: calibrate wrote the 72B program (11 experiments),
  arm-template wrote the unsigned template, watch-plist wrote the plist, watch --once ticks and correctly
  blocks the gate (no signed intent). HONEST State-B boundary unchanged: the two final arming acts (operator
  signature + launchctl load of the auto-activating agent) are the operator's, by design; everything up to
  them is built, tested, and one command each away. CI: PR #24 greened frontend + rustfmt; clippy cascade debt
  remains. Committed to PR #23; no merge, no activation.
- Wave F2 (2026-07-17, Unattended Condenser successor control plane): invoked via the authoritative master
  goal HAWKING_UNATTENDED_CONDENSER_MASTER_GOAL.md (State B target: legacy running). E0 audit (6-agent
  read-only workflow + direct capture) established: legacy 72B still running (shard 32/37, supervisor pid
  48045, untouched), 176 GB free, CPU saturated (so successor work is Python-light, Rust defers to CI);
  the eco_* scaffold from Wave F1 is a DESCRIPTOR/PLANNING layer not a runtime (no event log, no journaled
  resume, admission faked from an id-map, pipeline validators unenforced); adapters INVERTED from naive
  reading (qwen2.5-dense adapter execution-ready but claim-restricted with treatment hooks
  lora_kd/blockwise_qat/strand_hessian UNSUPPORTED = only method=none; gpt-oss-120B adapter a fail-closed
  0.1-contract whose run refuses exit 78 with real blockers); CI pre-existing red on main (rustfmt drift +
  app/pnpm-workspace.yaml missing packages: under pnpm 9). BUILT the real successor control plane: 13
  tools/condense/succ_*.py modules + 3 test files (28 successor tests, 86 total green): succ_events
  (append-only hash-chained log, tamper-detect, resume), succ_state (BOOT..SEALED_PARENT enforced FSM +
  journaled checkpoint + exact resume + split-brain refusal), succ_queue (durable queue + full status
  vocabulary + section-11.2 row schema + 72B/120B/671B rows with honest blockers), succ_admission (SOURCE-
  BOUND adapter capability probe replacing the id-map: runs the adapter capabilities subcommand and ANDs the
  section-5.4 requirements; on live data qwen ready, gpt-oss not-ready with its 15 real blockers), succ_transition
  (one-use bound tamper-tested transition intent, gate re-derived from disk, all_pass-bypass refused, rollback),
  succ_watchdog (fcntl singleton lease, fail-closed adoption never on pid alone, launchd plist), succ_telegram
  (successor service: events, dedup cursor, bounded retry, heartbeat, redaction, injected+real sender),
  succ_doctor (typed mechanism registry of 13 lineages + controls, real MCKP/DP joint base+healing allocator
  matching brute force, causal-control set; unwired treatment hooks NOT selectable), succ_engine (acquisition
  function, source-bound program materialize/validate, GATED lightweight dispatch, idempotent ingest),
  succ_gc (evidence-closed retirement + safe GC, no-follow, receipts, never-delete classes), succ_eta
  (empirical per-segment ETA, refuses one global constant, marks 120B/giant uncalibrated), succ_audit (signed
  E0 packets), succ_cli (the `successor` command surface). VERIFIED END TO END ON LIVE DATA (read-only,
  plan-only): `successor compile` imports 187 terminal cells, probes the REAL adapters, builds the honest
  queue (72B waiting_old_release, 120B waiting_adapter, 671B waiting_source_authority), and boots the
  controller into WAIT_OLD_RELEASE; status/verify/explain-next/ping/resume/queue all work; the activation gate
  refuses while the campaign runs. NON-INTERFERENCE HELD (additive-only, no campaign write, no heavy launch,
  72B advanced untouched). CI: fixed on a separate scoped branch chore/ci-green (PR #24: cargo fmt across 158
  files + delete app/pnpm-workspace.yaml; fmt gate clean locally, clippy/build/test pending remote CI). VERDICT:
  State B core is REAL (durable event-sourced controller, exact resume, honest queue, transition machinery,
  telegram service, all tested); heavy 72B/120B/giant EXECUTION honestly gated (non-interference + adapter/disk
  blockers + treatment hooks unsupported). NEXT LEVER: on signed release, wire execute_transition to a launchd
  watcher and dispatch the first lightweight 72B calibration probe; build the missing qwen doctor treatment
  hooks and the 120B/deepseek adapters. Committed to PR #23; no merge, no activation.
- Wave F1 (2026-07-17, Condenser Ecosystem Frontier scaffold, isolated worktree): invoked via /goal with the
  frontier directive (governing refs: the Desktop bundle HAWKING_EVENT_HORIZON / _CONDENSER_ECOSYSTEM_FRONTIER
  / _CONDENSER_PULSE). NON-INTERFERENCE HELD: the 72B generation stayed live and untouched throughout
  (qwen2-5-72b__4bpw__codec-control advanced shard 25 -> 28 of 37 under the same supervisor pid 48045 while
  this ran); all work was additive, default-off, in worktree codex/condenser-ecosystem-frontier. BUILT: nine
  additive Python modules tools/condense/eco_*.py + eight test files (58 tests green) implementing the
  Press->Doctor->Horizon->Context->Continuum->Lens->Bridge->Passport->Capsule->Summon layer: (1) the one
  identity/receipt graph (Passport) binding all eight dimensions with claim-separation enforcement (physical
  bytes refuse runtime roles) and a content-addressed prefix/branch DAG; (2) an immutable read-only campaign
  import that validates each cell's on-disk result_sha256/disposition_sha256 byte-identically to the reporter
  and skips every non-terminal cell (verified on the live campaign: 187 terminal cells imported, 140 complete
  + 47 dispositions, all seals valid, running/blocked cells skipped); (3) the ADAPTIVE PLANNER replacing the
  fixed 320-cell matrix with an evidence-driven F0..F4 + diagnose-first + adaptive-descent frontier that emits
  one EXTREME candidate per parent, the Event-Horizon bracket, and the exact boundary probes; (4) the
  data-driven 10-stage pipeline state machine (validators, exact resume, rollback, offline hydration); (5) a
  fail-closed default-off activation gate; (6) 120B+ admission plans; (7) Telegram status/ETA reusing the
  notifier primitives; (8) a unified CLI + materialize. RESULT ON LIVE DATA (plan-only, read-only): under the
  campaign's own promotion gate (ppl rel delta <= 0.08, cap abs delta >= -0.05) NO rate passes yet; at 4 bpw
  the best Doctor branch reaches the very edge (0.5B doctor_full 0.0798) but does not clear it, and 2 bpw
  collapses (5x-28x ppl), so the small-Qwen standalone Event Horizon sits ABOVE 4 bpw under this contract;
  the collapse-boundary floor DESCENDS with scale (3.0 bpw at 0.5-3B, 2.0 bpw at 7-14B, slope ~-0.81/decade,
  the bit-floor-descends hypothesis on real sealed data), predicting 72B ~1.44 bpw / 120B ~1.28 bpw as
  SCHEDULING priors only. 72B TRANSITION: 72B has zero terminal cells (codec-control still running), so it is
  reported awaiting_evidence, bracketed from the smaller-scale anchors, never restarted; once it seals it is
  the planner's first calibration case. VERDICT: physical bytes SEALED (imported + revalidated), quality
  PROVISIONAL only (quality_claims_permitted:false end to end), zero public WIN asserted; EXTREME is UNPROVEN
  pending Doctor-recovery evidence, honestly. An adversarial review pass (1 reviewer) confirmed the seal
  byte-identity, non-interference, and PASS-honesty foundations and found defects that were FIXED with
  regression tests: activation all_pass now binds the pinned plan_sha256 (was only noted); checkpoint_accepted
  rejects empty-dict checkpoints; a campaign adaptive-defer disposition is labeled campaign_deferral (not a
  fake measured computation_collapse) and no longer sets the proven collapse boundary; import is None-status
  crash-safe; admission admissible_now includes disk feasibility; pipeline spec_sha256 is a deterministic
  content address; materialize passports flag their BPW as a planning_proxy. Docs: docs/plans/
  CONDENSER_ECOSYSTEM_FRONTIER.md (constitution mirrored, E0) + docs/plans/FRONTIER_ECOSYSTEM_SCAFFOLD.md
  (status, 72B transition, 120B+ admission, non-interference proof). ACTIVATION gate run against the live
  campaign REFUSES (133 non-terminal, no reporter checkpoints, running cell, supervisor alive, no signature)
  so activation is impossible while the campaign runs. NEXT LEVER: on the signed supersession boundary, make
  72B the first adaptive-planner calibration case; before that, the deferred Rust Continuum/Lens/Bridge crates
  and the real context-evaluation battery (E2..E12). No commit made; worktree only, pending operator approval
  (a draft PR is prepared, not opened).
- Wave 1 (2026-07-16, FIRST on-box session, M3 Ultra 96 GB): invoked via /goal with the added directive to
  fold in docs/plans/HIDE_CONDENSER_GOAL_PROMPT.md (the HIDE SOTA build ladder). ORIENT surfaced the material
  box-reality finding: the delivered box is an M3 Ultra 96 GB / ~162 GB-free, not the M1 Ultra 128 GB / 8 TB
  the whole doc stack assumes (see BOX CORRECTION above); this walls the frontier-resident + RAM-cliff
  moonshots on disk and retires 671B-resident on 96 GB, and it routes the highest-leverage work onto the
  box-cheap spine (thesis gate + HIDE condenser Phase A), which is exactly where the added directive points.
  RAN, all verified on-box: (1) cargo build --workspace GREEN (21.7s incremental, exit 0). (2) With operator
  approval to "stage parents in parallel", staged bf16 7B (Qwen2.5-7B-Instruct, 14 GB) + 14B
  (Qwen2.5-14B-Instruct, 28 GB) via tools/condense/procure.py (hf accelerators blocked by PEP 668, used the
  standard path); disk 162 -> 153 GB free, moonshot gate 1 now download-unblocked. (3) SERVE PATH PROVEN
  end to end: hawking serve --weights Qwen2.5-7B-Q4_K_M.gguf on 127.0.0.1, /v1/chat/completions returned
  coherent correct Rust (fn add -> a + b), finish_reason stop, temp=0 greedy, no template corruption -> the
  shared prerequisite for the thesis gate, microbench B, and HIDE M3 serve-coherence. (4) THESIS GATE RUN
  (SPINE-1, ahead of the moonshots per spine order): built an execution-grounded harness
  (tools/eval/thesis_gate.py + a 15-task original Python smoke corpus) because the frontier read is unanimous
  that substring/self-judge scoring is dead and only execution is a real accept signal; hawking-eval's
  existing scorer is substring-based so this harness supersedes it for the honest number. Result: 14/15 =
  93.3% pass@1, Wilson95 70.2-98.8%, receipt reports/eval/thesis_gate_qwen7b_q4km.json. The single fail was a
  REAL quant defect (the Q4_K 7B emitted `def count_ vowels(` with a space in the identifier, a token glitch),
  not a harness bug (verified by re-probing the raw output). VERDICT: thesis gate moves UNPROVEN(0) ->
  MEASURED-LAB 3.0; the local 7B-Q4_K can code easy Python at ~93% on this box, directionally answering the
  gate's core question, but this is R0/R1: a small smoke corpus, Python not Rust, debug build, no independent
  reproduction yet. NOT a public WIN. CATEGORY MOVEMENT: thesis gate 0 -> 3.0 (MEASURED-LAB); serve-coherence
  proven (feeds Native .tq serve / HIDE M3 once a .tq is baked). NEXT LEVER: (a) write the m3ultra device row;
  (b) thesis-gate tier 2 = EvalPlus HumanEval+ subset + a Rust-via-cargo corpus (the honest harder number);
  (c) HIDE condenser B1 (OpenAI tool round-trip fix) in parallel; (d) with 7B bf16 staged, microbench A
  (MPS-bf16 vs CPU-bf16 doctor) and moonshot gate 1's first doctor pass become runnable. No commit made;
  local only, pending operator approval.
- Wave 2 (2026-07-16, on-box, levers in sequence per operator): drove the reconciled spine
  (thesis gate + HIDE condenser Phase A). LEVER 1 (m3ultra constants): found the constants were ALREADY
  re-derived for this box (studio_manifest DEFAULT_HARDWARE = M3_ULTRA_96GB, weight_budget 78 GB, 819 GB/s,
  disk_reserve 150), so no code change; corrected the Wave 1 ledger over-claim and flagged the REAL residual
  gap (the manifest models a 1 TB SSD / 850 GB budget but the box has only ~162 GB actually free, so
  fits_storage overstates stageable frontier size). LEVER 2 (HIDE condenser B1, OpenAI tool round-trip):
  patched crates/hawking-serve/src/http.rs ChatMessage (content now Option, added tool_calls/tool_call_id/
  name for the standard round-trip) + a rendered_body() that emits the prior tool call and result in
  Hermes <tool_call>/<tool_response> tags so the interaction is VISIBLE next turn instead of dropped; 3 new
  unit tests + full hawking-serve suite green; VERIFIED LIVE e2e: the exact turn-2 history that used to 400
  (assistant content:null + tool_calls, then role:tool result) now returns HTTP 200 and the model correctly
  consumed the tool result (read `pub fn foo() -> i32 { 42 }` and explained it). B1 = MEASURED-LAB, e2e.
  LEVER 3 (thesis gate tier 2): extended tools/eval/thesis_gate.py with a Rust rustc compile-and-run path +
  a 12-task Rust corpus. First Rust run showed 1/12 with all fails "unknown start of token backtick" - caught
  as a HARNESS extraction bug (the Q4_K model opened fences with 2 backticks not 3, so raw backticks leaked
  into the compiled source); fixed extract_code to accept 2+ backtick fences and strip stray fence lines
  (reporting the fake 8.3% would have been a fake-LOSS, banned like a fake-win). Re-run HONEST: Rust 10/12 =
  83.3% (Wilson95 55.2-95.3%), the 2 fails REAL model defects (.is_ascii_alphanumeric on &str; a syntax
  error); Python re-confirmed stable at 14/15 = 93.3% under the new extractor. CATEGORY MOVEMENT: thesis gate
  3.0 -> 3.5 (now has a Python AND a Rust exec-grounded number; Rust is the honest harder + more differentiated
  number for our stack); HIDE tool-call round-trip blocker (from the HIDE deep audit) CLOSED at the serve wire
  layer. Still R0/R1, small corpora, debug build, no independent repro = not a public WIN. NEXT LEVER (4):
  moonshot gate 1 first doctor pass on the staged 7B bf16 + microbench A (MPS-bf16 vs CPU-bf16 doctor
  throughput); needs careful preregistration (<=+2% ppl gate, effective bpw, MULTIWINDOW>=4) and is the heavy
  long-running job. Wave 2 artifacts uncommitted pending approval.
- Wave C (2026-07-11, wiring + hardening): with operator approval, committed the scaffold
  (commit 79f54420, 11 files, 47 tests) then landed three wiring/hardening pieces (commit
  14909a97, 469 insertions): (1) MCP REGISTRATION - `register_mcp_servers` resiliently connects a
  set of MCP servers and registers their tools into the registry (one bad server is recorded as
  an error, does not abort the rest); the client was built-but-never-called. Tested against a
  live stdio fake server. (2) NATIVE TOOL CALLING ON SERVE (Phase 1a) - `/v1/chat/completions`
  now accepts `tools`, renders a Hermes/Qwen `<tools>` system preamble into the prompt, and
  parses the completion back into OpenAI `tool_calls` with finish_reason; a self-contained
  `tool_calls` module (5 tests) wired in without breaking the 35-test serve suite. (3) EDIT.RS
  GUARDS - with approval, added the two minimal safety guards for the pre-existing bugs the
  review found: a backward/out-of-order hunk is now a CONFLICT not a panic, and a removal that
  does not match the file is a CONFLICT not silent corruption (2 tests). NOT re-introducing the
  reverted seek_sequence fuzzy matcher. A second adversarial verification pass over this wiring
  (12 agents) CONFIRMED 8 more real defects - including two in the edit.rs guards I had just
  added (an over-rejection of valid stripped-blank diffs, and a remaining blank-line corruption
  hole the removal guard did not cover), a serve false-positive (untagged prose JSON turned into
  a spurious tool_call discarding the real answer), a streaming gap (raw <tool_call> XML streamed
  instead of structured tool_calls), and three MCP resilience holes (no per-server timeout so one
  hung server stalls the catalog; duplicate ids silently shadowing tools; a false shutdown
  contract + no kill_on_drop). ALL 8 FIXED with 9 regression tests (commit 90de248a): edit.rs now
  keeps blank context lines in the located sequence and verifies them; serve requires an untagged
  call to name a declared tool and buffers streaming to emit real tool_calls; MCP gained a
  per-server timeout, duplicate-id refusal, kill_on_drop, and a corrected doc. A THIRD
  loop-until-dry pass then caught 2 regressions in those very fixes: the edit.rs empty-line
  change over-rejected a patch ending in a trailing blank, and the streaming buffer emitted its
  flush only in the failure-sentinel arm while the decode loop signals normal completion by
  DROPPING the sender - so a streaming chat with tools silently dropped the whole answer. Both
  fixed (commit b6d30db5): hunks now strip trailing blanks, and the streaming flush + [DONE]
  moved outside the token loop (also fixing a pre-existing missing-[DONE] on the non-tools
  success path). A FOURTH lean pass caught 1 narrow edge-case (an all-blank hunk body silently
  reported ok instead of CONFLICT); fixed with a guard (commit 903baa20). The verification loop
  CONVERGED: pass1 6 findings, pass2 8, pass3 2, pass4 1 - a clean decreasing trend, all 17
  fixed with regression tests. FINAL STATE: five commits (79f54420 scaffold, 14909a97 wiring,
  90de248a +8 fixes, b6d30db5 +2 fixes, 903baa20 +1 fix), all per operator approval, LOCAL ONLY
  no push. Full workspace: 392 passed / 7 failed where all 7 are the pre-existing q8_kv Metal
  tests in hawking-core (absent from every .metal source), zero non-q8kv failures. Delivered +
  verified: Phase 0 (parser + parse/lint/dedup/dispatch loop + parallel), Phase 2/3 (jump-forward
  + prompt-lookup spec-decode), Phase 1a (native serve tools/tool_calls incl. streaming), Phase
  1b (memory tool), Phase 1c (MCP registration). Grades stay MEASURED-LAB. STILL GATED for the
  tok/s + first-try-valid RECEIPT that would flip a WIN cell: our own .tq-served model (the
  thesis gate). Unbuilt/unblocked next: rest of catalog (plan.todo/notebook/batch/web/agent.spawn),
  Phase 0 -> live FSM driver wiring, and the runtime mask/spec wiring into forward_multiseq_*.
- Wave D (2026-07-11, deepen to integrated + a caught security bug): per operator "deepen even
  more, get to 10/10, commit/push/merge". Made the keystone REAL: wired the tool-call loop into
  the live agent driver (act_model now parses its output and dispatches emitted calls), with two
  end-to-end integration tests (a stub model emitting <tool_call> for fs.read dispatches through
  the real allow_all_dispatcher; a mutating call is refused). A FIFTH adversarial pass then caught
  a SECURITY vulnerability - exactly the payoff of the discipline: git.diff/git.log passed the
  model-controlled `ref` verbatim as a git arg, so `--output=FILE` was honored as an option and
  WROTE an arbitrary file, and the read-only auto-dispatch gate let a model step trigger it (an
  un-approved out-of-workspace write). Fixed BOTH layers (commit 9d012fbb): git ref guarded with
  --end-of-options (protects all callers, +2 tests), and the model-step gate hardened from
  annotation-only to a deny-by-default allowlist of pure in-process query tools
  (fs.read/list/stat/glob, search.text) - subprocess tools (git.*, shell.*) never auto-dispatch
  from a model step even when read-only (+1 integration test). Verification tally now 6/8/2/1/1 =
  21 real defects found+fixed across 5 passes. Commits this deepen: 7dc59220 (driver wiring),
  cbe2d0d7 (read-only gate), 9d012fbb (security fix). A sixth pass on the security fix is running
  before any push. HONEST 10/10 NOTE: per the audit this category's ceiling is 7.5 (structural
  truthfulness wall - "guaranteed" is an overclaim; honest ceiling is first-try-valid + repair);
  the tok/s WIN receipt is still gated on a served model. Not laundering a 10.
- Wave B (2026-07-11, catalog expansion start): added the `memory` tool (Phase 1b) - a durable
  cross-session scratchpad with `view|create|str_replace|insert|delete|rename`, modeled on
  Anthropic's memory tool, rooted at a private per-workspace directory. The security boundary
  (path-traversal rejection: absolute paths, `..` traversal, and percent-encoded `%2e/%2f/%5c`
  escapes) has a dedicated test per vector. Registered as the 23rd builtin; 8 tests green,
  hide-tools crate clean (52 tests, no warnings). Remaining catalog gaps: plan.todo, notebook,
  batch multi-edit, web.fetch/search, agent.spawn, and wiring the built-but-dormant MCP client.
- Wave A-verify (2026-07-11, adversarial verification + fixes): ran a 5-dimension review
  workflow (finder + independent refuting verifier per dimension, 11 agents) over the Wave A
  scaffold, hunting correctness / losslessness / safety / overclaims. It CONFIRMED 6 real
  defects, each reproduced end-to-end - proof the discipline earns its keep. FIXED 4 in the
  delivered scaffold, each with regression tests: (1) parser dropped a bare-JSON call when a
  `[...]` (markdown link/citation) preceded it -> now scans ALL balanced spans; (2) the loop's
  feedback formatter let untrusted tool output forge a `<tool_call>` (TT8 violation) -> now
  escapes the envelope delimiters in body+name; (5) `scaffold_for` forced a single arg key even
  when optional props existed (a real jump-forward LOSSLESSNESS violation for the shipped
  fs.read schema) -> now gated on a closed single-property schema; (6) audit doc per-file test
  counts corrected. The other 2 (a wrap-slice panic and an unverified `-` drop that silently
  corrupts a file) are PRE-EXISTING bugs in `edit.rs`, surfaced when Phase 1d was reverted;
  reported in audit section 3.1, not fixed (edit.rs is under separate management). Also: Phase
  1d (Codex fuzzy apply_patch) was reverted intentionally in the working tree, so it is
  WITHDRAWN from the scaffold. Honest test state after fixes: 39 owned scaffold tests green
  (parse 14 + runner 10 + shared mod.rs 2 + tool_spec_decode 13); full workspace 392 passed / 7
  failed where all 7 are the pre-existing q8_kv_parity Metal-kernel-missing tests in
  hawking-core (kernels absent from every .metal source), zero non-q8kv failures. Grades stay
  MEASURED-LAB. Commit pending approval. Next lever: drive-to-10 wiring (Phase 0 into the live
  FSM; serve `tools` field), still gated on a served model for the tok/s receipt.
- Wave A (2026-07-11, agentic tool system scaffold): executed the operator's scaffold-all
  directive on the agentic tool system plan (`docs/plans/agentic_tool_system_2026_07_11.md`),
  built from three deep-research passes (Claude Code tool architecture, Codex + other agents,
  constrained/speculative decoding). Landed as real, compiling, unit-tested library code,
  highest-leverage-first: (1) PHASE 0 keystone - the missing model-output tool-call parser
  (`hide-kernel/src/tools/parse.rs`, tolerant across Hermes/OpenAI/fenced/bare formats) plus the
  parse->lint->dedup->dispatch loop (`runner.rs`) that finally calls the built-but-dormant
  `lint_tool_call` + `IdempotencyLedger`, with Hermes-shaped self-correction feedback. (2) PHASE
  1d - upgraded `apply_patch`'s hunk locator to the Codex 4-pass fuzzy `seek_sequence` (exact ->
  trailing-ws -> both-ws -> typographic-unicode) and made context emission byte-exact from the
  file, with two adversarial drift tests. (3) PHASE 2/3 differentiator - the tool spec-decode
  layer (`hawking-orch/src/tool_spec_decode.rs`): schema-aware `ToolCallGrammar` jump-forward
  (envelope prefix, tool-name common-prefix, full skeleton once resolved, validity gate,
  forced_fraction) + `PromptLookup` n-gram drafter + `accepted_prefix_len` lossless accounting.
  (4) PHASE 4 - parallel + purity-gated dispatch primitives (read-only concurrent, mutating
  sequential, order-preserving). Evidence: 41 targeted unit tests green; full-workspace
  regression run recorded next. Grades are MEASURED-LAB (unit tests, below R3): no WIN cell
  flips. The load-bearing gap for the "fastest" claim stays honest and GATED - none of the
  constrained/spec primitives are wired into the batched serve lanes yet, and the tok/s win is
  UNPROVEN until measured on our own served model (the thesis-gate dependency). Audit +
  per-component /10 rating + the specialization thesis: `agentic_tool_system_audit_2026_07_11.md`.
  Commit pending approval. Next lever: confirm the full-suite regression is green, then the
  drive-to-10 wiring (Phase 0 into the live FSM, then serve `tools` field, then the decode-loop
  mask + jump-forward measurement once a served model exists).
- Wave 0-prep (off-box, M3 Pro 18 GB session): converted the parts of Wave 0 that are pure code,
  ahead of the box arriving, so the first on-box session starts from a shorter Wave 0. Landed: (1)
  SPINE-0 FIXED - studio_run.py P6 (`bench_baselines`) and P8 (`codec_bakeoff`) unpacked the 3-tuple
  `eval_targets` into 2 vars (ValueError, the conveyor had never finished); both now unpack 3, and
  `--go-plan` is green. (2) DEVICE CONSTANTS RE-DERIVED for the M1 Ultra 128 GB / 800 GB/s / 8 TB box
  (audit re-derivations 1-4): `size_frontier.DEVICES` gains an `m1ultra` row (112 GB budget, 8 TB,
  800 GB/s) and is the DEFAULT; `ladder.py` RAM 96->128, WEIGHT_BUDGET 84->112, CONDENSE_RESIDENT_MAX
  34->48; `ramcliff_bench`/`expert_cache_policy` bandwidth denominators 400->800 GB/s; the FRONTIER
  serve-fit gate 84->112 GB. (3) THE PIVOT: with the 112 GB budget, 235B-A22B (~39 GB), 405B-dense
  (~68 GB), and 671B@1.0 (~84 GB) all fit RESIDENT (verified via size_frontier: regime=RESIDENT,
  full speed) - the OOC expert pager (hardest serve-build item, Type-1 dead in free-RAM) drops OFF
  the critical path for the entire prize, shortening it to 5 steps; the pager is deferred to the
  deep frontier (744B/1T/3T) only. (4) STUDIO_GO.md locked context + serve-build critical path
  re-derived for the box; Mamba2 gained condense-track arch coverage. All tools py_compile green.
  Also landed: PROCUREMENT accelerated. The ~4.3 TB of bf16 frontier parents were the one real
  bottleneck; `tools/condense/procure.py` forces the fastest-SOTA path (hf_transfer Rust accelerator
  + hf_xet dedup backend + --max-workers, both already installed), turning download from a software
  cap into a pure link cap: the whole frontier manifest lands in ~10 h on gigabit (vs an
  unaccelerated multi-day pull). preflight.py now verifies the accelerators are active; the frontier
  "not staged" message points at procure.py; BASELINES.md documents it. --stream emits the advanced
  download+bake+delete fused plan (peak disk ~= one shard) gated on a baker --append-tq mode.
  Not done (needs the box): cargo build/test --workspace, the two day-0 microbenches, staging bf16
  parents, the oracle fixture, autotune. Commit pending approval. Next lever on-box: finish Wave 0
  (build + microbenches), then SPINE-1 the thesis gate.
- Wave -1 (2026-07-03, off-box, this session): the M1 Ultra potential audit was produced by a nine
  subsystem code-grounded pass, each grade independently adversarially verified (four scores cut, one
  raised). It set the start-of-run scoreboard above, named the two ceilings (bounded 6.33, maximal ~8.4),
  classified every wall as TIME (attackable, a build or a run) or STRUCTURAL (a theorem, bandwidth physics,
  or model capability), and pinned the spine and Wave 0. Verdict: the proven state is honestly 3.25/10 and
  correct as a pre-box reading; the prize is the gap to ~8.4, which is almost entirely never-run
  experiments and unbuilt code that unbounded wall-clock converts. Next lever: Wave 0 on the box (transfer +
  build green, fix the studio_run.py crash, regenerate device constants, stage bf16 parents, the two day-0
  microbenches, pin the oracle fixture), then SPINE-1 the thesis gate.
- CLEAN SLATE Stage A (2026-07-17, this session, additional directive): finished Gravity Forge
  preparation and froze the sub-bit run architecture; Stage B (repo condensation) authorized but NOT
  begun. Fresh live audit: no heavy hawking process (only the separate MoP project on CPU), Forge
  local-only on codex/gravity-forge. Resolved the key contradiction: the 120B tokenizer IS present
  and valid (vocab 200019 o200k_harmony + chat_template.jinja), unblocking real-token F2. Landed a
  4th materially-distinct family (ternary_factor), a real-token F2 fixture (real inputs route to 74
  experts, sub-bit transform_pq output divergence 1.26 vs 0.61 synthetic - the Gaussian proxy was
  optimistic), integrated a sealed launch-disabled Forge program into the ONE merged controller
  (materialize_forge_program; controller refuses launch), and composed 685B/1T/1.6T giant adapter
  contracts from read-only source authority. Built the AUTO-DERIVED pre-run readiness gate
  (hawking.gravity_forge.pre_run_readiness.v1): 12/12 live probes PASS -> authorizes condensation
  (NOT the heavy run). Honest guard: fixture-green means the apparatus runs, not that packing passes
  a capability bar; the science stays negative. 48 tests green. Commits HELD for approval (house
  rule). Next: begin Stage B condensation (needs commit approval) OR keep hardening Stage A (true
  residual-stream F2 via the block attention layer; wire the forge program as a live successor row).
