# STUDIO GO — the one-command entry point for the Hawking frontier program

> Paste target: when Hawking is on the Mac Studio, run preflight, then tell the coding agent "go".
> Everything downstream is already built, gated, resumable, and continuous. No pauses.

## STEP 1 — preflight (always run first)

```
cargo run -p hawking --bin hawking -- studio preflight
```

Checks Python deps, Rust toolchain, RAM/disk, that every `tools/condense/*.py` compiles, that
`cargo check --workspace` is clean, which model parents are staged, that the frontier refresh ledger and
refreshed HF metadata ledger can be written, that the model-aware frontier launch gate is green against
that refresh artifact, that `reports/condense/studio_preflight_summary.json` is written with a canonical
SHA-256 signature over check results plus machine/network/power/thermal evidence, and that the receipt
harness verifies.
Exits 0 (green, safe to `go`) or 1 (red, prints exactly what to fix). Do not run `go` on a red preflight.
The product-facing command delegates to `tools/condense/preflight.py`.
Verify a saved summary with:

```
cargo run -p hawking --bin hawking -- studio verify-summary --path reports/condense/studio_preflight_summary.json
```

## STEP 1B — signed environment receipt

```
cargo run -p hawking --bin hawking -- studio environment-capture --out reports/condense/studio_environment.json
cargo run -p hawking --bin hawking -- studio environment-verify --path reports/condense/studio_environment.json
```

This is a no-download proof-environment capture. It records the target machine class, RAM/disk envelope,
CPU/model identifiers, Hugging Face DNS/route/API reachability, expected link budget, power source,
thermal status, and powermetrics availability. The receipt is independent of preflight so benchmark,
RAM-cliff, and claim bundles can point to a stable signed "this exact Studio was ready" artifact.

## STEP 2 — THE COMMAND

```
python3.12 tools/condense/studio_run.py go
```

Runs the entire frontier program end-to-end, RAM-packed across the 128 GB, continuous, and
resumable (re-run `go` after any interruption — completed models/lanes skip via per-lane floor
files + receipts). Dry-preview first with:

```
python3.12 tools/condense/studio_run.py --go-plan
```

## WHAT `go` DOES (ten phases, automatic — see `docs/plans/quintessential_engine_2026_06_29.md` for the full design)

- **P0 CODEC TRIAGE + STAGE/ADVISE** — one-time: `codec_parallelism.py --catalog` scores every
  candidate codec/kernel design for decode PARALLELISM (not just density) before any Rust build
  time is spent — the direct lesson from the QTIP-on-Metal dead end (serial decode ate the
  bandwidth win). Per-model (inline in P1/P4): `auto_bits.py` + `size_frontier.py` +
  `doctor.py registry --select` + `arch_coverage.py` recommend the bit format, the serve regime
  (RESIDENT/MOE-PAGED/DENSE-OOC), the auto-composed recovery chain, and which Doctor levers are
  architecture-compatible (dense/SSM/MoE — Mamba2 and RWKV-7 both get their real flat-state math,
  not an approximation) — all before any bake. For MoE frontier models, `expert.py cache`
  simulates hot-expert cache hit-rate/blended-tok-s across cache sizes so the eventual OOC pager's
  cache size is chosen from a measured sweep, not a guess.
- **P1 CONDENSE** — the bit-floor-vs-scale curve across {0.5B,1.5B,7B,14B,32B} via the Doctor
  registry's auto-composed L0-L6 stack, multiwindow ppl + capability tripwire, one floor receipt
  per model, then the curve fit (H1 descent vs H0 flat).
  -> `reports/cron/bit_floor_curve.jsonl`, `receipts/official/*-floor.json`.
- **P2 SUBBIT** — the sub-1-bit frontier lane (PTQ1.61, residual two-part, codec-native/recover),
  gated per model by `subbit.py measure` (SUBBIT-0 entropy floor) and, for MoE, `expert.py sensitivity`.
  -> `reports/cron/bit_floor_subbit.jsonl`.
- **P3 SPEC** — `spec_revive.py` on the condensed substrate (7B) + capstone (32B): lossless-verify
  gate -> capture-retrain the eagle5 head -> acceptance measure -> governor bench (exact-match).
  Density (RAM) x spec (latency) stack multiplicatively.
- **P4 FRONTIER** — the 100B+ research prize (235B-A22B / 405B / 671B / DeepSeek-V4-Flash /
  DeepSeek-V4-Pro / GLM-5.2 / Kimi-K2.6 / Kimi-K2.7-Code / Kimi-K2-Instruct; exact HF ids in `BASELINES.md` and
  `tools/condense/studio_manifest.py`),
  serve-oriented since they don't fit the doctor
  budget. Runs on streamed shards (entropy floor + per-expert sensitivity + serve-fit record + the
  auto-composed recovery chain). The native-serve quality + RAM-cliff are the serve build.
- **P5 EVAL + LONG-CONTEXT** — `eval_suite.py` (capability + NIAH) + `ctx_extend.py` (YaRN) +
  `kv.py frontier` (int2/trellis KV, SSD-paging, SSM) + `kv.py hybrid` (STKV: exact recall + unbounded reach).
- **P6 BASELINE** — `bench_baselines.py`: the wedge gate vs llama.cpp IQ1_S/IQ2 + MLX-4bit at matched
  effective bpw. WIN iff it beats IQ2 on 7B+; else reframe to portfolio.
- **P7 CLIFF** — `ramcliff_bench.py --all`: RAM-cliff tok/s + energy J/tok — the headline + the
  energy moat. A CLIFF-WIN requires native serve + >10x tok/s + lower J/tok.
- **P8 CODEC** — `codec_bakeoff.py`: STRAND vs QTIP/QuIP#/AQLM at matched bpw (CUDA-locked rivals
  are offline-encode-only; STRAND is the lone Metal-native trellis serve).
- **P9 SYNTH + SCORECARD** — fit both lane curves + the 70B/405B extrapolation, then `scorecard.py`:
  the populated competitive matrix. **Refuses any WIN cell without an R3+ receipt.**
  -> `reports/condense/SCORECARD.md`. **The deliverable.**

## LOCKED CONTEXT — do NOT reopen

- Hardware: this M1 Ultra Studio, 128 GB unified, ~800 GB/s, 8 TB SSD. Metal/MPS only, NO CUDA, no cloud, no 512 GB box. One project owns
  the whole machine, one heavy job at a time (the RAM scheduler enforces it). Wall-clock is FREE,
  plugged in 24/7 — optimize for maximum proof, not speed. Use the highest-fidelity public source:
  bf16 where available, explicit compressed-source receipts where not.
- Respect the measured dead-ends: low-rank LoRA plateaus (use full-rank), NO uniform-STE through
  the trellis (codec-aware only), AWQ x residual is a non-win, calib = domain-matched not diverse,
  judge low-bit on 7B+ never on 0.5B. `subbit.py admm` already re-confirmed NanoQuant is a low-rank
  resurrection (KILLs on real qwen-05b) — do not iterate on it.

## PROOF DISCIPLINE (the program enforces this; do not relax it)

- EFFECTIVE bpw only (baker AGGREGATE incl. RHT + outlier + side-info), never nominal.
- Quality = output-space ppl vs the f16 parent with MULTIWINDOW>=4 + the multi_eval capability
  tripwire. A floor claim is void if ppl passes but a capability collapses.
- Production headline numbers are CPU-bf16. No public WIN below repro level R3.
- FAKE-WIN BAN: a rung counts ONLY if the compressed payload stays in RAM and decode is folded into
  the GEMV. Any recipe whose served tensor is rehydrated to f16 counts ZERO. Spec-decode counts ONLY
  under the exact-match (bit-lossless) gate.

## THE TWO GATES THAT DECIDE THE MOONSHOT (both currently UNMEASURED, not refuted)

1. Does doctor recovery work resident on 128 GB? (every +dr died on the 18 GB box by swap/timeout, not recipe.)
2. Is MoE expert sensitivity non-uniform? (dense was uniform ~3% spread = dead; MoE is a different regime.)

If both pass: build toward the dream — **DeepSeek-V3 671B @ 1.0 bpw = 84 GB, GLM-5.2 753B @ 1.0 bpw
= 94 GB, Kimi-K2.6/K2.7-Code 1.1T @ 0.75 bpw = 103 GB, and DeepSeek-V4-Pro 1.6T @ 0.50 bpw = 100 GB
served entirely from RAM (RESIDENT, no pager)**
on a single Studio where llama.cpp Q4_K cannot even load. On 128 GB, 235B/405B/671B/GLM-5.2 and the K2
resident stretch rungs fit without an expert pager. If recovery fails: density-only, usable floor
~3.3-3.8 bpw. If expert sensitivity is uniform: fall back to 405B @ 1.34 = 68 GB dense.
0.33/0.5 DENSE is below the information floor — fantasy; only MoE-amortized sub-1 is real.

## THE SERVE-BUILD CRITICAL PATH (the one gate on real wins, in order)

See `docs/plans/quintessential_engine_2026_06_29.md` §"Serve-build critical path" for the full spec.
RE-DERIVED FOR 128 GB: because 235B/405B/671B/DeepSeek-V4/GLM-5.2/K2 resident rungs fit RESIDENT, the OOC expert pager is NO LONGER on
the critical path for the prize (it is Type-1 dead in the free-RAM regime anyway); it is deferred to
the deep frontier only (beyond the resident rungs, SSD-bound). The shortened path:
(1) residual two-part GPU decode parity, (2) all-tensor `.tq` loader, (3) per-expert `.tq` writer +
resident heterogeneous MoE serve, (4) frontier native quality + RAM-cliff RESIDENT (flips P4/P7
GATED->MEASURED), (5) spec-decode governor. [deferred] the OOC pager, only for models > ~112 GB.
Until (1)-(4) land, the size/quality/tps numbers stay honestly GATED.

Proof-mode native serve must fail closed. For Qwen-family `.tq` receipts, run with:

```
HAWKING_QWEN_TQ=1 \
HAWKING_QWEN_TQ_STRICT=1 \
HAWKING_QWEN_TQ_REQUIRE_ALL_LINEAR=1 \
HAWKING_QWEN_TQ_REQUIRE_GPU=1
```

Those levers make a missing sidecar, partial all-linear coverage, or silent CPU fallback an error instead
of a backward-compatible no-op. A RAM-cliff or tok/s receipt is not admissible without this proof-mode
coverage line plus the served-forward parity command.

The laptop-safe runtime contract is now a signed artifact:
`reports/condense/studio_runtime_contract.local.json`. Build and verify it before launch-packet. It
hashes the product runtime source of truth for profiles, workload defaults, energy modes, and the strict
native `.tq` proof-mode receipt requirements; it does not claim that a model has served natively.

The scorecard reads native serve receipts from `reports/condense/<LABEL>_serve.json`. A passing receipt
must state `status=pass`, `native_tq=true`, `rehydrate_f16=false`, `tq_strict=true`, `all_linear=true`,
`gpu_bitslice=true`, `served_forward_pass=true`, and a positive `tok_s`.

## STAGING (download on the Studio; `go` skips what is not present)

14B/32B/72B/MoE/100B+ parents/checkpoints are owner-gated downloads (8 TB SSD). Exact HF ids + sizes are in
`BASELINES.md`. `go` runs whatever is staged and skips the rest, so you can start with 7B+14B
present and add 32B/72B/235B-A22B/671B/DeepSeek-V4/GLM-5.2/Kimi-K2 as they land. The 7B substrate + its
calib/recovery data are the baseline that makes P3 (spec) work. `procure.py --all-frontier
--link-mbs 300 --efficiency 0.7` estimates ~9.8 h download-only for the full nine-model frontier
manifest; perfect 300 MB/s sustain is ~6.9 h. `procure.py --cycle-frontier --link-mbs 300
--efficiency 0.7` is the operational view: download one source, bake/receipt the `.tq`, then release
that source before the next checkpoint. With a 200 GB scratch reserve and all `.tq` outputs retained,
plus a 128 GB HF/Xet cache reserve, that conservative cycle plan peaks around ~2.6 TB live disk instead
of needing the full source manifest resident. `frontier_ops.py storage-plan --storage-budget-gb 8000`
prints a more aggressive wave plan; at the 300 MB/s x 70 percent assumption it batches the nine targets
into checkpointed waves while keeping peak disk well under the 8 TB target.

Operator loop for the giant frontier:

```
cargo run -p hawking --bin hawking -- studio snapshot
cargo run -p hawking --bin hawking -- studio worktree-plan --out reports/condense/worktree_split_plan.local.json
cargo run -p hawking --bin hawking -- studio worktree-plan --verify reports/condense/worktree_split_plan.local.json
cargo run -p hawking --bin hawking -- studio runtime-contract-build --out reports/condense/studio_runtime_contract.local.json
cargo run -p hawking --bin hawking -- studio runtime-contract-verify --path reports/condense/studio_runtime_contract.local.json
cargo run -p hawking --bin hawking -- studio status --storage-budget-gb 8000
cargo run -p hawking --bin hawking -- studio storage-plan --storage-budget-gb 8000 --link-mbs 300 --efficiency 0.7
cargo run -p hawking --bin hawking -- studio lifecycle --storage-budget-gb 8000
cargo run -p hawking --bin hawking -- studio gate --phase procure --require-refresh reports/condense/frontier_refresh.preflight.json --storage-budget-gb 8000
cargo run -p hawking --bin hawking -- studio license-plan
cargo run -p hawking --bin hawking -- studio license-decisions draft --out reports/condense/frontier_license_decisions.draft.json
cargo run -p hawking --bin hawking -- studio license-decisions verify --path reports/condense/frontier_license_decisions.draft.json
cargo run -p hawking --bin hawking -- studio review-plan --refresh reports/condense/frontier_refresh.preflight.json --out reports/condense/frontier_review_plan.local.json
cargo run -p hawking --bin hawking -- studio review-decisions draft --refresh reports/condense/frontier_refresh.preflight.json --out reports/condense/frontier_refresh_review_decisions.draft.json
cargo run -p hawking --bin hawking -- studio review-decisions verify --path reports/condense/frontier_refresh_review_decisions.draft.json
cargo run -p hawking --bin hawking -- studio proof-pack --force
cargo run -p hawking --bin hawking -- studio launch-packet-build --out reports/condense/studio_wave0_launch_packet.json
cargo run -p hawking --bin hawking -- studio launch-packet-verify --path reports/condense/studio_wave0_launch_packet.json
cargo run -p hawking --bin hawking -- studio audit-grade-build --out reports/condense/studio_audit_grade.local.json
cargo run -p hawking --bin hawking -- studio audit-grade-verify --path reports/condense/studio_audit_grade.local.json
cargo run -p hawking --bin hawking -- studio run-next --require-refresh reports/condense/frontier_refresh.preflight.json
python3.12 tools/condense/frontier_ops.py ledger --refresh-hf --out reports/condense/frontier_ledger.launch.json
python3.12 tools/condense/frontier_ops.py refresh
cargo run -p hawking --bin hawking -- studio review-plan --refresh reports/condense/frontier_refresh.json --out reports/condense/frontier_review_plan.launch.json
cargo run -p hawking --bin hawking -- studio review-decisions draft --refresh reports/condense/frontier_refresh.json --out reports/condense/frontier_refresh_review_decisions.launch.json
cargo run -p hawking --bin hawking -- studio review-decisions verify --path reports/condense/frontier_refresh_review_decisions.launch.json
# for every REVIEW candidate printed by refresh:
cargo run -p hawking --bin hawking -- studio review-candidate <hf_id> --decision watch --by <name> --note <why>
# or, after editing/re-drafting a final signed workbook with --by/--note/--final:
cargo run -p hawking --bin hawking -- studio review-decisions apply --path reports/condense/frontier_refresh_review_decisions.launch.json --confirm
cargo run -p hawking --bin hawking -- studio license-plan
cargo run -p hawking --bin hawking -- studio record-license <label> --status accepted --by <name> --license <id> --terms-url <url> --allowed-use research --redistribution none --source-policy local-only-delete-after-bake --note <decision>
# or, after filling a final signed license workbook per model:
cargo run -p hawking --bin hawking -- studio license-decisions sign --path reports/condense/frontier_license_decisions.draft.json --out reports/condense/frontier_license_decisions.final.json
cargo run -p hawking --bin hawking -- studio license-decisions verify --path reports/condense/frontier_license_decisions.final.json
cargo run -p hawking --bin hawking -- studio license-decisions apply --path reports/condense/frontier_license_decisions.final.json --confirm
cargo run -p hawking --bin hawking -- studio storage-plan --storage-budget-gb 8000 --link-mbs 300 --efficiency 0.7
cargo run -p hawking --bin hawking -- studio lifecycle --storage-budget-gb 8000 --link-mbs 300 --efficiency 0.7
cargo run -p hawking --bin hawking -- studio proof-pack --force --out reports/condense/frontier_proof_pack.local.json
cargo run -p hawking --bin hawking -- studio source-provenance-plan
cargo run -p hawking --bin hawking -- studio coverage-plan
cargo run -p hawking --bin hawking -- studio source-provenance-receipt draft <label> --sign-draft
# after procurement fills final HF revision, source format, file manifest, and download/verify receipt:
cargo run -p hawking --bin hawking -- studio source-provenance-receipt sign <label>
cargo run -p hawking --bin hawking -- studio source-provenance-receipt verify <label>
cargo run -p hawking --bin hawking -- studio parity-receipt draft <label> --sign-draft
# after architecture parity runs fill final rows, exact commands, hashes, traces, and verified features:
cargo run -p hawking --bin hawking -- studio parity-receipt sign <label>
cargo run -p hawking --bin hawking -- studio parity-receipt verify <label>
cargo run -p hawking --bin hawking -- studio coverage-receipt draft <label> --kind both --sign-draft
# after baseline/eval runs fill final rows, exact commands, artifacts/receipts, and metrics:
cargo run -p hawking --bin hawking -- studio coverage-receipt sign <label> --kind both
cargo run -p hawking --bin hawking -- studio coverage-receipt verify <label> --kind both
cargo run -p hawking --bin hawking -- studio receipt-plan
# before native serve/RAM-cliff runs:
cargo run -p hawking --bin hawking -- studio receipt-record draft <label> --kind both --sign-draft
# after native serve emits a strict JSON report:
cargo run -p hawking --bin hawking -- studio serve-capture <label> --artifact <artifact.tq> --bench-json <serve_report.json> --command '<exact hawking serve bench command>' --load-receipt <load_trace.json> --served-forward-receipt <served_forward_trace.json> --parity-receipt <serve_parity_trace.json> --force
# after native serve/RAM-cliff rows are final, measured, and traced:
cargo run -p hawking --bin hawking -- studio receipt-record sign <label> --kind both
cargo run -p hawking --bin hawking -- studio receipt-record verify <label> --kind both
cargo run -p hawking --bin hawking -- studio experiment-plan
# before expensive-mode experiments:
cargo run -p hawking --bin hawking -- studio experiment-receipt draft <label> --sign-draft
# after seeds/ablations/rungs/repeats/nulls/rebake rows are final and traced:
cargo run -p hawking --bin hawking -- studio experiment-receipt sign <label>
cargo run -p hawking --bin hawking -- studio experiment-receipt verify <label>
cargo run -p hawking --bin hawking -- studio claim-bundle-build <label>
cargo run -p hawking --bin hawking -- studio claim-bundle-verify reports/condense/<LABEL>_claim_bundle.json
cargo run -p hawking --bin hawking -- studio run-next --storage-budget-gb 8000 --link-mbs 300 --efficiency 0.7
python3.12 tools/condense/frontier_ops.py launch-gate --phase procure --require-refresh reports/condense/frontier_refresh.json
python3.12 tools/condense/procure.py --cycle-frontier --link-mbs 300 --efficiency 0.7
python3.12 tools/condense/procure.py --cache-status
python3.12 tools/condense/procure.py <label> --retries 2 --min-observed-mbs 80 --verify --progress-interval-s 60 --stall-timeout-s 900
cargo run -p hawking --bin hawking -- studio status --storage-budget-gb 8000   # latest observed download MB/s + return code/stall state
# after bake + receipt + verify:
python3.12 tools/condense/frontier_ops.py record-event <label> --stage bake --status pass --duration-s <seconds> --artifact <path>
python3.12 tools/condense/frontier_ops.py artifact-inventory <label>
python3.12 tools/condense/frontier_ops.py release-source <label> --dry-run
```

`release-source` is dry-run by default and refuses to delete source checkpoints unless a `.tq` artifact
exists and either a frontier record or official receipt exists. Use `--yes` only after the dry-run evidence
is correct. `procure.py` pins `HF_HOME`, `HF_HUB_CACHE`, and `HF_XET_CACHE` under `scratch/` by default,
records cache deltas in every download receipt, and exposes `procure.py --cache-prune` as a dry-run cache
maintenance hook. The maximal download command retries resumably with fewer workers if the command fails
or under-runs the observed-throughput floor, records live progress samples, terminates a long no-progress
stall, attaches route/HF/DNS/network diagnostics to bad attempts, then runs `hf cache verify` against the
local dir before the attempt is considered green.
`license-plan` prints the accepted-terms command for every frontier model;
`reviewed` is not enough for procurement. The product-facing `hawking studio record-license` command
must record license id, terms URL, allowed use, redistribution policy, source-retention policy, signer,
and note for accepted terms. `license-decisions draft|sign|verify|apply` provides the same accepted-terms
gate as a signed batch workbook; apply still requires complete final rows and `--confirm`.
`frontier_ops.py lifecycle` is the
operator DAG: if it says `needs-license-review`, do not download; if it says `ready-bake`, bake; if it
says `ready-release-source`, run the release dry-run.
`run-next` is dry-run by default and refuses human-proof gates or placeholder commands; it requires explicit
flags before downloads or heavy compute, and downloads also require a `--require-refresh` artifact. `record-event`
makes bake/serve/eval/archive durations part of the ledger. `artifact-inventory` hashes the durable `.tq`
output and source release refuses to proceed without a matching inventory. Refresh candidates tagged `REVIEW`
must receive a `hawking studio review-candidate` decision before the launch gate can go green;
`review-plan` writes the durable candidate-decision queue from the refresh ledger, and
`review-decisions draft|verify|apply` provides a signed batch workbook for the same human decisions.
The Rust CLI now exposes the same proof state through `hawking studio`: `snapshot` reads signed local
artifacts and prints the current red/green wall; `worktree-plan` groups the dirty tree by subsystem for
review splitting and verifies the signed split receipt; `runtime-contract-build` and
`runtime-contract-verify` seal the native `.tq` proof-mode/runtime policy before launch-packet; `status`, `storage-plan`, `lifecycle`, `gate`,
`license-plan`, `record-license`, `license-decisions`, `review-plan`, `review-candidate`, `review-decisions`, `source-provenance-plan`,
`source-provenance-receipt`, `parity-receipt`, `coverage-plan`, `coverage-receipt`, `receipt-plan`, `experiment-plan`,
`claim-bundle-build`, `claim-bundle-verify`, `proof-pack`, `launch-packet-build`, `launch-packet-verify`, and
`audit-grade-build`, `audit-grade-verify`, `receipt-record`, `experiment-receipt`, `serve-capture` delegate to the guarded
operator tool; and `run-next` prints the next command without executing it. This is the product-facing
shell for Studio wave 0; heavy work still requires explicit proof gates and allow flags.
`proof-pack` is the one-command non-compute wall for the frontier manifest. It writes signed but blocked
draft envelopes for source provenance, parity, baseline/eval coverage, native serve/RAM-cliff, and
experiment depth for each frontier label, then builds `<LABEL>_claim_bundle.local.json` files that hash
those drafts and remain claim-inadmissible. It preserves final receipts unless `--force-final` is
explicitly passed, so it can be rerun before Studio measurements without erasing completed proof. Use
`hawking studio proof-pack --force` as the product-facing entry point; `frontier_ops.py proof-pack` is the
lower-level equivalent.
`launch-packet-build` signs the Studio wave-0 packet by hashing/summarizing the preflight summary,
environment receipt, signed worktree split plan, signed runtime contract, refresh ledger,
license/review workbooks, storage plan, lifecycle dry-run, procurement gate, proof pack, and run-next
dry-run. A valid packet can still be red; `launch-packet-verify` proves the packet did not drift while
preserving `procurement_permitted=false` until every launch gate is green.
`audit-grade-build` signs the audit-grade receipt by parsing the Studio deep-audit facet table, hashing
the external harsher audit, and summarizing the launch packet, proof pack, worktree split, runtime
contract, claim gate, procurement gate, and scorecard artifact. `audit-grade-verify` proves that receipt
did not drift. A valid audit-grade receipt can still report `target_reached=false`; on the laptop it
should also report `runtime_contract_ok=true` and `frontier_claims_walled=true` while the public-claim
gates remain red.
`source-provenance plan` prints the required source checkpoint provenance path for every frontier model.
Before a public claim, each `<LABEL>_source_provenance.json` must be final and signed with exact HF
revision, source kind, source format, procurement command, download/cache verification receipt, and
file-manifest evidence. Compressed/FP4/FP8 sources must record `source_is_prequantized=true` and a
source-format receipt; bf16 parents must record `source_is_prequantized=false` and `source_format=bf16`.
`hawking studio source-provenance-receipt draft` writes signed but blocked source-provenance envelopes for a label.
After verified procurement fills final HF revision, source kind/format, exact procurement command,
download/cache verification receipt, and file-manifest evidence, `hawking studio source-provenance-receipt sign`
refuses to sign anything draft, placeholder-filled, source-format-inconsistent, or missing verification
evidence. `hawking studio source-provenance-receipt verify` rejects unsigned, tampered, draft, or
placeholder source provenance.
`coverage-plan` prints the required baseline and eval receipt paths for every frontier model. Before
any public claim, each `<LABEL>_baselines.json` must contain same-box measured rows or explicit N/A rows
with reasons for llama.cpp Q4_K_M, llama.cpp IQ2_S, llama.cpp mmap OOC, MLX 4-bit, Unsloth Dyn 2.0,
and EXL3/PonyExl3. Each `<LABEL>_eval.json` must cover ppl multiwindow, capability QA, math, coding,
tool-use, long-context recall, RAM-cliff, and native-serve domains with pass or reasoned N/A rows.
`hawking studio parity-receipt draft` writes signed but blocked architecture parity envelopes for a label.
After the real reference-backend and Hawking/native runs are complete, `hawking studio parity-receipt sign`
refuses to sign unless the record is final, measured, threshold-clean, trace-backed, exact-commanded, and
confirms every family-specific required native feature from `frontier_parity.py`.
`hawking studio parity-receipt verify` rejects unsigned, tampered, draft, placeholder, loose-logit,
trace-free, or feature-incomplete parity records.
`hawking studio coverage-receipt draft` writes signed but blocked baseline/eval envelopes for a label.
After the real same-box runs are complete, `hawking studio coverage-receipt sign` refuses to sign unless
the record is final, covers every required row, includes exact non-placeholder commands, and carries a
receipt/artifact/metrics trace for every measured/pass row. `hawking studio coverage-receipt verify`
rejects unsigned or tampered records.
`receipt-plan` prints the stricter serve/RAM-cliff receipt contract. `<LABEL>_serve.json` must use
schema `hawking.frontier_serve.v1`, identify the artifact hash, record exact commands and commit, pass
native `.tq` proof mode, prove no f16 rehydrate, prove all-linear/GPU ownership, set `parity_pass=true`,
report positive tok/s, include a load trace, and prove positive peak/resident/unified-memory fields with
`resident_memory_ok=true`. `<LABEL>_ramcliff.json` must use schema `hawking.frontier_ramcliff.v1`,
be `source=measured`, not modeled, identify the artifact hash, show native `.tq` serving, Q4_K overflow,
>10x cliff, lower resident J/tok, exact commands, commit, and Studio machine class.
`hawking studio receipt-record draft` writes signed but blocked native serve/RAM-cliff envelopes for a label. After the
real native serve and RAM-cliff runs are complete, `hawking studio receipt-record sign` refuses to sign unless the record
is final, measured, strict-native, trace-backed, and non-placeholder. Serve records need a served-forward
or parity trace; RAM-cliff records need powermetrics/energy and baseline traces. `hawking studio receipt-record verify`
rejects unsigned, tampered, draft, placeholder, synthetic/modelled, or trace-free native receipts.
`serve-capture` is the Studio bridge for the native serve half. Feed it an existing `.tq` artifact, the
JSON report emitted by Hawking's native serve bench, the exact command, a load trace, and served-forward
plus parity trace receipts. It hashes the artifact and bench JSON, refuses f16 rehydrate/fallback reports,
requires strict/all-linear/GPU ownership, positive tok/s, resident memory proof, served-forward pass, and
parity pass, then writes the signed `<LABEL>_serve.json`. Use `hawking studio serve-capture` as the product-facing entry point;
`frontier_ops.py serve-capture` is the lower-level equivalent.
`experiment-plan` prints the expensive-mode matrix contract. `<LABEL>_experiment_matrix.json` must cover
at least 3 floor seeds, required calibration ablations, at least 4 bpw rungs, MoE expert allocation
or reasoned dense/N/A, 3 cold and 3 warm RAM-cliff runs, baseline variants, at least 2 archived null
certifications, and a rebake/hash verification row.
`hawking studio experiment-receipt draft` writes signed but blocked expensive-mode matrix envelopes for a label. After
the real experiment rows are complete, `hawking studio experiment-receipt sign` refuses to sign unless the matrix is
final, real/measured, covers every depth requirement, has exact non-placeholder commands, and carries
row-level receipt/artifact/metrics/command traces for every pass/measured/certified row. `hawking studio experiment-receipt
verify` rejects unsigned, tampered, draft, placeholder, trace-free, or depth-incomplete matrices.
`hawking studio claim-bundle-build` signs the final public-claim evidence by SHA-256 after signed source
provenance, signed parity, signed baseline/eval, signed native serve/RAM-cliff, and signed experiment
matrix files exist. `hawking studio claim-bundle-verify` rejects stale, missing, or
claim-inadmissible bundles, and `launch-gate --phase claim` treats missing signed bundles as a hard
failure.

Before any quality, tok/s, or RAM-cliff claim, run:

```
cargo run -p hawking --bin hawking -- studio coverage-plan
cargo run -p hawking --bin hawking -- studio source-provenance-receipt verify <label>
cargo run -p hawking --bin hawking -- studio coverage-receipt verify <label> --kind both
cargo run -p hawking --bin hawking -- studio parity-receipt verify <label>
cargo run -p hawking --bin hawking -- studio receipt-plan
cargo run -p hawking --bin hawking -- studio receipt-record verify <label> --kind both
cargo run -p hawking --bin hawking -- studio experiment-plan
cargo run -p hawking --bin hawking -- studio experiment-receipt verify <label>
python3.12 tools/condense/frontier_parity.py status
HAWKING_QWEN_TQ=1 HAWKING_QWEN_TQ_STRICT=1 HAWKING_QWEN_TQ_REQUIRE_ALL_LINEAR=1 HAWKING_QWEN_TQ_REQUIRE_GPU=1 \
  cargo test -p hawking-core --features tq --test qwen_tq_serve_parity -- --ignored --nocapture
cargo run -p hawking --bin hawking -- studio claim-bundle-build <label>
cargo run -p hawking --bin hawking -- studio claim-bundle-verify reports/condense/<LABEL>_claim_bundle.json
python3.12 tools/condense/frontier_ops.py launch-gate --phase claim --require-refresh reports/condense/frontier_refresh.json
```

The claim gate stays red until each frontier model has a passing signed `<LABEL>_parity.json` receipt plus
passing signed baseline/eval coverage receipts, strict signed native-serve/RAM-cliff receipts, and a complete
signed expensive-mode experiment matrix, then a signed `<LABEL>_claim_bundle.json` that verifies every evidence
file by hash. A modeled or synthetic RAM-cliff record is useful as a probe, but it is not claim-admissible.
