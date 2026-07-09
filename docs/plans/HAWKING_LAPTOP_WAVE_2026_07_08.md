# Hawking laptop wave - 2026-07-08

Scope: laptop-side stabilization against `/Users/scammermike/Downloads/project_audits/hawking_deep_audit_2026_07_08.md`,
`docs/plans/STUDIO_GO.md`, and `docs/plans/STUDIO_DEEP_AUDIT_2026_07_08.md`.

Rule for this wave: no large downloads and no bakes on the laptop.

## Current evidence

- Free disk: started around 12 GiB available on `/Users/scammermike/Downloads/hawking`; after the latest
  validation pass, `df -h .` showed about 62 GiB available, while preflight reported about 67 GB decimal.
  This remains below the 500 GB preflight minimum and is not enough for frontier procurement or broad
  rebuild churn.
- `python3.12 tools/condense/preflight.py`: red, as expected on the laptop.
  Green subchecks: Python deps, Rust toolchain, `tools/condense/*.py` compile, `cargo check --workspace`,
  staged 0.5B/1.5B/7B local ladder, HF transfer/Xet accelerators, frontier refresh, frontier ledger,
  receipt harness, signed preflight summary.
  Red subchecks: RAM 19 GB below Studio target, disk below minimum, HIDE app Node engine
  (`v20.17.0` vs `>=20.19`), frontier launch gate. The current preflight refresh artifact has 39/39
  review-worthy candidates awaiting accept/reject/watch decisions.
- `python3.12 tools/condense/preflight.py --verify-summary reports/condense/studio_preflight_summary.json`:
  pass.
- `target/debug/hawking studio verify-summary --path reports/condense/studio_preflight_summary.json`:
  pass, proving the signed preflight summary through the product CLI.
- `target/debug/hawking studio preflight --quiet`: correctly red on the laptop with exit code 1 and
  blockers `Hardware`, `HIDE app engine`, and `Frontier launch gate`; this also refreshed the signed
  summary without starting downloads or bakes.
- Signed preflight network evidence is now present in `reports/condense/studio_preflight_summary.json`:
  schema `hawking.studio_network_summary.v1`, DNS ok, HF API status 200, no-download probe latency about
  145 ms on this run, and route interface captured. `hawking studio snapshot` prints the same network
  line next to hardware/power/thermal evidence.
- `target/debug/hawking studio environment-capture --json`: correctly red on the laptop with 2 failures.
  It wrote `reports/condense/studio_environment.json`; blockers were RAM 19.327 GB below the 120 GB
  Studio target and disk free 66.074 GB below the 500 GB target. Network, AC power, thermal status, and
  `/usr/bin/powermetrics` availability were captured green without downloads.
- `target/debug/hawking studio environment-verify --path reports/condense/studio_environment.json --json`:
  pass for schema/signature, while preserving `environment_ok: false` for this laptop.
- `target/debug/hawking studio review-decisions draft --refresh reports/condense/frontier_refresh.preflight.json
  --out reports/condense/frontier_refresh_review_decisions.draft.json --json`: pass. The signed batch
  workbook contains 39 refresh-candidate rows and keeps all 39 marked `operator_required`, so it does not
  satisfy the launch gate by itself.
- `target/debug/hawking studio review-decisions verify --path
  reports/condense/frontier_refresh_review_decisions.draft.json --json`: pass for schema/signature and
  reports the same 39 operator confirmations still missing.
- `target/debug/hawking studio gate --phase procure --require-refresh
  reports/condense/frontier_refresh.preflight.json --storage-budget-gb 8000 --json`: correctly red. It
  still blocks on disk, license approval, and `39/39` refresh-candidate decisions, proving the signed
  review workbook did not bypass the human decision gate.
- `target/debug/hawking studio license-decisions draft --out
  reports/condense/frontier_license_decisions.draft.json --json`: pass. The signed batch workbook has
  9 frontier license rows, all marked `operator_required`, and remains `applyable: false`.
- `target/debug/hawking studio license-decisions verify --path
  reports/condense/frontier_license_decisions.draft.json --json`: pass for schema/signature and lists
  the missing accepted-license fields for every model.
- `target/debug/hawking studio license-decisions apply --path
  reports/condense/frontier_license_decisions.draft.json --confirm --json`: correctly red. Even with
  confirmation, incomplete accepted-license rows are not written to
  `reports/condense/frontier_license_acceptance.json`.
- `target/debug/hawking studio license-decisions sign --path
  reports/condense/frontier_license_decisions.draft.json --out /tmp/hawking_license_decisions.sign_test.json --json`
  and verify of that `/tmp` output: pass, proving edited license workbooks can be re-signed without
  applying them.
- `target/debug/hawking studio launch-packet-build --out reports/condense/studio_wave0_launch_packet.json --json`:
  pass as a signed red packet. It records `procurement_permitted: false`, 5 hard failures, and 0 warnings
  after the dirty branch was split into commits on `codex/hawking-studio-stabilization`.
  The hard failures are preflight summary red, environment receipt red, 0/9 accepted licenses,
  39/39 refresh candidates awaiting operator decisions, and the procurement launch gate red. The signed
  worktree split receipt now reports `risk: clean`, 0 dirty entries, and 0 subsystems. The signed runtime
  contract is green inside the packet with 5 runtime profiles, 5 workload defaults, 4 required native
  `.tq` proof-mode env vars, and `signature_ok: true`.
- `target/debug/hawking studio launch-packet-verify --path reports/condense/studio_wave0_launch_packet.json --json`:
  pass for schema/signature while preserving `packet_ok: false`; latest verification reports
  `warning_count: 0`. The packet's run-next dry-run selects 235B-A22B in `needs-license-review` with a
  placeholder `record-license` command, not a download.
- `target/debug/hawking studio runtime-contract-build --out reports/condense/studio_runtime_contract.local.json --json`:
  pass. It writes schema `hawking.studio_runtime_contract.v1` from the product runtime source of truth:
  `hawking_serve::RuntimeProfile::lever_plan`, `WorkloadPack::defaults`, and `EnergyMode`, plus the
  strict native `.tq` proof-mode receipt requirements.
- `target/debug/hawking studio runtime-contract-verify --path reports/condense/studio_runtime_contract.local.json --json`:
  pass with `signature_ok: true`, `profile_count: 5`, `workload_count: 5`, and
  `proof_mode_required: 4`.
- `target/debug/hawking studio density-receipt-build --out reports/condense/studio_density_receipt.local.json --json`
  and `target/debug/hawking studio density-receipt-verify --path reports/condense/studio_density_receipt.local.json --json`:
  pass. The signed local stabilization receipt records repo size, largest files, tracked LOC, disk
  headroom, and generated artifact/model mass; it records cleanup recommendations but does not delete
  evidence or unlock claim gates.
- `target/debug/hawking studio audit-grade-build --out reports/condense/studio_audit_grade.local.json --json`:
  pass. It writes a signed audit-grade receipt from the external harsher audit, Studio facet table, launch
  packet, proof pack, worktree split, runtime contract, density receipt, claim/procurement gates, and scorecard artifact.
  Current state: `target_grade: 8.4`, `target_reached: false`, `frontier_claims_walled: true`,
  `runtime_contract_ok: true`, `density_receipt_ok: true`, 23 facets parsed, 8 facets below target.
- `target/debug/hawking studio audit-grade-verify --path reports/condense/studio_audit_grade.local.json --json`:
  pass for schema/signature while preserving `target_reached: false` and `frontier_claims_walled: true`.
  Lowest facets are Product readiness 4.8, Doctor recovery stack 6.5, Sub-bit / MoE thesis 6.5,
  RAM-cliff and energy demo 6.6, Native `.tq` serving 6.9, Frontier architecture correctness 7.0,
  Evaluation suite 7.6, and Auto bpw resource maximization 8.0.
- `pnpm test` from the repo root: pass after adding a root package shim that delegates to `app`.
  The actual app test run is 101/101 passing. The app package still declares Node `>=20.19`; the local
  node used by the earlier app-level run was `v20.17.0`, so the engine warning remains a local setup risk.
- `cargo test -p hide-kernel -p hide-backend -p hawking-serve --quiet`: pass, 181 tests run and 4 ignored.
- `cargo test -p hawking-core --lib bsize_matrix --quiet`: pass; the long-running
  `model::qwen_dense::bsize_verify_diag::bsize_matrix` real-model diagnostic is classified with
  `#[ignore = "real-model diagnostic; loads Qwen 3B GGUF and sweeps B=1..8, run explicitly with --ignored"]`
  and is skipped in normal runs.
- `cargo test -p hawking-core --lib bsize_verify_diag --quiet`: pass; both bsize diagnostics are ignored,
  confirming the full `hawking-core --lib` suite no longer blocks on that known slow diagnostic by default.
- `cargo test -p hawking-core --features tq --lib tq_projection_parser_covers_gguf_and_hf_names -- --nocapture`:
  pass.
- `python3.12 tools/condense/frontier_coverage_runner.py selftest`: pass. It verifies blocked signed
  drafts, complete signed baseline/eval records, tamper rejection, missing machine/frozen-suite proof
  rejection, no same-box inference from `mode=real`, and operator-style draft output.
- `python3.12 tools/condense/frontier_parity.py selftest`: pass. The legacy parity readiness ledger now
  treats placeholder numeric fields as blocked evidence instead of crashing.
- `python3.12 tools/condense/frontier_parity_runner.py selftest`: pass. It verifies signed parity drafts
  remain blocked, complete final records sign and verify, tampered receipts fail, missing reference trace
  hashes and tensor-map contracts are rejected, and loose logit parity is rejected.
- `python3.12 tools/condense/frontier_receipt_runner.py selftest`: pass. It verifies signed native
  serve/RAM-cliff drafts remain blocked, complete final records sign and verify, tampered serve receipts
  fail, and RAM-cliff rows without energy traces are rejected.
- `python3.12 tools/condense/frontier_doctor_recovery.py selftest`: pass. It verifies signed Doctor
  recovery drafts remain blocked, complete 7B+ recovery records sign and verify, tampered receipts fail,
  heldout task collapse is rejected, and missing heldout traces are rejected.
- `python3.12 tools/condense/frontier_evidence_run.py selftest`: pass. It verifies complete same-run
  Studio evidence bundles sign and verify, tampered bundles fail, signed drafts remain blocked, and
  placeholder source-release decisions cannot pass.
- `python3.12 tools/condense/frontier_serve_capture.py selftest`: pass. It verifies native serve-bench
  JSON can sign a strict serve receipt, the signed receipt verifies through the existing native receipt
  runner, and f16 rehydrate or artifact-hash mismatch reports are blocked.
- `python3.12 tools/condense/frontier_experiment_runner.py selftest`: pass. It verifies signed
  experiment drafts remain blocked, complete final matrices sign and verify, tampered matrices fail,
  trace-free rows, missing trace hashes, and missing same-run IDs are rejected, and missing required
  experiment depth is rejected.
- `python3.12 tools/condense/frontier_provenance.py selftest`: pass. It verifies signed
  source-provenance drafts remain blocked, complete bf16 provenance signs and verifies,
  tamper/placeholder rejection, and compressed-source prequantization enforcement.
- `python3.12 tools/condense/frontier_claims.py selftest`: pass. It verifies blocked missing bundles,
  admissible synthetic bundles with signed source provenance, parity, native receipts, Doctor recovery,
  experiment matrices, and same-run Studio evidence-run bundles, signed-bundle verification, and
  stale-evidence rejection.
- `python3.12 tools/condense/frontier_ops.py selftest`: pass after adding the signed-claim-bundle launch
  gate, signed coverage/native/Doctor/experiment receipt operator paths, native serve-capture route,
  proof-pack route, lifecycle state, product-facing license/review command templates, signed worktree
  split planning, wave-0 packet worktree evidence, signed audit-grade receipts, completion-audit guard,
  same-run Studio evidence-run gate, signed density receipts, signed proof-pack manifests, and atomic JSON
  receipt writes.
- `python3.12 -m py_compile tools/condense/frontier_evidence_run.py tools/condense/frontier_claims.py tools/condense/frontier_ops.py`: pass.
- `cargo check -p hawking`: pass after adding the expanded `hawking studio` proof/lifecycle/plan surface.
- `cargo test -p hawking studio --quiet`: pass, 4 studio unit tests, including signed proof-pack summary
  extraction and Python-compatible signature canonicalization.
- `cargo build -p hawking --bin hawking`: pass after exposing `status`, `storage-plan`, `license-plan`,
  `record-license`, `review-plan`, `review-candidate`, `source-provenance-plan`,
  `source-provenance-receipt`, `coverage-plan`, `coverage-receipt`, `parity-receipt`, `receipt-plan`,
  `receipt-record`, `doctor-recovery-plan`, `doctor-recovery-receipt`, `experiment-plan`,
  `experiment-receipt`, `evidence-run-plan`, `evidence-run-receipt`, `claim-bundle-build`, and
  `claim-bundle-verify` through the product CLI.
  Later builds also passed after adding `preflight`, `verify-summary`, signed preflight network evidence,
  `worktree-plan`, `density-receipt-build|verify`, `audit-grade-build`, `audit-grade-verify`, and
  `completion-audit-build|verify`.
- `cargo run -p hawking --bin hawking -- studio snapshot`: pass. It reads the signed preflight summary,
  procurement gate, claim gate, review plan, and frontier ledger, then prints the current red wall.
- `target/debug/hawking studio worktree-plan --out reports/condense/worktree_split_plan.local.json --json`:
  pass. It writes schema `hawking.worktree_split_plan.v1`, classifies the current dirty tree as high risk,
  signs the receipt, and records 138 dirty entries across 7 subsystems: 102 staged, 15 unstaged,
  21 untracked, 1 deletion.
- `target/debug/hawking studio worktree-plan --verify reports/condense/worktree_split_plan.local.json --json`:
  pass. It verifies schema/signature and preserves `risk: high`, `entries: 138`, `subsystems: 7`.
- `target/debug/hawking studio status --storage-budget-gb 8000 --json`: pass, 9 frontier rows.
- `target/debug/hawking studio storage-plan --storage-budget-gb 8000 --json`: pass, prints the
  storage-wave/ETA model through the product CLI.
- `target/debug/hawking studio license-plan --json`: pass, 9 frontier license-command rows.
- `target/debug/hawking studio --help`: pass, exposes `record-license` and `review-candidate`.
- `target/debug/hawking studio license-plan --json`: pass, every generated command starts with
  `hawking studio record-license`.
- `target/debug/hawking studio source-provenance-plan --json`: pass as a read-only blocker plan, 9 rows,
  all currently blocked on missing final source provenance.
- `target/debug/hawking studio source-provenance-receipt verify 235B-A22B --json`: correctly red,
  because final signed source provenance does not exist yet.
- `target/debug/hawking studio source-provenance-receipt sign|verify 235B-A22B --out-dir <tmp> --json`:
  pass with a synthetic complete source-provenance record in a temp directory, proving the product wrapper
  can sign and verify without mutating real Studio evidence.
- `target/debug/hawking studio coverage-plan --json`: pass, 9 baseline/eval coverage rows.
- `target/debug/hawking studio coverage-receipt verify 235B-A22B --kind both --json`: correctly red,
  because signed final same-box baseline/eval receipts do not exist yet.
- `target/debug/hawking studio coverage-receipt sign|verify 235B-A22B --kind both --out-dir <tmp> --json`:
  pass with synthetic complete baseline/eval records in a temp directory, proving the product wrapper can
  sign and verify machine fingerprint, environment receipt, same-box group, frozen suite/score-set hashes,
  and trace-backed measured rows without mutating real Studio evidence.
- `target/debug/hawking studio parity-receipt verify 235B-A22B --json`: correctly red, because signed final
  architecture parity receipts do not exist yet.
- `target/debug/hawking studio parity-receipt sign|verify 235B-A22B --out-dir <tmp> --json`: pass with a
  synthetic complete parity record in a temp directory, proving the product wrapper can sign and verify
  the adapter/tensor-map, tokenizer/context, trace-hash, and unsupported-exit contract without mutating
  real Studio evidence.
- `target/debug/hawking studio receipt-plan --json`: pass, 9 native serve/RAM-cliff rows.
- `target/debug/hawking studio receipt-record verify 235B-A22B --kind both --json`: correctly red,
  because signed final native serve/RAM-cliff receipts do not exist yet.
- `target/debug/hawking studio receipt-record sign|verify 235B-A22B --kind both --out-dir <tmp> --json`:
  pass with synthetic complete serve/RAM-cliff records in a temp directory, proving the product wrapper can
  sign and verify without mutating real Studio evidence.
- `target/debug/hawking studio doctor-recovery-plan 235B-A22B --json`: pass as a read-only blocker plan,
  showing the required 7B+ measured recovery receipt fields and the `hawking studio
  doctor-recovery-receipt draft` command template.
- `target/debug/hawking studio doctor-recovery-receipt draft 235B-A22B --out-dir <tmp> --force --sign-draft --json`:
  pass while returning `ok=false` inside the JSON, writing an intentionally blocked signed Doctor recovery
  draft instead of treating the red receipt as a CLI failure.
- `target/debug/hawking studio doctor-recovery-receipt verify 235B-A22B --out-dir <tmp> --json`:
  correctly red, because signed final measured Doctor recovery evidence does not exist yet.
- `target/debug/hawking studio experiment-plan --json`: pass, 9 expensive-mode matrix rows.
- `target/debug/hawking studio experiment-receipt verify 235B-A22B --json`: correctly red, because the
  real expensive-mode experiment matrix remains a signed draft with placeholder rows.
- `target/debug/hawking studio experiment-receipt sign|verify 235B-A22B --out-dir <tmp> --json`: pass
  with a synthetic complete experiment matrix in a temp directory, proving the product wrapper can sign
  and verify without mutating real Studio evidence.
- `target/debug/hawking studio evidence-run-plan 235B-A22B --json`: pass as a read-only blocker plan,
  showing the nine upstream evidence files, the one-command same-run requirement, and the required source
  release decision.
- `target/debug/hawking studio evidence-run-receipt draft 235B-A22B --out-dir /tmp/hawking-evidence-cli --force --sign-draft --json`:
  pass while returning `ok=false` inside the JSON, writing an intentionally blocked signed same-run
  evidence bundle instead of treating the missing measured Studio run as a CLI failure.
- `target/debug/hawking studio evidence-run-receipt verify 235B-A22B --out-dir /tmp/hawking-evidence-cli --json`:
  correctly red, because final measured source provenance, parity, native `.tq` serve, same-box baseline,
  frozen eval, RAM-cliff/energy, Doctor recovery, experiment matrix, artifact inventory, and source
  release decision receipts do not exist yet.
- `target/debug/hawking studio review-plan --refresh reports/condense/frontier_refresh.preflight.json --out <tmp> --json`:
  pass as a read-only blocker plan, writes the review queue, currently reports 39 missing decisions, and
  every generated command starts with `hawking studio review-candidate`.
- `target/debug/hawking studio claim-bundle-verify --json`: correctly red with `ok=false`, 9 rows, because
  final public claim bundles do not exist yet.
- `target/debug/hawking studio claim-bundle-build 235B-A22B --out <tmp>/blocked_bundle.json --json`:
  correctly red while writing a signed blocked bundle with 284 blockers and `claim_admissible=false`.
- `target/debug/hawking studio claim-bundle-build 235B-A22B --root <tmp-root> --json` and
  `target/debug/hawking studio claim-bundle-verify <tmp-root>/reports/condense/235B-A22B_claim_bundle.json --root <tmp-root> --json`:
  pass in an isolated temp root with synthetic complete signed source-provenance, parity, coverage,
  native serve/RAM-cliff, Doctor recovery, and experiment receipts, proving the product wrapper can build
  and verify an admissible bundle without mutating real Studio evidence.
- `cargo run -p hawking --bin hawking -- studio lifecycle --storage-budget-gb 8000 --json`: pass. The
  current lifecycle has 9/9 `needs-license-review` nodes, and its next-command hints use
  `hawking studio record-license`.
- `target/debug/hawking studio record-license 235B-A22B --status accepted`: correctly red before any
  ledger write, because accepted license terms require signer, license id, terms URL, allowed use,
  redistribution policy, source policy, and note.
- `cargo run -p hawking --bin hawking -- studio run-next --require-refresh reports/condense/frontier_refresh.preflight.json --json`:
  pass. It prints the first human-proof license command and does not execute it.
- `cargo run -p hawking --bin hawking -- studio proof-pack --force --json`: pass. Product-facing wrapper
  for the all-frontier signed draft wall returned `ok=true`, 9 frontier models, 81 evidence rows, and 9
  intentionally blocked local claim bundles; the generated manifest verifies with `signature_ok=true`.
- `target/debug/hawking studio serve-capture 235B-A22B --artifact <tiny.tq> --bench-json <strict_report.json> --command <exact serve command> --load-receipt selftest://load --served-forward-receipt selftest://served-forward --parity-receipt selftest://parity --out <tmp>/serve.json --force --json`:
  pass. Product-facing wrapper returned `ok=true`, `status.ok=true`, `strict_ok=true`, wrote the receipt,
  and the signed receipt records `status=pass`, `native_tq=true`, `rehydrate_f16=false`, `tok_s=12.5`,
  load trace, and resident memory proof fields.
- `target/debug/hawking studio snapshot --json`: pass; `next_safe_commands` now includes
  `hawking studio source-provenance-receipt verify`, `hawking studio receipt-record verify --kind both`,
  `hawking studio doctor-recovery-plan`, `hawking studio doctor-recovery-receipt verify`,
  `hawking studio evidence-run-plan`, `hawking studio evidence-run-receipt verify`,
  `hawking studio experiment-receipt verify`, `hawking studio claim-bundle-build`, and
  `hawking studio claim-bundle-verify` before lifecycle gates.
- `target/debug/hawking studio source-provenance-plan 235B-A22B --json`: pass; generated command
  template now starts with `hawking studio source-provenance-receipt`.
- `target/debug/hawking studio experiment-plan 235B-A22B --json`: pass; generated command template now
  starts with `hawking studio experiment-receipt`.
- `target/debug/hawking studio gate --phase claim --json`: correctly red. Fresh gate output includes
  `frontier-doctor-recovery: 0/9 Doctor recovery receipts verify`,
  `frontier-studio-evidence-runs: 0/9 Studio evidence-run bundles verify`,
  `summary.doctor_recovery_blocked=9`, and `summary.studio_evidence_run_blocked=9`.

## Qwen dense long-tail classification

`model::qwen_dense::bsize_verify_diag::bsize_matrix` is now explicitly classified as a slow real-model
diagnostic with `#[ignore]`. It loads the local Qwen 3B GGUF when present and sweeps B=1..8 over a
repeating-cycle KV state. It is useful for manual regression diagnosis, but it should not sit in the
default local `hawking-core --lib` lane on a low-disk laptop.

Run it only when intended:

```bash
cargo test -p hawking-core --lib bsize_matrix -- --ignored --nocapture
```

The fast TQ parser proof remains in the default targeted lane with `--features tq`.

## Dirty tree by subsystem

The original dirty branch was mixed and should not have been treated as one reviewable unit. The
product-facing split artifact is `reports/condense/worktree_split_plan.local.json`, generated with:

```bash
hawking studio worktree-plan --out reports/condense/worktree_split_plan.local.json
hawking studio worktree-plan --verify reports/condense/worktree_split_plan.local.json
```

Pre-split generated split:

| subsystem | entries | staged | unstaged | untracked | risk | suggested branch |
|---|---:|---:|---:|---:|---|---|
| condense-frontier-proof | 26 | 0 | 11 | 15 | high | `codex/condense-frontier-proof` |
| hawking-core-runtime | 4 | 1 | 2 | 1 | high | `codex/hawking-core-runtime` |
| hide-ui-tauri-assets | 97 | 95 | 0 | 2 | medium | `codex/hide-ui-tauri-assets` |
| studio-docs-audits | 6 | 2 | 2 | 2 | low | `codex/studio-docs-audits` |
| hide-backend-kernel | 3 | 3 | 0 | 0 | medium | `codex/hide-backend-kernel` |
| ci-config | 1 | 1 | 0 | 0 | medium | `codex/ci-config` |
| local-deps-generated | 1 | 0 | 0 | 1 | high | do not commit |

Recommended split order from the artifact:

1. Condense/frontier proof gates and receipts.
2. Hawking core/runtime/CLI, including `hawking studio`.
3. HIDE desktop/Tauri/UI/assets.
4. Studio docs/audits.
5. HIDE backend/kernel.
6. CI config.
7. Local dependency/generated cleanup, especially `node_modules/`, kept out of source commits unless
   intentionally archived.

Post-split review stack on `codex/hawking-studio-stabilization`:

- `d3140462 Split HIDE launcher stabilization stack`
- `0c7015c3 Add Studio proof lifecycle and native TQ gates`
- `bacd6edb Record post-split Studio stabilization receipts`
- `6c1b5a0e Add Studio completion audit guard`
- `a6d27c28 Add Studio Doctor recovery receipt gate`
- this note now covers the same-run Studio evidence-run receipt/gate stabilization on top of those
  review commits.
- `reports/condense/worktree_split_plan.local.json` verifies with `risk: clean`, 0 dirty entries,
  0 staged entries, 0 unstaged entries, 0 untracked entries, and 0 subsystems.
- The root `node_modules/` artifact is ignored by `.gitignore`; it is 4 KB locally and is not part of
  the review stack.
- Local Node remains `v20.17.0` while `app/package.json` declares `>=20.19`; app tests still pass, and
  the mismatch is now captured as a signed `HIDE app engine` preflight blocker rather than a loose note.

Post-split density receipt:

| item | size |
|---|---:|
| repo checkout | 63G |
| `models/` | 35G |
| `scratch/` | 18G |
| `target/` | 4.2G |
| `.claude/` | 2.4G |
| `reports/` | 23M |
| root `node_modules/` | 4K |

Largest local artifacts remain model/source payloads, not source code: Qwen 7B GGUF 4.4G, MLX Qwen 7B
4-bit safetensors 4.0G, Qwen 7B scratch shards at 3.3-3.7G each, and Qwen 32B GGUF shards at 3.7G each.
No frontier downloads or bakes were started on the laptop.

## Non-compute launch artifacts prepared

Generated locally without downloads or bakes:

- `reports/condense/frontier_license_plan.local.json`: 0/9 accepted licenses; records the exact
  `hawking studio record-license` command template per frontier model.
- `reports/condense/frontier_review_plan.local.json`: 39/39 review-worthy refresh candidates still need
  accept/reject/watch decisions; records command templates.
- `reports/condense/frontier_coverage_plan.local.json`: 0/9 baseline and eval coverage; records required
  baseline/eval paths.
- `reports/condense/<LABEL>_source_provenance.json` for every frontier label: signed but blocked draft
  source-provenance envelopes. They verify as intentionally incomplete until final HF revision,
  source kind/format, procurement command, download/cache verification receipt, and file-manifest evidence
  replace TODO state. Compressed K2/DeepSeek sources require `source_is_prequantized=true` plus a
  source-format receipt; bf16 parents require `source_is_prequantized=false` and `source_format=bf16`.
- `reports/condense/235B-A22B_parity.json`: signed but blocked draft architecture parity envelope; it
  verifies as intentionally incomplete until final config/tokenizer hashes, exact commands,
  reference/native traces, logit thresholds, greedy-match windows, and verified native features replace
  TODO state.
- `reports/condense/235B-A22B_baselines.json` and `reports/condense/235B-A22B_eval.json`: signed but
  blocked draft coverage envelopes; they verify as intentionally incomplete until final same-box rows
  replace TODO state and fill machine/environment/frozen-suite proof.
- `reports/condense/frontier_receipt_plan.local.json`: 0/9 native serve and RAM-cliff receipts; records
  required receipt paths.
- `reports/condense/235B-A22B_serve.json` and `reports/condense/235B-A22B_ramcliff.json`: signed but
  blocked draft native receipts; they verify as intentionally incomplete until final artifact hashes,
  exact commands, native serve traces, powermetrics/energy traces, baseline traces, and metrics replace
  TODO state.
- `reports/condense/235B-A22B_doctor_recovery.json`: signed but blocked draft Doctor recovery receipt;
  it verifies as intentionally incomplete until final 7B+ measured PTQ/recovered receipts, heldout eval
  trace, artifact hash, degradation numbers, exact commands, and no-task-collapse gate replace TODO state.
- `reports/condense/frontier_experiment_plan.local.json`: 0/9 expensive-mode experiment matrices; records
  required matrix paths.
- `reports/condense/235B-A22B_experiment_matrix.json`: signed but blocked draft expensive-mode matrix;
  it verifies as intentionally incomplete until final seeds, ablations, bpw rungs, RAM-cliff repeats,
  baseline variants, null certifications, rebake/hash proof, exact commands, and row-level traces replace
  TODO state.
- `reports/condense/<LABEL>_studio_evidence_run.json` for every frontier label: signed but blocked
  draft same-run Studio evidence-run envelopes. They verify as intentionally incomplete until the final
  one-command Studio run links final source provenance, architecture parity, native `.tq` serve,
  same-box baselines, frozen eval coverage, RAM-cliff/energy, Doctor recovery, experiment matrix,
  artifact inventory, and source-release decision receipts with matching hashes.
- `reports/condense/235B-A22B_claim_bundle.local.json`: blocked signed bundle probe for 235B-A22B;
  `claim_admissible=false` with 284 blockers after signed-draft source provenance, parity, coverage,
  native receipt, Doctor recovery, experiment, and same-run evidence-run checks were added. Every
  evidence file in the bundle now exists and hashes; all blockers are draft/final-measurement
  requirements.
- `reports/condense/frontier_proof_pack.local.json`: signed all-frontier local proof-pack summary. It
  records 9/9 blocked local claim bundles, 81/81 signed draft evidence files present, and zero
  unsigned/missing draft envelopes across source provenance, parity, baseline/eval, native serve/RAM-cliff,
  Doctor recovery, experiment, and same-run evidence-run evidence.
- `reports/condense/<LABEL>_claim_bundle.local.json` for every frontier label: all nine local bundles
  are claim-inadmissible with 280-284 blockers each, and all nine evidence files per bundle exist and
  hash.
- `reports/condense/frontier_claim_launch_gate.local.json`: claim-phase launch gate snapshot with signed
  bundles, Doctor recovery, and same-run Studio evidence-run included as hard failures; fresh output
  records `doctor_recovery_blocked=9` and `studio_evidence_run_blocked=9`.
- `reports/condense/worktree_split_plan.local.json`: signed dirty-tree split receipt. It verifies through
  `hawking studio worktree-plan --verify` and is summarized by the wave-0 launch packet.
- `reports/condense/studio_runtime_contract.local.json`: signed runtime/proof-mode contract. It hashes the
  product runtime profile/workload/energy policy and the native `.tq` proof-mode requirements that make a
  missing sidecar, partial all-linear coverage, CPU fallback, f16 rehydrate, or missing served-forward
  parity trace inadmissible.
- `reports/condense/studio_wave0_launch_packet.json`: signed red launch packet. It hashes/summarizes the
  preflight summary, environment receipt, signed worktree split, runtime contract, refresh ledger,
  license/review workbooks, storage/lifecycle dry-runs, procurement gate, signed proof pack, and run-next
  dry-run.
- `reports/condense/studio_audit_grade.local.json`: signed audit-grade receipt. It hashes the external
  harsher audit and Studio audit table, summarizes the launch packet/proof pack/worktree split/gates, and
  records that the current 8.4 target is not proven while the runtime contract is green and all frontier
  claims are explicitly walled.
- `reports/condense/studio_completion_audit.local.json`: signed Hawking Studio 10/10 completion audit.
  It verifies as a valid red receipt through `hawking studio completion-audit-verify`, with 4/19
  requirements passing locally (`split_clean_worktree`, `native_tq_runtime_contract`,
  `density_receipt_signed`, `proof_pack_signed_wall`) and 15/19 blocked
  on the actual Studio evidence: preflight/environment, human license/review gates, procurement gate,
  native `.tq` serve, architecture parity, 7B+ Doctor recovery, RAM-cliff/energy, same-box baselines,
  frozen eval coverage, source provenance, experiment depth, same-run Studio evidence bundle, claim
  gate, signed claim bundles, and the final audit-grade target.

Current launch-gate wall:

- laptop RAM/disk fail by design;
- cycle/free-disk and storage-wave checks fail on the laptop;
- accepted licenses are 0/9;
- candidate decisions are 0/39 in the current preflight refresh/review-plan artifacts;
- frontier parity remains 9/9 blocked for public claims; the 235B-A22B parity envelope is a signed draft,
  not a final measured receipt;
- source provenance remains 9/9 blocked for public claims; the source-provenance envelopes are signed
  drafts, not final HF revision/file-manifest receipts;
- baseline/eval/native-serve/RAM-cliff/Doctor/experiment/same-run evidence remains incomplete until
  Studio or smaller proof-complete runtime work produces signed receipts;
- the 235B-A22B baseline/eval envelopes are signed drafts, not final receipts, so coverage remains 0/9;
- the 235B-A22B native serve/RAM-cliff envelopes are signed drafts, not final receipts, so native receipt
  gates remain 0/9;
- the 235B-A22B Doctor recovery envelope is a signed draft, not a final measured 7B+ recovery receipt, so
  Doctor recovery remains 0/9;
- the 235B-A22B experiment matrix envelope is a signed draft, not a final receipt, so experiment depth
  remains 0/9;
- the same-run Studio evidence-run envelopes are signed drafts, not final one-command measured bundles,
  so `frontier-studio-evidence-runs` remains 0/9;
- signed public claim bundles are 0/9. The `.local` proof-pack bundles correctly fail verification
  because the underlying evidence files are intentionally draft-blocked; verifier JSON now reports
  `signature_ok=true` for 9/9 local bundles, `claim_admissible=false` for 9/9 local bundles, and no
  evidence file is missing from any local bundle now.
- the signed worktree split receipt is valid and clean after the branch split, with 0 dirty entries and
  0 subsystems; the launch packet no longer carries a dirty-tree warning.
- the signed audit-grade receipt is valid, but `target_reached=false`; it records the external audit's
  current overall grade as 6.4/10, the Studio potential as 8.4/10, the operator-plan grade as 9.8/10,
  `runtime_contract_ok=true`, `density_receipt_ok=true`, `proof_pack_signature_ok=true`, and 8 Studio
  facets below the 8.4 target.
- the signed completion audit is valid, but `completion_ok=false`; it records the local stabilization
  boundary explicitly and refuses to treat drafts or missing receipts as native serve, parity, Doctor
  recovery, RAM-cliff/energy, baseline/eval, experiment, same-run evidence, or final claim proof.
- `hawking studio snapshot` correctly reports preflight red, procurement gate red, claim gate red,
  0/39 candidate decisions reviewed, 9/9 lifecycle nodes waiting on accepted license records, and
  signed proof-pack `blocked_claims=9/9`, `local_signed_count=9`, and 81 evidence rows.

## Next laptop-safe moves

- Reclaim disk before any broader Rust test sweep.
- Keep frontier downloads and bakes blocked on this machine.
- Fill human-only records only after actual terms/review decisions are made; do not fake accepted licenses
  or candidate decisions.
- Build the next custom-runtime pass around the smallest proof-complete native `.tq` serve path, with
  receipt schema and same-box baseline expectations preregistered before running it.
- On the actual Studio box, use `hawking studio evidence-run-plan` before the first final claim attempt
  so native `.tq` serve, parity, baselines, evals, RAM-cliff/energy, Doctor recovery, experiment depth,
  artifact inventory, and source-release decision are captured in one signed same-run bundle.
