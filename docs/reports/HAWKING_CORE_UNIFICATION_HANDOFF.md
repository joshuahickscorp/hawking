# Hawking core unification handoff

This is the integration record for `codex/hawking-core-unification`. It does
not authorize a merge, deployment, resident restart or live-state migration.

## Git basis and commit order

- Base: `ce5e1b2a551c6817b6e82f05cf937dbdc77bb00e`
- Worktree: `/Users/scammermike/Downloads/hawking/.worktrees/hawking-core-unification`
- Branch: `codex/hawking-core-unification`
- Implementation tip: `1146b6201a269341c96553ce73d0b4794dfecb99`
- `c1cc507fe41273b9366fbcbf5cf33d9898b62065` — accepted vocabulary,
  migration authority, architecture ownership and reproducible language census.
- `3914e2acf9d7891294d4b68258d0f0ce12f1b48c` — first native Gravity
  legal-region contract and ratio-zero plan adapter.
- `635bcf77b64a2d21c6df703fabe6f570a14e8b10` — first Hawking
  model/action surface and admitted/specimen split.
- `1146b6201a269341c96553ce73d0b4794dfecb99` — connects the native contract
  to the real full-sequence ratio-zero caller, closes validation gaps, hardens
  exact artifact selection and records the reviewed limitations.

The handoff document itself may be followed by one report-only commit that
records this implementation tip. The receiving lane should use `git rev-parse
codex/hawking-core-unification` for the current branch tip and treat the
implementation tip above as the last commit that changes behavior.

Apply the commits in that order. The active main checkout remains at the base
SHA and has substantial uncommitted Flash work. Reconcile in a disposable
integration worktree after that owner commits its side.

## Implemented ownership and vocabulary

Hawking is the public system. Gravity is its transformation and physical
optimization function; collapse is the verb. NR and NX remain separate semantic
and deployment identities. HCLI, HIDE, AgentOS, historical Deep Gravity,
Noetic and Singularity spellings survive only where current imports, schemas,
commands or evidence still depend on them. The complete caller/dependency and
sunset crosswalk is in `docs/HAWKING_NOMENCLATURE.md`.

The Rust `hawking` binary now exposes these working entry points:

```text
hawking gravity ...
hawking models [--json]
hawking actions [--json]
hawking select <artifact> execute
hawking select <artifact> serve
hawking select <artifact> web
```

`hawking gravity` is native. Model actions forward through the existing HCLI
implementation boundary. The resolver returns an admitted profile's exact path,
SHA-256 revision and supported actions. Rust validates that contract before
dispatch. `execute` refuses noninteractive stdin. Raw ModelLake directories are
reported under `specimens` and excluded from executable bodies. Fresh Web and
serve launches require an admitted binding. An existing Web resident is reused
only when its exact loaded path matches; otherwise Web requests an exact-path
switch and refuses a fixed-body mismatch.

The tracked native profile now has an explicit `admission.status`, contract and
evidence list. Its file digest is the selection revision. This is a schema
addition on the branch; no resident or persisted live selection was changed.

The action catalog is a typed Rust metadata projection and the three selection
actions share the same resolver and forwarding path. It does not yet generate
Clap grammar, HCLI parsing and the OpenWebUI menu from one schema. Two serve
contracts also remain: native `hawking gravity serve` and the admitted HCLI
profile adapter. Claiming a fully unified action graph or a single serve engine
would therefore be premature. Jobs, Status and Settings retain their existing
implementations and are not advertised as new root commands. Hawking Web source
is absent from committed HEAD; only a built bundle and source map exist, so menu
projection is deferred until its source authority is recovered.

## Native legal-region tranche

`hawking_core::gravity::execution` now owns a native semantic graph, legal
region, boundary, state/effect, backend-resource and schedule contract. It
checks:

- graph operation/output identity and a semantic SHA-256 fingerprint;
- acyclic dependencies, convex legal regions and exact incoming/outgoing data and
  control boundaries, including outside consumers;
- state declarations, read/write effects, loop-carried lifetime/ownership,
  alias safety, randomness effects and refusal of unsynchronized shared writable
  state;
- numerical policy, supported backend, checked scratch arithmetic, unique
  program IDs, exact operation coverage, dependency order and explicit waits
  across backend programs.

The model adapter builds a six-operation, four-program Metal plan with explicit
waits. `DeepSeekV4FullSeqAttentionDeviceExecutor::execute_position` calls
`DeepSeekV4Ratio0AttentionDeviceExecutor::prepare` before Metal encoding when
`compress_ratio == 0`, so an invalid plan now blocks a real execution path.
This is stronger than the earlier template-only wiring, but its qualification
remains `STRUCTURALLY_VALIDATED_PLAN_ONLY`. GPU execution, numerical parity,
throughput and full PhysicalGraph forwarding remain deferred to the protected
integration lane.

No numerical oracle was removed. No active Flash kernel was relocated. Python
PhysicalGraph remains an independent planning oracle because differential
fixtures do not yet justify deleting it.

## Duplicate ownership and automatic reuse status

This branch removes two ambiguous selection behaviors: a source specimen can no
longer become executable through discovery alone, and an explicit Web request
cannot silently retain a different resident. HCLI launch discovery is also
anchored to `HAWKING_HCLI_ROOT` or the Rust crate's repository root rather than
depending solely on an ambient installed Python package.

Broader owner consolidation is incomplete. The canonical method data is at
`tools/foundry/GRAVITY_METHOD_REGISTRY.json`, while the research
`gravity_potency.py` reader still defaults to its own directory. No future-work
caller automatically retrieves and invokes an applicable method, and no
compatible/incompatible execution demonstration was earned. The terminology
guard is now exercised by the ordinary focused Python suite, but it remains a
bounded added-line check rather than a repository-wide generated-interface
system. These are explicit unmet obligations, not completed reuse claims.

## Source census

`tools/loc/hawking_loc.py` preserves the repository's established physical-line
policy: tracked first-party active source; generated, documentation archive,
vendor and build output reported separately; Rust share uses Rust plus Python as
the denominator. Counts below are refreshed after the final code is staged.

| Scope | Base Rust | Base Python | Base Rust share | Final Rust | Final Python | Final Rust share |
|---|---:|---:|---:|---:|---:|---:|
| Whole active repository | 442,732 | 522,102 | 45.886857% | 444,252 | 522,565 | 45.949957% |
| Established minimum product | 373,881 | 86,002 | 81.299157% | 375,401 | 86,147 | 81.335202% |

The established minimum-product measure counts all physical lines in Rust
`src/`, including inline `#[cfg(test)]` modules. It is therefore a file-level
product measure, not a proven test-excluding production/runtime measure. It
exceeds 75%; the stricter production/runtime interpretation requested by the
mandate remains unproven. Whole-repository Rust share remains below 75%.
Research and verifier Python was retained rather than deleted or moved to alter
the denominator. Final combined active source is 1,155,461 physical lines in
2,387 files; established minimum product is 509,093 lines in 653 files. The
excluded counts remain 48,716 vendored lines, 76 archived lines and zero
generated lines.

## Validation evidence

The final focused validation set is recorded here after the last commit:

- `cargo fmt --manifest-path crates/hawking-core/Cargo.toml -- --check`
- `cargo fmt --manifest-path crates/hawking/Cargo.toml -- --check`
- `cargo test -p hawking-core gravity::execution --lib`: ten tests.
- `cargo test -p hawking-core ratio_zero_template_uses_validated_stateful_region --lib`:
  one test.
- `cargo test -p hawking public_surface_tests --bin hawking`: four tests.
- `cargo check -p hawking`.
- Catalog, admission, switching, explicit Web selection and OpenAI serve tests:
  37 passed.
- Nomenclature and Flash/science-boundary suite: 36 passed.
- `tools/verify/hawking_terminology.py --base <base>` and `--selfcheck`.
- Isolated command smokes for `hawking actions --json`, `hawking models --json`,
  resolve-only selection and refusal of an unqualified ModelLake directory.

No baseline failure was established in these focused tests. A mistakenly named
test path was an operator invocation error and is not counted as a product
failure. New failures are zero after the final focused run.

No protected performance measurement was attempted while Flash was active. The
model/action path adds one short Python resolver process and one Python action
process per invocation; it transfers metadata only, not model arrays. Memory,
copy counts and caller-to-result p50/p95 were not measured. Native region
validation allocation and latency are also unmeasured. This branch makes no
performance claim from those paths.

## Isolation and live-state evidence

Python tests used a worktree-local virtual environment, temporary HOME/XDG roots
where relevant, explicit `PYTHONPATH`, and isolated fixtures. Cargo used
`.worktree-state/cargo-target` with at most two build jobs. These paths are
ignored. Cargo fetched missing registry dependencies (`fastrand`, `tempfile`,
`getrandom` and `rustix`) into its normal dependency cache during the isolated
build; this was the only observed host-cache exception. No weights,
Metal/GPU/ANE kernels, package prefix, test database, symlink, launch agent,
receipt authority or ModelLake file was written. One early catalog smoke made a
GET request to the production `/health` endpoint before the explicit isolated
base option was added; it performed no selection or mutation. The hawkingd and
OpenWebUI processes were not restarted.

## Integration conflicts, migration and rollback

The exact branch/main dirty overlap at handoff is:

```text
crates/hawking/src/main.rs
docs/HCLI_WEB.md
hcli/catalog.py
hcli/nomenclature.py
hcli/serve.py
hcli/tests/test_catalog_and_switching.py
hcli/use.py
hcli/web.py
```

These files require semantic three-way reconciliation. The native execution
files do not overlap the current main dirty set, but the main owner must still
rerun them against its eventual committed Flash tip. Do not choose wholesale
`ours` or `theirs` for the overlapping surface files.

`CLAUDE-CODEX-EXODUS.md`, `CLAUDE_CODEX_EXODUS.json`,
`CLAUDE_ONLY_REMAINING` and `AGENTS.md` exist only as untracked files in the
active main checkout at this base. They were read as migration evidence and not
copied. The main owner should reconcile them from its own authoritative state.

There is no live-state migration command in this tranche. The only source schema
migration is the explicit admission object in
`hcli/hawking-native.sealed-3.14.json`; integrating the commit applies it.
Before activation, rollback means omitting the commits. After integration,
revert them in reverse order; no artifact or receipt rewrite is required. The
main Flash owner controls clean-prefix installation, protected source/route/
state/capability gates, GPU parity and performance checks, packaging, restart,
canary, final acceptance and rollback.
