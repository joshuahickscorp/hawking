# Hawking Tabula Rasa Shipping Plan

Status: proposed execution plan. No destructive cleanup is authorized by this
document.

## Objective

Turn the research workspace into a clean, reproducible shipping repository
without disturbing the detached Doctor V5 campaign, losing experiment evidence,
or weakening any scientific contract. Reduce accidental complexity and duplicate
implementations; do not optimize for line-count reduction by itself.

## Observed baseline (2026-07-13)

- Git history already attributes commits only to Joshua Hicks, under two email/name
  spellings. The configured author is `Joshua Hicks <joshuahicksboba@gmail.com>`.
- The working directory is roughly 336 GB, while tracked source is under 20 MB.
  The dominant local trees are `scratch/` (~309 GB), `target/` (~12 GB),
  `models/` (~9.5 GB), `.claude/` (~2.4 GB), and `reports/` (~2.0 GB).
- The repository has about 1,168 tracked files and roughly 324,000 tracked
  implementation lines. Runtime state, downloaded models,
  build products, evidence, and product source need clearer ownership boundaries.
- Exact duplicate and near-duplicate utility implementations exist, including two
  `strand_eval` trees and repeated frontier evidence/signing helpers.
- The current branch has 35 modified tracked paths and 46 nonignored untracked
  paths. The untracked campaign implementation is therefore part of the freeze
  problem, not disposable scratch.
- The product boundary is not yet explicit: the Rust workspace, excluded STRAND
  crates, HIDE application, and Python Studio/Doctor system currently share one
  repository.

These figures are a planning snapshot, not release claims. Re-measure them at the
start and end of every phase.

## Non-negotiable invariants

1. Freeze and verify the detached campaign before touching any path it reads or
   writes.
2. Preserve receipts, manifests, hashes, checkpoints, logs, and report ledgers.
3. Never delete original model sources through campaign garbage collection.
4. Keep third-party copyright, licenses, and required notices. Removing unwanted
   AI-tool attribution must not erase legitimate dependency attribution or product
   compatibility references.
5. No Git history rewrite, bulk file deletion, or active-data relocation without a
   separately reviewed manifest and explicit approval.
6. Every consolidation must be behavior-preserving first: characterize, migrate,
   verify, then remove the superseded path.
7. The detached campaign worktree is immutable. Shipping work uses a separate Git
   worktree, branch, and build directory; no cleanup process may resolve campaign
   code or binaries through shared paths.

## Phase -1 — Define the release boundary

Decide which surfaces ship together and which remain research or separately
versioned packages: the Hawking runtime/CLI, Studio and Condense tooling, HIDE, and
STRAND. Record supported entry points, compatibility promises, and ownership for
each surface before judging a file redundant.

Exit gate: every top-level package, workspace member, excluded crate, application,
and Python tool family has an explicit ship, separate-package, research-only, or
archive disposition.

## Phase 0 — Freeze the research launch

- Review the dirty tree and capture it as deliberately scoped campaign-baseline
  commits; do not make one opaque bulk commit. Record the baseline commit and a
  content manifest/tag before cleanup starts.
- Record the exact campaign plan hash, adapter hashes, binary hashes, model source
  manifests, report schema, PID/launch method, resource gates, and resume command.
- Save a read-only inventory of the dirty worktree and runtime-owned paths.
- Prove detach/resume from a checkpoint and prove that automatic reporting survives
  a worker restart.
- Mark runtime-owned paths with an explicit allowlist. Cleanup work must treat that
  allowlist as read-only until the campaign is paused and sealed.
- Create a separate cleanup worktree/branch with its own `CARGO_TARGET_DIR`, Python
  environment, logs, and generated outputs. Prove that no campaign subprocess can
  import, execute, or overwrite a path in that worktree.

Exit gate: the campaign can survive terminal/app closure and host restart without
manual reconstruction, its first checkpoint/report synchronization is valid, and
cleanup builds cannot change any source or binary hash used by the campaign.

## Phase 0.5 — Characterize before refactoring

Create small golden fixtures for CLI behavior, canonical JSON and hashing, schemas,
receipts, reports, checkpoint/resume, GC crash recovery, numerical output, and a
tiny end-to-end condensation. Add fault-injection tests for partial writes,
interrupted deletion, stale PIDs, source drift, and memory-pressure stops.

Exit gate: every intended consolidation seam has a pre-existing behavior oracle;
refactors do not invent their own acceptance criteria after the fact.

## Phase 1 — Separate product source from experimental mass

Define one external, configurable data root, for example `HAWKING_DATA_ROOT`, with
a typed path-ownership contract for immutable source, durable evidence, model
sources, derived artifacts, checkpoints, logs, caches, build outputs, user
configuration, and ephemeral scratch. Product code refers to logical artifact IDs
plus manifests, not repository-relative accidental paths.

- Keep source, schemas, small fixtures, curated result summaries, and reproducible
  manifests in Git.
- Keep downloaded weights, build outputs, transient decoded weights, raw logs, and
  bulk experiment payloads outside Git.
- Replace ad-hoc path discovery with one path/config module shared by CLIs and
  workers.
- Add a dry-run migration command that emits a content-addressed move manifest,
  checks free-space and open-file safety, and supports rollback. Execute it only
  after campaign paths are sealed.
- Tighten `.gitignore` and add size/leak checks in CI.
- Do not move the live 309 GB campaign tree during this phase. First create an
  immutable external evidence bundle and a small, tracked, hash-bound summary;
  perform a restore drill before deleting any source model or campaign payload.

Exit gate: a clean clone can point at the same external data root and reproduce a
selected receipt without copying the 300+ GB workspace into the repository.

## Phase 2 — Normalize ownership and public provenance

- Add `.mailmap` to normalize the two existing Joshua Hicks identities for reports
  and release tooling.
- Audit `CONTRIBUTING.md`, package metadata, generated report templates, comments,
  branch names recorded in docs, and credits. Public project authorship should name
  Joshua Hicks only unless a legally required third-party notice applies.
- Keep technical references such as OpenAI-compatible APIs or Claude-format inputs
  when they describe actual compatibility; they are not contributor attribution.
- Produce a reviewable findings file before removing anything.
- Classify every Codex, Claude/Anthropic, and OpenAI mention as contributor claim,
  session chatter, technical compatibility, citation, or legally required
  attribution. Remove only the first two categories where appropriate.
- Generate third-party notices and an SBOM for Cargo, npm, Python, copied code, and
  vendored sources. Audit source headers before removing provenance.
- Reject personal absolute paths and signing-key locations in shipping source,
  fixtures, and dependency locks unless an explicitly sanitized archive owns them.

Exit gate: `git shortlog` and release metadata show the intended identity, license
and SBOM scans remain clean, no personal path leaks, and no technical compatibility
claim or required attribution was silently removed.

## Phase 3 — Consolidate the documentation surface

Create a small canonical set:

- `README.md`: product promise, verified status, minimal quick start.
- `ARCHITECTURE.md`: stable component and data-flow contracts.
- `docs/condensation.md`: format, Doctor, quality, and evidence semantics.
- `docs/studio-campaign.md`: queue lifecycle, checkpoints, reports, and recovery.
- `docs/reproducibility.md`: exact commands, environments, hashes, and claims.
- `docs/operations.md`: storage, memory, thermal, and incident procedures.

Move historical run reports and superseded plans into a dated archive with an
index rather than leaving them as competing root-level instructions. Generate a
link graph and fail CI on broken internal links. Assign every document one explicit
disposition: canonical, generated, archived, or deleted after redirect. Preserve
`LICENSE`, `CONTRIBUTING`, and `CHANGELOG`; add `SECURITY.md` and third-party notices.

Exit gate: every current command and claim has exactly one canonical owner; archived
documents are clearly non-normative.

## Phase 4 — Consolidate code around contracts

Work in small, bisectable changes with characterization tests:

1. Make one canonical `strand_eval` package and migrate both duplicate trees,
   preserving copied-package and self-location behavior with characterization tests.
2. Extract shared frontier selection, evidence, signing, hashing, and receipt
   utilities; remove copies only after byte-for-byte fixture comparison.
3. Establish one campaign state machine and one adapter ABI. Older launchers become
   thin compatibility shims or are archived after their receipts can still be read.
4. Split oversized modules by ownership boundary—planning, admission, execution,
   evidence, and reporting—not by arbitrary file length.
5. Centralize model/rate/treatment registries so a cell is declared once and
   projected into plans, reports, and runtimes.
6. Remove unreachable flags and dead implementations only with coverage evidence.

Maintain a duplication ledger for deliberate generated assets, compatibility
copies, and fixtures; each exception needs an owner, reason, compatibility test,
and expiry condition.

Exit gate: no unexplained exact duplicate implementation remains, every CLI has one
documented entry point, and existing golden receipts/reports verify unchanged.

## Phase 5 — Build a release-shaped repository

- Define supported macOS/Apple Silicon and optional CUDA feature matrices.
- Make builds hermetic enough for a clean-clone release check: locked dependencies,
  deterministic generated schemas, small fixtures, and no hidden home-directory
  dependency.
- Add CI for formatting, linting, unit/integration tests, schema compatibility,
  receipt verification, source-size limits, secret scans, and license notices.
- Include the excluded STRAND crates and Python/Doctor tooling; fail when required
  tests are silently filtered or skipped. Add personal-path scanning, SBOM and
  dependency-license checks, packaging/notarization checks, and clean-user install.
- Package the CLI/runtime with explicit versioning and migration policy.
- Run a clean-user-account smoke test and reproduce one tiny end-to-end condensation
  plus report from only documented inputs.

Exit gate: a clean clone builds, tests, packages, and reproduces the reference
fixture without any private workspace state.

## Phase 6 — Optional history surgery

Only if explicitly approved after the shipping tree is stable: analyze whether old
large blobs or unwanted historical metadata justify a history rewrite. Prepare a
mirror backup, collaborator migration plan, old-to-new commit map, and remote
cutover procedure first. This phase is not required to ship a clean present-day
tree.

## Change sequence

Use one intentionally scoped commit series after the campaign launch is frozen:

1. release-scope decision;
2. reviewed campaign-baseline commits, immutable worktree, and reproducibility inventory;
3. separate cleanup worktree plus characterization fixtures;
4. external data-root/path-ownership contract and ignore rules;
5. authorship normalization, provenance, privacy, notices, and SBOM;
6. canonical documentation and archive index;
7. shared evidence/runtime utilities;
8. duplicate evaluator removal and duplication ledger;
9. release gates and clean-clone proof.

Each commit must state its measured before/after files, lines, disk footprint,
tests, and preserved compatibility surface. Do not mix mass data movement with code
consolidation.
