# Phase 0: Source and Legal Intake Ledger

Edition: 2026-07-19 · Program: HIDE BEST-OF-ALL Implementation Bible, Book I (RIP) + Book XVI §47.
Branch: `build/hide-impl-2026-07-19` (isolated worktree). Gate: every intake path legally and technically classified before any donor code enters HIDE.

## 1. Intake path classes (RIP §1.3)

- **A. Direct licensed port** - compatible OSS license; pin commit, preserve notices, record modified files, adapt to HIDE contracts, never copy branding.
- **B. Adapted reimplementation** - study behavior + source, write HIDE-native design, cite inspiration + license.
- **C. Clean-room behavioral recreation** - proprietary products; behavioral contract only, no proprietary source/strings/assets/prompts.

No donor code enters the tree except through A or B with this ledger updated. Proprietary donors (Cursor, Claude Code, hosted Codex, GPT-5.6 UI) are class C only.

## 2. Source ledger (pinned)

| Donor | Repo | Pinned commit (Bible §77) | License (stated) | Confirm-at-intake | Class | First mechanisms to intake |
|---|---|---|---|---|---|---|
| OpenAI Codex | `openai/codex` | `aa982319c2642918182e88bdace1ef9fea9ebc4b` | Apache-2.0 | yes | A/B | app-server thread/turn/item protocol, event stream, approvals-as-events, schema generation |
| Grok Build | `xai-org/grok-build` | `ba76b0a683fa52e4e60685017b85905451be17bc` | Apache-2.0 (+3rd-party notices) | yes | A/B | Rust checkpoint store, rewind/replay, hunk tracking, resumable subagents, ACP framing, headless entry, sandbox plumbing |
| OpenCode | `anomalyco/opencode` | pin exact commit at intake (not yet pinned) | MIT | yes | A/B | provider registry, client/server split, generated SDK, agent profiles, event hooks, LSP-as-evidence |
| Gemini CLI | `google-gemini/gemini-cli` | pin at intake | Apache-2.0 | yes | B | multimodal ingestion, trusted-folder enforcement, checkpoint UX, structured streaming, release channels |
| Zed ACP | ACP spec + reference impls | pin at intake | Apache-2.0 | yes | A/B | ACP server framing, editor-agent boundary |
| Aider | `Aider-AI/aider` | pin at intake | Apache-2.0 (CONFIRM) | REQUIRED | B | graph-ranked RepoMap, architect/editor split, edit-format metadata |
| Cline | `cline/cline` | pin at intake | Apache-2.0 (CONFIRM) | REQUIRED | B/C | human-in-loop diff review, checkpoints, browser evidence |
| Roo Code | `RooCodeInc/Roo-Code` | pin at intake | Apache-2.0 (CONFIRM) | REQUIRED | B/C | mode-specific permissions, staged approvals |

Rule: a donor with an unconfirmed license is class **C only** until its `LICENSE` file is read at the pinned commit and recorded here. No class-A/B intake proceeds on a "commonly known" license.

## 3. Clean-room-only donors (no source intake, ever)

| Donor | Study surface | Boundary |
|---|---|---|
| Cursor | changelog, docs, observed UX (side chats, transcript search, Design Mode, context report, tabs) | behavioral contract only; no proprietary source/strings/assets/prompts |
| Claude Code | official docs, the prior `hide_claude_parity_2026_07_19/` package | behavioral parity already specified clean-room; reuse that package, not Anthropic source |
| Hosted Codex / GPT-5.6 UI | docs, announcements | behavioral only |
| GitHub Copilot | docs | behavioral only |

The prior research package (`docs/plans/hide_claude_parity_2026_07_19/`, branch `research/hide-claude-parity-2026-07-19`) is HIDE's own clean-room work product and is a first-class input for all class-C behavior.

## 4. License matrix and notice plan

- **Apache-2.0 donors** (Codex, Grok Build, Gemini CLI, Zed ACP; Aider/Cline/Roo pending confirm): permit modification + redistribution with (a) retained copyright + license + NOTICE, (b) statement of changes, (c) no use of donor trademarks. HIDE ships a `docs/source-ledger/THIRD_PARTY_NOTICES.md` aggregating each ported file's origin commit + license + change summary.
- **MIT donors** (OpenCode): permit reuse with retained copyright + permission notice.
- **Apache/MIT compatibility**: both are permissive and compatible with HIDE's own license; no copyleft in the donor set. GPL/AGPL donors: none selected. If a transitively-pulled dependency is copyleft, it is flagged and excluded from static linking.
- **Third-party notice plan**: every class-A ported file carries a header block: `SPDX-License-Identifier`, `origin: <repo>@<commit>:<path>`, `changes: <summary>`. A CI check (Phase 2) fails the build if a file under `crates/hide-*` imports donor-derived code without a provenance header.

## 5. No-copy list (hard)

Never copy from any donor: product/brand names and logos; UI copy beyond short functional labels; hidden/system prompts; proprietary model weights or eval sets; marketing assets; trademarked spinner/progress vocabulary; Cursor/Claude Code/Codex visual assets. HIDE uses its own names, voice, and the Ando/Geist doctrine (`docs/hide-bible/DESIGN_DOCTRINE.md`).

## 6. Component ownership (who owns the mechanism in HIDE)

| Mechanism | HIDE owner crate | Primary donor inspiration | Intake class |
|---|---|---|---|
| Agent server protocol (thread/turn/item, events, init) | `hide-protocol` + `hide-agent-server` (new) | Codex app-server | B (reimpl from schema) |
| Checkpoint/rewind/replay, hunk provenance | `hide-state` + `hide-edit` | Grok Build | A/B |
| Provider registry, client/server, SDK gen | `hide-agent-server` + `hide-sdk` | OpenCode | B |
| ACP server | `hide-acp` (new) | Zed ACP | A/B |
| RepoMap + architect/editor | `hide-context` (reconciles packed `hawking-context`/`hawking-index`) | Aider | B |
| Steerable autonomy, migration readers | `hide-kernel`, `hide-extension-registry` | Claude Code (clean-room) | C |
| Programmatic tool runtime | `hide-program-runtime` (new) | GPT-5.6 programmatic tools (clean-room) | C |

Reconciliation: HIDE already has packed crates that own several of these (`hide-kernel`, `hawking-context`, `hawking-index`, `hide-tools`, `hawking-orch`). The Bible's proposed 28-crate map (§71) is the target; Phase 1 reuses the packed crates as the starting implementation and renames/splits toward the target map only where it reduces duplicate responsibility. See `PHASE0_CRATE_MAP.md`.

## 7. Provenance schema (per ported file)

```yaml
# hide.provenance.v1  (header block or sidecar .prov.yaml)
origin: "xai-org/grok-build@ba76b0a6:crates/.../checkpoint.rs"
license: "Apache-2.0"
intake_class: "A"          # A direct port | B adapted | C clean-room
ported_by: "phase-<n>"
date: "2026-07-19"
changes: "adapted to hide-core effect types; removed xAI auth; replaced logging"
tests: ["hide-state::checkpoint_roundtrip", "hide-state::rewind_parity"]
no_copy_checked: true       # brand/prompt/asset scan passed
```

## 8. Donor test plan (RIP Prove)

For each intaken mechanism, port or write a behavioral test that pins the contract, not the lines: e.g. checkpoint roundtrip byte-identity, rewind restores code+conversation, RepoMap ranking is deterministic under a fixed budget, thread fork preserves ancestry. A mechanism is not "intaken" until its behavioral test is green in HIDE.

## 9. Gate status

- Intake classes defined: DONE.
- Donor commits pinned where the Bible pinned them (Codex, Grok Build); others pin-at-intake: RECORDED.
- License matrix + notice plan: DONE; Aider/Cline/Roo licenses flagged CONFIRM-REQUIRED before class-A/B intake.
- No-copy list: DONE.
- Provenance schema: DONE.
- Clean-room boundaries: DONE (prior parity package is the class-C source).

Phase 0 gate is met for the crates rehydrated in Phase 1 (they are HIDE's own prior work at `5a99d0e2`, not donor code, so no external license applies). External donor intake (Codex/Grok/OpenCode/etc.) begins in Phase 2+ and each intake updates this ledger before code lands.
