# Hawking IDE (HIDE) — Documentation

HIDE is the second product in the Hawking family: a local-first agentic coding IDE that serves Hawking `.tq` models. This directory has two parts now — the **built backend** (reference) and the **front-end bible** (the active, forward-looking spec).

> **Where things stand:** the entire HIDE **backend is built and tested** (11 Rust crates, the agent loop is real). The work now is the **front end** — a Tauri 2 + React/Monaco shell that talks to that backend over a small, already-existing contract. The bible has been restructured to match: the old 13-chapter backend design spec is archived; what's active is the front-end document set.

---

## → Start here: the front-end bible

The active spec for building the UI skeleton lives in **[`frontend/`](frontend/)**:

| Doc | What it covers |
|---|---|
| [00 · Vision & the Backend Contract](frontend/00-vision-and-backend-contract.md) | The three surfaces, the Tauri/React/Monaco/xterm stack, and the **real** FE↔backend contract — the `Intent`/`UiEvent` wires, the `BackendHost`/`CommandRouter`/`UiEventBus` API, the connectors, the TS IPC client, and the Custom-name registry. |
| [01 · The Surfaces](frontend/01-surfaces.md) | The **AI IDE** (Monaco editor / diff review / file tree / terminal), the **AI Chat** (streaming turns), the **AI Workstation** (agent-run timeline / fleetview), and the **Context Stack** rail — component trees, store slices, intents sent, UiEvents consumed, and the binding state machines. |
| [02 · OSS Harvest Map](frontend/02-oss-harvest.md) | What front-end component/UX to rip from each open-source AI IDE — Cline, Void, OpenHands, Continue, Aider, Goose, Kilo, OpenCode (port) and Zed/Cursor/Copilot (study-only) — with licenses and target modules. |
| [03 · Build Sequencing](frontend/03-build-sequencing.md) | The skeleton-first plan, on top of the done backend: scaffold the Tauri app + the typed IPC client, then the panels in priority order (chat → editor → diff → tree → terminal → context stack → timeline → workstation), with milestones M-FE0…M-FE3. |

Everything in `frontend/` binds to the contract that **actually exists** in code (`crates/hide-core/src/api.rs`, `crates/hide-backend/`), not to a sketch — where the old design diverged from what was built, the docs follow what was built.

---

## The backend (built — reference, not active work)

The backend is implemented across `crates/hide-core`, `hide-kernel`, `hide-tools`, `hide-security`, `hide-fleet`, `hide-personalize`, `hide-backend`, `hawking-context`, `hawking-index`, `hawking-orch`, `hawking-research`. The code is the source of truth; these two docs index it:

- **[SCAFFOLD_STATUS.md](SCAFFOLD_STATUS.md)** — the completion record: per-crate real-vs-seam state, the deferred seams (live model wiring, the Tauri frontend, the moonshots), and a "when you return" checklist. **Read this for what the backend can do today.**
- **[SCAFFOLD_AUDIT.md](SCAFFOLD_AUDIT.md)** — the original gap audit (the "before" picture; historical).

The one deferred seam the front-end work fills: the **Tauri host layer**. `CommandRouter::handle(Intent)` is transport-agnostic and the `UiEventBus` already streams `UiEvent`s — the front end wraps these behind `#[tauri::command]` and an `ipc::Channel<UiEvent>` (see [frontend/03 §1](frontend/03-build-sequencing.md)).

---

## Archive

The original 13-chapter bible (the backend design spec, now implemented) is preserved under **[`archive/`](archive/)** — `00-vision-and-constitution` through `13-roadmap-and-build-sequencing`. It's design rationale; the implementation is the code + `SCAFFOLD_STATUS.md`. Kept for reference; not part of the active spec.

---

## License policy (for the front-end harvest)

Only **MIT / Apache-2.0** code may be ported into shipped `app/`. Zed (AGPL) and Cursor/Copilot (proprietary) are **study-only — never copy code**. Maintain `THIRD_PARTY_NOTICES.md`; the CI license gate fails on anything else. Full map in [frontend/02](frontend/02-oss-harvest.md).
