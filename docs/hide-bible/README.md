# Hawking IDE (HIDE) — Documentation

HIDE is the second product in the Hawking family: a local-first agentic coding IDE that serves Hawking `.tq` models, free and fully on-device. The design and the capability are one idea, "the box that radiates": privacy outward (nothing leaves your machine), transparency inward (the Context Stack).

## The single source of truth

**[HIDE_PLAN.md](HIDE_PLAN.md)** is the one authoritative document. It folds the strategy, the unified roadmap, the full design doctrine, and the product + contract reference into one place:

- **Part A: Strategy** — the spine, where we stand, the wedge ("fork and try 5, keep the best, free, watch all five radiate"), the proof demo, the risk register, the improvements.
- **Part B: The Roadmap** — the M0 to M8 interlock (backend capability married to front-end surface), the critical path, the design-to-capability synergy map.
- **Part C: The Design Doctrine** — SUPERSEDED. The canonical design system is now **[DESIGN_DOCTRINE.md](DESIGN_DOCTRINE.md)** (v3: a Tadao Ando grayscale-concrete and light-as-accent system; the v2 gold rim-light is retired). Read that for tokens, type, the three surfaces, and the Self-check ship gate.
- **Part D: Product and Contract Reference** — the surfaces, the `Intent`/`UiEvent` + `hide-serve` contract, the OSS harvest map.

Everything binds to the contract that actually exists in code (`crates/hide-core/src/api.rs`, `crates/hide-backend`, `crates/hide-serve`).

## What is built

- **Backend:** 11 crates, the agent loop real. See **[SCAFFOLD_STATUS.md](SCAFFOLD_STATUS.md)** (the per-crate reality) and **[SCAFFOLD_AUDIT.md](SCAFFOLD_AUDIT.md)** (the original gap audit, historical).
- **Transport:** `crates/hide-serve` (localhost HTTP/WS over `BackendHost`).
- **Front end:** `app/` (Vite + React + TS), built to Design Doctrine v3 (Ando concrete + light): the design system (`theme.css`), the `Volume`/`LightEdge` primitives, `wire.ts`/`ipc.ts`, the Zustand stores, the Shell, and the three surfaces plus the Context Stack. It runs alive on a mock transport; set `VITE_HIDE_TRANSPORT=live` to bind `hide-serve`.

## Archive

**[archive/](archive/)** holds superseded design material kept for provenance: the original 13-chapter backend bible (now implemented as code), the 5-doc front-end bible (now folded into `HIDE_PLAN.md`), and the standalone `MASTER_PLAN.md`. Reference only; the live spec is `HIDE_PLAN.md` and the code.
