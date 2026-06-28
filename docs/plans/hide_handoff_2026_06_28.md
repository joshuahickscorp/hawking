# HIDE handoff prompt (paste into the dedicated HIDE chat) — 2026-06-28

> Generated from the condensation/Studio chat. The condensation track (the `.tq` artifacts HIDE
> serves) is now fully scaffolded on the M3 Pro and otherwise Studio-gated. This hands the HIDE
> (front-end + serving product) work to its dedicated chat so the two tracks run in parallel and
> consolidate later. Everything below is paste-able as the opening prompt.

---

```
You are working on HIDE — the product surface of the Hawking project: the local-first inference
app + serving layer that runs Hawking's condensed `.tq` models on Apple Silicon. The repo is at
~/Downloads/hawking. Read these canonical docs FIRST and treat them as the contract:
  docs/hide-bible/MASTER_PLAN.md        (the unified roadmap — design + capability are one idea)
  docs/hide-bible/HIDE_PLAN.md          (the consolidated plan)
  docs/hide-bible/SCAFFOLD_STATUS.md    (what is REAL vs a seam, what is tested)
  docs/hide-bible/frontend/             (the front-end document set; the Design Doctrine lives here)

LOCKED CONTEXT — do not reopen:
- HIDE is the front-end + serving lane. The QUALITY/condensation science (the bit-floor curve,
  the doctor recovery stack, 14B/32B floors) lives in a SEPARATE chat and runs on a Mac Studio
  (M2 Max, 96 GB). Do NOT do condensation experiments here. HIDE's job is to SERVE what that
  track produces and to be the product around it.
- The bridge between the two tracks is NATIVE `.tq` SERVING (docs: native_tq_serving_impl.md):
  Stage A = HAWKING_QWEN_TQ=<path.tq> dequant-on-load to f16 and serve (correctness; tps loses
  on a fitting model, expected); Stage B = WeightKind::Tq native strand_bitslice.metal decode →
  GEMV (the real low-bit footprint win). Residual two-part serve (base bitslice + residual
  bitslice summed on-the-fly) serves the ~1:1 quality recipes. These are HIDE-track work.
- The four moats to build AROUND, not re-derive: (1) RWKV state — pass state not text;
  (2) economics — free local fleets / fork-and-try-N; (3) the `.tq` format itself;
  (4) logits access. The design ASSUMES these (fork&try-N, free fleets, no-jank tools).

HOUSE RULES (non-negotiable, enforced in all copy + UI):
- NO em-dashes, NO en-dashes, NO middot (·) in any user-facing copy. Use plain punctuation.
- Fonts: Geist = LOGO ONLY; Geist-Mono = UI; Cormorant = display.
- Palette: #060606 base, gold #F0B95B accent. NO blue, NO purple.
- Material brutalism, not flat. WCAG AA minimum.
- Aesthetic north star: "observatory / gold-rim / legible" — a box that radiates.

CURRENT STATE (from SCAFFOLD_STATUS.md, verify before trusting):
- All 11 hide-*/hawking-* crates are REAL and tested (~410 tests); the kernel loop is audited
  genuine; the system runs headless. The live-model load and the Tauri/desktop shell are SEAMS
  (stubbed boundaries), not yet wired to a real condensed model end to end.
- A localhost HTTP/WS transport over BackendHost exists (recent commits on serve/rwkv-multiseq-fix).
- The UI transport is decoupled from Tauri via a localhost HTTP/WS adapter (design doctrine doc).

WHAT TO DO (propose a plan, then execute against the docs — do not skip the read):
1. Confirm the scaffold: build + run the existing crate tests; confirm the headless serve path
   and the localhost HTTP/WS transport are green. Report what is real vs seam, precisely.
2. Native `.tq` serve Stage A (correctness): wire HAWKING_QWEN_TQ load so HIDE serves a real
   condensed artifact end to end (dequant-on-load to f16 first). Prove coherent generation
   through the actual product surface, not just a unit test. (A 7B `.tq` from the condensation
   track is the test input; if none is staged yet, use any existing baked artifact in scratch/.)
3. Close the live-model + Tauri seams against that Stage-A path so the desktop shell drives a
   real model.
4. THEN Stage B (native bitslice decode → GEMV) + residual two-part serve — the footprint/tps
   win. This is the headline serve number; gate it with parity vs Stage A (bit-identical decode).
5. Build the product moats into the surface: fork-and-try-N (free local fleets), state pass
   (RWKV), logits access, no-jank tools — per MASTER_PLAN.md.

DISCIPLINE:
- Respect the house rules in every pixel and every string.
- Parity-gate every serve-path change (Stage B must match Stage A bit-for-bit on a fixed prompt
  before it is trusted; re-run the parity check yourself, do not trust a green CI alone).
- Keep the two tracks clean: if you need a condensed artifact that does not exist yet, note it as
  a dependency on the condensation/Studio track rather than running condensation here.
- Git attribution: do NOT add any Claude/AI co-author or "Generated with" trailers to commits or
  PRs (the owner's standing rule).

Begin by reading docs/hide-bible/MASTER_PLAN.md and SCAFFOLD_STATUS.md, then give me a plan for
step 1 (confirm scaffold) before touching code.
```

---

## Why this split (for the consolidation later)

- **Condensation/Studio track** (the other chat): produces the `.tq` artifacts + the bit-floor
  science. Studio-gated. M3-Pro scaffolding is done (receipts harness, env pin, baselines).
- **HIDE track** (this handoff): the product + serving surface that runs those artifacts. Most of
  it is independent of the Studio — Stage A serve, seam-closing, the product moats can all proceed
  on any machine NOW.
- **Consolidation point:** when the Studio produces a real condensed 14B/32B `.tq` AND HIDE has
  Stage B native serve, the two meet at the RAM-cliff tps bench (§5 of studio_maximization) — the
  headline "condensed model runs faster because it fits" demo. That is the moment to merge.
