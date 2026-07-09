# HIDE refinement roadmap

Date: 2026-07-05
Branch: feat/hide-launcher
Source: 14-facet multi-agent audit (2 waves) + web research on premium grayscale/mono dev-tool UIs, over the retina screenshot set in reports/hide_ux_2026_07_05/ and the app/ + crates/hide-* source.

## TL;DR

The "vibe coded" feeling is real but it is NOT the doctrine. Grayscale + Geist Mono + Ando is the anti-slop position; the research is unanimous that a strict locked constraint set is exactly what separates crafted tools from AI-generated ones. What reads as unrefined is **execution drift**: the ambition of the doctrine leaks through median-quality details. Concretely, three design root causes and one packaging fact produce almost every complaint:

1. **No type token scale** (14 font sizes, fractional 11.5/12.5px, faux-bold 700 the loaded font cannot render). This is the loudest tell and it is worst exactly where you said: the landing greeting (34px/600 mono banner) over eight identical 22px "metrics" where `7 AM` and `qwen2.5-7b` are set like numerals.
2. **Cool surfaces under warm light + flat-black working views.** The 13-hex palette is disciplined, but the surface ramp drifts blue (rendered titlebar composites to `#18181b` = Tailwind zinc-900, the stock shadcn dark color) while every text token is warm; and every working panel collapses to pure `#070707`, so panels read as wireframe outlines on flat black (the "parking garage" failure mode of Ando).
3. **No shared gridline on the landing.** Hero, digest card, and composer sit on three different left edges; the heatmap leaves ~60% of the digest as dead void; the fleet cards are clipped mid-card at the fold.

And separately from the look: **the shipped app does not actually run.** The DMG ships the mock transport, and even if it did not, the Tauri shell launches the engine with a flag the engine rejects, so the backend exits at boot. There is also a drive-by RCE surface (unauthenticated localhost + CORS Any). These are graded below and gated first in the plan.

The bones are genuinely good (disciplined store, typed event stream, 300 real Rust tests, a coherent 13-color palette, a started motion system). This is a refinement job on a strong base, not a rebuild.

---

## Scorecard

Scores are current state, 0-10, where 10 is Linear/Vercel/Zed-grade. Bimodal facets are split per the completeness critic; the product facet is on a market-completeness scale, not a craft scale, so read it separately.

| Facet | Score | One-line verdict |
|---|---|---|
| **Design / look-and-feel** | | |
| Typography and hierarchy | **4.5** | Declared 6-role system the code ignores; landing is flat by construction. Loudest fix. |
| Color, surfaces and depth | **5.0** | Disciplined 13-hex palette, but cool-blue drift under warm text and flat-black panels kill the material story. |
| Layout landing chamber | **3.5** | No shared gridline, dead-void heatmap, fleet clipped at fold. The surface you dislike. |
| Layout IDE chamber | **7.0** | Genuinely disciplined, tight seams, shared grid. The surface you like. Leave it. |
| Motion and animation | **6.0** | Real system underneath (4 duration tokens, one easing, global reduced-motion), drift at the edges. |
| Copy and microcopy voice | **4.0** | Half doctrine, half drift; ships banned middots/em-dashes, 3-way casing, stale instructions. |
| **Frontend engineering** | | |
| CSS design-system rigor | **4.5** | ~400 lines dead/shadowed CSS, cascade by import-order, 4 parallel styling systems. |
| Frontend app architecture | **6.5** | Disciplined store + typed intents; mock seam leaks into UI, App.tsx is a second ad-hoc store. |
| Performance feel | **4.0** | 4.5MB single chunk (Monaco at chat boot), O(n^2) streaming render, fake rAF "governor". |
| Keyboard / focus system | **3.5** | Good primitives, vibe-coded layer: typing eats diff hunks, invisible focus ring, no menu. |
| First-run experience | **2.5** | Potemkin: mock persona, dead "Open folder", no model story, engine dead at boot. |
| **Backend engineering** | | |
| Backend architecture | **4.5** | Strong module craft, but drive-by RCE surface + a shipped sidecar that exits at boot. |
| Rust testing | **7.5** | 300 real tests with genuine edge cases, strict clippy CI. Actually good. |
| Ship-readiness | **2.0** | Mock ships, engine dead at boot, zero FE CI, errors vanish, updater is scaffold-only. |
| Privacy / egress verification | **5.0** | Behaviorally close to no-egress, but "verifiably" is unbuilt; runtime defaults to 0.0.0.0. |
| **Product** (market scale, read separately) | | |
| Differentiation and potential | **3.0 now / ~8.0 ceiling** | Pre-launch demo-ware today; defensible niche if the thesis gate returns GO. |

**How to read this:** the design facets converge on the 3 root causes above; fixing those lifts typography, color, landing-layout, and CSS-rigor together. The engineering facets converge on ~5 ship-blockers; fixing those lifts backend, ship-readiness, first-run, and product together. The audit found roughly 5 "high" design issues that collapse to 2 fixes, and the mock/sidecar root cause appears in 4 facets but is one afternoon of work.

---

## Root causes (deduped)

### Design: why it reads "vibe coded"

**RC-D1. There is no typography scale.** 14 distinct font sizes set as raw px, zero routed through a token, fractional 11.5/12.5px (a classic generated-CSS artifact), and two `font-weight:700` sites the loaded Geist Mono (400/500/600 only) fakes by smearing. A declared `.t-*` role system exists (theme.css:213-259) but is used 28 times against ~150 raw re-specifications. Evidence: `app/src/theme.css`, `app/src/styles/*.css`; landing at `app/src/styles/home.css:245`.

**RC-D2. The neutrals are the wrong temperature and the panels have no material.** The surface ramp (`#0e0e0f -> #222226`) and glass tints lean +1..+4 blue on the B channel while every text/light token is warm (`#f4f2ee/#9b9a95`); the rendered chrome composites to `#18181b` (Tailwind zinc-900). Working views (terminal, diff, executor, chat side panels) all render pure `#070707`, so volumes read as 1px outlines on flat black instead of lit masses. `--depth` is a black drop shadow on near-black = invisible by construction. ~40 ad-hoc alpha neutrals (two parallel white systems: `rgba(255,255,255,x)` AND `rgba(244,242,238,x)` at near-identical strengths) sit around the 5 real surface tokens. Evidence: `app/src/theme.css:16-19, 44, 58-59`.

**RC-D3. The landing page is not composed to a grid.** Three different left edges (digest 920px column, composer 760px, hero a third), a GitHub-contribution-heatmap clone that leaves the right ~60% of the card empty (and collapses to a single 11px column in 7d view), and fleet cards guillotined mid-card at the fold. This is why the IDE (one 760px chamber, shared grid) feels right and the landing does not. Evidence: `app/src/styles/home.css:183, 250-262, 332`.

Supporting: CSS carries ~400 lines of dead/shadowed rules and resolves cascade conflicts by import order (RC-D1/D2 amplifier); copy ships banned characters and 3-way casing (RC-D independent, cheap).

### Engineering: why it does not ship

**RC-E1. The shipped build runs the mock.** `ipc.ts:17` defaults transport to `"mock"`; the only `live` flag lives in `app/.env.local.bak`, which Vite never loads. The signed DMG is demo-ware replaying a fabricated run (your name, 222.9M fake tokens). **Verified.**

**RC-E2. The bundled engine exits at boot.** `app/src-tauri/src/main.rs:21` spawns `hide-serve --addr 127.0.0.1:8744`; `hide-serve` `parse_args` (`crates/hide-serve/src/main.rs:54`) bails "unknown flag --addr" (only `--port`/`-p`/positional/`HIDE_SERVE_ADDR` env accepted). `.spawn().ok()` returns `Some(child)` for an already-dead process, so nothing notices. Even fixed, no workspace root is passed (Finder cwd is `/`). **Verified.**

**RC-E3. Drive-by RCE surface.** `crates/hide-serve/src/lib.rs:64-66` sets `allow_origin(Any)/allow_methods(Any)/allow_headers(Any)` with no auth token on `POST /v1/hide/intent` (reaches `RunCommand`) or `/v1/hide/connector` (reaches fs read/write). Any website you visit can POST to `127.0.0.1:8744` and drive the agent. Separately, `crates/hawking-serve/src/lib.rs:381` defaults the model runtime to `0.0.0.0:8080` (all interfaces). **Verified.**

**RC-E4. The safety gate is theater.** Bypass mode is client-side (`App.tsx:96`), persisted to localStorage (survives restart), one button-cycle away with no confirm, and auto-approves everything including `sudo`/`rm -rf`. The classifier is a substring denylist trivially evaded (`bash -c "sudo ..."`, `/usr/bin/sudo`, `find / -delete`), and the approved exec path (`host.rs:1068`) is unsandboxed while the parallel tool path is fail-closed sandboxed.

**RC-E5. No frontend CI and errors vanish.** `.github/workflows/ci.yml` has no node step; 53 vitest specs are pure-logic (6% of the FE), zero component/DOM/E2E, the transport and wire contract untested. No `window.onerror`/`unhandledrejection`, sidecar stdout/stderr inherited into the void of a Finder-launched .app, no log file, updater is scaffold-only.

---

## Roadmap

Four phases. P0 makes it real, P1 kills the vibe-coded look (this is what you asked for), P2 makes it feel crafted in use, P3 makes it defensible. Impact/effort per task; S = hours, M = a day or two, L = a week+.

### P0 - Make the shipped app actually run and be safe (ship-blockers)

Nothing else matters if the DMG is demo-ware with an RCE. This is roughly one focused day.

- **[S] Fix the transport default.** `ipc.ts:17` -> `USE_MOCK = (VITE_HIDE_TRANSPORT ?? (import.meta.env.PROD ? "live" : "mock")) !== "live"`, and add `app/.env.production` with `VITE_HIDE_TRANSPORT=live`. Add a build tripwire: `grep -c MOCK_SESSION dist/assets/*.js` must be 0. (RC-E1)
- **[S] Fix the sidecar argv + pass a workspace root.** `main.rs:21` -> `.arg("--port").arg("8744")` (or set `HIDE_SERVE_ADDR` env), and pass the app-support data dir as the positional workspace instead of inheriting `/`. Add a `parse_args` unit test asserting the exact argv the shell sends, plus a smoke test that boots the built binary and GETs `/healthz`. (RC-E2)
- **[M] Authenticate + origin-pin the loopback plane.** Mint a per-boot bearer token, write it `0600`, require it in an axum middleware; replace `allow_origin(Any)` with the Tauri/Vite origins. Change `hawking-serve` default bind to `127.0.0.1:8080`; require an explicit `--allow-lan` to expose. (RC-E3)
- **[M] Gate Bypass server-side; stop persisting it.** Boot always in `ask`; bypass skips only routine prompts and still parks classified-dangerous commands; route the FE terminal through the sandboxed `shell.run` tool and delete the duplicate unsandboxed runner; tokenize the classifier (basename resolve, recurse into `sh -c`, expand `$HOME`). (RC-E4)
- **[M] Add a frontend CI job + global error handlers.** `pnpm typecheck && pnpm test && VITE_HIDE_TRANSPORT=live pnpm build` in CI; `window.onerror`/`unhandledrejection` -> the existing bounded notices UI; redirect sidecar stdout/stderr to `~/Library/Logs/HIDE/`. (RC-E5)

### P1 - Kill the vibe-coded look (the design refinement you asked for)

This is the answer to "make it look less vibe coded." Do it as three token PRs; each is mechanical once the token set is decided, and each is independently shippable.

**P1a. Typography token scale (RC-D1) - highest impact.**
- **[M] One tokenized type scale, delete every raw px.** Replace the `.t-*` block with tokens and sweep all five CSS files. Proposed scale (size / weight / tracking / line-height / use):
  - display 28px / 500 / -0.02em / 36px - landing greeting only, `clamp(24px,2vw,28px)`
  - title 16px / 500 / 0 / 24px - dialog + section titles
  - metric 24px / 500 / -0.01em / 28px - primary digest numerals (unit suffixes nested at 13px/400/--text-2)
  - body 14px / 400 / 0 / 22px - chat prose, `max-width: 68ch`
  - ui 13px / 400 / 0 / 20px - default controls, composer
  - small 12px / 400 / 0.01em / 16px - meta, chips, status bar
  - label 11px / 500 / 0.08em / 16px / UPPERCASE - the ONE label style
  - code 13px / 400 / 0 / 20px - all code (kills 12.5px)
  - Collapse: 10->11, 11.5->12, 12.5->13, 15->14, 17->16, 20->18(md h1), 26->24. Weights 400/500/600 only; fix the two 700s. Set `body { line-height: 20px }`.
- **[S] Rebuild the landing hero + digest.** Greeting drops to 28px/500/-0.02em with a terse eyebrow line above it (`SAT JUL 5  MAIN  QWEN2.5-7B READY`) so it is a lockup not a floating banner. Digest gets two tiers: 4 primary numerals at `metric`, and streaks/peak-hour/favorite-model move out of the numeral grid into one 12px flight-log footline; render unit letters (`d`, `AM`, `M`) as nested small spans, not full-size numerals. This is the specific fix for the surface you flagged.
- **[S] Cap prose at 68ch, fix the md heading ladder** (18/16/14 by weight, h4=h3).

**P1b. Color temperature + material (RC-D2) - highest-visibility "shades" fix.**
- **[S] Warm the entire neutral system to one temperature.** Re-tint every surface to the warm axis of `#F4F2EE` with perceptually even L* steps. Proposed ramp: void `#070707` (keep), concrete-1 `#0f0e0d`, concrete-2 `#161513`, concrete-3 `#1e1c1a`, concrete-4 `#262421`; glass tints warmed to match. Replace every pure-white `rgba(255,255,255,x)` hairline/glow with warm-white `rgba(244,242,238,x)`; collapse to two hairline strengths (0.07 resting, 0.13 hover). Fix the two AA-failing text tokens (`--mute` 4.42:1, `--text-3` on raised surfaces).
- **[M] Give working views material: child = parent + 1 step, never equal, never darker.** Panels get `concrete-1`, inner volumes `concrete-2`; only the editor and the chat transcript column stay on void (they are the chamber). Enforce with a comment rule: a surface may only sit on the token one step below it.
- **[S] Replace the invisible black shadow with a lit-edge recipe.** `--depth: 0 0 0 1px var(--line), inset 0 1px 0 rgba(244,242,238,0.05)` (hairline + top rim). Floating layers get a wide soft dark halo that reads because they float over lighter glass.
- **[S] Fix the heatmap** so empty cells are recessed tiles (one step below the card), not wireframe boxes; tokenize the light ramp.
- Research-backed technique to consider: per SRCL/Ghostty/Departure Mono, generate the ramp in OKLCH with fixed hue so steps are perceptually even; add a 2-4% `feTurbulence` grain layer on elevated surfaces only (reads as cast-concrete tooth, also kills near-black gradient banding). This is the "grayscale as material not missing color" move.

**P1c. Landing grid + CSS system hygiene (RC-D3 + supporting).**
- **[S] One column for the landing.** Add `--col-stage: 760px`; put hero text, digest, fleet, and composer on it so all four share two vertical gridlines. Share one horizontal-padding var between the conversation scroll and composer (kills the measured 40px offset when a side panel is open).
- **[M] Make the heatmap fill its container** (grid `repeat(weeks, 1fr)`, flush-right most-recent), and give 7d a different form (7 day-rows) instead of a lone column; rebalance the stack so fleet never clips (compress digest ~90px, add a scroll-mask fade).
- **[S] Legislate the 4px grid** (purge ~30 off-grid 3/5/6px values), **collapse 6 radii to 3**, and **delete ~400 lines of dead/shadowed CSS**; adopt `@layer tokens, base, components, surfaces` so the cascade is declared not accidental.
- **[M] Kill the JS styling system:** promote the 120 inline `style={{}}` objects and `parts.ts` atoms to classes; add an ESLint rule forbidding `fontSize/color/fontWeight` literals in `style=` so the fourth system cannot regrow.

**P1d. Copy sweep (cheap, high signal).**
- **[S] Mechanical character sweep + CI guard.** Fix the 11 shipped strings with middots/em-dashes/ellipses; add `voice.test.ts` asserting no `/[—–·…•]/` in string literals. One casing regime (sentence case controls, lowercase values); rewrite the fleet empty state that instructs a removed control; single-source the pitch sentence; ship a 10-line `docs/hide-bible/VOICE.md`.

### P2 - Make it feel crafted in use (interaction layer)

The look is static; "crafted" is felt in motion, keyboard, and latency. This is where the remaining gap between HIDE and Zed/Raycast lives.

- **[M] Keyboard: central keymap registry with input-target guards.** Fixes the real bug where typing `jar` into the composer accepts/rejects diff hunks (global single-letter handlers with no `e.target` guard). One Escape stack (LIFO, never destructive - today one Escape can reject a whole diff AND close the executor). Ship a real shortcut map + native macOS menu (no Cmd+N/W/,/menu today; Cmd+W kills the window). (Keyboard facet 3.5)
- **[S] Focus ring that is visible and doctrine-pure.** Today it is the same 11%-white hairline as resting borders (~1.4:1, fails WCAG 1.4.11). Replace with `0 0 0 1px var(--void), 0 0 0 2.5px var(--light)` - bone light on void is ~18:1 and reads as "light entering the volume", which is on-doctrine.
- **[M] Performance: split Monaco/xterm out of the boot path** (React.lazy; today one 4.5MB chunk parses the whole IDE before the chat greeting paints), **make streaming render incremental** (memoize settled messages, parse only the tail; today it re-parses the entire message through marked+DOMPurify every token batch = O(n^2)), and **coalesce token batches at the transport seam** (the rAF "governor" throttles nothing today). Cap `manifestRing`, move drag/resize off the render loop, keep the terminal alive across panel toggles. (Perf facet 4.0)
- **[S] Motion doctrine.** Write the 4-duration/1-easing spec into DESIGN_DOCTRINE.md and purge the 6 drift sites; give the six summoned surfaces (palette, gate, onboarding, settings, PiP, side panel) one shared entrance gesture instead of six improvisations; fix the reduced-motion hole (JS smooth-scroll bypasses the global kill-switch). (Motion facet 6.0)
- **[L] First-run that works.** Register the dialog plugin (the "Open folder" CTA is a silent no-op today), handle `open_folder`/`open_session` in the backend (both are dropped intents today), build a first-model step (detect/download/select - there is no model story at all, the composer's model button fires an unhandled intent), collapse the empty digest honestly instead of fabricating peak-hour/favorite-model at zero activity. (First-run facet 2.5)

### P3 - Make it defensible (product)

Everything here is contingent on the H3 thesis gate (can the local model actually code?), which memory says is unrun. Run it first.

- **[M] Run the thesis gate.** Stage-A F16-on-load for both RWKV-7 and Qwen `.tq`, point hawking-eval's coding bench at it, get GO/CONDITIONAL/KILL. No product score moves above 4 until one real local token streams through the real app.
- **[M] Verifiable no-egress as a checkable guarantee.** Ship `docs/hide-bible/NETWORK_LEDGER.md` enumerating every socket + an in-app Privacy panel; add an air-gap toggle; add a CI egress test that fails on any non-loopback packet; ship a Little Snitch profile. "The IDE your security team can tcpdump" is a sentence no competitor can say - but only if it is true and provable. (Egress facet 5.0)
- **[M] Ship local FIM tab completion.** The one table-stakes feature the whole field has and HIDE lacks - and the place local wins outright (sub-20ms first token from a resident `.tq` model beats any cloud round-trip). Cheapest daily proof the own-stack bet pays.
- **[M] Add an MCP client to hide-tools** (inherit the ecosystem instead of rebuilding it; keep the approval gate in front, default-deny network servers in air-gap mode). Make the Qwen `.tq` path first-class so a CONDITIONAL gate verdict is not fatal.
- **[L] The wedge demo.** Rebuild the pitch around what is structurally uncopyable now that parallel agents are commoditized (Cursor 3 runs 8, Claude Code has worktrees): `state.clone()` memcpy fork (5 branches, 0 re-prefill) + fully offline + unbounded local context. Film the "five futures, one read" demo. That demo is the go-to-market asset.

---

## Doctrine questions for the owner (do not implement without a ruling)

These are genuine forks the audit surfaced. The mono/grayscale doctrine stays; these are the edges where it strains.

1. **The hero line typeface. RULED 2026-07-05: build both, compare.** P1a ships the pure-mono hero (28px/500/-0.02em + eyebrow) as the default, AND produces a Geist-Sans-hero variant side by side before the owner rules which lands. The mono version ships either way; the Sans exception is evaluated against it, not in the abstract.
2. **Warm the void? RULED 2026-07-05: keep #070707 pure.** Surfaces above the void warm to one temperature (P1b ramp), but the base stays the doctrine value: "the unlit chamber has no temperature." Do not warm to #080706.
3. **Composed void vs occurring void.** (Open.) Ando voids aim light at something; today the void is residue (dead heatmap half, 500px hole in a fresh chat). Ruling needed: may content leave >30% of a card empty, and should a fresh chat vertically center the first exchange until content exceeds ~50vh (Claude Code does this)?
4. **Sanctioned hybrid egress mode.** (Open.) The doctrine bans a BYO-key escape hatch. Is verifiable-local-by-default with an explicit, packet-visible egress mode a betrayal of the doctrine or its strongest proof? Absolutism may cap the market to air-gap buyers; every local-first competitor treats hybrid as the pragmatic answer.

---

## Reference appendix - liftable systems

The research surfaced directly usable systems for a premium grayscale/mono UI. Full briefs in the audit output; the load-bearing ones:

- **Vercel Geist** (vercel.com/design.md) - role-encoded 10-step gray ramp (100 bg -> 1000 text), alpha-white border tokens (`#ffffff12`..`#ffffff3d`), 3 radii, forensic shadows (0.02-0.16 alpha), two-layer gap focus ring (`0 0 0 2px surface, 0 0 0 4px accent`). HIDE already uses Geist Mono; adopt the token structure.
- **SRCL / sacred.computer** (github.com/internet-development/www-sacred) - the most liftable: line-height as the universal vertical grid unit, ch-based indents, written lightness contracts ("window one step lighter than page", "shadow always darker than the body"), `oklch(from ...)` relative-color theming.
- **Ghostty** (ghostty.org) - terminal aesthetic that feels expensive: dual 10-step ramps, accent used only as 2-15% alpha gradient washes, scroll-edge fade masks instead of hard clips.
- **Departure Mono / US Graphics / Berkeley Mono** - grayscale as named material (`--enamel/--cement/--ash/--smoke/--carbon`), part-number nomenclature as ornament-through-utility, width+weight (not size) as the mono hierarchy toolkit.
- **Anthony Hobday's safe rules** - grayscale-ready numbers: dark-mode borders must be lighter than both container and background; inner radius = outer minus gap; one depth technique per interface; no drop shadows in dark UIs.
- **"Why your AI keeps building the same purple gradient" + AI-slop fix guides** - the tell list to encode as negative constraints in DESIGN_DOCTRINE.md: 1px gray border on every card, colored left-border strip (the single most reliable AI tell), rounded-2xl + shadow-lg, all-caps everywhere, 0.1-opacity shadows, any state left at browser default. Blocking defaults by name is what stops the median from leaking back in.

## Provenance

- 14 facet audits across 2 workflow waves (17 agents), completeness-critic pass, 2 web-research briefs.
- Screenshots: reports/hide_ux_2026_07_05/ (22 states, 2x retina).
- Ship-blocker claims (RC-E1/E2/E3) independently verified against source before this doc.
- Full audit JSON retained in the session scratchpad.
