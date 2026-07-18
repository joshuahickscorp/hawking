# HIDE — Ship Status (2026-06-29)

A long-form account of where HIDE stands on the road to shipping. Companion to the living gap table in
`hide_ship_readiness.md`; this doc is the narrative snapshot at the moment the app became a signed,
notarized, distributable macOS build.

---

## 0. The one-line state

**HIDE is a signed, notarized, self-contained macOS app today.** Everything that makes it an *app* is
done and verified. The only thing standing between "runs" and "runs and generates" is the model
(#8 — the Hawking weights / `hawking serve`), which is a separate workstream.

A user can take `HIDE_0.1.0_aarch64.dmg` (18 MB), open it, drag HIDE to Applications, and launch it
with no Gatekeeper warning. It opens, the editor/terminal/files all work, and it honestly reports
"model offline" instead of faking tokens until weights are wired.

---

## 1. What HIDE is (so "shipping" has a target)

HIDE (Hawking IDE) is a local-first agentic coding environment: a React/Vite front end (Monaco editor,
xterm terminal, Zustand state) over a Rust backend (`hide-*` crates) that talks to a local model
runtime (`hawking-*` crates). The product thesis is four ownership moats the cloud tools can't price:

- **M1 — pass *state*, not text.** RWKV-7's constant ~6-16 MB recurrent state is serializable; forking a
  session is a memcpy, not a re-prompt.
- **M2 — own the economics.** Zero marginal cost locally, so "try N attempts in parallel" is free.
- **M3 — own the `.tq` format.** The quantized weight format is ours; it buys effective context far
  past the nominal window.
- **M4 — own the logits.** Grammar-guaranteed tool calls.

The surface design is deliberate: Tadao-Ando concrete (grayscale `--void`, light-only accent, Geist
Mono), Liquid-Glass chrome, the "Radiate" event-horizon progress signature, and no budget/context
meter. The doctrine phrase that governs scope creep: *supercharged Rolls-Royce — maximal capability
under the hood, minimal luxurious surface* (not an economy car, not a Bugatti).

---

## 2. The ship critical path — DONE

Four things had to be real (not mocked) for HIDE to be a usable app rather than a styled prototype.
All four are wired and verified:

1. **Real workspace file I/O.** `FsConnector` (in `hide-backend`) does path-confined `tree`/`read_file`/
   `write_file`; the Explorer and Editor call it over the connector bus, with a mock fallback. Writes
   are confined to the workspace root (parent-dir/abs-path escapes rejected).
2. **Real integrated terminal.** `run_command_streamed` spawns a real process in the workspace-confined
   cwd and streams stdout/stderr back as `ToolProgress` events that xterm renders. A real runner, not a
   transcript.
3. **Tauri desktop shell — now functional, not just compiling.** The app bundles the engine as a
   sidecar: `HIDE.app/Contents/MacOS/` contains *both* `hide-desktop` (the shell) and `hide-serve` (the
   engine). `externalBin` in `tauri.conf.json` plus `spawn_engine()` in `main.rs` (which resolves the
   engine next to the app binary, with PATH/`HIDE_SERVE_BIN` fallbacks for dev) means launching the app
   starts its engine — no PATH dependency, no "reconnecting" dead-end. Transparent window + macOS
   vibrancy via `window-vibrancy`.
4. **Live token streaming + executor loop.** `generate_submit_turn` emits `TokenBatch` events; the store
   coalesces them into the assistant message. The plumbing is complete and unit-tested; it produces real
   tokens the moment a model is online.

---

## 3. Distribution — the milestone reached today

This is the part that turned "I built a `.app`" into "I can hand this to anyone."

- **Signed.** `Developer ID Application: Joshua-Hicks Kilongozi (B5R65FT2U3)`, full chain to Apple Root
  CA, with a secure timestamp. The embedded `hide-serve` sidecar is signed as part of the bundle.
- **Verified.** `codesign --verify --strict --deep` reports *valid on disk* and *satisfies its
  Designated Requirement*.
- **Notarized.** `spctl -a -t exec` reports *accepted, source = Notarized Developer ID*.
- **Stapled.** The notarization ticket is attached to both the `.dmg` and the `.app`, so Gatekeeper
  clears it even with no network on first launch.

The chain that produces this is a single guided script, `app/scripts/build-macos.sh`, rewritten to be
**interactive and Xcode-free**:

- It checks for the Command Line Tools (`codesign`/`notarytool`), not full Xcode.
- For the certificate it walks the **web** flow: a Keychain-Access certificate request, uploaded at
  `developer.apple.com` as a Developer ID Application cert, downloaded and installed.
- It then **auto-detects** the signing identity and Team ID from the keychain (`security find-identity`),
  so the only values typed are the Apple ID email and an app-specific password (entered hidden).
- It stages the sidecar (`stage-sidecar.sh`), builds, signs, notarizes, and staples — secrets live only
  inside that one run, nothing is written to disk.

This is repeatable for every future release.

---

## 4. Hardening and quality — done this pass

- **Security gate.** `dangerous_command()` blocks genuinely destructive/system commands (`sudo`,
  `rm -rf /` or `~`, `mkfs`, `dd of=/dev`, `curl|sh`, fork bombs) with a `SecurityGate` event before the
  runner spawns; ordinary dev commands (`cargo test`, `git push`, `rm -rf node_modules`) pass. File
  writes were already workspace-confined.
- **Front-end test suite.** Vitest wired (`pnpm test`), with store-reducer tests (session tracking,
  token coalescing, context-manifest folding). Tests are excluded from the production `tsc` so the build
  stays clean; node-environment for now, with room to add component/DOM tests.
- **State persistence.** Sidebar/chat/panel/open-file/tab UI state persists to localStorage across
  launches.
- **Error boundary.** A class-component boundary wraps the app with a recovery panel instead of a white
  screen on render failure.
- **Settings surface.** A real Settings modal (Model / Engine / Keyboard / About), opened from the gear.

---

## 5. Brand / logo — settled

The canonical mark is the user's `hide-mark.png`: a solid ball (the black hole) and a Geist-Black `h`
on a 45-degree diagonal. It was serialized to vector (`logo/hide-mark.svg` plus `LogoH`/`LogoMark`
React components built from the real glyph outline), with the ball/`h`/gap proportions matched to the
PNG. In the app the wordmark is just the `h`. The app icon was regenerated from the faithful mark onto a
dark concrete squircle (`logo/hide-icon-1024.png`) and baked into the `.icns`.

---

## 6. What's verified (the receipts)

- **Rust:** ~1,488 `#[test]` functions across ~355 files in 17 crates; `hide-backend` at 58 tests green
  including the new security-gate classifier.
- **Front end:** `pnpm typecheck` + `pnpm build` green; `pnpm test` (Vitest) green; 35 TS/TSX source files.
- **Desktop:** `tauri build` produces `HIDE.app` + `HIDE_0.1.0_aarch64.dmg`; `codesign`, `spctl`, and
  `stapler validate` all pass on both artifacts.

---

## 7. What remains

### Blocking full product value (separate workstream)
- **#8 — the model.** Wire the Hawking `.tq` weights / `hawking serve` so the app generates. Until then
  the app is fully usable as a shell but reports "model offline." This is the Studio/Hawking arc, not the
  app shell.

### Recommended polish (not blocking distribution)
- **Accessibility:** focus traps on the Settings/gate dialogs, ARIA on the file tree and diff hunks,
  finish the reduced-motion paths.
- **Onboarding:** a no-folder first-run state (today it assumes a workspace is open).
- **Auto-update:** the Tauri updater plus a release feed, so shipped builds can self-update (needs a host
  for the feed; signing is already in place).
- **Approve-to-run gate:** the security gate currently blocks destructive commands outright; a full
  approve-and-run round-trip (user approves, then it runs) is a later iteration.
- **Node version:** the dev machine is on Node 20.17; Vite prefers 20.19+/22.12+ (it builds today with a
  warning) — bump before it becomes a hard requirement.

---

## 8. Honest caveats

- "Self-contained" means the app + the `hide-serve` engine ship together; it does **not** mean a model
  ships inside. Weights are the #8 workstream.
- The front end's live features are real wiring on real intents; anywhere the backend bridge isn't ready
  uses clearly-labeled mock/optimistic state, never faked results.
- Throughput/context claims (the `.tq` multiplier, RWKV long-context advantage) are surfaced as labeled
  estimates from measured metadata, not as guarantees.

---

## 9. Bottom line

The app is shippable now. If the goal is "a thing people can install and trust," that exists today as a
signed, notarized `.dmg`. If the goal is "a thing that cods for you," that's gated on #8 — and the entire
surface, distribution chain, and agentic plumbing are already in place to light up the instant the model
is online.
