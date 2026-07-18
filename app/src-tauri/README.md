# HIDE desktop shell (Tauri v2) — build-ready scaffold

This wraps the HIDE front end in a native desktop window and supervises the local `hide-serve` engine.
It is a **scaffold**: it was authored offline and not yet compiled, because the Tauri crates and the
`@tauri-apps` npm packages need a network fetch first.

## What it does
- Native macOS **vibrancy** behind a transparent window (`window-vibrancy`), so the OS desktop and the
  apps behind HIDE show through the glass. The web glass (rim + grain) layers on top.
- **Supervises `hide-serve`**: spawns it on launch (loopback :8744), kills it on quit.

## To build (needs network once)
```
cd app
pnpm add -D @tauri-apps/cli@^2
pnpm add @tauri-apps/api@^2
pnpm exec tauri dev      # or: pnpm exec tauri build
```
Add to `app/package.json` scripts: `"tauri": "tauri"`. (Left out of the committed package.json so a
frozen-lockfile install cannot fail before the deps are fetched.)

## Still needed before this ships
- **Icons**: drop `icons/icon.icns` + `icons/icon.png` (use `pnpm exec tauri icon path/to/logo.png`).
- **`hide-serve` as a sidecar**: bundle the built binary as a Tauri `externalBin` (or set
  `HIDE_SERVE_BIN`) so the app finds the engine in production, not just on PATH in dev.
- **Code signing + notarization**: an Apple Developer ID cert (external resource) + `tauri build`
  signing config + `xcrun notarytool`. Cannot be done without the cert.
- **Auto-update**: the Tauri updater plugin + a signed release feed.

## Engine note
On macOS the Tauri webview is **WKWebView (WebKit-class)**, so `shell/glass.ts` resolves
`data-glass="frost"` and the refraction filter does not engage. The rim + frost + grain path is what
ships on macOS. Verify the glass on WKWebView before launch.
