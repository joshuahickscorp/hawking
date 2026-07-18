# HIDE — Auto-update (release wiring)

The code scaffold for auto-update is in the tree and tested (`app/src/shell/updater.ts` +
`updater.test.ts`, and the "check for updates" button in Settings → About). What remains is **infra
that needs the network and a signing key**, captured here so it is a paste-and-run when the desktop
build host is online. None of this needs the Mac Studio.

## 1. Add the updater plugin (needs network)

```sh
# Rust side (in app/src-tauri):
cargo add tauri-plugin-updater
# JS side (in app/):
pnpm add @tauri-apps/plugin-updater
```

Register it in `app/src-tauri/src/main.rs`:

```rust
tauri::Builder::default()
    .plugin(tauri_plugin_updater::Builder::new().build())
    // ...existing setup
```

`withGlobalTauri` is already `true` in `tauri.conf.json`, so once the plugin is registered the
runtime check in `updater.ts` (`window.__TAURI__.updater.check`) lights up — no FE change needed.

## 2. Generate the update signing keypair (one time)

```sh
pnpm tauri signer generate -w ~/.hide/hide-updater.key
# prints a PUBLIC key; keep the PRIVATE key + its password out of the repo (use CI secrets).
```

This is a **separate** minisign keypair from the Apple Developer ID codesign identity — it signs the
update *artifacts*, not the app bundle.

## 3. Add the updater config to `tauri.conf.json`

```jsonc
{
  "bundle": { "createUpdaterArtifacts": true },
  "plugins": {
    "updater": {
      "pubkey": "<PASTE THE PUBLIC KEY FROM STEP 2>",
      "endpoints": ["https://dl.hide.dev/updates/{{target}}-{{arch}}/latest.json"],
      "windows": { "installMode": "passive" }
    }
  }
}
```

(Kept out of the live `tauri.conf.json` until step 1 lands, so the current offline `tauri build`
doesn't require the not-yet-fetched crate.)

## 4. The release feed (`latest.json`)

`buildUpdateManifest` in `updater.ts` produces exactly this shape (and rejects a malformed one). An
example is in `docs/plans/hide_update_feed.example.json`. At release time:

```sh
# tauri build emits HIDE.app.tar.gz + HIDE.app.tar.gz.sig (when createUpdaterArtifacts is on).
# Build latest.json with the signature + the artifact URL, then upload both to the feed host:
#   https://dl.hide.dev/updates/darwin-aarch64/latest.json
#   https://dl.hide.dev/updates/.../HIDE_<ver>_aarch64.app.tar.gz
```

The host is any static file server (S3/R2/Pages). The feed URL in step 3 must match the upload paths.

## 5. CI hook

Extend `app/scripts/build-macos.sh` (or a CI job) to, after sign+notarize+staple: set
`createUpdaterArtifacts`, sign the artifact with the step-2 key (`TAURI_SIGNING_PRIVATE_KEY` +
`..._PASSWORD` env), generate `latest.json`, and publish to the feed host.

## Status

- **Done + tested (code):** `buildUpdateManifest` feed builder, runtime-detected `isUpdaterAvailable`/
  `checkForUpdate`, Settings "check for updates" affordance, `withGlobalTauri` enabled.
- **Pending (infra, network/key/host — no Studio):** steps 1-5 above.
