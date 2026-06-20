# STRAND.app — macOS packaging for `.sa` archives

Makes `.sa` a first-class macOS file type: double-clicking a STRAND archive
extracts it next to itself and reveals the result in Finder.

| File | Purpose |
|---|---|
| `StrandOpener.swift` | Tiny AppKit shell (no storyboard). Receives `.sa` documents from LaunchServices, shells out to the bundled `strand` CLI (`unpack <file> -o <dir>`), reveals the output dir, quits. Launched bare → one-line hint alert, quits. |
| `Info.plist` | Bundle metadata. Declares + **exports** the UTI `com.strand.archive` (conforms to `public.data`, `public.archive`; tag `.sa`) and claims it with role Viewer / rank **Owner**. |
| `make-app.sh` | Reproducible assembly: compile opener → lay out `STRAND.app` → embed CLI → sign inner-first → verify. |
| `setup-signing-keychain.sh` | Stands up `/tmp/strand-signing.keychain` with the Developer ID identity (see "Signing identity" below). |
| `build/` | Output (gitignore-worthy). |

No app icon yet — a future nicety is `Resources/STRAND.icns` + `CFBundleIconFile`.

## Pipeline

```sh
# 0. One-time per boot: signing identity into a dedicated keychain
./setup-signing-keychain.sh          # → "Developer ID Application: Joshua-Hicks Kilongozi (B5R65FT2U3)"

# 1. Build + sign (auto-picks the best identity; -c overrides CLI path)
./make-app.sh -c /tmp/strand-release-bin/strand

# 2. Notarize (DONE 2026-06-10: submission bcd0366e-1274-4a9d-bbaf-36d743d46725 Accepted, stapled)
xcrun notarytool store-credentials strand-notary \
    --apple-id joshuahicksboba@gmail.com --team-id B5R65FT2U3 \
    --password xxxx-xxxx-xxxx-xxxx          # app-specific password, one-time
ditto -c -k --keepParent build/STRAND.app /tmp/STRAND.zip
xcrun notarytool submit /tmp/STRAND.zip --keychain-profile strand-notary --wait --timeout 15m
# if Invalid: xcrun notarytool log <submission-id> --keychain-profile strand-notary

# 3. Staple (after Accepted)
xcrun stapler staple build/STRAND.app
spctl -a -t exec -vv build/STRAND.app        # now: "accepted, source=Notarized Developer ID"

# 4. Install + register
ditto build/STRAND.app /Applications/STRAND.app
/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister -f /Applications/STRAND.app

# 5. Smoke test
strand pack a.txt b.txt -o /tmp/demo.sa && open /tmp/demo.sa
# → /tmp/demo/ appears and is revealed in Finder
```

## Signing identity

The Developer ID Application certificate + private key live in the
**tailor** repo's gitignored secrets folder (single source of truth on this
machine, also held by GitHub Actions as encrypted secrets):

```
/Users/scammermike/Downloads/tailor/deploy/secrets/tailor-signing.key        # RSA-2048, unencrypted PEM
/Users/scammermike/Downloads/tailor/deploy/secrets/developerID_application.cer
```

`setup-signing-keychain.sh` imports both (plus Apple's Developer ID G2
intermediate, fetched from apple.com) into `/tmp/strand-signing.keychain`
with a throwaway password, pre-authorizes codesign via
`set-key-partition-list`, and prepends the keychain to the user search list.
This mirrors tailor's own CI recipe and avoids login-keychain password
prompts entirely.

**Ephemeral by design:** `/tmp` clears on reboot, taking the keychain (and a
then-dangling search-list entry) with it. Re-run the script to restore;
`security delete-keychain /tmp/strand-signing.keychain` to clean up early.

## Credentials needed and where they're configured

| Credential | Needed for | Status on this machine |
|---|---|---|
| Developer ID Application cert + key | codesign | **Present** — tailor/deploy/secrets (cert `B5R65FT2U3`, valid 2026-05-24 → 2031-05-25) |
| Developer ID G2 intermediate CA | chain building | Fetched to /tmp by setup script (not preinstalled) |
| Apple ID (joshuahicksboba@gmail.com) | notarytool | Known |
| Team ID (`B5R65FT2U3`) | notarytool | Known (read from cert subject) |
| **App-specific password** | notarytool | **MISSING locally.** Exists only as the GitHub Actions secret `APPLE_APP_PASSWORD` in `joshuahickscorp/tailor.ai` (write-only, not recoverable). Mint a fresh one: appleid.apple.com → Sign-In and Security → App-Specific Passwords, then run the `store-credentials` line above. An App Store Connect API key (`AuthKey_*.p8`) would also work (`notarytool --key/--key-id/--issuer`); none found on disk. |

## Status as of 2026-06-10 (this machine)

| Step | Status |
|---|---|
| Bundle build | ✅ `build/STRAND.app` (arm64, opener + embedded CLI) |
| Codesign | ✅ `Developer ID Application: Joshua-Hicks Kilongozi (B5R65FT2U3)`, hardened runtime, secure timestamp; `codesign --verify --deep --strict` passes |
| spctl | ⚠️ `rejected — source=Unnotarized Developer ID` (expected pre-notarization) |
| Notarize | ❌ Blocked: no app-specific password on this machine (see table above) |
| Staple | ❌ N/A until notarized |
| Install | ✅ `/Applications/STRAND.app` |
| lsregister | ✅ UTI `com.strand.archive` active+exported, claim rank Owner |
| Live test | ✅ `open demo.sa` → extracted (byte-identical) + revealed in ~4 s; re-open uniquifies to `demo 2` |

**Gatekeeper note:** locally built apps carry no quarantine attribute, so the
unnotarized app opens fine on THIS Mac. Any copy that travels (download,
AirDrop, etc.) gets quarantined and WILL be blocked until notarized+stapled —
finish steps 2–3 before distributing.
