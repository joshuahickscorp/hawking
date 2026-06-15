# Making `.sa` seamless on every desktop

STRAND archives (`.sa`) open on double-click on macOS, Linux, and Windows. Only
macOS needs a helper app; the other two are pure per-user configuration written
by the CLI itself. No platform needs admin rights.

| Platform | What handles the double-click | One-time setup |
|---|---|---|
| **macOS** | `STRAND.app` (headless, notarized) embeds the CLI | install the app, then `strand register` |
| **Linux** | the `strand` CLI via a hidden `.desktop` entry | `strand register` |
| **Windows** | `strand-open.exe` (windowless shim) → `strand.exe` | build the shim, then `strand register` |

`strand register` is the common front door — it writes the association for
whatever OS it runs on, and supports `--dry-run` to preview every change.

## macOS

The app is a background agent: no Dock icon, no menu bar (`LSUIElement`). A
double-click extracts the archive next to itself and reveals it in Finder;
nothing else is shown unless extraction fails.

```sh
# From a release download:
#   unzip STRAND-notarized.zip -d /Applications
# Or build + sign from a checkout:
cd packaging/macos
./setup-signing-keychain.sh         # loads the Developer ID (one-time per boot)
./make-app.sh                       # build + sign  (retry if the Apple
                                    #   timestamp server flakes — see below)
ditto build/STRAND.app /Applications/STRAND.app
strand register                     # re-point LaunchServices at the app
```

Notarization (for distributing to other Macs) is a stored one-liner once the
keychain profile exists — see [macos/README.md](macos/README.md). The signing
keychain is ephemeral and dies on reboot; re-run `setup-signing-keychain.sh`.
If `make-app.sh` reports "a timestamp was expected but was not found", that's a
transient Apple timestamp-server hiccup — just run it again.

## Linux

```sh
strand register          # writes ~/.local/share/mime + applications entries
```

This installs a shared-mime-info type (matched by both the `*.sa` glob **and**
the `STRA1` container magic, so renamed archives still resolve) and a
`NoDisplay` desktop entry that runs `strand unpack` on the file. Associations
apply immediately where `update-mime-database`/`xdg-mime` are present, otherwise
after the next login. No app, no daemon.

## Windows

Double-clicking the bare CLI would flash a console window, so a tiny
GUI-subsystem shim launches the extraction invisibly.

```powershell
# Build the windowless shim (needs rustc; no external crates):
.\packaging\windows\make-shim.ps1
# Put strand-open.exe next to strand.exe, then:
strand register          # writes HKCU\Software\Classes keys
```

`strand register` automatically points `.sa` at `strand-open.exe` when it finds
it beside `strand.exe`, and falls back to associating the CLI directly (with the
console flash) otherwise.

## Security note

Extraction is hardened against path traversal ("zip-slip"): every archived
path is validated to be relative with no `..` or absolute/root/drive component
before anything is written, and the destination join is re-checked. A malicious
`.sa` cannot write outside the chosen output directory. This is enforced in the
`strand-archive` crate (`unpack`/`unpack_into`), so it protects the CLI and the
double-click openers alike.
