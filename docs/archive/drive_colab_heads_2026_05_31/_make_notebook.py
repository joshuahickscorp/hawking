#!/usr/bin/env python3
"""Generator for consolidate_drive_dismantle.ipynb (one-shot Drive cleanup notebook).
Run: python3 _make_notebook.py  -> writes consolidate_drive_dismantle.ipynb next to it."""
import json, os

cells = []
def md(src):   cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})

md(r'''# Consolidate the dismantle Drive folders into one downloadable archive

One-shot ops notebook (2026-05-31). It mounts your Google Drive, bundles the
dismantle content into a **single `.tar`** written back to Drive, and prints a
`sha256` so you can verify the download. After you download + verify, delete the
source folders **and** the archive — the whole dismantle Drive presence goes.

### Pick a scope in the CONFIG cell
- **`keepers`** (default, ~13 GB): `dismantle_final_push/best_heads` +
  `dismantle_final_push/frozen_cache` + the unique `dsv2` head. Everything else
  on Drive is a duplicate older export or raw sweep intermediates that re-run
  from Colab — not worth downloading.
- **`everything`** (~75 GB): all three folders. Needs ~75 GB **free Drive quota**
  for the archive plus a ~75 GB download. Only for a full cold backup.

Run the cells top to bottom.''')

code(r'''# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')''')

code(r'''# 2. Locate the dismantle folders (handles My Drive vs Shared drive)
import glob
hits = sorted(glob.glob('/content/drive/MyDrive/dismantle*')
              + glob.glob('/content/drive/Shareddrives/*/dismantle*'))
print("dismantle folders visible after mount:")
for c in hits:
    print("   ", c)
if not hits:
    print("   (none found — open the Files pane on the left and check where they live)")
print("\nIf they are NOT directly under /content/drive/MyDrive, set ROOT in the "
      "next cell to their parent path shown above.")''')

code(r'''# 3. CONFIG  --  edit these, then run
import os, shlex, subprocess

SCOPE        = "keepers"     # "keepers" (~13 GB) or "everything" (~75 GB)
INCLUDE_DSV2 = True          # dsv2 head is the only copy anywhere (~1.7 GB). False to drop it.
ROOT         = "/content/drive/MyDrive"                                   # parent of the dismantle* folders
OUT          = "/content/drive/MyDrive/dismantle_consolidated_2026_05_31.tar"

DISMANTLE     = f"{ROOT}/dismantle"
FINAL_PUSH    = f"{ROOT}/dismantle_final_push"
HEADBANK_CORR = f"{ROOT}/dismantle_headbank_corrected"

if SCOPE == "keepers":
    items = [f"{FINAL_PUSH}/best_heads", f"{FINAL_PUSH}/frozen_cache"]
    if INCLUDE_DSV2:
        items.append(f"{DISMANTLE}/dismantle_export/headbank_500u_v2/dsv2")
elif SCOPE == "everything":
    items = [DISMANTLE, FINAL_PUSH, HEADBANK_CORR]
else:
    raise ValueError("SCOPE must be 'keepers' or 'everything'")

print(f"SCOPE = {SCOPE}")
print(f"archive -> {OUT}\n")
missing = [p for p in items if not os.path.isdir(p)]
for p in items:
    print(("  OK   " if os.path.isdir(p) else "  MISSING ->"), p)
if missing:
    raise SystemExit("\nSome source paths are missing. Fix ROOT (see cell 2) and re-run.")''')

code(r'''# 4. Show what will be archived + total size (and the Drive-quota cost of the archive)
total = 0
print("Per-item sizes:")
for p in items:
    b = int(subprocess.run(["du", "-sb", p], capture_output=True, text=True).stdout.split("\t")[0])
    total += b
    print(f"  {b/1e9:7.2f} GB   {p}")
print(f"\nTOTAL to archive: {total/1e9:.2f} GB")
print(f"NOTE: writing the .tar adds ~{total/1e9:.0f} GB to your Drive usage until you delete it.")
print("      If you are short on Drive quota, set OUT='/content/dismantle_consolidated_2026_05_31.tar'")
print("      (Colab local disk) instead and download from the Files pane.")''')

code(r'''# 5. Build the single archive.
# No compression: safetensors/npz are ~incompressible, so plain tar is fastest and
# predictable. (To compress anyway, use -czvf and name OUT '...tar.gz'.)
rel = [os.path.relpath(p, ROOT) for p in items]
relpaths = " ".join(shlex.quote(r) for r in rel)
cmd = f"tar -cvf {shlex.quote(OUT)} -C {shlex.quote(ROOT)} {relpaths}"
print("Running:", cmd, "\n")
get_ipython().system(cmd)
print("\nArchive build complete.")''')

code(r'''# 6. Final size + checksum  (compare sha256 after you download)
get_ipython().system(f"ls -lh {shlex.quote(OUT)}")
print("\nsha256 (verify with `shasum -a 256 <file>` on your Mac):")
get_ipython().system(f"sha256sum {shlex.quote(OUT)}")''')

md(r'''## After this finishes

1. **Download the one file** from Drive: find `dismantle_consolidated_2026_05_31.tar`
   in My Drive → right-click → **Download**. Single file, so no zip-in-parts.
2. **Verify** on your Mac (must match the sha256 printed above):
   ```
   shasum -a 256 ~/Downloads/dismantle_consolidated_2026_05_31.tar
   ```
   Then extract wherever you want, e.g.:
   ```
   mkdir -p ~/Downloads/dismantle/checkpoints/drive_archive && tar -xvf ~/Downloads/dismantle_consolidated_2026_05_31.tar -C ~/Downloads/dismantle/checkpoints/drive_archive
   ```
3. **Delete everything from Drive** (it all goes): trash `dismantle`,
   `dismantle_final_push`, `dismantle_headbank_corrected`, **and** the
   `dismantle_consolidated_2026_05_31.tar` archive — then empty Trash to actually
   reclaim the space (Drive holds trashed items 30 days otherwise).''')

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "consolidate_drive_dismantle.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
