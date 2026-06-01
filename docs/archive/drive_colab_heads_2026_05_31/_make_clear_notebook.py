#!/usr/bin/env python3
"""Generator for clear_drive_dismantle_and_root.ipynb (guarded Drive cleanup notebook).
Run: python3 _make_clear_notebook.py  -> writes clear_drive_dismantle_and_root.ipynb next to it."""
import json, os

cells = []
def md(src):   cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})

md(r'''# Clear the dismantle Drive footprint (folders + loose root checkpoints)

**Destructive.** This trashes ~115 GB from your Google Drive:
- the 3 folders `dismantle`, `dismantle_final_push`, `dismantle_headbank_corrected` (~74.7 GB)
- the 113 loose Eagle-5 training files sitting in My Drive **root** (~40.75 GB):
  `step_*.npz`, `latest.npz`, `head_final.safetensors`, `log.jsonl`, `tau.json`

**Preserved:** every personal/academic doc in root, and the folders TradingModels,
demucs, gravel, apartments, BHM_highlight_videos, Colab Notebooks. Also preserved:
your `dismantle_consolidated_*.tar` (delete that one yourself after verifying the download).

**Recoverable:** items are moved to Drive **Trash** (~30 days), not hard-deleted.
Empty Trash afterward to actually reclaim quota.

**Run this AFTER** you've run `consolidate_drive_dismantle.ipynb`, downloaded the tar, and
verified its sha256. Then: run cells top to bottom, eyeball the dry-run lists, set
`CONFIRM = True` in the config cell, re-run the config cell and the final cell.''')

code(r'''# 1. Authenticate to the Drive API (trash by ID = instant + recoverable, no 75 GB FUSE copy).
# If you hit a permission error here, run a cell with `from google.colab import drive; drive.mount('/content/drive')`
# first to grant Drive access, then re-run this cell.
from google.colab import auth
auth.authenticate_user()
from googleapiclient.discovery import build
from collections import Counter
service = build('drive', 'v3')
print("Drive API ready.")''')

code(r'''# 2. CONFIG + gather targets (preview only — this cell deletes nothing)
CONFIRM = False   # leave False to preview. Set True ONLY after your tar is downloaded + verified.

DISMANTLE_FOLDERS = {
    "dismantle":                    "1kZsJcBXYnvENZsaHrl5Xux7nSeo1H904",
    "dismantle_final_push":         "1O9ACoTgcj6pMvxf63ey0BHynfoE2Ya2C",
    "dismantle_headbank_corrected": "1Plni8zxdQ6XxWei-pN5TDBgssQzj3pei",
}
FOLDER_SIZE_GB = {"dismantle": 18.0, "dismantle_final_push": 48.7, "dismantle_headbank_corrected": 8.0}

def is_target_file(name):
    if name in ("latest.npz", "head_final.safetensors", "log.jsonl", "tau.json"):
        return True
    return name.startswith("step_") and name.endswith(".npz")

# find loose training files DIRECTLY in My Drive root (not inside any folder)
loose, page = [], None
while True:
    resp = service.files().list(
        q="'root' in parents and trashed = false",
        fields="nextPageToken, files(id,name,size,mimeType)",
        pageSize=1000, pageToken=page).execute()
    for f in resp.get("files", []):
        if f.get("mimeType") != "application/vnd.google-apps.folder" and is_target_file(f["name"]):
            loose.append(f)
    page = resp.get("nextPageToken")
    if not page:
        break

loose_bytes = sum(int(f.get("size", 0)) for f in loose)
kinds = Counter("step_*.npz" if f["name"].startswith("step_") else f["name"] for f in loose)

print(f"Loose root training files matched: {len(loose)}  ({loose_bytes/1e9:.2f} GB)")
for k, c in sorted(kinds.items()):
    print(f"   {c:4d}  {k}")
print("\nDismantle folders to trash:")
folder_gb = 0.0
for name in DISMANTLE_FOLDERS:
    gb = FOLDER_SIZE_GB.get(name, 0.0); folder_gb += gb
    print(f"   ~{gb:5.1f} GB  {name}")
print(f"\nGRAND TOTAL to clear: ~{folder_gb + loose_bytes/1e9:.1f} GB  "
      f"({len(loose)} loose files + {len(DISMANTLE_FOLDERS)} folders)")
print("PRESERVED: all personal docs in root + folders TradingModels/demucs/gravel/")
print("           apartments/BHM_highlight_videos/Colab Notebooks + your *.tar backup.")
print(f"\nCONFIRM = {CONFIRM}", "-> WILL DELETE when you run the final cell."
      if CONFIRM else "-> preview only; set CONFIRM=True (then re-run this cell) to enable deletion.")''')

code(r'''# 3. (optional) list every loose file that will be trashed — eyeball before confirming
for f in sorted(loose, key=lambda x: x["name"]):
    print(f"   {int(f.get('size',0))/1e6:8.1f} MB  {f['name']}")''')

code(r'''# 4. Trash everything (guarded by CONFIRM)
if not CONFIRM:
    print("CONFIRM is False - nothing deleted. Review the lists above, set CONFIRM=True in cell 2, "
          "re-run cell 2, then re-run this cell.")
else:
    for name, fid in DISMANTLE_FOLDERS.items():
        service.files().update(fileId=fid, body={"trashed": True}).execute()
        print("trashed folder:", name)
    for f in loose:
        service.files().update(fileId=f["id"], body={"trashed": True}).execute()
    print(f"trashed {len(loose)} loose root files")
    print(f"\nDone - {len(DISMANTLE_FOLDERS)} folders + {len(loose)} files moved to Trash (recoverable ~30 days).")
    print("Reclaim quota: drive.google.com -> Trash -> Empty trash.")
    print("Also delete your dismantle_consolidated_*.tar once your local copy is verified.")''')

md(r'''## Order of operations recap
1. `consolidate_drive_dismantle.ipynb` -> makes the keeper tar in Drive.
2. Download that tar, verify its sha256, extract locally.
3. **This notebook**: preview -> `CONFIRM = True` -> run final cell -> trashes the 3 folders + 113 root checkpoints.
4. Delete the `*.tar` and **empty Drive Trash** to reclaim the full ~115 GB.''')

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

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clear_drive_dismantle_and_root.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
