#!/usr/bin/env python3
"""Builder for ``cleanup_drive.ipynb``.

Run from repo root:
    python3 colab/_build_cleanup_notebook.py

Emits a Colab notebook that reclaims Drive space for the dismantle headbank
runs. Dry-run by default — prints every candidate path with its size and
total reclaim, but does not delete until APPLY=True is flipped explicitly.

Safety rails baked in:
  * Skips anything modified within RECENT_WINDOW_HOURS (default 2h) so a
    live Colab can't lose state.
  * Never touches frozen.npz, the apex leaderboard head, the live lab root's
    fresh-export, or the q1p5_corpus shards.
  * Three modes: CONSERVATIVE / MODERATE / AGGRESSIVE, in increasing order
    of reclaim. Defaults to MODERATE.
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text))


def code(text: str) -> None:
    CELLS.append(("code", text))


md(
    """# dismantle Drive cleanup

Reclaims storage from the headbank / overengineer / 500U Colab runs.

**This notebook does nothing destructive until you set `APPLY = True`** in
Cell 2 and re-run. Default is dry-run: every candidate is printed with size
+ reason + total-reclaim summary at the bottom.

Three modes (set `MODE`):

* `'conservative'` — only orphans, duplicates, half-written `.tmp` uploads,
  known-regressed polish/push checkpoints. Zero risk to active work.
* `'moderate'` (default) — conservative + delete `latest.npz` files from
  non-winner base-sweep checkpoints (keeps `head_final.safetensors` so the
  heads remain inference-usable, just not warm-start-resumable). Bigger
  reclaim, no impact on inference.
* `'aggressive'` — moderate + delete `q3b_ref_corpus/` and `q0p5_corpus/`
  (the headbank notebook recaptures these). Biggest reclaim, costs ~10
  Colab CU when headbank reruns.

Live-protection: anything modified within the last `RECENT_WINDOW_HOURS`
(default 2h) is skipped no matter the mode — so the running overengineer
Colab can't lose state.
"""
)

code(
    """# Cell 1 - Mount Drive

from pathlib import Path
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as e:
    print(f'[setup] Drive mount skipped/failed: {e}')

DRIVE_ROOT = Path('/content/drive/MyDrive/dismantle')
assert DRIVE_ROOT.exists(), f'No dismantle folder at {DRIVE_ROOT}; mount may have failed'
print(f'[cleanup] mounted, root = {DRIVE_ROOT}')
"""
)

code(
    """# Cell 2 - Settings (edit before running Cell 3)

# DO NOT DELETE ANYTHING IF THIS IS FALSE.
APPLY = False

# Increasing reclaim:
#   conservative — orphans / regressed checkpoints / .tmp only
#   moderate     — also deletes redundant latest.npz files (keeps heads)
#   aggressive   — also deletes recapturable corpora (q3b_ref, q0p5)
MODE = 'moderate'

# Don't touch anything modified within this many hours. Protects the
# currently running overengineer/headbank Colab from losing state.
RECENT_WINDOW_HOURS = 2.0

# The leaderboard apex you must never touch. Anything matching this slug
# is excluded from every deletion rule.
APEX_TAG = 'q1p5_b1_fast_b1_h16_ff40_s16_lr3e-4_rd000_cw12_seed0'

print(f'[cleanup] APPLY={APPLY}  MODE={MODE}  RECENT_WINDOW_HOURS={RECENT_WINDOW_HOURS}')
print(f'[cleanup] APEX_TAG={APEX_TAG}')
if not APPLY:
    print('[cleanup] DRY RUN — no files will be touched. Flip APPLY=True to actually delete.')
"""
)

code(
    """# Cell 3 - Plan + (optionally) execute

import os
import shutil
import time
from pathlib import Path

NOW = time.time()
RECENT_CUTOFF = NOW - (RECENT_WINDOW_HOURS * 3600.0)

# Paths we touch. Anchored to DRIVE_ROOT so a renamed Drive can't surprise us.
LAB_ROOT   = DRIVE_ROOT / 'maximal_spec_500u'
EXPORT_NEW = DRIVE_ROOT / 'dismantle_export' / 'maximal_spec_500u'
EXPORT_OLD = DRIVE_ROOT / 'maximal_spec_500u_export'
ROOT_OLD_EXPORT_FOLDER = DRIVE_ROOT.parent / 'dismantle_export'         # NOT the one inside dismantle/
ROOT_OLD_EXPORT_ZIP    = DRIVE_ROOT.parent / 'dismantle_export.zip'

REGRESSED_CKPT_SLUGS = [
    'q1p5_best_r1_polish_d8_w006_p060_g095_lr5e-5',
    'q1p5_best_r2_push_d8_w010_p080_g092_lr4e-5',
]


def is_recent(path: Path) -> bool:
    try:
        return path.stat().st_mtime > RECENT_CUTOFF
    except FileNotFoundError:
        return False


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for p in path.rglob('*'):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


# Each entry: (mode_required, label, path, reason). When mode_required is
# 'conservative', it runs in every mode. 'moderate' runs in moderate +
# aggressive. 'aggressive' runs only in aggressive.
plan: list[tuple[str, str, Path, str]] = []

# ── CONSERVATIVE ──────────────────────────────────────────────────────────
plan.append(('conservative', 'root_old_export_zip', ROOT_OLD_EXPORT_ZIP,
             'orphaned zipped export from earliest run (May 26-27)'))
plan.append(('conservative', 'root_old_export_folder', ROOT_OLD_EXPORT_FOLDER,
             'orphaned folder export at MyDrive root (NOT inside dismantle/)'))
plan.append(('conservative', 'lab_old_export', EXPORT_OLD,
             'superseded by dismantle/dismantle_export/maximal_spec_500u/'))

# Any *.tmp file under the export trees (half-written copies).
for root in [EXPORT_NEW, EXPORT_OLD]:
    if root.exists():
        for tmp in root.rglob('*.tmp'):
            plan.append(('conservative', f'tmp:{tmp.name}', tmp,
                         'half-written upload (.tmp); the completed sibling exists alongside it'))
        # Also catch `.tmp.N` retry artifacts.
        for tmp in root.rglob('*.tmp.*'):
            plan.append(('conservative', f'tmp:{tmp.name}', tmp,
                         'incomplete copy retry artifact'))

# Regressed checkpoints (Cell 6 of the tau8 handoff broke the head; these
# have tau ~1.6 vs apex 7.99 and live below apex on the leaderboard).
ckpt_root = LAB_ROOT / 'artifacts' / 'q1p5' / 'checkpoints'
if ckpt_root.exists():
    for slug_substring in REGRESSED_CKPT_SLUGS:
        for cand in ckpt_root.glob(f'{slug_substring}*'):
            plan.append(('conservative', f'regressed:{cand.name}', cand,
                         'regressed checkpoint from polish/push runs (tau collapsed)'))

# ── MODERATE ──────────────────────────────────────────────────────────────
# Delete redundant latest.npz from every base-sweep checkpoint that is
# NOT the apex. Keeps head_final.safetensors so the head stays usable for
# inference / eval; only the warm-start resume state goes away.
if ckpt_root.exists():
    for ckpt_dir in sorted(ckpt_root.glob('*')):
        if not ckpt_dir.is_dir():
            continue
        if APEX_TAG in ckpt_dir.name:
            continue  # never touch the apex
        if any(slug in ckpt_dir.name for slug in REGRESSED_CKPT_SLUGS):
            continue  # already covered by conservative whole-dir removal
        latest = ckpt_dir / 'latest.npz'
        if latest.exists():
            plan.append(('moderate', f'latest:{ckpt_dir.name}', latest,
                         'redundant warm-start state; head_final.safetensors retained'))
        # While we're here, also surface any step_*.npz intermediate ckpts.
        for step_npz in ckpt_dir.glob('step_*.npz'):
            plan.append(('moderate', f'step_npz:{step_npz.name}', step_npz,
                         'intermediate-step checkpoint; redundant once head_final exists'))

# ── AGGRESSIVE ────────────────────────────────────────────────────────────
# Recapturable corpora — the headbank notebook re-captures these into
# headbank_500u/<slug>/corpus/, so the originals are dead weight after the
# current Colab session finishes.
plan.append(('aggressive', 'q3b_ref_corpus', LAB_ROOT / 'corpora' / 'q3b_ref_corpus',
             'recapturable in the headbank notebook (q3b uses capture_layer=30 there)'))
plan.append(('aggressive', 'q0p5_corpus', LAB_ROOT / 'corpora' / 'q0p5_corpus',
             'recapturable in the headbank notebook (q05b uses capture_layer=20 there)'))
plan.append(('aggressive', 'q3b_ref_artifacts', LAB_ROOT / 'artifacts' / 'q3b_ref',
             'reference artifacts paired with q3b_ref_corpus; recaptured by headbank'))
plan.append(('aggressive', 'q0p5_artifacts', LAB_ROOT / 'artifacts' / 'q0p5',
             'reference artifacts paired with q0p5_corpus; recaptured by headbank'))

# ─── Filter by mode + safety rails ───────────────────────────────────────
MODE_ORDER = {'conservative': 0, 'moderate': 1, 'aggressive': 2}
selected_mode = MODE_ORDER.get(MODE, 1)


def mode_allowed(entry_mode: str) -> bool:
    return MODE_ORDER[entry_mode] <= selected_mode


actionable: list[dict] = []
for entry_mode, label, path, reason in plan:
    if not mode_allowed(entry_mode):
        continue
    if not path.exists():
        continue
    if is_recent(path):
        actionable.append({'label': label, 'path': path, 'size': 0,
                           'mode': entry_mode, 'reason': reason,
                           'skip': f'modified within last {RECENT_WINDOW_HOURS}h — protected'})
        continue
    if APEX_TAG in str(path):
        actionable.append({'label': label, 'path': path, 'size': 0,
                           'mode': entry_mode, 'reason': reason,
                           'skip': 'apex tag matched — refuse to touch'})
        continue
    size = dir_size(path)
    actionable.append({'label': label, 'path': path, 'size': size,
                       'mode': entry_mode, 'reason': reason, 'skip': None})

actionable.sort(key=lambda e: e['size'], reverse=True)

# ─── Print plan ──────────────────────────────────────────────────────────
print(f'\\n=== cleanup plan (mode={MODE}) ===\\n')
total_reclaim = 0
skipped_count = 0
to_delete_count = 0
for e in actionable:
    size_gb = e['size'] / 1e9
    if e['skip']:
        print(f"  SKIP  {size_gb:>7.3f} GB  [{e['mode']:>12s}]  {e['path']}  ({e['skip']})")
        skipped_count += 1
    else:
        print(f"  DEL   {size_gb:>7.3f} GB  [{e['mode']:>12s}]  {e['path']}")
        print(f"        reason: {e['reason']}")
        total_reclaim += e['size']
        to_delete_count += 1
print()
print(f'[plan] would delete: {to_delete_count} items, {total_reclaim/1e9:.2f} GB reclaim')
print(f'[plan] skipped (recent / apex): {skipped_count}')
if not APPLY:
    print('\\n[plan] DRY RUN. Set APPLY=True in Cell 2 and re-run to execute.')
else:
    print('\\n[plan] APPLY=True — executing deletions now...\\n')
    deleted = 0
    failed = 0
    freed = 0
    for e in actionable:
        if e['skip']:
            continue
        path = e['path']
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            deleted += 1
            freed += e['size']
            print(f"  DELETED {e['size']/1e9:>6.3f} GB  {path}")
        except Exception as exc:
            failed += 1
            print(f"  FAILED  {path}: {exc}")
    print(f'\\n[done] deleted={deleted} failed={failed} freed={freed/1e9:.2f} GB')
    print('\\n[!] Files deleted through the Drive mount go to TRASH and keep')
    print('    counting against your quota. Run Cell 4 to empty the trash and')
    print('    actually reclaim the space.')
"""
)

code(
    """# Cell 4 - Empty Drive trash (THE step that actually reclaims quota)

# Deletions via the Drive mount move files to Trash, which still counts
# against your storage quota until emptied. This lists the trash first
# (dry run), then empties it ONLY when EMPTY_TRASH=True.
#
# emptyTrash() permanently deletes EVERYTHING in your Drive trash, not just
# dismantle files. The listing below shows exactly what will go. If you have
# unrelated trash to keep, empty selectively at drive.google.com instead.

EMPTY_TRASH = False

from google.colab import auth
auth.authenticate_user()
from googleapiclient.discovery import build
_drive = build('drive', 'v3')

_resp = _drive.files().list(
    q='trashed = true',
    fields='files(id,name,size,modifiedTime)',
    pageSize=1000,
).execute()
_files = _resp.get('files', [])
_total = sum(int(f.get('size', 0)) for f in _files)
print(f'trashed files: {len(_files)}, {_total/1e9:.2f} GB total')
for f in sorted(_files, key=lambda x: int(x.get('size', 0)), reverse=True)[:30]:
    print(f"  {int(f.get('size',0))/1e9:6.2f} GB  {f.get('name')}")

if EMPTY_TRASH:
    _drive.files().emptyTrash().execute()
    print('\\n[trash] emptied — quota updates within ~1 min')
else:
    print('\\n[trash] DRY RUN. Set EMPTY_TRASH=True and re-run to reclaim the space.')
"""
)


def build_nb() -> dict:
    nb_cells = []
    for kind, text in CELLS:
        if text.startswith("\n"):
            text = text[1:]
        lines = text.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "colab": {"name": "cleanup_drive.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out_path = Path(__file__).resolve().parent / "cleanup_drive.ipynb"
    out_path.write_text(json.dumps(build_nb(), indent=1) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
