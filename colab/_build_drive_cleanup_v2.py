#!/usr/bin/env python3
"""Builder for ``drive_cleanup_v2.ipynb``.

Run from repo root:
    python3 colab/_build_drive_cleanup_v2.py

Emits a Colab notebook that reclaims Drive space for the *current* dismantle
Drive layout (the May 29 2026 generations):

    MyDrive/dismantle_final_push/        # newest: frozen_cache + best_heads + sweep
    MyDrive/dismantle_headbank_corrected # corrected headbank heads
    MyDrive/dismantle/dismantle_export/  # older 500u + headbank_v2 export

The older ``cleanup_drive.ipynb`` hard-codes ``MyDrive/dismantle/maximal_spec_500u``
which is now nested two levels down and only one of three generations, so its
delete rules no longer match what's actually on Drive. This notebook is layout
agnostic: it finds duplicates by Drive's server-side ``md5Checksum`` rather than
by guessing folder names, so it keeps working as the trees keep moving.

Why md5 and not download+hash: the Drive API returns an md5Checksum for every
binary file, computed server-side. We never download a byte — we list metadata,
group identical checksums, and keep exactly one copy of each.

Safety rails baked in (all active even when armed):
  * DRY RUN by default (APPLY=False). Prints the full plan + reclaim first.
  * Dedup ALWAYS keeps >=1 copy of every distinct md5 (never deletes a unique
    file via the dedup path).
  * Skips anything modified within RECENT_WINDOW_HOURS (default 2h) so a live
    upload/training Colab can't lose state.
  * PROTECT_GLOBS (manifests, leaderboards, eval json) are never deleted.
  * APPLY moves duplicates to Trash (reversible). A separate, separately-gated
    cell empties Trash — that is the step that actually reclaims quota.
  * Writes an audit CSV of everything trashed.
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
    """# dismantle Drive cleanup v2 (md5 dedup, current layout)

Reclaims Drive storage by finding **exact-content duplicates** across the
current dismantle trees and keeping one copy of each.

## What it targets

The frozen baselines and trained heads were uploaded across three generations:

| Tree | Created | Holds |
|------|---------|-------|
| `dismantle_final_push/` | newest | `frozen_cache/{q05b,q1p5b,q3b,q7b}/frozen_gguf.npz`, `best_heads/`, `sweep/` |
| `dismantle_headbank_corrected/` | mid | `heads_corrected/{q05b,q1p5b,q3b,q7b}/` |
| `dismantle/dismantle_export/` | oldest | `maximal_spec_500u/`, `headbank_500u_v2/` |

The ~1.2 GB q3b frozen, ~0.9 GB q1p5b, ~0.5 GB q05b, ~2.2 GB q7b baselines each
appear in **multiple** trees with identical bytes. This notebook detects that by
`md5Checksum` (server-side, no download) and trashes the redundant copies,
keeping the one in the highest-priority tree (`dismantle_final_push` by default).

## How to run

1. **Run All** with `APPLY = False` (the default) — prints the dedup plan and
   total reclaim. Nothing is touched.
2. Read the plan. Confirm the *kept* copy of each group is the one you want.
3. Set `APPLY = True` in Cell 2, re-run Cell 5 — duplicates move to **Trash**
   (still recoverable, still counts against quota until emptied).
4. Set `EMPTY_TRASH = True` in Cell 6, run it — **permanently** empties Drive
   trash. This is the step that actually frees quota.

Safety rails (always on): dedup never removes the last copy of a checksum;
anything modified in the last `RECENT_WINDOW_HOURS` is skipped; `PROTECT_GLOBS`
(manifests / leaderboards / eval json) are never deleted.
"""
)

code(
    """# Cell 1 - Authenticate + Drive API client (no mount, no downloads)

from googleapiclient.discovery import build
from google.colab import auth

auth.authenticate_user()
drive = build('drive', 'v3')
print('[auth] Drive v3 client ready')
"""
)

code(
    """# Cell 2 - Settings

# DRY RUN by default. Flip to True (and re-run Cell 5) to move dupes to Trash.
APPLY = False

# 'dedup'  -> only exact md5 duplicates + cruft (.tmp / zero-byte). Safe: every
#             distinct file keeps at least one copy.
# 'stale'  -> dedup + remove whole superseded subtrees listed in STALE_SUBTREES
#             below (gated on the newer generation existing). Bigger reclaim.
MODE = 'dedup'

# Top-level dismantle folders to scan, highest KEEP-priority first. When the
# same bytes live in several trees, the copy in the earliest-listed tree is the
# one kept; copies in later trees are trashed.
PREFER_ROOTS = [
    'dismantle_final_push',
    'dismantle_headbank_corrected',
    'dismantle',
]

# Never trash anything modified within this many hours (protects live Colabs).
RECENT_WINDOW_HOURS = 2.0

# Only dedup files at least this big (bytes). Tiny json/txt dupes aren't worth
# the risk; the storage is all in the npz / safetensors / tar / parquet blobs.
MIN_DEDUP_BYTES = 1 * 1024 * 1024  # 1 MB

# Glob-style name patterns that are NEVER trashed, even if duplicated.
PROTECT_GLOBS = [
    '*manifest*.json',
    'leaderboard.json',
    'frontier*.json',
    '*.runtime.json',
    'tau_eval.json',
    'progress*.json',
]

# Only used when MODE == 'stale'. Each entry is trashed wholesale ONLY IF its
# guard path still exists (so we never delete the old export before the new
# generation is confirmed present). Paths are relative to MyDrive.
STALE_SUBTREES = [
    # (subtree_to_remove, guard_that_must_exist)
    ('dismantle/dismantle_export/maximal_spec_500u',
     'dismantle_final_push/best_heads'),
]

print(f'[cfg] APPLY={APPLY}  MODE={MODE}  RECENT_WINDOW_HOURS={RECENT_WINDOW_HOURS}')
print(f'[cfg] PREFER_ROOTS={PREFER_ROOTS}')
print(f'[cfg] MIN_DEDUP_BYTES={MIN_DEDUP_BYTES/1e6:.1f} MB')
if not APPLY:
    print('[cfg] DRY RUN — nothing will be trashed. Flip APPLY=True in Cell 2.')
"""
)

code(
    """# Cell 3 - Discover roots + walk the whole tree (metadata only)

import fnmatch
from collections import defaultdict

FIELDS = 'nextPageToken, files(id,name,size,md5Checksum,parents,modifiedTime,mimeType,trashed,ownedByMe)'
FOLDER_MIME = 'application/vnd.google-apps.folder'


def _list(q):
    \"\"\"Yield every file matching q, following pagination.\"\"\"
    page = None
    while True:
        resp = drive.files().list(
            q=q, fields=FIELDS, pageSize=1000, pageToken=page,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        for f in resp.get('files', []):
            yield f
        page = resp.get('nextPageToken')
        if not page:
            break


# Resolve each PREFER_ROOTS name to a folder id (prefer one owned by me, not in
# trash). If a name resolves to several folders we take the most recent.
root_ids = {}
for name in PREFER_ROOTS:
    safe = name.replace("'", "\\\\'")
    cands = [f for f in _list(
        f"name = '{safe}' and mimeType = '{FOLDER_MIME}' and trashed = false")]
    cands.sort(key=lambda f: f.get('modifiedTime', ''), reverse=True)
    if cands:
        root_ids[name] = cands[0]['id']
        print(f"[scan] root '{name}' -> {cands[0]['id']}")
    else:
        print(f"[scan] root '{name}' NOT FOUND (skipping)")

assert root_ids, 'No dismantle roots found — check PREFER_ROOTS and your account.'

# BFS each root, recording every node and its parent so we can build paths.
node = {}          # id -> file dict
root_of = {}       # id -> which PREFER_ROOTS name it descends from
for rname, rid in root_ids.items():
    frontier = [rid]
    root_of[rid] = rname
    # fetch the root folder's own metadata so path building has a base
    try:
        node[rid] = drive.files().get(
            fileId=rid, fields='id,name,parents,mimeType',
            supportsAllDrives=True).execute()
    except Exception:
        node[rid] = {'id': rid, 'name': rname, 'mimeType': FOLDER_MIME}
    while frontier:
        nxt = []
        for pid in frontier:
            for f in _list(f"'{pid}' in parents and trashed = false"):
                node[f['id']] = f
                root_of[f['id']] = rname
                if f['mimeType'] == FOLDER_MIME:
                    nxt.append(f['id'])
        frontier = nxt

files = [f for f in node.values() if f.get('mimeType') != FOLDER_MIME]
print(f'[scan] {len(node)} nodes total, {len(files)} files across {len(root_ids)} roots')


def path_of(fid):
    parts = []
    cur = fid
    seen = set()
    while cur in node and cur not in seen:
        seen.add(cur)
        parts.append(node[cur].get('name', cur))
        parents = node[cur].get('parents') or []
        cur = parents[0] if parents else None
    return '/'.join(reversed(parts))


def is_protected(name):
    return any(fnmatch.fnmatch(name, g) for g in PROTECT_GLOBS)


total_bytes = sum(int(f.get('size', 0) or 0) for f in files)
print(f'[scan] tracked file bytes: {total_bytes/1e9:.2f} GB')
"""
)

code(
    """# Cell 4 - Build the plan (dedup + cruft [+ stale])

import time
from datetime import datetime, timezone

NOW = time.time()
RECENT_CUTOFF = NOW - RECENT_WINDOW_HOURS * 3600.0
ROOT_PRIORITY = {name: i for i, name in enumerate(PREFER_ROOTS)}  # lower = keep


def mtime_epoch(f):
    t = f.get('modifiedTime')
    if not t:
        return 0.0
    try:
        return datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp()
    except Exception:
        return 0.0


def is_recent(f):
    return mtime_epoch(f) > RECENT_CUTOFF


plan = []   # dicts: id, path, size, reason, keep_path(optional)

# ---- (1) exact-content duplicates by md5 -------------------------------------
groups = defaultdict(list)
for f in files:
    md5 = f.get('md5Checksum')
    size = int(f.get('size', 0) or 0)
    if not md5 or size < MIN_DEDUP_BYTES:
        continue
    groups[md5].append(f)

for md5, members in groups.items():
    if len(members) < 2:
        continue
    # Rank to decide which single copy to KEEP: lowest root priority, then
    # newest mtime, then shortest path (stable).
    def rank(f):
        return (ROOT_PRIORITY.get(root_of.get(f['id']), 99),
                -mtime_epoch(f),
                len(path_of(f['id'])))
    members_sorted = sorted(members, key=rank)
    keeper = members_sorted[0]
    keep_path = path_of(keeper['id'])
    for f in members_sorted[1:]:
        size = int(f.get('size', 0) or 0)
        p = path_of(f['id'])
        skip = None
        if is_protected(f['name']):
            skip = 'protected name'
        elif is_recent(f):
            skip = f'modified within {RECENT_WINDOW_HOURS}h'
        elif not f.get('ownedByMe', True):
            skip = 'not owned by me'
        plan.append({'id': f['id'], 'path': p, 'size': size,
                     'reason': f'dup of kept copy (md5={md5[:8]})',
                     'keep_path': keep_path, 'skip': skip})

# ---- (2) cruft: half-written .tmp uploads + zero-byte blobs -------------------
for f in files:
    name = f['name']
    size = int(f.get('size', 0) or 0)
    is_tmp = fnmatch.fnmatch(name, '*.tmp') or '.tmp.' in name
    is_zero = size == 0 and f.get('md5Checksum') is not None
    if not (is_tmp or is_zero):
        continue
    if is_protected(name):
        continue
    skip = f'modified within {RECENT_WINDOW_HOURS}h' if is_recent(f) else None
    plan.append({'id': f['id'], 'path': path_of(f['id']), 'size': size,
                 'reason': 'half-written .tmp upload' if is_tmp else 'zero-byte file',
                 'keep_path': None, 'skip': skip})

# ---- (3) stale subtrees (MODE == 'stale' only) -------------------------------
if MODE == 'stale':
    name_to_id = {path_of(i): i for i in node}
    for subtree, guard in STALE_SUBTREES:
        # match by suffix so the MyDrive/ prefix or root naming doesn't matter
        sub_id = next((i for p, i in name_to_id.items() if p.endswith(subtree)), None)
        guard_ok = any(p.endswith(guard) for p in name_to_id)
        if sub_id is None:
            print(f'[stale] {subtree} not present — skip')
            continue
        if not guard_ok:
            print(f'[stale] guard {guard} missing — refuse to remove {subtree}')
            continue
        # add every file under the subtree
        for f in files:
            p = path_of(f['id'])
            if subtree in p and not is_protected(f['name']):
                skip = f'modified within {RECENT_WINDOW_HOURS}h' if is_recent(f) else None
                plan.append({'id': f['id'], 'path': p,
                             'size': int(f.get('size', 0) or 0),
                             'reason': f'stale subtree (superseded; guard {guard} present)',
                             'keep_path': None, 'skip': skip})

# de-dup the plan itself (a file could match >1 rule)
seen_ids = set()
uniq = []
for e in plan:
    if e['id'] in seen_ids:
        continue
    seen_ids.add(e['id'])
    uniq.append(e)
plan = sorted(uniq, key=lambda e: e['size'], reverse=True)

# ---- print -------------------------------------------------------------------
reclaim = 0
to_trash = 0
skipped = 0
print(f'\\n=== Drive cleanup plan (MODE={MODE}) ===\\n')
for e in plan:
    gb = e['size'] / 1e9
    if e['skip']:
        print(f"  SKIP  {gb:>7.3f} GB  {e['path']}  ({e['skip']})")
        skipped += 1
        continue
    print(f"  TRASH {gb:>7.3f} GB  {e['path']}")
    print(f"        why: {e['reason']}")
    if e['keep_path']:
        print(f"        keep: {e['keep_path']}")
    reclaim += e['size']
    to_trash += 1
print()
print(f'[plan] would trash {to_trash} files, reclaim {reclaim/1e9:.2f} GB')
print(f'[plan] skipped (recent / protected / not-owned): {skipped}')
if not APPLY:
    print('\\n[plan] DRY RUN. Set APPLY=True in Cell 2 and re-run Cell 5 to trash.')
"""
)

code(
    """# Cell 5 - Apply: move planned files to Trash + write audit CSV

import csv, io
from datetime import datetime

if not APPLY:
    print('[apply] APPLY=False — nothing trashed. Flip APPLY=True in Cell 2.')
else:
    trashed, failed, freed = 0, 0, 0
    audit = io.StringIO()
    w = csv.writer(audit)
    w.writerow(['id', 'path', 'size_bytes', 'reason', 'keep_path'])
    print('[apply] moving planned files to Trash...\\n')
    for e in plan:
        if e['skip']:
            continue
        try:
            drive.files().update(fileId=e['id'], body={'trashed': True},
                                  supportsAllDrives=True).execute()
            trashed += 1
            freed += e['size']
            w.writerow([e['id'], e['path'], e['size'], e['reason'], e['keep_path'] or ''])
            print(f"  TRASHED {e['size']/1e9:>6.3f} GB  {e['path']}")
        except Exception as exc:
            failed += 1
            print(f"  FAILED  {e['path']}: {exc}")
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    audit_path = f'/content/drive_cleanup_audit_{stamp}.csv'
    with open(audit_path, 'w') as fh:
        fh.write(audit.getvalue())
    print(f'\\n[apply] trashed={trashed} failed={failed} freed={freed/1e9:.2f} GB')
    print(f'[apply] audit log: {audit_path}')
    print('[apply] files are in TRASH and still count against quota.')
    print('[apply] run Cell 6 with EMPTY_TRASH=True to actually reclaim space.')
"""
)

code(
    """# Cell 6 - Empty Drive trash (THE step that reclaims quota)

# emptyTrash() permanently removes EVERYTHING in your Drive trash, not only the
# files this notebook moved there. The listing below shows what will go. If you
# have unrelated trash to keep, leave EMPTY_TRASH=False and empty selectively at
# drive.google.com instead.

EMPTY_TRASH = False

resp = drive.files().list(q='trashed = true',
                          fields='files(id,name,size,modifiedTime)',
                          pageSize=1000).execute()
tf = resp.get('files', [])
tot = sum(int(f.get('size', 0) or 0) for f in tf)
print(f'trashed files: {len(tf)}, {tot/1e9:.2f} GB total')
for f in sorted(tf, key=lambda x: int(x.get('size', 0) or 0), reverse=True)[:30]:
    print(f"  {int(f.get('size',0) or 0)/1e9:6.2f} GB  {f.get('name')}")

if EMPTY_TRASH:
    drive.files().emptyTrash().execute()
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
            "colab": {"name": "drive_cleanup_v2.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out_path = Path(__file__).resolve().parent / "drive_cleanup_v2.ipynb"
    out_path.write_text(json.dumps(build_nb(), indent=1) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
