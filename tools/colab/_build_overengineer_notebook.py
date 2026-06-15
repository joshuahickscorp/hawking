#!/usr/bin/env python3
"""Builder for ``maximal_spec_tau8_overengineer.ipynb``.

Run from repo root:
    python3 colab/_build_overengineer_notebook.py

Keeps the notebook itself easy to audit by emitting it from typed cell
sources instead of hand-editing JSON. Re-running the script is idempotent.
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
    """# Maximal Spec Tau8 Overengineer

Standalone restart-safe notebook for the **overengineering ladder** that sits
on top of the tau8 handoff.

Pipeline when running all cells:

1. Mount Drive and refresh `/content/dismantle` to `codex/maximal-spec-colab`.
2. Pick the current best resumable q1p5 head from `leaderboard.json`.
3. Mine an initial hard-negative corpus from that head.
4. Run a **curriculum ladder** on the mined corpus. Each rung:
   * Warm-starts from the previous rung's head.
   * Uses the new `--rollout-depth-targets` multi-depth joint loss so
     depths {1, 2, 4, 8} are optimized together with explicit weights, not
     drowned out by geometric decay.
   * **Re-mines** hard negatives against the new rung's head before
     handing the corpus to the next rung. The training data adapts as the
     head improves.
5. Run a calibration-weighted variant on the latest mined corpus.
6. Evaluate τ and frontier policy for every produced head and merge into
   `leaderboard.json`.
7. **Export the leaderboard winner as a runtime profile JSON** that the
   dismantle runtime can read directly (head path + env hint block).
8. Mirror everything into `dismantle_export/` for safe download.

The notebook never overwrites tau8 heads. New heads are prefixed
`q1p5_overeng_*` and runtime profiles land under
`<lab_root>/runtime_profiles/`.
"""
)

code(
    """# Cell 1 - Restart-safe setup: Drive, repo, packages, GPU

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

REPO_URL = 'https://github.com/joshuahickscorp/dismantle.git'
BRANCH = 'codex/maximal-spec-colab'
REPO_DIR = Path('/content/dismantle')
DRIVE_ROOT = Path('/content/drive/MyDrive/dismantle')
LAB_ROOT = DRIVE_ROOT / 'maximal_spec_500u'
EXPORT_ROOT = DRIVE_ROOT / 'dismantle_export' / 'maximal_spec_500u'

# Toggle individual stages. The default is the full overengineer pass.
RUN_MINE = True
RUN_CURRICULUM = True
RUN_CALIB = True
RUN_EVAL = True
RUN_EXPORT = True
RUN_RUNTIME_PROFILE = True

# How long the curriculum trains each rung. Keep modest; the ladder is
# meant to milk a known-good head, not retrain from scratch.
CURRICULUM_EPOCHS = 2
CURRICULUM_MINE_KEEP_FRACTION = 0.25
CURRICULUM_MINE_MIN_ROWS = 3000
CURRICULUM_MINE_MAX_ROWS = 10000

# Re-mine cadence. When True, each curriculum rung gets a fresh mined corpus
# scored against the previous rung's head. When False, every rung trains on
# the same initial mine.
REMINE_BETWEEN_RUNGS = True

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as e:
    print(f'[setup] Drive mount skipped/failed: {e}')


def run(cmd, *, cwd=None):
    print('$', ' '.join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)

if not (REPO_DIR / '.git').exists():
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(['git', 'clone', '--depth', '1', '--branch', BRANCH, REPO_URL, REPO_DIR])
else:
    run(['git', '-C', REPO_DIR, 'fetch', 'origin', BRANCH, '--depth', '1'])
    run(['git', '-C', REPO_DIR, 'checkout', BRANCH])
    run(['git', '-C', REPO_DIR, 'reset', '--hard', f'origin/{BRANCH}'])

os.chdir(REPO_DIR)
HEAD_SHA = subprocess.check_output(['git', '-C', str(REPO_DIR), 'rev-parse', '--short', 'HEAD'], text=True).strip()
print(f'[setup] repo={REPO_DIR} branch={BRANCH} sha={HEAD_SHA}')

run([sys.executable, '-u', '-m', 'pip', 'install', '-q', 'pyarrow>=17', 'tqdm>=4.66', 'zstandard', 'safetensors>=0.4'])

import numpy as np
import torch

assert torch.cuda.is_available(), 'No CUDA device. Runtime > Change runtime type > GPU.'
GPU_NAME = torch.cuda.get_device_name(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
BIG_GPU = VRAM_GB >= 40
print(json.dumps({'gpu': GPU_NAME, 'vram_gb': VRAM_GB, 'big_gpu': BIG_GPU, 'repo_sha': HEAD_SHA}, indent=2))
"""
)

code(
    """# Cell 2 - Shared helpers and target config (q1p5 only for this notebook)

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
import hashlib

QWEN_LOCKED_ENV = {
    'DISMANTLE_QWEN_TCB': '1',
    'DISMANTLE_QWEN_VOCAB_PRUNE': '32000',
    'DISMANTLE_QWEN_Q4K_LMHEAD': '1',
    'DISMANTLE_QWEN_FFN_DOWN_Q4K': '1',
    'DISMANTLE_QWEN_Q4K_PREDEC': '1',
}

TARGETS = {
    'q1p5': {
        'model_id': 'Qwen/Qwen2.5-1.5B-Instruct',
        'artifact_dir': LAB_ROOT / 'artifacts' / 'q1p5',
        'corpus_dir': LAB_ROOT / 'corpora' / 'q1p5_corpus',
        'drive_frozen': LAB_ROOT / 'artifacts' / 'q1p5' / 'q1p5_frozen.npz',
        'local_frozen': Path('/content/q1p5_frozen.npz'),
        'capture_layer': 24,
        'train_max_rows': 18000 if BIG_GPU else 6000,
        'train_max_row_tokens': 384 if BIG_GPU else 192,
        'eval_max_windows': 24000 if BIG_GPU else 6000,
        'frontier_max_depth': 24 if BIG_GPU else 12,
        'base_tps_placeholder': 95.0,
        'spec_efficiency_placeholder': 0.80,
    },
}

TAU_DEPTH = 8
FRONTIER_DEPTHS = '2,4,6,8,12,16,24'
FRONTIER_WIDTHS = '2,3,4,6,8'
TARGET = 'q1p5'

MINE_ROOT = LAB_ROOT / 'corpora' / 'q1p5_hardneg'
RUNTIME_PROFILE_DIR = LAB_ROOT / 'runtime_profiles'


def load_json(path, default):
    path = Path(path)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n')
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def remount_drive_for_export():
    try:
        from google.colab import drive
        print('[export] remounting Drive after copy failure...')
        try:
            drive.flush_and_unmount()
            time.sleep(2)
        except Exception as e:
            print(f'[export] flush/unmount skipped: {e}')
        drive.mount('/content/drive', force_remount=True)
        time.sleep(2)
    except Exception as e:
        print(f'[export] Drive remount failed: {e}')


def copy_warm_start(src_ckpt_dir, dst_ckpt_dir, retries=3):
    src_latest = Path(src_ckpt_dir) / 'latest.npz'
    dst_ckpt_dir = Path(dst_ckpt_dir)
    dst_latest = dst_ckpt_dir / 'latest.npz'
    if dst_latest.exists():
        return True
    if not src_latest.exists():
        print(f'[warm] WARN no latest.npz at {src_latest}; training cold')
        return False
    dst_ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, retries + 1):
        tmp = dst_latest.with_suffix(dst_latest.suffix + f'.tmp.{attempt}')
        try:
            if tmp.exists():
                tmp.unlink()
            with open(src_latest, 'rb') as fsrc, open(tmp, 'wb') as fdst:
                shutil.copyfileobj(fsrc, fdst, length=16 * 1024 * 1024)
                fdst.flush()
                os.fsync(fdst.fileno())
            shutil.copystat(src_latest, tmp)
            os.replace(tmp, dst_latest)
            print(f'[warm] copied {src_latest} -> {dst_latest}')
            return True
        except OSError as e:
            last_error = e
            print(f'[warm] copy failed attempt {attempt}/{retries}: {e}')
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            if getattr(e, 'errno', None) == 107 or 'Transport endpoint is not connected' in str(e):
                remount_drive_for_export()
            time.sleep(min(20, 2 * attempt))
    raise OSError(f'warm-start copy failed: {src_latest} -> {dst_latest}: {last_error}')


def run_with_heartbeat(cmd, label='job', interval_sec=60):
    print('$', ' '.join(map(str, cmd)), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    start = time.time()
    last = start
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end='')
        now = time.time()
        if now - last >= interval_sec:
            mem = 'n/a'
            gpu = 'n/a'
            try:
                out = subprocess.check_output([
                    'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                    '--format=csv,noheader,nounits'
                ], text=True).strip().splitlines()[0].split(',')
                gpu = out[0].strip() + '%'
                mem = out[1].strip() + '/' + out[2].strip() + 'MB'
            except Exception:
                pass
            print(f'[{label}] RUNNING elapsed={(now-start)/60:.1f}m gpu={gpu} mem={mem}', flush=True)
            last = now
    rc = p.wait()
    print(f'[{label}] finished rc={rc} elapsed={(time.time()-start)/60:.1f}m', flush=True)
    if rc != 0:
        raise RuntimeError(f'{label} failed with rc={rc}')


def leaderboard_sort_key(row):
    return (
        float(row.get('offline_projected_tps') or 0.0),
        float(row.get('accepted_draft_tokens_per_verify') or 0.0),
        float(row.get('tau') or 0.0),
    )


def best_resumable_row(name):
    rows = load_json(LAB_ROOT / 'leaderboard.json', {'rows': []}).get('rows', [])
    candidates = []
    for row in rows:
        if row.get('target') != name or not row.get('head'):
            continue
        head = Path(row['head'])
        latest = head.parent / 'latest.npz'
        if head.exists() and latest.exists():
            candidates.append(row)
    candidates.sort(key=leaderboard_sort_key, reverse=True)
    if not candidates:
        raise RuntimeError(f'no resumable leaderboard head for {name}')
    return candidates[0]


def merge_leaderboard_rows(rows_to_merge):
    if not rows_to_merge:
        return []
    existing_rows = load_json(LAB_ROOT / 'leaderboard.json', {}).get('rows', [])
    by_key = {(r.get('target'), r.get('tag')): r for r in existing_rows}
    for row in rows_to_merge:
        by_key[(row.get('target'), row.get('tag'))] = row
    merged = sorted(by_key.values(), key=leaderboard_sort_key, reverse=True)
    write_json_atomic(LAB_ROOT / 'leaderboard.json', {'schema': 'dismantle-maximal-spec-leaderboard-v1', 'rows': merged})
    return merged


# Make sure frozen weights are staged locally; the trainer needs them on every call.
for name, target in TARGETS.items():
    target['artifact_dir'].mkdir(parents=True, exist_ok=True)
    if not target['local_frozen'].exists():
        assert target['drive_frozen'].exists(), f'Missing frozen artifact: {target["drive_frozen"]}'
        shutil.copy2(target['drive_frozen'], target['local_frozen'])
        print(f'[state:{name}] copied frozen -> {target["local_frozen"]}')
    assert target['corpus_dir'].exists(), f'Missing corpus: {target["corpus_dir"]}'

print('top resumable q1p5 row:')
best_row = best_resumable_row(TARGET)
print(json.dumps({
    'tag': best_row.get('tag'),
    'head': best_row.get('head'),
    'tau': best_row.get('tau'),
    'accepted_draft_tokens_per_verify': best_row.get('accepted_draft_tokens_per_verify'),
    'offline_projected_tps': best_row.get('offline_projected_tps'),
}, indent=2))
"""
)

code(
    """# Cell 3 - Hard-negative mine helper (callable, restart-safe)

# The miner writes a parquet directory shaped exactly like the original
# corpus, so the trainer ingests it without code changes. Each mine writes
# `mine_manifest.json` which doubles as the "skip if exists" signal.

from safetensors import safe_open


def _read_head_meta(head_path):
    try:
        with safe_open(str(head_path), framework='pt', device='cpu') as f:
            return f.metadata() or {}
    except Exception as e:
        print(f'[mine] WARN metadata read failed for {head_path}: {e}')
        return {}


def mine_hard_negatives(head_path, slug, *, force=False):
    '''Mine hard negatives against ``head_path``. Idempotent on slug.'''
    target = TARGETS[TARGET]
    head_path = Path(head_path)
    mine_dir = MINE_ROOT / slug
    manifest_path = mine_dir / 'mine_manifest.json'
    if manifest_path.exists() and not force:
        print(f'[mine] skip; reuse existing mine at {mine_dir}')
        return mine_dir
    meta = _read_head_meta(head_path)
    nb = int(meta.get('num_blocks', '1'))
    hh = int(meta.get('n_heads', '16'))
    ff = float(meta.get('ff_mult', '4.0'))
    cmd = [
        sys.executable, '-u', 'colab/eagle5_hard_neg_miner.py',
        '--ckpt', str(head_path),
        '--frozen', str(target['local_frozen']),
        '--corpus-dir', str(target['corpus_dir']),
        '--out-dir', str(mine_dir),
        '--keep-fraction', str(CURRICULUM_MINE_KEEP_FRACTION),
        '--keep-min-rows', str(CURRICULUM_MINE_MIN_ROWS),
        '--keep-max-rows', str(CURRICULUM_MINE_MAX_ROWS),
        '--shards-to-scan', '0',
        '--rows-per-output-shard', '200',
        '--max-row-tokens', str(target['train_max_row_tokens']),
        '--score', 'depth1_miss',
        '--num-blocks', str(nb),
        '--head-heads', str(hh),
        '--head-ff-mult', str(ff),
        '--device', 'cuda',
        '--seed', '0',
    ]
    run_with_heartbeat(cmd, label=f'mine_{slug}', interval_sec=120)
    assert manifest_path.exists(), f'mine manifest missing at {manifest_path}'
    return mine_dir


# Initial mine against the leaderboard winner.
best_row = best_resumable_row(TARGET)
best_head = Path(best_row['head'])
best_tag = best_head.parent.name
base_hash = hashlib.sha1(best_tag.encode()).hexdigest()[:8]

if RUN_MINE:
    INITIAL_MINE_DIR = mine_hard_negatives(best_head, f'from_{base_hash}')
else:
    INITIAL_MINE_DIR = MINE_ROOT / f'from_{base_hash}'
    print(f'[mine] RUN_MINE=False; expecting prior mine at {INITIAL_MINE_DIR}')

mine_manifest_path = INITIAL_MINE_DIR / 'mine_manifest.json'
mine_manifest = load_json(mine_manifest_path, {})
print(json.dumps({
    'mine_dir': str(INITIAL_MINE_DIR),
    'rows_kept': mine_manifest.get('write_stats', {}).get('rows_written'),
    'shards_written': mine_manifest.get('write_stats', {}).get('shards_written'),
    'cutoff_score': mine_manifest.get('score_summary', {}).get('cutoff_score'),
    'score_mean': mine_manifest.get('score_summary', {}).get('score_mean'),
    'score_p90': mine_manifest.get('score_summary', {}).get('score_p90'),
}, indent=2))

# Cache head architecture metadata for the curriculum/calib cells.
BASE_META = {
    'nb': int(_read_head_meta(best_head).get('num_blocks', '1')),
    'hh': int(_read_head_meta(best_head).get('n_heads', '16')),
    'ff': float(_read_head_meta(best_head).get('ff_mult', '4.0')),
}
"""
)

code(
    """# Cell 4 - Curriculum ladder with multi-depth joint loss + iterative remine

# Each rung trains for `CURRICULUM_EPOCHS` epochs and warm-starts from the
# previous rung's head. When REMINE_BETWEEN_RUNGS is True the corpus is
# refreshed against the freshly-trained head before the next rung starts,
# so the harder targets keep up with the head.

CURRICULUM_RUNGS = [
    {
        'name': 'rung1_d2_p050_w006_lr5e-5',
        'lr': 5e-5,
        'rollout_loss_weight': 0.06,
        'rollout_depth': 2,
        'rollout_starts_per_batch': 4,
        'rollout_draft_prob': 0.50,
        'rollout_depth_gamma': 0.95,
        'rollout_depth_targets': '1,2',
        'rollout_depth_target_weights': '1.0,0.8',
        'calib_loss_weight': 0.12,
        'residual_delta_loss_weight': 0.010,
    },
    {
        'name': 'rung2_d4_p070_w010_lr4e-5',
        'lr': 4e-5,
        'rollout_loss_weight': 0.10,
        'rollout_depth': 4,
        'rollout_starts_per_batch': 4,
        'rollout_draft_prob': 0.70,
        'rollout_depth_gamma': 0.93,
        'rollout_depth_targets': '1,2,4',
        'rollout_depth_target_weights': '1.0,0.7,0.5',
        'calib_loss_weight': 0.12,
        'residual_delta_loss_weight': 0.012,
    },
    {
        'name': 'rung3_d8_p085_w015_lr3e-5',
        'lr': 3e-5,
        'rollout_loss_weight': 0.15,
        'rollout_depth': 8,
        'rollout_starts_per_batch': 3,
        'rollout_draft_prob': 0.85,
        'rollout_depth_gamma': 0.90,
        'rollout_depth_targets': '1,2,4,8',
        'rollout_depth_target_weights': '1.0,0.7,0.5,0.3',
        'calib_loss_weight': 0.14,
        'residual_delta_loss_weight': 0.014,
    },
]


def _train_one_rung(rung_idx, rung, warm_dir, corpus_dir, base_meta):
    name = TARGET
    target = TARGETS[name]
    tag = f'{name}_overeng_curric_{rung["name"]}_from_{base_hash}'
    ckpt_dir = Path(target['artifact_dir']) / 'checkpoints' / tag
    head = ckpt_dir / 'head_final.safetensors'
    if head.exists():
        print(f'[curric] skip existing {head}')
        return head, ckpt_dir, tag
    copy_warm_start(warm_dir, ckpt_dir)
    batch = 48 if BIG_GPU else 20
    cmd = [
        sys.executable, '-u', 'colab/eagle5_train_pytorch.py',
        '--corpus-dir', str(corpus_dir),
        '--frozen', str(target['local_frozen']),
        '--ckpt-dir', str(ckpt_dir),
        '--epochs', str(CURRICULUM_EPOCHS),
        '--batch-size', str(batch),
        '--seq-len', '16',
        '--lr', str(rung['lr']),
        '--num-blocks', str(base_meta['nb']),
        '--head-heads', str(base_meta['hh']),
        '--head-ff-mult', str(base_meta['ff']),
        '--capture-layer', str(target['capture_layer']),
        '--max-rows', str(target['train_max_rows']),
        '--max-row-tokens', str(target['train_max_row_tokens']),
        '--sparsity-head', 'off',
        '--seed', str(9000 + rung_idx),
        '--calib-loss-weight', str(rung['calib_loss_weight']),
        '--residual-delta-loss-weight', str(rung['residual_delta_loss_weight']),
        '--rollout-loss-weight', str(rung['rollout_loss_weight']),
        '--rollout-depth', str(rung['rollout_depth']),
        '--rollout-starts-per-batch', str(rung['rollout_starts_per_batch']),
        '--rollout-draft-prob', str(rung['rollout_draft_prob']),
        '--rollout-depth-gamma', str(rung['rollout_depth_gamma']),
        '--rollout-depth-targets', str(rung.get('rollout_depth_targets', '')),
        '--rollout-depth-target-weights', str(rung.get('rollout_depth_target_weights', '')),
        '--save-safetensors',
    ]
    print(f'\\n=== [curric] train rung {rung_idx} {tag}')
    run_with_heartbeat(cmd, label=f'curric_{name}_{rung["name"]}', interval_sec=60)
    if not head.exists():
        raise FileNotFoundError(f'curriculum head missing after train: {head}')
    return head, ckpt_dir, tag


CURRICULUM_HEADS = []
if RUN_CURRICULUM:
    warm_dir = best_head.parent
    current_mine = INITIAL_MINE_DIR
    for idx, rung in enumerate(CURRICULUM_RUNGS, start=1):
        head_path, ckpt_dir, tag = _train_one_rung(idx, rung, warm_dir, current_mine, BASE_META)
        CURRICULUM_HEADS.append({
            'head': head_path,
            'ckpt_dir': ckpt_dir,
            'tag': tag,
            'rung': rung['name'],
            'mine_dir': str(current_mine),
        })
        warm_dir = ckpt_dir
        # Re-mine against the freshly-trained head before the next rung.
        if REMINE_BETWEEN_RUNGS and idx < len(CURRICULUM_RUNGS) and RUN_MINE:
            slug = f'from_{base_hash}_after_rung{idx}'
            try:
                current_mine = mine_hard_negatives(head_path, slug)
            except Exception as e:
                print(f'[curric] WARN remine after rung {idx} failed: {e}; reusing {current_mine}')
else:
    print('RUN_CURRICULUM=False; skipping curriculum ladder.')

for entry in CURRICULUM_HEADS:
    print(entry['rung'], '->', entry['head'], '(mined from', entry['mine_dir'], ')')
"""
)

code(
    """# Cell 5 - Calibration-weighted variant on the latest mined corpus

# Same architecture, brief training with a heavy calib_loss_weight. Targets
# the runtime's variable-K gating: we want the head's calibration signal to
# be a faithful acceptance-probability estimator. Trains on the freshest
# mine (the last rung's remined corpus when available, else the initial).

CALIB_SPEC = {
    'name': 'calib_heavy_w030_d4_p070_lr3e-5',
    'lr': 3e-5,
    'epochs': max(2, CURRICULUM_EPOCHS),
    'rollout_loss_weight': 0.08,
    'rollout_depth': 4,
    'rollout_starts_per_batch': 4,
    'rollout_draft_prob': 0.70,
    'rollout_depth_gamma': 0.92,
    'rollout_depth_targets': '1,2,4',
    'rollout_depth_target_weights': '1.0,0.6,0.4',
    'calib_loss_weight': 0.30,
    'residual_delta_loss_weight': 0.010,
}

CALIB_HEADS = []
if RUN_CALIB:
    target = TARGETS[TARGET]
    # Warm-start from the strongest curriculum rung when available, else best row.
    warm_dir = (CURRICULUM_HEADS[-1]['ckpt_dir'] if CURRICULUM_HEADS else best_head.parent)
    calib_mine_dir = Path(CURRICULUM_HEADS[-1]['mine_dir']) if CURRICULUM_HEADS else INITIAL_MINE_DIR
    tag = f'{TARGET}_overeng_{CALIB_SPEC["name"]}_from_{base_hash}'
    ckpt_dir = Path(target['artifact_dir']) / 'checkpoints' / tag
    head = ckpt_dir / 'head_final.safetensors'
    if head.exists():
        print(f'[calib] skip existing {head}')
    else:
        copy_warm_start(warm_dir, ckpt_dir)
        batch = 48 if BIG_GPU else 20
        cmd = [
            sys.executable, '-u', 'colab/eagle5_train_pytorch.py',
            '--corpus-dir', str(calib_mine_dir),
            '--frozen', str(target['local_frozen']),
            '--ckpt-dir', str(ckpt_dir),
            '--epochs', str(CALIB_SPEC['epochs']),
            '--batch-size', str(batch),
            '--seq-len', '16',
            '--lr', str(CALIB_SPEC['lr']),
            '--num-blocks', str(BASE_META['nb']),
            '--head-heads', str(BASE_META['hh']),
            '--head-ff-mult', str(BASE_META['ff']),
            '--capture-layer', str(target['capture_layer']),
            '--max-rows', str(target['train_max_rows']),
            '--max-row-tokens', str(target['train_max_row_tokens']),
            '--sparsity-head', 'off',
            '--seed', '9301',
            '--calib-loss-weight', str(CALIB_SPEC['calib_loss_weight']),
            '--residual-delta-loss-weight', str(CALIB_SPEC['residual_delta_loss_weight']),
            '--rollout-loss-weight', str(CALIB_SPEC['rollout_loss_weight']),
            '--rollout-depth', str(CALIB_SPEC['rollout_depth']),
            '--rollout-starts-per-batch', str(CALIB_SPEC['rollout_starts_per_batch']),
            '--rollout-draft-prob', str(CALIB_SPEC['rollout_draft_prob']),
            '--rollout-depth-gamma', str(CALIB_SPEC['rollout_depth_gamma']),
            '--rollout-depth-targets', str(CALIB_SPEC['rollout_depth_targets']),
            '--rollout-depth-target-weights', str(CALIB_SPEC['rollout_depth_target_weights']),
            '--save-safetensors',
        ]
        print(f'\\n=== [calib] train {tag}')
        run_with_heartbeat(cmd, label=f'calib_{TARGET}', interval_sec=60)
        if not head.exists():
            raise FileNotFoundError(f'calib head missing after train: {head}')
    CALIB_HEADS.append({
        'head': head,
        'ckpt_dir': ckpt_dir,
        'tag': tag,
        'rung': CALIB_SPEC['name'],
        'mine_dir': str(calib_mine_dir),
    })
else:
    print('RUN_CALIB=False; skipping calibration variant.')

for entry in CALIB_HEADS:
    print(entry['rung'], '->', entry['head'], '(mined from', entry['mine_dir'], ')')
"""
)

code(
    """# Cell 6 - Evaluate τ and frontier policy for every produced head


def eval_head(head_path):
    target = TARGETS[TARGET]
    head_path = Path(head_path)
    tag = head_path.parent.name if head_path.parent.name != 'heads' else head_path.stem
    out_dir = Path(target['artifact_dir']) / 'eval' / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tau_path = out_dir / 'tau.json'
    frontier_path = out_dir / 'frontier.json'

    meta = _read_head_meta(head_path)
    nb_e = meta.get('num_blocks', '1')
    hh_e = meta.get('n_heads', '16')
    ff_e = meta.get('ff_mult', '4.0')

    if not tau_path.exists():
        run_with_heartbeat([
            sys.executable, 'colab/eagle5_tau_eval_pytorch.py',
            '--ckpt', str(head_path),
            '--frozen', str(target['local_frozen']),
            '--corpus', str(target['corpus_dir']),
            '--out', str(tau_path),
            '--depth', str(TAU_DEPTH),
            '--max-windows', str(target['eval_max_windows']),
            '--max-row-tokens', str(target['train_max_row_tokens']),
            '--num-blocks', str(nb_e),
            '--head-heads', str(hh_e),
            '--head-ff-mult', str(ff_e),
            '--base-tps', str(target['base_tps_placeholder']),
            '--w4a8-multiplier', '1.0',
            '--spec-efficiency', str(target['spec_efficiency_placeholder']),
        ], label=f'eval-tau-{tag}', interval_sec=60)
    if not frontier_path.exists():
        run_with_heartbeat([
            sys.executable, 'colab/eagle5_frontier_policy.py',
            '--ckpt', str(head_path),
            '--frozen', str(target['local_frozen']),
            '--corpus', str(target['corpus_dir']),
            '--out', str(frontier_path),
            '--max-depth', str(target['frontier_max_depth']),
            '--depths', FRONTIER_DEPTHS,
            '--lattice-widths', FRONTIER_WIDTHS,
            '--max-windows', str(target['eval_max_windows']),
            '--max-row-tokens', str(target['train_max_row_tokens']),
            '--eval-batch-size', '192',
            '--num-blocks', str(nb_e),
            '--head-heads', str(hh_e),
            '--head-ff-mult', str(ff_e),
            '--base-tps', str(target['base_tps_placeholder']),
            '--w4a8-multiplier', '1.0',
            '--spec-efficiency', str(target['spec_efficiency_placeholder']),
        ], label=f'eval-frontier-{tag}', interval_sec=60)
    tau = load_json(tau_path, {})
    frontier = load_json(frontier_path, {})
    best = frontier.get('policies', {}).get('best_deployable', {})
    overall = frontier.get('policies', {}).get('best_overall', {})
    return {
        'target': TARGET,
        'tag': tag,
        'head': str(head_path),
        'tau_path': str(tau_path),
        'frontier_path': str(frontier_path),
        'tau': tau.get('tau'),
        'depth1_accept_rate': tau.get('depth1_accept_rate'),
        'best_deployable': best,
        'best_overall': overall,
        'offline_projected_tps': best.get('projected_dec_tps', 0.0),
        'accepted_draft_tokens_per_verify': best.get('accepted_draft_tokens_per_verify', 0.0),
        'policy_kind': best.get('kind'),
        'metadata': meta,
        'source_tag': best_tag,
    }


NEW_ROWS = []
HEAD_TO_ENTRY = {}
for entry in CURRICULUM_HEADS + CALIB_HEADS:
    HEAD_TO_ENTRY[str(entry['head'])] = entry

if RUN_EVAL:
    for entry in CURRICULUM_HEADS + CALIB_HEADS:
        row = eval_head(entry['head'])
        row['overengineer_rung'] = entry['rung']
        row['mine_dir'] = entry.get('mine_dir')
        NEW_ROWS.append(row)
        merged = merge_leaderboard_rows([row])
        top = merged[0]
        print(f"[eval] {row['tag']} tau={row.get('tau')} accepted={row.get('accepted_draft_tokens_per_verify')} tps={row.get('offline_projected_tps')}")
        print(f"[eval] leaderboard top now: {top.get('target')} {top.get('tag')} tau={top.get('tau')} tps={top.get('offline_projected_tps')}")
else:
    print('RUN_EVAL=False; skipping eval.')

print('\\n=== Overengineer rows ===')
for r in NEW_ROWS:
    print(f"{r['tag'][:72]:72s} tau={r.get('tau')} accepted={r.get('accepted_draft_tokens_per_verify')} tps={r.get('offline_projected_tps')} policy={r.get('policy_kind')}")
"""
)

code(
    """# Cell 7 - Frontier-policy → runtime profile JSON for the leaderboard winner

# Produces a single self-contained JSON the dismantle runtime can read:
#   * head safetensors path
#   * env block (QWEN locked + EAGLE5 policy hints derived from frontier.json)
#   * the winning policy snapshot
#   * provenance (repo sha, source frontier file, tau/accepted/projected tps)
#
# Default location: <lab_root>/runtime_profiles/q1p5_winner.runtime.json. We
# also write a per-tag copy so prior winners are preserved across runs.


def _normalize_for_env(v):
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, (int, float)):
        return str(v)
    return str(v)


def export_runtime_profile(row):
    if not row.get('frontier_path') or not row.get('head'):
        print('[profile] row missing head/frontier_path; skipping')
        return None
    frontier = load_json(row['frontier_path'], {})
    hints = frontier.get('runtime_hints', {}) or {}
    best = frontier.get('policies', {}).get('best_deployable', {}) or {}

    runtime_env = dict(QWEN_LOCKED_ENV)
    # The runtime expects an EAGLE5_HEAD pointer.
    runtime_env['EAGLE5_HEAD'] = str(row['head'])
    # Variable-K is the most directly actionable policy; always emit its
    # hints if frontier surfaced them.
    if hints.get('variable_k', {}).get('env'):
        for k, v in hints['variable_k']['env'].items():
            runtime_env[k] = _normalize_for_env(v)
    if hints.get('entropy_routing', {}).get('env'):
        for k, v in hints['entropy_routing']['env'].items():
            runtime_env[k] = _normalize_for_env(v)
    if hints.get('draft_lattice', {}).get('env'):
        for k, v in hints['draft_lattice']['env'].items():
            runtime_env[k] = _normalize_for_env(v)
    # If best_deployable was fixed-K, override the variable-K knob to match.
    if best.get('kind') == 'fixed_k':
        runtime_env.pop('DISMANTLE_EAGLE5_VARIABLE_K', None)
        runtime_env.pop('DISMANTLE_EAGLE5_CONF_THRESH', None)
        if best.get('max_depth') is not None:
            runtime_env['DISMANTLE_EAGLE5_FIXED_K'] = str(best['max_depth'])

    payload = {
        'schema': 'dismantle-eagle5-runtime-profile-v1',
        'created_at_unix': int(time.time()),
        'repo_sha': HEAD_SHA,
        'target': row.get('target'),
        'tag': row.get('tag'),
        'head': row.get('head'),
        'head_sha256': sha256_file(row['head']) if Path(row['head']).is_file() else None,
        'metrics': {
            'tau': row.get('tau'),
            'depth1_accept_rate': row.get('depth1_accept_rate'),
            'accepted_draft_tokens_per_verify': row.get('accepted_draft_tokens_per_verify'),
            'offline_projected_tps': row.get('offline_projected_tps'),
            'policy_kind': row.get('policy_kind'),
        },
        'best_deployable_policy': best,
        'runtime_env': runtime_env,
        'frontier_source': row.get('frontier_path'),
        'tau_source': row.get('tau_path'),
        'locked_qwen_env': QWEN_LOCKED_ENV,
        'mine_dir': row.get('mine_dir'),
        'overengineer_rung': row.get('overengineer_rung'),
    }

    RUNTIME_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    safe_tag = ''.join(c if c.isalnum() or c in '._-+' else '_' for c in str(row.get('tag') or 'unknown'))
    per_tag_path = RUNTIME_PROFILE_DIR / f'{safe_tag}.runtime.json'
    write_json_atomic(per_tag_path, payload)
    winner_path = RUNTIME_PROFILE_DIR / f'{row.get("target") or "unknown"}_winner.runtime.json'
    write_json_atomic(winner_path, payload)
    print(f"[profile] wrote per-tag profile: {per_tag_path}")
    print(f"[profile] wrote winner profile:  {winner_path}")
    return {'per_tag': per_tag_path, 'winner': winner_path, 'payload': payload}


WINNER_PROFILE = None
if RUN_RUNTIME_PROFILE:
    rows_all = load_json(LAB_ROOT / 'leaderboard.json', {'rows': []}).get('rows', [])
    rows_q = [r for r in rows_all if r.get('target') == TARGET and r.get('head') and r.get('frontier_path')]
    if not rows_q:
        print('[profile] no leaderboard rows with frontier_path; skipping')
    else:
        rows_q.sort(key=leaderboard_sort_key, reverse=True)
        WINNER_PROFILE = export_runtime_profile(rows_q[0])
else:
    print('RUN_RUNTIME_PROFILE=False; skipping runtime profile export.')

if WINNER_PROFILE is not None:
    payload = WINNER_PROFILE['payload']
    print(json.dumps({
        'winner_tag': payload.get('tag'),
        'projected_tps': payload['metrics'].get('offline_projected_tps'),
        'policy_kind': payload['metrics'].get('policy_kind'),
        'runtime_env_keys': sorted(payload['runtime_env'].keys()),
    }, indent=2))
"""
)

code(
    """# Cell 8 - Safe export of leaderboard, new heads, eval, and runtime profile

EXPORT_NOTE = 'post_overengineer'


def copy_file(src, dst, key, manifest, sha=False, retries=3):
    src = Path(src)
    dst = Path(dst)
    manifest.setdefault('copy_errors', {})
    last_error = None
    for attempt in range(1, retries + 1):
        tmp = dst.with_suffix(dst.suffix + f'.tmp.{attempt}')
        try:
            if not src.exists() or not src.is_file():
                manifest['missing'][key] = str(src)
                return None
            dst.parent.mkdir(parents=True, exist_ok=True)
            if tmp.exists():
                tmp.unlink()
            with open(src, 'rb') as fsrc, open(tmp, 'wb') as fdst:
                shutil.copyfileobj(fsrc, fdst, length=16 * 1024 * 1024)
                fdst.flush()
                os.fsync(fdst.fileno())
            shutil.copystat(src, tmp)
            os.replace(tmp, dst)
            info = {'source': str(src), 'exported': str(dst), 'bytes': int(dst.stat().st_size)}
            if sha:
                info['sha256'] = sha256_file(dst)
            manifest['files'][key] = info
            manifest['copy_errors'].pop(key, None)
            return info
        except OSError as e:
            last_error = repr(e)
            print(f'[export] copy failed attempt {attempt}/{retries}: {src} -> {dst}: {e}')
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            if getattr(e, 'errno', None) == 107 or 'Transport endpoint is not connected' in str(e):
                remount_drive_for_export()
            time.sleep(min(20, 2 * attempt))
    manifest['missing'][key] = str(src)
    manifest['copy_errors'][key] = last_error or 'unknown copy error'
    print(f'[export] giving up on optional copy key={key}; export will continue')
    return None


if RUN_EXPORT:
    manifest = {
        'schema': 'dismantle-maximal-spec-overengineer-export-v1',
        'created_at_unix': int(time.time()),
        'repo_sha': HEAD_SHA,
        'note': EXPORT_NOTE,
        'lab_root': str(LAB_ROOT),
        'export_root': str(EXPORT_ROOT),
        'files': {},
        'missing': {},
        'new_rows': [
            {
                'tag': r.get('tag'),
                'tau': r.get('tau'),
                'accepted_draft_tokens_per_verify': r.get('accepted_draft_tokens_per_verify'),
                'offline_projected_tps': r.get('offline_projected_tps'),
                'policy_kind': r.get('policy_kind'),
                'overengineer_rung': r.get('overengineer_rung'),
                'source_tag': r.get('source_tag'),
                'mine_dir': r.get('mine_dir'),
            }
            for r in NEW_ROWS
        ],
    }
    meta_dir = EXPORT_ROOT / 'metadata'
    copy_file(LAB_ROOT / 'leaderboard.json', meta_dir / 'leaderboard.json', 'leaderboard', manifest)

    # Mine manifests for every mine we materialised this run.
    seen_mines = set()
    for entry in CURRICULUM_HEADS + CALIB_HEADS:
        md = entry.get('mine_dir')
        if not md or md in seen_mines:
            continue
        seen_mines.add(md)
        copy_file(Path(md) / 'mine_manifest.json',
                  EXPORT_ROOT / 'mine' / Path(md).name / 'mine_manifest.json',
                  f'mine/{Path(md).name}', manifest)

    for r in NEW_ROWS:
        tag = str(r.get('tag') or 'unknown')
        safe_tag = ''.join(c if c.isalnum() or c in '._-+' else '_' for c in tag)
        if r.get('head'):
            copy_file(r['head'], EXPORT_ROOT / 'heads' / TARGET / f'overeng_{safe_tag}.safetensors',
                      f'heads/{TARGET}/{tag}', manifest, sha=True)
        for key in ['tau_path', 'frontier_path']:
            if r.get(key):
                copy_file(r[key], EXPORT_ROOT / 'eval' / TARGET / safe_tag / Path(r[key]).name,
                          f'eval/{TARGET}/{tag}/{Path(r[key]).name}', manifest)

    # Runtime profile output (per-tag and winner). Mirrors the lab dir layout.
    if WINNER_PROFILE is not None:
        for label, src_path in WINNER_PROFILE.items():
            if label == 'payload':
                continue
            copy_file(src_path,
                      EXPORT_ROOT / 'runtime_profiles' / Path(src_path).name,
                      f'runtime_profiles/{Path(src_path).name}', manifest)
    for prof in sorted(RUNTIME_PROFILE_DIR.glob('*.runtime.json')) if RUNTIME_PROFILE_DIR.exists() else []:
        copy_file(prof,
                  EXPORT_ROOT / 'runtime_profiles' / prof.name,
                  f'runtime_profiles/{prof.name}', manifest)

    manifest_path = EXPORT_ROOT / f'manifest_overengineer.json'
    write_json_atomic(manifest_path, manifest)
    print(f'[export] wrote {manifest_path}')
    total = sum(v['bytes'] for v in manifest['files'].values())
    print(f"[export] copied={len(manifest['files'])} missing={len(manifest['missing'])} total={total/1e9:.2f}GB")
else:
    print('RUN_EXPORT=False; skipping export.')

print('\\nDone. Inspect:')
print(f'  leaderboard: {LAB_ROOT / "leaderboard.json"}')
print(f'  runtime profiles: {RUNTIME_PROFILE_DIR}')
print(f'  export root: {EXPORT_ROOT}')
"""
)


def build_nb() -> dict:
    nb_cells = []
    for kind, text in CELLS:
        if text.startswith("\n"):
            text = text[1:]
        lines = text.splitlines(keepends=True)
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": lines,
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "maximal_spec_tau8_overengineer.ipynb",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out_path = Path(__file__).resolve().parent / "maximal_spec_tau8_overengineer.ipynb"
    out_path.write_text(json.dumps(build_nb(), indent=1) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
