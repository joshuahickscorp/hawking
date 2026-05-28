#!/usr/bin/env python3
"""Builder for ``maximal_spec_headbank_500u.ipynb``.

Run from repo root:
    python3 colab/_build_headbank_notebook.py

Produces a multi-model "head bank" notebook that, for every supported model,
runs the full Eagle5 pipeline:

    frozen-weights extraction
    → corpus capture (residual + intermediate + AWQ stats + top-k logits)
    → AWQ per-channel calibration
    → Eagle5 base head sweep
    → overengineer pass (mine + curriculum + calib variant)
    → τ + frontier eval
    → runtime profile JSON export
    → safe Drive export

Per-model toggles let you skip or re-run individual models without losing
work. Every stage is restart-safe via skip-existing artifacts.

The notebook is the spiritual sequel to ``maximal_spec_decode_500u.ipynb`` +
``maximal_spec_tau8_handoff.ipynb`` + ``maximal_spec_tau8_overengineer.ipynb``,
generalized over the model dimension.
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
    """# Maximal Spec Head Bank 500U

Multi-model head bank. Trains a polished Eagle5 spec-decode head per model,
then exports a runtime profile the dismantle laptop runtime can source.

## Models

**Verified (served by dismantle-core today, default ON):**
`q05b`, `q3b`, `q7b` (Qwen2.5) + `dsv2` (DeepSeek-V2-Lite).

**Rust-wired but UNVERIFIED (no GGUF smoke run yet, default OFF):**
`llama32_3b`, `mistral7b` (LlamaDense engine), `gemma2_2b`, `phi35_mini`.
These four were added to dismantle-core but not yet validated end-to-end —
the Rust smoke tests pin-then-guard a hash the moment a GGUF is dropped in.
**Their heads can still be trained here right now** because this whole
pipeline is pure PyTorch + transformers and never touches the Rust runtime;
flip `RUN_<slug> = True` to train one and it's ready the moment Rust
verification lands. Llama / Mistral / Gemma are gated on HuggingFace — set
`HF_TOKEN` in Cell 1.

## Pipeline per model

1. Frozen-weights extraction (`token_embd`, `lm_head`, `output_norm`;
   handles tied embeddings for Gemma/Llama).
2. Corpus capture: residual + intermediate at the capture layer, plus
   per-channel activation stats for AWQ.
3. AWQ per-channel smoothing calibration.
4. Eagle5 base head **sweep** (4 variants: fast, wide, compact, tiny-distill).
5. **Two-tier eval** — quick rank of all 4 variants (cheap window count, no
   lattice search), then full eval only on the per-model winner. Cuts ~3/4
   of frontier-search cost without changing which head wins.
6. Runtime profile JSON for the per-model winner (head + AWQ scales + locked
   env + EAGLE5 policy hints + `rust_serving_verified` flag).
7. Safe export to `dismantle_export/headbank_500u/<slug>/` + aggregate
   `headbank_manifest.json`.

## What's deliberately OFF

The **overengineer pass** (rollout/multi-depth curriculum) is disabled by
default. The q1p5 run on 2026-05-28 proved it REGRESSES tau under the
teacher-forced eval: all rungs dropped tau from the apex 7.99 to ~2-3 while
depth-1 acceptance held at ~99.95%. The base sweep's pure depth-1 CE head
is the winner. See the `RUN_OVERENGINEER` comment in Cell 1.

## Notes

Per-model toggles (`RUN_<slug>`) + per-stage toggles let a single Colab
session resume cleanly after a runtime kill; every stage skips existing
artifacts. Defaults target Colab Pro+ A100 background execution. With only
the four verified models on, budget ~8-10 hr.
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
LAB_ROOT = DRIVE_ROOT / 'headbank_500u'
EXPORT_ROOT = DRIVE_ROOT / 'dismantle_export' / 'headbank_500u'

# Per-model toggles. Flip any to False to skip; the rest still run normally.
#
# VERIFIED archs (qwen2, deepseek2) are served by the dismantle Rust runtime
# today, so a trained head is immediately usable locally.
RUN_Q05B = True
RUN_Q3B  = True
RUN_Q7B  = True
RUN_DSV2 = True
#
# UNVERIFIED archs (llama, gemma2, phi3) were wired into dismantle-core but
# NOT yet validated end-to-end (no GGUF smoke run). The HEAD can still be
# trained here right now — this whole pipeline is pure PyTorch+transformers
# and never touches the Rust runtime. Flip one to True to train its head; it
# will be ready the moment the Rust smoke tests confirm that arch serves
# correctly. Default OFF so we don't spend CU on heads that can't be served
# locally yet.
RUN_LLAMA32_3B = False
RUN_MISTRAL7B  = False
RUN_GEMMA2_2B  = False
RUN_PHI35_MINI = False

# Llama / Mistral / Gemma are gated on HuggingFace — you must accept the
# license on the model page and provide a token. Phi-3.5 and Qwen are open.
# Set HF_TOKEN here (or leave '' and `huggingface-cli login` in a scratch
# cell) before enabling any gated model.
HF_TOKEN = ''

# Per-stage toggles (apply to every enabled model).
RUN_FROZEN = True
RUN_CORPUS = True
RUN_AWQ = True
RUN_BASE_SWEEP = True
# OVERENGINEER IS OFF BY DEFAULT. The q1p5 overengineer run (2026-05-28)
# proved rollout/multi-depth curriculum fine-tuning REGRESSES tau on this
# head + teacher-forced eval combination: every rung dropped tau from the
# apex 7.99 down to ~2-3 while depth-1 acceptance stayed ~99.95%. The eval
# feeds ground-truth captured residuals at every depth, so a pure depth-1
# CE head (what the base sweep produces) already generalizes to all depths;
# training on self-drafted tokens creates a train/eval mismatch that only
# hurts. The base sweep IS the head-production step. Leave this False unless
# the eval methodology changes to true autoregressive (non-teacher-forced).
RUN_OVERENGINEER = False
RUN_EVAL = True
RUN_RUNTIME_PROFILE = True
RUN_EXPORT = True
RUN_HEADBANK_MANIFEST = True

# Two-tier eval for speed without quality loss: rank the sweep variants on a
# cheap window count, then run the FULL eval only on each model's winner. The
# winner's reported metrics + runtime profile come from the full eval; the
# losers only ever get the quick eval (which is plenty to rank them). This
# cuts ~3/4 of the frontier-search cost on the non-winning variants.
QUICK_EVAL_WINDOWS_BIG = 6000
QUICK_EVAL_WINDOWS_SMALL = 3000

# Mining + curriculum knobs (apply to every model's overengineer pass).
CURRICULUM_EPOCHS = 2
MINE_KEEP_FRACTION = 0.25
MINE_MIN_ROWS = 3000
MINE_MAX_ROWS = 10000
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

run([sys.executable, '-u', '-m', 'pip', 'install', '-q',
     'pyarrow>=17', 'tqdm>=4.66', 'zstandard',
     'safetensors>=0.4', 'transformers>=4.45',
     'datasets>=2.18', 'accelerate>=0.32'])

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
    """# Cell 2 - Per-model registry + shared helpers

# Capture layer heuristic: ~85% of total layer count (penultimate-zone capture
# is where the residual is most predictive of next-token in practice).
#
# base_tps_placeholder is what the offline projection assumes for the model
# on the target laptop. Local benchmarks should ground-truth these.

MODELS = {
    'q05b': {
        'enabled': RUN_Q05B,
        'hf_id': 'Qwen/Qwen2.5-0.5B-Instruct',
        'arch': 'qwen2',
        'capture_layer': 20,
        'corpus_max_sequences': 1500,
        'awq_calibrate_mode': 'adaptive-alpha',
        'base_tps_placeholder': 140.0,
        'spec_efficiency_placeholder': 0.80,
        'corpus_max_row_tokens': 384,
        'frontier_max_depth': 24,
        'train_batch_size_big': 64,
        'train_batch_size_small': 24,
        'capture_batch_size': 8,
        'gguf_name': 'qwen2.5-0.5b-instruct-q4_k_m.gguf',
        'profile_name': 'qwen05b-instruct-q4k.m3pro18.json',
    },
    'q3b': {
        'enabled': RUN_Q3B,
        'hf_id': 'Qwen/Qwen2.5-3B-Instruct',
        'arch': 'qwen2',
        'capture_layer': 30,
        'corpus_max_sequences': 2500,
        'awq_calibrate_mode': 'adaptive-alpha',
        'base_tps_placeholder': 65.0,
        'spec_efficiency_placeholder': 0.80,
        'corpus_max_row_tokens': 384,
        'frontier_max_depth': 24,
        'train_batch_size_big': 48,
        'train_batch_size_small': 16,
        'capture_batch_size': 4,
        'gguf_name': 'qwen2.5-3b-instruct-q4_k_m.gguf',
        'profile_name': 'qwen3b-instruct-q4k.m3pro18.json',
    },
    'q7b': {
        'enabled': RUN_Q7B,
        'hf_id': 'Qwen/Qwen2.5-7B-Instruct',
        'arch': 'qwen2',
        'capture_layer': 24,
        'corpus_max_sequences': 2000,
        'awq_calibrate_mode': 'adaptive-alpha',
        'base_tps_placeholder': 35.0,
        'spec_efficiency_placeholder': 0.80,
        'corpus_max_row_tokens': 384,
        'frontier_max_depth': 24,
        'train_batch_size_big': 32,
        'train_batch_size_small': 12,
        'capture_batch_size': 2,
        'gguf_name': 'qwen2.5-7b-instruct-q4_k_m.gguf',
        'profile_name': 'qwen7b-instruct-q4k.m3pro18.json',
    },
    'dsv2': {
        'enabled': RUN_DSV2,
        'hf_id': 'deepseek-ai/DeepSeek-V2-Lite-Chat',
        'arch': 'deepseek2',
        'capture_layer': 22,
        'corpus_max_sequences': 1500,
        'awq_calibrate_mode': 'adaptive-alpha',
        'base_tps_placeholder': 30.0,
        'spec_efficiency_placeholder': 0.80,
        'corpus_max_row_tokens': 384,
        'frontier_max_depth': 24,
        'train_batch_size_big': 32,
        'train_batch_size_small': 12,
        'capture_batch_size': 2,
        'gguf_name': 'deepseek-v2-lite-q4_k_m.gguf',
        'profile_name': 'deepseek-v2-lite-q4.m3pro18.json',
    },
    # ── Rust-wired but UNVERIFIED archs (no GGUF smoke run yet). Heads train
    #    fine here (pure PyTorch); local serving waits on dismantle-core's
    #    smoke tests. Capture layers follow the ~85%-of-depth heuristic.
    'llama32_3b': {
        'enabled': RUN_LLAMA32_3B,
        'hf_id': 'meta-llama/Llama-3.2-3B-Instruct',
        'arch': 'llama',
        'verified': False,
        'gated': True,
        'capture_layer': 24,          # Llama-3.2-3B has 28 layers
        'corpus_max_sequences': 2500,
        'awq_calibrate_mode': 'adaptive-alpha',
        'base_tps_placeholder': 60.0,
        'spec_efficiency_placeholder': 0.80,
        'corpus_max_row_tokens': 384,
        'frontier_max_depth': 24,
        'train_batch_size_big': 48,
        'train_batch_size_small': 16,
        'capture_batch_size': 4,
        'gguf_name': 'llama-3.2-3b-instruct-q4_k_m.gguf',
        'profile_name': 'llama32-3b-instruct-q4k.m3pro18.json',
    },
    'mistral7b': {
        'enabled': RUN_MISTRAL7B,
        'hf_id': 'mistralai/Mistral-7B-Instruct-v0.3',
        'arch': 'llama',              # served by the LlamaDense engine
        'verified': False,
        'gated': True,
        'capture_layer': 27,          # Mistral-7B has 32 layers
        'corpus_max_sequences': 2000,
        'awq_calibrate_mode': 'adaptive-alpha',
        'base_tps_placeholder': 35.0,
        'spec_efficiency_placeholder': 0.80,
        'corpus_max_row_tokens': 384,
        'frontier_max_depth': 24,
        'train_batch_size_big': 32,
        'train_batch_size_small': 12,
        'capture_batch_size': 2,
        'gguf_name': 'mistral-7b-instruct-v0.3-q4_k_m.gguf',
        'profile_name': 'mistral7b-instruct-q4k.m3pro18.json',
    },
    'gemma2_2b': {
        'enabled': RUN_GEMMA2_2B,
        'hf_id': 'google/gemma-2-2b-it',
        'arch': 'gemma2',
        'verified': False,
        'gated': True,
        'capture_layer': 22,          # Gemma-2-2B has 26 layers
        'corpus_max_sequences': 2000,
        'awq_calibrate_mode': 'adaptive-alpha',
        'base_tps_placeholder': 70.0,
        'spec_efficiency_placeholder': 0.80,
        'corpus_max_row_tokens': 384,
        'frontier_max_depth': 24,
        'train_batch_size_big': 48,
        'train_batch_size_small': 16,
        'capture_batch_size': 4,
        'gguf_name': 'gemma-2-2b-it-q4_k_m.gguf',
        'profile_name': 'gemma2-2b-it-q4k.m3pro18.json',
    },
    'phi35_mini': {
        'enabled': RUN_PHI35_MINI,
        'hf_id': 'microsoft/Phi-3.5-mini-instruct',
        'arch': 'phi3',
        'verified': False,
        'gated': False,               # Phi-3.5 is open
        'capture_layer': 27,          # Phi-3.5-mini has 32 layers
        'corpus_max_sequences': 2000,
        'awq_calibrate_mode': 'adaptive-alpha',
        'base_tps_placeholder': 55.0,
        'spec_efficiency_placeholder': 0.80,
        'corpus_max_row_tokens': 384,
        'frontier_max_depth': 24,
        'train_batch_size_big': 48,
        'train_batch_size_small': 16,
        'capture_batch_size': 4,
        'gguf_name': 'phi-3.5-mini-instruct-q4_k_m.gguf',
        'profile_name': 'phi35-mini-instruct-q4k.m3pro18.json',
    },
}

# Decorate every entry with derived paths so the rest of the notebook just
# reads from MODELS[slug][...]. Verified/gated default sensibly for the
# original qwen2/deepseek2 entries that don't set them explicitly.
for slug, m in MODELS.items():
    m['slug'] = slug
    m.setdefault('verified', True)
    m.setdefault('gated', False)
    m['model_root'] = LAB_ROOT / slug
    m['frozen_path']  = m['model_root'] / 'frozen.npz'
    m['corpus_dir']   = m['model_root'] / 'corpus'
    m['awq_dir']      = m['model_root'] / 'awq'
    m['ckpt_root']    = m['model_root'] / 'checkpoints'
    m['eval_root']    = m['model_root'] / 'eval'
    m['profile_dir']  = m['model_root'] / 'runtime_profiles'
    m['leaderboard']  = m['model_root'] / 'leaderboard.json'

TAU_DEPTH = 8
FRONTIER_DEPTHS = '2,4,6,8,12,16,24'
FRONTIER_WIDTHS = '2,3,4,6,8'

# Locked env blocks per arch. The runtime profile writer merges the right one.
QWEN_LOCKED_ENV = {
    'DISMANTLE_QWEN_TCB': '1',
    'DISMANTLE_QWEN_VOCAB_PRUNE': '32000',
    'DISMANTLE_QWEN_Q4K_LMHEAD': '1',
    'DISMANTLE_QWEN_FFN_DOWN_Q4K': '1',
    'DISMANTLE_QWEN_Q4K_PREDEC': '1',
}
DEEPSEEK_LOCKED_ENV = {
    'DISMANTLE_DSV2_TCB': '1',
    'DISMANTLE_DSV2_Q8KV': '1',
}
# llama / gemma2 / phi3 run the Metal-hybrid path in dismantle-core today
# (no TCB+predec arena port yet), so there are no model-specific lock flags
# to bake in. Left empty deliberately; the EAGLE5_* policy hints from the
# frontier eval still get merged into runtime_env. When the Rust side ports
# the fast arena path for these archs, add the flags here.
LLAMA_LOCKED_ENV = {}
GEMMA2_LOCKED_ENV = {}
PHI3_LOCKED_ENV = {}
LOCKED_ENV_BY_ARCH = {
    'qwen2': QWEN_LOCKED_ENV,
    'deepseek2': DEEPSEEK_LOCKED_ENV,
    'llama': LLAMA_LOCKED_ENV,
    'gemma2': GEMMA2_LOCKED_ENV,
    'phi3': PHI3_LOCKED_ENV,
}


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


def leaderboard_sort_key(row):
    return (
        float(row.get('offline_projected_tps') or 0.0),
        float(row.get('accepted_draft_tokens_per_verify') or 0.0),
        float(row.get('tau') or 0.0),
    )


def merge_model_leaderboard(slug, rows_to_merge):
    if not rows_to_merge:
        return []
    lb_path = MODELS[slug]['leaderboard']
    existing_rows = load_json(lb_path, {}).get('rows', [])
    by_key = {(r.get('target'), r.get('tag')): r for r in existing_rows}
    for row in rows_to_merge:
        by_key[(row.get('target'), row.get('tag'))] = row
    merged = sorted(by_key.values(), key=leaderboard_sort_key, reverse=True)
    write_json_atomic(lb_path, {'schema': 'dismantle-headbank-leaderboard-v1', 'rows': merged})
    return merged


def best_resumable_row(slug):
    rows = load_json(MODELS[slug]['leaderboard'], {'rows': []}).get('rows', [])
    candidates = []
    for row in rows:
        head = Path(row.get('head', ''))
        latest = head.parent / 'latest.npz'
        if head.exists() and latest.exists():
            candidates.append(row)
    candidates.sort(key=leaderboard_sort_key, reverse=True)
    if not candidates:
        raise RuntimeError(f'no resumable head for {slug}')
    return candidates[0]


# Authenticate to HuggingFace if a token was supplied — required for the
# gated llama/mistral/gemma models. Also export it so subprocesses inherit.
if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN
    os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        print('[bank] HuggingFace login OK')
    except Exception as e:
        print(f'[bank] HF login skipped/failed: {e}')

# Per-model error isolation. If a model throws in any stage (e.g. dsv2's MoE
# loader chokes), it gets added to FAILED_SLUGS and every later stage skips
# it — the other models still complete, eval, and export. A multi-hour
# unattended run should never lose 3 good models because the 4th failed.
FAILED_SLUGS = set()


def active_slugs():
    return [s for s in ENABLED_SLUGS if s not in FAILED_SLUGS]


def run_stage(label, slug, fn):
    '''Run fn() for one model; on error, log + mark the model failed + continue.'''
    if slug in FAILED_SLUGS:
        print(f'[{label}:{slug}] skipped — model previously failed')
        return None
    try:
        return fn()
    except Exception as e:
        import traceback
        FAILED_SLUGS.add(slug)
        print(f'[{label}:{slug}] FAILED: {e}')
        traceback.print_exc()
        print(f'[{label}:{slug}] marked failed; remaining models continue.')
        return None


ENABLED_SLUGS = [s for s, m in MODELS.items() if m['enabled']]
print(f'[bank] enabled models: {ENABLED_SLUGS}')
for s in ENABLED_SLUGS:
    m = MODELS[s]
    m['model_root'].mkdir(parents=True, exist_ok=True)
    flags = []
    if not m['verified']:
        flags.append('UNVERIFIED-rust')
    if m['gated']:
        flags.append('gated-hf')
    flag_str = (' [' + ','.join(flags) + ']') if flags else ''
    print(f"  {s:11s} hf={m['hf_id']:42s} arch={m['arch']:9s} capture_layer={m['capture_layer']}{flag_str}")
    if m['gated'] and not HF_TOKEN:
        print(f"    WARN {s} is gated on HF but HF_TOKEN is empty — frozen extraction + capture will fail. "
              f"Set HF_TOKEN in Cell 1 or huggingface-cli login.")
"""
)

code(
    """# Cell 3 - Frozen-weights extraction (per model)

# Pulls (token_embd, lm_head, output_norm) from HuggingFace and writes the
# .npz the Eagle5 trainer needs. Reuses colab/build_qwen3b_frozen_hf.py which
# is HF-arch-agnostic via the --model flag.


def extract_frozen(slug):
    m = MODELS[slug]
    if m['frozen_path'].exists() and not False:
        print(f'[frozen:{slug}] exists ({m["frozen_path"].stat().st_size/1e9:.2f}GB); skip')
        return
    cmd = [
        sys.executable, '-u', 'colab/build_qwen3b_frozen_hf.py',
        '--model', m['hf_id'],
        '--out', str(m['frozen_path']),
    ]
    run_with_heartbeat(cmd, label=f'frozen_{slug}', interval_sec=60)
    if not m['frozen_path'].exists():
        raise FileNotFoundError(f'frozen extraction failed for {slug}')


if RUN_FROZEN:
    for slug in ENABLED_SLUGS:
        run_stage('frozen', slug, lambda s=slug: extract_frozen(s))
else:
    print('RUN_FROZEN=False; skipping frozen extraction.')
"""
)

code(
    """# Cell 4 - Corpus capture (per model)

# Uses colab/mega_calibrate.py: captures residual + intermediate at
# capture_layer plus per-channel activation stats used by AWQ. Idempotent
# per-shard so a runtime kill resumes cleanly.


def capture_corpus(slug):
    m = MODELS[slug]
    out_dir = m['corpus_dir']
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / 'capture_done.json'
    if sentinel.exists():
        print(f'[capture:{slug}] sentinel present; skip')
        return
    cmd = [
        sys.executable, '-u', 'colab/mega_calibrate.py',
        '--model', m['hf_id'],
        '--out', str(out_dir),
        '--capture-layer', str(m['capture_layer']),
        '--max-sequences', str(m['corpus_max_sequences']),
        '--batch-size', str(m['capture_batch_size']),
        '--shard-size', '8',
    ]
    run_with_heartbeat(cmd, label=f'capture_{slug}', interval_sec=120)
    # Mark the capture as done so re-runs skip cleanly even if the script
    # didn't drop its own sentinel.
    shards = sorted(out_dir.glob('shard_*.parquet'))
    if not shards:
        raise FileNotFoundError(f'no shards captured for {slug} at {out_dir}')
    write_json_atomic(sentinel, {
        'schema': 'dismantle-headbank-capture-v1',
        'shards': len(shards),
        'capture_layer': m['capture_layer'],
        'hf_id': m['hf_id'],
        'completed_at_unix': int(time.time()),
    })


if RUN_CORPUS:
    for slug in active_slugs():
        run_stage('capture', slug, lambda s=slug: capture_corpus(s))
else:
    print('RUN_CORPUS=False; skipping corpus capture.')
"""
)

code(
    """# Cell 5 - AWQ per-channel calibration (per model)

# Reads per_site_activation_stats.npz from the capture and produces
# adaptive-alpha smoothing factors. The runtime loads these as AWQ scales.


def calibrate_awq(slug):
    m = MODELS[slug]
    m['awq_dir'].mkdir(parents=True, exist_ok=True)
    out_path = m['awq_dir'] / 'awq_smoothing.json'
    if out_path.exists():
        print(f'[awq:{slug}] exists; skip')
        return out_path
    stats_path = m['corpus_dir'] / 'per_site_activation_stats.npz'
    if not stats_path.exists():
        print(f'[awq:{slug}] WARN missing {stats_path}; skipping AWQ for {slug}')
        return None
    cmd = [
        sys.executable, '-u', 'colab/awq_per_channel_calibrate.py',
        '--stats', str(stats_path),
        '--out', str(out_path),
        '--mode', m['awq_calibrate_mode'],
    ]
    run_with_heartbeat(cmd, label=f'awq_{slug}', interval_sec=60)
    return out_path if out_path.exists() else None


AWQ_PATHS = {}
if RUN_AWQ:
    for slug in active_slugs():
        # AWQ is optional polish — a failure here must NOT fail the model.
        try:
            AWQ_PATHS[slug] = calibrate_awq(slug)
        except Exception as e:
            print(f'[awq:{slug}] WARN calibration failed (non-fatal): {e}')
            AWQ_PATHS[slug] = None
else:
    print('RUN_AWQ=False; skipping AWQ calibration.')
    for slug in ENABLED_SLUGS:
        p = MODELS[slug]['awq_dir'] / 'awq_smoothing.json'
        AWQ_PATHS[slug] = p if p.exists() else None

print('AWQ outputs:')
for s, p in AWQ_PATHS.items():
    print(f'  {s}: {p}')
"""
)

code(
    """# Cell 6 - Base head sweep (per model)

# Four architecture variants per model. Picked to match the q1p5 500U
# sweep's most-informative axes:
#   b1_fast    — 1 block, h16, ff_mult=4.0, lr=3e-4   (the q1p5 winner shape)
#   b2_wide    — 2 blocks, h16, ff_mult=6.0, lr=5e-4  (capacity probe)
#   b3_compact — 3 blocks, h16, ff_mult=4.0, lr=3e-4  (depth probe)
#   b1_tiny    — 1 block, h8,  ff_mult=2.0, lr=3e-4   (distill candidate)


SWEEP_VARIANTS = [
    {'name': 'b1_fast',    'num_blocks': 1, 'head_heads': 16, 'head_ff_mult': 4.0, 'lr': 3e-4, 'epochs': 4, 'seed': 0,
     'calib_loss_weight': 0.12, 'residual_delta_loss_weight': 0.000,
     'rollout_loss_weight': 0.0, 'rollout_depth': 1, 'rollout_starts_per_batch': 4,
     'rollout_draft_prob': 0.0, 'rollout_depth_gamma': 0.85},
    {'name': 'b2_wide',    'num_blocks': 2, 'head_heads': 16, 'head_ff_mult': 6.0, 'lr': 5e-4, 'epochs': 4, 'seed': 1,
     'calib_loss_weight': 0.20, 'residual_delta_loss_weight': 0.020,
     'rollout_loss_weight': 0.0, 'rollout_depth': 1, 'rollout_starts_per_batch': 4,
     'rollout_draft_prob': 0.0, 'rollout_depth_gamma': 0.85},
    {'name': 'b3_compact', 'num_blocks': 3, 'head_heads': 16, 'head_ff_mult': 4.0, 'lr': 3e-4, 'epochs': 4, 'seed': 0,
     'calib_loss_weight': 0.30, 'residual_delta_loss_weight': 0.030,
     'rollout_loss_weight': 0.0, 'rollout_depth': 1, 'rollout_starts_per_batch': 4,
     'rollout_draft_prob': 0.0, 'rollout_depth_gamma': 0.85},
    {'name': 'b1_tiny',    'num_blocks': 1, 'head_heads': 8,  'head_ff_mult': 2.0, 'lr': 3e-4, 'epochs': 4, 'seed': 2,
     'calib_loss_weight': 0.12, 'residual_delta_loss_weight': 0.000,
     'rollout_loss_weight': 0.0, 'rollout_depth': 1, 'rollout_starts_per_batch': 4,
     'rollout_draft_prob': 0.0, 'rollout_depth_gamma': 0.85},
]


def train_base_variant(slug, variant):
    m = MODELS[slug]
    tag = (
        f"{slug}_{variant['name']}_b{variant['num_blocks']}_h{variant['head_heads']}"
        f"_ff{int(variant['head_ff_mult']*10):02d}_lr{int(variant['lr']*10000):04d}"
        f"_seed{variant['seed']}"
    )
    ckpt_dir = m['ckpt_root'] / tag
    head = ckpt_dir / 'head_final.safetensors'
    if head.exists():
        print(f'[sweep:{slug}] skip existing {head}')
        return head, tag, ckpt_dir
    batch = m['train_batch_size_big'] if BIG_GPU else m['train_batch_size_small']
    cmd = [
        sys.executable, '-u', 'colab/eagle5_train_pytorch.py',
        '--corpus-dir', str(m['corpus_dir']),
        '--frozen', str(m['frozen_path']),
        '--ckpt-dir', str(ckpt_dir),
        '--epochs', str(variant['epochs']),
        '--batch-size', str(batch),
        '--seq-len', '16',
        '--lr', str(variant['lr']),
        '--num-blocks', str(variant['num_blocks']),
        '--head-heads', str(variant['head_heads']),
        '--head-ff-mult', str(variant['head_ff_mult']),
        '--capture-layer', str(m['capture_layer']),
        '--max-rows', '24000',
        '--max-row-tokens', str(m['corpus_max_row_tokens']),
        '--sparsity-head', 'off',
        '--seed', str(variant['seed']),
        '--calib-loss-weight', str(variant['calib_loss_weight']),
        '--residual-delta-loss-weight', str(variant['residual_delta_loss_weight']),
        '--rollout-loss-weight', str(variant['rollout_loss_weight']),
        '--rollout-depth', str(variant['rollout_depth']),
        '--rollout-starts-per-batch', str(variant['rollout_starts_per_batch']),
        '--rollout-draft-prob', str(variant['rollout_draft_prob']),
        '--rollout-depth-gamma', str(variant['rollout_depth_gamma']),
        '--save-safetensors',
    ]
    print(f'\\n=== [sweep:{slug}] train {tag} batch={batch}')
    run_with_heartbeat(cmd, label=f'sweep_{slug}_{variant["name"]}', interval_sec=60)
    if not head.exists():
        raise FileNotFoundError(f'sweep head missing for {tag}')
    return head, tag, ckpt_dir


BASE_HEADS = {slug: [] for slug in ENABLED_SLUGS}
if RUN_BASE_SWEEP:
    for slug in active_slugs():
        def _sweep(s=slug):
            for variant in SWEEP_VARIANTS:
                head, tag, ckpt_dir = train_base_variant(s, variant)
                BASE_HEADS[s].append({'head': head, 'tag': tag, 'ckpt_dir': ckpt_dir, 'variant': variant['name']})
        run_stage('sweep', slug, _sweep)
else:
    print('RUN_BASE_SWEEP=False; collecting existing base heads only.')
    for slug in ENABLED_SLUGS:
        for variant in SWEEP_VARIANTS:
            tag = (
                f"{slug}_{variant['name']}_b{variant['num_blocks']}_h{variant['head_heads']}"
                f"_ff{int(variant['head_ff_mult']*10):02d}_lr{int(variant['lr']*10000):04d}"
                f"_seed{variant['seed']}"
            )
            head = MODELS[slug]['ckpt_root'] / tag / 'head_final.safetensors'
            if head.exists():
                BASE_HEADS[slug].append({'head': head, 'tag': tag,
                                         'ckpt_dir': MODELS[slug]['ckpt_root'] / tag,
                                         'variant': variant['name']})

for s in ENABLED_SLUGS:
    print(f'[sweep:{s}] heads: {[h["tag"] for h in BASE_HEADS[s]]}')
"""
)

code(
    """# Cell 7 - τ + frontier eval for every base head + leaderboard merge

# Each head gets a tau.json and frontier.json. The leaderboard per-model
# decides which head feeds the overengineer pass.

from safetensors import safe_open


def _read_head_meta(head_path):
    try:
        with safe_open(str(head_path), framework='pt', device='cpu') as f:
            return f.metadata() or {}
    except Exception as e:
        print(f'[eval] WARN metadata read failed for {head_path}: {e}')
        return {}


def eval_head(slug, head_path, *, source_tag=None, quick=False):
    '''Evaluate a head. quick=True ranks cheaply (fewer windows, no lattice
    search); quick=False is the full eval that feeds the runtime profile.'''
    m = MODELS[slug]
    head_path = Path(head_path)
    tag = head_path.parent.name
    out_dir = m['eval_root'] / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = '_quick' if quick else ''
    tau_path = out_dir / f'tau{suffix}.json'
    frontier_path = out_dir / f'frontier{suffix}.json'
    meta = _read_head_meta(head_path)
    nb = meta.get('num_blocks', '1')
    hh = meta.get('n_heads', '16')
    ff = meta.get('ff_mult', '4.0')

    if quick:
        windows = QUICK_EVAL_WINDOWS_BIG if BIG_GPU else QUICK_EVAL_WINDOWS_SMALL
        # Cheap frontier: skip the lattice-width sweep, narrow the depth grid.
        front_depths = '2,4,8,16,24'
        front_widths = '2'
    else:
        windows = 24000 if BIG_GPU else 6000
        front_depths = FRONTIER_DEPTHS
        front_widths = FRONTIER_WIDTHS

    if not tau_path.exists():
        run_with_heartbeat([
            sys.executable, 'colab/eagle5_tau_eval_pytorch.py',
            '--ckpt', str(head_path),
            '--frozen', str(m['frozen_path']),
            '--corpus', str(m['corpus_dir']),
            '--out', str(tau_path),
            '--depth', str(TAU_DEPTH),
            '--max-windows', str(windows),
            '--max-row-tokens', str(m['corpus_max_row_tokens']),
            '--num-blocks', str(nb),
            '--head-heads', str(hh),
            '--head-ff-mult', str(ff),
            '--base-tps', str(m['base_tps_placeholder']),
            '--w4a8-multiplier', '1.0',
            '--spec-efficiency', str(m['spec_efficiency_placeholder']),
        ], label=f"eval-tau{suffix}-{slug}-{tag}", interval_sec=60)
    if not frontier_path.exists():
        run_with_heartbeat([
            sys.executable, 'colab/eagle5_frontier_policy.py',
            '--ckpt', str(head_path),
            '--frozen', str(m['frozen_path']),
            '--corpus', str(m['corpus_dir']),
            '--out', str(frontier_path),
            '--max-depth', str(m['frontier_max_depth']),
            '--depths', front_depths,
            '--lattice-widths', front_widths,
            '--max-windows', str(windows),
            '--max-row-tokens', str(m['corpus_max_row_tokens']),
            '--eval-batch-size', '192',
            '--num-blocks', str(nb),
            '--head-heads', str(hh),
            '--head-ff-mult', str(ff),
            '--base-tps', str(m['base_tps_placeholder']),
            '--w4a8-multiplier', '1.0',
            '--spec-efficiency', str(m['spec_efficiency_placeholder']),
        ], label=f"eval-frontier{suffix}-{slug}-{tag}", interval_sec=60)
    tau = load_json(tau_path, {})
    frontier = load_json(frontier_path, {})
    best = frontier.get('policies', {}).get('best_deployable', {})
    return {
        'target': slug,
        'tag': tag,
        'head': str(head_path),
        'tau_path': str(tau_path),
        'frontier_path': str(frontier_path),
        'tau': tau.get('tau'),
        'depth1_accept_rate': tau.get('depth1_accept_rate'),
        'best_deployable': best,
        'offline_projected_tps': best.get('projected_dec_tps', 0.0),
        'accepted_draft_tokens_per_verify': best.get('accepted_draft_tokens_per_verify', 0.0),
        'policy_kind': best.get('kind'),
        'metadata': meta,
        'source_tag': source_tag,
        'eval_tier': 'quick' if quick else 'full',
    }


def _eval_model(slug):
    entries = BASE_HEADS[slug]
    if not entries:
        print(f'[eval:{slug}] no base heads; skip')
        return
    # Tier 1: quick-eval every variant to rank them cheaply.
    quick_rows = []
    for entry in entries:
        quick_rows.append(eval_head(slug, entry['head'], source_tag=entry['tag'], quick=True))
    quick_rows.sort(key=leaderboard_sort_key, reverse=True)
    winner = quick_rows[0]
    print(f"[eval:{slug}] quick-rank winner = {winner['tag']} "
          f"(tau={winner.get('tau')}, tps={winner.get('offline_projected_tps')})")
    # Tier 2: full-eval only the winner. Its full row overwrites its quick
    # row in the leaderboard (same target+tag key); losers keep quick rows.
    full_winner = eval_head(slug, winner['head'], source_tag=winner.get('source_tag'), quick=False)
    merged = merge_model_leaderboard(slug, quick_rows + [full_winner])
    if merged:
        top = merged[0]
        print(f"[eval:{slug}] leaderboard top: {top.get('tag')} tau={top.get('tau')} "
              f"tps={top.get('offline_projected_tps')} tier={top.get('eval_tier')}")


if RUN_EVAL:
    for slug in active_slugs():
        run_stage('eval', slug, lambda s=slug: _eval_model(s))
else:
    print('RUN_EVAL=False; skipping base-sweep eval.')
"""
)

code(
    """# Cell 8 - Overengineer pass (mine + curriculum + calib) for each model

# Same pattern as the standalone overengineer notebook, looped over every
# model. For each model:
#   1. Pick the current leaderboard winner.
#   2. Mine hard negatives against it.
#   3. Run a 3-rung multi-depth curriculum (rung_i re-mines against rung_{i-1}).
#   4. Train a calibration-heavy variant on the latest mine.
#   5. Eval each new head and merge into the per-model leaderboard.


CURRICULUM_RUNGS = [
    {'name': 'rung1_d2_p050_w006_lr5e-5', 'lr': 5e-5, 'rollout_loss_weight': 0.06, 'rollout_depth': 2,
     'rollout_starts_per_batch': 4, 'rollout_draft_prob': 0.50, 'rollout_depth_gamma': 0.95,
     'rollout_depth_targets': '1,2', 'rollout_depth_target_weights': '1.0,0.8',
     'calib_loss_weight': 0.12, 'residual_delta_loss_weight': 0.010},
    {'name': 'rung2_d4_p070_w010_lr4e-5', 'lr': 4e-5, 'rollout_loss_weight': 0.10, 'rollout_depth': 4,
     'rollout_starts_per_batch': 4, 'rollout_draft_prob': 0.70, 'rollout_depth_gamma': 0.93,
     'rollout_depth_targets': '1,2,4', 'rollout_depth_target_weights': '1.0,0.7,0.5',
     'calib_loss_weight': 0.12, 'residual_delta_loss_weight': 0.012},
    {'name': 'rung3_d8_p085_w015_lr3e-5', 'lr': 3e-5, 'rollout_loss_weight': 0.15, 'rollout_depth': 8,
     'rollout_starts_per_batch': 3, 'rollout_draft_prob': 0.85, 'rollout_depth_gamma': 0.90,
     'rollout_depth_targets': '1,2,4,8', 'rollout_depth_target_weights': '1.0,0.7,0.5,0.3',
     'calib_loss_weight': 0.14, 'residual_delta_loss_weight': 0.014},
]
CALIB_SPEC = {
    'name': 'calib_heavy_w030_d4_p070_lr3e-5', 'lr': 3e-5, 'epochs': max(2, CURRICULUM_EPOCHS),
    'rollout_loss_weight': 0.08, 'rollout_depth': 4, 'rollout_starts_per_batch': 4,
    'rollout_draft_prob': 0.70, 'rollout_depth_gamma': 0.92,
    'rollout_depth_targets': '1,2,4', 'rollout_depth_target_weights': '1.0,0.6,0.4',
    'calib_loss_weight': 0.30, 'residual_delta_loss_weight': 0.010,
}


def mine_hard_negatives(slug, head_path, mine_slug, *, force=False):
    m = MODELS[slug]
    head_path = Path(head_path)
    mine_dir = m['model_root'] / 'hardneg' / mine_slug
    manifest_path = mine_dir / 'mine_manifest.json'
    if manifest_path.exists() and not force:
        print(f'[mine:{slug}] reuse {mine_dir}')
        return mine_dir
    meta = _read_head_meta(head_path)
    cmd = [
        sys.executable, '-u', 'colab/eagle5_hard_neg_miner.py',
        '--ckpt', str(head_path),
        '--frozen', str(m['frozen_path']),
        '--corpus-dir', str(m['corpus_dir']),
        '--out-dir', str(mine_dir),
        '--keep-fraction', str(MINE_KEEP_FRACTION),
        '--keep-min-rows', str(MINE_MIN_ROWS),
        '--keep-max-rows', str(MINE_MAX_ROWS),
        '--shards-to-scan', '0',
        '--rows-per-output-shard', '200',
        '--max-row-tokens', str(m['corpus_max_row_tokens']),
        '--score', 'depth1_miss',
        '--num-blocks', str(meta.get('num_blocks', '1')),
        '--head-heads', str(meta.get('n_heads', '16')),
        '--head-ff-mult', str(meta.get('ff_mult', '4.0')),
        '--device', 'cuda',
        '--seed', '0',
    ]
    run_with_heartbeat(cmd, label=f'mine_{slug}_{mine_slug}', interval_sec=120)
    return mine_dir


def train_overeng(slug, name, warm_dir, corpus_dir, spec, base_meta, base_hash, *, epochs=None):
    m = MODELS[slug]
    tag = f'{slug}_overeng_{name}_from_{base_hash}'
    ckpt_dir = m['ckpt_root'] / tag
    head = ckpt_dir / 'head_final.safetensors'
    if head.exists():
        print(f'[overeng:{slug}] skip {head}')
        return head, ckpt_dir, tag
    copy_warm_start(warm_dir, ckpt_dir)
    batch = m['train_batch_size_big'] if BIG_GPU else m['train_batch_size_small']
    cmd = [
        sys.executable, '-u', 'colab/eagle5_train_pytorch.py',
        '--corpus-dir', str(corpus_dir),
        '--frozen', str(m['frozen_path']),
        '--ckpt-dir', str(ckpt_dir),
        '--epochs', str(epochs if epochs is not None else CURRICULUM_EPOCHS),
        '--batch-size', str(batch),
        '--seq-len', '16',
        '--lr', str(spec['lr']),
        '--num-blocks', str(base_meta['nb']),
        '--head-heads', str(base_meta['hh']),
        '--head-ff-mult', str(base_meta['ff']),
        '--capture-layer', str(m['capture_layer']),
        '--max-rows', '18000',
        '--max-row-tokens', str(m['corpus_max_row_tokens']),
        '--sparsity-head', 'off',
        '--seed', str(9000 + hash(name) % 1000),
        '--calib-loss-weight', str(spec['calib_loss_weight']),
        '--residual-delta-loss-weight', str(spec['residual_delta_loss_weight']),
        '--rollout-loss-weight', str(spec['rollout_loss_weight']),
        '--rollout-depth', str(spec['rollout_depth']),
        '--rollout-starts-per-batch', str(spec['rollout_starts_per_batch']),
        '--rollout-draft-prob', str(spec['rollout_draft_prob']),
        '--rollout-depth-gamma', str(spec['rollout_depth_gamma']),
        '--rollout-depth-targets', str(spec.get('rollout_depth_targets', '')),
        '--rollout-depth-target-weights', str(spec.get('rollout_depth_target_weights', '')),
        '--save-safetensors',
    ]
    print(f'\\n=== [overeng:{slug}] {tag}')
    run_with_heartbeat(cmd, label=f'overeng_{slug}_{name}', interval_sec=60)
    if not head.exists():
        raise FileNotFoundError(f'overeng head missing for {tag}')
    return head, ckpt_dir, tag


OVERENG_HEADS = {slug: [] for slug in ENABLED_SLUGS}
if RUN_OVERENGINEER:
    for slug in ENABLED_SLUGS:
        try:
            base_row = best_resumable_row(slug)
        except RuntimeError as e:
            print(f'[overeng:{slug}] WARN {e}; skip')
            continue
        base_head = Path(base_row['head'])
        base_tag = base_head.parent.name
        base_hash = hashlib.sha1(base_tag.encode()).hexdigest()[:8]
        meta = _read_head_meta(base_head)
        base_meta = {
            'nb': int(meta.get('num_blocks', '1')),
            'hh': int(meta.get('n_heads', '16')),
            'ff': float(meta.get('ff_mult', '4.0')),
        }
        current_mine = mine_hard_negatives(slug, base_head, f'from_{base_hash}')
        warm_dir = base_head.parent
        for idx, rung in enumerate(CURRICULUM_RUNGS, start=1):
            head, ckpt_dir, tag = train_overeng(slug, rung['name'], warm_dir, current_mine, rung, base_meta, base_hash)
            OVERENG_HEADS[slug].append({'head': head, 'ckpt_dir': ckpt_dir, 'tag': tag,
                                        'rung': rung['name'], 'mine_dir': str(current_mine)})
            warm_dir = ckpt_dir
            if REMINE_BETWEEN_RUNGS and idx < len(CURRICULUM_RUNGS):
                slug_mine = f'from_{base_hash}_after_rung{idx}'
                try:
                    current_mine = mine_hard_negatives(slug, head, slug_mine)
                except Exception as e:
                    print(f'[overeng:{slug}] remine after rung {idx} failed: {e}; reusing prior mine')
        # Calibration-heavy variant on the freshest mine.
        head, ckpt_dir, tag = train_overeng(
            slug, CALIB_SPEC['name'], warm_dir, current_mine, CALIB_SPEC, base_meta, base_hash,
            epochs=CALIB_SPEC['epochs'])
        OVERENG_HEADS[slug].append({'head': head, 'ckpt_dir': ckpt_dir, 'tag': tag,
                                    'rung': CALIB_SPEC['name'], 'mine_dir': str(current_mine)})

# Eval every overengineer head and merge.
if RUN_EVAL:
    for slug in ENABLED_SLUGS:
        rows = []
        for entry in OVERENG_HEADS[slug]:
            row = eval_head(slug, entry['head'], source_tag=entry['tag'])
            row['overengineer_rung'] = entry['rung']
            row['mine_dir'] = entry['mine_dir']
            rows.append(row)
        merged = merge_model_leaderboard(slug, rows)
        if merged:
            top = merged[0]
            print(f"[overeng:{slug}] top now: {top.get('tag')} tau={top.get('tau')} tps={top.get('offline_projected_tps')}")
"""
)

code(
    """# Cell 9 - Runtime profile JSON for each model's leaderboard winner


def _normalize_for_env(v):
    if isinstance(v, bool): return '1' if v else '0'
    if isinstance(v, (int, float)): return str(v)
    return str(v)


def export_runtime_profile(slug):
    m = MODELS[slug]
    rows = load_json(m['leaderboard'], {'rows': []}).get('rows', [])
    rows = [r for r in rows if r.get('head') and r.get('frontier_path')]
    if not rows:
        print(f'[profile:{slug}] no eligible row; skip')
        return None
    rows.sort(key=leaderboard_sort_key, reverse=True)
    row = rows[0]
    frontier = load_json(row['frontier_path'], {})
    hints = frontier.get('runtime_hints', {}) or {}
    best = frontier.get('policies', {}).get('best_deployable', {}) or {}

    locked_env = dict(LOCKED_ENV_BY_ARCH.get(m['arch'], {}))
    runtime_env = dict(locked_env)
    runtime_env['EAGLE5_HEAD'] = str(row['head'])
    if AWQ_PATHS.get(slug):
        runtime_env['DISMANTLE_AWQ_SCALES'] = str(AWQ_PATHS[slug])
    for key in ('variable_k', 'entropy_routing', 'draft_lattice'):
        block = hints.get(key, {}).get('env') or {}
        for k, v in block.items():
            runtime_env[k] = _normalize_for_env(v)
    if best.get('kind') == 'fixed_k':
        runtime_env.pop('DISMANTLE_EAGLE5_VARIABLE_K', None)
        runtime_env.pop('DISMANTLE_EAGLE5_CONF_THRESH', None)
        if best.get('max_depth') is not None:
            runtime_env['DISMANTLE_EAGLE5_FIXED_K'] = str(best['max_depth'])

    payload = {
        'schema': 'dismantle-eagle5-runtime-profile-v1',
        'created_at_unix': int(time.time()),
        'repo_sha': HEAD_SHA,
        'target': slug,
        'hf_id': m['hf_id'],
        'arch': m['arch'],
        'rust_serving_verified': m['verified'],
        'gated': m['gated'],
        'gguf_name': m['gguf_name'],
        'profile_name': m['profile_name'],
        'tag': row.get('tag'),
        'head': row.get('head'),
        'head_sha256': sha256_file(row['head']) if Path(row['head']).is_file() else None,
        'awq_scales': str(AWQ_PATHS.get(slug)) if AWQ_PATHS.get(slug) else None,
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
        'locked_env': locked_env,
    }
    m['profile_dir'].mkdir(parents=True, exist_ok=True)
    safe_tag = ''.join(c if c.isalnum() or c in '._-+' else '_' for c in str(row.get('tag') or 'unknown'))
    per_tag = m['profile_dir'] / f'{safe_tag}.runtime.json'
    winner  = m['profile_dir'] / f'{slug}_winner.runtime.json'
    write_json_atomic(per_tag, payload)
    write_json_atomic(winner, payload)
    print(f"[profile:{slug}] wrote {per_tag} and {winner}")
    return {'payload': payload, 'per_tag': per_tag, 'winner': winner}


WINNER_PROFILES = {}
if RUN_RUNTIME_PROFILE:
    for slug in active_slugs():
        WINNER_PROFILES[slug] = run_stage('profile', slug, lambda s=slug: export_runtime_profile(s))
else:
    print('RUN_RUNTIME_PROFILE=False; skipping runtime profile export.')
"""
)

code(
    """# Cell 10 - Aggregate head bank manifest

# Indexes every enabled model with its winner head, AWQ scales, metrics, and
# runtime profile path. This is what `tools/headbank/pull.sh` reads.


def build_headbank_manifest():
    entries = []
    for slug in ENABLED_SLUGS:
        wp = WINNER_PROFILES.get(slug)
        if not wp:
            continue
        p = wp['payload']
        entries.append({
            'slug': slug,
            'hf_id': p['hf_id'],
            'arch': p['arch'],
            'rust_serving_verified': p.get('rust_serving_verified', True),
            'gated': p.get('gated', False),
            'gguf_name': p['gguf_name'],
            'profile_name': p['profile_name'],
            'head_path': p['head'],
            'head_sha256': p['head_sha256'],
            'awq_scales': p.get('awq_scales'),
            'runtime_profile': str(wp['winner']),
            'metrics': p['metrics'],
        })
    manifest = {
        'schema': 'dismantle-headbank-manifest-v1',
        'created_at_unix': int(time.time()),
        'repo_sha': HEAD_SHA,
        'lab_root': str(LAB_ROOT),
        'export_root': str(EXPORT_ROOT),
        'entries': entries,
    }
    path = LAB_ROOT / 'headbank_manifest.json'
    write_json_atomic(path, manifest)
    print(f'[headbank] wrote {path}')
    for e in entries:
        m = e['metrics']
        print(f"  {e['slug']:5s} tps={m.get('offline_projected_tps'):.0f} tau={m.get('tau'):.2f} head={e['head_path']}")
    return path


HEADBANK_MANIFEST_PATH = None
if RUN_HEADBANK_MANIFEST:
    HEADBANK_MANIFEST_PATH = build_headbank_manifest()
"""
)

code(
    """# Cell 11 - Safe export of the head bank to Drive

# Mirrors:
#   * leaderboard.json per model
#   * head safetensors per model (winner only)
#   * eval JSONs for the winner
#   * runtime profile JSONs
#   * AWQ scales
#   * mine manifests (audit trail)
#   * top-level headbank_manifest.json
#
# Skips large corpora and intermediate base-sweep heads to keep the export
# slim and downloadable.


def _copy_file(src, dst, key, manifest, sha=False, retries=3):
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
            if tmp.exists(): tmp.unlink()
            with open(src, 'rb') as fsrc, open(tmp, 'wb') as fdst:
                shutil.copyfileobj(fsrc, fdst, length=16 * 1024 * 1024)
                fdst.flush(); os.fsync(fdst.fileno())
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
                if tmp.exists(): tmp.unlink()
            except Exception: pass
            if getattr(e, 'errno', None) == 107 or 'Transport endpoint is not connected' in str(e):
                remount_drive_for_export()
            time.sleep(min(20, 2 * attempt))
    manifest['missing'][key] = str(src)
    manifest['copy_errors'][key] = last_error or 'unknown copy error'
    return None


if RUN_EXPORT:
    manifest = {
        'schema': 'dismantle-headbank-export-v1',
        'created_at_unix': int(time.time()),
        'repo_sha': HEAD_SHA,
        'lab_root': str(LAB_ROOT),
        'export_root': str(EXPORT_ROOT),
        'files': {},
        'missing': {},
    }
    if HEADBANK_MANIFEST_PATH and HEADBANK_MANIFEST_PATH.exists():
        _copy_file(HEADBANK_MANIFEST_PATH, EXPORT_ROOT / 'headbank_manifest.json',
                   'headbank_manifest', manifest)
    for slug in ENABLED_SLUGS:
        m = MODELS[slug]
        wp = WINNER_PROFILES.get(slug)
        slug_root = EXPORT_ROOT / slug
        # leaderboard per model
        if m['leaderboard'].exists():
            _copy_file(m['leaderboard'], slug_root / 'leaderboard.json',
                       f'{slug}/leaderboard', manifest)
        # AWQ scales
        if AWQ_PATHS.get(slug):
            _copy_file(AWQ_PATHS[slug], slug_root / 'awq' / 'awq_smoothing.json',
                       f'{slug}/awq', manifest)
        # winner head + eval + runtime profiles
        if wp:
            p = wp['payload']
            tag = p['tag']
            safe_tag = ''.join(c if c.isalnum() or c in '._-+' else '_' for c in str(tag))
            _copy_file(p['head'], slug_root / 'heads' / f'{safe_tag}.safetensors',
                       f'{slug}/heads/{tag}', manifest, sha=True)
            for label, src_path in wp.items():
                if label == 'payload': continue
                _copy_file(src_path, slug_root / 'runtime_profiles' / Path(src_path).name,
                           f'{slug}/runtime_profiles/{Path(src_path).name}', manifest)
            for k in ('tau_source', 'frontier_source'):
                if p.get(k):
                    _copy_file(p[k], slug_root / 'eval' / safe_tag / Path(p[k]).name,
                               f'{slug}/eval/{tag}/{Path(p[k]).name}', manifest)
        # mine manifests
        for mp in (m['model_root'] / 'hardneg').glob('*/mine_manifest.json') if (m['model_root'] / 'hardneg').exists() else []:
            _copy_file(mp, slug_root / 'mines' / mp.parent.name / 'mine_manifest.json',
                       f'{slug}/mines/{mp.parent.name}', manifest)

    manifest_path = EXPORT_ROOT / 'export_manifest.json'
    write_json_atomic(manifest_path, manifest)
    total = sum(v['bytes'] for v in manifest['files'].values())
    print(f'[export] wrote {manifest_path}')
    print(f"[export] copied={len(manifest['files'])} missing={len(manifest['missing'])} total={total/1e9:.2f}GB")
else:
    print('RUN_EXPORT=False; skipping export.')

print('\\nDone. Inspect:')
print(f'  per-model lab roots: {LAB_ROOT}')
print(f'  export root:         {EXPORT_ROOT}')
print(f'  manifest:            {LAB_ROOT / "headbank_manifest.json"}')

succeeded = [s for s in ENABLED_SLUGS if s not in FAILED_SLUGS]
print(f'\\n[bank] SUCCEEDED: {succeeded}')
if FAILED_SLUGS:
    print(f'[bank] FAILED (excluded from later stages): {sorted(FAILED_SLUGS)}')
    print('[bank] Re-run after fixing the cause; succeeded models skip via existing artifacts.')
else:
    print('[bank] all enabled models completed.')
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
            "accelerator": "GPU",
            "colab": {"name": "maximal_spec_headbank_500u.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out_path = Path(__file__).resolve().parent / "maximal_spec_headbank_500u.ipynb"
    out_path.write_text(json.dumps(build_nb(), indent=1) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
