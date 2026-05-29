#!/usr/bin/env python3
"""Emit `colab/headbank_corrected.ipynb` — the corrected Eagle5 headbank
retrain notebook.

WHY THIS EXISTS (2026-05-29)
----------------------------
The original headbank heads gave ~0-1.7% real spec-decode acceptance. Two
root causes, both now fixed and validated locally on Qwen-3B:

  1. FROZEN-WEIGHTS MISMATCH (the big one). The head's baseline is
     `argmax(RMSNorm(residual, output_norm) @ lm_head)`. The old frozen npz
     was an HF fp16 export; the runtime serves the GGUF's Q6_K-dequantized
     weights. They differ by up to 0.27/element — enough to flip the argmax
     and tank the lens ceiling to 0%. Rebuilding frozen weights from the GGUF
     dequant (tools/orchestrator/build_frozen_gguf.py) lifts the q3b lens
     ceiling to 74%.

  2. SELF-REFERENTIAL OBJECTIVE. Training/eval targeted
     `argmax(RMSNorm(captured_residual)@lm_head)` (the head's own baseline)
     instead of the model's REAL next token. `--target-mode corpus` fixes
     both train and eval to use the captured real next token.

Plus the residuals are now captured from the QUANTIZED runtime (Q4_K_M) at
layer n-1, not the fp16 calibration model — eliminating the fp16→quant shift.

Validated locally: q3b head reaches 73.9% depth-1 accuracy on training-aligned
inputs, matching the 74.2% lens ceiling.

INPUTS (uploaded to Drive by the local side)
--------------------------------------------
Under DRIVE_ROOT/, per model slug:
  <slug>/corpus_shards/shard_*.parquet   # locally-captured quantized residuals
  <slug>/frozen_gguf.npz                 # GGUF-dequant frozen weights

OUTPUTS
-------
  DRIVE_ROOT/heads_corrected/<slug>/head_final.safetensors
  DRIVE_ROOT/heads_corrected/<slug>/tau_eval.json
  DRIVE_ROOT/heads_corrected/headbank_manifest.json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/joshuahickscorp/dismantle.git"
BRANCH = "codex/maximal-spec-colab"

# slug -> (capture_layer = n-1, human label). hidden/vocab are inferred from
# frozen_gguf.npz by the trainer, so we don't hardcode them.
MODELS = {
    "q05b": {"capture_layer": 23, "label": "Qwen2.5-0.5B-Instruct"},
    "q1p5b": {"capture_layer": 27, "label": "Qwen2.5-1.5B-Instruct"},
    "q3b": {"capture_layer": 35, "label": "Qwen2.5-3B-Instruct"},
    "q7b": {"capture_layer": 27, "label": "Qwen2.5-7B-Instruct"},
}

# Training hyperparameters validated on q3b (loss converged ~1.0, 73.9% depth-1).
# Rollout loss trains multi-depth prediction (depths 1-4 from one residual,
# autoregressive token feeding) — required for K>1 speculation speedup; a
# depth-1-only head can only re-predict the token we already have.
TRAIN = dict(
    epochs=8, batch_size=24, seq_len=16, lr=1e-3,
    rollout_loss_weight=0.5, rollout_depth=5,
    rollout_depth_targets="1,2,3,4", rollout_draft_prob=0.75,
)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build() -> dict:
    cells = []

    cells.append(md(
        "# Eagle5 Headbank — Corrected Retrain\n"
        "\n"
        "Retrains the spec-decode head for every Qwen model with the two fixes "
        "validated locally on Qwen-3B (lens ceiling 0%→74%, head depth-1 73.9%):\n"
        "\n"
        "1. **Frozen weights from the GGUF dequant** (not HF fp16) so the head's "
        "baseline `RMSNorm(residual)@lm_head` matches the runtime verifier.\n"
        "2. **`--target-mode corpus`** — train + eval against the model's REAL "
        "next token, not the self-referential baseline proxy.\n"
        "3. Residuals captured from the **quantized (Q4_K_M)** runtime at layer n-1.\n"
        "\n"
        "**Before running:** upload the locally-prepared inputs to Drive under "
        "`DRIVE_ROOT/<slug>/{corpus_shards/, frozen_gguf.npz}`. The local side "
        "produces these via `tools/orchestrator/capture_all.sh` + "
        "`build_frozen_gguf.py` and `stage_headbank_upload.py`.\n"
    ))

    cells.append(md("## 1. Mount Drive + config"))
    cells.append(code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "\n"
        "# EDIT THIS to where you uploaded the inputs:\n"
        "DRIVE_ROOT = '/content/drive/MyDrive/dismantle_headbank_corrected'\n"
        "\n"
        "import os\n"
        "assert os.path.isdir(DRIVE_ROOT), f'DRIVE_ROOT not found: {DRIVE_ROOT}'\n"
        "print('DRIVE_ROOT:', DRIVE_ROOT)\n"
        "print('contents:', os.listdir(DRIVE_ROOT))\n"
    ))

    cells.append(md("## 2. Clone repo (fixed trainer) + install deps"))
    cells.append(code(
        f"REPO_URL = {REPO_URL!r}\n"
        f"BRANCH = {BRANCH!r}\n"
        "import os\n"
        "if not os.path.isdir('/content/dismantle'):\n"
        "    !git clone --depth 1 --branch $BRANCH $REPO_URL /content/dismantle\n"
        "else:\n"
        "    !cd /content/dismantle && git fetch --depth 1 origin $BRANCH && git checkout $BRANCH && git reset --hard origin/$BRANCH\n"
        "!pip -q install pyarrow safetensors gguf packaging\n"
        "import sys\n"
        "sys.path.insert(0, '/content/dismantle/colab')\n"
        "print('repo + deps ready')\n"
    ))

    cells.append(md("## 3. Model configs + input verification"))
    cells.append(code(
        f"MODELS = {json.dumps(MODELS, indent=4)}\n"
        f"TRAIN = {json.dumps(TRAIN)}\n"
        "\n"
        "import os\n"
        "ready = {}\n"
        "for slug, cfg in MODELS.items():\n"
        "    shards = os.path.join(DRIVE_ROOT, slug, 'corpus_shards')\n"
        "    frozen = os.path.join(DRIVE_ROOT, slug, 'frozen_gguf.npz')\n"
        "    have_shards = os.path.isdir(shards) and any(f.endswith('.parquet') for f in os.listdir(shards)) if os.path.isdir(shards) else False\n"
        "    have_frozen = os.path.isfile(frozen)\n"
        "    ready[slug] = have_shards and have_frozen\n"
        "    print(f'{slug:6s} shards={have_shards} frozen={have_frozen} -> {\"READY\" if ready[slug] else \"MISSING\"}')\n"
        "TRAINABLE = [s for s, r in ready.items() if r]\n"
        "print('\\ntrainable:', TRAINABLE)\n"
    ))

    cells.append(md("## 4. Train every ready head (corrected pipeline)"))
    cells.append(code(
        "import subprocess, os\n"
        "OUT_ROOT = os.path.join(DRIVE_ROOT, 'heads_corrected')\n"
        "os.makedirs(OUT_ROOT, exist_ok=True)\n"
        "TRAINER = '/content/dismantle/colab/eagle5_train_pytorch.py'\n"
        "\n"
        "for slug in TRAINABLE:\n"
        "    cfg = MODELS[slug]\n"
        "    shards = os.path.join(DRIVE_ROOT, slug, 'corpus_shards')\n"
        "    frozen = os.path.join(DRIVE_ROOT, slug, 'frozen_gguf.npz')\n"
        "    ckpt = os.path.join(OUT_ROOT, slug)\n"
        "    os.makedirs(ckpt, exist_ok=True)\n"
        "    cmd = ['python', TRAINER,\n"
        "           '--corpus-dir', shards,\n"
        "           '--frozen', frozen,\n"
        "           '--ckpt-dir', ckpt,\n"
        "           '--device', 'cuda',\n"
        "           '--target-mode', 'corpus',\n"
        "           '--capture-layer', str(cfg['capture_layer']),\n"
        "           '--epochs', str(TRAIN['epochs']),\n"
        "           '--batch-size', str(TRAIN['batch_size']),\n"
        "           '--seq-len', str(TRAIN['seq_len']),\n"
        "           '--lr', str(TRAIN['lr']),\n"
        "           '--rollout-loss-weight', str(TRAIN['rollout_loss_weight']),\n"
        "           '--rollout-depth', str(TRAIN['rollout_depth']),\n"
        "           '--rollout-depth-targets', TRAIN['rollout_depth_targets'],\n"
        "           '--rollout-draft-prob', str(TRAIN['rollout_draft_prob']),\n"
        "           '--save-safetensors']\n"
        "    print('\\n===', slug, cfg['label'], '===')\n"
        "    print(' '.join(cmd))\n"
        "    r = subprocess.run(cmd, capture_output=True, text=True)\n"
        "    print(r.stdout[-2000:])\n"
        "    if r.returncode != 0:\n"
        "        print('STDERR:', r.stderr[-2000:])\n"
        "    else:\n"
        "        print('OK ->', os.path.join(ckpt, 'head_final.safetensors'))\n"
    ))

    cells.append(md(
        "## 5. Tau eval (corpus mode = real next token)\n"
        "Reports the honest depth-1 / multi-depth acceptance ceiling against the "
        "model's actual next token (not the old self-referential proxy)."
    ))
    cells.append(code(
        "import subprocess, os, json\n"
        "EVAL = '/content/dismantle/colab/eagle5_tau_eval_pytorch.py'\n"
        "tau_results = {}\n"
        "for slug in TRAINABLE:\n"
        "    cfg = MODELS[slug]\n"
        "    ckpt = os.path.join(OUT_ROOT, slug, 'latest.npz')\n"
        "    frozen = os.path.join(DRIVE_ROOT, slug, 'frozen_gguf.npz')\n"
        "    shards = os.path.join(DRIVE_ROOT, slug, 'corpus_shards')\n"
        "    out = os.path.join(OUT_ROOT, slug, 'tau_eval.json')\n"
        "    if not os.path.isfile(ckpt):\n"
        "        print(slug, 'no checkpoint, skipping'); continue\n"
        "    cmd = ['python', EVAL,\n"
        "           '--ckpt', ckpt, '--frozen', frozen, '--corpus', shards,\n"
        "           '--out', out, '--device', 'cuda',\n"
        "           '--target-mode', 'corpus', '--depth', '4']\n"
        "    r = subprocess.run(cmd, capture_output=True, text=True)\n"
        "    if r.returncode == 0 and os.path.isfile(out):\n"
        "        tau_results[slug] = json.load(open(out))\n"
        "        d1 = tau_results[slug].get('depth1_accept_rate', 0)\n"
        "        tau = tau_results[slug].get('tau', 0)\n"
        "        print(f'{slug:6s} depth1={d1:.1%} tau={tau:.2f}')\n"
        "    else:\n"
        "        print(slug, 'eval failed:', r.stderr[-800:])\n"
    ))

    cells.append(md("## 6. Emit headbank manifest + summary"))
    cells.append(code(
        "import json, os, hashlib\n"
        "def sha256(p):\n"
        "    h = hashlib.sha256()\n"
        "    with open(p, 'rb') as f:\n"
        "        for b in iter(lambda: f.read(1<<20), b''): h.update(b)\n"
        "    return h.hexdigest()\n"
        "entries = []\n"
        "for slug in TRAINABLE:\n"
        "    head = os.path.join(OUT_ROOT, slug, 'head_final.safetensors')\n"
        "    if not os.path.isfile(head): continue\n"
        "    entries.append({\n"
        "        'slug': slug,\n"
        "        'label': MODELS[slug]['label'],\n"
        "        'arch': 'qwen2',\n"
        "        'capture_layer': MODELS[slug]['capture_layer'],\n"
        "        'head_path': head,\n"
        "        'head_sha256': sha256(head),\n"
        "        'metrics': tau_results.get(slug, {}),\n"
        "    })\n"
        "manifest = {'schema': 'dismantle-headbank-corrected-v1', 'entries': entries}\n"
        "mpath = os.path.join(OUT_ROOT, 'headbank_manifest.json')\n"
        "json.dump(manifest, open(mpath, 'w'), indent=2)\n"
        "print('wrote', mpath)\n"
        "for e in entries:\n"
        "    m = e['metrics']\n"
        "    print(f\"{e['slug']:6s} depth1={m.get('depth1_accept_rate',0):.1%} tau={m.get('tau',0):.2f}\")\n"
        "print('\\nDownload heads_corrected/ from Drive and stage with tools/headbank/pull.py')\n"
    ))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build()
    out = Path(__file__).resolve().parent / "headbank_corrected.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({len(nb['cells'])} cells)")
