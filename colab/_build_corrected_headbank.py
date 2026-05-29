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
REPO_SLUG = "joshuahickscorp/dismantle"
BRANCH = "codex/maximal-spec-colab"
# GitHub release holding the locally-captured corpora + GGUF-dequant frozen
# weights. Assets: <slug>_corpus.tar + <slug>_frozen.npz. Public repo, so the
# notebook wget's them with no auth — the user uploads nothing.
RELEASE_TAG = "headbank-corpus-v1"

# slug -> (capture_layer = n-1, human label). hidden/vocab are inferred from
# frozen_gguf.npz by the trainer, so we don't hardcode them.
MODELS = {
    "q05b": {"capture_layer": 23, "label": "Qwen2.5-0.5B-Instruct",
             "hf_repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
             "hf_file": "qwen2.5-0.5b-instruct-q4_k_m.gguf"},
    "q1p5b": {"capture_layer": 27, "label": "Qwen2.5-1.5B-Instruct",
              "hf_repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
              "hf_file": "qwen2.5-1.5b-instruct-q4_k_m.gguf"},
    "q3b": {"capture_layer": 35, "label": "Qwen2.5-3B-Instruct",
            "hf_repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "hf_file": "qwen2.5-3b-instruct-q4_k_m.gguf"},
    "q7b": {"capture_layer": 27, "label": "Qwen2.5-7B-Instruct",
            "hf_repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "hf_file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf"},
}

# Training hyperparameters. CHAINED-HIDDEN rollout (EAGLE-style) is the key:
# the head feeds its OWN draft_hidden forward as the next-depth residual,
# learning to advance its hidden state. Validated on q3b — lifted multi-depth
# acceptance dramatically (depth-2 16%→47%, accepted-prefix 1.0→1.6 ≈ ~2.6×
# decode potential) vs fixed-residual rollout (which caps at depth-1).
TRAIN = dict(
    epochs=10, batch_size=24, seq_len=16, lr=1e-3,
    rollout_loss_weight=1.0, rollout_depth=5,
    rollout_depth_targets="1,2,3,4", rollout_draft_prob=0.75,
    rollout_chain_hidden=True,
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
        "**Just `Runtime → Run all`.** Inputs (corpora + frozen weights) auto-"
        "download from the GitHub release — nothing to upload. Trained heads are "
        "written to your Drive.\n"
    ))

    cells.append(md("## 1. Mount Drive (for output) + clone repo + deps"))
    cells.append(code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "OUT_ROOT = '/content/drive/MyDrive/dismantle_headbank_corrected/heads_corrected'\n"
        "import os\n"
        "os.makedirs(OUT_ROOT, exist_ok=True)\n"
        "\n"
        f"REPO_URL = {REPO_URL!r}\n"
        f"BRANCH = {BRANCH!r}\n"
        "if not os.path.isdir('/content/dismantle'):\n"
        "    !git clone --depth 1 --branch $BRANCH $REPO_URL /content/dismantle\n"
        "else:\n"
        "    !cd /content/dismantle && git fetch --depth 1 origin $BRANCH && git checkout $BRANCH && git reset --hard origin/$BRANCH\n"
        "!pip -q install pyarrow safetensors gguf packaging\n"
        "import sys\n"
        "sys.path.insert(0, '/content/dismantle/colab')\n"
        "print('output ->', OUT_ROOT)\n"
        "print('repo + deps ready')\n"
    ))

    cells.append(md(
        "## 2. Auto-fetch inputs (corpus from release, frozen rebuilt from HF)\n"
        "Corpora (locally-captured quantized residuals) download from the GitHub "
        "release. Frozen weights are rebuilt on Colab from the **official Qwen "
        "HF GGUFs** (fast Colab pipe, not your slow upstream) and **verified "
        "against an `output_norm` fingerprint** — a mismatch fails safe (skips "
        "the model) rather than training a bad head."
    ))
    cells.append(code(
        f"REPO_SLUG = {REPO_SLUG!r}\n"
        f"RELEASE_TAG = {RELEASE_TAG!r}\n"
        f"MODELS = {json.dumps(MODELS, indent=4)}\n"
        "import os, json, tarfile, urllib.request, subprocess, numpy as np\n"
        "!pip -q install huggingface_hub >/dev/null\n"
        "from huggingface_hub import hf_hub_download\n"
        "DATA_ROOT = '/content/headbank_inputs'\n"
        "BASE = f'https://github.com/{REPO_SLUG}/releases/download/{RELEASE_TAG}'\n"
        "BUILDER = '/content/dismantle/tools/orchestrator/build_frozen_gguf.py'\n"
        "os.makedirs(DATA_ROOT, exist_ok=True)\n"
        "# fingerprints for frozen verification\n"
        "fp_path = os.path.join(DATA_ROOT, 'frozen_fingerprints.json')\n"
        "urllib.request.urlretrieve(f'{BASE}/frozen_fingerprints.json', fp_path)\n"
        "FP = json.load(open(fp_path))\n"
        "\n"
        "READY = []\n"
        "for slug, cfg in MODELS.items():\n"
        "    dst = os.path.join(DATA_ROOT, slug)\n"
        "    shards = os.path.join(dst, 'corpus_shards')\n"
        "    os.makedirs(shards, exist_ok=True)\n"
        "    frozen = os.path.join(dst, 'frozen_gguf.npz')\n"
        "    # 1) corpus from release\n"
        "    if not any(f.endswith('.parquet') for f in os.listdir(shards)):\n"
        "        tarp = os.path.join(dst, 'corpus.tar')\n"
        "        print(f'{slug}: downloading corpus...')\n"
        "        urllib.request.urlretrieve(f'{BASE}/{slug}_corpus.tar', tarp)\n"
        "        with tarfile.open(tarp) as t: t.extractall(shards)\n"
        "        os.remove(tarp)\n"
        "    # 2) frozen: rebuild from HF GGUF, verify against fingerprint\n"
        "    if not os.path.isfile(frozen):\n"
        "        print(f'{slug}: downloading GGUF {cfg[\"hf_repo\"]}/{cfg[\"hf_file\"]} ...')\n"
        "        gguf = hf_hub_download(repo_id=cfg['hf_repo'], filename=cfg['hf_file'])\n"
        "        print(f'{slug}: building frozen from GGUF dequant...')\n"
        "        r = subprocess.run(['python', BUILDER, '--gguf', gguf, '--out', frozen],\n"
        "                           capture_output=True, text=True)\n"
        "        if r.returncode != 0:\n"
        "            print(f'{slug}: build_frozen failed:', r.stderr[-400:]); continue\n"
        "    # 3) verify output_norm fingerprint\n"
        "    z = np.load(frozen)\n"
        "    on = z['output_norm'].astype(np.float32)\n"
        "    ref = np.array(FP[slug]['output_norm'], dtype=np.float32)\n"
        "    ok = on.shape == ref.shape and float(np.abs(on - ref).max()) < 1e-3\n"
        "    n = len([f for f in os.listdir(shards) if f.endswith('.parquet')])\n"
        "    print(f'{slug}: {n} shards, frozen_verified={ok}')\n"
        "    if ok and n > 0:\n"
        "        READY.append(slug)\n"
        "    else:\n"
        "        print(f'   !! {slug} SKIPPED — frozen fingerprint mismatch (GGUF source differs). '\n"
        "              f'Upload {slug}_frozen.npz to the release to train it.')\n"
        "print('\\nREADY:', READY)\n"
    ))

    cells.append(md("## 3. Training config"))
    cells.append(code(
        f"TRAIN = {json.dumps(TRAIN)}\n"
        "TRAINABLE = READY  # set in cell 2 (corpus present + frozen fingerprint verified)\n"
        "print('trainable:', TRAINABLE)\n"
        "assert TRAINABLE, 'no models ready — check cell 2 output'\n"
    ))

    cells.append(md("## 4. Train every ready head (corrected pipeline)"))
    cells.append(code(
        "import subprocess, os\n"
        "TRAINER = '/content/dismantle/colab/eagle5_train_pytorch.py'\n"
        "\n"
        "for slug in TRAINABLE:\n"
        "    cfg = MODELS[slug]\n"
        "    shards = os.path.join(DATA_ROOT, slug, 'corpus_shards')\n"
        "    frozen = os.path.join(DATA_ROOT, slug, 'frozen_gguf.npz')\n"
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
        "    if TRAIN.get('rollout_chain_hidden'):\n"
        "        cmd.append('--rollout-chain-hidden')\n"
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
        "    frozen = os.path.join(DATA_ROOT, slug, 'frozen_gguf.npz')\n"
        "    shards = os.path.join(DATA_ROOT, slug, 'corpus_shards')\n"
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
