#!/usr/bin/env python3
"""Emit `colab/final_push.ipynb` — THE final-push, all-heavy-compute notebook.

Goal: spend cloud compute ONLY on levers that convert directly to dismantle
decode tps, measured by tps-predictive metrics. Single-stream decode is
bandwidth-bound (~90 tps ceiling on M3 Pro for q3b); the only ways past it are
(a) the MULTIPLIER — speculative tokens-per-weight-read — and (b) the
DENOMINATOR — fewer weight bytes read per token (sparsity / lower-bit). This
notebook trains every cloud-trainable piece of both.

TRACKS
------
A. Setup — clone repo, deps, fetch corpora (release) + rebuild frozen (HF GGUF,
   fingerprint-verified). [proven]
B. HEAD MAXIMIZATION SWEEP [proven, default-ON, the core heavy compute] —
   grid over {num_blocks, ff_mult, rollout_depth, epochs} with chained-hidden
   rollout; eval each with the RUNTIME-PREDICTIVE accepted-prefix
   (--chain-hidden); leaderboard; pick the best head per model. This maximizes
   the spec MULTIPLIER (accepted-prefix 1.6 -> target 2.5+).
C. SPARSITY PREDICTOR [experimental, default-OFF] — train a tiny per-layer FFN
   active-block predictor (the DENOMINATOR lever). REQUIRES a local
   FFN-activation capture (`--capture-ffn` — not yet built); the cell documents
   the expected data format and trains if present.
D. Q2/Q3 DISTILLED DRAFT [experimental, default-OFF] — a cheaper draft so the
   runtime can afford deeper/wider (tree) drafts. Documented + coded; gated.
E. EMIT — best head per model + leaderboard + manifest to Drive.

NOTE on tree-drafting: tree-structured speculation is a RUNTIME feature (top-k
per node + tree-attention verify), NOT a separate training objective — a
well-trained chained-hidden head (Track B) already produces the per-node
distributions a tree drafter consumes. So tree-draft heavy-lifting is local,
not cloud; Track B is what makes it strong.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/joshuahickscorp/dismantle.git"
REPO_SLUG = "joshuahickscorp/dismantle"
BRANCH = "codex/maximal-spec-colab"
RELEASE_TAG = "headbank-corpus-v1"

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

# Head-maximization grid. Each entry is (num_blocks, ff_mult, rollout_targets,
# epochs). chained-hidden + rollout_loss_weight 1.0 are fixed (proven). Bigger
# blocks / deeper rollout = better depth-2+ = higher accepted-prefix, at more
# train time. Ordered cheap->expensive so a time-boxed run still gets results.
SWEEP = [
    (1, 4.0, "1,2,3,4", 12),     # the proven baseline config
    (2, 4.0, "1,2,3,4", 12),     # bigger head
    (1, 4.0, "1,2,3,4,5,6", 14), # deeper rollout
    (2, 4.0, "1,2,3,4,5,6", 14), # bigger + deeper (usually the winner)
]


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


def build() -> dict:
    cells = []

    cells.append(md(
        "# dismantle — FINAL PUSH (all heavy compute)\n"
        "\n"
        "Single-stream decode is bandwidth-bound (~90 tps ceiling on M3 Pro for "
        "q3b; llama.cpp ~50). The only ways past that are the **multiplier** "
        "(spec tokens per weight-read) and the **denominator** (fewer bytes/"
        "token via sparsity / low-bit). This notebook trains every cloud piece "
        "of both, measured by **tps-predictive** metrics.\n"
        "\n"
        "- **Track B (default ON):** maximize the spec head (accepted-prefix "
        "1.6 → target 2.5+). This is the proven core heavy-compute.\n"
        "- **Tracks C/D (default OFF, audit before enabling):** sparsity "
        "predictor + Q2 draft — experimental, depend on local capture.\n"
        "\n"
        "`Runtime → Run all`. Best head per model + leaderboard land in Drive."
    ))

    # ── A. Setup ──────────────────────────────────────────────────────────
    cells.append(md("## A. Setup — Drive (output), repo, deps, inputs"))
    cells.append(code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "OUT_ROOT = '/content/drive/MyDrive/dismantle_final_push'\n"
        "import os; os.makedirs(OUT_ROOT, exist_ok=True)\n"
        f"REPO_URL={REPO_URL!r}; BRANCH={BRANCH!r}\n"
        "if not os.path.isdir('/content/dismantle'):\n"
        "    !git clone --depth 1 --branch $BRANCH $REPO_URL /content/dismantle\n"
        "else:\n"
        "    !cd /content/dismantle && git fetch --depth 1 origin $BRANCH && git checkout $BRANCH && git reset --hard origin/$BRANCH\n"
        "!pip -q install pyarrow safetensors gguf packaging huggingface_hub\n"
        "import sys; sys.path.insert(0,'/content/dismantle/colab')\n"
        "print('ready; output ->', OUT_ROOT)\n"
    ))
    cells.append(md("### A2. Fetch corpora (release) + rebuild frozen (HF GGUF, verified)"))
    cells.append(code(
        f"REPO_SLUG={REPO_SLUG!r}; RELEASE_TAG={RELEASE_TAG!r}\n"
        f"MODELS={json.dumps(MODELS, indent=4)}\n"
        "import os, json, tarfile, urllib.request, subprocess, numpy as np\n"
        "from huggingface_hub import hf_hub_download\n"
        "DATA_ROOT='/content/headbank_inputs'\n"
        "BASE=f'https://github.com/{REPO_SLUG}/releases/download/{RELEASE_TAG}'\n"
        "BUILDER='/content/dismantle/tools/orchestrator/build_frozen_gguf.py'\n"
        "os.makedirs(DATA_ROOT, exist_ok=True)\n"
        "fp_path=os.path.join(DATA_ROOT,'frozen_fingerprints.json')\n"
        "urllib.request.urlretrieve(f'{BASE}/frozen_fingerprints.json', fp_path)\n"
        "FP=json.load(open(fp_path))\n"
        "READY=[]\n"
        "for slug,cfg in MODELS.items():\n"
        "    dst=os.path.join(DATA_ROOT,slug); shards=os.path.join(dst,'corpus_shards')\n"
        "    os.makedirs(shards, exist_ok=True); frozen=os.path.join(dst,'frozen_gguf.npz')\n"
        "    if not any(f.endswith('.parquet') for f in os.listdir(shards)):\n"
        "        tarp=os.path.join(dst,'corpus.tar')\n"
        "        urllib.request.urlretrieve(f'{BASE}/{slug}_corpus.tar', tarp)\n"
        "        with tarfile.open(tarp) as t: t.extractall(shards)\n"
        "        os.remove(tarp)\n"
        "    if not os.path.isfile(frozen):\n"
        "        gguf=hf_hub_download(repo_id=cfg['hf_repo'], filename=cfg['hf_file'])\n"
        "        r=subprocess.run(['python',BUILDER,'--gguf',gguf,'--out',frozen],capture_output=True,text=True)\n"
        "        if r.returncode!=0: print(slug,'build_frozen failed:',r.stderr[-300:]); continue\n"
        "    z=np.load(frozen); on=z['output_norm'].astype(np.float32)\n"
        "    ref=np.array(FP[slug]['output_norm'],dtype=np.float32)\n"
        "    ok=on.shape==ref.shape and float(np.abs(on-ref).max())<1e-3\n"
        "    n=len([f for f in os.listdir(shards) if f.endswith('.parquet')])\n"
        "    print(f'{slug}: shards={n} frozen_verified={ok}')\n"
        "    if ok and n>0: READY.append(slug)\n"
        "print('READY:',READY)\n"
    ))

    # ── B. Head sweep ─────────────────────────────────────────────────────
    cells.append(md(
        "## B. HEAD MAXIMIZATION SWEEP  *(the core heavy compute)*\n"
        "For each model, train a grid of head configs (chained-hidden rollout) "
        "and keep the one with the highest **accepted-prefix** (the "
        "runtime-predictive spec multiplier). Ordered cheap→expensive so a "
        "time-boxed run still produces a winner."
    ))
    cells.append(code(
        f"SWEEP={json.dumps(SWEEP)}\n"
        "TRAINER='/content/dismantle/colab/eagle5_train_pytorch.py'\n"
        "EVAL='/content/dismantle/colab/eagle5_tau_eval_pytorch.py'\n"
        "import subprocess, os, json\n"
        "BEST={}; LEADER=[]\n"
        "for slug in READY:\n"
        "    cfg=MODELS[slug]; shards=os.path.join(DATA_ROOT,slug,'corpus_shards')\n"
        "    frozen=os.path.join(DATA_ROOT,slug,'frozen_gguf.npz')\n"
        "    best_prefix=-1.0; best_dir=None\n"
        "    for (nb,ffm,rt,ep) in SWEEP:\n"
        "        tag=f'b{nb}_ff{ffm}_rt{rt.replace(\",\",\"-\")}_e{ep}'\n"
        "        ckpt=os.path.join(OUT_ROOT,'sweep',slug,tag); os.makedirs(ckpt,exist_ok=True)\n"
        "        tr=['python',TRAINER,'--corpus-dir',shards,'--frozen',frozen,'--ckpt-dir',ckpt,\n"
        "            '--device','cuda','--target-mode','corpus','--capture-layer',str(cfg['capture_layer']),\n"
        "            '--epochs',str(ep),'--batch-size','24','--seq-len','16','--lr','1e-3',\n"
        "            '--num-blocks',str(nb),'--head-ff-mult',str(ffm),\n"
        "            '--rollout-loss-weight','1.0','--rollout-depth','8',\n"
        "            '--rollout-depth-targets',rt,'--rollout-draft-prob','0.75',\n"
        "            '--rollout-chain-hidden','--save-safetensors']\n"
        "        r=subprocess.run(tr,capture_output=True,text=True)\n"
        "        if r.returncode!=0: print(f'{slug}/{tag} TRAIN FAIL:',r.stderr[-300:]); continue\n"
        "        ev=['python',EVAL,'--ckpt',os.path.join(ckpt,'latest.npz'),'--frozen',frozen,\n"
        "            '--corpus',shards,'--out',os.path.join(ckpt,'tau.json'),'--device','cuda',\n"
        "            '--target-mode','corpus','--chain-hidden','--depth',str(min(8,max(4,int(rt.split(\",\")[-1])))),\n"
        "            '--num-blocks',str(nb),'--head-ff-mult',str(ffm)]\n"
        "        re=subprocess.run(ev,capture_output=True,text=True)\n"
        "        prefix=-1.0\n"
        "        tj=os.path.join(ckpt,'tau.json')\n"
        "        if re.returncode==0 and os.path.isfile(tj):\n"
        "            d=json.load(open(tj)); prefix=d.get('tau',-1.0)\n"
        "        LEADER.append({'slug':slug,'config':tag,'accepted_prefix':prefix,\n"
        "                       'depth1':json.load(open(tj)).get('depth1_accept_rate') if os.path.isfile(tj) else None})\n"
        "        print(f'{slug:6s} {tag:28s} accepted_prefix={prefix:.2f} -> ~{1+prefix:.2f}x')\n"
        "        if prefix>best_prefix: best_prefix=prefix; best_dir=ckpt\n"
        "    if best_dir: BEST[slug]={'dir':best_dir,'accepted_prefix':best_prefix}\n"
        "    print(f'=== {slug} BEST: {best_dir} accepted_prefix={best_prefix:.2f} ===')\n"
        "print('\\nBEST per model:', {k:round(v[\"accepted_prefix\"],2) for k,v in BEST.items()})\n"
    ))

    # ── C. Sparsity predictor (experimental) ──────────────────────────────
    cells.append(md(
        "## C. SPARSITY PREDICTOR  *(experimental — default OFF; audit first)*\n"
        "The DENOMINATOR lever: predict which FFN weight blocks are active for a "
        "token so the runtime skips the rest (read fewer bytes/token → direct "
        "tps gain that multiplies with spec). **Requires a local FFN-activation "
        "capture** (`dismantle ... --capture-ffn <path>`, NOT yet built — see "
        "the after-steps). Set `RUN_SPARSITY=True` only once that capture exists "
        "and is uploaded as `<slug>/ffn_act_shards/`. Trainer:\n"
        "`colab/sparsity_predictor_train.py` (committed in the repo)."
    ))
    cells.append(code(
        "RUN_SPARSITY=False  # AUDIT: enable only when local FFN-activation capture exists\n"
        "if RUN_SPARSITY:\n"
        "    SP='/content/dismantle/colab/sparsity_predictor_train.py'\n"
        "    import subprocess,os\n"
        "    for slug in READY:\n"
        "        ffn=os.path.join(DATA_ROOT,slug,'ffn_act_shards')\n"
        "        frozen=os.path.join(DATA_ROOT,slug,'frozen_gguf.npz')\n"
        "        if not os.path.isdir(ffn): print(slug,'no ffn capture; skip'); continue\n"
        "        out=os.path.join(OUT_ROOT,'sparsity',slug); os.makedirs(out,exist_ok=True)\n"
        "        r=subprocess.run(['python',SP,'--ffn-dir',ffn,'--frozen',frozen,'--out-dir',out,\n"
        "                           '--device','cuda','--epochs','6','--block-size','256'],\n"
        "                          capture_output=True,text=True)\n"
        "        print(slug, 'sparsity:', r.stdout[-400:] if r.returncode==0 else r.stderr[-400:])\n"
        "else:\n"
        "    print('sparsity track OFF (default). See after-steps for the local FFN capture.')\n"
    ))

    # ── D. Q2 draft (experimental) ────────────────────────────────────────
    cells.append(md(
        "## D. Q2/Q3 DISTILLED DRAFT  *(experimental — default OFF; audit first)*\n"
        "A cheaper standalone draft (small + low-bit) so the runtime can afford "
        "deeper/wider TREE drafts. The chained head already serves as the draft; "
        "this is a stretch lever. Default OFF; documented for the audit. A "
        "distillation harness would train a tiny model to mimic the target's "
        "next-token distribution on the corpus, then quantize to Q3_K/Q2_K."
    ))
    cells.append(code(
        "RUN_Q2_DRAFT=False  # AUDIT: design + validate before spending compute\n"
        "print('Q2 draft track OFF (default). Stretch lever; the chained head is the primary draft.')\n"
    ))

    # ── E. Emit ───────────────────────────────────────────────────────────
    cells.append(md("## E. Emit best heads + leaderboard + manifest"))
    cells.append(code(
        "import json,os,shutil,hashlib\n"
        "def sha(p):\n"
        "    h=hashlib.sha256();\n"
        "    with open(p,'rb') as f:\n"
        "        for b in iter(lambda:f.read(1<<20),b''): h.update(b)\n"
        "    return h.hexdigest()\n"
        "FINAL=os.path.join(OUT_ROOT,'best_heads'); os.makedirs(FINAL,exist_ok=True)\n"
        "entries=[]\n"
        "for slug,info in BEST.items():\n"
        "    src=os.path.join(info['dir'],'head_final.safetensors')\n"
        "    if not os.path.isfile(src): continue\n"
        "    dstd=os.path.join(FINAL,slug); os.makedirs(dstd,exist_ok=True)\n"
        "    dst=os.path.join(dstd,'head_final.safetensors'); shutil.copy2(src,dst)\n"
        "    entries.append({'slug':slug,'label':MODELS[slug]['label'],\n"
        "                    'capture_layer':MODELS[slug]['capture_layer'],\n"
        "                    'accepted_prefix':info['accepted_prefix'],\n"
        "                    'projected_speedup':round(1+info['accepted_prefix'],2),\n"
        "                    'head_path':dst,'head_sha256':sha(dst)})\n"
        "json.dump({'schema':'dismantle-final-push-v1','entries':entries,'leaderboard':LEADER},\n"
        "          open(os.path.join(FINAL,'manifest.json'),'w'),indent=2)\n"
        "print('=== FINAL HEADBANK ===')\n"
        "for e in entries: print(f\"{e['slug']:6s} accepted_prefix={e['accepted_prefix']:.2f} -> ~{e['projected_speedup']}x  {e['head_path']}\")\n"
        "print('\\nFull leaderboard + heads in', FINAL)\n"
    ))

    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                         "accelerator":"GPU","colab":{"provenance":[]}},
            "nbformat":4,"nbformat_minor":5}


if __name__ == "__main__":
    nb=build()
    out=Path(__file__).resolve().parent/"final_push.ipynb"
    out.write_text(json.dumps(nb,indent=1))
    print(f"wrote {out} ({len(nb['cells'])} cells)")
