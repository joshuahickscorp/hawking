# On-device instruct post-train: RWKV7-g1-0.4B (M3 Pro, $0, overnight)

Goal: take the already-instruct-pretrained **`BlinkDL/rwkv7-g1` 0.4B** state-space
model and post-train it (light SFT → DPO → optional Qwen2.5-3B on-policy
distillation) **entirely on this M3 Pro** — no RunPod, no cloud, ~overnight, $0.
Capture is local; training runs on `mps` (slower than cloud-GPU, but free).

This is a **post-train of an existing instruct model**, NOT a from-scratch
distill. A prior research pass concluded that is the cheapest path to a coherent
~0.4B instruct SSM.

> **Status:** the *optimized* runbook on branch `rwkv7/posttrain-opt` (extends
> the original prep pass on `rwkv7/posttrain-prep`). The corpus builder, the
> batched-capture CLI path, the streaming SFT trainer, and the CPU trainer-lever
> smoke are committed and CPU-validated. The GPU capture + training steps are
> **deliberately not executed** — sequenced for after the in-flight perf
> benchmark frees the GPU. Everything is one-command-ready.

---

## ⚡ What's optimized (and the new wall-clock)

The original prep pipeline was correct but slow (~10–12 h) because it (a) decoded
the teacher corpus **one prompt at a time** and (b) ran capture, then SFT, then
DPO strictly back-to-back. Four levers cut that to **~2.5–3.5 h** with **zero
quality change** (same teacher, same data, same greedy decode, same loss):

| # | Lever | Mechanism | Speedup | Validated |
|---|---|---|---|---|
| 1 | **Batched teacher capture** | `hawking generate --batched-capture` routes the prompt corpus through the **multiseq path** (B=8 sequences/pass, one Q4_K weight read amortised across the group) instead of the serial loop | **~6–8×** on capture | CLI built + CPU unit-tested; GPU run deferred |
| 2 | **torch-mps trainer (MLX rejected)** | MLX has **no RWKV-7** support → NO-GO (see §7). Optimize torch-mps instead: grad-accum for big effective batch at batch-1 memory, bf16 autocast where mps supports it, grad-checkpointing | **~1.3–1.7×** on train | trainer-lever smoke PASS on CPU |
| 3 | **Pipeline capture → SFT** | Capture writes **sharded JSONL**; the streaming trainer (`--watch`) starts SFT on finished shards while capture continues | overlaps the two longest phases → `max()` not `sum()` | dry-run validated; orchestrator `rwkv7_pipeline.sh` |
| 4 | **Aggressive-but-safe memory** | capture-to-disk-first (never teacher+trainer resident together) + tuned `max_length`/grad-accum/grad-checkpointing to fill 18 GB without OOM | enables 1–3 | budget in §5 |

**New projected wall-clock (math in §8):**

```
            OLD (serial, 1-at-a-time)         NEW (batched + overlapped)
capture     ~5–12.5 h  (1 prompt/pass)        ~0.8–1.6 h  (B=8/pass)      ← lever 1
SFT         ~1.5–3.5 h                         ~1.0–2.0 h                  ← lever 2
DPO         ~3–4 h                             ~1.5–2.5 h                  ← lever 2
overlap     (none: sum)                        capture ‖ SFT              ← lever 3
            ────────────                        ────────────
TOTAL       ~10–12 h                           ~2.5–3.5 h
```

The headline is lever 1: at ~17 tok/s single-stream the teacher capture dominated
the budget; the multiseq path turns each weight read into B=8 sequences' work, so
the same 0.77 M teacher tokens come out in roughly an eighth of the wall-clock.
Levers 2–4 then keep the train side from re-inflating it.

---

## 0. Hardware + the one toolchain gotcha (READ FIRST)

| | |
|---|---|
| Chip | Apple M3 Pro (6P+6E, 12 cores) |
| Unified memory | **18 GB** (this is the binding constraint) |
| Backend | PyTorch `mps` (Metal). **MLX investigated → NO-GO for RWKV-7, see §7.** |
| Student | RWKV7-g1 0.4B — ~0.45B params, hidden 1024, 24 layers, 32 heads × 64, vocab 65536, ctx 8192-trained |
| Teacher | Qwen2.5-3B-Instruct (for the optional distillation pass) |

**GOTCHA — the system `python3` is 3.14, which torch/mlx do NOT support yet.**
`python3 --version` here reports **3.14.5** (Homebrew). PyTorch ships wheels up
to 3.13 and MLX up to 3.13, so a `pip install torch` against the default
interpreter will fail or pull nothing usable. This machine already has
**`python3.12` (3.12.6)** and **`python3.11` (3.11.15)** installed, plus
**`uv` 0.11.7**. Build the training venv against **3.12** (or 3.11), never the
bare `python3`.

```sh
# always create the env with an explicit 3.12 interpreter
uv venv .venv-rwkv --python 3.12
source .venv-rwkv/bin/activate
python --version          # must say 3.12.x, NOT 3.14
```

---

## 1. Toolchain install (CPU only — safe to run now)

Audit result on this box at prep time:

| Package | Installed (system py3.14) | Needed for | Action |
|---|---|---|---|
| `numpy`, `pyarrow` | ✅ 2.4.6 / 24.0.0 | corpus IO | already present |
| `torch` (mps) | ❌ | SFT/DPO/distill on `mps` | install in 3.12 venv |
| `transformers` | ❌ | model + tokenizer + Trainer | install |
| `datasets` | ❌ (verified install OK as 5.0.0 in a 3.12 venv) | corpus pull | install |
| `trl` | ❌ | `SFTTrainer` + `DPOTrainer` | install |
| `accelerate` | ❌ | device placement / mps | install |
| **`flash-linear-attention` (`fla`)** | ❌ | **hard dep** — `fla-hub/rwkv7-0.4B-g1`'s `modeling_rwkv7.py` does `from fla.models.rwkv7 import RWKV7ForCausalLM`. The model will not load without it. | install |
| `peft` | ❌ | optional LoRA (cuts memory; see §6) | install |
| `huggingface_hub` | ❌ (CLI handy) | model/data fetch | install |
| `sentencepiece` | ❌ | not needed (RWKV uses its own world vocab), harmless | optional |
| `mlx` / `mlx-lm` | ❌ | optional faster path (see §7) | optional |

One-shot install into the 3.12 venv (CPU/IO; downloads wheels, no GPU):

```sh
uv venv .venv-rwkv --python 3.12 && source .venv-rwkv/bin/activate

# core training stack (torch mps wheels are the default macOS arm64 wheels)
uv pip install \
  "torch>=2.4" \
  "transformers>=4.48" \
  "datasets>=3.0" \
  "trl>=0.12" \
  "accelerate>=1.0" \
  "peft>=0.13" \
  "flash-linear-attention" \
  "huggingface_hub" \
  "numpy>=1.26" "pyarrow>=17"

python -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
#   must print mps True
```

> If `flash-linear-attention` pulls a cloud-GPU-only Triton path, that's fine — on
> `mps` the fla RWKV7 layer falls back to its native/torch recurrence. We are
> not using Triton kernels here. If import fails, pin `triton` out:
> `uv pip install flash-linear-attention --no-deps` then add `einops`
> + `transformers` manually.

---

## 2. Models on disk

### Student — RWKV7-g1 0.4B

The de-risk pass already fetched the **GGUF** form (inference only):
`models/rwkv7-g1-04/rwkv7-0.4B-g1.Q4_K_M.gguf` (300 MB). **That GGUF cannot be
trained** — we need the HF safetensors checkpoint. Two upstream forms exist:

- **`fla-hub/rwkv7-0.4B-g1`** — `model.safetensors` (902 MB) **+ `config.json` +
  `modeling_rwkv7.py` + `rwkv_vocab_v20230424.txt`**. This is the HF
  `RWKV7ForCausalLM` form that `trl` and `convert_hf_to_gguf.py` consume.
  **← fetch this one for training.**
- `BlinkDL/rwkv7-g1` — native `.pth` (`rwkv7-g1d-0.4b-20260210-ctx8192.pth`,
  902 MB). RWKV-LM-native; would require the BlinkDL `RWKV-LM` trainer, not trl.
  Skip unless you deliberately want the BlinkDL training loop.

```sh
# fetch the trainable HF student (~903 MB, IO only)
huggingface-cli download fla-hub/rwkv7-0.4B-g1 \
  --local-dir models/rwkv7-g1-04-hf
# sanity: must contain model.safetensors, config.json, modeling_rwkv7.py,
#         rwkv_vocab_v20230424.txt  (the last is required by the GGUF export)
ls models/rwkv7-g1-04-hf
```

### Teacher — Qwen2.5-3B-Instruct

- GGUF already on disk: `models/Qwen2.5-3B-Instruct-Q4_K_M.gguf` (1.93 GB) —
  fine for **generating teacher *text*** (run it through `hawking generate` or
  `llama.cpp`), which is all the DPO/distill step strictly needs.
- For **teacher *logits*** (true KL distillation) you'd want HF weights:
  `Qwen/Qwen2.5-3B-Instruct` safetensors ≈ **6.17 GB**. Only download if you do
  logit-level KD (§4c, optional). Text-level on-policy DPO does **not** need it.

> Recommendation for 18 GB: do **text-level on-policy DPO** with the GGUF
> teacher (cheap, fits). Reserve logit-KD for a later cloud run if quality
> stalls. Running the 3B teacher (HF, fp16 ≈ 6 GB) and the 0.4B student
> (fp32 ≈ 1.8 GB + optimizer) simultaneously on `mps` is tight but feasible if
> you generate teacher data **first, to disk**, then train — never both at once.

---

## 3. Build the instruct corpus (CPU/IO — safe to run now)

Builder: `tools/training/rwkv7_build_corpus.py`. Pulls single-turn instruct
pairs skewed **code + chat** from OpenHermes-2.5 (code/reasoning) and
UltraChat-200k (chat), renders them in the RWKV-7 g1 chat format, and writes a
gitignored corpus under `artifacts/rwkv7_posttrain/`. Falls back to an in-repo
curated seed if HF is unreachable, so it always produces a well-formed file.

RWKV-7 g1 chat format (from the fla-hub tokenizer config):

```
<|rwkv_tokenizer_end_of_text|>User: {user}\n\nAssistant: {assistant}\n\n
```

`\n\n` is the model's EOS string. The builder emits that literal `text` plus a
`messages` array.

```sh
source .venv-rwkv/bin/activate     # needs `datasets`

# ~3000 examples, 45% code / 40% chat / 15% reasoning (the default skew)
python tools/training/rwkv7_build_corpus.py --n 3000

# outputs (gitignored):
#   artifacts/rwkv7_posttrain/sft.jsonl          ~3000 rows  (training input)
#   artifacts/rwkv7_posttrain/dpo_prompts.jsonl  dedup prompts (DPO scaffold)
#   artifacts/rwkv7_posttrain/manifest.json      provenance + sha256
```

A 40-row committed sample lives at `tools/training/data/rwkv7_sft_sample.jsonl`
(provenance + format reference only — the real corpus is rebuilt from the
command above and never committed). Offline build for CI/air-gapped:
`python tools/training/rwkv7_build_corpus.py --offline --n 200`.

---

## 4. The post-train pipeline

Trainers below are short, self-contained driver scripts. They are written to be
copy-pasteable; create each as a file under `tools/training/` (or inline with
`python - <<'PY'`). All assume `source .venv-rwkv/bin/activate` and the corpus
from §3.

### 4a. (GPU) Teacher data capture — **BATCHED (lever 1)** — RUN WHEN GPU FREE

> ⛔ **Do not run this while the perf benchmark is using the GPU.** This is the
> first GPU step. Everything above (install, download, corpus) is CPU/IO and can
> run alongside the bench; this cannot.

For text-level on-policy DPO we need, per prompt, a **"chosen"** completion from
the strong teacher and a **"rejected"** completion from the current student.
Generate both to disk first, then train CPU/GPU-decoupled.

**The optimization:** the original runbook redirected `generate --prompts-file`
to a file, but that path decodes **one prompt at a time** (`max_batch_size = 1`)
— each Q4_K weight read serves a single sequence. hawking already has a
**multiseq / continuous-batch path** (B=8 independent sequences, weight read once
per position across the whole group; `forward_multiseq_greedy_tokens` +
`prefill_slots_parallel` in `qwen_dense.rs`). The new `--batched-capture` flag
routes the prompt corpus through it.

> **Quality is identical.** Batched capture uses the *same* per-prompt greedy
> prefill (`forward_token_greedy_tcb`) and the *same* Q4_K-LM-head argmax the
> single-stream path uses — it is the `--profile exact`, temperature-0 teacher,
> just with B sequences sharing each weight read. No sampling, no draft, no
> approximation. (`crates/hawking/src/capture.rs`, unit-tested.)

**THE EXACT FIRST COMMAND once the bench frees the GPU** (teacher "chosen" set):

```sh
# (A) teacher "chosen" completions — Qwen2.5-3B GGUF, BATCHED greedy, sharded JSONL.
#   --batched-capture : route through the B=8 multiseq path (~6–8× vs serial)
#   --capture-out     : per-group shards <stem>.shard-NNNN.jsonl (for streaming SFT)
#   --capture-batch 8 : sequences per pass (max weight-read amortisation)
#   --profile exact   : clean, un-traded teacher target (no vocab-prune-32k trade)
cargo run --release -p hawking -- generate \
  --weights models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --prompts-file artifacts/rwkv7_posttrain/dpo_prompts.prompts.txt \
  --batched-capture \
  --capture-out artifacts/rwkv7_posttrain/teacher_chosen.jsonl \
  --capture-batch 8 \
  --max-new-tokens 256 \
  --max-seq-len 4096 \
  --profile exact
# writes artifacts/rwkv7_posttrain/teacher_chosen.shard-0000.jsonl, .shard-0001.jsonl, …
# each line: {"idx", "prompt", "completion", "stop"}  (idx = line in prompts file)
```

`--prompts-file` wants a newline-delimited *plain prompt* file, so first flatten
the DPO scaffold (CPU, one-liner):

```sh
python3.12 - <<'PY'
import json
src="artifacts/rwkv7_posttrain/dpo_prompts.jsonl"
dst="artifacts/rwkv7_posttrain/dpo_prompts.prompts.txt"
with open(src) as f, open(dst,"w") as o:
    for ln in f:
        o.write(json.loads(ln)["prompt"].replace("\n","\\n")+"\n")
print("wrote", dst)
PY
```

Then (B) the **"rejected"** set = the *student before training* on the same
prompts (so DPO has a contrast), also batched:

```sh
cargo run --release -p hawking -- generate \
  --weights models/rwkv7-g1-04/rwkv7-0.4B-g1.Q4_K_M.gguf \
  --prompts-file artifacts/rwkv7_posttrain/dpo_prompts.prompts.txt \
  --batched-capture \
  --capture-out artifacts/rwkv7_posttrain/student_rejected.jsonl \
  --capture-batch 8 \
  --max-new-tokens 256 --max-seq-len 4096 --profile exact
```

Finally zip chosen+rejected into a trl-DPO file `dpo.jsonl` with rows
`{"prompt","chosen","rejected"}` — pair by `idx` (CPU one-liner; concat shards
with `cat *.shard-*.jsonl`, then join chosen↔rejected on `idx`).

> **Greedy-only:** `--batched-capture` is the temperature-0 multiseq lane, which
> is exactly the teacher's greedy target. For *sampled* capture (you almost
> never want it for a teacher set) drop `--batched-capture` and use the serial
> `--prompts-file` path with `--temperature`.
>
> **Non-Qwen / non-macOS:** `--batched-capture` errors with a clear message if
> the engine lacks the multiseq seam; fall back to the serial path there.

**Wall-clock for capture — the big win (lever 1):**

| | serial (old) | batched B=8 (new) |
|---|---|---|
| effective decode tps | ~17 tok/s | ~17 tok/s × (B amortisation) — the GPU does B sequences' work per weight read |
| 3000 prompts × 256 tok = 0.77 M tok | **~12.5 h** | **~1.6 h** (≈ 8× fewer weight-read passes) |
| 1200-prompt subset | ~5 h | **~0.8 h** |

The exact multiplier is bounded by how compute- vs bandwidth-bound the q3b decode
is at B=8 on this M3 Pro — measure it on the first real run (the `[capture] DONE:
… prompts/s` line reports it). Even a conservative 5–6× (not the full 8×) puts
the full 3000-prompt teacher set under ~2.5 h. **Measure, don't assume 8×.**

### 4b. (GPU) SFT — 1 epoch on `mps`, shard-streaming (levers 2+3)

**Preferred path — the committed streaming trainer** (`tools/training/
rwkv7_sft_stream.py`). It trains directly off the capture shards from §4a,
optionally **streaming** them with `--watch` so SFT overlaps capture (lever 3),
and bakes in the tuned 18 GB config (grad-accum 16, grad-checkpointing,
`use_reentrant=False`, optional bf16) — lever 2:

```sh
# SFT on the teacher_chosen shards (post-capture, or with --watch DURING capture).
# CPU dry-run first (no GPU) to sanity the data + config path:
python3.12 tools/training/rwkv7_sft_stream.py \
  --model models/rwkv7-g1-04-hf \
  --shards-glob 'artifacts/rwkv7_posttrain/teacher_chosen.shard-*.jsonl' \
  --out artifacts/rwkv7_posttrain/sft_out --dry-run

# Real SFT (GPU). Drop --dry-run; add --watch to overlap with an in-flight capture
# (see §4-pipeline). The trainer renders {prompt,completion} shards into the RWKV-7
# chat `text` automatically, and also accepts pre-rendered {text} rows (e.g. the
# corpus builder's sft.jsonl), so you can train on either source.
python3.12 tools/training/rwkv7_sft_stream.py \
  --model models/rwkv7-g1-04-hf \
  --shards-glob 'artifacts/rwkv7_posttrain/teacher_chosen.shard-*.jsonl' \
  --out artifacts/rwkv7_posttrain/sft_out \
  --max-length 1024 --grad-accum 16
```

The trainer mechanics (bf16 autocast, grad-accum == big-batch, lossless
grad-checkpointing) are **CPU-validated** by `tools/training/rwkv7_train_smoke.py`
(run it before the GPU step; it PASSes in seconds with no GPU).

<details><summary>Equivalent inline `trl.SFTTrainer` snippet (over the corpus
builder's <code>sft.jsonl</code> instead of capture shards) — for reference</summary>

```sh
python - <<'PY'
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL="models/rwkv7-g1-04-hf"
tok=AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype=torch.float32).to("mps")
ds=load_dataset("json", data_files="artifacts/rwkv7_posttrain/sft.jsonl", split="train")

cfg=SFTConfig(
    output_dir="artifacts/rwkv7_posttrain/sft_out",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,     # effective batch 16
    learning_rate=1e-5,
    max_length=1024,                    # mps memory: keep seqs short
    bf16=False, fp16=False,             # mps: train fp32 (stable); see §6
    dataset_text_field="text",
    logging_steps=10, save_steps=500, report_to=[],
    gradient_checkpointing=True,        # trades compute for memory
)
SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok).train()
model.save_pretrained("artifacts/rwkv7_posttrain/sft_out/final")
tok.save_pretrained("artifacts/rwkv7_posttrain/sft_out/final")
print("SFT done")
PY
```
</details>

**Wall-clock (SFT, mps, fp32, 0.4B):** ≈ **1.5–4 s/step** at batch-1/seq-1024 on
M3 Pro (fp32 + gradient-checkpointing). 3000 examples ÷ 16 grad-accum ≈ 188
optimizer steps × 16 microsteps ≈ 3000 forward/backward ≈ **~1.5–3.5 h** for one
epoch — and when run with `--watch` it **overlaps capture** (lever 3), so it adds
only ~`max(0, train − capture)` to the wall-clock instead of its full duration.
Optional `--bf16` shaves the dense-matmul time *if* it doesn't trip an fp32
fallback on your mps build (measure; keep fp32 if unsure — §6).

### 4-pipeline. (GPU) Overlap capture ‖ SFT (lever 3)

The capture (§4a) writes shards as each B=8 group finishes; the streaming trainer
(§4b) can begin on the first shard while capture continues. The committed
orchestrator wires both together:

```sh
# capture (background, writes shards)  ‖  streaming SFT (foreground, --watch)
tools/training/rwkv7_pipeline.sh \
  artifacts/rwkv7_posttrain/dpo_prompts.prompts.txt \
  models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  models/rwkv7-g1-04-hf \
  artifacts/rwkv7_posttrain/sft_out 256 8
```

This turns `capture_time + sft_time` into ≈ `max(capture_time, sft_time)` + a
one-shard lead-in. **Memory note (lever 4):** the Qwen teacher (~2.3 GB Q4_K) and
the RWKV-7 student (~6–8 GB) are both resident in this mode. That fits 18 GB
*because the capture decode is light* (Q4_K, B=8 token-only readback), but if you
observe memory pressure, run capture to completion first (the plain §4a command,
no pipeline) and SFT after — **capture-to-disk-first is always the safe
fallback**, and you still keep lever 1's ~6–8× on capture.

### 4c. (GPU) DPO — preference alignment on `mps`

`trl` `DPOTrainer` over the `dpo.jsonl` built in §4a (prompt/chosen/rejected).

```sh
python - <<'PY'
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

BASE="artifacts/rwkv7_posttrain/sft_out/final"   # start from the SFT'd model
tok=AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(
    BASE, trust_remote_code=True, torch_dtype=torch.float32).to("mps")
ds=load_dataset("json", data_files="artifacts/rwkv7_posttrain/dpo.jsonl", split="train")

cfg=DPOConfig(
    output_dir="artifacts/rwkv7_posttrain/dpo_out",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    beta=0.1,
    max_length=1024, max_prompt_length=512,
    bf16=False, fp16=False,
    gradient_checkpointing=True,
    logging_steps=10, report_to=[],
)
# ref_model=None → trl makes an internal frozen copy; on 18 GB this doubles the
# 0.4B footprint (~3.6 GB fp32) which still fits. If tight, pass a LoRA policy
# (peft) so the ref is the base adapter-off model and no copy is needed.
DPOTrainer(model=model, ref_model=None, args=cfg,
           train_dataset=ds, processing_class=tok).train()
model.save_pretrained("artifacts/rwkv7_posttrain/dpo_out/final")
tok.save_pretrained("artifacts/rwkv7_posttrain/dpo_out/final")
print("DPO done")
PY
```

**Wall-clock (DPO, mps):** ~2× SFT per step (chosen+rejected both forwarded).
≈ **2–5 h** for the prompt set. Run SFT and DPO on consecutive nights, or
back-to-back if the machine is free.

> Optional logit-KD (true distillation) instead of/after DPO needs the HF
> teacher (§2, ~6 GB) resident alongside the student — on 18 GB do this with
> LoRA on the student and the teacher in 4-bit, or defer to cloud. Not required
> to ship a coherent instruct 0.4B; DPO with teacher *text* is the cheap path.

### 4d. (CPU) Export to the GGUF hawking loads

hawking's `rwkv7` loader reads `rwkv7.*` GGUF metadata. Convert the trained HF
safetensors with the in-repo converter (`Rwkv7ForCausalLM` is supported):

```sh
# the converter's _set_vocab_rwkv_world() requires rwkv_vocab_v20230424.txt in
# the model dir — copy it from the original fla-hub download:
cp models/rwkv7-g1-04-hf/rwkv_vocab_v20230424.txt \
   artifacts/rwkv7_posttrain/dpo_out/final/

python tools/strand/tools/gguf/convert_hf_to_gguf.py \
  artifacts/rwkv7_posttrain/dpo_out/final \
  --outfile models/rwkv7-g1-04-posttrained-f16.gguf \
  --outtype f16

# (optional) quantize to Q4_K_M to match the existing inference path.
# Use llama.cpp's llama-quantize if available:
#   llama-quantize models/rwkv7-g1-04-posttrained-f16.gguf \
#                  models/rwkv7-g1-04-posttrained-Q4_K_M.gguf Q4_K_M
```

### 4e. (mixed) Eval gate

Ship only if the post-trained model **ties on chat AND wins on long-context tps**
vs the size-matched baseline **Qwen2.5-0.5B-Instruct**
(`models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`, already on disk).

1. **Chat/instruct quality (mostly CPU; tiny GPU for generation):**
   - **IFEval** (instruction-following) and a small **MT-Bench** subset — run
     both models' generations, score with the standard harness. Gate: post-train
     **≥** the pre-train g1 on IFEval, and **ties** (within noise) with
     Qwen2.5-0.5B on the MT-Bench subset.
   - Generate with: `cargo run --release -p hawking -- generate --weights
     <gguf> --prompt "<ifeval prompt>" --max-new-tokens 256`.
2. **Recall probe (the SSM's known weak spot):** a needle-in-a-haystack / NIAH
   probe at 1k / 2k / 4k context. RWKV's fixed state can lose mid-context facts;
   confirm the post-train didn't regress recall below the pre-train g1, and note
   the gap vs Qwen2.5-0.5B (attention will win recall — that's expected; the
   SSM's win is **tps + flat memory at long ctx**).
3. **Long-context tps (GPU, quick):** the SSM's headline advantage — O(1) state,
   no growing KV. Measure decode tps at 4k context for both:
   ```sh
   cargo run --release -p hawking -- generate \
     --weights models/rwkv7-g1-04-posttrained-Q4_K_M.gguf \
     --prompt "$(python - <<'PY'
print("Summarize the following.\n\n"+("lorem ipsum dolor sit amet. "*450))
PY
)" --max-new-tokens 128 --max-seq-len 8192 --explain-performance
   ```
   vs the same for `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`. **Gate: RWKV tps > Qwen
   0.5B tps at 4k**, with the gap widening as context grows (RWKV decode is
   ctx-independent; Qwen's grows with KV).

**Ship criterion (one line):** ties Qwen2.5-0.5B on chat (IFEval/MT-Bench
subset), does not regress recall vs pre-train g1, and **wins decode tps at ≥4k
context**.

---

## 5. Memory budget (18 GB unified) — per phase (lever 4)

Everything competes for the same 18 GB; macOS + apps already hold ~3–4 GB, so the
working budget is **~14 GB**. The optimizations are tuned to fill that *without*
OOM — high utilisation, not timidity.

| Phase | Resident | Fits ~14 GB? | Notes |
|---|---|---|---|
| Corpus build (CPU) | < 1 GB | ✅ trivially | IO only |
| **Batched capture** (Qwen 3B Q4_K_M, B=8 multiseq) | ~2.3 GB weights + ~0.6 GB 8-slot KV (4k ctx) ≈ **~3 GB** | ✅ comfortable | B=8 token-only readback is tiny (32 B/step); the 8-slot KV arena is the only batch cost |
| Teacher capture (Qwen 3B **HF fp16**, only if logit-KD) | ~6.2 GB | 🟡 alone yes; not with student | not used in the text-level DPO path |
| **SFT** (0.4B fp32 + AdamW m+v + acts, gc on, seq 1024) | ~1.8 + ~3.6 + ~1–3 ≈ **~6–8 GB** | ✅ headroom | grad-checkpointing recomputes acts → keeps the last term low |
| **Pipeline** (capture ‖ SFT, lever 3) | capture ~3 GB **+** SFT ~6–8 GB ≈ **~9–11 GB** | 🟡 fits, watch pressure | safe because capture decode is light; fallback = capture-first then SFT |
| **DPO** (0.4B fp32 + frozen ref copy + 2× acts) | ~9–12 GB | 🟡 tight — LoRA or seq 768 | chosen+rejected both forwarded; ref copy doubles weights |
| Export/convert (CPU) | ~2 GB | ✅ | |

**Rules for 18 GB (the OOM guards):**
1. **Capture-to-disk-first is the invariant.** Never hold teacher-generate +
   trainer resident *unless* using the pipeline mode (where it's measured to fit
   because capture is light). If pipeline shows pressure, fall back to sequential.
2. `gradient_checkpointing=True` **always** (`use_reentrant=False`) — the single
   biggest activation-memory lever; validated lossless in the smoke.
3. `max_length ≤ 1024` for SFT; **drop to 768** if DPO OOMs.
4. **batch-1 + grad-accum** (effective batch 16) — raises the effective batch
   with **zero** extra resident memory (validated exact in the smoke). Never
   raise `per_device_train_batch_size` above 1 on 18 GB.
5. **DPO OOM fallback ladder:** (a) `max_length 768` → (b) `max_prompt_length
   384` → (c) **LoRA student** (§6) so the frozen ref needs no copy (frees
   ~1.8 GB) → (d) if still tight, reduce DPO `beta`-set to a 1500-prompt subset.
   One of these always fits.

---

## 6. Where `mps` is weak + fallbacks

- **bf16/fp16 on mps is partial.** Some RWKV7 ops (the WKV recurrence, certain
  reductions) fall back to **fp32** on mps or silently run on CPU. Train **fp32**
  (the config above does) for stability; it's slower but correct. Do not assume
  `bf16=True` will speed things up — it often triggers fp32 fallbacks anyway.
- **Op coverage gaps → `PYTORCH_ENABLE_MPS_FALLBACK=1`.** If a custom fla op is
  unimplemented on mps, set this env var so it falls back to CPU instead of
  crashing:
  ```sh
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```
  (Expect a slowdown on whichever op falls back; acceptable for an overnight run.)
- **LoRA to cut memory (recommended for DPO):** wrap the student with `peft`
  (`LoraConfig` on the time-mix + channel-mix projections, r=16). Cuts trainable
  params ~100×, frees the DPO ref copy, and fp32 LoRA on mps is well-supported.
- **CPU-validation (committed):** before the GPU step, run the trainer-lever
  smoke — it validates bf16 autocast, grad-accumulation (== big-batch, to 1e-8),
  and lossless grad-checkpointing on CPU in seconds, with no GPU:
  ```sh
  PYTORCH_ENABLE_MPS_FALLBACK=1 python3.12 tools/training/rwkv7_train_smoke.py
  # add --fla to also run a few CPU steps on the REAL fla RWKV-7 (needs fla + model)
  ```
  And dry-run the streaming trainer's data path (no training):
  ```sh
  python3.12 tools/training/rwkv7_sft_stream.py --model models/rwkv7-g1-04-hf \
    --shards-glob 'artifacts/rwkv7_posttrain/teacher_chosen.shard-*.jsonl' --dry-run
  ```
  Do not run the *full* train on CPU (it would take days) — these prove wiring,
  not throughput.

---

## 7. The MLX path — investigated, **NO-GO for RWKV-7** (verdict)

`mlx-lm` is Apple-native and typically faster than torch-mps for small models, so
it was the obvious candidate for lever 2. It was investigated and **rejected** —
honestly, with the reasons, so nobody re-treads it:

- **`mlx-lm` has no RWKV model at all.** Its `mlx_lm/models/` registry ships
  `mamba.py` / `mamba2.py` for state-space models but **no `rwkv*.py`** (checked
  upstream, June 2026). The fla-hub `rwkv7-0.4B-g1` HF checkpoint loads only via
  its bundled `modeling_rwkv7.py` + the `flash-linear-attention` library, neither
  of which MLX consumes. So `mlx_lm.lora` / `mlx_lm.fine_tune` **cannot load this
  model**.
- **The one community MLX RWKV port is v5-inference-only.** `dc-dc-dc/mlx-rwkv`
  is WIP (~7 commits), RWKV-**v5**, **inference only** — no v7, no training, no
  delta-rule/WKV-7 state evolution. Not usable for RWKV-7 SFT/DPO.
- **A from-scratch MLX port is a multi-day model build, not a drop-in win.** It
  would mean reimplementing RWKV-7's WKV-7 delta-rule time-mix + channel-mix as
  an MLX autograd `nn.Module` and writing the training loop, then re-passing the
  parity gate — high-risk, and the WKV-7 kernel is owned elsewhere in this repo.
  No validated speedup justifies that here.

**Verdict: stay on torch-mps; do not claim an MLX speedup.** Lever 2 is therefore
the *torch-mps optimizations* (grad-accum, grad-checkpointing, bf16-where-safe),
which are validated (`rwkv7_train_smoke.py`). If someone later wants MLX, the
prerequisite is an upstreamed MLX RWKV-7 *training* module — revisit only then.

---

## 8. End-to-end command summary (ordered, optimized)

```sh
# --- CPU/IO: safe NOW, alongside the GPU bench ---
uv venv .venv-rwkv --python 3.12 && source .venv-rwkv/bin/activate
uv pip install torch transformers datasets trl accelerate peft \
               flash-linear-attention huggingface_hub numpy pyarrow
huggingface-cli download fla-hub/rwkv7-0.4B-g1 --local-dir models/rwkv7-g1-04-hf
python3.12 tools/training/rwkv7_build_corpus.py --n 3000
# validate the trainer levers on CPU (seconds, no GPU):
PYTORCH_ENABLE_MPS_FALLBACK=1 python3.12 tools/training/rwkv7_train_smoke.py

# --- GPU: ONLY after the perf bench frees the GPU ---
# (0) flatten the DPO scaffold to a plain newline-delimited prompts file (§4a)
python3.12 - <<'PY'
import json
src="artifacts/rwkv7_posttrain/dpo_prompts.jsonl"; dst="artifacts/rwkv7_posttrain/dpo_prompts.prompts.txt"
open(dst,"w").writelines(json.loads(l)["prompt"].replace("\n","\\n")+"\n" for l in open(src))
PY

# (1+3) BATCHED teacher capture ‖ streaming SFT — lever 1 + lever 3 in one go:
tools/training/rwkv7_pipeline.sh \
  artifacts/rwkv7_posttrain/dpo_prompts.prompts.txt \
  models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  models/rwkv7-g1-04-hf \
  artifacts/rwkv7_posttrain/sft_out 256 8        # capture(~1.6h) ‖ SFT(~1–2h)

# (2) student "rejected" capture — batched (§4a, for DPO contrast)   ~0.3–0.5 h
# (4) DPO from sft_out/final                       (§4c)             ~1.5–2.5 h
# (5) export to GGUF                               (§4d)   CPU
# (6) eval gate                                    (§4e)
```

**Projected total on-device wall-clock (optimized):**

```
capture (chosen, B=8) ............ ~1.6 h   ┐  overlapped via rwkv7_pipeline.sh:
SFT (1 epoch, --watch) ........... ~1–2 h   ┘  wall ≈ max(1.6, 1–2) ≈ ~1.6–2.0 h
capture (rejected, B=8, smaller) . ~0.3–0.5 h
DPO (1 epoch) .................... ~1.5–2.5 h
export + eval (CPU/quick) ........ ~0.2 h
                                   ───────────
TOTAL ............................ ~2.5–3.5 h   (was ~10–12 h)  — $0, one sitting
```

vs the original **~10–12 h across one or two nights**. The win is dominated by
lever 1 (batched capture turning ~12.5 h of serial teacher decode into ~1.6 h);
lever 3 then hides SFT behind capture, and levers 2+4 keep DPO from re-inflating
the budget. Peak memory stays under the ~14 GB working budget if §5's guards are
followed. **The multipliers are conservative estimates pending the first GPU run
— the `[capture] DONE: … prompts/s` line and the SFT step time will confirm the
real numbers; trust those over these projections.**
