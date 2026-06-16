# On-device instruct post-train: RWKV7-g1-0.4B (M3 Pro, $0, overnight)

Goal: take the already-instruct-pretrained **`BlinkDL/rwkv7-g1` 0.4B** state-space
model and post-train it (light SFT → DPO → optional Qwen2.5-3B on-policy
distillation) **entirely on this M3 Pro** — no RunPod, no cloud, ~overnight, $0.
Capture is local; training runs on `mps` (slower than CUDA, but free).

This is a **post-train of an existing instruct model**, NOT a from-scratch
distill. A prior research pass concluded that is the cheapest path to a coherent
~0.4B instruct SSM.

> **Status:** this is the *runbook* produced by the CPU/IO prep pass on branch
> `rwkv7/posttrain-prep`. The corpus builder and a small sample are committed.
> The GPU capture + training steps below are written but **deliberately not
> executed** — they are sequenced for after the in-flight perf benchmark frees
> the GPU. Everything is one-command-ready.

---

## 0. Hardware + the one toolchain gotcha (READ FIRST)

| | |
|---|---|
| Chip | Apple M3 Pro (6P+6E, 12 cores) |
| Unified memory | **18 GB** (this is the binding constraint) |
| Backend | PyTorch `mps` (Metal). `mlx` is the faster-but-narrower alternative. |
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

> If `flash-linear-attention` pulls a CUDA-only Triton path, that's fine — on
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
  fine for **generating teacher *text*** (run it through `dismantle generate` or
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

### 4a. (GPU) Teacher data capture — **RUN WHEN THE GPU IS FREE**

> ⛔ **Do not run this while the perf benchmark is using the GPU.** This is the
> first GPU step. Everything above (install, download, corpus) is CPU/IO and can
> run alongside the bench; this cannot.

For text-level on-policy DPO we need, for each prompt, a **"chosen"** completion
from the strong teacher and a **"rejected"** completion from the current student.
Generate both to disk first, then train CPU/GPU-decoupled.

**THE EXACT FIRST COMMAND to kick off once the bench frees the GPU** (teacher
"chosen" set, via the in-repo engine + the GGUF teacher already on disk):

```sh
# (A) teacher "chosen" completions — Qwen2.5-3B GGUF, greedy, to JSONL.
# Use --profile exact for the teacher: we want clean, un-traded output as the
# "chosen" target (the `fast` profile applies vocab-prune-32k, a mild quality
# trade). `--profile` is a global flag and works after the subcommand.
cargo run --release -p dismantle -- generate \
  --weights models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --prompts-file artifacts/rwkv7_posttrain/dpo_prompts.jsonl.prompts.txt \
  --max-new-tokens 256 \
  --max-seq-len 4096 \
  --profile exact \
  > artifacts/rwkv7_posttrain/teacher_chosen.raw.jsonl
```

`--prompts-file` wants a newline-delimited *plain prompt* file, so first flatten
the DPO scaffold (CPU, one-liner):

```sh
python - <<'PY'
import json
src="artifacts/rwkv7_posttrain/dpo_prompts.jsonl"
dst="artifacts/rwkv7_posttrain/dpo_prompts.jsonl.prompts.txt"
with open(src) as f, open(dst,"w") as o:
    for ln in f:
        o.write(json.loads(ln)["prompt"].replace("\n","\\n")+"\n")
print("wrote", dst)
PY
```

Then (B) the **"rejected"** set = the *student before training* on the same
prompts (so DPO has a contrast). Use the student GGUF the de-risk pass produced:

```sh
cargo run --release -p dismantle -- generate \
  --weights models/rwkv7-g1-04/rwkv7-0.4B-g1.Q4_K_M.gguf \
  --prompts-file artifacts/rwkv7_posttrain/dpo_prompts.jsonl.prompts.txt \
  --max-new-tokens 256 --max-seq-len 4096 --profile exact \
  > artifacts/rwkv7_posttrain/student_rejected.raw.jsonl
```

Finally zip chosen+rejected into a trl-DPO file `dpo.jsonl` with rows
`{"prompt","chosen","rejected"}` (CPU one-liner; pair by line index).

> Alternative if you prefer not to build the engine: run the same two
> generations with `llama.cpp`'s `llama-cli -m <gguf> -f prompts.txt -n 256`.
> Either way the output is just text — no logits required for this path.

**Wall-clock for capture (on `mps`/Metal, GPU free):** Qwen2.5-3B Q4_K_M decodes
≈ 17 tok/s here; 3000 prompts × 256 tok ≈ 0.77M tok ≈ **~12.5 h** at that rate.
To fit "overnight", either (i) cut to **~1200 prompts** (≈ 5 h), or (ii) cap
`--max-new-tokens 160`, or (iii) generate chosen for a 1200-subset and reuse the
SFT set for the rest. The student "rejected" pass is ~3× faster (smaller model).

### 4b. (GPU) SFT — 1 epoch on `mps`

`trl` `SFTTrainer` over `sft.jsonl` (the `text` field). fp32 weights (RWKV g1
ships fp32; mps bf16 is partial — see §6). LoRA optional (§6) to cut memory.

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

**Wall-clock (SFT, mps, fp32, 0.4B):** ≈ **1.5–4 s/step** at batch-1/seq-1024 on
M3 Pro (fp32 + gradient-checkpointing). 3000 examples ÷ 16 grad-accum ≈ 188
optimizer steps × 16 microsteps ≈ 3000 forward/backward ≈ **~1.5–3.5 h** for one
epoch. Comfortably overnight.

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

### 4d. (CPU) Export to the GGUF dismantle loads

dismantle's `rwkv7` loader reads `rwkv7.*` GGUF metadata. Convert the trained HF
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
   - Generate with: `cargo run --release -p dismantle -- generate --weights
     <gguf> --prompt "<ifeval prompt>" --max-new-tokens 256`.
2. **Recall probe (the SSM's known weak spot):** a needle-in-a-haystack / NIAH
   probe at 1k / 2k / 4k context. RWKV's fixed state can lose mid-context facts;
   confirm the post-train didn't regress recall below the pre-train g1, and note
   the gap vs Qwen2.5-0.5B (attention will win recall — that's expected; the
   SSM's win is **tps + flat memory at long ctx**).
3. **Long-context tps (GPU, quick):** the SSM's headline advantage — O(1) state,
   no growing KV. Measure decode tps at 4k context for both:
   ```sh
   cargo run --release -p dismantle -- generate \
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

## 5. Memory budget (18 GB unified)

Everything competes for the same 18 GB; macOS + apps already hold ~3–4 GB.

| Phase | Resident | Fits 18 GB? |
|---|---|---|
| Corpus build (CPU) | < 1 GB | ✅ trivially |
| Teacher capture (Qwen 3B Q4_K_M, GGUF) | ~2.3 GB weights + KV | ✅ |
| Teacher capture (Qwen 3B **HF fp16**, only if logit-KD) | ~6.2 GB | 🟡 alone yes; not with student |
| SFT (0.4B fp32 + Adam states + acts, gc on, seq 1024) | ~5–8 GB | ✅ |
| DPO (0.4B fp32 + frozen ref copy + 2× acts) | ~9–12 GB | 🟡 tight — use LoRA or seq 768 |
| Export/convert (CPU) | ~2 GB | ✅ |

**Rules for 18 GB:** (1) never run teacher-generate and training simultaneously —
capture to disk first; (2) keep `max_length` ≤ 1024 (drop to 768 if you OOM in
DPO); (3) `gradient_checkpointing=True` always; (4) batch-1 + grad-accum, never
batch > 1; (5) if DPO OOMs, switch the student to **LoRA** (§6) so the frozen ref
is free.

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
- **CPU-validation fallback:** if mps misbehaves, run a **tiny** sanity train on
  CPU (`.to("cpu")`, `--n 50` via the offline corpus, 5 steps) just to confirm
  the loss decreases and the export round-trips. Do not run the full train on
  CPU (it would take days).

---

## 7. Optional: the MLX path (faster, narrower)

`mlx-lm` has native Apple-silicon training and is typically **faster than
torch-mps** for small models, but RWKV7-g1 is not a first-class `mlx-lm`
architecture, so this needs an MLX RWKV7 implementation (more wiring). If torch-
mps wall-clock is acceptable (§4), **stay on torch-mps**. Revisit MLX only if you
want to push the run from ~overnight toward a few hours and are willing to port
the RWKV7 layer to MLX.

---

## 8. End-to-end command summary (ordered)

```sh
# --- CPU/IO: safe NOW, alongside the GPU bench ---
uv venv .venv-rwkv --python 3.12 && source .venv-rwkv/bin/activate
uv pip install torch transformers datasets trl accelerate peft \
               flash-linear-attention huggingface_hub numpy pyarrow
huggingface-cli download fla-hub/rwkv7-0.4B-g1 --local-dir models/rwkv7-g1-04-hf
python tools/training/rwkv7_build_corpus.py --n 3000

# --- GPU: ONLY after the perf bench frees the GPU ---
# (0) flatten the DPO scaffold to a plain newline-delimited prompts file (§4a)
# (1) FIRST GPU COMMAND — teacher "chosen" capture
cargo run --release -p dismantle -- generate \
  --weights models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --prompts-file artifacts/rwkv7_posttrain/dpo_prompts.jsonl.prompts.txt \
  --max-new-tokens 256 --max-seq-len 4096 --profile exact \
  > artifacts/rwkv7_posttrain/teacher_chosen.raw.jsonl
# (2) student "rejected" capture  (§4a)
# (3) SFT 1 epoch                 (§4b)   ~1.5–3.5 h
# (4) DPO                         (§4c)   ~2–5 h
# (5) export to GGUF              (§4d)   CPU
# (6) eval gate                   (§4e)
```

**Realistic total on-device wall-clock:** capture ~5 h (at ~1200 prompts) +
SFT ~2–3 h + DPO ~3–4 h ≈ **~10–12 h across one or two nights**, $0, all on the
M3 Pro. Peak memory stays under 18 GB if §5's rules are followed.
