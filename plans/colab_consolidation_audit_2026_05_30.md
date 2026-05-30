# Colab Consolidation Audit — 2026-05-30

Scope: the three recent Bible Colabs:

- `colab/01_awq_bytecut.ipynb`
- `colab/02_eagle3_train.ipynb`
- `colab/03_qtip_3bit.ipynb`

## Findings Fixed

1. `01_awq_bytecut` allowed dependency solver failures with `check=False`, then
   imported `transformers` later. This let a broken `sklearn -> scipy -> numpy`
   optional import chain surface as an `AutoTokenizer` error.
   - Fixed by moving dependency setup to the first code cell, uninstalling unused
     `sklearn/scipy`, pinning Transformers to 4.x, and smoke-testing imports
     before any expensive model work.

2. `01_awq_bytecut` installed unbounded `gptqmodel`. Current GPTQModel releases
   can pull incompatible NumPy/Transformers versions and may compile CUDA
   extensions on older GPUs.
   - Fixed by making GPTQ W3 lazy and disabled by default on cc < 8.0. AWQ W4
     now always runs first as the cheap quality signal.

3. `02_eagle3_train` asserted `head_final.safetensors` but did not pass
   `--save-safetensors` to the trainer.
   - Fixed.

4. `02_eagle3_train` passed the checkpoint directory to tau eval, but
   `eagle5_tau_eval_pytorch.py` expects a checkpoint file.
   - Fixed by passing the emitted `head_final.safetensors`.

5. `02_eagle3_train` evaluated tau without `--chain-hidden`, making the metric
   less predictive of the real runtime draft path.
   - Fixed by training with chained-hidden rollout and evaluating with
     `--chain-hidden`.

6. `03_qtip_3bit` used stale upstream QTIP script paths
   (`hessian_offline_llama.py`, root-level `quantize_finetune_llama.py`).
   - Fixed to the current module paths, but the notebook is now guarded with
     `RUN_QTIP=False` because it is still a research scaffold.

## Consolidation Decision

Do not merge all three into one run-all notebook.

- Consolidate the byte-cut decision around `01_awq_bytecut`: f16 denominator,
  AWQ W4, and optional GPTQ W3 belong together.
- Keep `02_eagle3_train` separate: it consumes M3-produced Q4_K_M capture
  artifacts and has a different success gate.
- Quarantine `03_qtip_3bit`: it is not compute-unit-efficient until AWQ/GPTQ
  and the M3 trellis-kernel work justify it.

## Compute-Unit Order

1. Run `01_awq_bytecut.ipynb`.
   - T4/L4: trust AWQ W4 first.
   - A100/L4 or cc >= 8.0: enable GPTQ W3 if the W3 answer is worth the extra run.
2. Run `02_eagle3_train.ipynb` only after M3 Q4_K_M captures and
   `qwen3b_frozen.npz` exist under `/content/artifacts/eagle5/`.
3. Do not run `03_qtip_3bit.ipynb` by default.

## Verification Performed Locally

- `python3 -m py_compile colab/01_awq_bytecut.py colab/02_eagle3_train.py colab/03_qtip_3bit.py`
- Regenerated all three `.ipynb` files via `colab/py_to_ipynb.py`.
- Parsed notebook JSON and compiled joined code cells for all three notebooks.

GPU execution is still required to validate quantization quality and tau.
