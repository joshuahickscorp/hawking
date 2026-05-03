# Kernel Profile

A kernel profile is a JSON file that records which dispatch strategy
performed best on a specific `(model, shader_version, device)` triple. The
runtime loads it on startup and selects the corresponding Metal kernel path
without re-benchmarking.

## Profile fields

```json
{
  "schema_version": 1,
  "profile_id": "<hash>",
  "profile_name": "m3-pro-18gb",
  "model_id": "DeepSeek-V2-Lite-Chat",
  "model_arch": "deepseek2",
  "tensor_layout_hash": "<hash>",
  "device_name": "Apple M3 Pro",
  "shader_hash": "<hash>",
  "selected": {
    "id": "gpu-greedy-frontier",
    "moe_schedule": "indexed-no-pack-one-cb",
    ...
  },
  "evidence": {
    "measurements": [...],
    "target_tps": 60.0
  }
}
```

- **`shader_hash`** — SHA-256 prefix of all compiled Metal sources. If this
  does not match the running binary's shader hash (check with
  `dismantle shader-hash`), the runtime logs a warning and falls back to the
  compiled-in default. It does **not** crash.
- **`selected.moe_schedule`** — the active MoE dispatch strategy:
  - `"indexed-no-pack-one-cb"` — production default; batched expert GEMV,
    no byte-packing, one command buffer per MoE block.
  - `"single-kernel"` — opt-in; strict fused FlashMoE
    (`moe_block_fused_v2lite`). Correct at atol < 1e-3 but ~90× slower on
    single-token decode due to redundant intermediate compute. Use for
    batch/prefill experiments or future redesign work.
- **`tensor_layout_hash`** — hash of tensor names, shapes, dtypes, and byte
  offsets. Changing quantization or re-exporting the GGUF will invalidate
  this and trigger a fallback.

## M3 Pro 18 GB reference profile

The repository ships `profiles/deepseek-v2-lite-q4.m3pro18.json` with
measured results from a 14-core M3 Pro, 18 GB unified memory. This is the
reference hardware for all v0.1.0 benchmarks:

- **dec\_tps**: 1.61 median (DeepSeek-V2-Lite Q4\_K\_M, 3 trials × 64 tokens)
- **Selected**: `gpu-greedy-frontier` / `indexed-no-pack-one-cb`
- **Score**: 157 (deterministic scoring over 5 candidates)

The profile is valid as long as the shader hash matches your binary. Check:

```sh
dismantle shader-hash
# compare with profiles/deepseek-v2-lite-q4.m3pro18.json .shader_hash
```

To regenerate for your hardware:

```sh
dismantle autotune \
  --weights models/deepseek-v2-lite-q4.gguf \
  --profile <your-hardware-name> \
  --max-hours 8 \
  --out profiles/deepseek-v2-lite-q4.<your-hardware>.json
```

See [docs/autotune.md](autotune.md) for full autotune documentation.
