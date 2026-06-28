# Roadmap

Hawking is a from-scratch LLM inference engine for Apple Silicon, written in
Rust with hand-written Metal kernels. It runs quantized GGUF models end to end
on the GPU, with no PyTorch, llama.cpp, or BLAS. The work ahead is about
pushing quality, memory, and context length further on Apple hardware, with the
heavy runs on an Apple M2 Max Mac Studio (96 GB).

## Now (works today)
- Dense Qwen2.5 forward pass on Metal, GGUF-native, with Q4_K / Q6_K kernels,
  RoPE / RMSNorm / attention / GPU sampling, and an OpenAI-compatible server.
- CPU-to-GPU numerical-parity tests and golden-hash regression gates running in
  CI on real Apple Silicon.

## Next
- RWKV-7 (SSM) long-context path: flat-cost decode with no KV-cache wall.
- Per-channel int4 KV cache, to cut KV memory by roughly three quarters.
- Close the remaining decode-throughput gap to llama.cpp / MLX (kernel and
  scheduling work).

## Later
- Condense: an out-of-core, memory-budgeted low-bit compression pipeline that
  can quantize models too large to hold resident, so a single Mac can prepare
  and serve models well beyond its own memory.
- Broader verified architecture coverage (MoE, Mamba2, more dense families)
  under the same correctness-before-speed gates.
