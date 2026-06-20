#!/usr/bin/env python3
"""batch_ceiling.py — arithmetic-intensity ceiling oracle for continuous-batching DECODE.

Direction B (aggregate-tps prize). CPU-only, no deps. Models the bus-bandwidth
ceiling on aggregate decode throughput at batch B for Qwen2.5-3B-Q4_K_M on a
single ~150 GB/s unified bus (M3 Pro, 18 GB).

PHYSICS
-------
One decode step at batch B reads, over the shared bus:
  weight_bytes   = 1.93e9  (all Q4_K weights, read ONCE — amortized across the
                            whole batch by a weight-stationary kernel like v3w)
  kv_bytes       = B * KV_read(seqlen)   (each sequence has its OWN KV cache)
where, for f16 KV with GQA:
  KV_read(L) = 2 (K and V) * n_layers * n_kv_heads * head_dim * L * 2 bytes

  time_per_step  = (weight_bytes + kv_bytes) / BW
  aggregate_tps  = B / time_per_step      (B tokens produced per step)
  speedup(B)     = aggregate_tps(B) / aggregate_tps(1)

The ceiling: as B grows, kv_bytes (which scales with B) eventually rivals then
dominates weight_bytes (fixed). Once kv_bytes ~ weight_bytes the bus is
"re-saturated" by KV traffic and additional concurrency stops buying throughput.

Qwen2.5-3B-Q4_K_M config (read from the GGUF metadata):
  n_layers=36, hidden=2048, n_heads=16, n_kv_heads=2 (GQA 8:1), head_dim=128.
  => kv_dim = n_kv_heads*head_dim = 256 elems per K (and per V) per token per layer.
"""

# ---- model / hardware constants ----
BW = 150e9                # bytes/sec, one shared unified bus (M3 Pro)
WEIGHT_BYTES = 1.93e9     # Q4_K_M weight read per decode step (amortized once)
N_LAYERS = 36
N_KV_HEADS = 2            # GQA — this is the lever; MHA would be 16 (8x KV)
HEAD_DIM = 128
KV_DTYPE_BYTES = 2        # f16 KV
EMBED_LMHEAD_BYTES = 0.0  # folded into WEIGHT_BYTES (read once); left explicit=0


def kv_read_per_seq(seqlen: int) -> float:
    """Bytes one sequence's KV cache contributes to ONE decode step's bus read.
    Reads the full [0..seqlen) K and V for every layer."""
    return 2.0 * N_LAYERS * N_KV_HEADS * HEAD_DIM * seqlen * KV_DTYPE_BYTES


def step_bytes(B: int, seqlen: int) -> float:
    return WEIGHT_BYTES + B * kv_read_per_seq(seqlen)


def aggregate_tps(B: int, seqlen: int) -> float:
    t = step_bytes(B, seqlen) / BW
    return B / t


def per_seq_tps(B: int, seqlen: int) -> float:
    return aggregate_tps(B, seqlen) / B


def kv_saturation_B(seqlen: int) -> float:
    """B at which B*KV_read == WEIGHT_BYTES (KV traffic equals weight traffic).
    Past this, >50% of the bus is KV and the weight-amortization win is mostly spent."""
    return WEIGHT_BYTES / kv_read_per_seq(seqlen)


def report():
    seqlens = [512, 2048]
    batches = [1, 2, 4, 8, 16]
    print("=" * 78)
    print("batch_ceiling.py — Qwen2.5-3B-Q4_K_M continuous-batching DECODE ceiling")
    print(f"  BW={BW/1e9:.0f} GB/s  W={WEIGHT_BYTES/1e9:.3f} GB  "
          f"n_layers={N_LAYERS} n_kv_heads={N_KV_HEADS} head_dim={HEAD_DIM} f16-KV")
    print("=" * 78)

    # KV per seq per step at each seqlen.
    for L in seqlens:
        kvb = kv_read_per_seq(L)
        print(f"\nseqlen={L}: KV/seq/step = {kvb/1e6:.2f} MB "
              f"({kvb/WEIGHT_BYTES*100:.2f}% of weight read)  "
              f"| KV==W at B={kv_saturation_B(L):.1f}")

    for L in seqlens:
        base = aggregate_tps(1, L)
        print(f"\n--- seqlen={L}  (single-stream agg_tps baseline = {base:.2f}) ---")
        print(f"{'B':>3} {'step_MB':>9} {'kv_frac':>8} {'agg_tps':>9} "
              f"{'per_seq':>8} {'speedup':>8} {'marg/step':>10}")
        prev = None
        for B in batches:
            sb = step_bytes(B, L)
            kvf = (B * kv_read_per_seq(L)) / sb
            agg = aggregate_tps(B, L)
            ps = per_seq_tps(B, L)
            sp = agg / base
            # marginal speedup gained per added concurrency unit since prev row
            marg = "" if prev is None else f"+{(agg-prev[1])/ (B-prev[0]):.1f}/seq"
            print(f"{B:>3} {sb/1e6:>9.1f} {kvf*100:>7.1f}% {agg:>9.1f} "
                  f"{ps:>8.2f} {sp:>7.2f}x {marg:>10}")
            prev = (B, agg)

    # ---- cross-check vs published vllm-mlx anchors ----
    # Anchor: ~2.6x at 16-concurrent for an 8B on M4 Max (546 GB/s);
    #         ~3.7x for 0.6B same setup. Our model, applied to THOSE configs,
    #         should land in the same neighborhood — validates the shape.
    print("\n" + "=" * 78)
    print("CROSS-CHECK vs vllm-mlx published anchors (same arithmetic, their configs)")
    print("=" * 78)
    cross_check()

    # ---- the verdict line ----
    print("\n" + "=" * 78)
    print("VERDICT (Qwen2.5-3B-Q4_K_M, 150 GB/s)")
    print("=" * 78)
    for L in seqlens:
        sp16 = aggregate_tps(16, L) / aggregate_tps(1, L)
        sp8 = aggregate_tps(8, L) / aggregate_tps(1, L)
        knee = kv_saturation_B(L)
        print(f"  seqlen={L:>4}: B=8 -> {sp8:.2f}x   B=16 -> {sp16:.2f}x   "
              f"KV==W knee at B~={knee:.0f}")


def model_speedup(weight_bytes, kv_per_seq_per_step, B):
    """Generic speedup(B) for any model: weight read once + B*KV."""
    agg1 = 1.0 / (weight_bytes + 1 * kv_per_seq_per_step)
    aggB = B / (weight_bytes + B * kv_per_seq_per_step)
    return aggB / agg1


def cross_check():
    # vllm-mlx setup: M4 Max ~546 GB/s. KV traffic scales the same way; the
    # speedup ratio is BW-independent (BW cancels), so we only need
    # weight_bytes and kv_per_seq_per_step in consistent units. Use a
    # representative decode context; the anchor papers run ~moderate context.
    #
    # 8B model (Llama-3.1-8B-class), 4-bit MLX: ~4.3 GB weights @4bpw.
    #   Llama-3.1-8B: 32 layers, 8 KV heads, head_dim 128 -> kv_dim 1024.
    # 0.6B model (Qwen3-0.6B-class), 4-bit: ~0.4 GB weights.
    #   Qwen3-0.6B: 28 layers, 8 KV heads (GQA), head_dim 128 -> hmm small;
    #   use published-ish 0.5B: 24 layers, 2 KV heads, head_dim 64.
    def kv_bytes(n_layers, n_kv_heads, head_dim, L, dt=2):
        return 2.0 * n_layers * n_kv_heads * head_dim * L * dt

    scenarios = [
        # name, weight_bytes, (n_layers,n_kv,head_dim), pub_anchor_at16
        ("8B  4-bit (anchor ~2.6x@16)", 4.3e9, (32, 8, 128), 2.6),
        ("0.6B 4-bit (anchor ~3.7x@16)", 0.40e9, (28, 8, 128), 3.7),
        ("0.5B 4-bit (alt small geom)", 0.40e9, (24, 2, 64), 3.7),
    ]
    for L in (1024, 2048):
        print(f"\n  [assumed decode context seqlen={L}]")
        print(f"  {'model':<32}{'pred@16':>9}{'anchor':>8}{'pred@8':>8}{'pred@4':>8}")
        for name, wb, (nl, nkv, hd), anchor in scenarios:
            kpp = kv_bytes(nl, nkv, hd, L)
            s16 = model_speedup(wb, kpp, 16)
            s8 = model_speedup(wb, kpp, 8)
            s4 = model_speedup(wb, kpp, 4)
            print(f"  {name:<32}{s16:>8.2f}x{anchor:>7.1f}x{s8:>7.2f}x{s4:>7.2f}x")
    print("\n  NOTE: the published anchors include kernel-launch / attention-"
          "overhead\n  realism that a pure-bus model omits, so a pure-bus model is an"
          "\n  UPPER bound — it should sit at or above the measured anchor. The 8B "
          "\n  case (low KV-fraction, big weights) is the cleanest comparison.")


if __name__ == "__main__":
    report()
