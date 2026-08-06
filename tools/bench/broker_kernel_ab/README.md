# Broker kernel A/B harness

Scaffold for correctness-first kernel tuning on the open-source-model brokers
(DeepSeek-V4-Flash Terra, Qwen3-Coder Luna, Frankenstein body).

## Rules

1. **Parity before speed.** A candidate that is faster but fails the sealed
   oracle is rejected. There is no serve-path promote path in this scaffold.
2. **No forward-lane edits.** A/B lives here and in
   `crates/hawking-core/{src/broker_kernel_ab.rs,examples/broker_kernel_ab_harness.rs}`.
   Do not patch `gravity_deepseek_v4*.rs` from this harness.
3. **No TPS claims** from these runs. Use `tools/bench/coexist_bench.sh` /
   clean-room scripts only after a candidate is manually integrated and sealed.
4. **Costs come from receipts**, not vibes. See `receipt_costs.json`.

## Run (Rust scaffold)

```sh
# Dry-run: prints registry + gate policy; no Metal required.
cargo run -p hawking-core --example broker_kernel_ab_harness -- --dry-run

# Simulated A/B (synthetic vectors): demonstrates reject-without-parity.
cargo run -p hawking-core --example broker_kernel_ab_harness -- \
  --kernel fp4_expert_matvec --simulate-parity fail

# Simulated A/B with parity pass + speed win → CandidateReady (not promoted).
cargo run -p hawking-core --example broker_kernel_ab_harness -- \
  --kernel act_quant --simulate-parity pass --simulate-speed-ratio 0.85
```

## Tests

```sh
cargo test -p hawking-core --test broker_kernel_ab_gate
```

## Wiring a real oracle later

1. Load sealed authority receipt / component oracle (existing
   `gravity_deepseek_v4_*_oracle` / sweep examples).
2. Dispatch authority kernel and candidate on the **same** device buffers.
3. Score with `hawking_core::numeric_parity` V2.1 (or byte-exact hashes for
   act_quant).
4. Feed `parity_pass` + optional GPU p50 ratio into
   `hawking_core::broker_kernel_ab::decide_promotion`.
5. Write a receipt JSON; never flip serve defaults from this tool.
