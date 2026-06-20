#!/usr/bin/env python3
import argparse
import collections
import json
import sys
from pathlib import Path


# M3 Pro 18GB unified memory: 150 GB/s vendor spec, ~120-135 sustained.
M3_PRO_PEAK_GBPS = 150.0
M3_PRO_SUSTAINED_GBPS = 130.0


# Per-token total weight-read footprint, Q4_K_M, by model. Nearly the whole
# weight file streams per decode token at batch=1.
V2_LITE_BYTES_PER_TOKEN = int(1.82 * 1024 ** 3)  # ~1.82 GB (DeepSeek-V2-Lite)
QWEN3B_BYTES_PER_TOKEN = int(1.93 * 1024 ** 3)   # ~1.93 GB (Qwen2.5-3B GGUF)
MODEL_BYTES = {
    "qwen3b": QWEN3B_BYTES_PER_TOKEN,
    "v2lite": V2_LITE_BYTES_PER_TOKEN,
}


def find_first_key(obj, key):
    """First value for `key` anywhere in a nested JSON doc, else None."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_first_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_first_key(v, key)
            if r is not None:
                return r
    return None


def find_samples_and_tps(obj, path=""):
    """Walk nested JSON looking for dispatch_samples + decode_tps."""
    samples, dec_tps, dec_ms = None, None, None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "dispatch_samples" and samples is None and isinstance(v, list):
                samples = v
            elif k == "decode_tps" and dec_tps is None and isinstance(v, (int, float)):
                dec_tps = v
            elif k == "decode_ms" and dec_ms is None and isinstance(v, (int, float)):
                dec_ms = v
            else:
                sub_s, sub_t, sub_m = find_samples_and_tps(v)
                samples = samples or sub_s
                dec_tps = dec_tps or sub_t
                dec_ms = dec_ms or sub_m
    elif isinstance(obj, list):
        for v in obj:
            sub_s, sub_t, sub_m = find_samples_and_tps(v)
            samples = samples or sub_s
            dec_tps = dec_tps or sub_t
            dec_ms = dec_ms or sub_m
    return samples, dec_tps, dec_ms


def find_completion_tokens(obj):
    """First completion_tokens encountered, walking nested."""
    if isinstance(obj, dict):
        if "completion_tokens" in obj:
            return obj["completion_tokens"]
        for v in obj.values():
            r = find_completion_tokens(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_completion_tokens(v)
            if r is not None:
                return r
    return None


def summarize(samples):
    """Aggregate samples by kernel name."""
    by_k = collections.defaultdict(lambda: {"n": 0, "cpu_us": 0, "gpu_us": 0})
    for s in samples:
        k = s["kernel_name"]
        by_k[k]["n"] += 1
        by_k[k]["cpu_us"] += s.get("wall_us", 0) or 0
        by_k[k]["gpu_us"] += s.get("gpu_us") or 0
    return by_k


def by_layer(samples):
    """Aggregate samples by (layer_hint, kernel_name)."""
    by_lk = collections.defaultdict(lambda: {"n": 0, "gpu_us": 0})
    for s in samples:
        layer = s.get("layer_hint")
        k = s["kernel_name"]
        by_lk[(layer, k)]["n"] += 1
        by_lk[(layer, k)]["gpu_us"] += s.get("gpu_us") or 0
    return by_lk


def print_per_kernel(by_k, total_gpu_us, tokens):
    print(f"\n{'kernel':45s} {'n':>5s}  {'gpu_us_total':>13s}  {'us/call':>9s}  {'us/token':>9s}  {'% GPU':>7s}")
    print("-" * 100)
    rows = sorted(by_k.items(), key=lambda kv: -kv[1]["gpu_us"])
    for k, v in rows:
        gpu = v["gpu_us"]
        if gpu == 0 and v["n"] == 0:
            continue
        n = v["n"]
        pct = (gpu / total_gpu_us * 100) if total_gpu_us else 0
        per_call = gpu / max(n, 1)
        per_token = gpu / max(tokens, 1)
        flag = ""
        if k == "other":
            flag = "  <-- UNMAPPED; expand static_kernel_name"
        print(f"{k:45s} {n:5d}  {gpu:13d}  {per_call:9.1f}  {per_token:9.0f}  {pct:7.2f}%{flag}")


def print_bandwidth(total_gpu_us, tokens, model_bytes_per_token):
    if total_gpu_us == 0 or tokens == 0:
        return
    per_token_us = total_gpu_us / tokens
    per_token_s = per_token_us / 1_000_000
    # Effective bandwidth = bytes / time
    eff_gbps = model_bytes_per_token / per_token_s / 1024**3
    print(f"\n--- Bandwidth ---")
    print(f"per-token GPU time:       {per_token_us/1000:.2f} ms")
    print(f"per-token model reads:    {model_bytes_per_token/1024**3:.2f} GiB (V2-Lite Q4_K_M)")
    print(f"effective GPU bandwidth:  {eff_gbps:.1f} GiB/s")
    print(f"M3 Pro vendor peak:       {M3_PRO_PEAK_GBPS:.0f} GiB/s ({eff_gbps/M3_PRO_PEAK_GBPS*100:.0f}% util)")
    print(f"M3 Pro sustained anchor:  {M3_PRO_SUSTAINED_GBPS:.0f} GiB/s ({eff_gbps/M3_PRO_SUSTAINED_GBPS*100:.0f}% util)")
    print(f"NOTE: split-CB mode adds per-dispatch sync overhead; production")
    print(f"      single-CB-per-block bandwidth will be HIGHER than this.")


def print_by_layer(by_lk):
    print(f"\n--- Per-layer breakdown (top kernel per layer) ---")
    layers = sorted({l for (l, _) in by_lk.keys() if l is not None})
    for layer in layers[:10]:  # cap at 10 to keep output readable
        items = [(k, v["gpu_us"], v["n"]) for (l, k), v in by_lk.items() if l == layer]
        items.sort(key=lambda x: -x[1])
        total = sum(g for _, g, _ in items)
        if total == 0:
            continue
        print(f"\nlayer {layer}  total {total/1000:.1f} ms")
        for k, g, n in items[:5]:
            print(f"  {k:42s} n={n:3d}  gpu_us={g:8d}  ({g/total*100:5.1f}%)")
    if len(layers) > 10:
        print(f"\n... ({len(layers) - 10} more layers; rerun with --by-layer-all for full)")


def methodology_gate(samples, by_k, total_gpu_us, tokens,
                     model_bytes_per_token, dec_ms, doc):
    """Bible §1 — four physical invariants that refuse a lying result.

    Returns (failures, notes, eff_gbps). A non-empty `failures` means the
    measurement violates physics and must not drive a decision.
    """
    failures, notes = [], []

    # INV1 — busy-time bandwidth cannot exceed hardware peak. This alone kills
    # any "N% idle" claim that would imply pushing the weights at >100% of peak
    # during the busy window. Also catches a halved token count (gpu_us/token
    # too small ⇒ inflated BW) — the exact artifact that misled twice.
    eff_gbps = None
    if total_gpu_us > 0 and tokens > 0:
        per_token_s = (total_gpu_us / tokens) / 1_000_000
        eff_gbps = model_bytes_per_token / per_token_s / 1024 ** 3
        if eff_gbps > M3_PRO_PEAK_GBPS * 1.001:
            failures.append(
                f"INV1: busy-time BW {eff_gbps:.1f} > peak {M3_PRO_PEAK_GBPS:.0f} "
                f"GiB/s — bytes/token too high or gpu_us undercounted "
                f"(token miscount? wrong --model?).")

    # INV2 — per-kernel accounting closes: the unmapped 'other' bucket must be
    # small, and Σ per-kernel must match a measured GPU-busy if the trace has one.
    if total_gpu_us > 0:
        other = by_k.get("other", {}).get("gpu_us", 0)
        other_share = other / total_gpu_us
        if other_share > 0.05:
            failures.append(
                f"INV2: unmapped 'other' bucket {other_share*100:.1f}% > 5% "
                f"— expand static_kernel_name; accounting drift.")
    gpu_busy = find_first_key(doc, "gpu_busy_us")
    if gpu_busy and total_gpu_us > 0:
        drift = abs(total_gpu_us - gpu_busy) / gpu_busy
        if drift > 0.05:
            failures.append(
                f"INV2: Σkernel {total_gpu_us} vs measured gpu_busy {gpu_busy} "
                f"drift {drift*100:.1f}% > 5%.")

    # INV3 — token count must come from sample_* dispatches, never completion_tokens.
    traced = sum(1 for x in samples
                 if x.get("kernel_name", "").startswith("sample_"))
    if traced == 0:
        notes.append(
            "INV3: no sample_* dispatch found — token count NOT verified from "
            "argmax; per-token math may be off (the ÷64-vs-÷32 trap).")

    # INV4 — bit-identical greedy parity is a separate Rust gate; surface it.
    parity = find_first_key(doc, "parity_bit_identical")
    if parity is False:
        failures.append("INV4: trace marked parity_bit_identical=false.")
    elif parity is None:
        notes.append(
            "INV4: bit-identical parity not recorded here — enforce via the "
            "correctness parity gate for every kernel change.")

    # Headline busy fraction (the '~85% busy ⇒ kernel-bound' claim).
    if dec_ms and total_gpu_us > 0 and tokens:
        busy_frac = (total_gpu_us / tokens) / (dec_ms * 1000)
        notes.append(f"GPU-busy ≈ {busy_frac*100:.0f}% of decode wall "
                     f"(Bible: ~85% ⇒ kernel-bound, not gap-bound).")
        if busy_frac > 1.05:
            failures.append(
                f"INV1: busy fraction {busy_frac*100:.0f}% > 100% — serial decode "
                f"cannot be busy longer than the wall; measurement is wrong.")
    return failures, notes, eff_gbps


def print_gate(failures, notes, strict):
    print("\n=== METHODOLOGY GATE (Bible §1) ===")
    for n in notes:
        print(f"  [note] {n}")
    if failures:
        for f in failures:
            print(f"  [FAIL] {f}")
        print(f"  GATE: FAILED ({len(failures)} invariant violation(s)) — "
              f"this result must not drive a decision.")
    else:
        print("  GATE: PASS — all checkable invariants hold.")
    return bool(failures) and strict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_json", type=Path)
    ap.add_argument("--by-layer", action="store_true",
                    help="Show per-layer breakdown (default: aggregate)")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON summary instead of human table")
    ap.add_argument("--model", choices=sorted(MODEL_BYTES), default="qwen3b",
                    help="Model footprint preset for bandwidth/gate math (default: qwen3b).")
    ap.add_argument("--model-bytes-per-token", type=float, default=None,
                    help="Override per-token read footprint (bytes). Default: from --model.")
    ap.add_argument("--no-gate", action="store_true",
                    help="Print the §1 gate but do not exit non-zero on violation.")
    args = ap.parse_args()
    if args.model_bytes_per_token is None:
        args.model_bytes_per_token = MODEL_BYTES[args.model]

    if not args.trace_json.exists():
        sys.exit(f"trace not found: {args.trace_json}")
    doc = json.loads(args.trace_json.read_text())
    samples, dec_tps, dec_ms = find_samples_and_tps(doc)
    tokens = find_completion_tokens(doc) or 1
    if not samples:
        sys.exit("no dispatch_samples found in trace JSON (was --trace-json passed?)")

    # The ProdCbGpu counter buffer captures only the tokens it traced, which
    # can be FEWER than completion_tokens (e.g. 32 of 64). `sample_argmax_f32`
    # fires exactly once per greedy decode token, so its count is the true
    # number of traced tokens. Using completion_tokens here would halve
    # per-token GPU time and double the apparent inter-dispatch gap.
    traced = sum(1 for x in samples
                 if x.get("kernel_name", "").startswith("sample_"))
    if traced > 0 and traced != tokens:
        print(f"[note] trace covers {traced} tokens (sample_* count), not "
              f"completion_tokens={tokens}; using {traced} for per-token math.")
        tokens = traced

    by_k = summarize(samples)
    total_gpu_us = sum(v["gpu_us"] for v in by_k.values())
    with_gpu = sum(v["n"] for v in by_k.values() if v["gpu_us"] > 0)
    has_gpu = total_gpu_us > 0

    failures, notes, eff_gbps = methodology_gate(
        samples, by_k, total_gpu_us, tokens,
        args.model_bytes_per_token, dec_ms, doc)

    if args.json:
        out = {
            "trace": str(args.trace_json),
            "samples": len(samples),
            "samples_with_gpu_us": with_gpu,
            "tokens": tokens,
            "decode_tps": dec_tps,
            "total_gpu_us": total_gpu_us,
            "per_token_gpu_us": total_gpu_us / tokens if tokens else 0,
            "by_kernel": {
                k: {"n": v["n"], "gpu_us": v["gpu_us"], "cpu_us": v["cpu_us"]}
                for k, v in by_k.items()
            },
        }
        if has_gpu:
            per_token_s = (total_gpu_us / tokens) / 1_000_000
            out["effective_bandwidth_gibps"] = (
                args.model_bytes_per_token / per_token_s / 1024**3
            )
        out["gate"] = {"passed": not failures, "failures": failures, "notes": notes}
        print(json.dumps(out, indent=2))
        if failures and not args.no_gate:
            sys.exit(2)
        return

    print(f"--- Trace: {args.trace_json} ---")
    print(f"samples:                  {len(samples)}")
    print(f"samples with gpu_us:      {with_gpu}/{len(samples)}")
    print(f"tokens decoded:           {tokens}")
    if dec_tps:
        print(f"decode_tps (this run):    {dec_tps:.2f}")
    print(f"total GPU time:           {total_gpu_us/1000:.2f} ms")
    if tokens:
        print(f"per-token GPU time:       {total_gpu_us/tokens/1000:.2f} ms")
    if not has_gpu:
        print("\nWARNING: no gpu_us values populated. Run with HAWKING_TCB_TRACE=gpu.")
        print("With HAWKING_TCB_TRACE=cpu (or =1) you get CPU encode times only.")

    print_per_kernel(by_k, total_gpu_us, tokens)
    if has_gpu:
        print_bandwidth(total_gpu_us, tokens, args.model_bytes_per_token)
    if args.by_layer:
        print_by_layer(by_layer(samples))

    should_exit = print_gate(failures, notes, strict=not args.no_gate)
    if should_exit:
        sys.exit(2)


if __name__ == "__main__":
    main()
