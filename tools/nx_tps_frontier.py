#!/usr/bin/env python3
"""What TPS does this machine's measured weight-movement roof permit?

Every number here is read from a receipt, never assumed. The output is an
UPPER BOUND on TPS: it prices weight movement only, and charges zero for
non-GEMV compute, dispatch topology, encode, and host tail. A candidate that
cannot clear 100 TPS in this model cannot clear it in reality.

  ./tools/nx_tps_frontier.py --out receipts/ascent-2026-08-16/NX_TPS_FRONTIER.json
"""
import argparse
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROOF = ROOT / "receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.json"
LEDGER = ROOT / "receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json"
AMORT = ROOT / "receipts/ascent-2026-08-16/NX_MATMUL_K_AMORTIZATION.json"

# Language parameter count. BPW denominator, always (ledger ASSUMPTIONS).
N_PARAMS = 26_895_998_464
G0_COMPLETE_BPW = 4.255954555664

# Grouped-codec footprint, group size 64: code bytes + one f16 scale.
# q4: 32 B codes + 2 B scale = 34 B / 64 elem. q3: 24 + 2 = 26 B / 64 elem.
GROUP = 64
CODEC_BYTES_PER_GROUP = {"q4": 34, "q3": 26, "q2": 18}

# Measured interleaved-rANS coded bits per element on the real q3 stream
# (directive section 4). Scale bytes are NOT entropy coded, so only the code
# half shrinks.
RANS_CODED_BITS_PER_ELEM = {"L8": 2.5591, "L32": 2.5825, "L64": 2.6138}


def codec_bytes_per_elem(codec):
    return CODEC_BYTES_PER_GROUP[codec] / GROUP


def rans_bytes_per_elem(interleave):
    """q3 codes replaced by a rANS stream; the f16 group scale still lands."""
    return RANS_CODED_BITS_PER_ELEM[interleave] / 8.0 + 2.0 / GROUP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    roof = json.loads(ROOF.read_text())
    ledger = json.loads(LEDGER.read_text())

    adj = roof["adjudication"]
    kernel_roof_gb_s = adj["kernel_roof_gb_s"]
    achieved_gb_s = adj["defended_gb_s"]
    published_peak_gb_s = roof["hardware"]["published_peak_gb_s"]
    g0_weight_ns = adj["defended_time_ns"]

    org = roof["byte_count_adjudication"]["gemv_payload_breakdown"]
    # Organ classes, all currently q4 group-64 in G0.
    organs = {
        "mlp": org["mlp_bytes"],
        "linear_attn": org["linear_attn_bytes"],
        "full_attn": org["full_attn_bytes"],
        "lm_head": org["lm_head_bytes"],
    }
    gemv_payload = org["gemv_payload"]
    assert abs(sum(organs.values()) - gemv_payload) < 1, "organ census must close"

    # Elements per organ, back-derived from the q4 footprint that produced
    # these byte counts. This is the invariant that survives a codec change.
    q4_bpe = codec_bytes_per_elem("q4")
    elems = {k: v / q4_bpe for k, v in organs.items()}

    production_gpu_ns = ledger["median_gpu_ns"]
    non_gemv_ns = production_gpu_ns - g0_weight_ns

    def candidate(name, per_organ, note=""):
        """per_organ maps organ -> bytes-per-element for that organ."""
        by_organ = {k: elems[k] * per_organ[k] for k in elems}
        total_bytes = sum(by_organ.values())
        # Complete BPW: every stored bit over N, including the embed table and
        # the norms the GEMV kernel does not stream. Held at the G0 ratio so
        # the accounting stays complete, not payload-only.
        complete_overhead = (
            G0_COMPLETE_BPW * N_PARAMS / 8.0
        ) - gemv_payload  # embed table + norms + headers, codec-invariant
        complete_bpw = (total_bytes + complete_overhead) * 8.0 / N_PARAMS
        out = {
            "name": name,
            "note": note,
            "active_gemv_bytes": total_bytes,
            "active_gemv_gb": total_bytes / 1e9,
            "complete_bpw": complete_bpw,
            "bytes_per_organ": by_organ,
        }
        for label, gb_s in (
            ("at_measured_kernel_roof", kernel_roof_gb_s),
            ("at_measured_production_rate", achieved_gb_s),
            ("at_published_peak", published_peak_gb_s),
        ):
            weight_ns = total_bytes / gb_s  # bytes / (B/ns) == bytes/(GB/s)
            out[label] = {
                "gb_s": gb_s,
                "weight_ms": weight_ns / 1e6,
                "tps_if_non_gemv_were_free": 1e9 / weight_ns,
                "tps_with_todays_non_gemv": 1e9 / (weight_ns + non_gemv_ns),
                "clears_100_tps_even_with_free_non_gemv": (1e9 / weight_ns) >= 100.0,
            }
        return out

    def uniform(codec):
        return {k: codec_bytes_per_elem(codec) for k in elems}

    def endpoints_q4(body_bpe):
        p = {k: body_bpe for k in elems}
        p["lm_head"] = codec_bytes_per_elem("q4")
        return p

    cands = [
        candidate("G0 uniform-q4-v1", uniform("q4"), "current speed leader, measured"),
        candidate(
            "mixed-q3mlp-q3attn-v1",
            endpoints_q4(codec_bytes_per_elem("q3")),
            "current coherent density floor, 10/10 capability",
        ),
        candidate(
            "rANS-L32 over q3 body",
            endpoints_q4(rans_bytes_per_elem("L32")),
            "projected: interleaved rANS, consume-direct, endpoints still q4",
        ),
        candidate(
            "rANS-L8 over q3 body",
            endpoints_q4(rans_bytes_per_elem("L8")),
            "projected: smallest measured interleave",
        ),
        candidate("uniform-q2 (DEAD)", uniform("q2"), "capability dead, listed as a bound only"),
    ]

    # Inverse question: what does 100 TPS actually demand?
    demands = []
    for label, gb_s in (
        ("at_measured_kernel_roof", kernel_roof_gb_s),
        ("at_published_peak", published_peak_gb_s),
    ):
        for non_gemv_ms in (0.0, 2.0, float(non_gemv_ns) / 1e6):
            budget_ns = 10_000_000.0 - non_gemv_ms * 1e6
            if budget_ns <= 0:
                demands.append(
                    {
                        "roof": label,
                        "non_gemv_ms_allowed": non_gemv_ms,
                        "verdict": "IMPOSSIBLE: non-GEMV alone exceeds the 10 ms token",
                    }
                )
                continue
            max_bytes = budget_ns * gb_s
            overhead = (G0_COMPLETE_BPW * N_PARAMS / 8.0) - gemv_payload
            demands.append(
                {
                    "roof": label,
                    "gb_s": gb_s,
                    "non_gemv_ms_allowed": non_gemv_ms,
                    "max_active_gemv_gb": max_bytes / 1e9,
                    "max_complete_bpw": (max_bytes + overhead) * 8.0 / N_PARAMS,
                }
            )

    # MEASURED amortization, not projection. NX_MATMUL_K_AMORTIZATION.json times
    # the R x K tiled geo_tpr64 kernel against the K=1 kernel on the same codes.
    measured_amort = None
    if AMORT.exists():
        am = json.loads(AMORT.read_text())
        best = None
        for res in am["results"]:
            for row in res.get("by_rk", []):
                if not row.get("per_position_control_holds"):
                    continue
                if best is None or row["amortization_x"] > best["amortization_x"]:
                    best = {**row, "payload_mib": res["payload_mib"]}
        if best:
            k1 = next(
                r for res in am["results"] for r in res["by_k"]
                if r["k"] == 1 and res["payload_mib"] == best["payload_mib"]
            )
            measured_amort = {
                "best_tile": f"R={best['r']} K={best['k']}",
                "kernel": best["kernel"],
                "amortization_x": best["amortization_x"],
                "k1_code_gb_s": k1["achieved_gb_s"],
                "payload_mib": best["payload_mib"],
                "per_position_control_holds": best["per_position_control_holds"],
            }
            gb_s = k1["achieved_gb_s"]
            amort = best["amortization_x"]
            rows_out = []
            for c in cands:
                per_token_ns = c["active_gemv_bytes"] / gb_s / amort
                budget_ns = 10_000_000.0 - per_token_ns
                rows_out.append({
                    "candidate": c["name"],
                    "complete_bpw": c["complete_bpw"],
                    "weight_ms_per_token": per_token_ns / 1e6,
                    "non_gemv_ms_budget_for_100_tps": budget_ns / 1e6,
                    "reachable": budget_ns > 0,
                    "non_gemv_cut_needed_x": (non_gemv_ns / budget_ns) if budget_ns > 0 else None,
                })
            measured_amort["what_100_tps_needs_now"] = rows_out

    # One full weight sweep per token is an ASSUMPTION of the current genome,
    # not a law. If K tokens are emitted per sweep (multi-token head,
    # speculative verify, matryoshka draft-then-verify), weight movement is
    # amortized K ways while per-position non-GEMV work is not.
    multi_token = []
    for cand_name, per_organ in (
        ("G0 uniform-q4-v1", uniform("q4")),
        ("mixed-q3mlp-q3attn-v1", endpoints_q4(codec_bytes_per_elem("q3"))),
        ("rANS-L32 over q3 body", endpoints_q4(rans_bytes_per_elem("L32"))),
    ):
        total_bytes = sum(elems[k] * per_organ[k] for k in elems)
        sweep_ns = total_bytes / kernel_roof_gb_s
        row = {"candidate": cand_name, "sweep_ms": sweep_ns / 1e6, "by_k": []}
        for k in (1, 2, 4, 8):
            weight_per_token_ns = sweep_ns / k
            budget_ns = 10_000_000.0 - weight_per_token_ns
            row["by_k"].append(
                {
                    "k": k,
                    "weight_ms_per_token": weight_per_token_ns / 1e6,
                    "non_gemv_ms_budget_for_100_tps": budget_ns / 1e6,
                    "reachable": budget_ns > 0,
                    "non_gemv_reduction_needed_x": (
                        (non_gemv_ns / budget_ns) if budget_ns > 0 else None
                    ),
                }
            )
        multi_token.append(row)

    doc = {
        "schema": "hawking.nos.nx_tps_frontier.v1",
        "measured_amortization": measured_amort,
        "multi_token_escape": multi_token,
        "multi_token_note": (
            "K is tokens emitted per full weight sweep. K>1 requires a verified "
            "multi-token or speculative path (G091/G141); the arithmetic here says "
            "only what it would be worth, not that it works. Non-GEMV is charged "
            "per token at today's rate, which is pessimistic for the parts that "
            "amortize and optimistic for none."
        ),
        "question": "Which representations can physically reach 100 TPS on this machine?",
        "method": (
            "Weight movement only. Organ byte census and both bandwidth roofs are read "
            "from HONEST_ROOF_WEIGHT_ADDRESSING.json; non-GEMV time is production GPU "
            "minus defended weight-addressing time from QWEN38_TOKEN_NS_LEDGER.json. "
            "TPS reported here is an UPPER BOUND: dispatch topology tax, encode, host "
            "tail and every non-GEMV kernel are charged at zero in the headline column."
        ),
        "inputs": {
            "kernel_roof_gb_s": kernel_roof_gb_s,
            "achieved_production_gb_s": achieved_gb_s,
            "published_peak_gb_s": published_peak_gb_s,
            "g0_weight_addressing_ns": g0_weight_ns,
            "g0_production_gpu_ns": production_gpu_ns,
            "g0_non_gemv_ns": non_gemv_ns,
            "n_params": N_PARAMS,
            "g0_complete_bpw": G0_COMPLETE_BPW,
            "roof_source_commit": roof.get("source_head_at_measurement"),
            "ledger_source_commit": ledger.get("commit"),
            "roof_contamination_note": roof.get("contamination_note"),
        },
        "candidates": cands,
        "what_100_tps_demands": demands,
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip(),
    }

    print(f"kernel roof {kernel_roof_gb_s:.2f} GB/s | production {achieved_gb_s:.2f} GB/s "
          f"| peak {published_peak_gb_s:.0f} GB/s")
    print(f"G0 non-GEMV remainder = {non_gemv_ns/1e6:.3f} ms/token\n")
    print(f"{'candidate':<26} {'BPW':>7} {'GB/tok':>7} {'wt ms':>7} {'TPS*':>7} {'TPS now':>8}  100?")
    for c in cands:
        r = c["at_measured_kernel_roof"]
        print(
            f"{c['name']:<26} {c['complete_bpw']:>7.4f} {c['active_gemv_gb']:>7.3f} "
            f"{r['weight_ms']:>7.3f} {r['tps_if_non_gemv_were_free']:>7.2f} "
            f"{r['tps_with_todays_non_gemv']:>8.2f}  "
            f"{'YES' if r['clears_100_tps_even_with_free_non_gemv'] else 'NO'}"
        )
    print("\n* TPS at the measured kernel roof charging ZERO for non-GEMV work.\n")
    print("100 TPS demands:")
    for d in demands:
        if "verdict" in d:
            print(f"  {d['roof']:<26} non-GEMV {d['non_gemv_ms_allowed']:.3f} ms -> {d['verdict']}")
        else:
            print(
                f"  {d['roof']:<26} non-GEMV {d['non_gemv_ms_allowed']:.3f} ms -> "
                f"<= {d['max_active_gemv_gb']:.3f} GB/token, complete BPW <= {d['max_complete_bpw']:.4f}"
            )

    print("\nmulti-token escape (K tokens per full weight sweep):")
    for row in doc["multi_token_escape"]:
        print(f"  {row['candidate']} (sweep {row['sweep_ms']:.3f} ms)")
        for b in row["by_k"]:
            if not b["reachable"]:
                print(f"    K={b['k']}: weight {b['weight_ms_per_token']:.3f} ms/token -> 100 TPS IMPOSSIBLE")
            else:
                print(
                    f"    K={b['k']}: weight {b['weight_ms_per_token']:.3f} ms/token, "
                    f"non-GEMV budget {b['non_gemv_ms_budget_for_100_tps']:.3f} ms "
                    f"({b['non_gemv_reduction_needed_x']:.2f}x cut needed)"
                )

    if measured_amort:
        print(f"\nMEASURED amortization: {measured_amort['best_tile']} = "
              f"{measured_amort['amortization_x']:.2f}x "
              f"(K=1 code rate {measured_amort['k1_code_gb_s']:.1f} GB/s, controls hold)")
        for r in measured_amort["what_100_tps_needs_now"]:
            if not r["reachable"]:
                print(f"  {r['candidate']:<26} weight {r['weight_ms_per_token']:.3f} ms -> 100 TPS IMPOSSIBLE")
            else:
                print(f"  {r['candidate']:<26} weight {r['weight_ms_per_token']:.3f} ms/token, "
                      f"non-GEMV budget {r['non_gemv_ms_budget_for_100_tps']:.3f} ms "
                      f"({r['non_gemv_cut_needed_x']:.2f}x cut needed)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
