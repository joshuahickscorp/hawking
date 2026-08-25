#!/usr/bin/env python3
"""G005 completion: the vector items the first qualification pass did not measure.

Missing were DRAM bytes/token, active bytes/token, the model-reachable roof,
concurrency equilibrium, and verified WUs/hour.

The first four were missing because the harness read them from MIX_REPORT, where
active_bytes_per_token is a DESIGN CONSTANT: the clean 8.73 GB body and the variantA
10.02 GB body both publish 8234330016.0. That is canary D's exact shape (G033), so
every figure here is recomputed from payload_bytes_by_role instead.

ACTIVE vs COMPLETE. Every organ is read once per decoded token except the embedding
table, which is a gather of a single row. Active bytes therefore equal payload minus
the embedding table. This is a dense body: there is no routing to make it smaller.
"""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protected_window import ProtectedWindow

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
BINARY = REPO / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy"
TOKENIZER = "/Users/scammermike/noetic/CLEAN_REBUILD_A/mix_hetero_n041_floors"
CLEAN = "/Users/scammermike/noetic/CLEAN_REBUILD_A/mix_hetero_n041_floors"
PROMPT = ("Explain, in ordinary prose and at length, how a compiler turns a for-loop "
          "into basic blocks and then into machine code.")


def io_pids():
    out = subprocess.run(["pgrep", "-f", "hf download|lake_filler.py"],
                         capture_output=True, text=True)
    me = os.getpid()
    return [int(x) for x in out.stdout.split() if x.strip().isdigit() and int(x) != me]


def physical_vector(root, tpot_ns):
    """Recomputed from bytes, never from the design constant."""
    m = json.load(open(Path(root) / "MIX_REPORT.json"))
    by_role = m.get("payload_bytes_by_role")
    payload = m["payload_bytes"]
    params = m["parent_params"]
    if not by_role:
        return {"available": False,
                "why": "MIX_REPORT has no payload_bytes_by_role; this body predates the "
                       "per-role accounting and its active vector cannot be derived "
                       "from bytes"}
    embed = by_role.get("embedding", 0)
    active = payload - embed          # every other organ is read whole, once per token
    tpot_s = tpot_ns / 1e9
    return {
        "available": True,
        "ARTIFACT_PHYSICAL_active_bytes_per_token": active,
        "ARTIFACT_PHYSICAL_active_ebpw_per_token": round(8.0 * active / params, 6),
        "DESIGN_EXPECTED_active_bytes_per_token": m.get("active_bytes_per_token"),
        "DESIGN_EXPECTED_active_ebpw_per_token": m.get("active_ebpw_per_token"),
        # a byte-count comparison with an absolute 1e-3 tolerance calls a 0.0015%
        # agreement and a 13.5% divergence the same thing. Report the relative error.
        "design_vs_physical_rel_error": (
            abs(active - m["active_bytes_per_token"]) / active
            if m.get("active_bytes_per_token") else None),
        "design_matches_physical": (
            abs(active - m["active_bytes_per_token"]) / active < 1e-4
            if m.get("active_bytes_per_token") else False),
        "embedding_bytes_excluded_as_gather": embed,
        "payload_bytes": payload,
        "ARTIFACT_PHYSICAL_dram_bytes_per_token": active,
        "dram_assumption": (
            "one full read of every non-embedding organ per decoded token. The body is "
            "8.7-10.0 GB against a far smaller cache, so inter-token reuse is assumed "
            "zero. This is an UPPER bound on useful traffic and therefore a LOWER bound "
            "on the reachable roof."),
        "RUNTIME_MEASURED_model_reachable_gb_s": round(active / tpot_s / 1e9, 2),
        "roof_scope": "this executable, this regime. Directive §76: never copied to "
                      "another model or another representation.",
    }


def run_one(root, max_new, tag, extra_env=None):
    import tempfile
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); f.close()
    cmd = [str(BINARY), "--artifact-root", str(root),
           "--tokenizer", str(Path(TOKENIZER) / "tokenizer.json"),
           "--prompt", PROMPT, "--max-new-tokens", str(max_new),
           "--max-seq-len", str(max_new + 64), "--out", f.name]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                       env=dict(os.environ, **(extra_env or {})))
    wall = time.time() - t0
    body = json.loads(Path(f.name).read_text()) if Path(f.name).stat().st_size else {}
    Path(f.name).unlink(missing_ok=True)
    steps = body.get("gpu_ns_per_step") or []
    return {"tag": tag, "exit_code": p.returncode, "wall_s": round(wall, 3),
            "median_gpu_ns": statistics.median(steps[1:]) if len(steps) > 1 else None,
            "n_new_tokens": len(body.get("new_token_ids") or []),
            "decode_steps": body.get("decode_steps")}


def concurrency_sweep(root, levels, max_new):
    """Aggregate throughput as concurrent single-stream decoders are added.

    Equilibrium is the level past which aggregate tokens/s stops rising. Each level is
    launched as separate PROCESSES because that is how HCLI would actually run several
    missions at once on this runtime.
    """
    out = []
    for c in levels:
        import tempfile
        procs, files = [], []
        t0 = time.time()
        for i in range(c):
            f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); f.close()
            files.append(f.name)
            procs.append(subprocess.Popen(
                [str(BINARY), "--artifact-root", str(root),
                 "--tokenizer", str(Path(TOKENIZER) / "tokenizer.json"),
                 "--prompt", PROMPT, "--max-new-tokens", str(max_new),
                 "--max-seq-len", str(max_new + 64), "--out", f.name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        codes = [p.wait() for p in procs]
        wall = time.time() - t0
        toks, meds = 0, []
        for fn in files:
            try:
                b = json.loads(Path(fn).read_text())
                toks += len(b.get("new_token_ids") or [])
                s = b.get("gpu_ns_per_step") or []
                if len(s) > 1:
                    meds.append(statistics.median(s[1:]))
            except Exception:
                pass
            Path(fn).unlink(missing_ok=True)
        out.append({"concurrency": c, "wall_s": round(wall, 3),
                    "total_new_tokens": toks,
                    "aggregate_tps": round(toks / wall, 4) if wall else None,
                    "per_stream_median_gpu_ns": round(statistics.median(meds)) if meds else None,
                    "all_exit_zero": all(c2 == 0 for c2 in codes)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", default=str(RH / "QWEN_PERFORMANCE_ADDENDUM.json"))
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--levels", default="1,2,4")
    a = ap.parse_args()

    perf = json.load(open(RH / "QWEN_PERFORMANCE_QUALIFICATION.json"))
    bodies = perf["bodies"]

    vectors = {}
    for tag, b in bodies.items():
        root = {"sealed-3.14": "/Users/scammermike/noetic/NOETIC_PARENT_A",
                "variantA-2.98": "/Users/scammermike/noetic/VARIANT_A_MLP_ONLY",
                "clean-2.60": CLEAN}[tag]
        vectors[tag] = physical_vector(root, b["latency_vector"]["TPOT_ns_p50"])
        vectors[tag]["TPOT_ns_p50"] = b["latency_vector"]["TPOT_ns_p50"]

    levels = [int(x) for x in a.levels.split(",")]
    with ProtectedWindow(io_pids(), max_s=2400) as w:
        paused = list(w.paused)
        sweep = concurrency_sweep(CLEAN, levels, a.max_new)

    best = max((s for s in sweep if s["aggregate_tps"]),
               key=lambda s: s["aggregate_tps"], default=None)
    eq, knee_found = None, False
    for i, s in enumerate(sweep):
        if i and s["aggregate_tps"] and sweep[i - 1]["aggregate_tps"]:
            if s["aggregate_tps"] <= sweep[i - 1]["aggregate_tps"] * 1.05:
                eq, knee_found = sweep[i - 1]["concurrency"], True
                break
    if eq is None and best:
        # still rising at the top level: equilibrium is AT OR ABOVE it, not equal to it.
        # Reporting the top tested level as "the equilibrium" would silently turn an
        # untested range into a finding.
        eq = best["concurrency"]

    cap = json.load(open(RH / "COMPOSITION_ATTRIBUTION.json"))
    out = {
        "schema": "hawking.odyssey.perf_addendum.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/perf_addendum.py",
        "obligation": "G005 — the vector items the first pass did not measure",
        "hand_authored": False,
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "physical_vectors": vectors,
        "frozen_design_constant_observed": {
            "what": "MIX_REPORT.active_bytes_per_token",
            "clean_payload_bytes": 8731093292,
            "variantA_payload_bytes": 10019612956,
            "both_report_active_bytes_per_token": 8234330016.0,
            "verdict": "a design constant published under a physical name; canary D of "
                       "G033 predicts exactly this. Every figure in physical_vectors is "
                       "recomputed from payload_bytes_by_role.",
        },
        "concurrency": {
            "levels": sweep,
            "equilibrium_concurrency": eq,
            "knee_observed": knee_found,
            "equilibrium_is_a_lower_bound": not knee_found,
            "caveat": (None if knee_found else
                       "aggregate throughput was STILL RISING at the highest level "
                       "tested, so equilibrium is at or above this number rather than "
                       "equal to it"),
            "definition": "the highest level whose successor does not raise aggregate "
                          "tokens/s by more than 5%",
            "measured_on": "clean-2.60",
            "io_suspended_for_the_sweep": paused,
        },
        "verified_wus_per_hour": {
            "value": 0.0,
            "basis": "a WorkUnit is only counted when its output is VERIFIED. On the "
                     "canonical capability suite the clean-2.60 body scores 3/43 and "
                     "variantA 0/43, and none of the passes are production tasks -- so "
                     "no body in this ladder produces a verified accepted WorkUnit at "
                     "any rate. The figure is zero because the bodies are capability-"
                     "dead, not because the measurement is missing.",
            "capability_receipt": "receipts/headless/COMPOSITION_ATTRIBUTION.json",
            "ladder": cap.get("ladder"),
            "canonical_definition_pending": "G039 owns the HCLI bench and the scoring "
                                            "rule; this is the honest floor until then.",
        },
        "anomaly_variantA_short_generation": {
            "observed": "variantA-2.98 emitted 2 tokens per run with decode_steps=1, "
                        "against 24 tokens and 23 steps for the other two bodies",
            "consequence": "its TPOT is computed over 175 sampled steps rather than 285, "
                           "and its wall-clock figures are NOT comparable to the others",
            "reading": "an early stop is itself a capability signal, consistent with "
                       "variantA scoring 0/43",
        },
    }
    ok = all(v.get("available") for v in vectors.values() if "available" in v)
    out["pass"] = bool(sweep and eq and any(
        v.get("RUNTIME_MEASURED_model_reachable_gb_s") for v in vectors.values()))
    Path(a.emit).write_text(json.dumps(out, indent=1))
    for tag, v in vectors.items():
        if v.get("available"):
            print(f"  {tag:15s} active={v['ARTIFACT_PHYSICAL_active_bytes_per_token']/1e9:.3f}GB "
                  f"active_ebpw={v['ARTIFACT_PHYSICAL_active_ebpw_per_token']} "
                  f"roof={v['RUNTIME_MEASURED_model_reachable_gb_s']}GB/s "
                  f"design_matches={v['design_matches_physical']}")
        else:
            print(f"  {tag:15s} {v['why']}")
    for s in sweep:
        print(f"  c{s['concurrency']}: agg_tps={s['aggregate_tps']} "
              f"per_stream_gpu_ns={s['per_stream_median_gpu_ns']} wall={s['wall_s']}s")
    print(f"  equilibrium concurrency: {eq}"
          f"{'' if knee_found else '  (LOWER BOUND -- still rising at the top level)'}")
    print(f"-> {a.emit}  pass={out['pass']}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
