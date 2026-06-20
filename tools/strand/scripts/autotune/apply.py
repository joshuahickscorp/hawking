#!/usr/bin/env python3
"""The demo consumer — print the tuned env/flags for each known launcher.

THE OPT-IN PATTERN (the autotuner contract, see sweep.py): the tuner writes
research/tuned-profile.toml and changes NOTHING else. A consumer opts in by
reading the profile, checking the machine fingerprint matches the machine it is
about to run on, and applying the recommendation to its own launch line. This
script is that pattern, executable: it prints ready-to-paste env prefixes /
flags per launcher, falling back loudly to the hand-picked default whenever a
tunable is not TUNED or the fingerprint mismatches.

Usage:
    python3 tools/autotune/apply.py            # human-readable, all launchers
    python3 tools/autotune/apply.py --env      # just the env prefix line
"""

import argparse
import os
import sys
import tomllib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import DEFAULT_OUT, machine_identity  # noqa: E402


def load(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def pick(prof, name):
    """(value, provenance) — tuned best if TUNED, else the declared default."""
    t = prof.get("tunable", {}).get(name)
    if t is None:
        return None, "not in profile"
    if t.get("status") == "TUNED" and "best" in t:
        return t["best"], "tuned"
    return t.get("default"), f"default ({t.get('status', '?')})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default=DEFAULT_OUT)
    ap.add_argument("--env", action="store_true",
                    help="print only the env-prefix line (for eval in launchers)")
    args = ap.parse_args()

    if not os.path.exists(args.profile):
        print(f"no profile at {args.profile} — run tools/autotune/sweep.py "
              f"(or ops/autotune.sh) first; launchers keep their hand-picked defaults.")
        return 1
    prof = load(args.profile)

    _, fp_now = machine_identity()
    fp_prof = prof.get("meta", {}).get("machine_fingerprint", "?")
    fp_ok = fp_now == fp_prof
    if not fp_ok:
        print(f"WARNING: profile fingerprint {fp_prof} != this machine {fp_now} — "
              f"recommendations below are for ANOTHER machine; defaults are safer.")

    dec, dec_src = pick(prof, "decode_rayon_threads")
    enc, enc_src = pick(prof, "encode_threads")
    s37, s37_src = pick(prof, "interleave_s_k3l7")
    s212, s212_src = pick(prof, "interleave_s_k2l12")
    kd, kd_src = pick(prof, "kd_chunk")
    omp, omp_src = pick(prof, "eval_omp_threads")
    gpu, gpu_src = pick(prof, "eval_gpu_gb")

    if args.env:
        print(f"RAYON_NUM_THREADS={dec}")
        return 0

    print(f"tuned profile: {args.profile} (git {prof['meta'].get('git', '?')}, "
          f"generated {prof['meta'].get('generated', '?')}, advisory, "
          f"fingerprint {'MATCH' if fp_ok else 'MISMATCH'})")
    print()
    print("# decode (any decode_q12_par consumer: runtime loads, gate bins)")
    print(f"  env: RAYON_NUM_THREADS={dec}        # {dec_src}")
    print()
    print("# requant / quantize-model (PV rung-3 inner loop, strand-7b-ppl.sh Step 1)")
    print(f"  flag: --threads {enc}               # {enc_src}; cap at the tensor-job count")
    print()
    print("# interleaved single-core decode (compile-time const generic — pick the")
    print("# monomorphization at the call site, decode_q12_interleave::<S>)")
    print(f"  S = {s37}  for k=3 L=7   # {s37_src}")
    print(f"  S = {s212}  for k=2 L=12  # {s212_src}")
    print()
    print("# scripts/strand-qat.py KD chunk (tunable defined, sweep disabled: needs MPS)")
    print(f"  KD_CHUNK = {kd}                     # {kd_src}")
    print()
    print("# eval launchers (defined, sweeps disabled: torch-stack / pod-side)")
    print(f"  env: OMP_NUM_THREADS={omp}          # {omp_src} (strand-7b-ppl.sh eval)")
    print(f"  env: EVAL_GPU_GB={gpu}              # {gpu_src} (ops/pod-chain-v2.sh, POD ONLY)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
