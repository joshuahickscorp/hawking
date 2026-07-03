#!/usr/bin/env python3.12
"""procure.py — fastest-SOTA model procurement for the frontier run. Downloads are the ONE tedious
bottleneck left (~4 TB of bf16 frontier parents), so this forces the fastest path Hugging Face has
and saturates whatever link you are on instead of being software-bottlenecked.

The stack, fastest-first (all already installed on the box: hf 1.13, hf_transfer 0.1.9, hf_xet):
  1. hf_xet          - the Xet content-addressed backend. Chunk-level dedup + parallel range gets;
                        the default for modern large repos, faster and more resumable than plain LFS.
  2. HF_HUB_ENABLE_HF_TRANSFER=1 - the Rust chunked transfer accelerator (parallel byte ranges per
                        file). Saturates a fast link where the pure-Python downloader caps low.
  3. --max-workers N - parallel FILES (shards) in flight at once. A frontier parent is 100s of
                        shards; pulling many concurrently overlaps per-file setup latency.
Together these turn "download bandwidth" from a software cap into a pure link cap. On gigabit fiber
(~125 MB/s) a 1.3 TB 671B lands in ~3 hours; the default single-stream path would take far longer.

The procurement STRATEGY (not just speed):
  - CONDENSE targets (the whole ladder + frontier) need the bf16 PARENT to bake and to measure the
    floor against the f16 baseline - no shortcut, download bf16.
  - The one real reducer for the giant frontier: STREAM-CONDENSE. Because block-wise condense
    processes one shard at a time, you can download shard N, bake it, delete the bf16 shard, fetch
    N+1 - so you never need the full 1.3 TB resident on disk AND the bake compute hides under
    download I/O wait. procure.py --stream emits that fused plan (the pager-free frontier makes this
    clean: the .tq is what stays, the bf16 is transient). This is the advanced path; the serve build
    must support shard-incremental bake for it to run unattended.

Usage:
  procure.py --check                       # confirm hf_transfer + hf_xet are active; print the env
  procure.py <label|hf_id> [--dir DIR]     # download one, fastest path (label resolves via FRONTIER)
  procure.py --all-frontier [--link-mbps 1000]  # queue all frontier parents serially, disk-aware, w/ ETA
  procure.py --stream <label>              # emit the stream-condense fused plan (download+bake+delete)
"""
import sys, os, subprocess, shutil, importlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

# the frontier manifest (kept in sync with studio_run.FRONTIER: label -> (hf_id, bf16_GB, local dir))
FRONTIER = {
    "235B-A22B": ("Qwen/Qwen3-235B-A22B",           470, "scratch/qwen3-235b-a22b"),
    "405B":      ("meta-llama/Llama-3.1-405B-Instruct", 810, "scratch/llama31-405b"),
    "671B":      ("deepseek-ai/DeepSeek-V3",         1342, "scratch/deepseek-v3"),
    "744B":      ("zai-org/GLM-4.5",                 1488, "scratch/glm-744b"),
    # the doctor ladder parents (smaller, most already staged)
    "14B":       ("Qwen/Qwen2.5-14B-Instruct",       28, "scratch/qwen-14b"),
    "32B":       ("Qwen/Qwen2.5-32B-Instruct",       64, "scratch/qwen-32b"),
    "72B":       ("Qwen/Qwen2.5-72B-Instruct",      140, "scratch/qwen-72b"),
}
FAST_ENV = {"HF_HUB_ENABLE_HF_TRANSFER": "1"}       # Rust chunked transfer; hf_xet auto-activates if installed
MAX_WORKERS = str(min(16, (os.cpu_count() or 8) * 2))


def _have(mod):
    try:
        importlib.import_module(mod); return True
    except Exception:
        return False


def check():
    hf = shutil.which("hf") or shutil.which("huggingface-cli")
    xet, xfer = _have("hf_xet"), _have("hf_transfer")
    print(f"hf CLI          : {hf or 'MISSING (pip install huggingface_hub[cli])'}", file=sys.stderr)
    print(f"hf_xet backend  : {'ACTIVE (Xet dedup + parallel range gets)' if xet else 'MISSING (pip install hf_xet)'}", file=sys.stderr)
    print(f"hf_transfer     : {'ACTIVE (Rust chunked accelerator)' if xfer else 'MISSING (pip install hf_transfer)'}", file=sys.stderr)
    print(f"env to export   : HF_HUB_ENABLE_HF_TRANSFER=1   --max-workers {MAX_WORKERS}", file=sys.stderr)
    ok = bool(hf) and xet and xfer
    print(f"=> procurement path is {'FASTEST-SOTA (link-bound, not software-bound)' if ok else 'DEGRADED - install the missing piece above'}", file=sys.stderr)
    return ok


def _resolve(label_or_id):
    if label_or_id in FRONTIER:
        return FRONTIER[label_or_id]
    # a raw HF id
    dirn = "scratch/" + label_or_id.split("/")[-1].lower()
    return (label_or_id, None, dirn)


def eta_hours(gb, link_mbps):
    mbytes_s = link_mbps / 8.0                       # Mbps -> MB/s (link ceiling; hf_transfer saturates it)
    return round(gb * 1000 / mbytes_s / 3600, 1) if mbytes_s else None


def download(label_or_id, dir_override=None, dry=False):
    hf_id, gb, dirn = _resolve(label_or_id)
    dirn = dir_override or dirn
    env = {**os.environ, **FAST_ENV}
    cmd = ["hf", "download", hf_id, "--local-dir", dirn, "--max-workers", MAX_WORKERS]
    print(f"[procure] {label_or_id} -> {dirn}" + (f"  (~{gb} GB bf16)" if gb else ""), file=sys.stderr)
    print(f"[procure] HF_HUB_ENABLE_HF_TRANSFER=1 {' '.join(cmd)}", file=sys.stderr)
    if dry:
        return 0
    if os.path.isdir(dirn) and os.listdir(dirn):
        print(f"[procure] {dirn} already populated - hf resumes/skips complete shards", file=sys.stderr)
    return subprocess.run(cmd, env=env).returncode


def all_frontier(link_mbps):
    tot = sum(gb for (_id, gb, _d) in FRONTIER.values() if gb)
    print(f"[procure] frontier manifest ({len(FRONTIER)} parents, ~{tot/1000:.1f} TB bf16 total):", file=sys.stderr)
    for lbl, (hf_id, gb, dirn) in sorted(FRONTIER.items(), key=lambda x: x[1][1] or 0):
        staged = "STAGED" if os.path.isdir(dirn) and os.listdir(dirn) else f"~{eta_hours(gb, link_mbps)}h @ {link_mbps}Mbps"
        print(f"  {lbl:12s} {gb:>5} GB  {hf_id:38s} -> {dirn}  [{staged}]", file=sys.stderr)
    print(f"# total ~{eta_hours(tot, link_mbps)}h at {link_mbps} Mbps (serial); download biggest LAST so the "
          f"ladder + condense start immediately on the small ones. Disk: 8 TB holds all + the .tq outputs.", file=sys.stderr)
    print(f"# to actually run one: procure.py <label>   (fastest path auto-applied)", file=sys.stderr)


def stream_plan(label):
    hf_id, gb, dirn = _resolve(label)
    print(f"# STREAM-CONDENSE plan for {label} ({gb} GB bf16 -> a low-bpw .tq): download+bake+delete per", file=sys.stderr)
    print(f"# shard so peak disk ~= one shard + the .tq, and bake compute hides under download I/O wait.", file=sys.stderr)
    print(f"#   for each shard S in {hf_id}:", file=sys.stderr)
    print(f"#     hf download {hf_id} <S> --local-dir {dirn}   # HF_HUB_ENABLE_HF_TRANSFER=1", file=sys.stderr)
    print(f"#     quantize-model --in {dirn}/<S> --append-tq {dirn}.tq   # shard-incremental bake", file=sys.stderr)
    print(f"#     rm {dirn}/<S>                                # reclaim the bf16 shard", file=sys.stderr)
    print(f"# GATE: needs the baker's shard-incremental --append-tq mode (serve-build item). Until then,", file=sys.stderr)
    print(f"# download the full parent (8 TB fits) and bake whole. The fused path is the disk+time optimizer.", file=sys.stderr)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if a == "--check":
        sys.exit(0 if check() else 1)
    elif a == "--all-frontier":
        link = int(sys.argv[sys.argv.index("--link-mbps") + 1]) if "--link-mbps" in sys.argv else 1000
        all_frontier(link)
    elif a == "--stream":
        stream_plan(sys.argv[2])
    elif a == "--help":
        print(__doc__)
    else:
        d = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else None
        sys.exit(download(a, d, dry="--dry" in sys.argv))
