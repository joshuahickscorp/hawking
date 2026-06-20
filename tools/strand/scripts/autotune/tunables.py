"""The TUNABLE registry — every hand-picked performance constant, declared.

Each tunable is a declaration, not a behavior change: the sweep engine
(sweep.py) measures candidates and writes research/tuned-profile.toml; nothing
in the codebase reads the profile unless a consumer opts in (apply.py prints
the opt-in env/flags per launcher). THE CONTRACT: the tuner NEVER changes
defaults in code.

A tunable dict:
  name        str   unique key (becomes [tunable.<name>] in the profile)
  status      "enabled" | "disabled"   (disabled = defined, documented, not run)
  kind        "sweep"  -> run cmd(value) once per candidate value
              "batch"  -> run cmd() once; parse() returns {value: metric}
  values      candidate list (ints)
  default     the current hand-picked constant (what we ship today)
  direction   "max" | "min"  (is a bigger metric better?)
  metric      human description of what parse() returns
  requires    repo-relative paths that must exist or the tunable is SKIPped
              (sibling-wave gate bins may be absent mid-rebuild: degrade, never fail)
  cmd         sweep: cmd(value) -> (argv, env_overrides)
              batch: cmd()      -> (argv, env_overrides)
  parse       sweep: parse(text) -> float | None
              batch: parse(text) -> {value: float}
  guard       guard(text) -> (ok: bool, reason: str). A sample only counts if its
              guard PASSES — for decode paths that is the gate bin's own
              bit-identity assertion line (the bins refuse to print perf on any
              Q12 drift, will.md §5.2); for encode it is paired with invariance.
  invariance  optional invariance(text) -> str. The returned key must be
              IDENTICAL across every sample of the tunable (e.g. per-tensor
              rel-RMS fingerprint: a thread count that changes the encoded
              result is a bug, not a tuning point). None = not applicable.
  notes       why this constant exists / where it is consumed
"""

import re

# ---------------------------------------------------------------- helpers

_FLOAT = r"([0-9]+(?:\.[0-9]+)?)"


def _line_metric(label, text):
    """First float on the line starting with `label` (column = Mw/s)."""
    for line in text.splitlines():
        if line.strip().startswith(label):
            m = re.search(_FLOAT, line[len(label) + line.find(label):])
            if m:
                return float(m.group(1))
    return None


def _decode_speed_cmd(threads):
    return (["target/release/gate-decode-speed"], {"RAYON_NUM_THREADS": str(threads)})


def _decode_speed_parse(text):
    return _line_metric("parallel (rayon)", text)


def _decode_speed_guard(text):
    ok = "bit-identity @ bench size" in text and "... OK" in text
    return ok, "gate-decode-speed bit-identity line" if ok else "bit-identity line missing"


def _interleave_cmd():
    return (["target/release/gate-interleave"], {})


def _interleave_guard(text):
    n = text.count("determinism: 8/8 variants byte-identical")
    ok = n >= 2  # both deploy points must pass their identity gate
    return ok, f"gate-interleave determinism 8/8 x{n}" if ok else f"determinism line x{n} (<2)"


def _interleave_section(text, header):
    """Slice the per-point section ('3-bit deploy' / '2-bit reopen')."""
    start = text.find(f"== {header}")
    if start < 0:
        return ""
    nxt = text.find("\n== ", start + 1)
    return text[start:nxt] if nxt > 0 else text[start:]


def _interleave_parse(header):
    def parse(text):
        out = {}
        for line in _interleave_section(text, header).splitlines():
            m = re.match(r"\s*interleave S=(\d+)\s+" + _FLOAT + r"\s+Mw/s", line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
        return out
    return parse


_QM = "target/release/quantize-model"
_QM_MODEL = "scratch/qwen-05b/model.safetensors"


def _encode_cmd(threads):
    argv = [_QM, "--in", _QM_MODEL, "--measure-only", "--bits", "2", "--l", "4",
            "--only", "self_attn.q_proj", "--threads", str(threads)]
    # STRAND_NO_GPU forces the SIMD CPU encode path (encode.rs): the tuner runs
    # CPU-only by contract, and quant-is-CPU is the ops truth anyway (will.md §7).
    return (argv, {"STRAND_NO_GPU": "1"})


def _encode_parse(text):
    m = re.search(r"MEASURE-ONLY: (\d+) tensors quantized in " + _FLOAT + "s", text)
    return float(m.group(2)) if m else None


def _encode_guard(text):
    m = re.search(r"MEASURE-ONLY: (\d+) tensors quantized", text)
    if not m:
        return False, "MEASURE-ONLY completion line missing"
    n = int(m.group(1))
    return (n == 24), f"{n} tensors quantized" + ("" if n == 24 else " (expected 24)")


def _encode_invariance(text):
    """Fingerprint of WHAT was encoded: per-tensor (name,bits,bpw,rel-RMS), sorted
    (worker completion order varies with thread count; the results must not)."""
    rows = sorted(re.findall(
        r"\[done \d+/\d+\] (\S+)\s+bits=(\d+) bpw=([0-9.]+) rel-RMS=([0-9.]+)%", text))
    agg = re.search(r"AGGREGATE effective bpw = [0-9.]+ over \d+ quantized weights"
                    r" ; weighted rel-RMS = [0-9.]+%", text)
    return repr(rows) + "|" + (agg.group(0) if agg else "NO-AGG")


# ---------------------------------------------------------------- the registry

TUNABLES = [
    {
        "name": "decode_rayon_threads",
        "status": "enabled",
        "kind": "sweep",
        "values": [1, 2, 4, 6, 8, 10, 12],
        "default": 12,  # rayon's default = available_parallelism
        "direction": "max",
        "metric": "block-parallel decode throughput, Mw/s ('parallel (rayon)' row of gate-decode-speed; 67.9M-weight ffn_down shape, 3-bit)",
        "requires": ["target/release/gate-decode-speed", ],
        "cmd": _decode_speed_cmd,
        "parse": _decode_speed_parse,
        "guard": _decode_speed_guard,
        "invariance": None,  # the gate bin asserts PAR==SIMD==fast itself before any perf line
        "notes": "Consumers: any decode_q12_par caller (runtime, gate bins) via RAYON_NUM_THREADS. "
                 "The M3 Pro is 6P+6E; the best thread count is not obviously 12.",
    },
    {
        "name": "interleave_s_k3l7",
        "status": "enabled",
        "kind": "batch",
        "values": [2, 4, 6, 8, 16],
        "default": 4,
        "direction": "max",
        "metric": "single-core interleaved decode Mw/s at k=3 L=7 (gate-interleave '3-bit deploy' point)",
        "requires": ["target/release/gate-interleave"],
        "cmd": _interleave_cmd,
        "parse": _interleave_parse("3-bit deploy"),
        "guard": _interleave_guard,
        "invariance": None,  # gate asserts all S byte-identical to decode_q12_fast pre-perf
        "notes": "S is a const generic (decode_q12_interleave::<S>) — compile-time. The profile "
                 "records the per-machine winner; consumers pick the monomorphization at the call site.",
    },
    {
        "name": "interleave_s_k2l12",
        "status": "enabled",
        "kind": "batch",
        "values": [2, 4, 6, 8, 16],
        "default": 4,
        "direction": "max",
        "metric": "single-core interleaved decode Mw/s at k=2 L=12 (gate-interleave '2-bit reopen' point)",
        "requires": ["target/release/gate-interleave"],
        "cmd": _interleave_cmd,
        "parse": _interleave_parse("2-bit reopen"),
        "guard": _interleave_guard,
        "invariance": None,
        "notes": "Same gate run as interleave_s_k3l7 (the engine memoizes identical commands); "
                 "the 16KB L=12 LUT changes the L1 story, so the best S may differ per point.",
    },
    {
        "name": "encode_threads",
        "status": "enabled",
        "kind": "sweep",
        "values": [1, 2, 4, 6, 8, 10, 12],
        "default": 12,  # quantize-model default = available_parallelism
        "direction": "min",
        "metric": "wall seconds to measure-quantize 24 q_proj tensors (19.3M weights) at bits=2 l=4, CPU-only (STRAND_NO_GPU=1)",
        "requires": ["target/release/quantize-model", "scratch/qwen-05b/model.safetensors"],
        "cmd": _encode_cmd,
        "parse": _encode_parse,
        "guard": _encode_guard,
        "invariance": _encode_invariance,
        "notes": "Consumers: quantize-model --threads (requant inner loop of PV rung 3; "
                 "strand-7b-ppl.sh Step 1). Parallelism is per-tensor, so the winner transfers as "
                 "threads = min(n_jobs, best). Invariance guard: identical per-tensor rel-RMS "
                 "fingerprint across all thread counts (threads must never change the encode).",
    },
    # ---------------- defined but DISABLED (documented, not run) ----------------
    {
        "name": "kd_chunk",
        "status": "disabled",
        "kind": "sweep",
        "values": [32, 64, 128, 256],
        "default": 128,
        "direction": "min",
        "metric": "seconds/step of the KD loss chunk loop in scripts/strand-qat.py (KD_CHUNK constant)",
        "requires": ["scripts/strand-qat.py"],
        "cmd": None, "parse": None, "guard": None, "invariance": None,
        "notes": "DISABLED: measuring it requires an MPS training step — excluded while the box is "
                 "contended and by the no-MPS rule (QAT runs ALONE on the 18GB box, will.md §7 freeze "
                 "trap). Sweep shape when enabled: 5 steps of strand-qat.py --kd per chunk size, "
                 "metric = median step seconds, guard = loss bit-identical across chunk sizes "
                 "(chunking changes only the reduction batching).",
    },
    {
        "name": "eval_omp_threads",
        "status": "disabled",
        "kind": "sweep",
        "values": [4, 6, 8, 12],
        "default": 8,  # the strand-act2 drivers hand-pick OMP_NUM_THREADS=8
        "direction": "min",
        "metric": "seconds for a fixed-window CPU PPL eval (strand-7b-ppl.sh eval stage) per OMP_NUM_THREADS",
        "requires": ["scripts/strand-7b-ppl.sh"],
        "cmd": None, "parse": None, "guard": None, "invariance": None,
        "notes": "DISABLED: needs the torch eval stack + a recon model on disk (minutes per sample, "
                 "contends with science runs). Guard when enabled: PPL identical across thread counts.",
    },
    {
        "name": "eval_gpu_gb",
        "status": "disabled",
        "kind": "sweep",
        "values": [14, 16, 18, 20],
        "default": 18,  # ops/pod-chain-v2.sh hand-picks EVAL_GPU_GB=18
        "direction": "min",
        "metric": "seconds per offload PPL eval (ops/eval-ppl.py) per EVAL_GPU_GB on the pod",
        "requires": [],
        "cmd": None, "parse": None, "guard": None, "invariance": None,
        "notes": "DISABLED: cloud offload split — not measurable on this Apple machine; "
                 "the local profile must never claim it. Listed so the registry is the complete "
                 "census of hand-picked constants.",
    },
]
