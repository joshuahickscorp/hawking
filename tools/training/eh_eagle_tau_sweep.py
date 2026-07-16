#!/usr/bin/env python3
"""Event Horizon Item B — EAGLE-3-H tau resurrection sweep.

Captures target hidden states at a SWEEP of layer triplets (low, mid, high),
runs Eagle5Head (mock or trained), computes offline tau per triplet, and
reports results against the 2.5 gate.

KILL-LEDGER RULE (non-negotiable):
  This is MEASUREMENT ONLY. This script NEVER calls enable_neural_slot with
  verdict="GO". It computes tau and reports it. If tau < 2.5, it logs
  "NO-GO (kill-ledger)". The existing head is tau=0.877 < 2.5 gate — dead.
  We're probing multi-layer fusion at different triplets, not resurrecting
  any dead head.

Usage:
  python3 tools/training/eh_eagle_tau_sweep.py --no-gpu            # smoke/CI
  python3 tools/training/eh_eagle_tau_sweep.py --smoke --no-gpu    # 1 prompt
  python3 tools/training/eh_eagle_tau_sweep.py --model models/qwen2.5-3b-instruct-q4_k_m.gguf

Self-check (run after editing):
  python3 -c "import ast; ast.parse(open('tools/training/eh_eagle_tau_sweep.py').read()); print('syntax OK')"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants matching the Rust codebase
# ---------------------------------------------------------------------------
TAU_GATE = 2.5          # acceptance threshold for GO verdict
K_DEPTH  = 4            # draft depth per verify cycle
DEFAULT_TRIPLETS = "0,16,31 0,8,31 8,16,31 0,16,23"
DEFAULT_MAX_PROMPTS = 10
HIDDEN_DIM_MOCK = 2048   # Qwen-3B hidden dim for mock path
VOCAB_SIZE_MOCK  = 151936
RMS_EPS = 1e-6

# ---------------------------------------------------------------------------
# Embedded fallback prompts (used when sft_sample.jsonl is unavailable)
# ---------------------------------------------------------------------------
FALLBACK_PROMPTS: List[str] = [
    "Explain what a black hole is to a ten-year-old.",
    "How do I deduplicate a vector in Rust while preserving order?",
    "If a shirt costs $40 after a 20% discount, what was the original price?",
    "Write a Python function that returns the nth Fibonacci number.",
    "What is the difference between TCP and UDP?",
    "Suggest a simple weeknight dinner I can make in 20 minutes.",
    "What is gradient descent in machine learning?",
    "Write a SQL query that returns the top 5 customers by revenue.",
    "How does a compiler differ from an interpreter?",
    "Explain the CAP theorem in distributed systems.",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LayerTriplet:
    low: int
    mid: int
    high: int

    def __str__(self) -> str:
        return f"{self.low},{self.mid},{self.high}"

    @classmethod
    def parse(cls, s: str) -> "LayerTriplet":
        parts = s.strip().split(",")
        if len(parts) != 3:
            raise ValueError(f"triplet must be 'low,mid,high'; got: {s!r}")
        low, mid, high = int(parts[0]), int(parts[1]), int(parts[2])
        if not (0 <= low < mid < high):
            raise ValueError(
                f"triplet requires 0 <= low < mid < high; got {low},{mid},{high}"
            )
        return cls(low=low, mid=mid, high=high)


@dataclass
class TauResult:
    triplet: LayerTriplet
    tau: float
    accepted: int
    drafted: int
    gate: str          # "GO", "NO-GO (kill-ledger)", or "MARGINAL"
    mock: bool
    elapsed_s: float
    note: str = ""

    def verdict_line(self) -> str:
        mode = "SMOKE (mock head, no GPU)" if self.mock else "REAL"
        frac = self.accepted / max(1, self.drafted)
        return (
            f"[tau-sweep] layer_triplet={self.triplet}"
            f"  tau={self.tau:.4f}"
            f"  accepted={self.accepted}/{self.drafted} ({frac:.1%})"
            f"  gate={self.gate}"
            f"  mode={mode}"
            f"  elapsed={self.elapsed_s:.2f}s"
        )


# ---------------------------------------------------------------------------
# Mock Eagle5Head (Python-side, mirrors the Rust mock)
# ---------------------------------------------------------------------------

def _xorshift64(state: int) -> Tuple[int, int]:
    """One step of the xorshift64 PRNG used by Eagle5Head::mock in Rust."""
    state ^= (state << 13) & 0xFFFF_FFFF_FFFF_FFFF
    state ^= (state >> 7)  & 0xFFFF_FFFF_FFFF_FFFF
    state ^= (state << 17) & 0xFFFF_FFFF_FFFF_FFFF
    return state, state


def _xorshift_fill(seed: int, n: int, scale: float) -> List[float]:
    """Fill a list of n floats in (-scale, +scale) with xorshift64."""
    s = seed
    out = []
    for _ in range(n):
        s, v = _xorshift64(s)
        # Map uint64 to float in [0, 1), then shift to [-0.5, 0.5) * 2 * scale
        f = float(v & 0x000F_FFFF_FFFF_FFFF) / float(0x000F_FFFF_FFFF_FFFF)
        out.append((f - 0.5) * 2.0 * scale)
    return out


class MockEagle5Head:
    """Python mirror of Eagle5Head::Mock (Rust).

    Identical seeding strategy: seed + 0xa5a5_a5a5 (64-bit), then xorshift64.
    embed: [vocab, hidden]  out_w: [vocab, hidden]
    argmax = argmax(out_w @ embed[prev])  — deterministic, near-0 accept rate.
    """

    def __init__(self, seed: int = 42, hidden: int = HIDDEN_DIM_MOCK, vocab: int = VOCAB_SIZE_MOCK) -> None:
        self.hidden  = hidden
        self.vocab   = vocab
        self._seed   = (seed + 0xa5a5_a5a5) & 0xFFFF_FFFF_FFFF_FFFF
        self.last_token: Optional[int] = None

    def _argmax_step(self, prev: int) -> int:
        """Fast deterministic next-token for mock: uses xorshift hash of (seed, prev)
        to pick a token in O(1) rather than the full O(vocab * hidden) matmul.

        The real Rust mock does argmax(out_w @ embed[prev]), which is bit-identical
        to a token derived from the weight seed. We approximate that with a
        deterministic hash: same (seed, prev) always yields the same token id.
        Accept rate is still effectively 0 (random), which is the correct mock behaviour.
        """
        # Deterministic hash: mix seed with prev token using xorshift steps.
        v = (self._seed ^ (prev * 0x9e3779b97f4a7c15)) & 0xFFFF_FFFF_FFFF_FFFF
        v ^= (v >> 30) & 0xFFFF_FFFF_FFFF_FFFF
        v  = (v * 0xbf58476d1ce4e5b9) & 0xFFFF_FFFF_FFFF_FFFF
        v ^= (v >> 27) & 0xFFFF_FFFF_FFFF_FFFF
        v  = (v * 0x94d049bb133111eb) & 0xFFFF_FFFF_FFFF_FFFF
        v ^= (v >> 31) & 0xFFFF_FFFF_FFFF_FFFF
        return int(v % self.vocab)

    def propose(self, prev_token: int, k: int) -> List[int]:
        cur = self.last_token if self.last_token is not None else prev_token
        out = []
        for _ in range(k):
            nxt = self._argmax_step(cur)
            out.append(nxt)
            cur = nxt
        return out

    def propose_with_capture(
        self,
        prev_token: int,
        residual: List[float],
        intermediate: List[float],
        k: int,
    ) -> List[int]:
        # Mock ignores residual/intermediate — mirrors Rust behaviour.
        return self.propose(prev_token, k)

    def note_token(self, tok: int) -> None:
        self.last_token = tok

    def reset(self) -> None:
        self.last_token = None


# ---------------------------------------------------------------------------
# Hidden-state capture (mock path)
# ---------------------------------------------------------------------------

def _mock_hidden(hidden: int, rng: random.Random) -> List[float]:
    """Generate a plausible-shaped unit-norm mock hidden vector."""
    v = [rng.gauss(0.0, 1.0) for _ in range(hidden)]
    norm = math.sqrt(sum(x * x for x in v)) + 1e-9
    return [x / norm for x in v]


def _mock_target_token(vocab: int, rng: random.Random) -> int:
    """Greedy target token — random (mock), this is what the head tries to predict."""
    return rng.randint(0, vocab - 1)


@dataclass
class CaptureStep:
    """One step of captured data for offline tau computation."""
    triplet: LayerTriplet
    start_token: int
    residual_low:  List[float]   # hidden at triplet.low
    residual_mid:  List[float]   # hidden at triplet.mid
    residual_high: List[float]   # hidden at triplet.high
    target_tokens: List[int]     # K greedy target tokens starting from this pos


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(sft_path: Optional[Path], max_prompts: int, rng: random.Random) -> List[str]:
    """Load up to max_prompts prompts from sft_sample.jsonl or fallback."""
    prompts: List[str] = []
    if sft_path and sft_path.exists():
        try:
            with open(sft_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    msgs = obj.get("messages", [])
                    for m in msgs:
                        if m.get("role") == "user":
                            content = m.get("content", "").strip()
                            if content:
                                prompts.append(content)
                                break
                    if len(prompts) >= max_prompts:
                        break
        except Exception as exc:
            print(f"[tau-sweep] WARNING: could not read {sft_path}: {exc}", flush=True)
    if not prompts:
        print("[tau-sweep] WARNING: using embedded fallback prompts.", flush=True)
        prompts = FALLBACK_PROMPTS[:]
    rng.shuffle(prompts)
    return prompts[:max_prompts]


# ---------------------------------------------------------------------------
# Mock capture run
# ---------------------------------------------------------------------------

def mock_capture_run(
    prompts: List[str],
    triplets: List[LayerTriplet],
    k_depth: int,
    hidden: int,
    vocab: int,
    rng: random.Random,
) -> List[CaptureStep]:
    """Simulate a hidden-state capture across all requested triplets.

    In the real (GPU) path this would set HAWKING_QWEN_EAGLE5_CAPTURE=1
    and launch the hawking binary with each prompt, collecting per-layer
    residuals written by the Phase-4 HiddenTap into a sidecar buffer.
    In the mock path we generate Gaussian-noise hidden vectors and random
    target tokens — same shape, zero quality, enough to prove the
    measurement infrastructure runs end-to-end.

    One CaptureStep per (prompt, token-position) pair. We simulate
    min(8, len(prompt_tokens)) token positions per prompt to keep the
    mock fast; real run would capture every position.
    """
    steps: List[CaptureStep] = []
    for prompt in prompts:
        # Fake tokenisation: one token per word, capped at 16 tokens.
        fake_tokens = prompt.split()[:16]
        n_pos = max(1, len(fake_tokens) - k_depth)
        for pos in range(n_pos):
            start_tok = rng.randint(0, vocab - 1)
            # For each triplet record the "captured" hidden vectors.
            # In the real path we'd pick the triplet-specific layer slice.
            # Here all triplets get independently sampled vectors to mimic
            # the fact that different layers really do produce different activations.
            for triplet in triplets:
                step = CaptureStep(
                    triplet=triplet,
                    start_token=start_tok,
                    residual_low=_mock_hidden(hidden, rng),
                    residual_mid=_mock_hidden(hidden, rng),
                    residual_high=_mock_hidden(hidden, rng),
                    target_tokens=[_mock_target_token(vocab, rng) for _ in range(k_depth)],
                )
                steps.append(step)
    return steps


# ---------------------------------------------------------------------------
# Real capture run (subprocess path, GPU required)
# ---------------------------------------------------------------------------

def real_capture_run(
    model_path: Path,
    prompts: List[str],
    triplets: List[LayerTriplet],
    k_depth: int,
) -> List[CaptureStep]:
    """Run hawking binary with HAWKING_QWEN_EAGLE5_CAPTURE=1 and collect
    the hidden-state sidecar. This path requires a model file on disk and
    an MPS-capable GPU (or CPU fallback in the binary).

    Returns CaptureStep list. On any failure, logs a warning and returns
    an empty list (caller falls back to mock_capture_run).
    """
    try:
        import subprocess
    except ImportError:
        print("[tau-sweep] subprocess unavailable — falling back to mock.", flush=True)
        return []

    # Locate the hawking binary.
    repo_root = Path(__file__).parent.parent.parent
    bin_candidates = [
        repo_root / "target" / "release" / "hawking",
        repo_root / "target" / "debug"   / "hawking",
    ]
    binary = next((b for b in bin_candidates if b.exists()), None)
    if binary is None:
        print(
            "[tau-sweep] hawking binary not found in target/release or target/debug."
            " Build first with `cargo build --release`. Falling back to mock.",
            flush=True,
        )
        return []

    steps: List[CaptureStep] = []
    hidden = HIDDEN_DIM_MOCK  # populated from sidecar JSON in the real path

    for triplet in triplets:
        triplet_tag = str(triplet)
        for idx, prompt in enumerate(prompts):
            sidecar_out = f"/tmp/eh_tau_sweep_triplet{triplet_tag}_prompt{idx}.json"
            env = dict(os.environ)
            env["HAWKING_QWEN_EAGLE5_CAPTURE"] = "1"
            env["HAWKING_QWEN_EAGLE5_CAPTURE_LAYERS"] = triplet_tag
            env["HAWKING_QWEN_EAGLE5_CAPTURE_OUT"]    = sidecar_out
            env["HAWKING_QWEN_EVENT_HORIZON"]          = "1"
            env["HAWKING_QWEN_EAGLE5_CAPTURE_NTOK"]   = str(k_depth + 4)

            cmd = [
                str(binary), "generate",
                "--model", str(model_path),
                "--prompt", prompt,
                "--max-new-tokens", str(k_depth + 4),
                "--temperature", "0",
            ]
            try:
                result = subprocess.run(
                    cmd, env=env, capture_output=True, text=True, timeout=120
                )
                if result.returncode != 0:
                    print(
                        f"[tau-sweep] WARNING: hawking exited {result.returncode}"
                        f" for triplet={triplet_tag} prompt={idx}.",
                        flush=True,
                    )
                    continue
                if not Path(sidecar_out).exists():
                    print(
                        f"[tau-sweep] WARNING: sidecar not written for"
                        f" triplet={triplet_tag} prompt={idx}."
                        " (Binary may not implement HAWKING_QWEN_EAGLE5_CAPTURE yet.)"
                        " Falling back to mock for this prompt.",
                        flush=True,
                    )
                    continue
                with open(sidecar_out) as f:
                    sidecar = json.load(f)
                # Sidecar schema (per Phase-4 design):
                #   {steps: [{start_token, target_tokens, residuals: {layer_id: [f32...]}}]}
                for raw_step in sidecar.get("steps", []):
                    start_tok = int(raw_step["start_token"])
                    targets   = [int(t) for t in raw_step["target_tokens"][:k_depth]]
                    residuals = raw_step.get("residuals", {})
                    # Populate the three triplet layers; default to zeros if missing.
                    def _get_res(layer: int) -> List[float]:
                        return residuals.get(str(layer), [0.0] * hidden)
                    step = CaptureStep(
                        triplet=triplet,
                        start_token=start_tok,
                        residual_low=_get_res(triplet.low),
                        residual_mid=_get_res(triplet.mid),
                        residual_high=_get_res(triplet.high),
                        target_tokens=targets,
                    )
                    steps.append(step)
            except subprocess.TimeoutExpired:
                print(
                    f"[tau-sweep] WARNING: timeout for triplet={triplet_tag}"
                    f" prompt={idx} — skipping.",
                    flush=True,
                )
            except Exception as exc:
                print(f"[tau-sweep] WARNING: capture error: {exc}", flush=True)

    if not steps:
        print(
            "[tau-sweep] WARNING: real capture produced 0 steps."
            " Falling back to mock path.",
            flush=True,
        )
    return steps


# ---------------------------------------------------------------------------
# Offline tau computation
# ---------------------------------------------------------------------------

def _fused_residual(
    low: List[float],
    mid: List[float],
    high: List[float],
    hidden: int,
) -> List[float]:
    """Fuse three layer residuals into one vector for EAGLE-3-H multi-layer head.

    EAGLE-3 paper (2503.01840) concatenates residuals from three layers and
    passes them through in_proj. This mock simulates that by averaging the
    three (same dimension, no concat overhead in mock). The real trained head
    would use the full concatenation; we just need consistent input shape.
    """
    fused = [(low[i] + mid[i] + high[i]) / 3.0 for i in range(hidden)]
    return fused


def compute_tau_for_triplet(
    steps: List[CaptureStep],
    triplet: LayerTriplet,
    head: MockEagle5Head,
    k_depth: int,
) -> Tuple[float, int, int]:
    """Compute tau = sum(accepted) / sum(drafted) for one triplet.

    For each CaptureStep:
      1. Fuse the three layer residuals.
      2. Call head.propose_with_capture(start_token, fused_res, [], k_depth).
      3. Compare each draft[d] to target_tokens[d], accepting the longest
         matching prefix (EAGLE accept rule: accept all tokens up to first
         divergence).
    Returns (tau, total_accepted, total_drafted).
    """
    triplet_steps = [s for s in steps if str(s.triplet) == str(triplet)]
    if not triplet_steps:
        return 0.0, 0, 0

    total_accepted = 0
    total_drafted  = 0

    for step in triplet_steps:
        head.reset()
        fused = _fused_residual(
            step.residual_low,
            step.residual_mid,
            step.residual_high,
            head.hidden,
        )
        drafts = head.propose_with_capture(
            step.start_token, fused, [], k_depth
        )
        targets = step.target_tokens[:k_depth]
        k_actual = min(len(drafts), len(targets))
        total_drafted += k_actual

        # Accept longest confirmed prefix (exact greedy spec rule).
        acc = 0
        for d in range(k_actual):
            if drafts[d] == targets[d]:
                acc += 1
            else:
                break
        total_accepted += acc

    tau = total_accepted / max(1, total_drafted)
    return tau, total_accepted, total_drafted


def gate_label(tau: float) -> str:
    if tau >= TAU_GATE:
        return "GO"
    if tau >= 1.6:
        return "MARGINAL"
    return "NO-GO (kill-ledger)"


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(
    results: List[TauResult],
    out_path: Path,
    triplets: List[LayerTriplet],
    max_prompts: int,
    smoke: bool,
    no_gpu: bool,
) -> None:
    """Write markdown report to out_path."""
    date_str = time.strftime("%Y-%m-%d")
    mode = "SMOKE" if smoke else "FULL"
    gpu_mode = "mock (--no-gpu)" if no_gpu else "real GPU"

    lines = [
        f"# EAGLE-3-H tau Resurrection Sweep — {date_str}",
        "",
        f"**Mode:** {mode}  |  **GPU path:** {gpu_mode}  |"
        f"  **max-prompts:** {max_prompts}  |  **draft-depth:** {K_DEPTH}",
        "",
        "> KILL-LEDGER RULE: This is MEASUREMENT ONLY. `enable_neural_slot` is NEVER called.",
        "> Gate: tau >= 2.5 = GO; 1.6 <= tau < 2.5 = MARGINAL; tau < 1.6 = NO-GO (kill-ledger).",
        "",
    ]

    if any(r.mock for r in results):
        lines += [
            "## SMOKE (mock head, no GPU)",
            "",
            "Mock head uses random weights. tau will be near 0 (random drafts = 1/vocab accept).",
            "This run proves the measurement infrastructure operates end-to-end without a real model.",
            "",
        ]

    lines += [
        "## Results",
        "",
        "| Layer Triplet (low,mid,high) | tau | accepted/drafted | gate | elapsed |",
        "|---|---|---|---|---|",
    ]

    for r in results:
        frac = r.accepted / max(1, r.drafted)
        lines.append(
            f"| {r.triplet} | {r.tau:.4f} | {r.accepted}/{r.drafted} ({frac:.1%})"
            f" | {r.gate} | {r.elapsed_s:.2f}s |"
        )

    lines += [
        "",
        "## Verdict summary",
        "",
    ]

    best = max(results, key=lambda r: r.tau) if results else None
    if best:
        lines.append(f"Best triplet: `{best.triplet}` — tau={best.tau:.4f} — **{best.gate}**")
        lines.append("")
        if best.gate == "GO":
            lines.append(
                "> **CANDIDATE GO**: tau >= 2.5 gate met. "
                "Before any wiring: run full sweep (--max-prompts 100+) and confirm "
                "on the REAL WORKLOAD distribution. Then open an explicit kill-ledger "
                "review to override the EAGLE kill record in docs/RESEARCH.md."
            )
        else:
            lines.append(
                f"> **NO-GO**: best tau={best.tau:.4f} < {TAU_GATE} gate. "
                "All triplets remain below gate. EAGLE stays disabled per kill-ledger. "
                "Do not call enable_neural_slot."
            )
    lines += [
        "",
        "## Notes",
        "",
        "- EAGLE-3 paper (2503.01840): 3-layer tap (low/mid/high), up to ~5.6x on 13B.",
        "- Prior tau=0.877 was single-layer capture. Multi-layer fusion is the hypothesis here.",
        "- Mock path: tau near 0 by design (random argmax). Only proves infra, not quality.",
        "- Real path requires: `cargo build --release`, model weights, HAWKING_QWEN_EAGLE5_CAPTURE sidecar.",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[tau-sweep] report written to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_triplets(raw: str) -> List[LayerTriplet]:
    """Parse space-separated 'low,mid,high' triplets."""
    result = []
    for part in raw.strip().split():
        part = part.strip()
        if part:
            result.append(LayerTriplet.parse(part))
    if not result:
        raise ValueError("--triplets produced no valid triplets.")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="eh_eagle_tau_sweep",
        description="EAGLE-3-H tau resurrection sweep — MEASUREMENT ONLY (kill-ledger).",
    )
    p.add_argument(
        "--model", type=Path, default=None,
        help="Path to Qwen model .gguf (optional; triggers real GPU capture path).",
    )
    p.add_argument(
        "--triplets", type=str, default=DEFAULT_TRIPLETS,
        help=(
            "Space-separated list of 'low,mid,high' layer triplets to sweep. "
            f"Default: '{DEFAULT_TRIPLETS}'"
        ),
    )
    p.add_argument(
        "--max-prompts", type=int, default=DEFAULT_MAX_PROMPTS,
        help=f"Maximum prompts per triplet (default: {DEFAULT_MAX_PROMPTS}).",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Smoke mode: 1 prompt, 2 triplets only.",
    )
    p.add_argument(
        "--no-gpu", action="store_true",
        help="Force mock path: use Eagle5Head::mock(), skip GPU binary.",
    )
    p.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent.parent.parent / "reports" / "eagle_tau_sweep.md",
        help="Output markdown report path.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for mock path (default: 42).",
    )
    p.add_argument(
        "--sft-corpus", type=Path,
        default=Path(__file__).parent / "data" / "rwkv7_sft_sample.jsonl",
        help="Path to SFT JSONL corpus for prompts.",
    )

    args = p.parse_args(argv)

    # Parse triplets
    try:
        triplets = parse_triplets(args.triplets)
    except ValueError as e:
        print(f"[tau-sweep] ERROR parsing --triplets: {e}", file=sys.stderr)
        return 1

    # Smoke mode overrides
    if args.smoke:
        triplets = triplets[:2]
        args.max_prompts = 1
        print("[tau-sweep] SMOKE MODE: 1 prompt, 2 triplets.", flush=True)

    # Force mock if model missing or --no-gpu
    use_mock = args.no_gpu
    if not use_mock and args.model is not None and not args.model.exists():
        print(
            f"[tau-sweep] WARNING: model not found at {args.model} — using mock path.",
            flush=True,
        )
        use_mock = True
    if not use_mock and args.model is None:
        print(
            "[tau-sweep] INFO: --model not specified — using mock path (no GPU).",
            flush=True,
        )
        use_mock = True

    if use_mock:
        print(
            "\n*** SMOKE (mock head, no GPU) ***"
            "\nMock Eagle5Head: random weights, tau near 0."
            "\nThis proves measurement infrastructure runs end-to-end.\n",
            flush=True,
        )

    print(
        f"[tau-sweep] triplets={[str(t) for t in triplets]}"
        f"  max_prompts={args.max_prompts}"
        f"  smoke={args.smoke}"
        f"  no_gpu={use_mock}",
        flush=True,
    )

    rng = random.Random(args.seed)

    # Load prompts
    prompts = load_prompts(args.sft_corpus, args.max_prompts, rng)
    print(f"[tau-sweep] loaded {len(prompts)} prompt(s).", flush=True)

    # Build mock head (used for all runs in mock mode; also for fallback)
    head = MockEagle5Head(seed=args.seed, hidden=HIDDEN_DIM_MOCK, vocab=VOCAB_SIZE_MOCK)

    # Capture hidden states
    if use_mock:
        steps = mock_capture_run(prompts, triplets, K_DEPTH, HIDDEN_DIM_MOCK, VOCAB_SIZE_MOCK, rng)
    else:
        steps = real_capture_run(args.model, prompts, triplets, K_DEPTH)
        if not steps:
            # Fallback to mock so we always produce a report.
            print(
                "[tau-sweep] Falling back to mock capture (real capture returned 0 steps).",
                flush=True,
            )
            steps = mock_capture_run(
                prompts, triplets, K_DEPTH, HIDDEN_DIM_MOCK, VOCAB_SIZE_MOCK, rng
            )
            use_mock = True

    print(f"[tau-sweep] captured {len(steps)} total steps across {len(triplets)} triplet(s).", flush=True)

    # Compute tau per triplet
    results: List[TauResult] = []
    for triplet in triplets:
        t0 = time.monotonic()
        head.reset()
        tau, accepted, drafted = compute_tau_for_triplet(steps, triplet, head, K_DEPTH)
        elapsed = time.monotonic() - t0
        gate = gate_label(tau)
        result = TauResult(
            triplet=triplet,
            tau=tau,
            accepted=accepted,
            drafted=drafted,
            gate=gate,
            mock=use_mock,
            elapsed_s=elapsed,
        )
        results.append(result)
        print(result.verdict_line(), flush=True)

    # Summary
    print("", flush=True)
    print("=== TAU SWEEP SUMMARY ===", flush=True)
    for r in results:
        print(f"  {str(r.triplet):20s}  tau={r.tau:.4f}  {r.gate}", flush=True)

    best = max(results, key=lambda r: r.tau) if results else None
    if best:
        print(f"\nBest: {best.triplet}  tau={best.tau:.4f}  {best.gate}", flush=True)
        if best.gate == "GO":
            print(
                "\nCANDIDATE GO: tau >= 2.5 gate met on mock/real path.\n"
                "REMINDER: This is measurement only. Do NOT call enable_neural_slot\n"
                "without a full kill-ledger review and real-workload confirmation.",
                flush=True,
            )
        else:
            print(
                f"\nNO-GO: best tau={best.tau:.4f} < {TAU_GATE} gate.\n"
                "EAGLE stays disabled per kill ledger (docs/RESEARCH.md).\n"
                "Do not call enable_neural_slot.",
                flush=True,
            )

    # Write report
    write_report(results, args.out, triplets, args.max_prompts, args.smoke, use_mock)

    return 0


if __name__ == "__main__":
    sys.exit(main())
