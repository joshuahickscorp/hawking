#!/usr/bin/env python3.12
"""Model-feel screening gate on the real GLM-5.2 flagship artifact.

Per `HAWKING_MODEL_FEEL_PARITY_CONTRACT.md`: greedy continuation of a real prompt, cheap,
decisive-fail-only. The Llama-3.2-1B instrument at 0.877 BPW failed this outright --
" settle settle settle settle" -- and the oracle agreed that was the artifact's real
behavior, not a runtime defect. This is the same test on the actual flagship rather than a
proxy for it: does 744B-total MoE structure survive 0.883 whole-model BPW where a 1B dense
model did not.

Each step re-runs the whole growing sequence from an empty cache -- GravityGlmSource has
no persistent-cache API yet, and the dominant cost per call is decoding ~40B active
parameters' worth of weight bytes, which does not shrink by adding incremental caching for
a screening run this short. Simplicity over speed for a one-time signal.

    python3.12 tools/condense/glm52_flagship_screening.py --prompt "The capital of France is" --tokens 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import jinja2
import numpy as np
from tokenizers import Tokenizer

import glm52_reference as ref  # noqa: E402
from glm52_gravity_source import GravityGlmSource  # noqa: E402

DEFAULT_MODEL_DIR = (
    Path.home() / "Library/Application Support/Hawking/Models/GLM-5.2"
    "/b4734de4facf877f85769a911abafc5283eab3d9/General-R0")


def render_prompt(chat_template_path: Path, user_text: str, *, enable_thinking: bool) -> str:
    """GLM-5.2 is chat/instruction-tuned, not a base completion model: its own
    template opens every input with `[gMASK]<sop>` and role markers, and a raw
    text completion is out of distribution for it regardless of compression
    rate. Rendering through the model's own pinned template is what makes a
    screening-gate failure attributable to the ARTIFACT rather than to having
    fed it an input shape it was never trained to see.
    """
    template = jinja2.Template(chat_template_path.read_text())
    return template.render(
        messages=[{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        tools=None,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--raw-prompt", action="store_true",
                    help="skip chat-template rendering; encode --prompt verbatim")
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    index_json = args.dir / "model.gravity.index.json"
    arch = json.loads(index_json.read_text())["architecture"]
    tok = Tokenizer.from_file(str(args.dir / "tokenizer/tokenizer.json"))

    print("opening...", file=sys.stderr)
    source = GravityGlmSource(args.dir, index_json=index_json, verify_hash=False)

    if args.raw_prompt:
        rendered = args.prompt
    else:
        rendered = render_prompt(args.dir / "tokenizer/chat_template.jinja",
                                 args.prompt, enable_thinking=args.enable_thinking)
    print(f"rendered prompt: {rendered!r}", file=sys.stderr)
    ids = tok.encode(rendered).ids
    print(f"prompt tokens: {ids}", file=sys.stderr)
    generated: list[int] = []
    step_seconds: list[float] = []
    t_start = time.time()

    for step in range(args.tokens):
        t0 = time.time()
        seq = np.array([ids + generated], dtype=np.int64)
        logits, _cache, _trace = ref.main_forward(seq, source, arch)
        elapsed = time.time() - t0
        step_seconds.append(elapsed)
        next_id = int(np.argmax(logits[0, -1]))
        generated.append(next_id)
        piece = tok.decode([next_id])
        print(f"step {step}: {elapsed:.1f}s -> token {next_id} = {piece!r}", file=sys.stderr)

    total = time.time() - t_start
    text = tok.decode(ids + generated)
    result = {
        "schema": "hawking.glm52.flagship_screening.v1",
        "prompt": args.prompt,
        "prompt_tokens": ids,
        "generated_tokens": generated,
        "continuation_text": tok.decode(generated),
        "full_text": text,
        "step_seconds": step_seconds,
        "total_seconds": total,
        "artifact": str(args.dir),
        "complete_bpw": arch and json.loads(index_json.read_text())
        .get("coverage", {}).get("verdict"),
    }
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
