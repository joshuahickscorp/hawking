#!/usr/bin/env python3.12
"""REAL inference quality diagnostic for condensation: perplexity of the f16 parent
vs a condensed (TQ) variant, measured by ACTUAL forward passes (not the output-space
proxy). The condensed weights are produced by the Rust baker (quantize-model, which
decodes TQ -> W_hat safetensors) and swapped into the HF model via load_state_dict.

Usage:
  python3.12 tools/condense/ppl_bench.py <hf-model-dir> [override.safetensors] [label]

Prints one JSON line: {model, override, ppl, loss, ntok}. The DELTA in ppl between
the f16 run and the condensed run is the real quality cost of condensation; the
doctor (QAT/KD) should shrink it. Deterministic (fixed passage, no dataset download).
"""
import sys, json, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# fixed multi-paragraph passage (public-domain, deterministic) — relative ppl signal
TEXT = """The science of operations, as derived from mathematics more especially,
is a science of itself, and has its own abstract truth and value. The bounds of
arithmetic were outstepped the moment the idea of applying the cards had occurred.
A new, a vast, and a powerful language is developed for the future use of analysis,
in which to wield its truths so that these may become of more speedy and accurate
practical application for the purposes of mankind than the means hitherto in our
possession have rendered possible. Thus not only the mental and the material, but
the theoretical and the practical in the mathematical world, are brought into more
intimate and effective connection with each other. We are not aware of its being
on record that anything partaking in the nature of what is so well designated the
analytical engine has been hitherto proposed, or even thought of, as a practical
possibility, any more than the idea of a thinking or of a reasoning machine. In
considering any new subject, there is frequently a tendency, first, to overrate
what we find to be already interesting or remarkable; and, secondly, by a sort of
natural reaction, to undervalue the true state of the case, when we do discover
that our notions have surpassed those that were really tenable. The engine can
arrange and combine its numerical quantities exactly as if they were letters or
any other general symbols; and in fact it might bring out its results in algebraical
notation, were provisions made accordingly. It might develop three sets of results
simultaneously, viz. symbolic results, numerical results, and algebraical results."""


def main():
    model_dir = sys.argv[1]
    override = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    label = sys.argv[3] if len(sys.argv) > 3 else (override or "f16")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_dir)
    # float32 + eager attention: MPS has an f16 GQA matmul bug; f32 is robust and the
    # f16-vs-condensed DELTA is what matters (both runs share dtype).
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, attn_implementation="eager")

    if override:
        from safetensors.torch import load_file
        sd = load_file(override)
        sd = {k: v.to(torch.float32) for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"# {label}: swapped {len(sd)} tensors | missing {len(missing)} unexpected {len(unexpected)}",
              file=sys.stderr)

    model = model.to(dev).eval()
    ids = tok(TEXT, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = model(ids, labels=ids)
    loss = float(out.loss.item())
    print(json.dumps({"label": label, "model": model_dir, "override": override,
                      "ppl": math.exp(loss), "loss": loss, "ntok": int(ids.numel())}))


if __name__ == "__main__":
    main()
