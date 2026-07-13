#!/usr/bin/env python3.12
"""Deterministic downstream CAPABILITY eval for condensed models — the sanity
check that "near-1:1 perplexity" actually means "capability preserved".

Perplexity is an averaged surrogate: a condensed model can shave ppl back to ~1:1
on held-out prose while quietly losing arithmetic, recall, or code completion
(the long-tail, high-information tokens). This runs a handful of BUILT-IN, hardcoded
tasks (NO dataset download) and scores them deterministically (greedy / exact
forced-choice), so the doctor's ppl win can be cross-checked against real behavior.

Tasks (all hardcoded below):
  * qa     : short closed-book factual QA (first generated token, exact match)
  * cloze  : sentence completion, lowest-loss among 3-4 forced candidates
  * math   : single-token arithmetic answers (greedy decode of the answer span)
  * code   : tiny code-completion (next token after a deterministic prefix)

Scoring is greedy/argmax only -> deterministic and override-comparable. Mirrors
ppl_bench.py: same loading pattern, same DOCTOR_DEVICE/DOCTOR_DTYPE env contract
(0.5B -> mps/float32; 7B -> cpu/bfloat16; NEVER float16).

Usage:
  python3.12 tools/condense/multi_eval.py <hf-model-dir> [override.safetensors] [label] [adapter]

Prints one JSON line: {label, model, override, per_task{...}, aggregate, n}.
Run it on f16 and on a condensed override; the DELTA in aggregate accuracy is the
capability cost the ppl number alone can hide.
"""
import sys, os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from adapter_contract import AdapterContractError, validate_for_model

DEV = os.environ.get("DOCTOR_DEVICE")
DTYPE = getattr(torch, os.environ.get("DOCTOR_DTYPE", "float32"))

# ---------------- hardcoded tasks ----------------
# closed-book QA: (prompt, accepted answer prefixes). Scored by greedy-decoding a
# few tokens and checking the decoded text starts with any accepted answer.
QA = [
    ("Q: What is the capital of France?\nA:", ["paris"]),
    ("Q: What is the capital of Japan?\nA:", ["tokyo"]),
    ("Q: What is the chemical symbol for water?\nA:", ["h2o"]),
    ("Q: How many days are in a week?\nA:", ["seven", "7"]),
    ("Q: What color do you get mixing blue and yellow?\nA:", ["green"]),
    ("Q: What planet is known as the Red Planet?\nA:", ["mars"]),
]

# cloze / forced-choice: (context, [candidates], correct_index). Scored by the
# candidate with the lowest continuation loss (argmin) — deterministic.
CLOZE = [
    ("The opposite of hot is", [" cold", " warm", " fast", " loud"], 0),
    ("Water freezes at zero degrees", [" Celsius", " kilometers", " apples", " music"], 0),
    ("A dog is a kind of", [" animal", " mineral", " number", " color"], 0),
    ("The sun rises in the", [" east", " west", " ceiling", " ocean"], 0),
    ("Two plus two equals", [" four", " seven", " purple", " Tuesday"], 0),
]

# arithmetic: (prompt, answer string). Greedy-decode the answer span, exact match.
MATH = [
    ("2 + 2 =", "4"),
    ("10 - 3 =", "7"),
    ("6 * 7 =", "42"),
    ("100 / 4 =", "25"),
    ("9 + 8 =", "17"),
    ("12 * 12 =", "144"),
]

# code completion: (prefix, accepted next-token texts). Greedy first token.
CODE = [
    ("def add(a, b):\n    return a +", ["b"]),
    ("for i in range(10):\n    print(", ["i"]),
    ("x = [1, 2, 3]\nx.app", ["end"]),
    ("import nu", ["mpy"]),
    ("if x ==", ["="]),
]


def _attach_lora_adapter(model, adapter, *, expected_model, expected_wbase, dev, dtype):
    """Validate the complete artifact and model mapping before attaching any factor."""
    if not expected_wbase:
        raise AdapterContractError("a Hawking adapter requires its declared base override")
    import torch.nn as nn
    import torch.nn.functional as F
    from safetensors import safe_open

    modules = dict(model.named_modules())
    staged = []
    with safe_open(adapter, framework="pt") as handle:
        contract = validate_for_model(
            handle, model, expected_model=expected_model, expected_wbase=expected_wbase,
            expected_target_regex="all",
        )
        compatibility = []
        for entry in contract["entries"]:
            module = modules.get(entry["name"])
            if not isinstance(module, nn.Linear):
                compatibility.append(f"{entry['name']}: no matching nn.Linear")
                continue
            expected_a = (module.weight.shape[0], contract["rank"])
            expected_b = (contract["rank"], module.weight.shape[1])
            if entry["a_shape"] != expected_a or entry["b_shape"] != expected_b:
                compatibility.append(
                    f"{entry['name']}: A{entry['a_shape']} B{entry['b_shape']}, expected "
                    f"A{expected_a} B{expected_b}"
                )
        if compatibility:
            raise AdapterContractError(
                "adapter/model shape mismatch: " + "; ".join(compatibility)
            )
        # Stage all tensors only after every mapping has passed. The model remains untouched if a
        # tensor read/conversion fails, and no partially attached adapter can be evaluated.
        for entry in contract["entries"]:
            staged.append((
                modules[entry["name"]],
                handle.get_tensor(entry["a_key"]).to(dev, dtype),
                handle.get_tensor(entry["b_key"]).to(dev, dtype),
            ))

    for module, factor_a, factor_b in staged:
        module._hawking_lora_A = factor_a
        module._hawking_lora_B = factor_b
        module.forward = (lambda x, mm=module: F.linear(x, mm.weight, mm.bias)
                          + F.linear(F.linear(x, mm._hawking_lora_B),
                                     mm._hawking_lora_A))
    return len(staged), contract


def main():
    model_dir = sys.argv[1]
    override = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    label = sys.argv[3] if len(sys.argv) > 3 else (override or "f16")
    adapter = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
    dev = DEV or ("mps" if torch.backends.mps.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=DTYPE, attn_implementation="eager")

    if override:
        from safetensors.torch import load_file
        sd = load_file(override)
        sd = {k: v.to(DTYPE) for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"# {label}: swapped {len(sd)} tensors | missing {len(missing)} "
              f"unexpected {len(unexpected)}", file=sys.stderr)
        del sd

    model = model.to(dev).eval()

    if adapter:
        attached, contract = _attach_lora_adapter(
            model, adapter, expected_model=model_dir, expected_wbase=override,
            dev=dev, dtype=DTYPE,
        )
        print(f"# {label}: attached {attached} LoRA adapters "
              f"schema={contract['metadata']['adapter_schema']} "
              f"v{contract['metadata']['adapter_version']} "
              f"orientation={contract['orientation']}", file=sys.stderr)

    def greedy_text(prompt, max_new=6):
        enc = tok(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(enc.input_ids, attention_mask=enc.attention_mask,
                                 max_new_tokens=max_new, do_sample=False,
                                 num_beams=1, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

    def cand_loss(context, cand):
        """Mean NLL of `cand` tokens conditioned on `context` (forced-choice)."""
        ctx = tok(context, return_tensors="pt").input_ids.to(dev)
        full = tok(context + cand, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            logits = model(full).logits
        # score only the continuation tokens
        cont = full[0, ctx.shape[1]:]
        if cont.numel() == 0:
            return float("inf")
        lp = torch.log_softmax(logits[0, ctx.shape[1] - 1:-1], dim=-1)
        return float(-lp[range(cont.numel()), cont].mean())

    results = {}

    # QA: greedy decode, normalized startswith
    hit = 0
    for prompt, answers in QA:
        gen = greedy_text(prompt).strip().lower()
        gen = gen.lstrip(".:- ").strip()
        if any(gen.startswith(a) for a in answers):
            hit += 1
    results["qa"] = hit / len(QA)

    # cloze: argmin loss over candidates
    hit = 0
    for ctx, cands, correct in CLOZE:
        losses = [cand_loss(ctx, c) for c in cands]
        if int(min(range(len(losses)), key=lambda i: losses[i])) == correct:
            hit += 1
    results["cloze"] = hit / len(CLOZE)

    # math: greedy decode, exact token match on the leading number
    hit = 0
    for prompt, ans in MATH:
        gen = greedy_text(prompt).strip()
        # take the first whitespace-delimited chunk and strip trailing punctuation
        first = gen.split()[0].rstrip(".") if gen.split() else ""
        if first == ans:
            hit += 1
    results["math"] = hit / len(MATH)

    # code: greedy first token, normalized startswith
    hit = 0
    for prefix, accepted in CODE:
        gen = greedy_text(prefix, max_new=3).strip()
        if any(gen.startswith(a) for a in accepted):
            hit += 1
    results["code"] = hit / len(CODE)

    n = len(QA) + len(CLOZE) + len(MATH) + len(CODE)
    weights = {"qa": len(QA), "cloze": len(CLOZE), "math": len(MATH), "code": len(CODE)}
    aggregate = sum(results[k] * weights[k] for k in results) / n

    print(json.dumps({
        "schema": "hawking.multi_eval.v1", "suite": "hawking.multi_eval.v1",
        "label": label, "model": model_dir, "override": override, "adapter": adapter,
        "per_task": {k: round(v, 4) for k, v in results.items()},
        "task_n": weights, "aggregate": round(aggregate, 4), "n": n,
    }))


if __name__ == "__main__":
    main()
