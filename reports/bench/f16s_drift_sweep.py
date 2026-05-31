#!/usr/bin/env python3
"""A6.5 f16s broad drift sweep — OFF vs ON over a diverse prompt corpus.

Drives `dismantle bench-server` twice (one model load each): once with
DISMANTLE_QWEN_PREDEC_F16SCALES unset (OFF), once =1 (ON). Same locked env
otherwise, greedy temp=0, 32 new tokens. Compares the decoded text per prompt.

The engine is a deterministic greedy decoder with a fixed tokenizer, so
token-ID-sequence equality <=> decoded-text equality. For a diverged prompt
we also report the first-divergence position (as a fraction of the OFF text
length) — under autoregression, once one token-id differs every later token is
off-distribution, so the diverged-token count for that prompt is
(completion_tokens - tokens_before_first_divergence).

Output: JSON summary + human table to stdout. No source edits, no commits.
"""
import json
import os
import subprocess
import sys
import unicodedata

BIN = "./target/release/dismantle"
WEIGHTS = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
PROFILE = "profiles/qwen3b-instruct-q4k.m3pro18.json"
TOKENS = 32

LOCKED_ENV = {
    "DISMANTLE_QWEN_TCB": "1",
    "DISMANTLE_QWEN_VOCAB_PRUNE": "32000",
    "DISMANTLE_QWEN_Q4K_LMHEAD": "1",
    "DISMANTLE_QWEN_FFN_DOWN_Q4K": "1",
    "DISMANTLE_QWEN_Q4K_PREDEC": "1",
}

# (prompt-id, category) in the same order as the prompts file.
CATEGORIES = [
    "code", "code", "factual", "code-sql", "prose-edu",
    "prose", "math", "math", "math", "math",
    "dialogue", "dialogue", "lists", "lists", "nonenglish",
    "nonenglish", "nonenglish", "factual", "factual", "factual",
    "prose", "math", "code", "lists",
]


def load_prompts(path):
    prompts = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                prompts.append(s)
    return prompts


def run_server(prompts, f16s_on):
    """Spawn bench-server with the locked env + f16s toggle; feed all prompts;
    return {id: completion_text} and {id: completion_tokens}."""
    env = dict(os.environ)
    env.update(LOCKED_ENV)
    if f16s_on:
        env["DISMANTLE_QWEN_PREDEC_F16SCALES"] = "1"
    else:
        env.pop("DISMANTLE_QWEN_PREDEC_F16SCALES", None)

    # Cooperative scheduling per CLAUDE.md memory-coexist rule.
    cmd = [
        "nice", "-n", "19", "taskpolicy", "-b",
        BIN, "bench-server",
        "--weights", WEIGHTS,
        "--kernel-profile", PROFILE,
        "--stdin",
    ]
    reqs = "".join(
        json.dumps({"id": f"p{i:03d}", "prompt": p, "max_tokens": TOKENS}) + "\n"
        for i, p in enumerate(prompts)
    )
    proc = subprocess.run(
        cmd, input=reqs, env=env,
        capture_output=True, text=True, timeout=1800,
    )
    texts, ntok = {}, {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" in r and "completion_text" in r:
            texts[r["id"]] = r["completion_text"]
            ntok[r["id"]] = r.get("completion_tokens", 0)
    if not texts:
        sys.stderr.write("=== server stderr (no responses parsed) ===\n")
        sys.stderr.write(proc.stderr[-2000:] + "\n")
    return texts, ntok


def char_first_divergence(a, b):
    """Index of first differing char; len(min) if one is a prefix of the other."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1  # -1 == identical


def main():
    prompts = load_prompts("reports/bench/f16s_drift_prompts.txt")
    assert len(prompts) == len(CATEGORIES), (len(prompts), len(CATEGORIES))
    sys.stderr.write(f"[sweep] {len(prompts)} prompts, {TOKENS} tok, OFF then ON\n")

    sys.stderr.write("[sweep] === run OFF ===\n")
    off_t, off_n = run_server(prompts, f16s_on=False)
    sys.stderr.write(f"[sweep] OFF got {len(off_t)} responses\n")
    sys.stderr.write("[sweep] === run ON ===\n")
    on_t, on_n = run_server(prompts, f16s_on=True)
    sys.stderr.write(f"[sweep] ON got {len(on_t)} responses\n")

    rows = []
    cat_tot = {}      # category -> [drifted_tok, total_tok, n_prompts, n_drifted_prompts]
    tot_drift_tok = 0
    tot_tok = 0
    n_drifted_prompts = 0
    for i, p in enumerate(prompts):
        pid = f"p{i:03d}"
        cat = CATEGORIES[i]
        a = off_t.get(pid, "<MISSING>")
        b = on_t.get(pid, "<MISSING>")
        nt = off_n.get(pid, TOKENS) or TOKENS
        div = char_first_divergence(a, b)
        identical = (div == -1)
        # token-level drift estimate: fraction of the text (by char) past the
        # first divergence, applied to the token count. Conservative: if texts
        # differ at char k of an L-char OFF output, ~ (L-k)/L of the tokens are
        # downstream of the first divergence.
        if identical:
            drift_tok = 0
        else:
            L = max(len(a), 1)
            frac_after = max(0.0, (L - div) / L)
            drift_tok = round(frac_after * nt)
            n_drifted_prompts += 1
        c = cat_tot.setdefault(cat, [0, 0, 0, 0])
        c[0] += drift_tok
        c[1] += nt
        c[2] += 1
        c[3] += 1 if not identical else 0
        tot_drift_tok += drift_tok
        tot_tok += nt
        rows.append({
            "id": pid, "cat": cat, "identical": identical,
            "first_div_char": div, "off_len": len(a), "on_len": len(b),
            "n_tok": nt, "drift_tok_est": drift_tok,
            "off": a, "on": b,
        })

    # ---- print ----
    print("== A6.5 f16s drift sweep (OFF vs ON), 32-tok greedy, qwen2.5-3b-q4_k_m ==")
    print(f"prompts: {len(prompts)}   total tokens (OFF basis): {tot_tok}")
    print()
    print(f"{'id':>5} {'cat':>11} {'ident':>6} {'1stdiv':>7} "
          f"{'offlen':>7} {'ntok':>5} {'drift~':>6}")
    print("-" * 56)
    for r in rows:
        print(f"{r['id']:>5} {r['cat']:>11} {str(r['identical']):>6} "
              f"{r['first_div_char']:>7} {r['off_len']:>7} {r['n_tok']:>5} "
              f"{r['drift_tok_est']:>6}")
    print()
    print("== per-category ==")
    print(f"{'cat':>12} {'prompts':>8} {'drifted_p':>10} "
          f"{'drift_tok':>10} {'tot_tok':>8} {'drift%':>7}")
    print("-" * 60)
    for cat in sorted(cat_tot):
        dt, tt, npc, ndp = cat_tot[cat]
        pct = 100.0 * dt / tt if tt else 0.0
        print(f"{cat:>12} {npc:>8} {ndp:>10} {dt:>10} {tt:>8} {pct:>6.2f}%")
    print("-" * 60)
    pct = 100.0 * tot_drift_tok / tot_tok if tot_tok else 0.0
    print(f"{'TOTAL':>12} {len(prompts):>8} {n_drifted_prompts:>10} "
          f"{tot_drift_tok:>10} {tot_tok:>8} {pct:>6.2f}%")
    print()
    print(f"CORPUS DRIFT: {tot_drift_tok}/{tot_tok} tokens "
          f"({pct:.2f}%)  |  drifted prompts: {n_drifted_prompts}/{len(prompts)}")

    with open("reports/bench/f16s_drift_sweep_result.json", "w") as f:
        json.dump({
            "tokens": TOKENS, "n_prompts": len(prompts),
            "total_drift_tok": tot_drift_tok, "total_tok": tot_tok,
            "drift_pct": pct, "n_drifted_prompts": n_drifted_prompts,
            "per_category": {k: {"drift_tok": v[0], "total_tok": v[1],
                                 "n_prompts": v[2], "n_drifted_prompts": v[3]}
                             for k, v in cat_tot.items()},
            "rows": rows,
        }, f, indent=2)
    print("wrote reports/bench/f16s_drift_sweep_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
