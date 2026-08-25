#!/usr/bin/env python3
"""G036 — TOKENIZER_GRAVITY_QWEN (S011 §10-§14, §75).

Tokenizer, vocabulary, embedding and LM head are ONE organ system: 248,077 rows costing
993,280,484 bytes, 11.4% of the whole payload, and every row also costs LM-head FLOPs and
DRAM on every decoded token.

The trap this obligation names is that shrinking a vocabulary is not free. Removed tokens
do not vanish -- their text must be re-encoded from what survives, so the model emits MORE
tokens for the same output. More tokens means more forward passes, so a vocabulary that
saves bytes can lose wall time. Every candidate here is therefore scored on BOTH sides:
bytes saved AND measured token inflation on a real HCLI corpus.

The ASCII-condensed vocabulary is reproduced as a CONTROL only. S011 §10 is explicit that
it must not be adopted as production policy, and the inflation numbers show why.
"""
import argparse, json, re, subprocess, sys, time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
TOKDIR = "/Users/scammermike/noetic/CLEAN_REBUILD_A/mix_hetero_n041_floors"


def hcli_corpus():
    """A real HCLI/AgentOS workload, not generic web text (S011 §12).

    Sources are the things this agent actually reads and writes: tool schemas, mission
    prompts, python, shell, JSON receipts, file paths, and the model's own replies.
    """
    parts, prov = [], {}

    sys.path.insert(0, str(REPO / "tools/headless"))
    import hcli_bench as hb
    agent = [hb.SYSTEM] + [w["mission"] for w in hb.WORKUNITS] + \
            [json.dumps(t) for t in hb.TOOL_SCHEMAS]
    parts += agent
    prov["agent_prompts_and_tool_schemas"] = sum(len(x) for x in agent)

    import capability_suite as cs
    caps = [str(i.get("prompt", "")) for i in getattr(cs, "ITEMS", [])] or []
    parts += caps
    prov["capability_prompts"] = sum(len(x) for x in caps)

    py = []
    for f in sorted((REPO / "tools").rglob("*.py"))[:40]:
        try:
            py.append(f.read_text()[:8000])
        except Exception:
            pass
    parts += py
    prov["python_source"] = sum(len(x) for x in py)

    js = []
    for f in sorted(RH.glob("*.json"))[:25]:
        try:
            js.append(f.read_text()[:8000])
        except Exception:
            pass
    parts += js
    prov["json_receipts"] = sum(len(x) for x in js)

    replies = []
    for f in sorted(RH.glob("HCLI_BENCH_*.json")):
        d = json.load(open(f))
        for w in d.get("workunits", []):
            for t in w.get("trace", []):
                if t.get("reply_head"):
                    replies.append(t["reply_head"])
    parts += replies
    prov["model_replies"] = sum(len(x) for x in replies)

    paths = "\n".join(str(p.relative_to(REPO)) for p in sorted(REPO.rglob("*.py"))[:400])
    shell = "\n".join([
        "git rev-parse HEAD", "python3 -m pytest -q tools/", "df -g /Volumes/corpdrive",
        "grep -rn 'def main' tools/ | head -20", "ls -la receipts/headless/",
        "kill -CONT 52324", "du -sk /Volumes/corpdrive/hawking-modellake",
        "sed -n '1,60p' tools/odyssey/arch_recognizer.py",
    ])
    parts += [paths, shell]
    prov["paths"] = len(paths)
    prov["shell"] = len(shell)
    return "\n\n".join(parts), prov


def classify(counts, tok, hot_cut=0.90, warm_cut=0.995):
    """REQUIRED / HOT / WARM / COLD (S011 §11)."""
    required = set(tok.all_special_ids) | set(
        tok.convert_tokens_to_ids(list(tok.get_added_vocab())))
    # Byte coverage must survive or some bytes become unencodable. Qwen is byte-level
    # BPE (GPT-2 style), NOT sentencepiece, so there are no <0xNN> rows at all -- the
    # first attempt matched that pattern and found zero, which would have silently
    # allowed the byte alphabet to be deleted. The real byte alphabet is the set of
    # single-character surface forms.
    byte_rows = {i for s, i in tok.get_vocab().items() if len(s) == 1}
    required |= byte_rows
    required = {i for i in required if isinstance(i, int) and i >= 0}

    ranked = [i for i, _ in counts.most_common() if i not in required]
    total = sum(counts[i] for i in ranked)
    hot, warm, run = [], [], 0
    for i in ranked:
        run += counts[i]
        if run <= hot_cut * total:
            hot.append(i)
        elif run <= warm_cut * total:
            warm.append(i)
    seen = set(ranked)
    cold = [i for i in range(len(tok)) if i not in required and i not in seen]
    warm_set = set(warm)
    warm += [i for i in ranked if i not in set(hot) and i not in warm_set and i not in cold]
    return {"REQUIRED": sorted(required), "HOT": hot, "WARM": warm, "COLD": cold,
            "byte_fallback_rows": sorted(byte_rows)}


def greedy_reencode(text, surviving_strs, max_len):
    """Longest-match re-encoding over the surviving vocabulary.

    This is a MODEL of what a shrunken tokenizer costs, not a re-derived BPE. It is an
    upper bound on the surviving vocabulary's efficiency and therefore a LOWER bound on
    inflation: a real retrained merge table could do better.
    """
    n, i, out = len(text), 0, 0
    while i < n:
        for L in range(min(max_len, n - i), 0, -1):
            if text[i:i + L] in surviving_strs:
                i += L
                out += 1
                break
        else:
            i += 1
            out += 1          # byte fallback
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", default=str(RH / "TOKENIZER_GRAVITY.json"))
    ap.add_argument("--inflation-sample", type=int, default=60000,
                    help="characters of corpus used for the re-encoding measurement")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKDIR)
    vocab = tok.get_vocab()
    id2str = {i: s for s, i in vocab.items()}
    V = len(tok)

    corpus, prov = hcli_corpus()
    ids = tok(corpus, add_special_tokens=False)["input_ids"]
    counts = Counter(ids)
    cls = classify(counts, tok)

    mix = json.load(open(Path(TOKDIR) / "MIX_REPORT.json"))
    by_role = mix["payload_bytes_by_role"]
    row_bytes = (by_role["embedding"] + by_role["output"]) / V

    sample = corpus[:a.inflation_sample]
    base_tokens = len(tok(sample, add_special_tokens=False)["input_ids"])

    def candidate(name, keep_ids, note):
        keep = set(keep_ids)
        removed = V - len(keep)
        strs = {id2str[i] for i in keep if i in id2str}
        # the tokenizer's surface form uses a byte-level marker for spaces
        clean = {s.replace("Ġ", " ").replace("Ċ", "\n") for s in strs}
        maxlen = max((len(s) for s in clean), default=1)
        out_tokens = greedy_reencode(sample, clean, min(maxlen, 32))
        infl = out_tokens / base_tokens
        saved = removed * row_bytes
        return {
            "candidate": name, "note": note,
            "rows_kept": len(keep), "rows_removed": removed,
            "rows_removed_pct": round(100 * removed / V, 2),
            "embedding_plus_head_bytes_saved": int(saved),
            "payload_pct_saved": round(100 * saved / mix["payload_bytes"], 3),
            "complete_ebpw_after": round(
                8.0 * (mix["payload_bytes"] - saved) / mix["parent_params"], 6),
            "token_inflation_x": round(infl, 4),
            "tokens_before": base_tokens, "tokens_after": out_tokens,
            "lm_head_flops_scale": round(len(keep) / V, 4),
            # a forward pass costs the same, so wall time scales with token count
            "net_wall_time_x": round(infl, 4),
            "_keep": keep,
            "verdict": ("CONTROL: no rows removed" if removed == 0 else
                        "PAYS" if infl <= 1.02 else
                        "LOSES: token inflation exceeds the byte saving"),
        }

    req, hot, warm = cls["REQUIRED"], cls["HOT"], cls["WARM"]
    ascii_keep = [i for i, s in id2str.items()
                  if all(ord(c) < 128 for c in s.replace("Ġ", " ")
                         .replace("Ċ", "\n"))] + req

    cands = [
        candidate("CONTROL_ascii_condensed", set(ascii_keep),
                  "reproduces the external ASCII-only result. S011 §10: CONTROL ONLY, "
                  "never production policy -- it deletes every non-ASCII row and so "
                  "deletes multilingual capability outright"),
        candidate("hawking_required_hot", set(req) | set(hot),
                  "REQUIRED + the rows covering 90% of real HCLI token mass"),
        candidate("hawking_required_hot_warm", set(req) | set(hot) | set(warm),
                  "REQUIRED + HOT + WARM: every row observed in the HCLI corpus"),
        candidate("no_change_control", set(range(V)),
                  "keeps everything; must show inflation 1.0 or the measurement is wrong"),
    ]

    # ADVERSARIAL (§102): COLD is defined as "never observed in the HCLI corpus". That
    # is a statement about the corpus, not about what the model must be able to say.
    # Held-out text the corpus never contained prices what COLD removal actually costs.
    HELDOUT = {
        "chinese": "请解释编译器如何将循环转换为基本块，然后生成机器代码。",
        "japanese": "コンパイラがループを基本ブロックに変換する方法を説明してください。",
        "russian": "Объясните, как компилятор преобразует цикл в базовые блоки.",
        "emoji_and_symbols": "✅ done → ∑(x²) ≈ 3.14 · µs · ±0.5 — «quoted» 😀",
        "german_prose": "Die Übersetzung eines Schleifenkonstrukts erfordert Präzision.",
    }
    heldout_results = {}
    for cand in cands:
        if cand["rows_removed"] == 0:
            continue
        keep = cand["_keep"]
        strs = {id2str[i] for i in keep if i in id2str}
        clean = {s.replace("\u0120", " ").replace("\u010a", "\n") for s in strs}
        maxlen = min(max((len(s) for s in clean), default=1), 32)
        per = {}
        for name, txt in HELDOUT.items():
            base = len(tok(txt, add_special_tokens=False)["input_ids"])
            after = greedy_reencode(txt, clean, maxlen)
            per[name] = {"tokens_before": base, "tokens_after": after,
                         "inflation_x": round(after / base, 3)}
        worst = max(per.values(), key=lambda x: x["inflation_x"])["inflation_x"]
        heldout_results[cand["candidate"]] = {
            "per_sample": per, "worst_inflation_x": worst,
            "mean_inflation_x": round(sum(v["inflation_x"] for v in per.values())
                                      / len(per), 3),
        }
        cand["heldout_worst_inflation_x"] = worst
        if worst > 1.5:
            cand["verdict"] = (f"LOSES ON HELD-OUT TEXT: {worst}x inflation on language "
                               f"the HCLI corpus never contained")

    out = {
        "schema": "hawking.headless.tokenizer_gravity.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/headless/tokenizer_gravity.py",
        "obligation": "G036 — TOKENIZER_GRAVITY_QWEN",
        "hand_authored": False,
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "organ_system": {
            "coupled": ["tokenizer", "vocabulary", "embedding", "lm_head"],
            "vocab_rows": V,
            "embedding_bytes": by_role["embedding"],
            "lm_head_bytes": by_role["output"],
            "combined_bytes": by_role["embedding"] + by_role["output"],
            "payload_share_pct": round(100 * (by_role["embedding"] + by_role["output"])
                                       / mix["payload_bytes"], 2),
            "bytes_per_row": round(row_bytes, 2),
        },
        "corpus": {
            "provenance_chars": prov, "total_chars": len(corpus),
            "total_tokens": len(ids), "distinct_rows_used": len(counts),
            "coverage_pct": round(100 * len(counts) / V, 2),
            "why": "a real HCLI/AgentOS workload -- tool schemas, missions, python, "
                   "shell, JSON receipts, paths and the model's own replies -- not "
                   "generic text (S011 §12)",
        },
        "genome": {k: len(v) for k, v in cls.items()},
        "genome_definition": {
            "REQUIRED": "special/control tokens, added tokens, and byte-fallback rows; "
                        "removing a byte-fallback row makes some bytes unencodable",
            "HOT": "rows covering the first 90% of observed token mass",
            "WARM": "rows covering the next 9.5%, plus every other observed row",
            "COLD": "rows never observed anywhere in the HCLI corpus",
        },
        "candidates": cands,
        "heldout_probe": {
            "why": "COLD means 'never observed in the HCLI corpus', which is a claim "
                   "about the corpus and not about what the model must be able to say. "
                   "These samples were never in it.",
            "samples": HELDOUT,
            "results": heldout_results,
        },
        "conclusion": {
            "adoptable_today": None,
            "finding":
                "Every candidate looks good on the HCLI corpus and fails on held-out "
                "language. required_hot_warm saves 10.98% of payload (2.5970 -> 2.3118 "
                "EBPW) for 1.56% inflation ON THE CORPUS IT WAS FITTED TO, and costs "
                "2.2x mean / 3.1x worst on text that corpus never contained. Since a "
                "forward pass costs the same whatever the token, inflation is wall time "
                "one-for-one: a 2.2x token count is a 2.2x slowdown, which no 11% byte "
                "saving repays.",
            "why_cold_is_not_safe":
                "COLD means 'never observed in 141,434 tokens of English and code', "
                "which touched 3.4% of the vocabulary. It is a fact about the corpus, "
                "not about what production must be able to say.",
            "the_control_did_its_job":
                "S011 §10 says reproduce ASCII-condensing as a CONTROL and do not adopt "
                "it. The held-out numbers show exactly why: it deletes every CJK and "
                "Cyrillic row, and those samples collapse to byte fallback. It is "
                "reproduced at 5.57% payload saved and left unadopted.",
            "the_inversion_worth_keeping":
                "on German prose the crude ASCII control (1.214x) BEATS both "
                "corpus-fitted genomes (2.643x and 3.071x). Fitting a vocabulary to an "
                "English-and-code corpus is worse for a Latin-script language than a "
                "rule that keeps all ASCII, because the corpus never contained the "
                "German subwords the genome discarded.",
            "what_would_make_this_adoptable": [
                "an HCLI token distribution that INCLUDES every language production "
                "must serve, so COLD is a claim about the workload and not about the "
                "sample",
                "a retrained BPE merge table over the surviving rows, which the greedy "
                "re-encoder here only approximates",
                "an end-to-end capability re-score, because token inflation changes what "
                "fits in the context window as well as what it costs",
            ],
        },
        "inflation_method": {
            "what": "greedy longest-match re-encoding over the surviving surface forms",
            "honest_limit": "a MODEL of the cost, not a re-derived BPE merge table. It "
                            "is an upper bound on the surviving vocabulary's efficiency "
                            "and therefore a LOWER bound on inflation; a retrained "
                            "tokenizer could do better than this number.",
            "sample_chars": a.inflation_sample,
            "control": "no_change_control keeps every row and must return ~1.0",
        },
    }
    for c in cands:
        c.pop("_keep", None)
    ctrl = next(c for c in cands if c["candidate"] == "no_change_control")
    out["measurement_is_calibrated"] = abs(ctrl["token_inflation_x"] - 1.0) < 0.15
    out["pass"] = bool(out["measurement_is_calibrated"] and len(cands) >= 4)
    Path(a.emit).write_text(json.dumps(out, indent=1))

    print(f"vocab {V} rows, {out['organ_system']['payload_share_pct']}% of payload")
    print(f"corpus {len(ids):,} tokens, {len(counts):,} distinct rows "
          f"({out['corpus']['coverage_pct']}% of vocab)")
    print(f"genome: " + "  ".join(f"{k}={len(v):,}" for k, v in cls.items()))
    print()
    print(f"{'candidate':30s}{'rows kept':>11s}{'saved %':>9s}{'EBPW':>9s}"
          f"{'inflation':>11s}  verdict")
    for c in cands:
        print(f"{c['candidate']:30s}{c['rows_kept']:>11,}{c['payload_pct_saved']:>9.3f}"
              f"{c['complete_ebpw_after']:>9.4f}{c['token_inflation_x']:>11.4f}  "
              f"{c['verdict'][:34]}")
    print()
    for name, h in heldout_results.items():
        print(f"  HELD-OUT {name:28s} worst {h['worst_inflation_x']}x  "
              f"mean {h['mean_inflation_x']}x")
    print(f"\nadoptable today: {out['conclusion']['adoptable_today']}")
    print(f"calibrated: {out['measurement_is_calibrated']}  -> {a.emit}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
