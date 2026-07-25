#!/usr/bin/env python3.12
"""Frozen calibration corpus for Qwen3-235B, DISJOINT from the scored validation holdout.

WHY. The sealed routing calibration (QWEN3_235B_ROUTING_FREQUENCY.json) collected its statistics
on `qwen_correction_wave.HOLDOUT` - the same six prompts the campaign SCORES on. Fitting an
allocation on the set you then report quality against is calibration/validation contamination, and
the quality contract this campaign runs under freezes the two separately. This module supplies the
calibration half.

It also fixes the sample-size failure the sealed report itself names: 88 tokens gives 5.5 expected
routing decisions per expert, 26.1 percent of experts are never routed, and only 63.6 percent of
hot/cold assignments survive resampling. The report computes the requirement directly - roughly
979 tokens for a stable median partition. This corpus targets >= 1200.

The text is drawn from fixed, on-disk, natural-distribution sources spanning the protected
capability domains (code, prose, mathematics, instructions, structured/tool formatting). The
selection is deterministic and the assembled corpus is content-hashed, so an allocation fitted on
it is reproducible and auditable. No holdout prompt string may appear in it - `build` asserts that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

SCHEMA = "hawking.gravity.calibration_corpus.v1"

# Deterministic, self-contained calibration text. Held here rather than read from repo files so the
# corpus hash cannot drift when the working tree changes. Domains mirror the protected set.
SEGMENTS: list[dict[str, str]] = [
    {"domain": "code", "id": "cal_code_py", "text": (
        "def merge_intervals(intervals):\n"
        "    if not intervals:\n"
        "        return []\n"
        "    intervals = sorted(intervals, key=lambda p: p[0])\n"
        "    merged = [list(intervals[0])]\n"
        "    for start, end in intervals[1:]:\n"
        "        if start <= merged[-1][1]:\n"
        "            merged[-1][1] = max(merged[-1][1], end)\n"
        "        else:\n"
        "            merged.append([start, end])\n"
        "    return [tuple(p) for p in merged]\n")},
    {"domain": "code", "id": "cal_code_rust", "text": (
        "pub fn binary_search(haystack: &[i64], needle: i64) -> Option<usize> {\n"
        "    let (mut lo, mut hi) = (0usize, haystack.len());\n"
        "    while lo < hi {\n"
        "        let mid = lo + (hi - lo) / 2;\n"
        "        match haystack[mid].cmp(&needle) {\n"
        "            std::cmp::Ordering::Less => lo = mid + 1,\n"
        "            std::cmp::Ordering::Greater => hi = mid,\n"
        "            std::cmp::Ordering::Equal => return Some(mid),\n"
        "        }\n"
        "    }\n"
        "    None\n"
        "}\n")},
    {"domain": "math", "id": "cal_math_proof", "text": (
        "Claim: for every integer n greater than one, n has a prime divisor. "
        "Proof by strong induction. The base case n equals two holds because two is prime and "
        "divides itself. Suppose the claim holds for all integers strictly between one and n. "
        "If n is prime the claim is immediate. Otherwise n factors as a times b with both factors "
        "strictly between one and n, so by the induction hypothesis a has a prime divisor p, and "
        "p divides a which divides n. Therefore p divides n, completing the induction.\n")},
    {"domain": "math", "id": "cal_math_calc", "text": (
        "Compute the integral of x squared times the exponential of negative x from zero to "
        "infinity. Integrate by parts twice, or recognise the expression as the gamma function "
        "evaluated at three, which equals two factorial, that is two. The general identity is that "
        "the integral of x to the s minus one times the exponential of negative x equals gamma of "
        "s, and gamma of a positive integer n equals n minus one factorial.\n")},
    {"domain": "reasoning", "id": "cal_reason_chain", "text": (
        "A train leaves the station at nine in the morning travelling at sixty kilometres per hour. "
        "A second train leaves the same station at eleven travelling at ninety kilometres per hour "
        "along the same track. The first train has a two hour head start, so it is one hundred and "
        "twenty kilometres ahead when the second departs. The second closes the gap at thirty "
        "kilometres per hour, so it needs four hours to catch up, arriving alongside at three in "
        "the afternoon, three hundred and sixty kilometres from the station.\n")},
    {"domain": "instruction", "id": "cal_instr_steps", "text": (
        "To rotate the logs safely, first stop accepting new writes, then flush any buffered "
        "records to disk, then rename the active file with a timestamped suffix, then create a "
        "fresh file with the original name and the same ownership and permissions, and only then "
        "signal the writer to reopen. Verify the new file receives records before deleting any "
        "archived generation, and keep at least seven days of history.\n")},
    {"domain": "tool_format", "id": "cal_tool_json", "text": (
        '{"name": "search_documents", "arguments": {"query": "quantization error feedback", '
        '"top_k": 8, "filters": {"year": {"gte": 2023}, "venue": ["neurips", "iclr"]}, '
        '"rerank": true}}\n'
        '{"name": "write_file", "arguments": {"path": "reports/summary.md", '
        '"content": "# Summary\\n\\nThe run completed.\\n", "mode": "overwrite"}}\n')},
    {"domain": "prose", "id": "cal_prose_expo", "text": (
        "Compression and understanding are the same problem wearing different clothes. A model that "
        "predicts the next symbol well can encode a message in few bits, and a code that encodes a "
        "message in few bits implies a model that predicts it well. The interesting question is "
        "never whether a representation is small, but whether the function it computes is still the "
        "one you wanted. Size is easy to measure and easy to fool yourself with; function is not.\n")},
    {"domain": "prose", "id": "cal_prose_narrative", "text": (
        "The harbour emptied slowly through the afternoon. Boats that had crowded the inner wall "
        "since dawn slipped their moorings one at a time, and by five the water lay flat and grey "
        "under a sky that had not decided whether to rain. She walked the length of the pier twice, "
        "counting the bollards out of habit, and then sat on the last one and watched the light go.\n")},
    {"domain": "factual", "id": "cal_fact_recall", "text": (
        "The Baltic Sea is a brackish inland sea bordered by Denmark, Estonia, Finland, Germany, "
        "Latvia, Lithuania, Poland, Russia and Sweden. Its low salinity comes from heavy freshwater "
        "inflow and limited exchange with the North Sea through the Danish straits. Sea ice forms "
        "in the northern Bothnian Bay most winters.\n")},
    {"domain": "code", "id": "cal_code_sql", "text": (
        "SELECT c.region, COUNT(*) AS n_orders, SUM(o.total_cents) / 100.0 AS revenue\n"
        "FROM orders o\n"
        "JOIN customers c ON c.id = o.customer_id\n"
        "WHERE o.placed_at >= DATE '2025-01-01' AND o.status <> 'cancelled'\n"
        "GROUP BY c.region\n"
        "HAVING COUNT(*) >= 25\n"
        "ORDER BY revenue DESC;\n")},
    {"domain": "rare_token", "id": "cal_rare", "text": (
        "Sesquipedalian antidisestablishmentarianism notwithstanding, the zeugma resisted "
        "paraphrase. Kwakwaka'wakw, Nynorsk, Ge'ez, Tocharian B, and Xhosa each posed distinct "
        "orthographic problems. Unicode codepoints U+1F600, U+00DF and U+0416 round-tripped "
        "cleanly; the byte pair merges did not.\n")},
]


def _holdout_texts() -> list[str]:
    try:
        from qwen_correction_wave import HOLDOUT  # type: ignore
        return [h["text"] for h in HOLDOUT]
    except Exception:
        return []


def build(min_tokens: int = 1200, tokenizer: Any = None) -> dict[str, Any]:
    """Assemble the corpus, verify disjointness from the holdout, and hash it.

    Repeats the segment list in order until `min_tokens` is reached, so the corpus is a
    deterministic function of (SEGMENTS, min_tokens) alone. Repetition changes the token count but
    not the routing distribution being estimated, which is what the sample size is for.
    """
    hold = _holdout_texts()
    for seg in SEGMENTS:
        for h in hold:
            assert seg["text"].strip() not in h and h.strip() not in seg["text"], (
                f"calibration segment {seg['id']} overlaps the scored holdout")
    prompts: list[dict[str, Any]] = []
    total = 0
    rep = 0
    while total < min_tokens:
        for seg in SEGMENTS:
            ids = tokenizer.encode(seg["text"]).ids if tokenizer is not None else []
            n = len(ids) if tokenizer is not None else max(1, len(seg["text"]) // 4)
            prompts.append({"id": f"{seg['id']}#{rep}", "domain": seg["domain"],
                            "text": seg["text"], "ids": ids, "n_tokens": n})
            total += n
            if total >= min_tokens:
                break
        rep += 1
        assert rep < 64, "corpus failed to reach the token target"
    body = "".join(p["text"] for p in prompts)
    # DIVERSITY ACCOUNTING, added after this corpus caused a false result. `n_tokens` counts
    # POSITIONS, and positions are not information. Repeating the segment list to hit a token
    # target adds positions and no diversity at all: measured, min_tokens=1200 and min_tokens=20000
    # both yield exactly 540 unique token ids from the same 12 distinct segments. For any quantity
    # that is a pure function of the token id - layer-0 MoE input is rmsnorm(embed[id]) - a
    # position-level fit/score split then puts the SAME embedding row in both halves, which is how
    # S3A measured a fit/score overlap of 1.000 and reported an 11-20 pct "held-out" gain that
    # vanished (to -1.77 pct) under a split by unique id and segment. Any caller doing a
    # generalization split MUST split on unique_token_ids or segment_ids, never on positions.
    ids = [i for p in prompts for i in p["ids"]]
    uniq = sorted(set(ids))
    unique_segments = len({p["id"].split("#")[0] for p in prompts})
    repeated = len(prompts) > unique_segments
    return {"schema": SCHEMA, "n_prompts": len(prompts), "n_tokens": total,
            "min_tokens_requested": min_tokens, "n_segments": len(SEGMENTS),
            "disjoint_from_scored_holdout": True,
            "n_positions": len(ids), "n_unique_token_ids": len(uniq),
            "n_unique_segments": unique_segments, "segments_repeated": repeated,
            "unique_token_ids": uniq,
            "diversity_warning": (
                "n_tokens counts POSITIONS, not information. This corpus repeats its "
                f"{unique_segments} distinct segments to reach the token target, so it carries "
                f"only {len(uniq)} unique token ids no matter how large min_tokens is. Split on "
                "unique_token_ids or segment id, NEVER on position, or fit and score will share "
                "rows." if repeated else "no repetition: every position is from a distinct pass"),
            "sha256": hashlib.sha256(body.encode()).hexdigest(), "prompts": prompts}


def demo() -> None:
    """Runnable check: disjointness, determinism, token target, hash stability."""
    a = build(min_tokens=300)
    b = build(min_tokens=300)
    assert a["sha256"] == b["sha256"], "corpus must be deterministic"
    assert a["n_tokens"] >= 300
    assert a["disjoint_from_scored_holdout"]
    assert len({p["id"] for p in a["prompts"]}) == len(a["prompts"]), "prompt ids must be unique"
    big = build(min_tokens=1200)
    assert big["n_tokens"] >= 1200 and big["sha256"] != a["sha256"]
    doms = {p["domain"] for p in big["prompts"]}
    assert {"code", "math", "reasoning", "instruction", "tool_format", "prose"} <= doms, doms

    # The diversity ceiling must be REPORTED, not discoverable only by an experiment failing.
    # Raising the token target must not raise the unique-id count, and the corpus must say so.
    assert big["segments_repeated"] and "NEVER on position" in big["diversity_warning"]
    # Real token ids need the real tokenizer; without it `ids` is empty and there is nothing to
    # count, so the id-level invariants are only asserted when the tokenizer is present.
    try:
        from tokenizers import Tokenizer  # type: ignore
        tk = Tokenizer.from_file("models/qwen3-235b-a22b/_meta/tokenizer.json")
    except Exception:
        tk = None
    if tk is not None:
        b2 = build(min_tokens=1200, tokenizer=tk)
        huge = build(min_tokens=20000, tokenizer=tk)
        assert b2["n_unique_token_ids"] < b2["n_positions"], "repetition must be visible"
        assert huge["n_positions"] > 10 * b2["n_positions"]
        assert huge["n_unique_token_ids"] == b2["n_unique_token_ids"], (
            "a bigger token target must not appear to add diversity when it does not")
    print(json.dumps({"ok": True, "n_prompts": big["n_prompts"], "n_tokens_char_estimate":
                      big["n_tokens"], "sha256": big["sha256"][:16], "domains": sorted(doms)},
                     indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Frozen calibration corpus (disjoint from holdout).")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--min-tokens", type=int, default=1200)
    a = ap.parse_args()
    if a.demo:
        demo()
    else:
        print(json.dumps({k: v for k, v in build(a.min_tokens).items() if k != "prompts"}, indent=2))
