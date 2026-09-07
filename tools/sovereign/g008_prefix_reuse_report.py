#!/usr/bin/env python3
"""Prefix reuse, reported as SIX separate numbers. Never one blended rate.

Receipt: receipts/runtime/PREFIX_REUSE_SIX_NUMBERS.json

The directive is explicit: "Do not inflate effective TPS by hiding fresh work."
An append-path call can read ~4,900 tok/s of prompt, but almost none of it is
fresh. Quoting that as a prefill rate would be a lie, so fresh and effective are
carried side by side, always.

Three calls in ONE process, because reuse only exists across calls:
  1 cold      nothing to reuse
  2 diverges from call 1 but matches the stored checkpoint  -> RESTORE path
  3 strictly extends call 2                                  -> APPEND path
"""
from __future__ import annotations
import json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "receipts/runtime/PREFIX_REUSE_SIX_NUMBERS.json"
sys.path.insert(0, str(REPO))

from hcli.hawking_native import HawkingNativeConnector, config_for_model_path
from hcli.engine import _exact_tokenizer

tok = _exact_tokenizer()
F = ("Botanical taxonomy distinguishes bryophytes from tracheophytes by vascular tissue. "
     "Lichens pair a mycobiont with a photobiont in obligate symbiosis. ")
unit = len(tok.encode(F).ids)
BASE = F * max(1, 4000 // unit)

cfg = config_for_model_path(str(REPO / "ascension_envelope.hawking.json"))
cfg.max_seq_len = 34000
conn = HawkingNativeConnector(cfg)


def call(text: str, label: str) -> dict:
    payload = {"model": "local", "messages": [{"role": "user", "content": text}],
               "temperature": 0.0, "max_tokens": 8,
               "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    body = conn.complete_payload(payload, timeout=2400)
    wall = time.perf_counter() - t0
    h = body.get("hawking") or {}
    nm = h.get("native_metrics") or {}
    pre = nm.get("prefill") or {}
    prompt_positions = int((body.get("usage") or {}).get("prompt_tokens") or 0)
    reused = int(h.get("prefix_reused_tokens") or 0)
    fresh = int(h.get("prefill_tokens_stepped") or pre.get("steps") or 0)
    prefill_wall_s = float(pre.get("wall_ns") or 0) / 1e9
    return {
        "label": label,
        "prefix_source": h.get("prefix_source"),
        # THE SIX, never collapsed into one
        "prompt_positions": prompt_positions,
        "reused_positions": reused,
        "fresh_positions": fresh,
        "fresh_compute_tok_s": (fresh / prefill_wall_s) if prefill_wall_s > 0 else None,
        "effective_tok_s": (prompt_positions / prefill_wall_s) if prefill_wall_s > 0 else None,
        "wall_s": wall,
        # supporting
        "prefill_wall_s": prefill_wall_s,
        "reuse_fraction": (reused / prompt_positions) if prompt_positions else 0.0,
    }


rows = [
    call(BASE + "\n\nReply with the single word OK and nothing else.", "1_cold"),
    call(BASE + "\n\nReply with the single word FINE and nothing else.", "2_restore_path"),
    call(BASE + "\n\nReply with the single word FINE and nothing else. Then stop.", "3_append_path"),
]
doc = {
    "schema": "hawking.prefix_reuse.six_numbers.v1",
    "produced_by": "tools/sovereign/g008_prefix_reuse_report.py",
    "command": "python3 tools/sovereign/g008_prefix_reuse_report.py",
    "produced_at": datetime.now(timezone.utc).isoformat(),
    "law": ("fresh_compute_tok_s and effective_tok_s are ALWAYS reported together. "
            "effective counts reused positions the machine did not recompute; quoting it "
            "alone as a prefill rate hides the fresh work."),
    "calls": rows,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
for r in rows:
    print("%-15s src=%-19s prompt=%5d reused=%5d fresh=%5d  fresh=%8s tok/s  effective=%9s tok/s  wall=%7.2fs" % (
        r["label"], str(r["prefix_source"]), r["prompt_positions"], r["reused_positions"],
        r["fresh_positions"],
        ("%.1f" % r["fresh_compute_tok_s"]) if r["fresh_compute_tok_s"] else "n/a",
        ("%.1f" % r["effective_tok_s"]) if r["effective_tok_s"] else "n/a", r["wall_s"]))
