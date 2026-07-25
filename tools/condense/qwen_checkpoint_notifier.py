#!/usr/bin/env python3.12
"""Telegram notifier for the Qwen Studentization campaign: one message per newly sealed checkpoint.

CONTRACT (from the full-course directive):
  * exactly one message per newly sealed checkpoint, keyed by row_id + checkpoint seal (the sha256
    the campaign already writes into every row), so a re-scan, a controller restart or a replayed
    transition cannot double-send;
  * the dedup ledger advances ONLY after a delivery actually succeeds - a failed send leaves the
    row unsent so the next poll retries it, rather than silently marking it done;
  * a send is observable but NEVER evidence, and a delivery failure never raises into the caller.
    The campaign's state machine must never be abortable by the chat.

It reuses the hardened send primitives and Keychain service names in
doctor_v5_telegram_rung_notifier (loaded by file path, the same way succ_telegram does). It never
re-implements the wire format and never takes a token or chat id as an argument, so no secret ever
reaches an argv, a log line, or this file.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(_HERE).resolve().parents[1]
CAMPAIGN = ROOT / "reports/condense/general_frontier/QWEN_GRAVITY"
CHECKPOINTS = CAMPAIGN / "checkpoints"
LEDGER = CAMPAIGN / "telegram_delivered.json"

SCHEMA = "hawking.qwen.checkpoint_notifier.v1"


def _notifier():
    """Load the campaign notifier module by path for its hardened _send / Keychain primitives."""
    p = Path(_HERE) / "doctor_v5_telegram_rung_notifier.py"
    spec = importlib.util.spec_from_file_location("_dv5_notifier", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def default_sender() -> Callable[[str], bool]:
    """Real sender. Returns True only on a confirmed delivery; never raises."""
    try:
        n = _notifier()
    except Exception:
        return lambda _text: False

    def send(text: str) -> bool:
        # n._send reads the Keychain itself, builds the sendMessage payload, and only returns
        # once Telegram has echoed a real message_id. Anything else raises and counts as a
        # failure, which is what keeps the dedup ledger from advancing on a non-delivery.
        try:
            return isinstance(n._send(text).get("message_id"), int)
        except Exception:
            return False
    return send


def load_ledger() -> dict[str, Any]:
    if LEDGER.is_file():
        try:
            return json.loads(LEDGER.read_text())
        except Exception:
            pass
    return {"schema": SCHEMA, "delivered": {}}


def save_ledger(led: dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(led, indent=1, sort_keys=True))
    os.replace(tmp, LEDGER)


def event_key(row: dict[str, Any]) -> str:
    """row_id + the seal the campaign already computed. Both must be present or the row is skipped."""
    return f"{row['row_id']}@{row.get('sha256', '')}"


def compose(row: dict[str, Any], progress: tuple[int, int], best: dict[str, Any] | None,
            next_row: str | None, pid: int | None) -> str:
    """Terse, honest, and it never dresses a collapse up as progress."""
    dv = row.get("divergence_vs_parent") or {}
    bpw = (row.get("bpw") or {}).get("whole_model_bpw")
    diag = row.get("diagnostic_only")
    lines = [
        f"HAWKING qwen3-235b {'[CAUSAL CONTROL]' if diag else ''}".strip(),
        f"row {row['row_id']}  ({progress[0]}/{progress[1]})",
        f"variant {row.get('variant')}  domain {row.get('domain')}",
    ]
    if bpw is not None:
        legal = (row.get("bpw") or {}).get("legal_under_one_bit_ceiling")
        tag = "" if legal is None else ("  LEGAL" if legal else "  NOT A LEGAL ARTIFACT")
        lines.append(f"complete BPW {bpw:.9f}{tag}")
    if dv:
        lines.append(f"symKL {dv.get('mean_sym_kl'):.4f}  argmax {dv.get('next_token_argmax_agreement'):.4f}"
                     f"  cos {dv.get('mean_logit_cosine'):.3f}  top5 {dv.get('mean_top5_overlap'):.3f}")
    q = row.get("quality") or {}
    if q.get("perplexity") is not None:
        lines.append(f"ppl {q['perplexity']:.1f}  n_tok {row.get('n_tokens')}")
    lines.append(f"verdict {row.get('verdict')}  capability_pass {row.get('capability_pass')}")
    if best:
        lines.append(f"best so far {best['variant']} symKL {best['sym_kl']:.4f} argmax {best['argmax']:.4f}")
    if next_row:
        lines.append(f"next {next_row}")
    if pid:
        lines.append(f"pid {pid}")
    return "\n".join(lines)


def _rows() -> list[dict[str, Any]]:
    out = []
    for f in sorted(glob.glob(str(CHECKPOINTS / "*.json"))):
        if os.path.basename(f).startswith("PROBE__"):
            continue                      # a truncated stack is not science and is never announced
        try:
            out.append(json.load(open(f)))
        except Exception:
            continue
    return out


def _best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    cands = [r for r in rows if r.get("kind") == "packed" and not r.get("diagnostic_only")
             and (r.get("divergence_vs_parent") or {}).get("mean_sym_kl") is not None]
    if not cands:
        return None
    b = min(cands, key=lambda r: r["divergence_vs_parent"]["mean_sym_kl"])
    return {"variant": b["variant"], "sym_kl": b["divergence_vs_parent"]["mean_sym_kl"],
            "argmax": b["divergence_vs_parent"]["next_token_argmax_agreement"]}


def scan_once(send: Callable[[str], bool] | None = None, *, pid: int | None = None,
              dry_run: bool = False) -> dict[str, Any]:
    """Send one message per undelivered sealed row. Ledger advances only on confirmed delivery."""
    send = send or default_sender()
    led = load_ledger()
    delivered: dict[str, Any] = led.setdefault("delivered", {})
    rows = _rows()
    best = _best(rows)
    pending = [r for r in rows if r.get("sha256") and event_key(r) not in delivered]
    sent, failed, composed = 0, 0, []
    for i, row in enumerate(pending):
        nxt = pending[i + 1]["row_id"] if i + 1 < len(pending) else None
        text = compose(row, (rows.index(row) + 1, len(rows)), best, nxt, pid)
        composed.append(text)
        if dry_run:
            continue
        if send(text):
            delivered[event_key(row)] = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                         "row_id": row["row_id"]}
            sent += 1
            save_ledger(led)          # persist after EACH success, so a crash cannot lose or repeat
        else:
            failed += 1
            break                     # stop on first failure; the next poll retries this row
    return {"schema": SCHEMA, "rows_seen": len(rows), "pending": len(pending),
            "sent": sent, "failed": failed, "dry_run": dry_run,
            "composed": composed if dry_run else []}


def demo() -> None:
    """Self-check: dedup, ledger-only-on-success, retry after failure, PROBE exclusion."""
    import tempfile
    global CHECKPOINTS, LEDGER
    with tempfile.TemporaryDirectory() as d:
        CHECKPOINTS = Path(d) / "checkpoints"
        CHECKPOINTS.mkdir()
        LEDGER = Path(d) / "led.json"
        row = {"row_id": "p1__V", "sha256": "abc", "variant": "V", "domain": "math",
               "n_tokens": 5, "kind": "packed", "verdict": "collapse", "capability_pass": False,
               "bpw": {"whole_model_bpw": 0.9, "legal_under_one_bit_ceiling": True},
               "quality": {"perplexity": 12.0},
               "divergence_vs_parent": {"mean_sym_kl": 1.5, "next_token_argmax_agreement": 0.2,
                                        "mean_logit_cosine": 0.5, "mean_top5_overlap": 0.3}}
        (CHECKPOINTS / "p1__V.json").write_text(json.dumps(row))
        (CHECKPOINTS / "PROBE__p1__V.json").write_text(json.dumps(dict(row, row_id="probe")))

        # a failing sender must NOT advance the ledger
        r = scan_once(send=lambda _t: False)
        assert r["sent"] == 0 and r["failed"] == 1, r
        assert load_ledger()["delivered"] == {}, "ledger advanced on a failed delivery"

        # retry succeeds, exactly one message, PROBE excluded
        seen: list[str] = []
        r = scan_once(send=lambda t: (seen.append(t), True)[1])
        assert r["sent"] == 1, r
        assert len(seen) == 1 and "p1__V" in seen[0] and "probe" not in seen[0]

        # second scan is a no-op: dedup by row_id + seal
        r = scan_once(send=lambda t: (seen.append(t), True)[1])
        assert r["sent"] == 0 and r["pending"] == 0, r
        assert len(seen) == 1, "duplicate send"

        # a re-sealed row (new sha) IS a new event
        (CHECKPOINTS / "p1__V.json").write_text(json.dumps(dict(row, sha256="def")))
        r = scan_once(send=lambda t: (seen.append(t), True)[1])
        assert r["sent"] == 1 and len(seen) == 2, r

        # a diagnostic row is labelled as a control, never as a candidate
        drow = dict(row, row_id="p2__D", sha256="xyz", diagnostic_only=True,
                    verdict="CAUSAL_CONTROL_not_a_candidate",
                    bpw={"whole_model_bpw": 16.0, "legal_under_one_bit_ceiling": False})
        (CHECKPOINTS / "p2__D.json").write_text(json.dumps(drow))
        r = scan_once(send=lambda t: (seen.append(t), True)[1], dry_run=False)
        assert "CAUSAL CONTROL" in seen[-1] and "NOT A LEGAL ARTIFACT" in seen[-1], seen[-1]
    print(json.dumps({"ok": True, "messages": len(seen)}, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Telegram notifier for sealed Qwen checkpoints.")
    ap.add_argument("--watch", action="store_true", help="poll until interrupted")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    if args.demo:
        demo()
        return 0
    if not args.watch:
        print(json.dumps(scan_once(pid=args.pid, dry_run=args.dry_run), indent=2))
        return 0
    while True:
        r = scan_once(pid=args.pid)
        if r["sent"] or r["failed"]:
            print(json.dumps({k: v for k, v in r.items() if k != "composed"}), flush=True)
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
