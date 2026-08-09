#!/usr/bin/env python3
"""Prepare a broad, source-bound L0 route-capture input for activation pricing.

Diagnostic only. Not a capability, HCLI, TPS, or coherence claim.

Builds many one-user native chat prompts (diverse length/domain/structure),
tokenizes them with the source Qwen3-Coder tokenizer, and writes the sealed
input JSON consumed by `ascension_qwen30_current_hcli_layer0_route_capture`
under the broad-activation schema.

The three-probe HCLI quality path is left untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TOKENIZER = Path(
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/"
    "qwen-30b/Qwen3-Coder-30B-A3B-Instruct/tokenizer.json"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/"
    "quality-diagnostics/broad-activation-v1"
)

BROAD_INPUT_SCHEMA = "hawking.ascension.qwen30_broad_activation_layer0_route_capture_input.v1"
STATUS = "NEW_DIAGNOSTIC_NOT_HISTORICAL"

# ---------------------------------------------------------------------------
# Prompt corpus — intentionally diverse so constant-mean null can fall.
# Domains: code, prose, structured/JSON, multi-turn, long-context, math,
# instructions, dialogue, mixed-language-ish technical, lists.
# ---------------------------------------------------------------------------


def build_corpus() -> list[dict[str, str]]:
    """Return list of {probe_id, domain, text} (user message body only)."""
    long_prose = (
        "Write a careful essay about why measurement instruments dominate "
        "conclusions in scientific work. Discuss selection bias, sample size, "
        "and the difference between operator recovery and distribution-local "
        "fit. Use three short sections with headings, then a one-paragraph "
        "conclusion that refuses to overclaim."
    )
    long_code = '''\
Implement a pure-Python priority queue that supports:
1) push(item, priority)
2) pop() -> item with lowest priority
3) decrease_key(item, new_priority)
Use a binary heap of (priority, counter, item) triples so equal priorities are
FIFO-stable. Include a minimal unittest block under if __name__ == "__main__".
Do not use heapq's nlargest. Comment each public method in one sentence.
'''
    long_json_spec = (
        "Design a JSON schema for a lab receipt with fields: schema (string), "
        "status (enum of EARNED/REFUSED/INCOMPLETE), claim_boundary (object of "
        "booleans), binding (object with absolute paths and sha256 hex strings), "
        "and measurements (array of objects with name, value, unit). "
        "Return ONLY valid JSON for one example instance populated with "
        "plausible diagnostic values. No markdown fences."
    )
    multi_turn_body = (
        "Conversation so far:\n"
        "User: What is a MoE expert?\n"
        "Assistant: A specialist sub-network selected by a router for a token.\n"
        "User: Why would weight cosine mislead packing quality?\n"
        "Assistant: Because output error on real activations can disagree with "
        "weight-space distance.\n"
        "User: Given that, list three concrete checks you would run on a new "
        "activation capture before trusting a family ranking. Be terse."
    )
    long_context = (
        "Context dump (use only what you need):\n"
        + "\n".join(
            f"- note[{i}]: activation diversity requires domain mix; null falls when "
            f"expert outputs are not near-constant across prompts; sample-{i}."
            for i in range(40)
        )
        + "\n\nTask: In at most five bullets, state how you would detect a "
        "constant-mean null trap from held-out expert outputs alone."
    )
    rows: list[tuple[str, str, str]] = [
        ("code_fib_iter", "code", "Write an iterative Fibonacci function in Rust that returns u64 and panics on overflow."),
        ("code_py_bisect", "code", "Write a Python binary search that returns the insertion index for a sorted list of ints."),
        ("code_sql_window", "code", "Write a SQL query using ROW_NUMBER() to pick the latest event per user_id from events(user_id, ts, payload)."),
        ("code_go_http", "code", "Show a minimal Go net/http handler that returns {\"ok\":true} as JSON with status 200."),
        ("code_ts_reduce", "code", "TypeScript: write a typed reduce that sums numbers and rejects non-finite values at runtime."),
        ("code_long_pq", "code", long_code),
        ("prose_measure", "prose", long_prose),
        ("prose_short_haiku", "prose", "Compose one English haiku about fog over a harbor. No title."),
        ("prose_argument", "prose", "Argue in two paragraphs that negative experimental results are first-class scientific deliverables."),
        ("prose_narrative", "prose", "Narrate 120-150 words of a technician discovering a high constant-mean null in a capture log, without inventing false numbers."),
        ("json_status_strict", "structured", 'Return exactly this JSON and nothing else: {"status":"ok","phase":"diagnostic"}'),
        ("json_schema_example", "structured", long_json_spec),
        ("json_array_table", "structured", "Return a JSON array of 5 objects with keys expert_id (int), hits (int), null_cosine (float). Invent realistic diagnostic numbers only."),
        ("json_nested_error", "structured", 'Return JSON: {"error":{"code":"INSUFFICIENT_DIVERSITY","detail":"null>=0.9","retry":{"min_prompts":12}}} with no prose.'),
        ("multi_turn_moe", "multi_turn", multi_turn_body),
        ("multi_turn_debug", "multi_turn", "User earlier said the server on :18430 must not be restarted. Assistant acknowledged. Now answer: list safe steps to capture L0 activations without disturbing that server."),
        ("multi_turn_review", "multi_turn", "Prior message claimed surplus +0.04 with weight cosine 0.46. Reply as a skeptical reviewer: what is still unproven?"),
        ("long_ctx_notes", "long_context", long_context),
        ("long_ctx_log", "long_context", "Log excerpt:\n" + "\n".join(f"t={i:04d} expert={i%128} cos={0.9+((i*17)%10)*0.001:.3f}" for i in range(80)) + "\nSummarize whether these cosines alone prove operator recovery. Answer in 3 sentences."),
        ("math_bayes", "math", "A test has sensitivity 0.9 and specificity 0.95. Prevalence is 1%. What is PPV? Show the arithmetic."),
        ("math_rank", "math", "For a matrix W in R^{m x n} and rank r, how many free parameters does a factored LR form have? Give the formula and a short explanation."),
        ("math_entropy", "math", "Compute Shannon entropy in bits for the distribution [0.5, 0.25, 0.125, 0.125]. Show steps."),
        ("instr_checklist", "instruction", "Produce a numbered checklist (8 items) for source-bound activation capture provenance. Imperative verbs only."),
        ("instr_refuse", "instruction", "If asked to claim full-model coherence from a component probe, refuse in one sentence and name the missing evidence."),
        ("instr_compare", "instruction", "Compare in a table with 4 rows: weight cosine vs output cosine vs null-subtracted surplus vs BPW. Columns: metric, what it measures, common failure mode."),
        ("dialogue_lab", "dialogue", "A: Is 0.95 output cosine good?\nB: Only relative to the constant-mean null.\nA: Ours is 0.94 null.\nB: Then surplus is thin.\nContinue one more A/B exchange that ends with a concrete next measurement."),
        ("list_domains", "list", "List 12 prompt domain tags useful for lowering activation nulls, one per line, no numbers."),
        ("mixed_api", "mixed", "Given this partial OpenAPI path /v1/capture with POST body {prompts: string[]}, write a curl example and a one-line claim_boundary comment."),
        ("mixed_regex", "mixed", "Write a regex that matches sha256 hex digests and a Python snippet that validates a path is absolute."),
        ("code_rust_parse", "code", "Rust: parse CLI flags --input-json and --output-dir that must be absolute Paths; return a struct or error string."),
        ("prose_policy", "prose", "Explain in plain language why absolute output cosine is inadmissible when the constant-mean null is enormous."),
        ("structured_yamlish", "structured", "Emit a YAML-like plain text config (not JSON) with keys: capture_layer, max_seq_len, probe_count, claim: diagnostic_only."),
    ]
    return [
        {"probe_id": pid, "domain": domain, "text": text}
        for pid, domain, text in rows
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def token_ids_u32le_sha256(ids: list[int]) -> str:
    return sha256_bytes(b"".join(struct.pack("<I", int(x)) for x in ids))


def one_user_native_prompt(tokenizer: Any, user_text: str) -> list[int]:
    """Match Qwen chat shape used by the sealed three-probe capture tails.

    <|im_start|>user\\n{text}<|im_end|>\\n<|im_start|>assistant\\n
    """
    rendered = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    # tokenizers encode with special tokens already present as literal text
    ids = list(tokenizer.encode(rendered).ids)
    if not ids:
        raise RuntimeError("empty tokenization")
    if ids[0] != 151644:  # <|im_start|>
        raise RuntimeError(f"expected chat start token, got {ids[0]}")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-json", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="optional cap for smoke tests (0 = full corpus)",
    )
    args = parser.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        print(
            "tokenizers package required (pip install tokenizers). "
            f"Import failed: {exc}",
            file=sys.stderr,
        )
        return 2

    tok_path = args.tokenizer_json.expanduser().resolve()
    if not tok_path.is_file():
        print(f"tokenizer missing: {tok_path}", file=sys.stderr)
        return 2
    tokenizer = Tokenizer.from_file(str(tok_path))

    corpus = build_corpus()
    if args.max_prompts > 0:
        corpus = corpus[: args.max_prompts]
    if len(corpus) < 12:
        print("need at least 12 prompts for broad schema", file=sys.stderr)
        return 2

    probes: list[dict[str, Any]] = []
    domain_counts: dict[str, int] = {}
    for row in corpus:
        ids = one_user_native_prompt(tokenizer, row["text"])
        domain_counts[row["domain"]] = domain_counts.get(row["domain"], 0) + 1
        probes.append(
            {
                "probe_id": row["probe_id"],
                "domain": row["domain"],
                "source_one_user_native_prompt": {
                    "token_ids": ids,
                    "token_count": len(ids),
                    "token_ids_u32le_sha256": token_ids_u32le_sha256(ids),
                    "add_special_tokens": True,
                    "chat_shape": "one_user_message_no_system_no_tools",
                    "user_text_sha256": sha256_bytes(row["text"].encode("utf-8")),
                },
                "user_text": row["text"],
            }
        )

    out_dir = args.out_dir.expanduser().resolve()
    req_dir = out_dir / "requests"
    req_dir.mkdir(parents=True, exist_ok=True)

    document: dict[str, Any] = {
        "schema": BROAD_INPUT_SCHEMA,
        "status": STATUS,
        "prepared_at": utc_now(),
        "purpose": (
            "broaden L0 router-input activations so constant-mean null can be "
            "re-priced honestly for activation-aware family ranking"
        ),
        "claim_boundary": {
            "model_execution_started": False,
            "new_diagnostic_not_historical": True,
            "diagnostic_activation_pricing_only": True,
            "does_not_claim_coherence_hcli_tps_or_capability": True,
            "not_hcli_compiler_trace_bound": True,
            "source_tokenizer_one_user_native_prompts": True,
            "production_server_not_used": True,
        },
        "tokenizer_binding": {
            "path": str(tok_path),
            "sha256": sha256_file(tok_path),
        },
        "corpus_summary": {
            "probe_count": len(probes),
            "domain_counts": domain_counts,
            "total_tokens": sum(p["source_one_user_native_prompt"]["token_count"] for p in probes),
            "min_tokens": min(p["source_one_user_native_prompt"]["token_count"] for p in probes),
            "max_tokens": max(p["source_one_user_native_prompt"]["token_count"] for p in probes),
            "mean_tokens": round(
                sum(p["source_one_user_native_prompt"]["token_count"] for p in probes) / len(probes),
                2,
            ),
        },
        "probes": probes,
    }

    # Content-addressed filename
    body_for_hash = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = sha256_bytes(body_for_hash.encode("utf-8"))
    out_path = req_dir / f"QWEN30_BROAD_ACTIVATION_L0_ROUTE_CAPTURE_INPUT_{digest[:16]}.json"
    out_path.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    file_sha = sha256_file(out_path)

    provenance = {
        "schema": "hawking.ascension.qwen30_broad_activation_capture_prepare.v1",
        "prepared_at": utc_now(),
        "input_path": str(out_path),
        "input_sha256": file_sha,
        "corpus_summary": document["corpus_summary"],
        "claim_boundary": document["claim_boundary"],
        "tokenizer_binding": document["tokenizer_binding"],
        "note": (
            "Input only. Metal L0 capture not started. Live production server "
            "must not be disturbed; run capture serialized if GPU residency conflicts."
        ),
    }
    prov_path = out_dir / "PREPARE_PROVENANCE.json"
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"input_path": str(out_path), "input_sha256": file_sha, **document["corpus_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
