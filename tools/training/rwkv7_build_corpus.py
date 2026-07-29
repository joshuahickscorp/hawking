#!/usr/bin/env python3
"""Build an instruct SFT corpus for the on-device RWKV7-g1-0.4B post-train.

CPU/IO ONLY. This script does NO model forward and touches NO GPU — it only
downloads text (IO) and reshapes it (CPU). It is safe to run while another
process is benchmarking the GPU.

What it produces (under a gitignored data dir, default `artifacts/rwkv7_posttrain/`):

  sft.jsonl       one JSON object per line:
                    {"messages": [{"role": "user", "content": ...},
                                  {"role": "assistant", "content": ...}],
                     "text": "<rwkv-chat-formatted string>",
                     "source": "openhermes2.5" | "ultrachat" | "seed",
                     "bucket": "code" | "chat" | "reason"}
  dpo_prompts.jsonl   prompt-only scaffold for the on-policy DPO step
                      (the chosen/rejected pairs are filled later by the
                      Qwen2.5-3B teacher on the GPU box — see the runbook):
                    {"prompt": "<rwkv-chat string ending in 'Assistant:'>",
                     "bucket": ...}
  manifest.json   provenance: counts per source/bucket, dataset names,
                  the exact filter rules, and a content hash.

The RWKV-7 "g1" chat format (from fla-hub/rwkv7-0.4B-g1 tokenizer_config.json):

    <|rwkv_tokenizer_end_of_text|>User: {user}\n\nAssistant: {assistant}\n\n

`\n\n` is both the turn separator and the model's EOS string. We emit that
literal text in the "text" field so the SFT trainer can tokenize it directly
with the RWKV vocab; `messages` is kept so a HF chat_template path works too.

Usage
-----
    # default: ~3000 examples skewed code+chat, written to artifacts/
    python3 tools/training/rwkv7_build_corpus.py

    # bigger, custom split, custom out dir
    python3 tools/training/rwkv7_build_corpus.py \
        --n 6000 --code-frac 0.45 --chat-frac 0.40 --reason-frac 0.15 \
        --out artifacts/rwkv7_posttrain

    # offline / air-gapped: skip HF, use only the in-repo curated seed
    python3 tools/training/rwkv7_build_corpus.py --offline --n 200

    # also (re)write the small committed sample under tools/training/data/
    python3 tools/training/rwkv7_build_corpus.py --write-sample 40

Network is OPTIONAL. With `--offline` (or if HF is unreachable) the script
falls back to the curated seed set below and still emits a well-formed file,
so the build is deterministic and CI-safe. The seed set is tiny but real
(hand-written instruct triples) — enough to smoke-test the trainer wiring;
the HF path is what produces the few-thousand-example training corpus.

Idempotent: re-running overwrites the output files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

# RWKV-7 g1 chat control strings (see module docstring / tokenizer_config.json).
BOS = "<|rwkv_tokenizer_end_of_text|>"
SEP = "\n\n"  # turn separator AND eos


# --------------------------------------------------------------------------- #
# Curated offline seed. Real, hand-written instruct triples across the three
# buckets. Deliberately small — this is the air-gapped fallback + wiring
# smoke-test, NOT the training corpus. Keep these short and clean.
# --------------------------------------------------------------------------- #
SEED: list[tuple[str, str, str]] = [
    # ---- code ----
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively.",
     "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\nThis runs in O(n) time and O(1) space."),
    ("code", "In Rust, how do I read a file into a String?",
     "Use `std::fs::read_to_string`:\n\n```rust\nuse std::fs;\nlet contents = fs::read_to_string(\"path.txt\")?;\n```\n\nIt returns `io::Result<String>`, so call it inside a function that returns a `Result` and use `?` to propagate errors."),
    ("code", "What does `git rebase --onto` do?",
     "`git rebase --onto <newbase> <upstream> <branch>` replays the commits in `<branch>` that are not in `<upstream>` on top of `<newbase>`. It's how you move a range of commits onto a different base without dragging along the commits in between."),
    ("code", "Explain the difference between a list and a tuple in Python.",
     "A list is mutable: you can append, remove, and reassign elements after creation, and it's written with `[]`. A tuple is immutable: once created its contents can't change, and it's written with `()`. Tuples are slightly faster and can be used as dictionary keys; lists cannot."),
    ("code", "How do I deduplicate a vector in Rust while preserving order?",
     "Track seen items in a `HashSet`:\n\n```rust\nuse std::collections::HashSet;\nlet mut seen = HashSet::new();\nv.retain(|x| seen.insert(x.clone()));\n```\n\n`HashSet::insert` returns `false` if the value was already present, so `retain` drops duplicates while keeping the first occurrence's order."),
    ("code", "Write a SQL query to get the second-highest salary from an Employee table.",
     "```sql\nSELECT MAX(salary) AS second_highest\nFROM Employee\nWHERE salary < (SELECT MAX(salary) FROM Employee);\n```\n\nThe subquery finds the top salary, and the outer query takes the largest value strictly below it."),
    # ---- chat ----
    ("chat", "I'm feeling overwhelmed by my to-do list. Any advice?",
     "Start by writing everything down so it's out of your head. Then pick just the top three items that actually matter today and ignore the rest for now. Breaking big tasks into a single concrete next action makes them feel far less daunting, and finishing even one builds momentum."),
    ("chat", "Suggest a simple weeknight dinner I can make in 20 minutes.",
     "Try a quick stir-fry: cook some rice or noodles, then sautee whatever vegetables you have with garlic, soy sauce, and a protein like eggs, tofu, or pre-cooked chicken. It's flexible, fast, and uses one pan."),
    ("chat", "What's a good way to remember someone's name when I first meet them?",
     "Repeat their name back right after they say it (\"Nice to meet you, Sam\"), then use it once more during the conversation. Associating the name with a vivid detail about them also helps it stick."),
    ("chat", "Explain what a black hole is to a ten-year-old.",
     "A black hole is a place in space where gravity pulls so hard that nothing can escape it, not even light. It forms when a giant star runs out of fuel and collapses. Because no light comes back out, it looks completely black, which is how it got its name."),
    ("chat", "Give me a one-sentence pep talk before a job interview.",
     "You've already earned this conversation by being qualified, so walk in curious rather than scared, and just let them meet the person who's clearly good enough to be in the room."),
    # ---- reason ----
    ("reason", "If a shirt costs $40 after a 20% discount, what was the original price?",
     "Let the original price be P. After a 20% discount you pay 80% of P, so 0.8 * P = 40. Dividing both sides by 0.8 gives P = 50. The original price was $50."),
    ("reason", "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much is the ball?",
     "Let the ball cost x. The bat is x + 1.00, and together they are x + (x + 1.00) = 1.10. So 2x + 1.00 = 1.10, meaning 2x = 0.10 and x = 0.05. The ball costs 5 cents."),
    ("reason", "Is 91 a prime number? Explain.",
     "No. 91 = 7 * 13, so it has divisors other than 1 and itself, which means it is composite, not prime. A quick check is that 91 divided by 7 is exactly 13."),
    ("reason", "I have three apples and I give away two, then buy five more. How many do I have?",
     "Start with 3. Giving away 2 leaves 3 - 2 = 1. Buying 5 more gives 1 + 5 = 6. You have 6 apples."),
    ("reason", "Which is heavier: a kilogram of feathers or a kilogram of steel?",
     "They weigh the same. A kilogram is a kilogram regardless of the material, so a kilogram of feathers and a kilogram of steel both weigh exactly one kilogram. The steel just takes up much less space."),
]


def rwkv_format(user: str, assistant: str) -> str:
    """Render one single-turn example in the RWKV-7 g1 chat format."""
    return f"{BOS}User: {user.strip()}{SEP}Assistant: {assistant.strip()}{SEP}"


def rwkv_prompt(user: str) -> str:
    """Prompt-only render (ends right before the assistant turn) for DPO/eval."""
    return f"{BOS}User: {user.strip()}{SEP}Assistant:"


def _clean(s: str) -> str:
    return (s or "").replace("\r\n", "\n").strip()


def _ok_text(user: str, assistant: str, min_a: int, max_a: int) -> bool:
    if not user or not assistant:
        return False
    if len(user) < 8 or len(assistant) < min_a:
        return False
    if len(assistant) > max_a:
        return False
    # Drop turns that smell like multi-turn role markers leaking through.
    low = (user + assistant).lower()
    if "<|im_start|>" in low or "<|endoftext|>" in low:
        return False
    return True


def _looks_codey(text: str) -> bool:
    t = text.lower()
    hits = ("```" in text) + sum(
        kw in t for kw in (
            "def ", "class ", "import ", "function", "const ", "let ",
            "public ", "return ", "#include", "sql", "select ", "println",
            "fn ", "use std", "async ", "await ", "regex", "stack trace",
            "compile", "git ", "docker", "bash", "shell", "json", "api",
        )
    )
    return ("```" in text) or hits >= 2


def from_openhermes(n: int, min_a: int, max_a: int, rng: random.Random):
    """Pull single-turn (user->assistant) pairs from OpenHermes-2.5.

    OpenHermes is code+reasoning heavy, which is exactly the skew we want.
    Yields (bucket_hint, user, assistant).
    """
    from datasets import load_dataset  # local import: only needed online
    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    taken = 0
    for row in ds:
        convs = row.get("conversations") or []
        user = assistant = None
        for i in range(len(convs) - 1):
            a, b = convs[i], convs[i + 1]
            if a.get("from") in ("human", "user") and b.get("from") in ("gpt", "assistant"):
                user, assistant = _clean(a.get("value")), _clean(b.get("value"))
                break
        if not _ok_text(user, assistant, min_a, max_a):
            continue
        bucket = "code" if _looks_codey(user + "\n" + assistant) else "reason"
        yield bucket, user, assistant
        taken += 1
        if taken >= n:
            break


def from_ultrachat(n: int, min_a: int, max_a: int, rng: random.Random):
    """Pull single-turn chat pairs from UltraChat-200k (first user/assistant)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    taken = 0
    for row in ds:
        msgs = row.get("messages") or []
        user = assistant = None
        for i in range(len(msgs) - 1):
            if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant":
                user, assistant = _clean(msgs[i]["content"]), _clean(msgs[i + 1]["content"])
                break
        if not _ok_text(user, assistant, min_a, max_a):
            continue
        yield "chat", user, assistant
        taken += 1
        if taken >= n:
            break


def gather(n: int, code_frac: float, chat_frac: float, reason_frac: float,
           min_a: int, max_a: int, offline: bool, rng: random.Random):
    """Return a list of dict examples, bucket-balanced as closely as the
    available sources allow. Falls back to the curated seed for any shortfall
    or when --offline / HF is unreachable."""
    want_code = int(round(n * code_frac))
    want_chat = int(round(n * chat_frac))
    want_reason = max(0, n - want_code - want_chat)  # remainder -> reason
    buckets: dict[str, list[dict]] = {"code": [], "chat": [], "reason": []}

    def add(source: str, bucket: str, user: str, assistant: str):
        buckets[bucket].append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "text": rwkv_format(user, assistant),
            "source": source,
            "bucket": bucket,
        })

    if not offline:
        # OpenHermes supplies code + reason; pull extra so the codey filter
        # can sort them. UltraChat supplies chat.
        try:
            need_hermes = int((want_code + want_reason) * 1.4) + 16
            for bucket, u, a in from_openhermes(need_hermes, min_a, max_a, rng):
                if bucket == "code" and len(buckets["code"]) < want_code:
                    add("openhermes2.5", "code", u, a)
                elif bucket == "reason" and len(buckets["reason"]) < want_reason:
                    add("openhermes2.5", "reason", u, a)
                if len(buckets["code"]) >= want_code and len(buckets["reason"]) >= want_reason:
                    break
        except Exception as e:  # noqa: BLE001 - any HF/network failure -> fallback
            print(f"[warn] OpenHermes unavailable ({e!r}); using seed for code+reason",
                  file=sys.stderr)
        try:
            for bucket, u, a in from_ultrachat(want_chat + 16, min_a, max_a, rng):
                if len(buckets["chat"]) < want_chat:
                    add("ultrachat", "chat", u, a)
                else:
                    break
        except Exception as e:  # noqa: BLE001
            print(f"[warn] UltraChat unavailable ({e!r}); using seed for chat",
                  file=sys.stderr)

    # Top up any shortfall (incl. fully-offline) from the curated seed, cycling.
    seed_by_bucket: dict[str, list[tuple[str, str, str]]] = {"code": [], "chat": [], "reason": []}
    for b, u, a in SEED:
        seed_by_bucket[b].append((b, u, a))
    for bucket, want in (("code", want_code), ("chat", want_chat), ("reason", want_reason)):
        pool = seed_by_bucket[bucket] or SEED
        i = 0
        while len(buckets[bucket]) < want and pool:
            _, u, a = pool[i % len(pool)]
            add("seed", bucket, u, a)
            i += 1
            # avoid an infinite loop if want is huge and pool tiny: cap seed
            # duplication at 50x the pool, then stop padding this bucket.
            if i > 50 * len(pool):
                break

    out = buckets["code"] + buckets["chat"] + buckets["reason"]
    rng.shuffle(out)
    return out, {"code": len(buckets["code"]), "chat": len(buckets["chat"]),
                 "reason": len(buckets["reason"])}


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3000, help="total examples (default 3000)")
    ap.add_argument("--code-frac", type=float, default=0.45)
    ap.add_argument("--chat-frac", type=float, default=0.40)
    ap.add_argument("--reason-frac", type=float, default=0.15)
    ap.add_argument("--min-answer-chars", type=int, default=40)
    ap.add_argument("--max-answer-chars", type=int, default=2400,
                    help="drop very long answers (keeps seqs trainable on mps)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/rwkv7_posttrain"))
    ap.add_argument("--offline", action="store_true",
                    help="skip HF; build from the curated seed only")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--write-sample", type=int, default=0, metavar="K",
                    help="also write tools/training/data/rwkv7_sft_sample.jsonl "
                         "with K examples (committed; provenance only)")
    args = ap.parse_args()

    tot = args.code_frac + args.chat_frac + args.reason_frac
    if abs(tot - 1.0) > 1e-6:
        print(f"[warn] fracs sum to {tot:.3f}, not 1.0 — normalizing", file=sys.stderr)
        args.code_frac /= tot
        args.chat_frac /= tot
        args.reason_frac /= tot

    rng = random.Random(args.seed)
    rows, counts = gather(args.n, args.code_frac, args.chat_frac, args.reason_frac,
                          args.min_answer_chars, args.max_answer_chars,
                          args.offline, rng)

    sft_path = args.out / "sft.jsonl"
    dpo_path = args.out / "dpo_prompts.jsonl"
    manifest_path = args.out / "manifest.json"

    write_jsonl(sft_path, rows)
    # DPO prompt scaffold: dedup user prompts, render prompt-only.
    seen = set()
    dpo_rows = []
    for r in rows:
        u = r["messages"][0]["content"]
        key = u[:200]
        if key in seen:
            continue
        seen.add(key)
        dpo_rows.append({"prompt": rwkv_prompt(u), "bucket": r["bucket"]})
    write_jsonl(dpo_path, dpo_rows)

    # content hash over the SFT text for provenance/repro.
    h = hashlib.sha256()
    for r in rows:
        h.update(r["text"].encode("utf-8"))
    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    manifest = {
        "n": len(rows),
        "counts_by_bucket": counts,
        "counts_by_source": by_source,
        "dpo_prompts": len(dpo_rows),
        "format": "rwkv7-g1 chat: '<|rwkv_tokenizer_end_of_text|>User: {u}\\n\\nAssistant: {a}\\n\\n'",
        "sources": {
            "openhermes2.5": "teknium/OpenHermes-2.5 (code+reasoning heavy)",
            "ultrachat": "HuggingFaceH4/ultrachat_200k split=train_sft (chat)",
            "seed": "in-repo curated fallback (tools/training/rwkv7_build_corpus.py SEED)",
        },
        "filters": {
            "min_answer_chars": args.min_answer_chars,
            "max_answer_chars": args.max_answer_chars,
            "single_turn_only": True,
            "code_detector": "fenced-block or >=2 code keywords -> bucket=code",
        },
        "offline": args.offline,
        "seed": args.seed,
        "sha256_of_sft_text": h.hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.write_sample:
        sample_path = Path("tools/training/data/rwkv7_sft_sample.jsonl")
        # Prefer seed rows for the committed sample so it never embeds large
        # third-party text; fall back to head() if seed underrepresented.
        seed_rows = [r for r in rows if r["source"] == "seed"]
        pick = (seed_rows or rows)[: args.write_sample]
        write_jsonl(sample_path, pick)
        print(f"[ok] wrote committed sample: {sample_path} ({len(pick)} rows)")

    print(f"[ok] SFT corpus:   {sft_path}  ({len(rows)} rows)")
    print(f"[ok] DPO prompts:  {dpo_path}  ({len(dpo_rows)} rows)")
    print(f"[ok] manifest:     {manifest_path}")
    print(f"      by bucket: {counts}")
    print(f"      by source: {by_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
