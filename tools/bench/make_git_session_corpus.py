#!/usr/bin/env python3
"""Build a NON-CIRCULAR coding-session corpus from real git history, for the
§8.1 L1.2 prefix-cache oracle (`oracle_prefix_cache.py`).

WHY (vs the earlier proxy)
--------------------------
The first prefix-cache proxy re-read the same repo files + appended a synthetic
diff — it *engineered* the shared prefix, so its 86.8% reuse was partly circular.
This generator instead models an agent iterating on a feature using REAL commit
history: a working set of files is re-sent each turn, and the only thing that
changes between consecutive turns is the ACTUAL diff of the next commit that
touched those files. The prefix overlap is therefore whatever real consecutive
commits happen to share — not something we put in. It's fully standalone (any
local repo; defaults to this one), no external product or workload needed.

MODEL
-----
A "session" = a contiguous window of commits. Its working set = the `--ws` files
most frequently touched in that window (text, size-capped). Each turn = a fixed
system preamble + the working-set files' contents AT that commit (`git show
<sha>:<path>`). Turns that don't touch the working set are skipped so the session
stays coherent (an agent iterating on those files). Consecutive turns differ only
by real diffs → realistic, un-engineered prefix overlap.

Emits one JSONL per session ({"request": ...} per line, in commit order) — the
exact input `oracle_prefix_cache.py --jsonl` expects.

Run:
    tools/bench/make_git_session_corpus.py --repo . --out-dir /tmp/git_sessions
    for f in /tmp/git_sessions/session_*.jsonl; do
        tools/bench/oracle_prefix_cache.py --jsonl "$f" --out "${f%.jsonl}_report.md"
    done
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

PREAMBLE = (
    "You are a coding assistant working on a software repository. The current "
    "working set of files is included below in full. Read them, then make the "
    "change described in the task. Preserve existing style and keep the diff "
    "minimal.\n\n=== WORKING SET ===\n"
)

TEXT_EXT = {
    ".rs", ".py", ".metal", ".md", ".toml", ".sh", ".txt", ".json", ".jsonl",
    ".c", ".h", ".cpp", ".hpp", ".js", ".ts", ".yaml", ".yml", ".cfg", ".ini",
}


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, errors="replace",
    )


def is_texty(path):
    return os.path.splitext(path)[1].lower() in TEXT_EXT


def files_touched(repo, sha):
    r = git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", sha)
    return [f for f in r.stdout.splitlines() if f.strip()]


def file_at(repo, sha, path, max_bytes):
    r = git(repo, "show", f"{sha}:{path}")
    if r.returncode != 0:
        return None  # file didn't exist at this commit (added later / renamed)
    content = r.stdout
    if not content or len(content.encode("utf-8", "replace")) > max_bytes:
        return None
    return content


def build_session(repo, window, ws_size, max_bytes):
    # working set = most-frequently-touched texty files across the window
    counts = {}
    for sha in window:
        for f in files_touched(repo, sha):
            if is_texty(f):
                counts[f] = counts.get(f, 0) + 1
    ws = [f for f, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:ws_size]]
    if not ws:
        return []
    ws_set = set(ws)
    turns = []
    for sha in window:
        if not (ws_set & set(files_touched(repo, sha))):
            continue  # skip turns that don't touch the working set
        req = PREAMBLE
        for f in ws:
            content = file_at(repo, sha, f, max_bytes)
            if content is None:
                continue
            req += f"\n# FILE: {f}\n{content}\n"
        req += f"\n=== TASK ===\nApply the change for commit {sha[:10]}.\n"
        turns.append(req)
    return turns


def main():
    ap = argparse.ArgumentParser(prog="make_git_session_corpus")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out-dir", default="/tmp/git_sessions")
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--turns", type=int, default=10,
                    help="commits per session window")
    ap.add_argument("--ws", type=int, default=3,
                    help="working-set size (files re-sent each turn)")
    ap.add_argument("--max-file-bytes", type=int, default=24000)
    ap.add_argument("--skip-recent", type=int, default=0,
                    help="skip the N most recent commits (e.g. avoid this session's own)")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    need = args.sessions * args.turns + args.skip_recent
    r = git(repo, "log", "--no-merges", "--format=%H", "-n", str(need))
    shas = r.stdout.split()[args.skip_recent:]
    shas = list(reversed(shas))  # oldest -> newest

    written = []
    for s in range(args.sessions):
        window = shas[s * args.turns:(s + 1) * args.turns]
        if len(window) < 2:
            break
        turns = build_session(repo, window, args.ws, args.max_file_bytes)
        if len(turns) < 2:
            continue
        out = out_dir / f"session_{s:02d}.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for t in turns:
                fh.write(json.dumps({"request": t}) + "\n")
        written.append((out, len(turns)))
        print(f"[corpus] {out.name}: {len(turns)} turns, "
              f"ws={args.ws}, ~{sum(len(t) for t in turns)//1024} KB")

    print(f"[corpus] wrote {len(written)} session files to {out_dir} "
          f"from real history of {repo.name}")
    if not written:
        print("[corpus] WARNING: no sessions produced (history too shallow or "
              "working sets empty)")


if __name__ == "__main__":
    main()
