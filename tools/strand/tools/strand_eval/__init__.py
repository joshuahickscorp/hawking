# strand_eval — THE canonical PPL eval module (audit measurement.md §3.1/§3.2).
#
# Consolidates the three diverging eval copies (strand-7b-ppl.sh's PPL_PY heredoc,
# ops/eval-ppl.py, strand-qat.py's eval_ppl) into ONE module with by-construction
# safety: derived output names (ppl_<model>_<tag>.json, collision-checked at write),
# a harness_key on every record (cross-harness comparisons mechanically detectable),
# and a results ledger with the 15-digit tell as code.
#
# Self-location is via __file__ (never cwd): the module works when invoked from /,
# from a copied tree, or via symlink. If the package is copied OUT of a strand repo,
# set STRAND_ROOT explicitly — it refuses to guess.
#
# Canon protocol (docs/STRAND-eval-canon.md): WikiText-2-raw-v1 test split, tokens
# joined "\n\n", NON-overlapping ctx windows, sum-CE, ppl = exp(Σnll / Σtok).
# bf16 (fp16 → NaN on Qwen). 64 windows = screening, 146 = anchors.

import os

HARNESS_VERSION = "1.0.0"
SCHEMA = 1

_PKG_DIR = os.path.dirname(os.path.realpath(__file__))


def _looks_like_repo(d):
    c = os.path.join(d, "Cargo.toml")
    try:
        with open(c, "r") as f:
            return "strand-quant" in f.read()
    except OSError:
        return False


def locate_repo_root():
    """Find the strand repo root by walking UP from this file (cwd-independent).

    Sentinel = a Cargo.toml mentioning strand-quant (the audit 3.3 location proof).
    Falls back to $STRAND_ROOT (verified against the same sentinel). Raises with
    every path it tried — a loud line-3 death, never a silent //Cargo.toml.
    """
    tried = []
    d = _PKG_DIR
    while True:
        tried.append(d)
        if _looks_like_repo(d):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    env = os.environ.get("STRAND_ROOT", "")
    if env:
        if _looks_like_repo(env):
            return os.path.realpath(env)
        raise RuntimeError(
            f"STRAND_ROOT={env!r} is not a strand repo (no Cargo.toml with strand-quant)")
    raise RuntimeError(
        "strand_eval cannot locate the repo root. Walked up from "
        f"{_PKG_DIR} through: {tried}. Set STRAND_ROOT=/path/to/strand.")


def default_ledger_path():
    return os.path.join(locate_repo_root(), "research", "results-ledger.jsonl")
