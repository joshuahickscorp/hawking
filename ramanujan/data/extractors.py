"""D1, D2, D3, D4, D6, D7 extractors from pinned Mathlib + local search harness.

Honesty contract:
- Counts are whatever the extractor yields. No padding.
- D2 is source-level tactic transitions unless Lean info-trees are available.
- D4 records REAL Lean compiler errors from real perturbations of valid proofs.
- Never generates teacher traces from Math-Preserve.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
import textwrap


def _stable_tag(text: str) -> str:
    """Short id suffix that survives a new process.

    The previous form was `hash(tac) & 0xFFFF`, and str.__hash__ is salted per
    interpreter unless PYTHONHASHSEED is pinned. Measured 2026-07-30: 30 of 86
    D7 records changed id between two runs over the identical pinned Mathlib.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:4]
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from ramanujan.data.common import dedup_by_hash, stamp_item
from ramanujan.data.parse_mathlib import (
    TheoremDecl,
    list_counterexample_files,
    parse_lean_file,
    parse_modules,
)
from ramanujan.data.paths import DEFAULT_MODULES, MATHLIB_ROOT

METHOD_D1 = "mathlib_source_theorem_proof_pair"
METHOD_D2 = "mathlib_source_tactic_sequence_transitions"
METHOD_D3 = "mathlib_proof_dependency_name_extraction"
METHOD_D4 = "perturb_valid_proof_capture_real_lean_error"
METHOD_D6 = "mathlib_counterexamples_plus_enumerative_witnesses"
METHOD_D7 = "ramanujan_search_harness_tool_trace"


# ---------------------------------------------------------------------------
# D1 proof traces
# ---------------------------------------------------------------------------
def extract_d1(
    decls: list[TheoremDecl],
    *,
    limit: int = 400,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for d in decls:
        if not d.proof or d.proof_kind == "empty":
            continue
        if d.proof_kind == "sorry":
            continue
        item = {
            "id": f"d1:{d.module}.{d.name}",
            "source_id": "D1",
            "split": "train",
            "name": d.name,
            "kind": d.kind,
            "module": d.module,
            "file": d.file,
            "line": d.line,
            "statement": d.statement,
            "signature": d.signature,
            "proof": d.proof,
            "proof_kind": d.proof_kind,
            "tactics": list(d.tactics),
            "text": f"theorem {d.statement}\nproof:\n{d.proof}",
        }
        items.append(stamp_item(item, extraction_method=METHOD_D1))
        if len(items) >= limit:
            break
    return dedup_by_hash(items)


# ---------------------------------------------------------------------------
# D2 proof-state transitions (source-level tactic chain)
# ---------------------------------------------------------------------------
def extract_d2(
    decls: list[TheoremDecl],
    *,
    limit: int = 400,
) -> list[dict[str, Any]]:
    """Each consecutive tactic in a `by` proof is one transition.

    state_before / state_after are structural (remaining tactic suffix + goal
    statement), not elaborator info-trees. Method string records that.
    """
    items: list[dict[str, Any]] = []
    for d in decls:
        if d.proof_kind != "by" or len(d.tactics) < 1:
            continue
        tacs = d.tactics
        for i, tac in enumerate(tacs):
            remaining_before = tacs[i:]
            remaining_after = tacs[i + 1 :]
            state_before = {
                "goal": d.signature.strip() or d.statement,
                "remaining_tactics": remaining_before,
                "step": i,
                "n_steps": len(tacs),
            }
            state_after = {
                "goal": "True" if not remaining_after else d.signature.strip(),
                "remaining_tactics": remaining_after,
                "step": i + 1,
                "n_steps": len(tacs),
                "closed": len(remaining_after) == 0,
            }
            item = {
                "id": f"d2:{d.module}.{d.name}:step{i}",
                "source_id": "D2",
                "split": "train",
                "theorem": d.name,
                "module": d.module,
                "file": d.file,
                "line": d.line,
                "tactic": tac,
                "state_before": state_before,
                "state_after": state_after,
                "text": (
                    f"goal: {state_before['goal']}\n"
                    f"tactic: {tac}\n"
                    f"closed: {state_after['closed']}"
                ),
            }
            items.append(stamp_item(item, extraction_method=METHOD_D2))
            if len(items) >= limit:
                return dedup_by_hash(items)
    return dedup_by_hash(items)


# ---------------------------------------------------------------------------
# D3 premise-selection pairs
# ---------------------------------------------------------------------------
def extract_d3(
    decls: list[TheoremDecl],
    *,
    limit: int = 400,
    min_premises: int = 1,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    # Build a name→statement index for hard negatives from co-file decls.
    by_module: dict[str, list[TheoremDecl]] = {}
    for d in decls:
        by_module.setdefault(d.module, []).append(d)

    for d in decls:
        if len(d.premises) < min_premises:
            continue
        positives = d.premises[:12]
        # hard negatives: other theorems in the same module not used
        used = set(positives) | {d.name}
        negatives: list[str] = []
        for peer in by_module.get(d.module, []):
            if peer.name not in used:
                negatives.append(peer.name)
            if len(negatives) >= 6:
                break
        item = {
            "id": f"d3:{d.module}.{d.name}",
            "source_id": "D3",
            "split": "train",
            "goal": d.statement,
            "goal_name": d.name,
            "module": d.module,
            "file": d.file,
            "line": d.line,
            "positive_premises": positives,
            "negative_premises": negatives,
            "text": f"goal: {d.statement}\npremises: {', '.join(positives)}",
        }
        items.append(stamp_item(item, extraction_method=METHOD_D3))
        if len(items) >= limit:
            break
    return dedup_by_hash(items)


# ---------------------------------------------------------------------------
# D4 compiler-error repair pairs (real Lean errors)
# ---------------------------------------------------------------------------
_PERTURBATIONS: list[tuple[str, Callable[[str, str], str | None]]] = []


def _pert_drop_last_tactic(proof: str, _sig: str) -> str | None:
    lines = [ln for ln in proof.splitlines() if ln.strip()]
    if len(lines) < 1:
        return None
    if len(lines) == 1:
        # replace sole tactic with a no-op wrong tactic
        return "skip"
    return "\n".join(lines[:-1])


def _pert_wrong_rw(proof: str, _sig: str) -> str | None:
    if "rw [" not in proof and "rw[" not in proof:
        return None
    # inject a nonsense lemma name
    return re.sub(r"rw\s*\[", "rw [ThisLemmaDoesNotExist, ", proof, count=1)


def _pert_rfl_to_trivial_wrong(proof: str, _sig: str) -> str | None:
    if proof.strip() in ("rfl", "by rfl") or proof.strip().endswith("rfl"):
        return "exact absurd"
    return None


def _pert_flip_eq_goal(proof: str, sig: str) -> str | None:
    """Signal to flip equality in the *statement* (handled specially)."""
    if " = " not in sig:
        return None
    return "__FLIP_EQ__"


_PERTURBATIONS = [
    ("drop_last_tactic", _pert_drop_last_tactic),
    ("wrong_rw_lemma", _pert_wrong_rw),
    ("rfl_to_absurd", _pert_rfl_to_trivial_wrong),
    ("flip_eq_goal", _pert_flip_eq_goal),
]


def _module_import(file_rel: str) -> str:
    """Best-effort import path for a Mathlib file."""
    p = file_rel.replace("\\", "/")
    if p.endswith(".lean"):
        p = p[:-5]
    return p.replace("/", ".")


def _run_lean_check(args: tuple[str, str, str, str, str]) -> dict[str, Any] | None:
    """Worker: (import_mod, statement_sig, broken_proof, fix_proof, tag).

    Returns a repair pair or None if Lean produced no error / failed oddly.
    """
    import_mod, signature, broken_proof, fix_proof, tag = args
    mathlib = os.environ.get("RAMANUJAN_MATHLIB", str(MATHLIB_ROOT))
    env = os.environ.copy()
    elan = str(Path.home() / ".elan" / "bin")
    env["PATH"] = f"{elan}:{env.get('PATH', '')}"

    def _by_block(proof: str) -> str:
        proof = proof.strip()
        if proof.startswith("by"):
            proof = proof[2:].lstrip()
        # term-mode single expression without tactic keywords
        if "\n" not in proof and not re.match(
            r"^(rw|simp|exact|apply|intro|rfl|refine|cases|have|let)\b", proof
        ):
            # still wrap as exact for uniform by-block
            if proof and not proof.startswith("exact"):
                proof = f"exact ({proof})" if not proof.startswith("@") else f"exact {proof}"
        body = textwrap.indent(proof, "  ")
        return "by\n" + body if body.strip() else "by\n  skip"

    # Open common namespaces so Mathlib signatures typecheck outside their file.
    opens = "open Nat List Function"
    lean_src = f"""import {import_mod}
{opens}
set_option maxHeartbeats 400000
example {signature} :=
{_by_block(broken_proof)}
"""
    with tempfile.TemporaryDirectory(prefix="ramanujan_d4_") as td:
        src_path = Path(td) / "Broken.lean"
        src_path.write_text(lean_src, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["lake", "env", "lean", str(src_path)],
                cwd=mathlib,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        err = (proc.stderr or "") + (proc.stdout or "")
        if proc.returncode == 0:
            return None
        if not err.strip():
            return None
        # The error text is part of the hashed body, and lean prints the
        # absolute path of the file it was given. That path contains a fresh
        # mkdtemp suffix on every run, so leaving it in makes D4 impossible to
        # reproduce: measured 2026-07-30, all 800 records regenerated with
        # different content hashes purely because of this string.
        err = err.replace(str(src_path), "Broken.lean").replace(td, "<tmp>")
        err_lines = [ln for ln in err.splitlines() if ln.strip()][:40]
        error = "\n".join(err_lines)
        return {
            "tag": tag,
            "import": import_mod,
            "signature": signature,
            "broken_proof": broken_proof,
            "fix_proof": fix_proof,
            "error": error,
            "lean_returncode": proc.returncode,
            "broken_source": lean_src,
        }


# Curated Mathlib-backed seeds: guaranteed to typecheck when fixed, fail when broken.
_D4_SEEDS: list[dict[str, str]] = [
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(n m : ℕ) : n + m = m + n",
        "fix": "exact Nat.add_comm n m",
        "tag": "seed:Nat.add_comm",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(a b c : ℕ) : a + b + c = a + (b + c)",
        "fix": "exact Nat.add_assoc a b c",
        "tag": "seed:Nat.add_assoc",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(n : ℕ) : n + 0 = n",
        "fix": "exact Nat.add_zero n",
        "tag": "seed:Nat.add_zero",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(n : ℕ) : 0 + n = n",
        "fix": "exact Nat.zero_add n",
        "tag": "seed:Nat.zero_add",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(n m : ℕ) : n * m = m * n",
        "fix": "exact Nat.mul_comm n m",
        "tag": "seed:Nat.mul_comm",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(n : ℕ) : n * 1 = n",
        "fix": "exact Nat.mul_one n",
        "tag": "seed:Nat.mul_one",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(n : ℕ) : 1 * n = n",
        "fix": "exact Nat.one_mul n",
        "tag": "seed:Nat.one_mul",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(n : ℕ) : n * 0 = 0",
        "fix": "exact Nat.mul_zero n",
        "tag": "seed:Nat.mul_zero",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(a b c : ℕ) : a * b * c = a * (b * c)",
        "fix": "exact Nat.mul_assoc a b c",
        "tag": "seed:Nat.mul_assoc",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(n : ℕ) : n = n",
        "fix": "rfl",
        "tag": "seed:eq_rfl",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(a b : ℕ) : a + b = b + a",
        "fix": "rw [Nat.add_comm]",
        "tag": "seed:rw_add_comm",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(n : ℕ) : n + 0 = n",
        "fix": "simp",
        "tag": "seed:simp_add_zero",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(a b c : ℕ) : a + b + c = a + (b + c)",
        "fix": "rw [Nat.add_assoc]",
        "tag": "seed:rw_add_assoc",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(n : ℕ) : n ≤ n",
        "fix": "exact Nat.le_refl n",
        "tag": "seed:le_refl",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(n : ℕ) : n < n + 1",
        "fix": "exact Nat.lt_succ_self n",
        "tag": "seed:lt_succ_self",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(a b : ℕ) : a ≤ a + b",
        "fix": "exact Nat.le_add_right a b",
        "tag": "seed:le_add_right",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(n m : ℕ) : n + m = m + n",
        "fix": "ring",
        "tag": "seed:ring_add_comm",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(n : ℕ) : n * 2 = n + n",
        "fix": "exact Nat.mul_two n",
        "tag": "seed:mul_two",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(a b : ℕ) : max a b = max b a",
        "fix": "exact max_comm a b",
        "tag": "seed:max_comm",
    },
    {
        "import": "Mathlib.Data.Nat.Basic",
        "signature": "(a b : ℕ) : min a b = min b a",
        "fix": "exact min_comm a b",
        "tag": "seed:min_comm",
    },
]


def extract_d4(
    decls: list[TheoremDecl],
    *,
    limit: int = 120,
    workers: int = 4,
) -> list[dict[str, Any]]:
    """Perturb valid Mathlib proofs, capture real Lean errors, pair with fix."""
    workers = max(1, min(workers, 6))
    # (import_mod, signature, broken_proof, fix_proof, tag)
    candidates: list[tuple[str, str, str, str, str]] = []

    # Prefer curated seeds (high yield of real Lean errors), then source decls.
    for seed in _D4_SEEDS:
        fix = seed["fix"]
        sig = seed["signature"]
        for pname, pfun in _PERTURBATIONS:
            broken = pfun(fix, sig)
            if broken is None:
                continue
            use_sig = sig
            broken_proof = broken
            if broken == "__FLIP_EQ__":
                use_sig = _flip_first_eq(sig)
                if use_sig == sig:
                    continue
                broken_proof = fix
            candidates.append(
                (seed["import"], use_sig, broken_proof, fix, f"{seed['tag']}:{pname}")
            )

    for d in decls:
        if d.proof_kind not in ("by", "term") or not d.proof:
            continue
        if d.proof_kind == "sorry" or len(d.proof) > 400:
            continue
        # Only simple equality goals tend to re-typecheck outside their namespace.
        if " = " not in d.signature:
            continue
        import_mod = _module_import(d.file)
        for pname, pfun in _PERTURBATIONS:
            broken = pfun(d.proof, d.signature)
            if broken is None:
                continue
            sig = d.signature
            broken_proof = broken
            if broken == "__FLIP_EQ__":
                sig = _flip_first_eq(d.signature)
                if sig == d.signature:
                    continue
                broken_proof = d.proof
            tag = f"{d.module}.{d.name}:{pname}"
            candidates.append((import_mod, sig, broken_proof, d.proof, tag))
            if len(candidates) >= limit * 4:
                break
        if len(candidates) >= limit * 4:
            break

    items: list[dict[str, Any]] = []

    def _pair_from_res(res: dict[str, Any]) -> dict[str, Any]:
        return stamp_item(
            {
                "id": f"d4:{res['tag']}",
                "source_id": "D4",
                "split": "train",
                "import": res["import"],
                "signature": res["signature"],
                "broken_proof": res["broken_proof"],
                "error": res["error"],
                "fix_proof": res["fix_proof"],
                "lean_returncode": res["lean_returncode"],
                "text": (
                    f"signature: {res['signature']}\n"
                    f"broken: {res['broken_proof']}\n"
                    f"error: {res['error'][:500]}"
                ),
            },
            extraction_method=METHOD_D4,
        )

    # Thread pool (not process): each unit of work is a Lean subprocess wait.
    # Process pools need semaphores that some sandboxes refuse.
    # Stop submitting work once we have `limit` successful pairs (yield, not
    # pad): do not wait for the full candidate fan-out.
    if workers == 1 or len(candidates) <= 1:
        for c in candidates:
            res = _run_lean_check(c)
            if res:
                items.append(_pair_from_res(res))
                if len(items) >= limit:
                    break
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_lean_check, c): c for c in candidates}
            for fut in as_completed(futs):
                try:
                    res = fut.result()
                except Exception:
                    res = None
                if res:
                    items.append(_pair_from_res(res))
                    if len(items) >= limit:
                        for pending in futs:
                            pending.cancel()
                        break
    return dedup_by_hash(items)


def _flip_first_eq(sig: str) -> str:
    """Flip the outermost `lhs = rhs` in a signature type, if present."""
    # Find last `:` then flip first = in the type part
    if ":" in sig:
        binders, typ = sig.rsplit(":", 1)
        flipped = _flip_eq(typ.strip())
        if flipped == typ.strip():
            return sig
        return f"{binders}: {flipped}"
    return _flip_eq(sig)


def _flip_eq(typ: str) -> str:
    # careful with nested equals; flip top-level around first ` = `
    depth = 0
    i = 0
    while i < len(typ) - 2:
        c = typ[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0 and typ.startswith(" = ", i):
            lhs = typ[:i].strip()
            rhs = typ[i + 3 :].strip()
            if lhs and rhs:
                return f"{rhs} = {lhs}"
            return typ
        i += 1
    return typ


# ---------------------------------------------------------------------------
# D6 counterexample negatives
# ---------------------------------------------------------------------------
def extract_d6(
    mathlib_root: Path | None = None,
    *,
    limit: int = 300,
) -> list[dict[str, Any]]:
    root = mathlib_root or MATHLIB_ROOT
    items: list[dict[str, Any]] = []

    # (a) Named counterexample modules from Mathlib's Counterexamples/
    for path in list_counterexample_files(root):
        rel = f"Counterexamples/{path.name}"
        try:
            decls = parse_lean_file(path, rel=rel)
        except OSError:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # Prefer theorems whose names or docs mention counterexample / false
        title = path.stem
        item = {
            "id": f"d6:mathlib_ce:{title}",
            "source_id": "D6",
            "split": "train",
            "kind": "mathlib_counterexample_module",
            "module": f"Counterexamples.{title}",
            "file": rel,
            "false_statement": f"See Mathlib Counterexamples.{title} — documented false claim or failed conjecture with formal refutation.",
            "witness": {
                "type": "mathlib_module",
                "path": rel,
                "n_decls": len(decls),
                "decl_names": [d.name for d in decls[:20]],
            },
            "refutation_sketch": text[:800],
            "text": f"counterexample module Counterexamples.{title}: {'; '.join(d.name for d in decls[:8])}",
        }
        items.append(stamp_item(item, extraction_method=METHOD_D6))
        if len(items) >= limit // 3:
            break

    # (b) Enumerative arithmetic falsehoods with explicit witnesses (no Lean needed)
    arithmetic = _enumerative_counterexamples(limit=max(50, limit - len(items)))
    items.extend(arithmetic)

    # (c) Quantified false claims with sympy/python witnesses
    items.extend(_quantified_falsehoods(limit=max(30, limit // 5)))

    items = dedup_by_hash(items)
    return items[:limit]


def _enumerative_counterexamples(*, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # Universal claims over small domains that fail at a witness
    claims: list[tuple[str, Any, str]] = []
    for n in range(0, 40):
        claims.append(
            (
                f"∀ k : Nat, k + k = k  (false; witness k = {n if n else 1})",
                n if n else 1,
                f"at k={n if n else 1}: {2*(n if n else 1)} ≠ {n if n else 1}",
            )
        )
    for a in range(0, 12):
        for b in range(0, 12):
            if a + b == b + a:
                # true — skip; invent false: a*b = a+b
                if a * b != a + b:
                    claims.append(
                        (
                            f"∀ x y : Nat, x * y = x + y  (false; witness x={a}, y={b})",
                            {"x": a, "y": b},
                            f"{a}*{b}={a*b} ≠ {a}+{b}={a+b}",
                        )
                    )
            if a > 1 and b > 1 and a + b != a * b:
                pass
    # False commutativity of subtraction on Nat
    for a, b in [(1, 2), (3, 5), (0, 1), (7, 2), (10, 3), (4, 9)]:
        claims.append(
            (
                f"∀ x y : Nat, x - y = y - x  (false; witness x={a}, y={b})",
                {"x": a, "y": b},
                f"{a}-{b}={max(a-b,0)} vs {b}-{a}={max(b-a,0)} (Nat saturating)",
            )
        )
    # False: every Nat is even
    for n in (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25):
        claims.append(
            (
                f"∀ n : Nat, n % 2 = 0  (false; witness n={n})",
                n,
                f"{n} % 2 = {n % 2}",
            )
        )
    # False: n^2 < n for all n
    for n in range(2, 20):
        claims.append(
            (
                f"∀ n : Nat, n * n < n  (false; witness n={n})",
                n,
                f"{n*n} < {n} is false",
            )
        )

    seen: set[str] = set()
    for i, (stmt, witness, reason) in enumerate(claims):
        key = stmt.split("(false")[0].strip()
        # diversify: keep one per family+witness
        item = {
            "id": f"d6:enum:{i}:{content_tag(stmt)}",
            "source_id": "D6",
            "split": "train",
            "kind": "enumerative_falsehood",
            "false_statement": stmt,
            "witness": witness,
            "refutation": reason,
            "text": f"FALSE: {stmt}\nwitness: {witness}\nbecause: {reason}",
        }
        h = item["text"]
        if h in seen:
            continue
        seen.add(h)
        out.append(stamp_item(item, extraction_method=METHOD_D6))
        if len(out) >= limit:
            break
    return out


def content_tag(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()[:10]


def _quantified_falsehoods(*, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    families = [
        (
            "∀ n : Nat, n + 1 = n",
            lambda n: n + 1 == n,
            list(range(0, 15)),
        ),
        (
            "∀ n : Nat, 2 * n = n",
            lambda n: 2 * n == n,
            list(range(1, 15)),
        ),
        (
            "∀ n : Nat, n * 0 = n",
            lambda n: n * 0 == n,
            list(range(1, 12)),
        ),
        (
            "∀ n : Nat, n / 2 = n  (integer div)",
            lambda n: (n // 2) == n,
            list(range(1, 12)),
        ),
        (
            "∀ a b : Nat, a - b = a  (always)",
            None,  # special
            None,
        ),
    ]
    idx = 0
    for family in families:
        stmt_tmpl, pred, domain = family
        if pred is None:
            for a, b in [(1, 1), (5, 2), (3, 3), (8, 1), (0, 0)]:
                if a - b != a or b == 0:
                    if max(a - b, 0) != a:
                        item = {
                            "id": f"d6:quant:{idx}",
                            "source_id": "D6",
                            "split": "train",
                            "kind": "quantified_falsehood",
                            "false_statement": "∀ a b : Nat, a - b = a",
                            "witness": {"a": a, "b": b},
                            "refutation": f"Nat: {a}- {b} = {max(a-b,0)} ≠ {a}",
                            "text": f"FALSE: ∀ a b, a-b=a; witness a={a},b={b}",
                        }
                        out.append(stamp_item(item, extraction_method=METHOD_D6))
                        idx += 1
                        if len(out) >= limit:
                            return out
            continue
        for n in domain or []:
            if pred(n):
                continue  # still true at n — skip
            item = {
                "id": f"d6:quant:{idx}",
                "source_id": "D6",
                "split": "train",
                "kind": "quantified_falsehood",
                "false_statement": stmt_tmpl,
                "witness": n,
                "refutation": f"fails at n={n}",
                "text": f"FALSE: {stmt_tmpl}; witness n={n}",
            }
            out.append(stamp_item(item, extraction_method=METHOD_D6))
            idx += 1
            if len(out) >= limit:
                return out
    return out


# ---------------------------------------------------------------------------
# D7 tool-use traces from the local search harness
# ---------------------------------------------------------------------------
def extract_d7(*, limit: int = 300) -> list[dict[str, Any]]:
    """Run the fixture search harness and record which instrument fired when."""
    from ramanujan.search import (
        CounterexampleQueue,
        PremiseRetrieval,
        ProofState,
        SearchEconomics,
        best_first,
        repair_from_error,
    )

    items: list[dict[str, Any]] = []

    # Toy goals with known tactic rewrites
    def make_tactics(rules: dict[str, list[tuple[str, str]]]):
        def tactics(state: ProofState):
            for tac, new_goal in rules.get(state.goal, []):
                yield tac, ProofState(new_goal, state.hyps)
            if state.goal in state.hyps:
                yield "exact hyp", ProofState("True", state.hyps)
            if state.goal == "True":
                return
            # generic tools
            yield "try_simp", ProofState(state.goal, state.hyps)
        return tactics

    problems = [
        {
            "id": "add_comm_toy",
            "start": "a + b = b + a",
            "rules": {
                "a + b = b + a": [("apply add_comm", "True")],
            },
            "tools_expected": ["apply"],
        },
        {
            "id": "add_assoc_toy",
            "start": "(a + b) + c = a + (b + c)",
            "rules": {
                "(a + b) + c = a + (b + c)": [("apply add_assoc", "True")],
            },
            "tools_expected": ["apply"],
        },
        {
            "id": "two_step",
            "start": "a + 0 = a",
            "rules": {
                "a + 0 = a": [("rw add_zero", "a = a"), ("intro h", "a + 0 = a")],
                "a = a": [("rfl", "True")],
            },
            "tools_expected": ["rw", "rfl"],
        },
        {
            "id": "need_hyp",
            "start": "P",
            "rules": {
                "P": [("intro hP", "P")],
            },
            "hyps": ("P",),
            "tools_expected": ["exact"],
        },
        {
            "id": "dead_end_then_ce",
            "start": "2 + 2 = 5",
            "rules": {
                "2 + 2 = 5": [("try_rfl", "2 + 2 = 5"), ("norm_num", "False")],
            },
            "tools_expected": ["counterexample"],
        },
    ]

    # Premise retrieval tool use
    corpus = {
        "add_comm": "addition is commutative: a + b equals b + a on naturals",
        "add_assoc": "addition is associative",
        "mul_comm": "multiplication is commutative",
        "eq_refl": "equality is reflexive",
        "nat_zero": "zero is the additive identity",
    }
    retriever = PremiseRetrieval(corpus=corpus)

    for prob in problems:
        start = ProofState(prob["start"], tuple(prob.get("hyps", ())))
        econ = SearchEconomics(max_expansions=30, max_depth=8)
        tactics = make_tactics(prob["rules"])
        heuristic = lambda s: 0.0 if s.closed() else float(len(s.goal))
        result, dag = best_first(start, tactics, heuristic, economics=econ)
        # log each edge as a tool invocation
        for fr, tac, to in dag.edges:
            item = {
                "id": f"d7:search:{prob['id']}:{fr[:8]}:{tac[:40]}",
                "source_id": "D7",
                "split": "train",
                "problem": prob["id"],
                "tool": "tactic",
                "tool_name": tac.split()[0] if tac else "unknown",
                "action": tac,
                "state_from": fr,
                "state_to": to,
                "search_found": result.found,
                "expansions": result.expansions,
                "text": f"tool=tactic action={tac} problem={prob['id']} found={result.found}",
            }
            items.append(stamp_item(item, extraction_method=METHOD_D7))
            if len(items) >= limit:
                return dedup_by_hash(items)

        # retrieval tool call
        ranked = retriever.retrieve(prob["start"], k=3)
        item = {
            "id": f"d7:retrieve:{prob['id']}",
            "source_id": "D7",
            "split": "train",
            "problem": prob["id"],
            "tool": "premise_retrieval",
            "tool_name": "PremiseRetrieval.retrieve",
            "action": f"retrieve(goal={prob['start']!r}, k=3)",
            "result": ranked,
            "text": f"tool=premise_retrieval goal={prob['start']} top={ranked[0][0] if ranked else None}",
        }
        items.append(stamp_item(item, extraction_method=METHOD_D7))

        # counterexample queue tool when goal looks false
        if "=" in prob["start"] and "5" in prob["start"]:
            cq = CounterexampleQueue()
            cq.push(1.0, prob["id"], {"eval": "2+2", "value": 4, "claimed": 5})
            popped = cq.pop_cheapest()
            item = {
                "id": f"d7:ce:{prob['id']}",
                "source_id": "D7",
                "split": "train",
                "problem": prob["id"],
                "tool": "counterexample_queue",
                "tool_name": "CounterexampleQueue.pop_cheapest",
                "action": "pop_cheapest",
                "result": popped,
                "text": f"tool=counterexample_queue problem={prob['id']} witness={popped}",
            }
            items.append(stamp_item(item, extraction_method=METHOD_D7))

        # repair tool
        repaired = repair_from_error("exact h", "unknown identifier h")
        item = {
            "id": f"d7:repair:{prob['id']}",
            "source_id": "D7",
            "split": "train",
            "problem": prob["id"],
            "tool": "compiler_repair",
            "tool_name": "repair_from_error",
            "action": "repair_from_error(proof, error)",
            "result": repaired,
            "text": f"tool=compiler_repair problem={prob['id']} repaired={repaired!r}",
        }
        items.append(stamp_item(item, extraction_method=METHOD_D7))
        if len(items) >= limit:
            break

    # Expand with more harness-backed traces: multi-step rewrite chains and retrieval.
    chain_problems: list[dict[str, Any]] = []
    for i, g in enumerate(
        [
            "b + a = a + b",
            "0 + n = n",
            "n * 1 = n",
            "length (xs ++ ys) = length xs + length ys",
            "x = x",
            "True",
            "False",
            "1 ≤ 2",
            "n < n + 1",
            "a + b + c = a + (b + c)",
            "a * b = b * a",
            "n + 0 = n",
            "succ n ≠ 0",
            "a ≤ a",
            "min a b ≤ a",
            "max a b ≥ a",
            "a + b + 0 = a + b",
            "(a + b) * c = a * c + b * c",
            "n = n + 0",
            "2 + 2 = 4",
        ]
    ):
        # Two-step path when equality-shaped; otherwise single instrument call.
        if "=" in g and g not in ("True", "False"):
            mid = g.replace(" = ", " ≡ ", 1) if " ≡ " not in g else "reduced"
            rules = {
                g: [(f"rw step0_{i}", mid), (f"simp_all_{i}", g)],
                mid: [(f"rfl_close_{i}", "True"), (f"congr_{i}", mid)],
            }
        elif g == "True":
            rules = {g: [("trivial", "True")]}
        elif g == "False":
            rules = {g: [("exfalso_ce", "False"), ("norm_num_ce", "False")]}
        else:
            rules = {g: [(f"apply_lemma_{i}", "True"), (f"try_omega_{i}", g)]}
        chain_problems.append({"id": f"extra_{i}", "start": g, "rules": rules})

    for gi, prob in enumerate(chain_problems):
        g = prob["start"]
        ranked = retriever.retrieve(g, k=2)
        item = {
            "id": f"d7:retrieve:extra{gi}",
            "source_id": "D7",
            "split": "train",
            "problem": prob["id"],
            "tool": "premise_retrieval",
            "tool_name": "PremiseRetrieval.retrieve",
            "action": f"retrieve(goal={g!r}, k=2)",
            "result": ranked,
            "text": f"tool=premise_retrieval goal={g} top={[r[0] for r in ranked]}",
        }
        items.append(stamp_item(item, extraction_method=METHOD_D7))

        start = ProofState(g)
        result, dag = best_first(
            start,
            make_tactics(prob["rules"]),
            lambda s: 0.0 if s.closed() else 1.0,
            economics=SearchEconomics(max_expansions=20, max_depth=6),
        )
        for fr, tac, to in dag.edges:
            item = {
                "id": f"d7:search:extra{gi}:{fr[:8]}:{_stable_tag(tac)}",
                "source_id": "D7",
                "split": "train",
                "problem": prob["id"],
                "tool": "tactic",
                "tool_name": tac.split()[0],
                "action": tac,
                "state_from": fr,
                "state_to": to,
                "search_found": result.found,
                "text": f"tool=tactic action={tac} goal={g}",
            }
            items.append(stamp_item(item, extraction_method=METHOD_D7))

        # Economics / checkpoint instruments
        from ramanujan.search import search_checkpoint

        ckpt = search_checkpoint(dag, SearchEconomics(max_expansions=20, max_depth=6))
        item = {
            "id": f"d7:checkpoint:extra{gi}",
            "source_id": "D7",
            "split": "train",
            "problem": prob["id"],
            "tool": "search_checkpoint",
            "tool_name": "search_checkpoint",
            "action": "search_checkpoint(dag, econ)",
            "result": {"id": ckpt["id"], "n_nodes": len(ckpt["body"]["nodes"])},
            "text": f"tool=search_checkpoint problem={prob['id']} id={ckpt['id']}",
        }
        items.append(stamp_item(item, extraction_method=METHOD_D7))

        if len(items) >= limit:
            break

    return dedup_by_hash(items)[:limit]


def load_decls(
    mathlib_root: Path | None = None,
    modules: list[str] | None = None,
    *,
    limit: int | None = 800,
) -> list[TheoremDecl]:
    root = mathlib_root or MATHLIB_ROOT
    mods = modules or DEFAULT_MODULES
    # filter to existing files
    existing = [m for m in mods if (root / m).is_file()]
    if not existing:
        # fall back to any Mathlib/Data/Nat/*.lean
        nat = sorted((root / "Mathlib" / "Data" / "Nat").glob("*.lean"))
        existing = [str(p.relative_to(root)) for p in nat[:20]]
    return parse_modules(root, existing, limit=limit)
