"""Read-only extraction of theorem/lemma declarations from Mathlib .lean sources.

This is a source-level parser, not a full Lean elaborator. It is honest about that:
proof terms and tactic sequences are taken from the surface syntax of the pinned
Mathlib checkout. No write to Mathlib.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Top-level declaration start (after optional modifiers).
_DECL_START = re.compile(
    r"^(?P<indent>\s*)"
    r"(?:(?:public|private|protected|scoped|noncomputable|unsafe)\s+)*"
    r"(?P<kind>theorem|lemma)\s+"
    r"(?P<name>[A-Za-z_][\w'.]*)"
)

# Names that look like Lean identifiers used as premises (not keywords).
_IDENT = re.compile(r"\b([A-Za-z_][\w']*(?:\.[A-Za-z_][\w']*)+)\b|\b([A-Za-z_][\w']{2,})\b")

_TACTIC_LINE = re.compile(
    r"^\s*(?P<tac>"
    r"rw|rwa|simp|simpa|exact|apply|refine|intro|intros|rintro|cases|rcases|"
    r"induction|constructor|ext|funext|congr|linarith|nlinarith|ring|ring_nf|"
    r"field_simp|norm_num|omega|lia|aesop|decide|rfl|trivial|assumption|"
    r"contradiction|exfalso|left|right|use|exists|convert|change|have|let|"
    r"suffices|obtain|specialize|simpa?|push_neg|by_cases|by_contra|contrapose|"
    r"calc|gcongr|positivity|finiteness|continuity|measurability|"
    r"grind|omega|native_decide|dsimp|erw|nth_rw|conv|show|skip"
    r")\b"
)

_KEYWORDS = frozenset(
    {
        "by", "where", "fun", "do", "if", "then", "else", "match", "with", "let",
        "have", "show", "calc", "sorry", "admit", "from", "using", "only", "at",
        "in", "open", "section", "namespace", "end", "variable", "variables",
        "example", "theorem", "lemma", "def", "structure", "class", "instance",
        "inductive", "mutual", "set_option", "attribute", "import", "public",
        "private", "protected", "noncomputable", "universe", "universes",
        "abbrev", "alias", "export", "extends", "deriving", "where", "Prop",
        "Type", "Sort", "true", "false", "True", "False", "And", "Or", "Not",
        "Eq", "HEq", "Iff", "Exists", "forall", "fun", "λ",
    }
)


@dataclass
class TheoremDecl:
    name: str
    kind: str  # theorem | lemma
    signature: str  # binders + type, without proof
    statement: str  # name + signature (training-facing goal text)
    proof: str
    proof_kind: str  # by | term | empty | sorry
    tactics: list[str] = field(default_factory=list)
    premises: list[str] = field(default_factory=list)
    file: str = ""
    line: int = 0
    module: str = ""


def _strip_block_comments(src: str) -> str:
    """Replace /- ... -/ with spaces (preserve newlines for line numbers)."""
    out: list[str] = []
    i = 0
    n = len(src)
    while i < n:
        if src.startswith("/-", i):
            j = src.find("-/", i + 2)
            if j < 0:
                out.append(" " * (n - i))
                break
            chunk = src[i : j + 2]
            out.append("".join("\n" if c == "\n" else " " for c in chunk))
            i = j + 2
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def _strip_line_comments(src: str) -> str:
    lines = []
    for line in src.splitlines(keepends=True):
        # crude: drop -- comments outside strings (Mathlib rarely has -- in strings for decls)
        if "--" in line:
            # keep string-ish safety: only strip if not obviously inside quotes
            in_str = False
            buf: list[str] = []
            k = 0
            while k < len(line):
                c = line[k]
                if c == '"' and not in_str:
                    in_str = True
                    buf.append(c)
                elif c == '"' and in_str:
                    in_str = False
                    buf.append(c)
                elif not in_str and line.startswith("--", k):
                    buf.append("\n" if line.endswith("\n") else "")
                    break
                else:
                    buf.append(c)
                k += 1
            lines.append("".join(buf))
        else:
            lines.append(line)
    return "".join(lines)


def _module_of(rel_path: str) -> str:
    p = rel_path.replace("\\", "/")
    if p.endswith(".lean"):
        p = p[:-5]
    return p.replace("/", ".")


def _split_proof(body: str) -> tuple[str, str, str]:
    """Split declaration body into (signature, proof, proof_kind).

    body starts after the theorem name.
    """
    # Find := that starts the proof (not in binders {x := ...} which use := rarely at top)
    # Prefer the last-ish top-level := before a by/term.
    depth = 0
    i = 0
    n = len(body)
    assign_at = -1
    while i < n:
        c = body[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        elif c == ":" and depth == 0 and not body.startswith(":=", i):
            # type colon; continue
            pass
        elif body.startswith(":=", i) and depth == 0:
            assign_at = i
            break
        i += 1
    if assign_at < 0:
        return body.strip(), "", "empty"
    signature = body[:assign_at].strip()
    proof = body[assign_at + 2 :].strip()
    if proof.startswith("by"):
        # drop leading by
        rest = proof[2:].lstrip()
        if rest.startswith("\n"):
            rest = rest[1:]
        return signature, rest if rest else "by", "by"
    if proof in ("sorry", "admit") or proof.startswith("sorry") or proof.startswith("admit"):
        return signature, proof, "sorry"
    return signature, proof, "term"


def extract_tactics(proof: str, proof_kind: str) -> list[str]:
    if proof_kind != "by" or not proof:
        if proof_kind == "term" and proof:
            return [f"exact ({proof[:120]})" if len(proof) > 120 else f"exact ({proof})"]
        return []
    tactics: list[str] = []
    for raw in proof.splitlines():
        line = raw.strip()
        if not line or line.startswith("--"):
            continue
        # multi-tactic one-liners: `rw [...]; simp`
        parts = re.split(r"\s*;\s*", line)
        for part in parts:
            part = part.strip()
            if not part or part in ("·", "{", "}", "|", "=>"):
                continue
            # skip pure binders / patterns
            if part.startswith("|") or part.startswith("·"):
                part = part.lstrip("·| ").strip()
            m = _TACTIC_LINE.match(part)
            if m:
                tactics.append(part)
            elif part and not part.startswith("intro ") and tactics:
                # continuation of previous (e.g. calc steps) — keep as fragment
                if re.match(r"^[A-Za-z_]", part):
                    tactics.append(part)
    return tactics


def extract_premises(proof: str, own_name: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for m in _IDENT.finditer(proof):
        name = m.group(1) or m.group(2)
        if not name or name == own_name or name in _KEYWORDS:
            continue
        if name[0].islower() and "." not in name and len(name) < 4:
            continue  # skip short locals a,b,n,h1
        if name.startswith("h") and name[1:].isdigit():
            continue
        if name not in seen:
            seen.add(name)
            found.append(name)
    return found


def _iter_decl_blocks(text: str) -> Iterator[tuple[int, str, str, str]]:
    """Yield (line_no_1based, kind, name, body_including_name_through_proof)."""
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        m = _DECL_START.match(lines[i])
        if not m:
            i += 1
            continue
        kind = m.group("kind")
        name = m.group("name")
        start_line = i + 1
        base_indent = len(m.group("indent"))
        # Collect until next top-level non-empty non-comment at indent <= base that starts a decl/cmd
        block = [lines[i]]
        i += 1
        while i < len(lines):
            ln = lines[i]
            stripped = ln.strip()
            if not stripped:
                block.append(ln)
                i += 1
                continue
            # blank-ish attribute lines etc.
            cur_indent = len(ln) - len(ln.lstrip(" "))
            if cur_indent <= base_indent and stripped and not stripped.startswith("--"):
                # new top-level?
                if _DECL_START.match(ln) or re.match(
                    r"^\s*(?:(?:public|private|protected|scoped|noncomputable)\s+)*"
                    r"(?:def|theorem|lemma|example|instance|class|structure|inductive|"
                    r"abbrev|alias|opaque|axiom|section|namespace|end|variable|"
                    r"open|attribute|set_option|export|universe|#)",
                    ln,
                ):
                    break
            block.append(ln)
            i += 1
        raw = "".join(block)
        # body after name
        after_name = raw.split(name, 1)[1] if name in raw else raw
        yield start_line, kind, name, after_name


def parse_lean_file(path: Path, *, rel: str | None = None) -> list[TheoremDecl]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned = _strip_line_comments(_strip_block_comments(raw))
    rel_s = rel or path.name
    module = _module_of(rel_s)
    out: list[TheoremDecl] = []
    for line_no, kind, name, body in _iter_decl_blocks(cleaned):
        signature, proof, proof_kind = _split_proof(body)
        if not signature and not proof:
            continue
        tactics = extract_tactics(proof, proof_kind)
        premises = extract_premises(proof, name)
        statement = f"{name} {signature}".strip()
        out.append(
            TheoremDecl(
                name=name,
                kind=kind,
                signature=signature,
                statement=statement,
                proof=proof.strip(),
                proof_kind=proof_kind,
                tactics=tactics,
                premises=premises,
                file=rel_s,
                line=line_no,
                module=module,
            )
        )
    return out


def parse_modules(
    mathlib_root: Path,
    modules: list[str],
    *,
    limit: int | None = None,
) -> list[TheoremDecl]:
    decls: list[TheoremDecl] = []
    for mod in modules:
        path = mathlib_root / mod
        if not path.is_file():
            continue
        for d in parse_lean_file(path, rel=mod):
            decls.append(d)
            if limit is not None and len(decls) >= limit:
                return decls
    return decls


def list_counterexample_files(mathlib_root: Path) -> list[Path]:
    d = mathlib_root / "Counterexamples"
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.lean") if p.is_file())
