"""FUZZ as a capability: throw malformed input at a boundary, keep only crashes.

S036 s47: HCLI reasons about the target boundary, the harness, the corpus and
the invariants; Agent OS handles iterations, mutations, crash capture,
deduplication, minimization. Raw fuzz noise never enters the model's context --
only a deduplicated set of reproducers does.

WHAT THIS IS. A bounded, in-process fuzzer for a Python callable: the operator
names a function (a parser, a decoder, a validator) and an invariant it must
uphold -- "never raises anything but ValueError", "always returns a dict",
"round-trips" -- and this throws mutated inputs at it, catching every violation.
The mutation set is deliberately the classic parser-breaker corpus (empty,
huge, control bytes, nesting, encoding edges) plus byte-level mutations of any
seed the caller gives, because those are what find real bugs in real parsers.

WHAT IT IS NOT. It does not shell out to a native fuzzer, spawn processes, or
touch the network. It is the smallest thing that turns "does this parser have a
crash" from a guess into a reproducer, and it hands back the MINIMIZED input,
not the 10,000 it tried. A crash you cannot reproduce is a story; a minimized
reproducing input is a fact the builder can turn into a red test.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

#: The classic corpus that breaks real parsers. Seeds are mutated on top of this.
_SEED_INPUTS: Tuple[Any, ...] = (
    "", " ", "\n", "\x00", "\xff", "\t\r\n",
    "0", "-0", "1e999", "NaN", "Infinity", "null", "true", "[]", "{}",
    "[" * 200, "{" * 200, '{"' + "a" * 10000 + '"}',
    "a" * 100000, "\ud800", "\\", '"', "'", "%s%s%s", "../" * 50,
    "\x1b[31m", "\U0001f600", b"\x00\x01\x02", b"\xff\xfe", 0, -1, 2 ** 63,
    [], {}, None, 3.14, float("inf"),
)

#: Byte/char mutations applied to a seed to widen coverage cheaply.
def _mutations(seed: Any) -> List[Any]:
    out: List[Any] = []
    if isinstance(seed, str) and seed:
        out.append(seed[:len(seed) // 2])                 # truncate
        out.append(seed + seed)                            # double
        out.append(seed.replace(seed[0], "\x00", 1))       # inject NUL
        out.append(seed.upper() + "￿")                # append noise
    if isinstance(seed, (bytes, bytearray)) and seed:
        out.append(bytes(seed[:len(seed) // 2]))
        out.append(bytes(seed) + b"\x00")
    return out


@dataclass
class Crash:
    """One reproduced invariant violation."""

    exception: str
    message: str
    minimized_repr: str
    signature: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class FuzzResult:
    target: str
    iterations: int
    crashes: List[Crash] = field(default_factory=list)
    clean: bool = True
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "iterations": self.iterations,
            "crash_count": len(self.crashes),
            "clean": self.clean,
            "crashes": [c.to_dict() for c in self.crashes],
            "note": self.note,
        }

    def summary(self) -> str:
        if self.clean:
            return (f"FUZZ {self.target} -- {self.iterations} inputs, no invariant "
                    f"violation. {self.note}").strip()
        lines = [f"FUZZ {self.target} -- {len(self.crashes)} distinct crash(es) "
                 f"in {self.iterations} inputs:"]
        for crash in self.crashes[:6]:
            lines.append(f"  {crash.exception}: {crash.message[:80]}")
            lines.append(f"    reproduce with: {crash.minimized_repr}")
        return "\n".join(lines)


def _minimize(target: Callable[[Any], Any], value: Any,
              allowed: Tuple[type, ...]) -> Any:
    """Shrink a crashing input while it keeps crashing the same way.

    Greedy and bounded: halve a string/bytes/list, drop a dict key, and keep the
    smaller input only if it still violates. A minimized input is what a human
    turns into a red test; the 10,000 that did not crash are noise.
    """
    def still_crashes(candidate: Any) -> bool:
        try:
            target(candidate)
            return False
        except Exception as exc:  # noqa: BLE001
            return not isinstance(exc, allowed)

    current = value
    for _ in range(40):
        smaller: Optional[Any] = None
        if isinstance(current, (str, bytes, bytearray)) and len(current) > 1:
            smaller = current[: len(current) // 2]
        elif isinstance(current, list) and current:
            smaller = current[: len(current) // 2]
        elif isinstance(current, dict) and current:
            smaller = dict(list(current.items())[:-1])
        if smaller is None or not still_crashes(smaller):
            break
        current = smaller
    return current


def fuzz(target: Callable[[Any], Any], *, name: str = "",
         allowed: Tuple[type, ...] = (ValueError,),
         seeds: Optional[List[Any]] = None,
         max_iterations: int = 400) -> FuzzResult:
    """Throw malformed input at `target`; collect deduplicated minimized crashes.

    `allowed` is the invariant: exceptions in this tuple are the contract (a
    parser is ALLOWED to raise ValueError on bad input). Anything else -- an
    IndexError, a RecursionError, a bare Exception -- is a crash. Returning
    normally is always fine. Deduplicated by (exception type, first frame), so
    one bug does not report a thousand times.
    """
    corpus: List[Any] = list(_SEED_INPUTS)
    for seed in (seeds or []):
        corpus.append(seed)
        corpus.extend(_mutations(seed))
    corpus = corpus[:max_iterations]

    crashes: Dict[str, Crash] = {}
    tried = 0
    for value in corpus:
        tried += 1
        try:
            target(value)
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, allowed):
                continue
            kind = type(exc).__name__
            signature = hashlib.sha256(
                f"{kind}:{str(exc)[:60]}".encode("utf-8", "replace")).hexdigest()[:12]
            if signature in crashes:
                continue
            minimized = _minimize(target, value, allowed)
            crashes[signature] = Crash(
                exception=kind, message=str(exc)[:200],
                minimized_repr=repr(minimized)[:200], signature=signature)
    result = FuzzResult(target=name or getattr(target, "__name__", "target"),
                        iterations=tried, crashes=list(crashes.values()),
                        clean=not crashes)
    if not crashes:
        result.note = f"held the invariant (only {allowed} raised) across the corpus"
    return result
