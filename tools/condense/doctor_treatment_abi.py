#!/usr/bin/env python3.12
"""Doctor Prime Treatment ABI: treatments serialize to bytes you can COUNT, not prose you price.

WHY THIS EXISTS. doctor_gen3.TREATMENTS prices every treatment with a sentence
("correction indices + amortized correction codebook + one bf16 scale per protected row + 1 bit
per row protection bitmap"). A sentence is not a serializer. The promotion gate requires "exact
artifact serialized" and complete_bits/original_weight_count <= 1/1, so a treatment that cannot
emit a byte layout cannot be promoted, only estimated. This module is the layout.

THE SHAPE. One generic bit-exact codec over a DECLARATIVE layout, not nine bespoke formats:

    header (192 bits, fixed) | params block | fields | pad to byte boundary

A field is (name, width, count, ledger_component, native?). Every payload value is an unsigned
integer of a declared width; bf16 values are carried as their raw uint16 bit pattern (see
bf16_bits / from_bf16) so encode/decode is EXACT and round-trip equality is integer equality, not
float tolerance. Counts are derived from the target shape plus an ordered param tuple that is
itself serialized, so decode needs nothing but the bytes.

WHAT IS BILLED. Everything. Header bits are `metadata`. Pad bits are `alignment`. Codebooks are
`codebooks` even when "amortized" - amortization is a division you do in the ledger, not a
permission to omit. A conditional treatment bills `installed_bits` in full; `active_bits_per_token`
is reported SEPARATELY and an assert proves installed >= active so a dynamic treatment can never
bill at its average.

NO DENSE SHADOW. An artifact that stores native-precision copies of every element of the tensor it
repairs is not a treatment, it is the parent tensor wearing a header. `requires_dense_shadow` is an
explicit predicate over the native-tagged fields and is tested.

HONESTY. Nothing here measures capability, error, or recovery. It counts bits. A layout being
exact says nothing about whether the treatment helps.
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
from dataclasses import dataclass, field as dc_field
from fractions import Fraction
from typing import Any, Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(os.path.dirname(_HERE), "foundry")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from one_bit_ceiling import (  # noqa: E402
    COMPONENTS, RESERVE, CompleteByteLedger, assert_complete_bpw_le_one,
)

SCHEMA = "hawking.doctor.treatment_abi.v1"
MAGIC = 0x48445458  # "HDTX"
VERSION = 1
HEADER_BITS = 32 + 8 + 8 + 16 + 16 + 8 + 32 + 32 + 8 + 32  # 192 = 24 bytes, fixed
PARAM_WIDTH = 32

ORGANS = ("gate_proj", "up_proj", "down_proj", "router", "attn", "norm", "other")


# ── bf16 carriage (exact, integer) ────────────────────────────────────────────────────────────
def bf16_bits(x: float) -> int:
    """Raw uint16 bit pattern of bf16(x), round-to-nearest-even. Payloads carry these, not floats."""
    u = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded & 0xFFFF


def from_bf16(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (int(b) & 0xFFFF) << 16))[0]


# ── bit codec ─────────────────────────────────────────────────────────────────────────────────
class BitWriter:
    __slots__ = ("_acc", "_n")

    def __init__(self) -> None:
        self._acc = 0
        self._n = 0

    def put(self, value: int, width: int) -> None:
        v = int(value)
        if width <= 0 or v < 0 or v >> width:
            raise ValueError(f"value {v} does not fit in {width} unsigned bits")
        self._acc = (self._acc << width) | v
        self._n += width

    @property
    def bits(self) -> int:
        return self._n

    def to_bytes(self) -> bytes:
        pad = (-self._n) % 8
        return ((self._acc << pad) if pad else self._acc).to_bytes((self._n + pad) // 8, "big")


class BitReader:
    __slots__ = ("_buf", "_pos")

    def __init__(self, data: bytes) -> None:
        self._buf = int.from_bytes(data, "big")
        self._pos = len(data) * 8

    def take(self, width: int) -> int:
        if width > self._pos:
            raise ValueError("truncated artifact")
        self._pos -= width
        return (self._buf >> self._pos) & ((1 << width) - 1)


# ── layout ────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Field:
    name: str
    width: int
    count: int
    component: str            # one of one_bit_ceiling.COMPONENTS
    native: bool = False      # stores full-precision elements of the repaired tensor

    def __post_init__(self) -> None:
        assert self.component in COMPONENTS, f"{self.name}: {self.component} is not a ledger slot"
        assert self.width > 0 and self.count >= 0, f"{self.name}: bad layout"

    @property
    def bits(self) -> int:
        return self.width * self.count


def _idx_w(k: int) -> int:
    return max(1, math.ceil(math.log2(int(k))))


# Layout functions: (rows, cols, params) -> list[Field]. `params` is an ordered tuple, documented
# per treatment, and is itself serialized so decode is self-describing.
def _l_sparse_residual(rows: int, cols: int, p: tuple[int, ...]) -> list[Field]:
    """params = (n_protected_rows, dim, k)  - Doctor Gen1 sparse residual."""
    R, dim, k = p
    assert cols % dim == 0, "sparse residual: dim must divide cols"
    return [
        Field("protection_bitmap", 1, rows, "indices"),
        Field("indices", _idx_w(k), R * (cols // dim), "indices"),
        Field("codebook", 16, k * dim, "codebooks"),
        Field("row_scales", 16, R, "scales"),
    ]


def _l_protected_islands(rows: int, cols: int, p: tuple[int, ...]) -> list[Field]:
    """params = (n_protected_rows,) - native rows kept at bf16 plus their selector bitmap."""
    (R,) = p
    return [
        Field("island_bitmap", 1, rows, "protected_islands"),
        Field("native_rows", 16, R * cols, "protected_islands", native=True),
    ]


def _l_residual_codebook(rows: int, cols: int, p: tuple[int, ...]) -> list[Field]:
    """params = (dim, k) - one extra additive RVQ stage over the whole tensor."""
    dim, k = p
    assert cols % dim == 0, "residual codebook: dim must divide cols"
    return [
        Field("indices", _idx_w(k), rows * (cols // dim), "doctor"),
        Field("codebook", 16, k * dim, "codebooks"),
    ]


def _l_router_bias(rows: int, cols: int, p: tuple[int, ...]) -> list[Field]:
    """params = (n_survivors,) - one bf16 logit bias per surviving expert."""
    (K,) = p
    return [Field("survivor_bias", 16, K, "runtime_tables")]


def _l_router_lowrank(rows: int, cols: int, p: tuple[int, ...]) -> list[Field]:
    """params = (n_survivors, d_model, rank) - additive low-rank router logit correction."""
    K, d_model, r = p
    return [
        Field("factor_a", 16, K * r, "runtime_tables"),
        Field("factor_b", 16, r * d_model, "runtime_tables"),
    ]


def _l_kronecker(rows: int, cols: int, p: tuple[int, ...]) -> list[Field]:
    """params = (m1, n1, m2, n2, rank) with m1*m2 == rows, n1*n2 == cols."""
    m1, n1, m2, n2, r = p
    assert m1 * m2 == rows and n1 * n2 == cols, "kronecker factors must tile the target"
    return [
        Field("kron_a", 16, r * m1 * n1, "doctor"),
        Field("kron_b", 16, r * m2 * n2, "doctor"),
    ]


def _l_adaptive_scales(rows: int, cols: int, p: tuple[int, ...]) -> list[Field]:
    """params = (with_bias,) - refit per-row scale, optionally a per-row bias."""
    (with_bias,) = p
    out = [Field("row_scale", 16, rows, "scales")]
    if with_bias:
        out.append(Field("row_bias", 16, rows, "scales"))
    return out


def _l_conditional(rows: int, cols: int, p: tuple[int, ...]) -> list[Field]:
    """params = (n_triggers, inner_len_bytes, fire_num, fire_den) - wrapper around an inner artifact.

    The inner artifact is carried verbatim as bytes. INSTALLED bits are the whole thing; the
    trigger fires on fire_num/fire_den of tokens and that only affects active_bits_per_token.
    """
    n_trig, inner_len, _num, _den = p
    return [
        Field("trigger_table", 16, n_trig, "runtime_tables"),
        Field("inner_blob", 8, inner_len, "doctor"),
    ]


@dataclass(frozen=True)
class Spec:
    tid: int
    name: str
    params: tuple[str, ...]
    layout: Callable[[int, int, tuple[int, ...]], list[Field]]
    conditional: bool = False


SPECS: tuple[Spec, ...] = (
    Spec(1, "sparse_residual", ("n_protected_rows", "dim", "k"), _l_sparse_residual),
    Spec(2, "protected_islands", ("n_protected_rows",), _l_protected_islands),
    Spec(3, "residual_additive_codebook", ("dim", "k"), _l_residual_codebook),
    Spec(4, "router_survivor_bias", ("n_survivors",), _l_router_bias),
    Spec(5, "router_lowrank_correction", ("n_survivors", "d_model", "rank"), _l_router_lowrank),
    Spec(6, "kronecker_repair", ("m1", "n1", "m2", "n2", "rank"), _l_kronecker),
    Spec(7, "adaptive_scales_biases", ("with_bias",), _l_adaptive_scales),
    Spec(8, "conditional", ("n_triggers", "inner_len_bytes", "fire_num", "fire_den"), _l_conditional,
         conditional=True),
)
BY_NAME = {s.name: s for s in SPECS}
BY_TID = {s.tid: s for s in SPECS}


# ── artifact ──────────────────────────────────────────────────────────────────────────────────
@dataclass
class Treatment:
    """One serializable treatment instance. `values` maps field name -> list[int] of that width."""
    name: str
    layer: int
    expert: int
    organ: str
    rows: int
    cols: int
    params: tuple[int, ...]
    values: dict[str, list[int]] = dc_field(default_factory=dict)

    @property
    def spec(self) -> Spec:
        return BY_NAME[self.name]

    def fields(self) -> list[Field]:
        assert len(self.params) == len(self.spec.params), \
            f"{self.name}: params must be {self.spec.params}"
        return self.spec.layout(self.rows, self.cols, self.params)

    # ── exact bit accounting ──────────────────────────────────────────────────────────────
    def payload_bits(self) -> int:
        return 8 + PARAM_WIDTH * len(self.params) + sum(f.bits for f in self.fields())

    def total_bits(self) -> int:
        """Header + payload + pad-to-byte. This is the number the ledger bills."""
        raw = HEADER_BITS + self.payload_bits()
        return raw + (-raw) % 8

    installed_bits = total_bits  # the law's name for it: conditional or not, you pay all of it

    def active_bits_per_token(self) -> Fraction:
        """Bits the runtime must READ per token. Reported separately, NEVER billed instead."""
        if not self.spec.conditional:
            return Fraction(self.total_bits())
        _n, _len, num, den = self.params
        assert 0 <= num <= den and den > 0, "fire rate must be a proper fraction"
        return Fraction(self.total_bits() * num, den)

    # ── the no-dense-shadow predicate ─────────────────────────────────────────────────────
    def native_elements(self) -> int:
        return sum(f.count for f in self.fields() if f.native)

    def requires_dense_shadow(self) -> bool:
        """True iff the artifact carries a full-precision copy of the tensor it repairs.

        Such a thing is the parent tensor with a header on it; it may not be promoted.
        """
        return self.native_elements() >= self.rows * self.cols

    # ── codec ─────────────────────────────────────────────────────────────────────────────
    def encode(self) -> bytes:
        flags = (1 if self.spec.conditional else 0) | (2 if self.requires_dense_shadow() else 0)
        w = BitWriter()
        for v, width in ((MAGIC, 32), (VERSION, 8), (self.spec.tid, 8), (self.layer, 16),
                         (self.expert, 16), (ORGANS.index(self.organ), 8), (self.rows, 32),
                         (self.cols, 32), (flags, 8), (self.payload_bits(), 32)):
            w.put(v, width)
        w.put(len(self.params), 8)
        for p in self.params:
            w.put(p, PARAM_WIDTH)
        for f in self.fields():
            vals = self.values.get(f.name, [])
            assert len(vals) == f.count, \
                f"{self.name}.{f.name}: {len(vals)} values, layout says {f.count}"
            for v in vals:
                w.put(v, f.width)
        out = w.to_bytes()
        assert len(out) * 8 == self.total_bits(), "layout bit count disagrees with the bytes emitted"
        return out


def decode(data: bytes) -> Treatment:
    r = BitReader(data)
    assert r.take(32) == MAGIC, "not a Doctor treatment artifact"
    assert r.take(8) == VERSION, "unsupported ABI version"
    spec = BY_TID[r.take(8)]
    layer, expert, organ = r.take(16), r.take(16), ORGANS[r.take(8)]
    rows, cols, flags, payload_bits = r.take(32), r.take(32), r.take(8), r.take(32)
    params = tuple(r.take(PARAM_WIDTH) for _ in range(r.take(8)))
    t = Treatment(spec.name, layer, expert, organ, rows, cols, params)
    for f in t.fields():
        t.values[f.name] = [r.take(f.width) for _ in range(f.count)]
    assert t.payload_bits() == payload_bits, "declared payload bits disagree with the layout"
    assert flags == ((1 if spec.conditional else 0) | (2 if t.requires_dense_shadow() else 0))
    return t


def wrap_conditional(inner: Treatment, *, triggers: list[int],
                     fire_num: int, fire_den: int) -> Treatment:
    """Conditional wrapper. Installed = the whole inner artifact + trigger table, always."""
    blob = inner.encode()
    t = Treatment("conditional", inner.layer, inner.expert, inner.organ, inner.rows, inner.cols,
                  (len(triggers), len(blob), fire_num, fire_den),
                  {"trigger_table": list(triggers), "inner_blob": list(blob)})
    assert Fraction(t.installed_bits()) >= t.active_bits_per_token(), \
        "a conditional treatment may not bill at its average"
    return t


def unwrap(t: Treatment) -> Treatment:
    assert t.spec.conditional, "not a conditional artifact"
    return decode(bytes(t.values["inner_blob"]))


# ── ledger adapter ────────────────────────────────────────────────────────────────────────────
def ledger_components(treatments: list[Treatment], *, reserve_bits: int = 0,
                      note: str = "") -> CompleteByteLedger:
    """Emit the ten one_bit_ceiling components for a treatment SET. Every bit lands in a slot."""
    bits = {c: 0 for c in COMPONENTS}
    for t in treatments:
        assert not t.requires_dense_shadow(), \
            f"{t.name} L{t.layer}E{t.expert}: dense shadow, not promotable"
        raw = HEADER_BITS + t.payload_bits()
        bits["metadata"] += HEADER_BITS + 8 + PARAM_WIDTH * len(t.params)
        bits["alignment"] += (-raw) % 8
        for f in t.fields():
            bits[f.component] += f.bits
    assert sum(bits.values()) == sum(len(t.encode()) * 8 for t in treatments), \
        "component split does not sum to the serialized bit total"
    return CompleteByteLedger(**bits, metadata_alignment_reserve_bits=reserve_bits, note=note)


def report(treatments: list[Treatment], original_weight_count: int,
           *, reserve_bits: int = 0) -> dict[str, Any]:
    led = ledger_components(treatments, reserve_bits=reserve_bits, note=SCHEMA)
    receipt = assert_complete_bpw_le_one(led, original_weight_count)
    installed = sum(t.total_bits() for t in treatments)
    active = sum((t.active_bits_per_token() for t in treatments), Fraction(0))
    assert Fraction(installed) >= active, "installed must dominate active-per-token"
    return {
        "schema": SCHEMA,
        "treatments": [{"name": t.name, "layer": t.layer, "expert": t.expert, "organ": t.organ,
                        "installed_bits": t.total_bits(),
                        "active_bits_per_token": str(t.active_bits_per_token()),
                        "serialized_bytes": len(t.encode())} for t in treatments],
        "installed_bits_total": installed,
        "active_bits_per_token_total": str(active),
        "ledger": led.as_dict(original_weight_count),
        "ceiling_receipt": receipt,
    }


# ── self-check ────────────────────────────────────────────────────────────────────────────────
def _demo(rows: int = 64, cols: int = 128) -> dict[str, Any]:
    import random
    rnd = random.Random(7)

    def fill(t: Treatment) -> Treatment:
        for f in t.fields():
            t.values[f.name] = [rnd.getrandbits(f.width) for _ in range(f.count)]
        return t

    made = [
        fill(Treatment("sparse_residual", 3, 11, "down_proj", rows, cols, (8, 8, 256))),
        fill(Treatment("protected_islands", 3, 11, "gate_proj", rows, cols, (4,))),
        fill(Treatment("residual_additive_codebook", 5, 2, "up_proj", rows, cols, (8, 256))),
        fill(Treatment("router_survivor_bias", 5, 0, "router", rows, cols, (64,))),
        fill(Treatment("router_lowrank_correction", 5, 0, "router", rows, cols, (64, 2048, 4))),
        fill(Treatment("kronecker_repair", 0, 1, "gate_proj", rows, cols, (8, 16, 8, 8, 2))),
        fill(Treatment("adaptive_scales_biases", 9, 4, "down_proj", rows, cols, (1,))),
    ]

    # 1. round-trip is exact for every treatment
    for t in made:
        assert decode(t.encode()) == t, f"{t.name}: round-trip lost information"

    # 2. EXACTNESS: an INDEPENDENT closed-form bit count for sparse_residual must equal the bytes.
    sr = made[0]
    R, dim, k = sr.params
    predicted = (HEADER_BITS + 8 + PARAM_WIDTH * 3
                 + rows * 1                       # protection bitmap
                 + R * (cols // dim) * 8          # indices, ceil(log2 256) = 8
                 + k * dim * 16                   # codebook, bf16
                 + R * 16)                        # per-row bf16 scale
    predicted += (-predicted) % 8
    assert predicted == sr.total_bits() == len(sr.encode()) * 8, \
        f"predicted {predicted} != serialized {len(sr.encode()) * 8}"

    # 3. conditional: installed is the whole thing, active is separate and strictly smaller
    cond = wrap_conditional(made[2], triggers=[1, 2, 3], fire_num=1, fire_den=1000)
    assert decode(cond.encode()) == cond
    assert unwrap(cond) == made[2]
    assert cond.installed_bits() > made[2].total_bits(), "wrapper must bill its trigger table too"
    assert Fraction(cond.installed_bits()) > cond.active_bits_per_token()
    assert cond.active_bits_per_token() == Fraction(cond.total_bits(), 1000)

    # 4. NO DENSE SHADOW: protecting every row is the parent tensor with a header on it
    shadow = fill(Treatment("protected_islands", 3, 11, "gate_proj", rows, cols, (rows,)))
    assert shadow.requires_dense_shadow()
    assert not made[1].requires_dense_shadow()
    try:
        ledger_components([shadow])
    except AssertionError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a dense shadow was admitted to the ledger")

    # 5. ledger adapter round-trips through the real ceiling, and the ceiling still bites
    weights = rows * cols * 4096
    rep = report(made + [cond], weights)
    assert rep["ledger"]["legal"] is True
    try:
        report(made + [cond], 1024)  # far too few weights: must violate 1/1
    except AssertionError:
        pass
    else:  # pragma: no cover
        raise AssertionError("the ceiling did not bite")
    return rep


def _fails_when_broken() -> bool:
    """Break the layout on purpose; the self-check must fail. A check that cannot fail is decor."""
    orig = SPECS[0].layout
    try:
        object.__setattr__(SPECS[0], "layout",
                           lambda r, c, p: [f if f.name != "codebook"
                                            else Field("codebook", 15, p[2] * p[1], "codebooks")
                                            for f in orig(r, c, p)])
        BY_NAME["sparse_residual"] = SPECS[0]
        try:
            _demo()
        except AssertionError:
            return True
        return False
    finally:
        object.__setattr__(SPECS[0], "layout", orig)
        BY_NAME["sparse_residual"] = SPECS[0]


if __name__ == "__main__":
    rep = _demo()
    rep["self_check_passed"] = True
    rep["self_check_fails_when_broken"] = _fails_when_broken()
    assert _demo(), "self-check must still pass after the broken-mode probe"
    print(json.dumps(rep, indent=1))
