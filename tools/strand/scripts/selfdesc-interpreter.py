#!/usr/bin/env python3
"""selfdesc-interpreter — the proof of the .strand self-description property.

A from-scratch, stdlib-only interpreter that decodes a `.strand` v2 archive's
bulk Q12 plane using ONLY the file: the STR2 header/descriptors, the SDSC
section (frozen LUTs + the decode arithmetic as straight-line integer
programs + layout constants), the payload/side-info regions, and the OUTL
section. It carries NO knowledge of the Rust implementation — every numeric
semantic (state advance, sub-scale, affine offset, reconstruction) is executed
from the programs embedded in the SDSC section, and the quantile LUT is the
raw Q12 table embedded there too.

The gate (the self-description theorem, demonstrated): for each decoded
tensor this script computes the SPV3 leaf hashes / tensor root over its Q12
stream (spec: docs/STRAND-v3-provenance-spec.md, the published canonical
serialization) and compares them against (a) the SPRV-stored roots and
self-test vectors inside the same file and (b) attest-strand's printed roots.
SHA-256 equality == byte equality of the decoded Q12 stream.

Usage:
    python3 tools/selfdesc-interpreter.py <archive.strand> [--tensors i,j,k] [--all]

Default tensor selection mirrors attest-strand's spot check: first / middle /
last. Exit status 0 = every comparison passed; 1 = any mismatch.

Honest boundary (SDSC v1): this decodes the deterministic integer Q12 plane —
the object the provenance roots hash. The weight-space steps above it
(Q12 * 2^-12 -> f32, inverse row-aware RHT from the descriptor seed, outlier
REPLACEMENT from the OUTL section) are specified in
docs/STRAND-self-describing-spec.md; the row-aware RHT is the declared v2 gap
(SDSC constant RHT_RULE = 0), so this interpreter reports the OUTL channel but
does not produce weight-space f32 tensors.
"""

import hashlib
import struct
import sys

PAGE = 4096

# ---------------------------------------------------------------- v2 header

def parse_v2(buf):
    if buf[:4] != b"STR2":
        raise SystemExit("not a .strand v2 archive (bad magic)")
    ver, = struct.unpack_from("<I", buf, 4)
    if ver != 2:
        raise SystemExit(f"unsupported v2 version {ver}")
    n_tensors, = struct.unpack_from("<I", buf, 12)
    p, descs = 56, []
    for _ in range(n_tensors):
        (nl,) = struct.unpack_from("<I", buf, p); p += 4
        name = buf[p:p + nl].decode(); p += nl
        (nd,) = struct.unpack_from("<I", buf, p); p += 4
        shape = struct.unpack_from(f"<{nd}Q", buf, p); p += 8 * nd
        (seed,) = struct.unpack_from("<Q", buf, p); p += 8
        l, k, d, f = buf[p], buf[p + 1], buf[p + 2], buf[p + 3]; p += 4
        (block_len,) = struct.unpack_from("<I", buf, p); p += 4
        total, n_blocks = struct.unpack_from("<QQ", buf, p); p += 16
        p += 8  # reserved
        toff, poff, pbytes, soff, sbytes = struct.unpack_from("<5Q", buf, p); p += 40
        descs.append(dict(name=name, shape=shape, seed=seed, l=l, k=k, d=d,
                          has_rht=bool(f & 1), tail=bool(f & 2), affine=bool(f & 4),
                          block_len=block_len, total=total, n_blocks=n_blocks,
                          table_off=toff, payload_off=poff, payload_bytes=pbytes,
                          sideinfo_off=soff, sideinfo_bytes=sbytes))
    return descs

# ------------------------------------------------- EOF trailer chain (16 B)

def chain_sections(buf):
    """Walk the EOF trailer chain; return {magic: (offset, nbytes, trailer_end)}."""
    out, end = {}, len(buf)
    for _ in range(8):
        if end < 16:
            break
        off, nbytes = struct.unpack_from("<QI", buf, end - 16)
        magic = buf[end - 4:end]
        if magic not in (b"SDSC", b"OUTL", b"SPRV") or off >= end or off % PAGE:
            break
        out[magic.decode()] = (off, nbytes, end)
        end = off
    return out

# ------------------------------------------------------------- SDSC section

CONST = {1: "SCALE_SHIFT", 2: "SUB_SCALE_SHIFT", 3: "SUB_BLOCK",
         4: "SUBSCALE_CODE_BITS", 5: "QUANTILE_SHIFT", 6: "PAYLOAD_BIT_ORDER",
         7: "SIDEINFO_LAYOUT", 8: "TAILBITE_RULE", 9: "OUTL_PATCH_RULE",
         10: "RHT_RULE"}
EXPR = {1: "ADVANCE", 2: "EFF_SCALE", 3: "OFFSET", 4: "RECON"}


def w64(x):
    return ((x + (1 << 63)) & ((1 << 64) - 1)) - (1 << 63)


def w32(x):
    return ((x + (1 << 31)) & ((1 << 32) - 1)) - (1 << 31)


def tdiv(a, b):
    if b == 0:
        return 0
    q = abs(a) // abs(b)
    return w64(q if (a < 0) == (b < 0) else -q)


def compile_expr(prog, n_slots):
    """Symbolically execute the SDSC stack program into one python lambda.

    The op set is the SDSC v1 algebra: wrapping two's-complement i64,
    straight-line, no control flow. Binary ops pop rhs then lhs.
    """
    st, p = [], 0
    while True:
        if p >= len(prog):
            raise SystemExit("sdsc: program missing END")
        op = prog[p]; p += 1
        if op == 0x00:  # END
            if p != len(prog) or len(st) != 1:
                raise SystemExit("sdsc: malformed program")
            break
        elif op == 0x01:  # IMM i64
            st.append(repr(struct.unpack_from("<q", prog, p)[0])); p += 8
        elif op == 0x02:  # LOAD slot
            s = prog[p]; p += 1
            if s >= n_slots:
                raise SystemExit("sdsc: LOAD slot out of range")
            st.append(f"s{s}")
        elif op in (0x10, 0x11, 0x12):  # ADD SUB MUL (wrapping)
            b, a = st.pop(), st.pop()
            st.append(f"w64(({a}){'+-*'[op - 0x10]}({b}))")
        elif op == 0x13:  # TDIV
            b, a = st.pop(), st.pop()
            st.append(f"tdiv({a},{b})")
        elif op == 0x14:  # NEG
            st.append(f"w64(-({st.pop()}))")
        elif op == 0x15:  # ABS
            st.append(f"w64(abs({st.pop()}))")
        elif op == 0x16:  # WRAP32
            st.append(f"w32({st.pop()})")
        elif op == 0x18:  # SHL
            b, a = st.pop(), st.pop()
            st.append(f"w64(({a})<<(({b})&63))")
        elif op == 0x19:  # ASR (python >> on ints IS arithmetic)
            b, a = st.pop(), st.pop()
            st.append(f"(({a})>>(({b})&63))")
        elif op in (0x1C, 0x1D, 0x1E):  # AND OR XOR
            b, a = st.pop(), st.pop()
            st.append(f"(({a}){'&|^'[op - 0x1C]}({b}))")
        elif op == 0x20:  # CLAMP (pops hi, lo, x)
            hi, lo, x = st.pop(), st.pop(), st.pop()
            st.append(f"min(max({x},{lo}),{hi})")
        else:
            raise SystemExit(f"sdsc: unknown opcode {op:#04x}")
    args = ",".join(f"s{i}" for i in range(n_slots))
    return eval(f"lambda {args}: {st[0]}", {"w64": w64, "w32": w32, "tdiv": tdiv,
                                            "min": min, "max": max, "abs": abs})


def parse_sdsc(buf, off, nbytes):
    s = buf[off:off + nbytes]
    if s[:4] != b"SDSC":
        raise SystemExit("sdsc: bad header magic")
    ver, flags, n_consts, n_exprs, n_luts = struct.unpack_from("<5I", s, 4)
    if ver != 1 or flags != 0:
        raise SystemExit(f"sdsc: unsupported version {ver} / flags {flags}")
    p, consts, exprs, luts = 48, {}, {}, {}
    for _ in range(n_consts):
        cid, v = struct.unpack_from("<Iq", s, p); p += 12
        consts[cid] = v
    for _ in range(n_exprs):
        eid, n_slots, plen = struct.unpack_from("<3I", s, p); p += 12
        exprs[eid] = compile_expr(s[p:p + plen], n_slots); p += plen
    for _ in range(n_luts):
        l, d, n_entries, _r = struct.unpack_from("<4I", s, p); p += 16
        if n_entries != (1 << l) * d:
            raise SystemExit("sdsc: LUT entry count mismatch")
        luts[(l, d)] = list(struct.unpack_from(f"<{n_entries}i", s, p))
        p += 4 * n_entries
    if p != nbytes:
        raise SystemExit("sdsc: trailing bytes after last LUT")
    return consts, exprs, luts

# ------------------------------------------------------------- OUTL section

def parse_outl(buf, off, nbytes, n_tensors):
    s = buf[off:off + nbytes]
    if s[:4] != b"OUTL" or struct.unpack_from("<I", s, 4)[0] != 1:
        raise SystemExit("outl: bad header")
    p, out = 32, []
    for _ in range(n_tensors):
        count, omax_bits, idx_bits, val_bits, _r = struct.unpack_from("<QIIII", s, p)
        p += 24
        if count:
            packed = (count * (idx_bits + val_bits) + 7) // 8
            p += packed
        out.append((count, omax_bits, idx_bits, val_bits))
    return out

# ------------------------------------------------------------- SPRV section

def parse_sprv(buf, off, nbytes, n_tensors):
    s = buf[off:off + nbytes]
    if s[:4] != b"SPRV":
        raise SystemExit("sprv: bad header magic")
    ver, nt, flags = struct.unpack_from("<3I", s, 4)
    if ver != 2 or nt != n_tensors:
        raise SystemExit("sprv: unsupported version / tensor count")
    model_root, p, recs = s[16:48], 64, []
    has_leaves = bool(flags & 1)
    for _ in range(n_tensors):
        root, digest = s[p:p + 32], s[p + 32:p + 64]; p += 64
        n_blocks, n_vectors, _r = struct.unpack_from("<QII", s, p); p += 16
        vectors = []
        for _ in range(n_vectors):
            (bi,) = struct.unpack_from("<Q", s, p)
            vectors.append((bi, s[p + 8:p + 40])); p += 40
        if has_leaves:
            p += 32 * n_blocks
        recs.append((root, digest, n_blocks, vectors))
    return model_root, recs

# ----------------------------------------------------------- the Q12 decode

def decode_tensor_q12_leaves(buf, d, consts, exprs, luts):
    """Decode one tensor blockwise from ONLY the file's bytes + SDSC semantics.

    Returns the SPV3 leaf list (one sha256 per 256-weight block). The decode
    loop's structure (block iteration, per-sub-block side-info, tail-biting
    prescan) is the published spec; every arithmetic step runs the SDSC
    programs, and every level comes from the SDSC-embedded LUT.
    """
    if consts[6] != 0 or consts[7] != 1 or consts[8] != 1:
        raise SystemExit("sdsc: unknown layout/rule ids — refusing to guess")
    if d["d"] > 1:
        raise SystemExit("sdsc v1: vector-trellis tensors not covered")
    lut = luts[(d["l"], 1)]
    sub_block, code_bits = consts[3], consts[4]
    adv, eff_e, off_e, rec_e = exprs[1], exprs[2], exprs[3], exprs[4]
    k, l = d["k"], d["l"]
    mask, kmask, ki = (1 << l) - 1, (1 << k) - 1, k

    # Whole payload as one int: stream bit j == that int's bit j (LSB-first).
    payload = int.from_bytes(buf[d["payload_off"]:d["payload_off"] + d["payload_bytes"]],
                             "little")
    n_blocks, total, bl = d["n_blocks"], d["total"], d["block_len"]
    # Offset table: (bit_offset u64, init_state u32, scale_q i32) per block.
    table = [struct.unpack_from("<QIi", buf, d["table_off"] + 16 * b)
             for b in range(n_blocks)]

    # Side-info layout 1: per-block sub-scale code bytes back-to-back, then
    # (if affine) 4-aligned per-block min_base_q i32s, then mins code bytes.
    soff = d["sideinfo_off"]
    ss_sizes, ss_starts, c = [], [], soff
    for b in range(n_blocks):
        n = bl if b + 1 < n_blocks else total - (n_blocks - 1) * bl
        nb = (((n + sub_block - 1) // sub_block) * code_bits + 7) // 8
        ss_starts.append(c); ss_sizes.append(nb); c += nb
    mins_base = (c + 3) & ~3 if d["affine"] and soff else None

    leaves, cmask = [], (1 << code_bits) - 1
    mins_codes_base = mins_base + 4 * n_blocks if mins_base is not None else None
    mc = mins_codes_base
    for b in range(n_blocks):
        n = bl if b + 1 < n_blocks else total - (n_blocks - 1) * bl
        n_sub = (n + sub_block - 1) // sub_block
        bit0, init_state, scale_q = table[b]
        sw = int.from_bytes(buf[ss_starts[b]:ss_starts[b] + ss_sizes[b]], "little")
        eff = [eff_e(scale_q, (sw >> (code_bits * i)) & cmask) for i in range(n_sub)]
        if mins_base is not None:
            (mbq,) = struct.unpack_from("<i", buf, mins_base + 4 * b)
            mw = int.from_bytes(buf[mc:mc + ss_sizes[b]], "little")
            mc += ss_sizes[b]
            offs = [off_e(mbq, (mw >> (code_bits * i)) & cmask) for i in range(n_sub)]
        else:
            offs = [0] * n_sub

        syms = [(payload >> (bit0 + i * ki)) & kmask for i in range(n)]
        if d["tail"] and n * k >= l:  # TAILBITE_RULE 1: prescan from state 0
            state = 0
            for sym in syms:
                state = adv(state, sym, ki, mask)
        else:
            state = init_state & mask
        q12 = []
        for i, sym in enumerate(syms):
            state = adv(state, sym, ki, mask)
            sb = i // sub_block
            q12.append(rec_e(eff[sb], lut[state], offs[sb]))
        # SPV3 block leaf (provenance spec §2.1).
        leaves.append(hashlib.sha256(
            b"SPV3.BLK" + struct.pack("<QI", b, n) + struct.pack(f"<{n}i", *q12)
        ).digest())
    return leaves


def tensor_root(leaves):
    h = hashlib.sha256(b"SPV3.TNS" + struct.pack("<Q", len(leaves)))
    for leaf in leaves:
        h.update(leaf)
    return h.digest()

# -------------------------------------------------------------------- main

def main():
    args = [a for a in sys.argv[1:]]
    if not args:
        raise SystemExit(__doc__)
    path, sel, decode_all = args[0], None, False
    i = 1
    while i < len(args):
        if args[i] == "--tensors":
            i += 1; sel = [int(x) for x in args[i].split(",")]
        elif args[i] == "--all":
            decode_all = True
        i += 1

    buf = open(path, "rb").read()
    descs = parse_v2(buf)
    n = len(descs)
    sections = chain_sections(buf)
    if "SDSC" not in sections:
        raise SystemExit("no SDSC section — the file does not carry its own decoder")
    consts, exprs, luts = parse_sdsc(buf, *sections["SDSC"][:2])
    print(f"archive  {path}: {n} tensors")
    print(f"SDSC     v1: consts={ {CONST.get(k, k): v for k, v in consts.items()} }")
    print(f"         luts={[(l, d2, len(t)) for (l, d2), t in luts.items()]} "
          f"exprs={[EXPR.get(e, e) for e in exprs]}")
    if consts.get(10, 0) == 0:
        print("         RHT_RULE=0: weight-space RHT inverse is NOT self-described "
              "(declared v2 gap; this run proves the Q12 plane)")
    if "OUTL" in sections:
        chans = parse_outl(buf, *sections["OUTL"][:2], n)
        with_ch = sum(1 for c in chans if c[0])
        print(f"OUTL     {with_ch}/{n} tensors carry a channel, "
              f"{sum(c[0] for c in chans)} entries (patch rule id "
              f"{consts.get(9)}: weight-space replacement — see spec)")
    sprv = None
    if "SPRV" in sections:
        model_root, sprv = parse_sprv(buf, *sections["SPRV"][:2], n)
        print(f"SPRV     stored model_root={model_root.hex()}")

    idxs = sel if sel is not None else (list(range(n)) if decode_all
                                        else sorted({0, n // 2, n - 1}))
    failures = 0
    for ti in idxs:
        d = descs[ti]
        leaves = decode_tensor_q12_leaves(buf, d, consts, exprs, luts)
        root = tensor_root(leaves)
        line = f"tensor_root {root.hex()}  {d['name']}  (n={d['total']}, {len(leaves)} blocks)"
        if sprv is not None:
            srec = sprv[ti]
            ok_root = srec[0] == root
            ok_vecs = all(leaves[bi] == h for bi, h in srec[3])
            verdict = ("PASS" if ok_root and ok_vecs else "FAIL")
            line += (f"  vs SPRV: root {'==' if ok_root else '!='} stored, "
                     f"{len(srec[3])} self-test vectors "
                     f"{'match' if ok_vecs else 'MISMATCH'}  [{verdict}]")
            if not (ok_root and ok_vecs):
                failures += 1
        print(line)

    if sprv is not None:
        print("VERDICT  " + ("PASS — interpreter Q12 byte-equals the committed plane "
                             "(SHA-256 equality over the decoded stream)"
                             if failures == 0 else f"FAIL — {failures} tensor(s) mismatched"))
    else:
        print("VERDICT  roots computed (no SPRV in file — compare against "
              "attest-strand --roots externally)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
