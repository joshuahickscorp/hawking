#!/usr/bin/env python3
# =============================================================================
# llama_logits_to_npy.py — convert a llama.cpp `--save-all-logits` /
#   `--kl-divergence-base` dump into a (T, V) float32 .npy for the small-draft
#   accept oracle (tools/bench/draft_accept_oracle.py --logits T.npy D.npy).
# =============================================================================
#
# WHY THIS EXISTS
#   `dismantle generate` cannot export logits (no flag, no .npy writer). The
#   only full-vocab logit exporter on this machine is
#       llama-perplexity -m M -f corpus.txt --save-all-logits OUT.bin
#   which writes the next-token logits for every scored position of the corpus
#   in one teacher-forced pass. The accept oracle wants those as a (T, V)
#   float32 array. This converter bridges the two.
#
# THE BINARY FORMAT  (llama.cpp examples/perplexity/perplexity.cpp,
#                     kl_divergence_result / the `--save-all-logits` writer)
#   The file written by `kl_divergence_base` / `--save-all-logits` begins with a
#   small header. Across llama.cpp versions the stable, documented prefix is:
#       int32  magic     (== 0x4c4c4b4b, the ASCII bytes "KKLL" little-endian,
#                          i.e. 'LLAMA_KV_LOGITS' marker used by the kl path)
#       int32  version
#       int32  n_vocab
#       int32  n_chunk   (number of context windows scored)
#   followed, per evaluated token, by `n_vocab` float32 logits (the modern
#   `--save-all-logits` writes RAW logits for every token; the older
#   kl-divergence-base wrote a compressed per-token record of
#   {max_logit (f32), [n_vocab] f16 deltas} — both layouts are handled below,
#   selected by the on-disk byte count).
#
#   IMPORTANT — VERSION FRAGILITY: the precise header magic/version differs
#   between llama.cpp builds. This converter:
#     (1) reads the 16-byte header,
#     (2) infers the record layout from the file size and n_vocab,
#     (3) FAILS LOUDLY with the observed header bytes + a size breakdown if it
#         cannot reconcile them — so the operator can confirm the llama.cpp
#         version and adjust, rather than silently emitting garbage logits.
#   Do NOT trust a converted array whose self-check (printed below) looks off.
#
# USAGE
#   tools/bench/llama_logits_to_npy.py IN.bin OUT.npy
#       [--n-vocab N]      # override header n_vocab (if header is non-standard)
#       [--header-bytes B] # override header size (default 16)
#       [--max-tokens T]   # cap rows (memory; default all)
#       [--layout auto|raw|kl]  # force the per-token record layout
#       [--dtype f32|f16]  # raw-logit element dtype on disk (default f32)
#
# OUTPUT
#   OUT.npy : float32 ndarray, shape (T, V). T = number of scored tokens, V =
#   n_vocab. This is exactly what draft_accept_oracle.py --logits expects; it
#   asserts target/draft shapes match, so run the SAME corpus + -c for both.
# =============================================================================
import argparse
import struct
import sys

import numpy as np

HEADER_BYTES = 16  # magic(i32) + version(i32) + n_vocab(i32) + n_chunk(i32)


def die(msg: str) -> "None":
    print(f"llama_logits_to_npy: error: {msg}", file=sys.stderr)
    sys.exit(2)


def read_header(raw: bytes, header_bytes: int):
    if len(raw) < header_bytes:
        die(f"file too small ({len(raw)} B) to contain a {header_bytes}-B header.")
    magic, version, n_vocab, n_chunk = struct.unpack_from("<iiii", raw, 0)
    return magic, version, n_vocab, n_chunk


def main() -> "None":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp", help="llama.cpp --save-all-logits .bin")
    ap.add_argument("out", help="output .npy ((T,V) float32)")
    ap.add_argument("--n-vocab", type=int, default=None,
                    help="override header n_vocab")
    ap.add_argument("--header-bytes", type=int, default=HEADER_BYTES,
                    help=f"header size in bytes (default {HEADER_BYTES})")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap number of rows (memory)")
    ap.add_argument("--layout", choices=["auto", "raw", "kl"], default="auto",
                    help="per-token record layout (auto = infer from size)")
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32",
                    help="raw-logit element dtype on disk")
    args = ap.parse_args()

    try:
        with open(args.inp, "rb") as fh:
            raw = fh.read()
    except OSError as ex:
        die(f"cannot read input '{args.inp}': {ex}. Did llama-perplexity "
            f"--save-all-logits run and write it?")
    total = len(raw)
    if total == 0:
        die(f"input '{args.inp}' is empty — the --save-all-logits run produced "
            f"no logits (check the perplexity log for OOM / a 0-chunk corpus).")

    magic, version, hdr_nv, n_chunk = read_header(raw, args.header_bytes)
    n_vocab = args.n_vocab if args.n_vocab is not None else hdr_nv
    print(f"  header: magic=0x{magic & 0xffffffff:08x} version={version} "
          f"n_vocab(header)={hdr_nv} n_chunk={n_chunk}  file={total} B")
    if n_vocab <= 0 or n_vocab > 1_000_000:
        die(f"implausible n_vocab={n_vocab} from header — your llama.cpp build's "
            f"--save-all-logits header differs. Re-run with --n-vocab <V> "
            f"(Qwen2.5 full vocab = 151936; dismantle decodes a 32000 pruned "
            f"slice but llama.cpp dumps the FULL vocab). Header bytes: "
            f"{raw[:args.header_bytes].hex()}")

    body = total - args.header_bytes
    elem = 4 if args.dtype == "f32" else 2

    # Infer layout from the body size.
    #   raw : per token = n_vocab * elem bytes.
    #   kl  : per token = 4 (max f32) + n_vocab * 2 (f16 deltas)  [legacy].
    raw_rec = n_vocab * elem
    kl_rec = 4 + n_vocab * 2
    layout = args.layout
    if layout == "auto":
        if raw_rec > 0 and body % raw_rec == 0:
            layout = "raw"
        elif kl_rec > 0 and body % kl_rec == 0:
            layout = "kl"
        else:
            die(f"cannot reconcile body={body} B with n_vocab={n_vocab}: "
                f"neither raw-record ({raw_rec} B, rem {body % raw_rec}) nor "
                f"kl-record ({kl_rec} B, rem {body % kl_rec}) divides it evenly. "
                f"Wrong --n-vocab, --header-bytes, or a different llama.cpp "
                f"format. Header bytes: {raw[:args.header_bytes].hex()}")

    if layout == "raw":
        n_tok = body // raw_rec
        print(f"  layout=raw  rec={raw_rec} B/token ({args.dtype})  T={n_tok}")
        np_dt = np.float32 if args.dtype == "f32" else np.float16
        arr = np.frombuffer(raw, dtype=np_dt, count=n_tok * n_vocab,
                            offset=args.header_bytes)
        arr = arr.reshape(n_tok, n_vocab).astype(np.float32, copy=False)
    else:  # kl legacy: reconstruct logits = max - delta (deltas are >=0 f16)
        n_tok = body // kl_rec
        print(f"  layout=kl(legacy)  rec={kl_rec} B/token  T={n_tok}  "
              f"(reconstructing logits = max_logit - f16_delta)")
        out = np.empty((n_tok, n_vocab), dtype=np.float32)
        off = args.header_bytes
        for t in range(n_tok):
            (maxl,) = struct.unpack_from("<f", raw, off)
            off += 4
            deltas = np.frombuffer(raw, dtype=np.float16, count=n_vocab, offset=off)
            off += n_vocab * 2
            out[t] = maxl - deltas.astype(np.float32)
        arr = out

    if args.max_tokens is not None and arr.shape[0] > args.max_tokens:
        arr = arr[: args.max_tokens]
        print(f"  capped to --max-tokens={args.max_tokens}")

    # Self-check: argmax distribution sanity (a real logit stream is not all the
    # same token; a corrupt parse usually collapses argmax to 0 or a constant).
    am = arr.argmax(axis=1)
    uniq = int(np.unique(am).size)
    print(f"  self-check: shape={arr.shape} dtype={arr.dtype} "
          f"distinct-argmax-tokens={uniq}/{arr.shape[0]} "
          f"logit[min,max]=[{float(arr.min()):.2f},{float(arr.max()):.2f}]")
    if uniq <= 1 and arr.shape[0] > 4:
        print("  WARN: argmax collapsed to a single token — likely a FORMAT "
              "MISMATCH (wrong n_vocab/header/layout), NOT a real stream. "
              "Verify your llama.cpp version and the header bytes above before "
              "trusting tau.", file=sys.stderr)

    np.save(args.out, arr)
    print(f"  wrote {args.out}  ({arr.shape[0]} x {arr.shape[1]} float32, "
          f"{arr.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
