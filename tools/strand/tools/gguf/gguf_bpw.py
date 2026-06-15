#!/usr/bin/env python3
"""gguf_bpw.py — effective bits-per-weight of a GGUF, billed two ways.

STRAND bills bpw over the quantized PROJECTION weights only (q/k/v/o/gate/
up/down_proj 2D weight matrices) including its outlier side-channel; norms,
biases and embeddings are not in STRAND's quantized set. To compare iso-bpw
honestly we bill the GGUF the SAME way:

  proj_bpw  = (bits in the 7 projection weight matrices) / (their element count)
  file_bpw  = (total file bytes * 8) / (all weight elements)   [whole-model]

Per-tensor on-disk size = GGML_QUANT_SIZES block bytes * n_blocks.

Usage: gguf_bpw.py <in.gguf>   ->  prints JSON {proj_bpw, file_bpw, ...}
"""
import json
import math
import os
import sys

import gguf

PROJ_SUFFIXES = (
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
)


def tensor_disk_bytes(t):
    block_size, type_size = gguf.GGML_QUANT_SIZES[t.tensor_type]
    n_elem = 1
    for d in t.shape:
        n_elem *= int(d)
    return (n_elem // block_size) * type_size, n_elem


def main():
    path = sys.argv[1]
    r = gguf.GGUFReader(path)
    proj_bits = 0
    proj_elems = 0
    all_bits = 0
    all_elems = 0
    type_hist = {}
    for t in r.tensors:
        nbytes, n_elem = tensor_disk_bytes(t)
        all_bits += nbytes * 8
        all_elems += n_elem
        if t.name.endswith(PROJ_SUFFIXES):
            proj_bits += nbytes * 8
            proj_elems += n_elem
            tn = gguf.GGMLQuantizationType(t.tensor_type).name
            type_hist[tn] = type_hist.get(tn, 0) + 1

    out = {
        "file": os.path.basename(path),
        "proj_bpw": proj_bits / proj_elems if proj_elems else None,
        "proj_elems": proj_elems,
        "file_bpw": all_bits / all_elems if all_elems else None,
        "file_bytes": os.path.getsize(path),
        "proj_type_hist": type_hist,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
