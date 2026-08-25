#!/usr/bin/env python3
"""G023 step 1c: write a real catalog + segments for a slice of model #2.

compile_mix is driven by the q4 INCUMBENT MANIFEST, which is a qwen38 artifact. Model #2
has none, so the packer cannot be pointed at it however good its organ resolver is. This
drives the same containers from the specimen's own safetensors index instead, which is
the generalization that stage actually needs.

It writes a slice, not a model: a complete layer 0 minus most experts, so the container
round-trip is proved without an hour of packing. Every segment is read back and compared
byte-for-byte, and every quantized one is decoded and scored against the source.
"""
import argparse, hashlib, json, struct, sys, time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/headless"))
import whole_model_native as w
from q3_mlp_q4_attn import pack_hgravu01


def sha(b):
    return hashlib.sha256(b).hexdigest()


def decode_hgravu01(payload):
    """Independent reader: parse the JSON envelope and reconstruct."""
    assert payload[:8] == b"HGRAVU01", payload[:8]
    hlen = struct.unpack_from("<I", payload, 8)[0]
    hdr = json.loads(payload[12:12 + hlen].decode())
    body = payload[12 + hlen:]
    rows, cols = hdr["shape"]
    group, bits = hdr["group_size"], hdr["bits"]
    ngroups = (cols + group - 1) // group
    nscale = rows * ngroups * 2
    scales = np.frombuffer(body[:nscale], dtype=np.float16).astype(np.float32)
    scales = scales.reshape(rows, ngroups, 1)
    codes_raw = np.frombuffer(body[nscale:], dtype=np.uint8)
    bound = (1 << (bits - 1)) - 1
    bitstr = np.unpackbits(codes_raw, bitorder="little")
    need = rows * ngroups * group * bits
    bitstr = bitstr[:need].reshape(-1, bits)
    vals = (bitstr * (1 << np.arange(bits))).sum(axis=1).astype(np.int32)
    q = (vals - bound).reshape(rows, ngroups, group).astype(np.float32)
    deq = (q / bound) * scales
    return deq.reshape(rows, -1)[:, :cols]


# family -> concrete codec, using the vocabulary the kernel library actually holds
FAMILY_CODEC = {
    "conventional_low_bit": {"codec": "ws_rtn_q4_g64", "bits": 4, "group": 64,
                             "gemv_storage_bpw": 4.25, "container": "HGRAVU01",
                             "catalog_codec": w.CODEC_UNIFORM},
    "q2_affine": {"codec": "q2f_g64", "bits": 2, "group": 64,
                  "gemv_storage_bpw": 2.25, "container": "HGRAVF01",
                  "catalog_codec": w.CODEC_AFFINE},
}
PLANNED = {}

# The planner names organs from the architecture recognizer; the packer names them from
# its genome. `gqa_attention` and `attention_gqa` are the same organ, and the mismatch
# made the plan lookup miss silently and fall back to the qwen38 genome -- which is how
# model #2's attention got packed at q3 g128 and decoded at 0.930 cosine.
PLANNER_TO_PACKER_ROLE = {
    "gqa_attention": "attention_gqa",
    "embed": "embedding",
    "lm_head": "output",
    "moe_expert": "moe_expert",
    "moe_router": "moe_router",
    "rmsnorm": "rmsnorm",
}


def load_plan():
    kp = json.load(open(REPO / "receipts/headless/KERNEL_PLANNER_MODEL2.json"))
    out = {}
    for r in kp["organ_plan"]:
        fam = r["selected_representation"]
        if fam and fam != "leftover_f32":
            out[PLANNER_TO_PACKER_ROLE.get(r["organ"], r["organ"])] = fam
    return out


def plan_assignment(role):
    fam = PLANNED.get(role)
    if fam is None:
        return None                      # leftover_f32 or unplanned: f32 passthrough
    spec = FAMILY_CODEC.get(fam)
    if spec is None:
        raise SystemExit(f"no concrete codec for planned family {fam!r}")
    return {"role": role, "planned_family": fam, **spec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--out", default="/Users/scammermike/noetic/MODEL2_SLICE")
    ap.add_argument("--emit", default=str(REPO / "receipts/headless/MODEL2_SLICE_PACK.json"))
    a = ap.parse_args()

    global PLANNED
    PLANNED = load_plan()
    import glob
    spec = glob.glob("/Volumes/corpdrive/hawking-modellake/specimens/"
                     "Qwen--Qwen3-30B-A3B@*")[0]
    idx = json.load(open(glob.glob(f"{spec}/**/*.index.json",
                                   recursive=True)[0]))["weight_map"]

    keep = []
    for n in sorted(idx):
        if ".layers.0." not in n:
            continue
        if ".mlp.experts." in n:
            e = int(n.split(".experts.")[1].split(".")[0])
            if e >= a.experts:
                continue
        keep.append(n)

    dest = Path(a.out)
    (dest / "segments").mkdir(parents=True, exist_ok=True)
    records, roles, verified, failures = [], {}, [], []
    payload_total = 0

    for n in keep:
        with safe_open(f"{spec}/{idx[n]}", framework="pt") as f:
            t = f.get_tensor(n).to(torch.float32).numpy()
        M = t.reshape(1, -1) if t.ndim == 1 else t
        role = w.organ_role(n)
        roles[role] = roles.get(role, 0) + 1
        # Representations come from the KernelPlanner's decisions FOR MODEL #2, not from
        # whole_model_native.GENOME, which is the qwen38 genome. Using that genome packed
        # model #2's attention at q3 g128 -- qwen38's choice -- and it decoded at
        # 0.930-0.939 cosine, while the planner had selected conventional_low_bit for
        # this organ, whose competent kernel is q4. The packer must not inherit another
        # model's representation just because the organ shares a name.
        assign = plan_assignment(role) if role in PLANNED else w.assignment_for(n)

        if assign is None:                       # leftover: f32 passthrough
            payload = struct.pack("<Q", M.size) + \
                np.ascontiguousarray(M, dtype=np.float32).tobytes()
            ext, codec, bpw = "f32v2", w.CODEC_F32, 32.0
        else:
            group, bits = int(assign["group"]), int(assign["bits"])
            if M.shape[-1] % group:
                failures.append({"tensor": n, "why": f"cols {M.shape[-1]} % {group}"})
                continue
            payload = pack_hgravu01(M, bits, group)
            ext = "hgravu01"
            codec, bpw = int(assign["catalog_codec"]), float(assign["gemv_storage_bpw"])

        fn = sha(n.encode()) + f".{ext}"
        p = dest / "segments" / fn
        p.write_bytes(payload)
        payload_total += len(payload)

        # read back from disk, never from the buffer still in hand
        back = p.read_bytes()
        byte_identical = back == payload
        rec = {"name": n, "role": role, "segment": fn, "bytes": len(payload),
               "codec": codec, "gemv_storage_bpw": bpw,
               "shape": list(M.shape), "elements": int(M.size),
               "sha256": sha(payload), "byte_identical_on_readback": byte_identical}
        if not byte_identical:
            failures.append({"tensor": n, "why": "segment did not read back identical"})
        if ext == "hgravu01":
            deq = decode_hgravu01(back)
            cos = float((M.ravel() @ deq.ravel()) /
                        (np.linalg.norm(M) * np.linalg.norm(deq) + 1e-12))
            rec["decode_cosine"] = round(cos, 6)
            verified.append(cos)
            if cos < 0.95:
                failures.append({"tensor": n, "why": f"decode cosine {cos:.4f} < 0.95"})
        records.append(rec)

    catalog = {"format": "hq38m20", "specimen": "Qwen/Qwen3-30B-A3B",
               "slice": f"layer 0, first {a.experts} experts",
               "n_tensors": len(records), "tensors": records}
    (dest / "catalog.hq38m20").write_text(json.dumps(catalog, indent=1))

    out = {
        "schema": "hawking.headless.model2_slice_pack.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/headless/moe_slice_pack.py",
        "obligation": "G023 step 1c — a real catalog and segments for model #2",
        "hand_authored": False,
        "artifact_root": str(dest),
        "driven_by": "the specimen's own safetensors index, NOT the q4 incumbent "
                     "manifest that compile_mix requires and model #2 does not have",
        "representations_from": "receipts/headless/KERNEL_PLANNER_MODEL2.json — the "
                                "planner's decisions for THIS model, not qwen38's genome",
        "planned": PLANNED,
        "n_tensors": len(records), "roles": roles,
        "payload_bytes": payload_total,
        "n_segments_written": len(records),
        "all_segments_read_back_identical": all(r["byte_identical_on_readback"]
                                                for r in records),
        "decode": {"n_quantized": len(verified),
                   "median_cosine": round(float(np.median(verified)), 6) if verified else None,
                   "worst_cosine": round(float(min(verified)), 6) if verified else None,
                   "decoder": "an INDEPENDENT reader in this file, not the packer's own "
                              "round-trip helper"},
        "failures": failures,
        "scope": {"is_a_slice_not_a_model": True,
                  "layers_packed": 1, "layers_in_model": 48,
                  "experts_packed": a.experts, "experts_per_layer": 128},
        "what_this_does_not_show": "that model #2 would be coherent. Local adequacy does "
                                   "not compose -- the ternary bracket failed at 1.85 bpw "
                                   "with every organ locally validated.",
    }
    out["pass"] = bool(not failures and records)
    Path(a.emit).write_text(json.dumps(out, indent=1))
    print(f"  packed {len(records)} tensors from layer 0, roles={roles}")
    print(f"  payload {payload_total:,} bytes, all read back identical: "
          f"{out['all_segments_read_back_identical']}")
    if verified:
        print(f"  independent decode: median cos {out['decode']['median_cosine']}, "
              f"worst {out['decode']['worst_cosine']} over {len(verified)} segments")
    print(f"  failures: {len(failures)}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
