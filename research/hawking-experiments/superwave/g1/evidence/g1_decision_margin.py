#!/usr/bin/env python3
"""Decision-margin measurement for Qwen3.8-27B. CPU only. No GPU. No model forward."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import struct
import sys
import time
from collections import defaultdict

import numpy as np

CAPTURE = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1"
BF16 = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16"
INDEX = os.path.join(BF16, "model.safetensors.index.json")
OUT = "/tmp/g1_decision_margin_out.json"
N_TOKENS = 256
HIDDEN = 5120
VOCAB = 248320
SOURCE_N = 26_895_998_464
G0_BYTES = 14_297_694_680
G0_BPW = 4.252735126866492
CHUNK = 4096
ISLAND = (3994, 3456, 310)


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] rss={rss_gb():.3f}G {msg}", flush=True)


def bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def parse_st_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def load_named_tensor(path: str, name: str) -> np.ndarray:
    header, data0 = parse_st_header(path)
    info = header[name]
    dtype = info["dtype"]
    shape = tuple(info["shape"])
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data0 + start)
        raw = f.read(end - start)
    if dtype == "BF16":
        arr = bf16_bytes_to_f32(raw)
    elif dtype == "F32":
        arr = np.frombuffer(raw, dtype="<f4").copy()
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    else:
        raise RuntimeError(f"unsupported dtype {dtype} for {name}")
    return arr.reshape(shape)


def load_hidden(layer: int) -> np.ndarray:
    path = os.path.join(CAPTURE, "hidden", f"L{layer:02d}.f32")
    x = np.fromfile(path, dtype="<f4")
    if x.size != N_TOKENS * HIDDEN:
        raise RuntimeError(f"{path} size {x.size}")
    return np.ascontiguousarray(x.reshape(N_TOKENS, HIDDEN), dtype=np.float32)


def q4_rtn_group64(w: np.ndarray) -> np.ndarray:
    rows, cols = w.shape
    if cols % 64 != 0:
        raise RuntimeError(f"cols {cols} not divisible by 64")
    g = w.reshape(rows, cols // 64, 64)
    scale = np.max(np.abs(g), axis=-1, keepdims=True) / 7.0
    scale = np.maximum(scale, 1e-12)
    q = np.clip(np.round(g / scale), -7.0, 7.0)
    return (q * scale).reshape(rows, cols)


def rms(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x), axis=axis))


def pct(xs: np.ndarray, ps) -> dict:
    return {str(p): float(np.percentile(xs, p)) for p in ps}


def main() -> int:
    t0 = time.perf_counter()
    report: dict = {
        "schema": "hawking.g1.decision_margin_budget.v1",
        "labels": {},
        "site": {},
        "capture": {},
        "tokens": {},
        "margins": {},
        "fragile": {},
        "special": {},
        "attribution": {},
        "screen_overlap": {},
        "q4_probe": {},
        "acceptance": {},
        "allocation": {},
        "bpw": {},
        "rss_gb_peak": None,
        "wall_s": None,
    }

    meta = json.load(open(os.path.join(CAPTURE, "capture-result.json")))
    prompts = meta["prompts"]
    assert meta["n_tokens"] == N_TOKENS
    ids = []
    prompt_of = []
    pos_in = []
    for pi, p in enumerate(prompts):
        for j, tok in enumerate(p["ids"]):
            ids.append(int(tok))
            prompt_of.append(pi)
            pos_in.append(j)
    ids = np.asarray(ids, dtype=np.int32)
    prompt_of = np.asarray(prompt_of, dtype=np.int32)
    pos_in = np.asarray(pos_in, dtype=np.int32)
    assert ids.size == N_TOKENS

    next_id = np.full(N_TOKENS, -1, dtype=np.int32)
    is_last = np.zeros(N_TOKENS, dtype=bool)
    for i in range(N_TOKENS):
        if i + 1 < N_TOKENS and prompt_of[i + 1] == prompt_of[i]:
            next_id[i] = ids[i + 1]
        else:
            is_last[i] = True

    vocab = json.load(open(os.path.join(BF16, "vocab.json")))
    id2 = {int(v): k for k, v in vocab.items()}
    tok_json = json.load(open(os.path.join(BF16, "tokenizer.json")))
    for a in tok_json.get("added_tokens") or []:
        id2[int(a["id"])] = a["content"]

    def dec(i: int) -> str:
        return id2.get(int(i), f"<unk_{i}>")

    decoded = [dec(i) for i in ids]
    report["capture"] = {
        "schema": meta["schema"],
        "status": meta["status"],
        "source": meta["source"],
        "n_tokens": N_TOKENS,
        "n_layers": 64,
        "hidden": HIDDEN,
        "n_prompts": len(prompts),
        "prompt_n": [int(p["n_tokens"]) for p in prompts],
        "prompt_text": [p["prompt"] for p in prompts],
        "wall_s_capture": meta["wall_s"],
        "sha256_self_claimed": meta["sha256_self"],
        "fit_kind": meta["fit_kind"],
        "label": "MEASURED",
    }

    h = hashlib.sha256()
    for layer in range(64):
        with open(os.path.join(CAPTURE, "hidden", f"L{layer:02d}.f32"), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    payload_sha = h.hexdigest()
    json_sha = hashlib.sha256(open(os.path.join(CAPTURE, "capture-result.json"), "rb").read()).hexdigest()
    report["capture"]["hidden_payload_sha256"] = payload_sha
    report["capture"]["capture_result_json_sha256"] = json_sha
    report["capture"]["sha256_self_matches_payload"] = payload_sha == meta["sha256_self"]

    SPECIAL = {
        248044: "<|endoftext|>",
        248045: "<|im_start|>",
        248046: "<|im_end|>",
        248058: "<tool_call>",
        248059: "</tool_call>",
        248066: "<tool_response>",
        248067: "</tool_response>",
        248068: "<think>",
        248069: "</think>",
    }
    STRUCTURAL_IDS = {248044, 248045, 248046, 248058, 248059, 248066, 248067, 248068, 248069, 198}
    ROLE_IDS = {8678, 846, 74455}  # system, user, assistant

    # Role / task class per position (current token).
    role = []
    task = []
    for i in range(N_TOKENS):
        pi = int(prompt_of[i])
        tok = int(ids[i])
        pids = prompts[pi]["ids"]
        # find spans
        try:
            first_im_end = pids.index(248046)
        except ValueError:
            first_im_end = len(pids)
        # user content between first user header and next im_end
        # template: im_start system nl SYS im_end nl im_start user nl USER im_end nl im_start assistant nl think nl
        r = "unknown"
        if tok == 248045:
            r = "im_start"
        elif tok == 248046:
            r = "im_end"
        elif tok == 248068:
            r = "think_open"
        elif tok == 248069:
            r = "think_close"
        elif tok in (248058, 248059, 248066, 248067):
            r = "tool_delim"
        elif tok == 8678:
            r = "role_system"
        elif tok == 846:
            r = "role_user"
        elif tok == 74455:
            r = "role_assistant"
        elif tok == 198:
            r = "newline"
        else:
            j = int(pos_in[i])
            if j < first_im_end:
                r = "system_text"
            else:
                # after first im_end
                # find user start
                user_starts = [k for k, t in enumerate(pids) if t == 846]
                asst_starts = [k for k, t in enumerate(pids) if t == 74455]
                user_k = user_starts[0] if user_starts else None
                asst_k = asst_starts[0] if asst_starts else None
                if user_k is not None and asst_k is not None and user_k < j < asst_k:
                    r = "user_text"
                elif asst_k is not None and j > asst_k:
                    r = "assistant_prefix"
                else:
                    r = "template_other"
        role.append(r)
        if pi == 0:
            tc = "factual"
        elif pi == 1:
            tc = "ordinary_instruction"
        elif pi == 2:
            tc = "code"
        elif pi == 3:
            tc = "math_reasoning"
        else:
            tc = "truncated_system"
        # override for shared template
        if r in ("im_start", "im_end", "think_open", "think_close", "role_system", "role_user",
                 "role_assistant", "newline", "system_text", "assistant_prefix", "template_other"):
            if r == "user_text":
                pass
            elif r == "system_text":
                tc = "system_template"
            elif r in ("think_open", "think_close"):
                tc = "think_delim"
            elif r in ("im_start", "im_end", "role_system", "role_user", "role_assistant",
                       "newline", "assistant_prefix", "template_other"):
                tc = "chat_template"
        if r == "user_text":
            if pi == 0:
                tc = "factual"
            elif pi == 1:
                tc = "ordinary_instruction"
            elif pi == 2:
                tc = "code"
            elif pi == 3:
                tc = "math_reasoning"
            else:
                tc = "truncated_system"
        task.append(tc)

    role = np.asarray(role)
    task = np.asarray(task)

    report["tokens"] = {
        "label": "MEASURED",
        "n": N_TOKENS,
        "n_last": int(is_last.sum()),
        "role_counts": {k: int((role == k).sum()) for k in sorted(set(role))},
        "task_counts": {k: int((task == k).sum()) for k in sorted(set(task))},
        "decoded_last_per_prompt": [
            {"prompt": prompts[pi]["prompt"], "last_id": int(ids[is_last & (prompt_of == pi)][0]),
             "last_tok": dec(int(ids[is_last & (prompt_of == pi)][0])),
             "n": int(prompts[pi]["n_tokens"])}
            for pi in range(len(prompts))
        ],
        "special_present": {SPECIAL[s]: int((ids == s).sum()) for s in SPECIAL},
        "note": "prompt 4 is 10 tokens of the shared system prefix only; no user text, no think.",
    }

    # ---- site identification ----
    log("loading layernorm weights for site check")
    weight_map = json.load(open(INDEX))["weight_map"]
    in_rms = []
    post_rms = []
    in_w = []
    for layer in range(64):
        n1 = f"language_model.model.layers.{layer}.input_layernorm.weight"
        n2 = f"language_model.model.layers.{layer}.post_attention_layernorm.weight"
        p1 = os.path.join(BF16, weight_map[n1])
        p2 = os.path.join(BF16, weight_map[n2])
        w1 = load_named_tensor(p1, n1).astype(np.float32).reshape(-1)
        w2 = load_named_tensor(p2, n2).astype(np.float32).reshape(-1)
        in_w.append(w1)
        in_rms.append(float(np.sqrt(np.mean(w1 * w1))))
        post_rms.append(float(np.sqrt(np.mean(w2 * w2))))
    in_w = np.stack(in_w, 0)  # 64 x 5120
    final_name = "language_model.model.norm.weight"
    final_w = load_named_tensor(os.path.join(BF16, weight_map[final_name]), final_name).astype(np.float32).reshape(-1)
    final_rms = float(np.sqrt(np.mean(final_w * final_w)))

    log("loading all 64 hidden layers")
    H = np.empty((64, N_TOKENS, HIDDEN), dtype=np.float32)
    for layer in range(64):
        H[layer] = load_hidden(layer)
    hid_rms = np.sqrt(np.mean(H ** 2, axis=(1, 2)))
    hid_rms_tok = np.sqrt(np.mean(H ** 2, axis=2))  # 64 x 256

    # correlation of layer-mean hidden RMS vs input_layernorm weight RMS
    def corr(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a = a - a.mean()
        b = b - b.mean()
        den = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / den) if den > 0 else 0.0

    # per-layer channel-wise: does |h| track |w_ln|?
    ch_corr = []
    for layer in range(64):
        hm = np.mean(np.abs(H[layer]), axis=0)
        ch_corr.append(corr(hm, np.abs(in_w[layer])))

    # L7 channel 3994 zero check (prior finding)
    l7_3994_nz = int(np.count_nonzero(H[7, :, 3994]))
    post7 = load_named_tensor(
        os.path.join(BF16, weight_map["language_model.model.layers.7.post_attention_layernorm.weight"]),
        "language_model.model.layers.7.post_attention_layernorm.weight",
    ).reshape(-1)
    post7_3994 = float(post7[3994])

    report["site"] = {
        "label": "MEASURED",
        "hidden_layer_rms": [float(x) for x in hid_rms],
        "input_ln_weight_rms": in_rms,
        "post_attn_ln_weight_rms": post_rms,
        "final_norm_weight_rms": final_rms,
        "corr_hidden_rms_vs_input_ln_rms": corr(hid_rms, in_rms),
        "corr_hidden_rms_vs_post_ln_rms": corr(hid_rms, post_rms),
        "mean_channel_corr_absH_vs_abs_input_ln": float(np.mean(ch_corr)),
        "per_layer_channel_corr_absH_vs_abs_input_ln": [float(x) for x in ch_corr],
        "L63_hidden_rms": float(hid_rms[63]),
        "L0_hidden_rms": float(hid_rms[0]),
        "L61_hidden_rms": float(hid_rms[61]),
        "L7_ch3994_nonzero": l7_3994_nz,
        "L7_post_attn_ln_w3994": post7_3994,
        "verdict": (
            "POST_INPUT_NORM_LIKELY" if corr(hid_rms, in_rms) > 0.8
            else "NOT_INPUT_LN_SCALE"
        ),
        "not_confirmed_final_norm": True,
        "note": "Logits below are L63 captured hidden @ BF16 lm_head.T. Site is not confirmed language_model.model.norm.",
    }

    # ---- logits from L63 @ lm_head ----
    X = np.ascontiguousarray(H[63], dtype=np.float32)
    lm_path = os.path.join(BF16, "model-00011-of-00011.safetensors")
    header, data0 = parse_st_header(lm_path)
    info = header["language_model.lm_head.weight"]
    assert info["shape"] == [VOCAB, HIDDEN]
    assert info["dtype"] == "BF16"
    start, end = info["data_offsets"]
    assert end - start == VOCAB * HIDDEN * 2

    log(f"computing logits L63 @ lm_head, chunk={CHUNK}")
    TOPK = 5
    top_z = np.full((N_TOKENS, TOPK), -np.inf, dtype=np.float32)
    top_id = np.full((N_TOKENS, TOPK), -1, dtype=np.int32)
    teacher_z = np.full(N_TOKENS, np.nan, dtype=np.float32)
    n_better_than_teacher = np.zeros(N_TOKENS, dtype=np.int32)
    special_z = {s: np.empty(N_TOKENS, dtype=np.float32) for s in SPECIAL}
    island_col_dot = {}  # filled later from rows

    # also keep W rows for specials + we'll fetch top1/top2 rows after
    special_rows = {}

    with open(lm_path, "rb") as f:
        for row0 in range(0, VOCAB, CHUNK):
            row1 = min(VOCAB, row0 + CHUNK)
            nbytes = (row1 - row0) * HIDDEN * 2
            f.seek(data0 + start + row0 * HIDDEN * 2)
            raw = f.read(nbytes)
            W = np.ascontiguousarray(bf16_bytes_to_f32(raw).reshape(row1 - row0, HIDDEN))
            z = X @ W.T  # 256 x chunk
            # teacher
            for i in range(N_TOKENS):
                nid = int(next_id[i])
                if nid >= row0 and nid < row1:
                    teacher_z[i] = z[i, nid - row0]
            # specials
            for s in SPECIAL:
                if row0 <= s < row1:
                    special_z[s][:] = z[:, s - row0]
                    special_rows[s] = W[s - row0].copy()
            # count better than teacher (partial; complete after we have teacher_z)
            # top-k merge
            for i in range(N_TOKENS):
                # take topk of this chunk
                if row1 - row0 >= TOPK:
                    part = np.argpartition(z[i], -TOPK)[-TOPK:]
                else:
                    part = np.arange(row1 - row0)
                vals = z[i, part]
                ids_c = part + row0
                # merge with existing
                all_v = np.concatenate([top_z[i], vals])
                all_i = np.concatenate([top_id[i], ids_c.astype(np.int32)])
                keep = np.argpartition(all_v, -TOPK)[-TOPK:]
                keep = keep[np.argsort(-all_v[keep])]
                top_z[i] = all_v[keep]
                top_id[i] = all_i[keep]
            if row0 % (CHUNK * 8) == 0:
                log(f"  lm_head rows {row0}-{row1}")

    # second pass for teacher rank: count how many logits > teacher_z
    # We didn't keep all logits. Re-scan only to count rank. Expensive but correct.
    log("second pass: teacher rank")
    with open(lm_path, "rb") as f:
        for row0 in range(0, VOCAB, CHUNK):
            row1 = min(VOCAB, row0 + CHUNK)
            nbytes = (row1 - row0) * HIDDEN * 2
            f.seek(data0 + start + row0 * HIDDEN * 2)
            raw = f.read(nbytes)
            W = np.ascontiguousarray(bf16_bytes_to_f32(raw).reshape(row1 - row0, HIDDEN))
            z = X @ W.T
            for i in range(N_TOKENS):
                if not np.isnan(teacher_z[i]):
                    n_better_than_teacher[i] += int(np.count_nonzero(z[i] > teacher_z[i]))

    margin = top_z[:, 0] - top_z[:, 1]
    argmax = top_id[:, 0]
    teacher_hit = np.zeros(N_TOKENS, dtype=bool)
    valid_tf = next_id >= 0
    teacher_hit[valid_tf] = argmax[valid_tf] == next_id[valid_tf]
    teacher_rank = np.where(valid_tf, n_better_than_teacher + 1, -1)

    # last-token predicted ids
    last_pred = []
    for pi in range(len(prompts)):
        idx = np.where(is_last & (prompt_of == pi))[0][0]
        last_pred.append({
            "prompt_i": pi,
            "prompt": prompts[pi]["prompt"],
            "cur_id": int(ids[idx]),
            "cur_tok": dec(int(ids[idx])),
            "pred_id": int(argmax[idx]),
            "pred_tok": dec(int(argmax[idx])),
            "second_id": int(top_id[idx, 1]),
            "second_tok": dec(int(top_id[idx, 1])),
            "margin": float(margin[idx]),
            "z1": float(top_z[idx, 0]),
            "z2": float(top_z[idx, 1]),
            "top5": [{"id": int(top_id[idx, k]), "tok": dec(int(top_id[idx, k])), "z": float(top_z[idx, k])}
                     for k in range(TOPK)],
        })

    def summarize(mask, name):
        m = margin[mask]
        if m.size == 0:
            return {"n": 0}
        hit = teacher_hit[mask & valid_tf]
        return {
            "n": int(m.size),
            "n_teacher": int(hit.size),
            "teacher_top1_acc": float(hit.mean()) if hit.size else None,
            "teacher_mean_rank": float(teacher_rank[mask & valid_tf].mean()) if hit.size else None,
            "margin_mean": float(m.mean()),
            "margin_median": float(np.median(m)),
            "margin_min": float(m.min()),
            "margin_max": float(m.max()),
            "margin_p": pct(m, [1, 5, 10, 25, 50, 75, 90, 95, 99]),
            "frac_m_lt_0_25": float((m < 0.25).mean()),
            "frac_m_lt_0_5": float((m < 0.5).mean()),
            "frac_m_lt_1": float((m < 1.0).mean()),
            "frac_m_lt_2": float((m < 2.0).mean()),
            "frac_m_lt_3": float((m < 3.0).mean()),
            "frac_m_ge_5": float((m >= 5.0).mean()),
            "frac_m_ge_10": float((m >= 10.0).mean()),
        }

    by_task = {t: summarize(task == t, t) for t in sorted(set(task))}
    by_role = {r: summarize(role == r, r) for r in sorted(set(role))}

    report["margins"] = {
        "label": "MEASURED_PROXY",
        "site": "L63_captured_hidden @ BF16_lm_head.T",
        "not_confirmed_final_norm": True,
        "all": summarize(np.ones(N_TOKENS, dtype=bool), "all"),
        "teacher_forced_only": summarize(valid_tf, "tf"),
        "last_token_only": summarize(is_last, "last"),
        "non_last": summarize(~is_last, "nonlast"),
        "by_task": by_task,
        "by_role": by_role,
        "last_token_predictions": last_pred,
        "teacher_top1_acc_all_tf": float(teacher_hit[valid_tf].mean()),
        "teacher_mean_rank_tf": float(teacher_rank[valid_tf].mean()),
        "teacher_median_rank_tf": float(np.median(teacher_rank[valid_tf])),
        "n_tf": int(valid_tf.sum()),
        "argmax_decoded_sample": [
            {"i": int(i), "cur": dec(int(ids[i])), "pred": dec(int(argmax[i])),
             "next": dec(int(next_id[i])) if next_id[i] >= 0 else None,
             "hit": bool(teacher_hit[i]), "m": float(margin[i]),
             "role": str(role[i]), "task": str(task[i])}
            for i in list(range(0, N_TOKENS, 32)) + [int(x) for x in np.where(is_last)[0]]
        ],
    }

    # ---- load top1/top2 rows for attribution ----
    log("loading top1/top2 lm_head rows for attribution")
    uniq_rows = sorted(set(int(x) for x in top_id[:, :2].reshape(-1)) | set(SPECIAL) | set(ISLAND))
    # map id -> row vector
    Wrow = {}
    with open(lm_path, "rb") as f:
        # sequential fetch by sorted id
        for tid in uniq_rows:
            f.seek(data0 + start + tid * HIDDEN * 2)
            raw = f.read(HIDDEN * 2)
            Wrow[tid] = bf16_bytes_to_f32(raw).copy()

    d = np.stack([Wrow[int(top_id[i, 0])] - Wrow[int(top_id[i, 1])] for i in range(N_TOKENS)])  # 256 x 5120
    # verify m ≈ <h, d>
    m_recon = np.sum(X * d, axis=1)
    recon_err = np.max(np.abs(m_recon - margin))
    report["margins"]["max_abs_margin_minus_h_dot_d"] = float(recon_err)
    report["margins"]["mean_abs_margin_minus_h_dot_d"] = float(np.mean(np.abs(m_recon - margin)))

    h_norm = np.linalg.norm(X, axis=1)
    d_norm = np.linalg.norm(d, axis=1)
    # worst-case critical relative perturbation
    # flip if ||δh|| > m / ||d||   (aligned against decision)
    crit_abs = margin / np.maximum(d_norm, 1e-12)
    crit_rel = crit_abs / np.maximum(h_norm, 1e-12)
    # isotropic 2-sigma: std(<δh,d>) = ||δh|| ||d|| / sqrt(dim)
    # flip ~ when ||δh|| / ||h|| > m * sqrt(dim) / (||h|| ||d||) / 2
    sqrt_d = np.sqrt(HIDDEN)
    crit_iso_rel = (margin * sqrt_d) / np.maximum(h_norm * d_norm, 1e-12)

    # plausible eps from typical Q4 output rel-L2 (CITED descent L63 organs 0.04-0.10)
    EPS_GRID = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]

    def frag_stats(mask):
        if mask.sum() == 0:
            return {"n": 0}
        cr = crit_rel[mask]
        ci = crit_iso_rel[mask]
        out = {
            "n": int(mask.sum()),
            "crit_rel_worst_p": pct(cr, [5, 10, 25, 50, 75, 90, 95]),
            "crit_rel_iso2sig_p": pct(ci, [5, 10, 25, 50, 75, 90, 95]),
            "frac_worst_flip_at_eps": {str(e): float((cr < e).mean()) for e in EPS_GRID},
            "frac_iso2sig_flip_at_eps": {str(e): float((ci < e).mean()) for e in EPS_GRID},
            "frac_m_lt": {str(t): float((margin[mask] < t).mean()) for t in [0.25, 0.5, 1, 2, 3, 5]},
        }
        return out

    report["fragile"] = {
        "label": "MEASURED_PROXY + DERIVED thresholds",
        "definition_worst": "position is worst-case-fragile at eps if ||δh||/||h|| < m/||d||/||h|| i.e. an aligned perturbation of relative size eps flips argmax",
        "definition_iso": "isotropic 2-sigma: rel eps flips when eps > m*sqrt(5120)/(||h||||d||)",
        "eps_grid": EPS_GRID,
        "eps_source": "EPS is a hypothesized residual relative error. Descent hold_output_rel_l2 at Q4 is 0.04-0.11 on scored organs (CITED g1-heterogeneous-allocation). Layer-63 output/input norm 2.60396 is amplification (CITED contract), not an eps.",
        "all": frag_stats(np.ones(N_TOKENS, dtype=bool)),
        "tf": frag_stats(valid_tf),
        "last": frag_stats(is_last),
        "user_text": frag_stats(role == "user_text"),
        "system_text": frag_stats(role == "system_text"),
        "chat_template_roles": frag_stats(np.isin(role, ["im_start", "im_end", "role_system", "role_user", "role_assistant", "newline", "assistant_prefix"])),
        "think_open": frag_stats(role == "think_open"),
        "code_user": frag_stats(task == "code"),
        "math_user": frag_stats(task == "math_reasoning"),
        "factual_user": frag_stats(task == "factual"),
        "ordinary_user": frag_stats(task == "ordinary_instruction"),
        "recon_err_max": float(recon_err),
    }

    # special-position table
    special_pos = []
    for i in range(N_TOKENS):
        tok = int(ids[i])
        if tok in SPECIAL or is_last[i] or role[i] in ("think_open", "im_start", "im_end"):
            rec = {
                "i": i,
                "prompt": int(prompt_of[i]),
                "pos": int(pos_in[i]),
                "cur_id": tok,
                "cur": dec(tok),
                "role": str(role[i]),
                "task": str(task[i]),
                "is_last": bool(is_last[i]),
                "pred_id": int(argmax[i]),
                "pred": dec(int(argmax[i])),
                "second_id": int(top_id[i, 1]),
                "second": dec(int(top_id[i, 1])),
                "margin": float(margin[i]),
                "teacher_id": int(next_id[i]),
                "teacher": dec(int(next_id[i])) if next_id[i] >= 0 else None,
                "teacher_hit": bool(teacher_hit[i]) if valid_tf[i] else None,
                "teacher_rank": int(teacher_rank[i]) if valid_tf[i] else None,
                "crit_rel_worst": float(crit_rel[i]),
                "crit_rel_iso2": float(crit_iso_rel[i]),
            }
            special_pos.append(rec)
    report["special"]["positions"] = special_pos
    report["special"]["label"] = "MEASURED_PROXY"

    # pairwise special-row geometry
    sids = sorted(SPECIAL)
    srows = np.stack([Wrow[s] for s in sids])
    sn = srows / np.maximum(np.linalg.norm(srows, axis=1, keepdims=True), 1e-12)
    scos = sn @ sn.T
    pair = []
    for a in range(len(sids)):
        for b in range(a + 1, len(sids)):
            pair.append({
                "a": SPECIAL[sids[a]], "b": SPECIAL[sids[b]],
                "cosine": float(scos[a, b]),
                "l2": float(np.linalg.norm(srows[a] - srows[b])),
            })
    pair.sort(key=lambda r: -r["cosine"])
    report["special"]["row_pairwise"] = {
        "label": "MEASURED",
        "rows": [SPECIAL[s] for s in sids],
        "row_l2": {SPECIAL[s]: float(np.linalg.norm(Wrow[s])) for s in sids},
        "nearest": pair[:12],
        "think_vs_think_close": float(
            np.dot(Wrow[248068], Wrow[248069])
            / (np.linalg.norm(Wrow[248068]) * np.linalg.norm(Wrow[248069]))
        ),
        "tool_call_vs_tool_call_close": float(
            np.dot(Wrow[248058], Wrow[248059])
            / (np.linalg.norm(Wrow[248058]) * np.linalg.norm(Wrow[248059]))
        ),
        "think_vs_tool_call": float(
            np.dot(Wrow[248068], Wrow[248058])
            / (np.linalg.norm(Wrow[248068]) * np.linalg.norm(Wrow[248058]))
        ),
        "im_end_vs_eot": float(
            np.dot(Wrow[248046], Wrow[248044])
            / (np.linalg.norm(Wrow[248046]) * np.linalg.norm(Wrow[248044]))
        ),
    }

    # ---- attribution ----
    log("attribution: channels and layers")
    contrib = X * d  # 256 x 5120, sums to margin
    # energy fraction of island channels
    island_frac = {}
    for ch in ISLAND:
        island_frac[str(ch)] = {
            "mean_abs_frac_of_margin": float(np.mean(np.abs(contrib[:, ch]) / np.maximum(np.abs(margin), 1e-12))),
            "mean_signed_frac_of_margin": float(np.mean(contrib[:, ch] / np.maximum(margin, 1e-12))),
            "mean_abs_h": float(np.mean(np.abs(X[:, ch]))),
            "mean_abs_d": float(np.mean(np.abs(d[:, ch]))),
            "mean_abs_h_over_mean_abs_h": float(np.mean(np.abs(X[:, ch])) / np.mean(np.abs(X))),
            "mean_abs_d_over_mean_abs_d": float(np.mean(np.abs(d[:, ch])) / np.mean(np.abs(d))),
        }

    # top channels by |contrib| averaged over last tokens and over all
    def top_channels(mask, k=16):
        c = np.mean(np.abs(contrib[mask]), axis=0)
        idx = np.argsort(-c)[:k]
        hmean = np.mean(np.abs(X[mask]), axis=0)
        dmean = np.mean(np.abs(d[mask]), axis=0)
        return [
            {
                "ch": int(i),
                "mean_abs_contrib": float(c[i]),
                "frac_of_mean_abs_margin": float(c[i] / max(float(np.mean(np.abs(margin[mask]))), 1e-12)),
                "mean_abs_h": float(hmean[i]),
                "mean_abs_d": float(dmean[i]),
                "is_island": int(i) in ISLAND,
            }
            for i in idx
        ]

    # layer alignment of captured hidden with d (d lives in lm_head / last-hidden space)
    # PROXY: cosine(h_L, d) — only apples-to-apples if h_L is same residual basis
    layer_cos = np.empty((64, N_TOKENS), dtype=np.float32)
    layer_dot = np.empty((64, N_TOKENS), dtype=np.float32)
    for layer in range(64):
        hn = H[layer]
        layer_dot[layer] = np.sum(hn * d, axis=1)
        layer_cos[layer] = layer_dot[layer] / np.maximum(
            np.linalg.norm(hn, axis=1) * d_norm, 1e-12
        )

    # which layer has max |cos| / max |dot| at last tokens
    last_idx = np.where(is_last)[0]
    layer_mean_abs_cos_last = np.mean(np.abs(layer_cos[:, last_idx]), axis=1)
    layer_mean_abs_dot_last = np.mean(np.abs(layer_dot[:, last_idx]), axis=1)
    layer_mean_abs_cos_all = np.mean(np.abs(layer_cos), axis=1)
    # delta of projection: if post-norm, not a residual increment; still a sequence
    dproj = np.diff(layer_dot, axis=0)  # 63 x 256

    top_layers_last = np.argsort(-layer_mean_abs_cos_last)[:8]
    top_layers_dproj = np.argsort(-np.mean(np.abs(dproj[:, last_idx]), axis=1))[:8]

    # GQA vs DeltaNet layers
    gqa_layers = [i for i in range(64) if (i + 1) % 4 == 0]
    dn_layers = [i for i in range(64) if (i + 1) % 4 != 0]
    report["attribution"] = {
        "label": "MEASURED_PROXY",
        "depends_on": "capture ranks a sharp set; magnitudes are underdetermined (contract). Channel ranking from |h*d| is a RANK. Flip fractions use measured m,||h||,||d||.",
        "margin_recon_max_abs_err": float(recon_err),
        "island_in_decision": island_frac,
        "top_channels_all": top_channels(np.ones(N_TOKENS, dtype=bool)),
        "top_channels_last": top_channels(is_last, 24),
        "top_channels_user_text": top_channels(role == "user_text") if (role == "user_text").any() else [],
        "top_channels_think_open": top_channels(role == "think_open") if (role == "think_open").any() else [],
        "overlap_top16_all_with_island": [c["ch"] for c in top_channels(np.ones(N_TOKENS, dtype=bool), 16) if c["is_island"]],
        "overlap_top16_last_with_island": [c["ch"] for c in top_channels(is_last, 16) if c["is_island"]],
        "rank_of_3994_by_abs_contrib_all": int(np.argsort(-np.mean(np.abs(contrib), axis=0)).tolist().index(3994) + 1),
        "rank_of_3456_by_abs_contrib_all": int(np.argsort(-np.mean(np.abs(contrib), axis=0)).tolist().index(3456) + 1),
        "rank_of_310_by_abs_contrib_all": int(np.argsort(-np.mean(np.abs(contrib), axis=0)).tolist().index(310) + 1),
        "rank_of_3994_by_abs_h_all": int(np.argsort(-np.mean(np.abs(X), axis=0)).tolist().index(3994) + 1),
        "rank_of_3994_by_abs_d_all": int(np.argsort(-np.mean(np.abs(d), axis=0)).tolist().index(3994) + 1),
        "layer_mean_abs_cos_last": [float(x) for x in layer_mean_abs_cos_last],
        "layer_mean_abs_dot_last": [float(x) for x in layer_mean_abs_dot_last],
        "layer_mean_abs_cos_all": [float(x) for x in layer_mean_abs_cos_all],
        "top_layers_by_abs_cos_last": [int(x) for x in top_layers_last],
        "top_layers_by_abs_dproj_last": [int(x) for x in top_layers_dproj],
        "gqa_mean_abs_cos_last": float(layer_mean_abs_cos_last[gqa_layers].mean()),
        "dn_mean_abs_cos_last": float(layer_mean_abs_cos_last[dn_layers].mean()),
        "late16_mean_abs_cos_last": float(layer_mean_abs_cos_last[48:].mean()),
        "early16_mean_abs_cos_last": float(layer_mean_abs_cos_last[:16].mean()),
        "heads": "UNMEASURED: mixer_x never captured; cannot attribute to GQA/DeltaNet heads without a mixer-site dump or a generate.",
    }

    # decision-subspace rank at last + small-margin positions
    log("decision subspace SVD")
    small = margin < 2.0
    for tag, mask in [
        ("last", is_last),
        ("m_lt_2", small),
        ("m_lt_1", margin < 1.0),
        ("all", np.ones(N_TOKENS, dtype=bool)),
        ("user_or_last", (role == "user_text") | is_last),
    ]:
        if mask.sum() < 2:
            continue
        D = d[mask]
        # thin SVD
        # D is n x 5120, n small
        # use covariance in the smaller dim
        if D.shape[0] <= HIDDEN:
            # SVD of D
            s = np.linalg.svd(D, compute_uv=False)
        else:
            s = np.sqrt(np.linalg.eigvalsh(D.T @ D)[::-1])
        s = np.maximum(s, 0)
        e = s ** 2
        tot = float(e.sum()) if e.sum() else 1.0
        cume = np.cumsum(e) / tot
        def rank_at(p):
            return int(np.searchsorted(cume, p) + 1)
        report["attribution"].setdefault("subspace", {})[tag] = {
            "n": int(mask.sum()),
            "sv_top8": [float(x) for x in s[:8]],
            "rank_90": rank_at(0.90),
            "rank_95": rank_at(0.95),
            "rank_99": rank_at(0.99),
            "eff_rank": float((e.sum() ** 2) / max(float((e ** 2).sum()), 1e-18)),
        }

    # ---- Q4 RTN probe on lm_head ----
    log("Q4 RTN probe on lm_head (doctor-compatible qmax=7 g64)")
    top_z_q = np.full((N_TOKENS, TOPK), -np.inf, dtype=np.float32)
    top_id_q = np.full((N_TOKENS, TOPK), -1, dtype=np.int32)
    # also keep candidate z at teacher top1/top2 for margin preservation
    zq_t1 = np.zeros(N_TOKENS, dtype=np.float32)
    zq_t2 = np.zeros(N_TOKENS, dtype=np.float32)
    with open(lm_path, "rb") as f:
        for row0 in range(0, VOCAB, CHUNK):
            row1 = min(VOCAB, row0 + CHUNK)
            nbytes = (row1 - row0) * HIDDEN * 2
            f.seek(data0 + start + row0 * HIDDEN * 2)
            raw = f.read(nbytes)
            W = np.ascontiguousarray(bf16_bytes_to_f32(raw).reshape(row1 - row0, HIDDEN))
            Wq = q4_rtn_group64(W)
            z = X @ Wq.T
            for i in range(N_TOKENS):
                t1 = int(top_id[i, 0])
                t2 = int(top_id[i, 1])
                if row0 <= t1 < row1:
                    zq_t1[i] = z[i, t1 - row0]
                if row0 <= t2 < row1:
                    zq_t2[i] = z[i, t2 - row0]
                if row1 - row0 >= TOPK:
                    part = np.argpartition(z[i], -TOPK)[-TOPK:]
                else:
                    part = np.arange(row1 - row0)
                vals = z[i, part]
                ids_c = part + row0
                all_v = np.concatenate([top_z_q[i], vals])
                all_i = np.concatenate([top_id_q[i], ids_c.astype(np.int32)])
                keep = np.argpartition(all_v, -TOPK)[-TOPK:]
                keep = keep[np.argsort(-all_v[keep])]
                top_z_q[i] = all_v[keep]
                top_id_q[i] = all_i[keep]
            if row0 % (CHUNK * 8) == 0:
                log(f"  q4 rows {row0}-{row1}")

    argmax_q = top_id_q[:, 0]
    margin_q = top_z_q[:, 0] - top_z_q[:, 1]
    flip = argmax_q != argmax
    # teacher-pair margin under Q4 (not full argmax): m_pair = zq_t1 - zq_t2
    m_pair_q = zq_t1 - zq_t2
    pair_flip = m_pair_q <= 0

    def qstats(mask):
        if mask.sum() == 0:
            return {"n": 0}
        return {
            "n": int(mask.sum()),
            "argmax_flip_rate": float(flip[mask].mean()),
            "pair_flip_rate": float(pair_flip[mask].mean()),
            "mean_m_bf16": float(margin[mask].mean()),
            "mean_m_q4": float(margin_q[mask].mean()),
            "mean_m_pair_q4": float(m_pair_q[mask].mean()),
            "median_m_ratio_q_over_bf": float(np.median(margin_q[mask] / np.maximum(margin[mask], 1e-6))),
            "frac_margin_preserved_half": float(((~flip[mask]) & (margin_q[mask] >= 0.5 * margin[mask])).mean()),
            "frac_same_argmax_and_m_ge_1": float(((~flip[mask]) & (margin_q[mask] >= 1.0)).mean()),
        }

    report["q4_probe"] = {
        "label": "MEASURED_PROXY",
        "codec": "symmetric group-64 absmax RTN qmax=7 (doctor probe, not HQ30UQ4 packer bit-exact)",
        "all": qstats(np.ones(N_TOKENS, dtype=bool)),
        "tf": qstats(valid_tf),
        "last": qstats(is_last),
        "m_lt_1": qstats(margin < 1.0),
        "m_lt_2": qstats(margin < 2.0),
        "m_ge_5": qstats(margin >= 5.0),
        "user_text": qstats(role == "user_text"),
        "think_open": qstats(role == "think_open"),
        "last_flips": [
            {
                "prompt": int(prompt_of[i]),
                "bf16": dec(int(argmax[i])),
                "q4": dec(int(argmax_q[i])),
                "m_bf16": float(margin[i]),
                "m_q4": float(margin_q[i]),
                "m_pair_q4": float(m_pair_q[i]),
            }
            for i in np.where(is_last & flip)[0]
        ],
    }

    # ---- acceptance metric definition + measured baseline ----
    # Margin preservation statistic (MPS) on a frozen teacher-forced set.
    # Does not require generate.
    tau = 1.0
    beta = 0.5
    # MPS_strict: argmax match AND (if m_t >= tau: m_c >= beta m_t) AND (if m_t < tau: still argmax match)
    # That's just: no flip, and confident teacher margins stay at least half.
    no_flip = ~flip
    confident = margin >= tau
    mps_i = no_flip & ((~confident) | (margin_q >= beta * margin))
    # weighted MPS: weight = 1 / max(m_t, tau) so fragile positions dominate
    w = 1.0 / np.maximum(margin, tau)
    w = w / w.sum()
    report["acceptance"] = {
        "label": "MEASURED on Q4-lm_head probe; metric DEFINITION is proposed",
        "name": "margin_preservation_statistic",
        "requires_generate": False,
        "requires": "teacher hidden (captured) and candidate lm_head, or first-order δh from a candidate body organ",
        "definition": {
            "per_position": "preserve iff argmax_c == argmax_t AND (m_t < tau OR m_c >= beta * m_t)",
            "tau_logit": tau,
            "beta": beta,
            "unweighted_MPS": "mean(preserve)",
            "weighted_MPS": "sum(w_i * preserve_i), w_i = 1/max(m_t, tau) normalized",
            "knife_edge_hold": "on positions with m_t < tau, require argmax match (already in per_position)",
            "not_a_product_of_cosines": True,
        },
        "q4_lm_head_baseline": {
            "unweighted_MPS_all": float(mps_i.mean()),
            "weighted_MPS_all": float(np.dot(w, mps_i.astype(np.float64))),
            "unweighted_MPS_last": float(mps_i[is_last].mean()),
            "unweighted_MPS_m_lt_2": float(mps_i[margin < 2].mean()) if (margin < 2).any() else None,
            "unweighted_MPS_m_ge_5": float(mps_i[margin >= 5].mean()) if (margin >= 5).any() else None,
            "argmax_hold_all": float(no_flip.mean()),
            "argmax_hold_last": float(no_flip[is_last].mean()),
        },
        "proposed_gate": {
            "unweighted_MPS_all": 0.99,
            "weighted_MPS_all": 0.98,
            "argmax_hold_last": 1.0,
            "argmax_hold_think_and_tool_delims": 1.0,
            "note": "Gates are POLICY, not measured thresholds. Last-token hold 1.0 is the doctor capability_gate. Cosine-alone is NEVER sufficient (CITED doctor).",
        },
        "how_to_score_a_body_candidate_without_generate": (
            "Apply candidate organ to captured X (legal site only). Inject output error into residual "
            "as first-order δh (down/o: 5120-wide write error; gate/up: refuse unless post_swiglu captured). "
            "δz = δh @ W_lm.T (or <δh, d> for pair). Recompute argmax via top-k refresh if |δz| can "
            "admit a new competitor. Score MPS. This is first-order; REOPEN_IF a generate disagrees."
        ),
    }

    # ---- screen overlap vs weight-space ----
    # Doctor floors (CITED): late down 8-bit, L0 gate/up 1-bit, lm_head 8, L63 o 4, L63 k 3
    # Energy island {3994,3456,310}
    # Weight-space would protect 3994; decision-space ranking is measured above.
    report["screen_overlap"] = {
        "label": "MEASURED comparison against CITED screens",
        "energy_island": list(ISLAND),
        "3994_rank_energy_abs_h": report["attribution"]["rank_of_3994_by_abs_h_all"],
        "3994_rank_decision_abs_contrib": report["attribution"]["rank_of_3994_by_abs_contrib_all"],
        "3994_rank_decision_direction_abs_d": report["attribution"]["rank_of_3994_by_abs_d_all"],
        "finding_template": "If energy rank << decision rank, the residual-island screen protects a high-energy channel that does not span the argmax. That is a different structure.",
        "doctor_lm_head_floor_bits": 8,
        "doctor_lm_head_last_ids_cited": [1596, 1596, 1596, 1596, 11553],
        "this_run_last_pred_ids": [int(argmax[i]) for i in np.where(is_last)[0]],
        "this_run_last_pred_match_cited": [
            int(argmax[i]) for i in np.where(is_last)[0]
        ] == [1596, 1596, 1596, 1596, 11553],
        "weight_space_hard_organs_cited": "late down_proj (descent L63 binary hold 0.730); L0 lin_o kurtosis 149.36 on row 3994",
        "heads": "UNMEASURED here; doctor swept 14/1216 heads and found L3 q/6 and L63 q/18 hold at 1-bit while L3 q/0,q/23 hold only at 8. That is a weight-space head non-uniformity. This lane cannot test whether those heads drive fragile margins (no mixer_x).",
    }

    # ---- allocation + BPW ----
    # Mass table (CITED hetero inventory, language-only)
    mass = {
        "mlp.gate_proj": 5_704_253_440,
        "mlp.up_proj": 5_704_253_440,
        "mlp.down_proj": 5_704_253_440,
        "dn.in_proj_qkvz": 4_026_531_840,
        "dn.out_proj": 1_509_949_440,
        "embed": 1_271_398_400,
        "lm_head": 1_271_398_400,
        "gqa.q_proj": 1_006_632_960,
        "gqa.o_proj": 503_316_480,
        "gqa.k_proj": 83_886_080,
        "gqa.v_proj": 83_886_080,
        "dn.in_proj_ba": 23_592_960,
        "small_f32": 1966080 + 327680 + 327680 + 24064,  # conv + 2 ln + rest, CITED hetero ~2.3M
    }
    # verify sum
    mass_sum = sum(mass.values())
    report["bpw"]["mass_sum"] = mass_sum
    report["bpw"]["source_n"] = SOURCE_N
    report["bpw"]["mass_sum_minus_source"] = mass_sum - SOURCE_N

    def bpw_of(bytes_: int) -> float:
        return 8.0 * bytes_ / SOURCE_N

    def payload_q(bits: int, elements: int, group: int = 64) -> int:
        # HQ30UQ4-faithful-ish: groups * (2 byte scale + group*bits/8)
        # plus a small header. Use hetero formula:
        # 32+4*rank + groups*2 + groups*64*bits/8 with rank=0 for uniform
        groups = (elements + group - 1) // group
        return 32 + groups * 2 + groups * group * bits // 8

    def payload_binary_g128(elements: int) -> int:
        # calibrated L0 gate 12534021 / 89128960 in hetero
        # general: header 261 + f16 scale/128 + 1-bit signs
        groups = (elements + 127) // 128
        return 261 + groups * 2 + (elements + 7) // 8

    def payload_ternary_g128(elements: int) -> int:
        groups = (elements + 127) // 128
        # 2-bit codes + two f16 tables / 128
        return groups * 4 + groups * (elements // max(groups, 1)) * 2 // 8  # messy
        # use calibrated ratio 25067853/89128960
    # Use calibrated ratios from hetero for 1/2 bit; exact formula for 3/4.

    ratio_bin = 12534021 / 89128960
    ratio_ter = 25067853 / 89128960

    def org_bytes(kind: str, elements: int, bits: int) -> int:
        if bits >= 32:
            return 8 + 4 * elements
        if bits == 1:
            return int(round(ratio_bin * elements))
        if bits == 2:
            return int(round(ratio_ter * elements))
        return payload_q(bits, elements)

    # Decision-keyed recipe (POLICY informed by this measurement)
    # After we have numbers we'll fill recommended bits in a second small block using measured ranks.
    # Compute several scenarios here.

    # Scenario A: G0 uniform q4
    g0 = G0_BYTES
    # Scenario B: protect lm_head special+observed-top2 rows exact bf16; rest q4
    observed_ids = set(int(x) for x in top_id[:, :2].reshape(-1)) | set(SPECIAL) | set(ids.tolist())
    n_exact_rows = len(observed_ids)
    lm_exact_elems = n_exact_rows * HIDDEN
    lm_rest_elems = VOCAB * HIDDEN - lm_exact_elems
    # vs q4 body of lm_head: replace those rows with bf16
    lm_q4 = payload_q(4, VOCAB * HIDDEN)
    lm_hybrid = 2 * lm_exact_elems + payload_q(4, lm_rest_elems)
    # Scenario C: protect decision subspace rank R as extra bf16 correction on residual writes
    # residual writes: down 64*(5120*17408) + out 64*(5120*6144)
    down_e = 64 * 5120 * 17408
    out_e = 64 * 5120 * 6144
    assert down_e == 5_704_253_440
    # a rank-R right-factor correction per write: R * in_dim  (store V) + out_dim * R (store U)
    # shared U across layers would be cheaper; quote both.

    subspace = report["attribution"].get("subspace", {})
    R95 = int(subspace.get("m_lt_2", subspace.get("last", {})).get("rank_95", 8))

    def subspace_bytes(R: int, shared_u: bool) -> int:
        # 64 down: U (5120 x R) + V (R x 17408), bf16
        # 64 out:  U (5120 x R) + V (R x 6144)
        if shared_u:
            u = 5120 * R * 2  # one U
            v = 64 * R * (17408 + 6144) * 2
            return u + v
        u = 64 * 5120 * R * 2 * 2  # two writes per layer (down+out)
        v = 64 * R * (17408 + 6144) * 2
        return u + v

    # Scenario D: decision-keyed hetero
    # Cheap mass (gate/up/qkv that are orthogonal-ish to residual write) at 2-bit
    # Residual writes: q4 + subspace correction
    # lm_head: q3 + exact special/top2 rows
    # embed: q3
    # small: f32
    # k/v: q4 (tiny)

    def scenario_bytes(name, bits_map, extra=0):
        total = extra
        detail = {}
        for k, e in mass.items():
            b = bits_map[k]
            by = org_bytes(k, e, b)
            detail[k] = {"bits": b, "elements": e, "bytes": by, "bpw_on_own_e": 8 * by / e}
            total += by
        return {"name": name, "bytes": total, "complete_bpw": bpw_of(total), "extra_bytes": extra, "detail": detail}

    bits_g0ish = {k: (32 if k == "small_f32" else 4) for k in mass}
    bits_cheap = {
        "mlp.gate_proj": 2, "mlp.up_proj": 2, "mlp.down_proj": 4,
        "dn.in_proj_qkvz": 2, "dn.out_proj": 4,
        "embed": 3, "lm_head": 3,
        "gqa.q_proj": 2, "gqa.o_proj": 4, "gqa.k_proj": 4, "gqa.v_proj": 4,
        "dn.in_proj_ba": 3, "small_f32": 32,
    }
    extra_sub_shared = subspace_bytes(R95, True)
    extra_sub_per = subspace_bytes(R95, False)
    extra_lm_rows = 2 * lm_exact_elems  # already switching lm_head to 3; add exact rows as extra
    # more honest: lm_head at 3 plus upgrade exact rows from 3-bit to 16-bit
    lm3 = org_bytes("lm_head", mass["lm_head"], 3)
    lm3_plus_exact = lm3 - org_bytes("lm_head", lm_exact_elems, 3) + 2 * lm_exact_elems

    sc_g0 = scenario_bytes("g0_uniform_q4_formula", bits_g0ish, 0)
    sc_dec = scenario_bytes("decision_keyed_2_4_3", bits_cheap, extra_sub_shared + (lm3_plus_exact - org_bytes("lm_head", mass["lm_head"], 3)))
    sc_dec_per = scenario_bytes("decision_keyed_2_4_3_perlayerU", bits_cheap, extra_sub_per + (lm3_plus_exact - org_bytes("lm_head", mass["lm_head"], 3)))

    # aggressive: downs at 3 + subspace
    bits_aggr = dict(bits_cheap)
    bits_aggr["mlp.down_proj"] = 3
    bits_aggr["dn.out_proj"] = 3
    bits_aggr["gqa.o_proj"] = 3
    sc_aggr = scenario_bytes("decision_keyed_aggressive", bits_aggr, extra_sub_shared + (lm3_plus_exact - org_bytes("lm_head", mass["lm_head"], 3)))

    report["allocation"] = {
        "label": "PROJECTED from MEASURED subspace rank + CITED codec formulas. Not a packed artifact.",
        "principle": "Protect the union of (w_top1-w_top2) at fragile positions in residual writes and lm_head rows. Cheap-quantize the orthogonal complement. Do not allocate by tensor RMS or |W| kurtosis unless that axis coincides with the decision span.",
        "n_exact_lm_head_rows": n_exact_rows,
        "exact_row_ids_n": n_exact_rows,
        "R95_m_lt_2_or_last": R95,
        "subspace_bytes_shared_U": extra_sub_shared,
        "subspace_bytes_per_layer_U": extra_sub_per,
        "subspace_bpw_shared": bpw_of(extra_sub_shared),
        "subspace_bpw_per_layer": bpw_of(extra_sub_per),
        "lm_head_exact_rows_bytes": 2 * lm_exact_elems,
        "lm_head_exact_rows_bpw": bpw_of(2 * lm_exact_elems),
        "metal_path": {
            "body": "Existing qwen_uniform_q4_group64_matvec_geo_tpr64_tg128 (or the 2/3-bit native sibling when packed). No expand-to-Q4-then-generic-GEMV.",
            "subspace_correction": "After the cheap GEMV, FMA a rank-R correction: y += U @ (V @ x). R is tens, so this is a tiny GEMV. U,V stay bf16 (or q8) in threadgroup. Discard V@x after the FMA.",
            "lm_head_hybrid": "Same Q3/Q4 GEMV over the body rows; add a dense bf16 gather-GEMV over the exact-row index list (n ~ hundreds). Argmax over the union. Do not materialize a full bf16 lm_head.",
            "reject": "packed-then-expand-to-Q4-then-generic-GEMV unless a GPU lane measures expansion still wins (this lane did not run GPU).",
        },
        "classification": {
            "ESSENTIAL": "decision subspace of residual writes (rank R95 of {w1-w2} on fragile positions); exact lm_head rows for specials + observed top-2; last-layer residual path at last-token / think / tool delims",
            "CONDITIONAL": "tool-call / think-close / stop rows of lm_head (exact, cheap); runtime refine-if-small-margin is CONDITIONAL compute, not stored bits",
            "COMPENSATABLE": "early-layer MLP (doctor L0 1-bit vacuous hold; this capture's early |cos| with d is low)",
            "PREDICTABLE": "chat-template positions with huge margin (im_start, role words) — error cannot flip",
            "SHARED": "one decision basis U across residual writes if per-layer U does not pay for itself",
            "REDUNDANT": "residual channels high in |h| but near-zero in |d| (test: 3994 if ranks diverge)",
            "UNKNOWN": "heads (no mixer_x); tool-formatted prompts (absent from capture); confirmed final-norm logits",
        },
    }
    report["bpw"]["g0_catalog_bytes"] = G0_BYTES
    report["bpw"]["g0_catalog_bpw"] = G0_BPW
    report["bpw"]["formula_g0ish"] = {"bytes": sc_g0["bytes"], "complete_bpw": sc_g0["complete_bpw"], "label": "DERIVED formula, not catalog"}
    report["bpw"]["decision_keyed"] = {
        "bytes": sc_dec["bytes"],
        "complete_bpw": sc_dec["complete_bpw"],
        "label": "PROJECTED",
        "bits": {k: v["bits"] for k, v in sc_dec["detail"].items()},
        "extra_shared_subspace_plus_lm_upgrade": extra_sub_shared + (lm3_plus_exact - org_bytes("lm_head", mass["lm_head"], 3)),
    }
    report["bpw"]["decision_keyed_perlayerU"] = {
        "bytes": sc_dec_per["bytes"],
        "complete_bpw": sc_dec_per["complete_bpw"],
        "label": "PROJECTED",
    }
    report["bpw"]["decision_keyed_aggressive"] = {
        "bytes": sc_aggr["bytes"],
        "complete_bpw": sc_aggr["complete_bpw"],
        "label": "PROJECTED",
        "bits": {k: v["bits"] for k, v in sc_aggr["detail"].items()},
    }
    report["bpw"]["delta_vs_g0_catalog"] = {
        "decision_keyed_minus_g0": sc_dec["complete_bpw"] - G0_BPW,
        "aggressive_minus_g0": sc_aggr["complete_bpw"] - G0_BPW,
    }
    # keep detail out of the huge file? include bit tables only
    report["bpw"]["decision_keyed_detail_bytes"] = {k: v["bytes"] for k, v in sc_dec["detail"].items()}

    report["rss_gb_peak"] = rss_gb()
    report["wall_s"] = time.perf_counter() - t0
    report["labels"] = {
        "MEASURED": "computed this process from on-disk tensors",
        "MEASURED_PROXY": "computed from L63 captured hidden x lm_head; site not confirmed final-norm",
        "DERIVED": "exact arithmetic on MEASURED",
        "PROJECTED": "codec-formula complete BPW of an unbuilt pack",
        "CITED": "prior g1 report / receipt, not re-run",
        "UNMEASURED": "absent capture or forbidden generate",
        "POLICY": "proposed gate, not a measured threshold",
    }

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    log(f"wrote {OUT} wall={report['wall_s']:.1f}s peak_rss={report['rss_gb_peak']:.3f}G")
    # compact stdout summary
    print("==== SUMMARY ====")
    print("payload_sha_match", payload_sha == meta["sha256_self"], payload_sha)
    print("site_corr_in_ln", report["site"]["corr_hidden_rms_vs_input_ln_rms"])
    print("site_verdict", report["site"]["verdict"])
    print("teacher_acc", report["margins"]["teacher_top1_acc_all_tf"])
    print("teacher_mean_rank", report["margins"]["teacher_mean_rank_tf"])
    print("margin_all_median", report["margins"]["all"]["margin_median"])
    print("frac_m<1", report["margins"]["all"]["frac_m_lt_1"])
    print("frac_m<2", report["margins"]["all"]["frac_m_lt_2"])
    print("last preds", report["margins"]["last_token_predictions"])
    print("3994 ranks h/d/contrib", report["attribution"]["rank_of_3994_by_abs_h_all"],
          report["attribution"]["rank_of_3994_by_abs_d_all"],
          report["attribution"]["rank_of_3994_by_abs_contrib_all"])
    print("q4 flip all/last", report["q4_probe"]["all"]["argmax_flip_rate"], report["q4_probe"]["last"]["argmax_flip_rate"])
    print("MPS", report["acceptance"]["q4_lm_head_baseline"])
    print("subspace", report["attribution"].get("subspace"))
    print("bpw decision_keyed", report["bpw"]["decision_keyed"]["complete_bpw"], "aggr", report["bpw"]["decision_keyed_aggressive"]["complete_bpw"])
    print("cited last match", report["screen_overlap"]["this_run_last_pred_match_cited"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
