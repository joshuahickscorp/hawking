# strand_eval.qat_shim — drop-in eval for scripts/strand-qat.py (copy #3 retires).
#
# strand-qat.py is owned by the selective-PV agent and is NOT edited here. When
# that agent is ready, the swap is two lines:
#
#     sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
#     from strand_eval.qat_shim import eval_ppl, eval_chunk_list, record_qat_eval
#
# `eval_ppl(model, eval_ch, device, tag="")` matches strand-qat.py's in-memory
# signature exactly (same return: the float ppl; same prints prefix "[qat]"),
# but the loop is the ONE canonical sum-CE loop from strand_eval.core — so QAT
# mid-evals become canon-comparable BY CONSTRUCTION instead of by-hand agreement.
#
# MPS notes preserved from the original: empty_cache before/after (defragment —
# eval needs contiguous GBs next to AdamW state), 512-row CE slices (full-vocab
# log_softmax over 2047x152k rows is a ~2.5GB transient), defrag+retry on OOM.

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from strand_eval import default_ledger_path
from strand_eval.core import build_record, eval_chunks
from strand_eval.ledger import append_record


def eval_ppl(model, eval_ch, device, tag=""):
    """Signature-compatible with strand-qat.py's eval_ppl. Returns the float ppl."""
    import torch
    model.eval()
    if device == "mps":
        torch.mps.empty_cache()
    t0 = time.time()
    ce_slice = 512 if device == "mps" else 0
    nll, ntok = eval_chunks(model, eval_ch, torch.device(device),
                            ce_slice=ce_slice, progress=False)
    ppl = math.exp(nll / ntok)
    print(f"[qat] eval{(' ' + tag) if tag else ''}: ppl={ppl:.4f}  "
          f"({len(eval_ch)} ch, {ntok} tok, {time.time()-t0:.0f}s)", flush=True)
    return ppl


# Alias for callers that want the canonical name.
eval_chunk_list = eval_ppl


def record_qat_eval(*, model_path, tag, ppl, ctx, chunks, tokens, device, dtype,
                    out_dir=None, ledger_path=None, extra=None):
    """Optional: give a QAT mid-eval a full canonical record + ledger line.

    QAT evals share the canon tokenization/windowing, so the harness_key makes
    them mechanically comparable to PTQ numbers at the same (device, dtype, ctx,
    chunks). dataset_id is the canon config (the shim assumes the strand-qat
    wikitext chain, which is the same chain)."""
    rec = build_record(
        model_path=model_path, tag=tag, ppl=ppl, ctx=ctx, chunks=chunks,
        tokens=tokens, device_resolved=device, device_mode=device, dtype=dtype,
        dataset_id="wikitext-2-raw-v1", dataset_fp=None,
        extra=dict({"provenance": "qat-mid-eval"}, **(extra or {})))
    if out_dir:
        import json
        from strand_eval.core import output_path
        out_json = output_path(out_dir, rec["model"], tag, model_path)
        rec["out_json"] = out_json
        with open(out_json, "w") as f:
            json.dump(rec, f, indent=2)
    append_record(ledger_path or default_ledger_path(), rec)
    return rec
