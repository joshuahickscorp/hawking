"""Diagnostic: per-step greedy rollout for KD draft anomaly.

Reuses load_student + load_rwkv7 (teacher) from the eval module. For each
student variant AND the teacher, runs the SAME 4-step greedy rollout that
greedy_accept_rate uses (context advances by the STUDENT's own token), and
prints per step: student_token, teacher_token, match, plus a collapse flag if
the student repeats a single token.

STRICT MEMORY SAFETY: exactly one model in memory at a time. We pre-decode the
teacher reference rollout per prompt (teacher's own greedy continuation) and the
teacher-argmax-on-student-context decision, but to keep only one model resident
we do TWO passes:
  Pass A (teacher): for each prompt, record teacher argmax at the prompt and at
    each student-advanced context. But student tokens differ per variant, so we
    cannot precompute teacher decisions independent of the student.

  => Instead we load BOTH the student and teacher for a variant, but free both
     before the next variant. Peak = student + teacher (0.4B). The teacher 0.4B
     fp32 ~ 1.5GB; the largest student 75M fp32 ~ 0.3GB. On CPU this is safe.
     We reload the teacher per variant to guarantee a clean free between variants.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from rwkv7_draft_ppl_eval import (
    load_student,
    read_sft_prompts,
    resolve_checkpoint,
)
from rwkv7_eval_ppl import load_tokenizer
from rwkv7_load_weights import load_rwkv7
from rwkv7_torch_model import RWKV7Config

DEVICE = "cpu"
DRAFT_K = 4
TEACHER_PATH = str(ROOT / "models/rwkv7-g1-04-hf/model.safetensors")
HF_DIR = str(ROOT / "models/rwkv7-g1-04-hf")
DATA = ROOT / "artifacts/rwkv7_posttrain/sft.jsonl"

VARIANTS = [
    ("draft_26m_probe", ROOT / "artifacts/lowbit_rwkv7/runs/kd_draft_26m_probe/final"),
    ("draft_35m_probe", ROOT / "artifacts/lowbit_rwkv7/runs/kd_draft_35m_probe/final"),
    ("draft_75m_probe", ROOT / "artifacts/lowbit_rwkv7/runs/kd_draft_75m_probe/final"),
]


def free(*models):
    for m in models:
        del m
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


@torch.no_grad()
def greedy_argmax(model, context):
    x = torch.tensor([context], dtype=torch.long, device=DEVICE)
    return int(model(x)[0, -1].argmax().item())


@torch.no_grad()
def rollout(student, teacher, prompt, decode):
    """Replicate greedy_accept_rate exactly: context advances by STUDENT token.
    Also track the TEACHER's own greedy continuation (teacher token advancing by
    teacher token) to see if the teacher path itself is stable."""
    context = list(prompt)
    teacher_self_ctx = list(prompt)
    student_tokens = []
    steps = []
    matched = 0
    for step in range(DRAFT_K):
        s_tok = greedy_argmax(student, context)        # student on shared (student-advanced) context
        t_tok = greedy_argmax(teacher, context)        # teacher on the SAME context (the accept metric)
        t_self = greedy_argmax(teacher, teacher_self_ctx)  # teacher's own greedy chain
        m = s_tok == t_tok
        matched += int(m)
        steps.append((step + 1, s_tok, t_tok, m, t_self))
        student_tokens.append(s_tok)
        context.append(s_tok)
        teacher_self_ctx.append(t_self)

    # collapse detection on student tokens
    uniq = set(student_tokens)
    collapsed = len(uniq) == 1 and len(student_tokens) > 1
    near_collapse = len(uniq) <= 2 and len(student_tokens) >= 4

    print(f"    prompt_len={len(prompt)}  student_tokens={student_tokens}  "
          f"unique={len(uniq)}  collapsed={collapsed}  near_collapse={near_collapse}")
    for (st, s_tok, t_tok, m, t_self) in steps:
        s_txt = repr(decode([s_tok]))
        t_txt = repr(decode([t_tok]))
        print(f"      step{st}: student={s_tok:<6} ({s_txt:<10}) "
              f"teacher={t_tok:<6} ({t_txt:<10}) match={m}  teacher_self={t_self}")
    return matched, collapsed


def main():
    print("[diag] loading tokenizer", file=sys.stderr, flush=True)
    encode = load_tokenizer(HF_DIR)

    # Decode via the RWKV greedy-trie tokenizer directly (avoids triton import
    # that AutoTokenizer+trust_remote_code triggers).
    import importlib.util as _ilu

    tok_script = Path(HF_DIR) / "hf_rwkv_tokenizer.py"
    vocab = Path(HF_DIR) / "rwkv_vocab_v20230424.txt"
    _spec = _ilu.spec_from_file_location("hf_rwkv_tokenizer", str(tok_script))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _rwkv_tok = _mod.RWKV_TOKENIZER(str(vocab))

    def decode(ids):
        try:
            return _rwkv_tok.decodeBytes(ids).decode("utf-8", errors="replace")
        except Exception:
            return "<?>"

    prompts = read_sft_prompts(DATA, encode, limit=3, max_length=128)
    print(f"[diag] {len(prompts)} prompts, lens={[len(p) for p in prompts]}", file=sys.stderr, flush=True)

    for variant, ckpt_dir in VARIANTS:
        ckpt = resolve_checkpoint(ckpt_dir)
        print(f"\n========== {variant} ({ckpt}) ==========", flush=True)
        print(f"[diag] loading student {variant}", file=sys.stderr, flush=True)
        student = load_student(variant, ckpt, DEVICE, use_chunked=False, chunk_size=32)
        n_layer = student.cfg.n_layer
        n_embd = student.cfg.n_embd
        params_m = sum(p.numel() for p in student.parameters()) / 1e6
        print(f"[diag] arch: n_layer={n_layer} n_embd={n_embd} params={params_m:.2f}M", flush=True)

        print(f"[diag] loading teacher 0.4B", file=sys.stderr, flush=True)
        teacher = load_rwkv7(TEACHER_PATH, RWKV7Config(), device=DEVICE, dtype=torch.float32)
        teacher.eval()

        total_match = 0
        total_steps = 0
        n_collapsed = 0
        for i, prompt in enumerate(prompts):
            print(f"  --- prompt {i} ---", flush=True)
            matched, collapsed = rollout(student, teacher, prompt, decode)
            total_match += matched
            total_steps += DRAFT_K
            n_collapsed += int(collapsed)
        acc = total_match / total_steps if total_steps else 0.0
        print(f"  >>> {variant}: accept={acc:.4f} ({total_match}/{total_steps})  "
              f"collapsed_prompts={n_collapsed}/{len(prompts)}", flush=True)

        free(student, teacher)
        print(f"[diag] freed {variant} + teacher", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
