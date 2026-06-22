"""Focused step-1 diagnostic.

For each student variant, on the REAL prompt (no rollout drift), check:
  - student top-5 vs teacher top-5 at the final position
  - is the teacher's argmax in the student's top-20? (sanity: not garbage logits)
  - student logit entropy + max prob (is the student confident/peaked or flat?)
  - per-step accept split: step-1 vs steps 2-4

One model in memory at a time.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from rwkv7_draft_ppl_eval import load_student, read_sft_prompts, resolve_checkpoint
from rwkv7_eval_ppl import load_tokenizer
from rwkv7_load_weights import load_rwkv7
from rwkv7_torch_model import RWKV7Config

DEVICE = "cpu"
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
def last_logits(model, ctx):
    x = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
    return model(x)[0, -1].float()


def topk(logits, k, decode):
    p = F.softmax(logits, dim=-1)
    vals, idx = p.topk(k)
    return [(int(i), round(float(v), 4), repr(decode([int(i)]))) for v, i in zip(vals, idx)]


def entropy(logits):
    p = F.softmax(logits, dim=-1)
    return float(-(p * (p + 1e-12).log()).sum())


def main():
    encode = load_tokenizer(HF_DIR)
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location("hf_rwkv_tokenizer", str(Path(HF_DIR) / "hf_rwkv_tokenizer.py"))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rwkv_tok = mod.RWKV_TOKENIZER(str(Path(HF_DIR) / "rwkv_vocab_v20230424.txt"))

    def decode(ids):
        try:
            return rwkv_tok.decodeBytes(ids).decode("utf-8", errors="replace")
        except Exception:
            return "<?>"

    prompts = read_sft_prompts(DATA, encode, limit=3, max_length=128)

    # teacher reference: argmax + top5 at each REAL prompt (load teacher once, free).
    print("[diag] loading teacher once for reference", file=sys.stderr, flush=True)
    teacher = load_rwkv7(TEACHER_PATH, RWKV7Config(), device=DEVICE, dtype=torch.float32)
    teacher.eval()
    teacher_ref = []
    for p in prompts:
        lg = last_logits(teacher, p)
        teacher_ref.append({
            "argmax": int(lg.argmax()),
            "top5": topk(lg, 5, decode),
            "logits": lg.clone(),  # keep to rank student tokens under teacher
        })
    free(teacher)
    print("[diag] freed teacher", file=sys.stderr, flush=True)

    for variant, ckpt_dir in VARIANTS:
        ckpt = resolve_checkpoint(ckpt_dir)
        student = load_student(variant, ckpt, DEVICE, use_chunked=False, chunk_size=32)
        print(f"\n===== {variant}  n_layer={student.cfg.n_layer} n_embd={student.cfg.n_embd} =====", flush=True)
        for i, p in enumerate(prompts):
            lg = last_logits(student, p)
            s_arg = int(lg.argmax())
            t_ref = teacher_ref[i]
            t_arg = t_ref["argmax"]
            # rank of teacher argmax in student distribution
            order = torch.argsort(lg, descending=True)
            t_rank_in_student = int((order == t_arg).nonzero(as_tuple=True)[0].item())
            # rank of student argmax in teacher distribution
            torder = torch.argsort(t_ref["logits"], descending=True)
            s_rank_in_teacher = int((torder == s_arg).nonzero(as_tuple=True)[0].item())
            print(f"  prompt {i}: student_argmax={s_arg} ({decode([s_arg])!r})  "
                  f"teacher_argmax={t_arg} ({decode([t_arg])!r})  match={s_arg==t_arg}")
            print(f"    student_top5  = {topk(lg, 5, decode)}")
            print(f"    teacher_top5  = {t_ref['top5']}")
            print(f"    student_entropy={entropy(lg):.3f}  student_maxprob={float(F.softmax(lg,-1).max()):.4f}")
            print(f"    teacher_argmax rank in STUDENT dist = {t_rank_in_student}  "
                  f"| student_argmax rank in TEACHER dist = {s_rank_in_teacher}")
        free(student)
        print(f"[diag] freed {variant}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
