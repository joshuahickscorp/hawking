#!/usr/bin/env python3
"""CPU smoke for the optimized RWKV-7 post-train trainer mechanics.

This is a **CPU-only** validation (NO GPU, NO mps) of the throughput/memory
levers the optimized runbook prescribes for the torch path:

  1. bf16 autocast forward + backward actually runs and the loss DECREASES
     (proves the bf16-where-supported lever is mechanically sound on this
     torch build), then
  2. gradient ACCUMULATION over micro-steps produces the same parameter update
     as a single large batch (proves effective-batch via accumulation is
     correct — the lever that lets us raise effective batch without raising
     resident memory), and
  3. gradient CHECKPOINTING runs a forward/backward without error and yields the
     same loss as the non-checkpointed forward (proves the OOM-guard lever is
     wired and lossless).

It deliberately uses a TINY synthetic causal-LM (a few-layer GPT-2-shaped
`transformers` model) rather than the 0.4B fla RWKV-7, because:
  * the point is to validate the *training loop mechanics / levers*, which are
    model-agnostic, on CPU in seconds — not to train RWKV-7 (that needs the GPU,
    deferred), and
  * it avoids the ~900 MB fla-hub download + the `flash-linear-attention`
    dependency on the smoke path. `rwkv7_train_smoke.py --fla` opts into the
    real RWKV-7 forward/backward smoke if `fla` + the HF model are present (also
    CPU, a few steps) for those who want the model-specific check.

Run (CPU, seconds):
    python3.12 tools/training/rwkv7_train_smoke.py
    python3.12 tools/training/rwkv7_train_smoke.py --fla   # if fla+model present

Exit code 0 = all checks passed. Non-zero = a lever is broken on this stack.
"""
from __future__ import annotations

import argparse
import sys


def _banner(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def smoke_tiny(steps: int = 6) -> bool:
    """Validate bf16 autocast + grad-accum + grad-checkpointing on CPU."""
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    # Force CPU — this smoke must never touch mps/GPU (the GPU is reserved for
    # the concurrent perf bench).
    device = torch.device("cpu")

    # Tiny GPT-2-shaped LM via transformers, so the path mirrors the real
    # AutoModelForCausalLM training loop (same .forward(labels=...) -> .loss API).
    from transformers import GPT2Config, GPT2LMHeadModel

    # Dropout = 0 so forward passes are deterministic — required for the
    # grad-accum and checkpointing EQUALITY checks below (stochastic dropout
    # would make two forwards differ for reasons unrelated to the lever).
    cfg = GPT2Config(
        vocab_size=256,
        n_positions=64,
        n_embd=64,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(cfg).to(device)
    model.train()

    B, T = 4, 32
    ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    def manual_ce(m: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """Explicit next-token cross-entropy (mean over all shifted positions).

        We compute the loss ourselves rather than relying on the HF model's
        internal `labels=` reduction, so the grad-accumulation arithmetic below
        is exact and independent of which averaging convention the transformers
        version uses for `num_items_in_batch`.
        """
        logits = m(input_ids=input_ids).logits  # [b, t, v]
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = input_ids[:, 1:].reshape(-1)
        return nn.functional.cross_entropy(shift_logits, shift_labels, reduction="mean")

    # ── Check 1: bf16 autocast forward/backward runs and loss decreases ──────
    _banner("Check 1: bf16 autocast forward/backward, loss decreases")
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    first_loss = None
    last_loss = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        # bf16 autocast on CPU — the mechanically-identical call the optimized
        # trainer makes on mps where bf16 is supported. CPU bf16 autocast is a
        # supported torch path, so this proves the lever executes.
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = model(input_ids=ids, labels=ids)
            loss = out.loss
        loss.backward()
        opt.step()
        lv = float(loss.detach().float())
        if first_loss is None:
            first_loss = lv
        last_loss = lv
        print(f"  step {step}: loss={lv:.4f}")
    ok1 = last_loss is not None and first_loss is not None and last_loss < first_loss
    print(f"  -> loss {first_loss:.4f} -> {last_loss:.4f}  decreased={ok1}")

    # ── Check 2: grad accumulation == single large batch (parameter update) ──
    _banner("Check 2: grad-accumulation matches single large batch")
    # Two micro-batches of B/2 accumulated (no zero_grad between) must produce
    # the SAME averaged gradient as one batch of B, when each micro-loss is
    # scaled by 1/accum. This is exactly what HF Trainer does with
    # gradient_accumulation_steps. We compare the resulting grad on one weight.
    def grad_on_full() -> torch.Tensor:
        m = GPT2LMHeadModel(cfg)
        m.load_state_dict(model.state_dict())
        m.eval()  # deterministic (dropout is 0 anyway; be explicit)
        m.zero_grad(set_to_none=True)
        loss = manual_ce(m, ids)
        loss.backward()
        return m.transformer.wte.weight.grad.detach().clone()

    def grad_on_accum(accum: int = 2) -> torch.Tensor:
        m = GPT2LMHeadModel(cfg)
        m.load_state_dict(model.state_dict())
        m.eval()
        m.zero_grad(set_to_none=True)
        micro = B // accum
        for a in range(accum):
            chunk = ids[a * micro : (a + 1) * micro]
            # Scale each micro-loss by 1/accum so the accumulated SUM equals the
            # mean over the full batch. Exact because every micro-batch has the
            # same #positions (equal rows × equal T) — the standard HF Trainer
            # gradient_accumulation_steps contract.
            loss = manual_ce(m, chunk) / accum
            loss.backward()
        return m.transformer.wte.weight.grad.detach().clone()

    g_full = grad_on_full()
    g_acc = grad_on_accum(2)
    max_abs = float((g_full - g_acc).abs().max())
    ok2 = max_abs < 1e-4
    print(f"  -> max|grad_full - grad_accum| = {max_abs:.2e}  match={ok2}")

    # ── Check 3: gradient checkpointing runs and is lossless ─────────────────
    _banner("Check 3: gradient checkpointing runs, loss matches non-checkpointed")
    m = GPT2LMHeadModel(cfg)
    m.load_state_dict(model.state_dict())
    m.eval()  # deterministic forward (no stochastic dropout)
    m.config.use_cache = False
    # non-checkpointed loss
    loss_plain = float(manual_ce(m, ids).detach().float())
    # checkpointed loss — use_reentrant=False is the modern, mps-friendly mode.
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    loss_c = manual_ce(m, ids)
    loss_ckpt = float(loss_c.detach().float())
    loss_c.backward()  # must not error
    grad_ok = m.transformer.wte.weight.grad is not None
    ok3 = abs(loss_plain - loss_ckpt) < 1e-4 and grad_ok
    print(
        f"  -> loss_plain={loss_plain:.4f} loss_ckpt={loss_ckpt:.4f} "
        f"grad_present={grad_ok} lossless={ok3}"
    )

    all_ok = ok1 and ok2 and ok3
    _banner(f"tiny-smoke result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def smoke_fla(steps: int = 3) -> bool:
    """Optional: a few CPU forward/backward steps on the REAL fla RWKV-7 model.

    Requires `flash-linear-attention` + a local HF checkpoint (default
    models/rwkv7-g1-04-hf). CPU only, a handful of steps — proves the actual
    RWKV-7 forward/backward differentiates on this stack. Skips with a clear
    message (return True) if fla or the model is absent, so CI stays green.
    """
    import importlib.util
    import os

    if importlib.util.find_spec("fla") is None:
        print("[fla-smoke] flash-linear-attention not installed; skipping "
              "(install it + the HF model to run the model-specific smoke).")
        return True
    model_dir = os.environ.get("RWKV7_HF_DIR", "models/rwkv7-g1-04-hf")
    if not os.path.isdir(model_dir):
        print(f"[fla-smoke] model dir {model_dir} absent; skipping. "
              f"Fetch with: huggingface-cli download fla-hub/rwkv7-0.4B-g1 "
              f"--local-dir {model_dir}")
        return True

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _banner(f"fla RWKV-7 CPU smoke: {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # fp32 on CPU (the runbook trains fp32 for RWKV-7 stability).
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=torch.float32
    ).to("cpu")
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    text = "<|rwkv_tokenizer_end_of_text|>User: Say hi.\n\nAssistant: Hi!\n\n"
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"][:, :64]

    first = last = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        lv = float(out.loss.detach())
        first = lv if first is None else first
        last = lv
        print(f"  step {step}: loss={lv:.4f}")
    ok = last is not None and last <= first + 1e-3  # should not diverge
    print(f"  -> RWKV-7 differentiates on CPU; loss {first:.4f} -> {last:.4f} ok={ok}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=6, help="tiny-smoke optimizer steps")
    ap.add_argument("--fla", action="store_true",
                    help="also run the real fla RWKV-7 CPU smoke (needs fla + model)")
    args = ap.parse_args()

    ok = smoke_tiny(args.steps)
    if args.fla:
        ok = smoke_fla() and ok
    if not ok:
        print("\nSMOKE FAILED — a trainer lever is broken on this stack.", file=sys.stderr)
        return 1
    print("\nALL SMOKE CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
