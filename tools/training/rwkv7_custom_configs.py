"""Custom RWKV-7 draft-model configs for spec-decode experiments.

The variants are sized for Apple Silicon/Metal geometry:
  * n_embd and n_ff are multiples of 256 for Q4_K GEMV block alignment.
  * RWKV-7 head_dim stays fixed at 64.
  * LoRA ranks are scaled from the 0.4B defaults and rounded to 16.
"""

from __future__ import annotations

try:
    from .rwkv7_torch_model import RWKV7Config
except ImportError:  # scripts are usually run from tools/training/
    from rwkv7_torch_model import RWKV7Config


VOCAB_SIZE = 65536
VARIANT_ORDER = ("draft_100m", "draft_150m", "draft_200m", "draft_300m")


CUSTOM_VARIANTS: dict[str, RWKV7Config] = {
    "draft_100m": RWKV7Config(
        n_embd=512,
        n_layer=10,
        n_ff=2048,
        head_dim=64,
        n_head=8,
        vocab_size=VOCAB_SIZE,
        decay_lora=32,
        iclr_lora=32,
        value_res_lora=16,
        gate_lora=64,
    ),
    "draft_150m": RWKV7Config(
        n_embd=768,
        n_layer=12,
        n_ff=1792,
        head_dim=64,
        n_head=12,
        vocab_size=VOCAB_SIZE,
        decay_lora=48,
        iclr_lora=48,
        value_res_lora=32,
        gate_lora=96,
    ),
    "draft_200m": RWKV7Config(
        n_embd=768,
        n_layer=17,
        n_ff=2048,
        head_dim=64,
        n_head=12,
        vocab_size=VOCAB_SIZE,
        decay_lora=48,
        iclr_lora=48,
        value_res_lora=32,
        gate_lora=96,
    ),
    "draft_300m": RWKV7Config(
        n_embd=1024,
        n_layer=15,
        n_ff=3072,
        head_dim=64,
        n_head=16,
        vocab_size=VOCAB_SIZE,
        decay_lora=64,
        iclr_lora=64,
        value_res_lora=32,
        gate_lora=128,
    ),
}


def estimated_params(cfg: RWKV7Config) -> int:
    """Return the sizing-formula parameter count from the training brief."""
    lora_sum = cfg.decay_lora + cfg.iclr_lora + cfg.value_res_lora + cfg.gate_lora
    layer_params = (
        4 * cfg.n_embd * cfg.n_embd
        + 2 * cfg.n_ff * cfg.n_embd
        + lora_sum * cfg.n_embd * 2
        + 9 * cfg.n_embd
    )
    return cfg.vocab_size * cfg.n_embd + cfg.n_layer * layer_params + cfg.vocab_size * cfg.n_embd


def q4k_bytes_per_forward(cfg: RWKV7Config) -> float:
    weights = (4 * cfg.n_embd * cfg.n_embd + 2 * cfg.n_ff * cfg.n_embd) * cfg.n_layer
    return weights * (4.5 / 32.0)


def theoretical_tok_s(cfg: RWKV7Config, bandwidth_bytes_s: float = 150e9) -> float:
    return bandwidth_bytes_s / q4k_bytes_per_forward(cfg)


def _assert_config(name: str, cfg: RWKV7Config) -> None:
    assert cfg.n_embd % 256 == 0, f"{name}: n_embd must be 256-aligned"
    assert cfg.n_ff % 256 == 0, f"{name}: n_ff must be 256-aligned"
    assert cfg.head_dim == 64, f"{name}: RWKV-7 head_dim must stay 64"
    assert cfg.n_head == cfg.n_embd // cfg.head_dim, f"{name}: n_head must be n_embd / 64"
    assert cfg.n_embd % cfg.head_dim == 0, f"{name}: n_embd must divide into heads"
    assert cfg.decay_lora >= 16 and cfg.decay_lora % 16 == 0, f"{name}: bad decay_lora"
    assert cfg.iclr_lora >= 16 and cfg.iclr_lora % 16 == 0, f"{name}: bad iclr_lora"
    assert cfg.value_res_lora >= 16 and cfg.value_res_lora % 16 == 0, f"{name}: bad value_res_lora"
    assert cfg.gate_lora >= 16 and cfg.gate_lora % 16 == 0, f"{name}: bad gate_lora"
    assert cfg.vocab_size == VOCAB_SIZE, f"{name}: vocab size must remain {VOCAB_SIZE}"


for _name, _cfg in CUSTOM_VARIANTS.items():
    _assert_config(_name, _cfg)

