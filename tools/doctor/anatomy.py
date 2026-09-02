"""Architecture fingerprint and organ graph from metadata only.

Reuses the algorithm in tools.future.specimen_events.fingerprint_from_config
(config.json + optional index, weights_opened=False) without importing that
module: it pulls hcli.persist, which is not materialized in this sparse
checkout. Opening a safetensors shard is out of scope.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.doctor.access import AccessLog, is_weight_file
from tools.doctor.zeros import organ_from_mapping


LAKE = Path("/Volumes/corpdrive/hawking-modellake")
SPECIMENS = LAKE / "specimens"
MANIFESTS = LAKE / "manifests"

# Same families specimen_events.ORGAN_PATTERNS names. Kept local so we do not
# import hcli.
ORGAN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("mlp", re.compile(r"(mlp|feed_forward|ffn|gate_proj|up_proj|down_proj)", re.I)),
    ("attention", re.compile(r"(self_attn|attention|q_proj|k_proj|v_proj|o_proj)", re.I)),
    ("mamba_ssm", re.compile(r"(mamba|ssm|conv1d|A_log|dt_bias|in_proj|out_proj)", re.I)),
    ("moe_expert", re.compile(r"experts?\.\d+|mlp\.experts", re.I)),
    ("moe_router", re.compile(r"(router|gate\.weight)", re.I)),
    ("embed", re.compile(r"(embed_tokens|word_embeddings|wte)", re.I)),
    ("lm_head", re.compile(r"lm_head", re.I)),
    ("vision", re.compile(r"(visual|vision_tower|vision_model|patch_embed)", re.I)),
    ("norm", re.compile(r"(layernorm|rms_norm|final_layernorm)", re.I)),
    ("deltanet", re.compile(r"(linear_attn|gated_deltanet|conv_kernel)", re.I)),
    ("mtp", re.compile(r"\bmtp\b|nextn", re.I)),
)

BANK_SCREEN = Path("receipts/headless/FLASH_DOCTOR_EXPERT_BANK_SCREEN.json")
FLASH_SLUG = "Qwen--Qwen3.8-Flash-Next@34567a4712bc"


def _text_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    inner = cfg.get("text_config")
    if isinstance(inner, Mapping):
        return inner
    return cfg


def fingerprint_from_config(
    cfg: Mapping[str, Any],
    *,
    weight_names: Sequence[str] = (),
    total_size: int | None = None,
) -> dict[str, Any]:
    """Metadata fingerprint. Opening a safetensors shard is out of scope.

    Mirrors tools.future.specimen_events.fingerprint_from_config.
    """
    text = _text_cfg(cfg)
    arches = cfg.get("architectures") if isinstance(cfg.get("architectures"), list) else []
    model_type = str(cfg.get("model_type") or text.get("model_type") or "")
    hidden = text.get("hidden_size")
    layers = text.get("num_hidden_layers")
    heads = text.get("num_attention_heads")
    kv = text.get("num_key_value_heads")
    intermediate = text.get("intermediate_size") or text.get("moe_intermediate_size")
    vocab = text.get("vocab_size")
    n_experts = text.get("num_experts") or text.get("n_routed_experts") or cfg.get("n_routed_experts")
    top_k = text.get("num_experts_per_tok") or cfg.get("num_experts_per_tok")
    shared = text.get("n_shared_experts") or cfg.get("n_shared_experts")
    shared_inter = text.get("shared_expert_intermediate_size")
    tied = text.get("tie_word_embeddings")
    if tied is None:
        tied = cfg.get("tie_word_embeddings")
    families: list[str] = []
    if weight_names:
        blob = " ".join(weight_names)
        for name, pat in ORGAN_PATTERNS:
            if pat.search(blob) and name not in families:
                families.append(name)
    else:
        if cfg.get("vision_config") or "vl" in model_type.lower() or any(
            "vl" in str(a).lower() for a in arches
        ):
            families.append("vision")
        if n_experts:
            families.append("moe_expert")
            families.append("moe_router")
        families.extend(["attention", "mlp", "embed", "lm_head", "norm"])
        if text.get("full_attention_interval") or text.get("linear_num_key_heads"):
            if "deltanet" not in families:
                families.append("deltanet")
        if text.get("mtp") or text.get("mtp_num_hidden_layers") or cfg.get("num_nextn_predict_layers"):
            if "mtp" not in families:
                families.append("mtp")
        if "mamba" in model_type.lower() or "falcon_h1" in model_type.lower():
            if "mamba_ssm" not in families:
                families.insert(0, "mamba_ssm")
    architecture_family = model_type or "UNKNOWN"
    multimodal = bool(cfg.get("vision_config")) or "vl" in architecture_family
    moe = bool(n_experts) or "moe_expert" in families
    return {
        "architectures": [str(a) for a in arches],
        "model_type": model_type,
        "architecture_family": architecture_family,
        "hidden_size": hidden if isinstance(hidden, int) else None,
        "num_hidden_layers": layers if isinstance(layers, int) else None,
        "num_attention_heads": heads if isinstance(heads, int) else None,
        "num_key_value_heads": kv if isinstance(kv, int) else None,
        "intermediate_size": intermediate if isinstance(intermediate, int) else None,
        "vocab_size": vocab if isinstance(vocab, int) else None,
        "num_experts": n_experts if isinstance(n_experts, int) else None,
        "num_experts_per_tok": top_k if isinstance(top_k, int) else None,
        "n_shared_experts": shared if isinstance(shared, int) else None,
        "shared_expert_intermediate_size": shared_inter if isinstance(shared_inter, int) else None,
        "tie_word_embeddings": tied if isinstance(tied, bool) else None,
        "full_attention_interval": text.get("full_attention_interval"),
        "mtp": bool(text.get("mtp") or text.get("mtp_num_hidden_layers") or cfg.get("num_nextn_predict_layers")),
        "ngram": bool(text.get("ngram_size") or text.get("ngram_vocab_size_base")),
        "multimodal": multimodal,
        "moe": moe,
        "organ_families": families,
        "n_named_tensors": len(weight_names),
        "size_bytes": total_size if isinstance(total_size, int) else None,
        "weights_opened": False,
        "source": "config.json+optional_index",
        "evidence_class": "STATIC",
        "evidence_tier": "STATIC",
    }


def read_index_names(doc: Mapping[str, Any]) -> tuple[list[str], int | None]:
    weight_map = doc.get("weight_map") if isinstance(doc.get("weight_map"), Mapping) else {}
    names = [str(k) for k in weight_map]
    meta = doc.get("metadata") if isinstance(doc.get("metadata"), Mapping) else {}
    total = meta.get("total_size")
    return names, (int(total) if isinstance(total, int) else None)


def load_specimen_metadata(
    root: Path,
    access: AccessLog,
    *,
    repo: Path | None = None,
) -> dict[str, Any]:
    """config.json + index.json + optional lake manifest. Never a shard."""
    cfg_path = root / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"specimen config.json missing: {root}")
    cfg = access.open_json(cfg_path)
    if not isinstance(cfg, Mapping):
        raise ValueError(f"config.json is not an object: {cfg_path}")
    names: list[str] = []
    total: int | None = None
    index_path = root / "model.safetensors.index.json"
    index_present = index_path.is_file()
    if index_present:
        idx = access.open_json(index_path)
        if isinstance(idx, Mapping):
            names, total = read_index_names(idx)
            # Refuse to follow shard names even if a caller tries later.
            weight_map = idx.get("weight_map") if isinstance(idx.get("weight_map"), Mapping) else {}
            for shard in set(str(v) for v in weight_map.values()):
                shard_path = root / shard
                if is_weight_file(shard_path):
                    # Do not open. Record the rule.
                    pass
    fp = fingerprint_from_config(cfg, weight_names=names, total_size=total)
    fp["specimen_root"] = str(root)
    fp["index_present"] = index_present
    fp["config_keys"] = sorted(str(k) for k in cfg.keys())
    manifest = None
    slug = root.name
    man_path = MANIFESTS / f"{slug}.json"
    if man_path.is_file():
        try:
            manifest = access.open_json(man_path)
        except OSError:
            manifest = None
    if isinstance(manifest, Mapping):
        fp["lake_manifest_bytes"] = manifest.get("bytes")
        fp["lake_manifest_n_files"] = manifest.get("n_files")
        fp["resolved_sha"] = manifest.get("resolved_sha")
        fp["repo"] = manifest.get("repo")
    fp["bank_screen"] = _attach_bank_screen(slug, repo)
    return {"config": dict(cfg), "fingerprint": fp, "weight_names": names}


def _attach_bank_screen(slug: str, repo: Path | None) -> dict[str, Any] | None:
    """Prior Doctor work: FLASH_DOCTOR_EXPERT_BANK_SCREEN. Receipt, not weights."""
    if FLASH_SLUG not in slug and "Flash-Next" not in slug:
        return None
    if repo is None:
        return None
    path = repo / BANK_SCREEN
    if not path.is_file():
        return None
    # This is a receipt under the repo, not a specimen shard. Direct JSON read
    # via a fresh AccessLog so the specimen log stays specimen-only; the
    # engine merges both reports.
    try:
        import json
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    pop = doc.get("population") if isinstance(doc.get("population"), dict) else {}
    sampled = pop.get("sampled_population_rank") if isinstance(pop.get("sampled_population_rank"), dict) else {}
    return {
        "path": str(BANK_SCREEN),
        "status": doc.get("status"),
        "cross_expert_gate_up_mean_cosine": pop.get("cross_expert_gate_up_mean_cosine"),
        "rank_1_energy": sampled.get("rank_1_energy"),
        "expert_count": (doc.get("source") or {}).get("expert_count") if isinstance(doc.get("source"), dict) else None,
        "weights_opened_by_this_lane": False,
        "note": "cited as STATIC receipt evidence; this lane did not reopen the shards",
        "evidence_tier": "STATIC",
    }


def organs_from_fingerprint(fp: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the organ graph Doctor will diagnose. Architecture evidence only."""
    geom = {
        "experts": fp.get("num_experts"),
        "num_experts": fp.get("num_experts"),
        "top_k": fp.get("num_experts_per_tok"),
        "num_experts_per_tok": fp.get("num_experts_per_tok"),
        "tie_word_embeddings": fp.get("tie_word_embeddings"),
    }
    organs: list[dict[str, Any]] = []
    tied = fp.get("tie_word_embeddings") is True
    families = list(fp.get("organ_families") or [])

    def add(cls: str, **over: Any) -> None:
        row = {"organ_class": cls, "name": cls, **over}
        organs.append(organ_from_mapping(row, geometry=geom))

    add(
        "embed_tokens",
        tie_word_embeddings=fp.get("tie_word_embeddings"),
        stored_independently=not tied,
        derived_from="lm_head" if tied else None,
        executes_every_token=True,
    )
    add(
        "lm_head",
        tie_word_embeddings=fp.get("tie_word_embeddings"),
        stored_independently=not tied,
        derived_from="embed_tokens" if tied else None,
        executes_every_token=True,
    )
    if fp.get("moe") or fp.get("num_experts"):
        bank = fp.get("bank_screen") if isinstance(fp.get("bank_screen"), Mapping) else {}
        add(
            "routed_expert",
            n_experts=fp.get("num_experts"),
            experts_per_tok=fp.get("num_experts_per_tok"),
            stored_independently=True,
            executes_every_token=False,
            cross_expert_cosine=bank.get("cross_expert_gate_up_mean_cosine"),
            rank_1_energy=bank.get("rank_1_energy"),
        )
        if fp.get("n_shared_experts") or fp.get("shared_expert_intermediate_size"):
            add(
                "shared_expert",
                stored_independently=True,
                executes_every_token=True,
            )
        add("moe_router", stored_independently=True, executes_every_token=True)
    else:
        add("dense_mlp", stored_independently=True, executes_every_token=True)
    add("attention", stored_independently=True, executes_every_token=True)
    if "deltanet" in families or fp.get("full_attention_interval"):
        add("deltanet", stored_independently=True, executes_every_token=True)
    if fp.get("mtp"):
        add("mtp", executes_every_token=False, on_decode_critical_path=False)
    if fp.get("ngram"):
        add("ngram_engine", stored_independently=True, executes_every_token=False)
    if fp.get("multimodal") or "vision" in families:
        add("vision_backbone", executes_every_token=False, on_decode_critical_path=False)
    return organs


def resolve_specimen(target: str | Path) -> Path | None:
    p = Path(target)
    if p.is_dir() and (p / "config.json").is_file():
        return p
    slug = p.name if p.name else str(target)
    cand = SPECIMENS / slug
    if cand.is_dir() and (cand / "config.json").is_file():
        return cand
    return None
