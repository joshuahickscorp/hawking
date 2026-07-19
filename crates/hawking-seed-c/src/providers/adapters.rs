//! **Adapter collapse.** A model adapter is a *declarative execution-plan provider*: it contains only
//! architecture identity, metadata mapping, tensor mapping, a tokenizer/protocol binding, execution-plan
//! generation, and architecture-specific exceptions. It emits a Seed [`crate::ir::Plan`] (and, for MoE, a
//! `Vec<MoeOp>`); the runtime is the Seed's — the adapter does NOT parse GGUF/Safetensors, run a decode
//! loop, manage Metal buffers, keep a KV cache, sample, or write receipts.
//!
//! The five heavy in-tree adapters (gemma2 600, phi3 694, olmoe 595, mamba2 605, mixtral 1117 LOC — each
//! re-implementing the runtime) collapse to the small descriptors below. One shared plan generator serves
//! every llama-family architecture; per-arch code is only the data that differs.
//!
//! ## The one runtime/parity plan builder
//!
//! The Seed's own [`crate::adapter::build_plan`] (GGUF-backed, TIED logits, per-tensor quant) is the ONE
//! runtime/parity authority — the CPU/Metal bit-identity path depends on its exact op order and it is left
//! untouched. The descriptor's [`ArchAdapter::build_plan`] below emits the SAME [`crate::ir`] op sequence
//! and order, but from declared config metadata rather than an mmap'd GGUF, purely to produce an evidence
//! plan-summary. It is never used to execute a model; when real weights are present the Seed's
//! GGUF-backed builder runs.

use super::provider::{Context, Provider, ProviderOutput, ResourceUsage};
use crate::ir::{MoeOp, Op, Plan, Reg, TensorRef};
use crate::pack::CapabilityKind;
use crate::Result;
use serde::{Deserialize, Serialize};

/// A dense transformer config, the metadata every llama-family adapter maps from source metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_ff: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
    pub quant: String,
}

/// The MoE routing shape for expert architectures (olmoe, mixtral, gpt-oss).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Moe {
    pub n_experts: usize,
    pub top_k: usize,
    /// The router weight tensor name template (`{l}` = layer).
    pub router: String,
    /// The per-expert down-projection tensor name template (`{l}`,`{e}`), used for scope identity.
    pub expert_down: String,
    pub router_quant: String,
    pub expert_quant: String,
}

/// GGUF-standard tensor names shared across llama-family GGUFs; adapters override only what differs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorNames {
    pub token_embd: String,
    pub attn_norm: String,
    pub attn_q: String,
    pub attn_k: String,
    pub attn_v: String,
    pub attn_o: String,
    pub ffn_norm: String,
    pub ffn_gate: String,
    pub ffn_up: String,
    pub ffn_down: String,
    pub output_norm: String,
    pub output: String,
}

impl Default for TensorNames {
    fn default() -> Self {
        TensorNames {
            token_embd: "token_embd.weight".into(),
            attn_norm: "blk.{l}.attn_norm.weight".into(),
            attn_q: "blk.{l}.attn_q.weight".into(),
            attn_k: "blk.{l}.attn_k.weight".into(),
            attn_v: "blk.{l}.attn_v.weight".into(),
            attn_o: "blk.{l}.attn_output.weight".into(),
            ffn_norm: "blk.{l}.ffn_norm.weight".into(),
            ffn_gate: "blk.{l}.ffn_gate.weight".into(),
            ffn_up: "blk.{l}.ffn_up.weight".into(),
            ffn_down: "blk.{l}.ffn_down.weight".into(),
            output_norm: "output_norm.weight".into(),
            output: "output.weight".into(),
        }
    }
}

fn subst(template: &str, l: usize) -> String {
    template.replace("{l}", &l.to_string())
}

/// One declarative adapter descriptor. This is the ENTIRE per-architecture surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchAdapter {
    pub arch: String,
    /// Source metadata key prefix (e.g. `llama`, `gemma2`, `gpt-oss`).
    pub meta_prefix: String,
    pub source_format: String,
    pub tokenizer: String,
    pub names: TensorNames,
    pub moe: Option<Moe>,
    /// Architecture-specific exceptions (e.g. SSM layers, sliding-window attention, tied embeddings).
    pub exceptions: Vec<String>,
    /// True when the architecture is NOT a standard attention transformer and needs IR extension.
    pub non_transformer: bool,
}

impl ArchAdapter {
    /// Generate the dense backbone Plan directly from config + declared tensor names. Mirrors the exact
    /// [`crate::ir`] op order of the Seed's runtime builder; used only to summarize a plan as evidence.
    pub fn build_plan(&self, cfg: &Config) -> Result<Plan> {
        if self.non_transformer {
            return Err(crate::Error::Adapter(format!(
                "{}: non-transformer architecture ({}); requires an IR extension, not a dense plan",
                self.arch,
                self.exceptions.join("; ")
            )));
        }
        let q = |name: &str, l: usize, rows: usize, cols: usize| TensorRef {
            name: subst(name, l),
            rows,
            cols,
            quant: cfg.quant.clone(),
        };
        let mut ops = Vec::new();
        ops.push(Op::Embed { out: Reg::X, weight: q(&self.names.token_embd, 0, cfg.vocab, cfg.hidden) });
        for l in 0..cfg.n_layers {
            ops.push(Op::Norm { src: Reg::X, dst: Reg::Xn, weight: q(&self.names.attn_norm, l, cfg.hidden, 1), eps: cfg.rms_eps, n: cfg.hidden });
            ops.push(Op::Linear { src: Reg::Xn, dst: Reg::Q, weight: q(&self.names.attn_q, l, cfg.n_heads * cfg.head_dim, cfg.hidden) });
            ops.push(Op::Linear { src: Reg::Xn, dst: Reg::K, weight: q(&self.names.attn_k, l, cfg.n_kv_heads * cfg.head_dim, cfg.hidden) });
            ops.push(Op::Linear { src: Reg::Xn, dst: Reg::V, weight: q(&self.names.attn_v, l, cfg.n_kv_heads * cfg.head_dim, cfg.hidden) });
            ops.push(Op::Rope { reg: Reg::Q, n_heads: cfg.n_heads, head_dim: cfg.head_dim, base: cfg.rope_base });
            ops.push(Op::Rope { reg: Reg::K, n_heads: cfg.n_kv_heads, head_dim: cfg.head_dim, base: cfg.rope_base });
            ops.push(Op::KvWrite { layer: l, k: Reg::K, v: Reg::V });
            ops.push(Op::Attention { q: Reg::Q, out: Reg::Attn, layer: l, n_heads: cfg.n_heads, n_kv_heads: cfg.n_kv_heads, head_dim: cfg.head_dim });
            ops.push(Op::Linear { src: Reg::Attn, dst: Reg::Xn, weight: q(&self.names.attn_o, l, cfg.hidden, cfg.n_heads * cfg.head_dim) });
            ops.push(Op::Residual { dst: Reg::X, add: Reg::Xn });
            ops.push(Op::Norm { src: Reg::X, dst: Reg::Xn, weight: q(&self.names.ffn_norm, l, cfg.hidden, 1), eps: cfg.rms_eps, n: cfg.hidden });
            if self.moe.is_none() {
                // dense FFN: gate/up/activate/down
                ops.push(Op::Linear { src: Reg::Xn, dst: Reg::G, weight: q(&self.names.ffn_gate, l, cfg.n_ff, cfg.hidden) });
                ops.push(Op::Linear { src: Reg::Xn, dst: Reg::U, weight: q(&self.names.ffn_up, l, cfg.n_ff, cfg.hidden) });
                ops.push(Op::Activate { gate: Reg::G, up: Reg::U, dst: Reg::A });
                ops.push(Op::Linear { src: Reg::A, dst: Reg::Xn, weight: q(&self.names.ffn_down, l, cfg.hidden, cfg.n_ff) });
                ops.push(Op::Residual { dst: Reg::X, add: Reg::Xn });
            }
            // MoE FFN is expressed via the MoE contract (see `moe_ops`), executed by the runtime's MoE path.
        }
        ops.push(Op::Norm { src: Reg::X, dst: Reg::Xn, weight: q(&self.names.output_norm, 0, cfg.hidden, 1), eps: cfg.rms_eps, n: cfg.hidden });
        ops.push(Op::Logits { src: Reg::Xn, dst: Reg::Logits, weight: q(&self.names.output, 0, cfg.vocab, cfg.hidden) });
        ops.push(Op::Sample { src: Reg::Logits });
        Ok(Plan { ops, n_layers: cfg.n_layers, hidden: cfg.hidden, vocab: cfg.vocab })
    }

    /// The MoE FFN op sequence (Route → Expert×top_k → WeightedCombine) for one layer, using the Seed's
    /// MoE contract. `None` for dense architectures.
    pub fn moe_ops(&self, cfg: &Config) -> Option<Vec<MoeOp>> {
        let m = self.moe.as_ref()?;
        let mut ops = vec![MoeOp::Route {
            router: TensorRef { name: subst(&m.router, 0), rows: m.n_experts, cols: cfg.hidden, quant: m.router_quant.clone() },
            n_experts: m.n_experts,
            top_k: m.top_k,
        }];
        for e in 0..m.top_k {
            ops.push(MoeOp::Expert { expert: e });
        }
        ops.push(MoeOp::WeightedCombine);
        Some(ops)
    }
}

// ---- The built-in declarative adapter registry (the adapter collapse). ----

/// Frontier default: dense llama family (the Seed's shipping path).
pub fn llama() -> ArchAdapter {
    ArchAdapter {
        arch: "llama".into(),
        meta_prefix: "llama".into(),
        source_format: "gguf".into(),
        tokenizer: "gguf.tokenizer".into(),
        names: TensorNames::default(),
        moe: None,
        exceptions: vec![],
        non_transformer: false,
    }
}

/// gemma2: dense, logit soft-capping + pre/post-norm (declared as exceptions the runtime honors).
pub fn gemma2() -> ArchAdapter {
    ArchAdapter {
        arch: "gemma2".into(),
        meta_prefix: "gemma2".into(),
        source_format: "gguf".into(),
        tokenizer: "gguf.tokenizer".into(),
        names: TensorNames::default(),
        moe: None,
        exceptions: vec!["attn_logit_softcap".into(), "final_logit_softcap".into(), "pre+post ffn norm".into()],
        non_transformer: false,
    }
}

/// phi3: dense, fused qkv + gate_up (declared exception: split at plan time).
pub fn phi3() -> ArchAdapter {
    ArchAdapter {
        arch: "phi3".into(),
        meta_prefix: "phi3".into(),
        source_format: "gguf".into(),
        tokenizer: "gguf.tokenizer".into(),
        names: TensorNames::default(),
        moe: None,
        exceptions: vec!["fused qkv proj".into(), "fused gate_up proj".into()],
        non_transformer: false,
    }
}

/// olmoe: MoE (64 experts, top-8).
pub fn olmoe() -> ArchAdapter {
    ArchAdapter {
        arch: "olmoe".into(),
        meta_prefix: "olmoe".into(),
        source_format: "gguf".into(),
        tokenizer: "gguf.tokenizer".into(),
        names: TensorNames::default(),
        moe: Some(Moe {
            n_experts: 64,
            top_k: 8,
            router: "blk.{l}.ffn_gate_inp.weight".into(),
            expert_down: "blk.{l}.ffn_down_exps.weight".into(),
            router_quant: "F32".into(),
            expert_quant: "Q4_K".into(),
        }),
        exceptions: vec!["qk-norm".into()],
        non_transformer: false,
    }
}

/// mixtral: MoE (8 experts, top-2), split per-expert tensors.
pub fn mixtral() -> ArchAdapter {
    ArchAdapter {
        arch: "mixtral".into(),
        meta_prefix: "llama".into(),
        source_format: "gguf".into(),
        tokenizer: "gguf.tokenizer".into(),
        names: TensorNames::default(),
        moe: Some(Moe {
            n_experts: 8,
            top_k: 2,
            router: "blk.{l}.ffn_gate_inp.weight".into(),
            expert_down: "blk.{l}.ffn_down.{e}.weight".into(),
            router_quant: "F32".into(),
            expert_quant: "Q4_K".into(),
        }),
        exceptions: vec!["split per-expert tensors (not fused *_exps)".into()],
        non_transformer: false,
    }
}

/// gpt-oss: MoE over safetensors with MXFP4 experts + BF16 router (the run-critical frontier path).
pub fn gpt_oss() -> ArchAdapter {
    ArchAdapter {
        arch: "gpt-oss".into(),
        meta_prefix: "gpt-oss".into(),
        source_format: "safetensors".into(),
        tokenizer: "o200k_harmony".into(),
        names: TensorNames::default(),
        moe: Some(Moe {
            n_experts: 128,
            top_k: 4,
            router: "block.{l}.mlp.gate.weight".into(),
            expert_down: "block.{l}.mlp.mlp1_weight".into(),
            router_quant: "BF16".into(),
            expert_quant: "MXFP4".into(),
        }),
        exceptions: vec!["MXFP4 blocks+scales expert layout".into(), "bounded expert execution (never densify 120B)".into()],
        non_transformer: false,
    }
}

/// mamba2: SSM — NOT an attention transformer. Declared as an architecture-specific exception; a dense
/// plan is refused. This is where the contract says "architecture-specific exceptions", not duplication.
pub fn mamba2() -> ArchAdapter {
    ArchAdapter {
        arch: "mamba2".into(),
        meta_prefix: "mamba2".into(),
        source_format: "gguf".into(),
        tokenizer: "gguf.tokenizer".into(),
        names: TensorNames::default(),
        moe: None,
        exceptions: vec!["selective state-space (SSM) layers, no attention".into(), "requires IR Ssm op extension".into()],
        non_transformer: true,
    }
}

/// The full declarative adapter set the nucleus ships (replacing the heavy in-tree adapters).
pub fn builtins() -> Vec<ArchAdapter> {
    vec![llama(), gemma2(), phi3(), olmoe(), mixtral(), gpt_oss(), mamba2()]
}

/// A `Provider` wrapper over one declarative adapter: input carries the source `Config`, output carries
/// the generated Plan summary + MoE shape as sealed evidence. No runtime, no parsing.
pub struct AdapterProvider {
    pub adapter: ArchAdapter,
    pub capability: String,
}

impl AdapterProvider {
    pub fn new(adapter: ArchAdapter) -> Self {
        let capability = format!("adapter.{}", adapter.arch.replace('-', "_"));
        AdapterProvider { adapter, capability }
    }
}

impl Provider for AdapterProvider {
    fn capability(&self) -> &str {
        &self.capability
    }
    fn kind(&self) -> CapabilityKind {
        CapabilityKind::ModelAdapter
    }
    fn run(&self, _ctx: &Context, input: serde_json::Value) -> Result<ProviderOutput> {
        let cfg: Config = serde_json::from_value(input)
            .map_err(|e| crate::Error::Adapter(format!("adapter {} needs a Config: {e}", self.adapter.arch)))?;
        let plan = self.adapter.build_plan(&cfg)?;
        let moe = self.adapter.moe_ops(&cfg);
        let result = serde_json::json!({
            "arch": self.adapter.arch,
            "plan_summary": plan.summary(),
            "n_ops": plan.ops.len(),
            "moe": self.adapter.moe.as_ref().map(|m| serde_json::json!({"n_experts": m.n_experts, "top_k": m.top_k})),
            "moe_ops": moe.as_ref().map(|o| o.len()),
            "exceptions": self.adapter.exceptions,
        });
        let metrics = serde_json::json!({ "declarative": true, "runtime_owned_by": "hawking", "ops": plan.ops.len() });
        Ok(ProviderOutput::sealed(result, metrics, ResourceUsage::default()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smol() -> Config {
        Config { n_layers: 2, hidden: 64, n_ff: 128, n_heads: 4, n_kv_heads: 2, head_dim: 16, vocab: 100, rms_eps: 1e-5, rope_base: 10000.0, quant: "Q4_K".into() }
    }

    #[test]
    fn dense_adapter_emits_valid_plan() {
        let a = llama();
        let plan = a.build_plan(&smol()).unwrap();
        // embed + 2 layers*(15 ops) + output_norm + logits + sample
        assert!(plan.ops.len() > 10);
        assert_eq!(plan.n_layers, 2);
        assert!(a.moe_ops(&smol()).is_none());
    }

    #[test]
    fn moe_adapter_emits_route_experts_combine() {
        let a = gpt_oss();
        let ops = a.moe_ops(&smol()).unwrap();
        assert!(matches!(ops[0], MoeOp::Route { n_experts: 128, top_k: 4, .. }));
        // 1 route + 4 experts + 1 combine
        assert_eq!(ops.len(), 6);
    }

    #[test]
    fn non_transformer_arch_refuses_dense_plan() {
        assert!(mamba2().build_plan(&smol()).is_err());
    }

    #[test]
    fn adapters_are_small_by_construction() {
        // Every built-in adapter is pure data (no runtime), so its serialized descriptor is tiny.
        for a in builtins() {
            let bytes = serde_json::to_vec(&a).unwrap().len();
            assert!(bytes < 2000, "{} descriptor too large: {bytes} bytes", a.arch);
        }
    }
}
