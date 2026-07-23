//! The compact, MoE-ready execution IR. A `Plan` is an ordered list of typed `Op`s over a small
//! register file. Weight ops carry tensor identity, shape, and quant format; the runtime executes each
//! directly on the compressed representation (CPU direct-quant or Metal), never on a dense f32 copy.
//! The IR supports dense Llama today and Route/Experts/WeightedCombine for MoE (exercised by the F2
//! bridge). Op kinds are added only when a fixture uses them.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Reg {
    X = 0,
    Xn = 1,
    Q = 2,
    K = 3,
    V = 4,
    Attn = 5,
    G = 6,
    U = 7,
    A = 8,
    Logits = 9,
}
pub const N_REGS: usize = 10;

#[derive(Debug, Clone, Serialize)]
pub struct TensorRef {
    pub name: String,
    pub rows: usize, // out_features
    pub cols: usize, // in_features
    pub quant: String,
}

#[derive(Debug, Clone, Serialize)]
pub enum Op {
    Embed { out: Reg, weight: TensorRef },
    Norm { src: Reg, dst: Reg, weight: TensorRef, eps: f32, n: usize },
    Linear { src: Reg, dst: Reg, weight: TensorRef },
    Rope { reg: Reg, n_heads: usize, head_dim: usize, base: f32 },
    KvWrite { layer: usize, k: Reg, v: Reg },
    Attention { q: Reg, out: Reg, layer: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize },
    Residual { dst: Reg, add: Reg },
    Activate { gate: Reg, up: Reg, dst: Reg }, // silu(gate)*up
    Logits { src: Reg, dst: Reg, weight: TensorRef },
    Sample { src: Reg },
}

/// MoE contract (exercised by the bounded F2 bridge; the same contract scales to 685B/1T/1.6T later).
#[derive(Debug, Clone, Serialize)]
pub enum MoeOp {
    /// router logits over `n_experts` from the hidden vector, then top-k selection.
    Route { router: TensorRef, n_experts: usize, top_k: usize },
    /// run one selected expert's compact linear (down·act(gate,up)) directly on quantized/sub-bit blocks.
    Expert { expert: usize },
    /// weighted-combine the selected experts' outputs by their (softmaxed) router weights.
    WeightedCombine,
}

#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub ops: Vec<Op>,
    pub n_layers: usize,
    pub hidden: usize,
    pub vocab: usize,
}

impl Plan {
    pub fn summary(&self) -> String {
        let mut linear = 0;
        let mut norm = 0;
        let mut attn = 0;
        for op in &self.ops {
            match op {
                Op::Linear { .. } | Op::Logits { .. } => linear += 1,
                Op::Norm { .. } => norm += 1,
                Op::Attention { .. } => attn += 1,
                _ => {}
            }
        }
        format!("{} ops: {linear} linear, {norm} norm, {attn} attention, {} layers", self.ops.len(), self.n_layers)
    }
}
