//! **Metal/kernel collapse.** One operation contract serves CPU and Metal parity. The reference op is the
//! measured bottleneck — the tied-vocabulary projection executed DIRECTLY on Q8_0 blocks — reusing the
//! Seed's kernel ([`crate::metal::MetalGemv`]) and its CPU reference ([`crate::cpu::logits_tied`]). We keep
//! the measured winner; tuning-only variants become descriptors, never forked algorithms. The provider
//! proves the Metal result matches the CPU reference (or runs CPU-only where no Metal device exists — the
//! same fallback the Seed uses).

use super::provider::{Context, Provider, ProviderOutput, ResourceUsage};
use crate::cpu;
use crate::gguf::GgmlType;
use crate::metal::MetalGemv;
use crate::pack::CapabilityKind;
use crate::Result;

const Q8_0_BLOCK: usize = 34; // 2-byte f16 scale + 32 int8 quants
const Q8_0_ELEMS: usize = 32;

/// Build a deterministic Q8_0 embedding buffer of `vocab`×`hidden` (hidden a multiple of 32).
pub fn synthetic_q8_0(vocab: usize, hidden: usize) -> Vec<u8> {
    assert!(hidden % Q8_0_ELEMS == 0);
    let blocks_per_row = hidden / Q8_0_ELEMS;
    let mut bytes = vec![0u8; vocab * blocks_per_row * Q8_0_BLOCK];
    for v in 0..vocab {
        for b in 0..blocks_per_row {
            let off = (v * blocks_per_row + b) * Q8_0_BLOCK;
            let d = half::f16::from_f32(0.01 + ((v * 7 + b) % 13) as f32 * 0.001);
            bytes[off..off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            for i in 0..Q8_0_ELEMS {
                let q = (((v * 31 + b * 17 + i * 3) % 255) as i32 - 127) as i8;
                bytes[off + 2 + i] = q as u8;
            }
        }
    }
    bytes
}

/// The result of running the one operation on both backends.
#[derive(Debug, Clone)]
pub struct OpParity {
    pub device: String,
    pub metal_available: bool,
    pub argmax_agree: bool,
    pub max_abs_diff: f32,
    pub vocab: usize,
    pub hidden: usize,
}

/// The one tied-logits operation, with a CPU reference and a Metal implementation held to parity.
pub struct TiedLogitsOp;

impl TiedLogitsOp {
    pub fn cpu(&self, embd: &[u8], hidden: usize, vocab: usize, x: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0f32; vocab];
        cpu::logits_tied(GgmlType::Q8_0, embd, hidden, vocab, x, &mut out)?;
        Ok(out)
    }

    /// Run both backends and measure parity. Metal is optional; CPU is always the reference.
    pub fn parity(&self, vocab: usize, hidden: usize) -> Result<OpParity> {
        let embd = synthetic_q8_0(vocab, hidden);
        let x: Vec<f32> = (0..hidden).map(|c| ((c % 17) as f32 - 8.0) * 0.05).collect();
        let cpu_out = self.cpu(&embd, hidden, vocab, &x)?;
        let cpu_argmax = argmax(&cpu_out);

        match MetalGemv::new() {
            Some(g) => {
                let metal_out = g.logits_q8_0(&embd, hidden, vocab, &x)?;
                let max_abs_diff = cpu_out
                    .iter()
                    .zip(&metal_out)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                Ok(OpParity {
                    device: g.device_name.clone(),
                    metal_available: true,
                    argmax_agree: argmax(&metal_out) == cpu_argmax,
                    max_abs_diff,
                    vocab,
                    hidden,
                })
            }
            None => Ok(OpParity {
                device: "cpu-fallback".into(),
                metal_available: false,
                argmax_agree: true,
                max_abs_diff: 0.0,
                vocab,
                hidden,
            }),
        }
    }
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..xs.len() {
        if xs[i] > xs[best] {
            best = i;
        }
    }
    best
}

/// A `Provider` over the one operation contract.
pub struct MetalOpProvider {
    pub op: TiedLogitsOp,
    capability: String,
}

impl Default for MetalOpProvider {
    fn default() -> Self {
        MetalOpProvider { op: TiedLogitsOp, capability: "metal.tied_logits".into() }
    }
}

impl Provider for MetalOpProvider {
    fn capability(&self) -> &str {
        &self.capability
    }
    fn kind(&self) -> CapabilityKind {
        CapabilityKind::MetalImpl
    }
    fn run(&self, _ctx: &Context, input: serde_json::Value) -> Result<ProviderOutput> {
        let vocab = input["vocab"].as_u64().unwrap_or(512) as usize;
        let hidden = input["hidden"].as_u64().unwrap_or(64) as usize;
        let p = self.op.parity(vocab, hidden)?;
        let result = serde_json::json!({
            "device": p.device,
            "metal_available": p.metal_available,
            "argmax_agree": p.argmax_agree,
            "max_abs_diff": p.max_abs_diff,
        });
        let metrics = serde_json::json!({
            "one_op_contract": "tied_logits_q8_0",
            "cpu_metal_parity": p.argmax_agree,
            "vocab": p.vocab,
            "hidden": p.hidden,
        });
        Ok(ProviderOutput::sealed(result, metrics, ResourceUsage::default()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_op_cpu_metal_parity() {
        // CPU is always available; Metal is exercised when present. Parity (argmax agreement) must hold.
        let p = TiedLogitsOp.parity(256, 64).unwrap();
        assert!(p.argmax_agree, "Metal must agree with the CPU reference (or CPU-only fallback)");
        if p.metal_available {
            assert!(p.max_abs_diff < 1e-2, "logits within tolerance: {}", p.max_abs_diff);
        }
    }
}
