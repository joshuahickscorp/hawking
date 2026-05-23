pub mod dispatch;

use crate::kernels::{gemv_f32, silu_mul, softmax_inplace};
use crate::Result;

/// Top-K gate over routed-expert logits.
///
/// Inputs:
///   - logits:   (n_experts,)
///   - top_k:    number of experts to route to
///   - normalize: if true, re-normalize the chosen weights to sum to 1
/// Returns:
///   - chosen:   `Vec<(expert_idx, weight)>` of length `top_k`
pub fn topk_gate(logits: &mut [f32], top_k: usize, normalize: bool) -> Vec<(usize, f32)> {
    softmax_inplace(logits);
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let mut out: Vec<(usize, f32)> = idx
        .into_iter()
        .take(top_k)
        .map(|i| (i, logits[i]))
        .collect();
    if normalize {
        let sum: f32 = out.iter().map(|(_, w)| *w).sum();
        if sum > 0.0 {
            for (_, w) in out.iter_mut() {
                *w /= sum;
            }
        }
    }
    out
}

/// Run one expert's SwiGLU FFN on a single token vector.
///
/// `gate_w`, `up_w`, `down_w` are row-major:
///   gate_w / up_w: (intermediate, hidden)
///   down_w:        (hidden, intermediate)
pub fn expert_ffn(
    x: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    hidden: usize,
    intermediate: usize,
    out: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(x.len(), hidden);
    debug_assert_eq!(out.len(), hidden);

    let mut gate = vec![0.0f32; intermediate];
    let mut up = vec![0.0f32; intermediate];
    let mut act = vec![0.0f32; intermediate];

    gemv_f32(gate_w, intermediate, hidden, x, &mut gate);
    gemv_f32(up_w, intermediate, hidden, x, &mut up);
    silu_mul(&gate, &up, &mut act);
    gemv_f32(down_w, hidden, intermediate, &act, out);
    Ok(())
}

/// Per-expert weight slice. The model layer constructs one of these
/// per expert by slicing into the GGUF expert tensors.
pub struct ExpertWeights<'a> {
    pub gate_w: &'a [f32],
    pub up_w: &'a [f32],
    pub down_w: &'a [f32],
}

/// MoE block forward -- single token, top-K routing, weighted sum.
///
/// `experts[i]` corresponds to expert id `i`. Only the experts in
/// comes from `topk_gate`.
pub fn moe_forward_token(
    x: &[f32],
    routes: &[(usize, f32)],
    experts: &[ExpertWeights],
    hidden: usize,
    intermediate: usize,
    out: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(out.len(), hidden);
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let mut tmp = vec![0.0f32; hidden];
    for (eid, w) in routes {
        let e = &experts[*eid];
        expert_ffn(
            x,
            e.gate_w,
            e.up_w,
            e.down_w,
            hidden,
            intermediate,
            &mut tmp,
        )?;
        for i in 0..hidden {
            out[i] += w * tmp[i];
        }
    }
    Ok(())
}

/// Add the contribution of N "shared" experts (always-on, no routing).
/// DeepSeek-V2-Lite has 2 shared experts; Qwen3-MoE has 0.
pub fn add_shared_experts(
    x: &[f32],
    shared: &[ExpertWeights],
    hidden: usize,
    intermediate: usize,
    out: &mut [f32],
) -> Result<()> {
    let mut tmp = vec![0.0f32; hidden];
    for e in shared {
        expert_ffn(
            x,
            e.gate_w,
            e.up_w,
            e.down_w,
            hidden,
            intermediate,
            &mut tmp,
        )?;
        for i in 0..hidden {
            out[i] += tmp[i];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_picks_largest() {
        let mut logits = vec![0.1, 5.0, 2.0, 0.5];
        let chosen = topk_gate(&mut logits, 2, true);
        assert_eq!(chosen.len(), 2);
        assert_eq!(chosen[0].0, 1);
        assert_eq!(chosen[1].0, 2);
        let sum: f32 = chosen.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
