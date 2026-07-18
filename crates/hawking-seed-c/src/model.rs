//! The Event Horizon runtime: mmap the compressed model, build the IR, and interpret it token-by-token
//! executing DIRECTLY on quantized weight views (per-row tile dequant) — no dense f32 model copy. Only
//! the tiny 1-D norm vectors are cached (f32, ~140 KB). Metal accelerates the tied-vocab bottleneck;
//! CPU is the parity reference. The prefill/decode loop matches the predecessor, so greedy output is
//! bit-identical to the golden.

use crate::adapter::{self, LlamaConfig};
use crate::gguf::GgufFile;
use crate::ir::{Op, Plan, Reg, N_REGS};
use crate::metal::MetalGemv;
use crate::{cpu, quant, Error, Result};
use std::collections::HashMap;

struct RtState {
    regs: Vec<Vec<f32>>,
    kcache: Vec<Vec<f32>>,
    vcache: Vec<Vec<f32>>,
    max_seq: usize,
    next_token: u32,
}

pub struct Model {
    g: GgufFile,
    pub cfg: LlamaConfig,
    pub plan: Plan,
    norms: HashMap<String, Vec<f32>>,
    metal: Option<MetalGemv>,
    st: RtState,
}

impl Model {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let g = GgufFile::open(path)?;
        let cfg = LlamaConfig::from_gguf(&g)?;
        let plan = adapter::build_plan(&g, &cfg)?;

        // cache only the tiny 1-D norm vectors as f32 (the big 2-D weights stay compressed).
        let mut norms = HashMap::new();
        for op in &plan.ops {
            if let Op::Norm { weight, .. } = op {
                if !norms.contains_key(&weight.name) {
                    let t = g.tensor(&weight.name)?;
                    let n: usize = t.dims.iter().product::<u64>() as usize;
                    let mut v = vec![0f32; n];
                    quant::dequant(t.dtype, g.tensor_bytes(&weight.name)?, &mut v)?;
                    norms.insert(weight.name.clone(), v);
                }
            }
        }
        let metal = MetalGemv::new();
        Ok(Model {
            g,
            cfg,
            plan,
            norms,
            metal,
            st: RtState { regs: vec![Vec::new(); N_REGS], kcache: Vec::new(), vcache: Vec::new(), max_seq: 0, next_token: 0 },
        })
    }

    pub fn metal_available(&self) -> bool {
        self.metal.is_some()
    }
    pub fn metal_device(&self) -> Option<&str> {
        self.metal.as_ref().map(|m| m.device_name.as_str())
    }
    pub fn mapped_bytes(&self) -> usize {
        self.g.mapped_bytes
    }

    fn reset_cache(&mut self, max_seq: usize) {
        let kv_dim = self.cfg.kv_dim();
        self.st.max_seq = max_seq;
        self.st.kcache = vec![vec![0f32; max_seq * kv_dim]; self.cfg.n_layers];
        self.st.vcache = vec![vec![0f32; max_seq * kv_dim]; self.cfg.n_layers];
    }

    pub fn forward(&mut self, token: u32, pos: usize) -> Result<()> {
        let ops = self.plan.ops.clone();
        for op in &ops {
            run_op(&self.g, &self.cfg, &self.norms, &mut self.st, op, token, pos)?;
        }
        Ok(())
    }

    pub fn generate(&mut self, prompt: &[u32], max_new: usize, eos: u32) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(Error::Model("empty prompt".into()));
        }
        let prompt_len = prompt.len();
        self.reset_cache(prompt_len + max_new);
        for (i, &t) in prompt.iter().enumerate() {
            self.forward(t, i)?;
        }
        let mut last = *prompt.last().unwrap();
        let mut out = Vec::with_capacity(max_new);
        for step in 0..max_new {
            self.forward(last, prompt_len + step)?;
            let next = self.st.next_token;
            out.push(next);
            if next == eos {
                break;
            }
            last = next;
        }
        Ok(out)
    }

    /// Benchmark + validate the Metal tied-vocab projection against the CPU reference on the final
    /// hidden state of one forward. Returns (cpu_ms, metal_ms, argmax_agree, max_abs_diff).
    pub fn bench_logits_cpu_vs_metal(&mut self, prompt: &[u32]) -> Result<(f64, f64, bool, f32)> {
        self.reset_cache(prompt.len() + 1);
        for (i, &t) in prompt.iter().enumerate() {
            self.forward(t, i)?;
        }
        // the normalized final hidden is in Xn after the last forward's Norm+Logits; recompute Xn.
        // Use the current X -> output_norm -> Xn as the LM-head input.
        let x = self.st.regs[Reg::X as usize].clone();
        let onorm = self.norms.get("output_norm.weight").unwrap().clone();
        let mut xn = vec![0f32; self.cfg.hidden];
        cpu::rmsnorm(&x, &onorm, self.cfg.rms_eps, &mut xn);

        let embd_bytes = self.g.tensor_bytes("token_embd.weight")?;
        let dtype = self.g.tensor("token_embd.weight")?.dtype;
        let (hidden, vocab) = (self.cfg.hidden, self.cfg.vocab);

        let t0 = std::time::Instant::now();
        let mut cpu_out = vec![0f32; vocab];
        cpu::logits_tied(dtype, embd_bytes, hidden, vocab, &xn, &mut cpu_out)?;
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let Some(m) = self.metal.as_ref() else {
            return Ok((cpu_ms, f64::NAN, false, f32::NAN));
        };
        // warm + timed
        let _ = m.logits_q8_0(embd_bytes, hidden, vocab, &xn)?;
        let t1 = std::time::Instant::now();
        let gpu_out = m.logits_q8_0(embd_bytes, hidden, vocab, &xn)?;
        let metal_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let agree = cpu::argmax(&cpu_out) == cpu::argmax(&gpu_out);
        let max_diff = cpu_out.iter().zip(&gpu_out).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        Ok((cpu_ms, metal_ms, agree, max_diff))
    }
}

#[allow(clippy::too_many_arguments)]
fn run_op(g: &GgufFile, cfg: &LlamaConfig, norms: &HashMap<String, Vec<f32>>, st: &mut RtState, op: &Op, token: u32, pos: usize) -> Result<()> {
    match op {
        Op::Embed { out, weight } => {
            let dtype = g.tensor(&weight.name)?.dtype;
            let bytes = g.tensor_bytes(&weight.name)?;
            let h = weight.cols;
            let mut o = vec![0f32; h];
            cpu::embed_lookup(dtype, bytes, token as usize, h, &mut o)?;
            st.regs[*out as usize] = o;
        }
        Op::Norm { src, dst, weight, eps, n } => {
            let x = st.regs[*src as usize].clone();
            let w = norms.get(&weight.name).ok_or_else(|| Error::Model(format!("norm not cached {}", weight.name)))?;
            let mut o = vec![0f32; *n];
            cpu::rmsnorm(&x, w, *eps, &mut o);
            st.regs[*dst as usize] = o;
        }
        Op::Linear { src, dst, weight } => {
            let dtype = g.tensor(&weight.name)?.dtype;
            let bytes = g.tensor_bytes(&weight.name)?;
            let x = st.regs[*src as usize].clone();
            let mut o = vec![0f32; weight.rows];
            cpu::gemv_quant(dtype, bytes, weight.rows, weight.cols, &x, &mut o)?;
            st.regs[*dst as usize] = o;
        }
        Op::Rope { reg, n_heads, head_dim, base } => {
            let mut x = st.regs[*reg as usize].clone();
            cpu::rope_neox(&mut x, *n_heads, *head_dim, *base, pos);
            st.regs[*reg as usize] = x;
        }
        Op::KvWrite { layer, k, v } => {
            let kv_dim = cfg.kv_dim();
            if pos >= st.max_seq {
                return Err(Error::Runtime(format!("kv cache full at {pos}")));
            }
            let off = pos * kv_dim;
            let kk = st.regs[*k as usize].clone();
            let vv = st.regs[*v as usize].clone();
            st.kcache[*layer][off..off + kv_dim].copy_from_slice(&kk);
            st.vcache[*layer][off..off + kv_dim].copy_from_slice(&vv);
        }
        Op::Attention { q, out, layer, n_heads, n_kv_heads, head_dim } => {
            let qv = st.regs[*q as usize].clone();
            let kv_dim = n_kv_heads * head_dim;
            let scale = 1.0f32 / (*head_dim as f32).sqrt();
            let group = n_heads / n_kv_heads;
            let seq = pos + 1;
            let mut o = vec![0f32; n_heads * head_dim];
            for h in 0..*n_heads {
                let kvh = h / group;
                let qh = &qv[h * head_dim..h * head_dim + head_dim];
                let mut scores = vec![0f32; seq];
                for t in 0..seq {
                    let koff = t * kv_dim + kvh * head_dim;
                    let krow = &st.kcache[*layer][koff..koff + head_dim];
                    let mut s = 0f32;
                    for i in 0..*head_dim {
                        s += qh[i] * krow[i];
                    }
                    scores[t] = s * scale;
                }
                cpu::softmax(&mut scores);
                let ob = h * head_dim;
                for t in 0..seq {
                    let voff = t * kv_dim + kvh * head_dim;
                    let vrow = &st.vcache[*layer][voff..voff + head_dim];
                    let sc = scores[t];
                    for i in 0..*head_dim {
                        o[ob + i] += sc * vrow[i];
                    }
                }
            }
            st.regs[*out as usize] = o;
        }
        Op::Residual { dst, add } => {
            let a = st.regs[*add as usize].clone();
            cpu::add_inplace(&mut st.regs[*dst as usize], &a);
        }
        Op::Activate { gate, up, dst } => {
            let gg = st.regs[*gate as usize].clone();
            let uu = st.regs[*up as usize].clone();
            let mut o = vec![0f32; gg.len()];
            cpu::silu_mul(&gg, &uu, &mut o);
            st.regs[*dst as usize] = o;
        }
        Op::Logits { src, dst, weight } => {
            let dtype = g.tensor(&weight.name)?.dtype;
            let bytes = g.tensor_bytes(&weight.name)?;
            let x = st.regs[*src as usize].clone();
            let mut o = vec![0f32; weight.rows];
            cpu::logits_tied(dtype, bytes, weight.cols, weight.rows, &x, &mut o)?;
            st.regs[*dst as usize] = o;
        }
        Op::Sample { src } => {
            st.next_token = cpu::argmax(&st.regs[*src as usize]);
        }
    }
    Ok(())
}
