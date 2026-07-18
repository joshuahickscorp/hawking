//! The runtime: load the GGUF, dequantize weights (embedding via f16 like the predecessor), build the
//! execution IR through the adapter, and interpret it token-by-token. Holds the register file and the
//! KV cache. `generate` reproduces the predecessor's exact prefill/decode loop (the last prompt token
//! is re-forwarded at pos=prompt_len as decode step 0), so greedy output is bit-identical.

use crate::adapter::{self, LlamaConfig};
use crate::gguf::GgufFile;
use crate::ir::{Op, Plan, Reg, N_REGS};
use crate::quant::{self};
use crate::{ops, Error, Result};
use std::collections::HashMap;

pub struct Model {
    pub cfg: LlamaConfig,
    pub plan: Plan,
    /// tied token embedding, stored as f16 bits (Q8_0 -> f32 -> f16), vocab*hidden.
    embed: Vec<u16>,
    /// dequantized f32 weights, keyed by tensor name (everything except token_embd).
    weights: HashMap<String, Vec<f32>>,
    regs: Vec<Vec<f32>>,
    kcache: Vec<Vec<f32>>, // per layer: max_seq * kv_dim
    vcache: Vec<Vec<f32>>,
    max_seq: usize,
    next_token: u32,
    pub source_bytes: usize,
    pub weight_f32_elems: usize,
}

impl Model {
    pub fn load(gguf_path: &std::path::Path) -> Result<Self> {
        let g = GgufFile::open(gguf_path)?;
        Self::from_gguf(&g)
    }

    /// Build the runtime from an already-parsed GGUF (avoids re-reading the 105 MB file).
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let cfg = LlamaConfig::from_gguf(g)?;
        let plan = adapter::build_plan(g, &cfg)?;
        let source_bytes = g.data.len();

        // Embedding: Q8_0 -> f32 -> round to f16 (lossy, reproduced), stored as f16 bits.
        let et = g.tensor("token_embd.weight")?;
        let n_embed: usize = et.dims.iter().product::<u64>() as usize;
        let mut ef32 = vec![0f32; n_embed];
        quant::dequant(et.dtype, g.tensor_bytes("token_embd.weight")?, &mut ef32)?;
        let embed: Vec<u16> = ef32.iter().map(|&v| quant::f32_to_f16_bits(v)).collect();

        // All other referenced tensors -> dequantized f32, deduped by name.
        let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
        let mut weight_f32_elems = 0usize;
        for op in &plan.ops {
            let name = match op {
                Op::RmsNorm { weight, .. } | Op::Linear { weight, .. } => weight.name.clone(),
                _ => continue,
            };
            if weights.contains_key(&name) {
                continue;
            }
            let t = g.tensor(&name)?;
            let n: usize = t.dims.iter().product::<u64>() as usize;
            let mut buf = vec![0f32; n];
            quant::dequant(t.dtype, g.tensor_bytes(&name)?, &mut buf)?;
            weight_f32_elems += n;
            weights.insert(name, buf);
        }

        Ok(Model {
            cfg,
            plan,
            embed,
            weights,
            regs: vec![Vec::new(); N_REGS],
            kcache: Vec::new(),
            vcache: Vec::new(),
            max_seq: 0,
            next_token: 0,
            source_bytes,
            weight_f32_elems,
        })
    }

    fn reset_cache(&mut self, max_seq: usize) {
        let kv_dim = self.cfg.kv_dim();
        self.max_seq = max_seq;
        self.kcache = vec![vec![0f32; max_seq * kv_dim]; self.cfg.n_layers];
        self.vcache = vec![vec![0f32; max_seq * kv_dim]; self.cfg.n_layers];
    }

    #[inline]
    fn reg(&self, r: Reg) -> &[f32] {
        &self.regs[r as usize]
    }
    #[inline]
    fn set_reg(&mut self, r: Reg, v: Vec<f32>) {
        self.regs[r as usize] = v;
    }
    fn weight(&self, name: &str) -> Result<&[f32]> {
        self.weights
            .get(name)
            .map(|v| v.as_slice())
            .ok_or_else(|| Error::Model(format!("weight not loaded: {name}")))
    }

    /// Execute the whole plan for one token at absolute position `pos`. Fills regs (incl. Logits) and,
    /// via the Sample op, `next_token`.
    pub fn forward(&mut self, token: u32, pos: usize) -> Result<()> {
        // clone the op list to avoid borrowing self immutably while mutating regs.
        let ops_list = self.plan.ops.clone();
        for op in &ops_list {
            self.run_op(op, token, pos)?;
        }
        Ok(())
    }

    fn run_op(&mut self, op: &Op, token: u32, pos: usize) -> Result<()> {
        match op {
            Op::Embed { out, weight } => {
                let h = weight.cols; // hidden
                let base = token as usize * h;
                let mut o = vec![0f32; h];
                for i in 0..h {
                    o[i] = quant::f16_to_f32(self.embed[base + i]);
                }
                self.set_reg(*out, o);
            }
            Op::RmsNorm { src, dst, weight, eps, n } => {
                let x = self.reg(*src).to_vec();
                let w = self.weight(&weight.name)?;
                let mut o = vec![0f32; *n];
                ops::rmsnorm(&x, w, *eps, &mut o);
                self.set_reg(*dst, o);
            }
            Op::Linear { src, dst, weight } => {
                let x = self.reg(*src).to_vec();
                let w = self.weight(&weight.name)?;
                let mut o = vec![0f32; weight.rows];
                ops::gemv(w, &x, weight.rows, weight.cols, &mut o);
                self.set_reg(*dst, o);
            }
            Op::Rope { reg, n_heads, head_dim, base } => {
                let mut x = self.reg(*reg).to_vec();
                ops::rope_neox(&mut x, *n_heads, *head_dim, *base, pos);
                self.set_reg(*reg, x);
            }
            Op::KvWrite { layer, k, v } => {
                let kv_dim = self.cfg.kv_dim();
                let kk = self.reg(*k).to_vec();
                let vv = self.reg(*v).to_vec();
                if pos >= self.max_seq {
                    return Err(Error::Runtime(format!("kv cache full at {pos}")));
                }
                let off = pos * kv_dim;
                self.kcache[*layer][off..off + kv_dim].copy_from_slice(&kk);
                self.vcache[*layer][off..off + kv_dim].copy_from_slice(&vv);
            }
            Op::Attention { q, out, layer, n_heads, n_kv_heads, head_dim } => {
                let qv = self.reg(*q).to_vec();
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
                        let krow = &self.kcache[*layer][koff..koff + head_dim];
                        let mut s = 0f32;
                        for i in 0..*head_dim {
                            s += qh[i] * krow[i];
                        }
                        scores[t] = s * scale;
                    }
                    ops::softmax(&mut scores);
                    let ob = h * head_dim;
                    for t in 0..seq {
                        let voff = t * kv_dim + kvh * head_dim;
                        let vrow = &self.vcache[*layer][voff..voff + head_dim];
                        let sc = scores[t];
                        for i in 0..*head_dim {
                            o[ob + i] += sc * vrow[i];
                        }
                    }
                }
                self.set_reg(*out, o);
            }
            Op::Residual { dst, add } => {
                let a = self.reg(*add).to_vec();
                ops::add_inplace(&mut self.regs[*dst as usize], &a);
            }
            Op::SiluMul { gate, up, dst } => {
                let g = self.reg(*gate).to_vec();
                let u = self.reg(*up).to_vec();
                let mut o = vec![0f32; g.len()];
                ops::silu_mul(&g, &u, &mut o);
                self.set_reg(*dst, o);
            }
            Op::Logits { src, dst, weight } => {
                let x = self.reg(*src).to_vec();
                let mut o = vec![0f32; weight.rows];
                ops::gemv_f16(&self.embed, &x, weight.rows, weight.cols, &mut o);
                self.set_reg(*dst, o);
            }
            Op::Sample { src } => {
                self.next_token = ops::argmax(self.reg(*src));
            }
        }
        Ok(())
    }

    /// Reproduce the predecessor's prefill/decode loop and return the generated token ids.
    pub fn generate(&mut self, prompt: &[u32], max_new: usize, eos: u32) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(Error::Model("empty prompt".into()));
        }
        let prompt_len = prompt.len();
        self.reset_cache(prompt_len + max_new);

        // prefill: forward every prompt token at pos = i.
        for (i, &t) in prompt.iter().enumerate() {
            self.forward(t, i)?;
        }

        // decode: re-forward the last prompt token at pos=prompt_len (step 0), then each new token.
        let mut last_id = *prompt.last().unwrap();
        let mut out = Vec::with_capacity(max_new);
        for step in 0..max_new {
            let pos = prompt_len + step;
            self.forward(last_id, pos)?;
            let next_id = self.next_token; // set by the Sample op
            out.push(next_id);
            if next_id == eos {
                break;
            }
            last_id = next_id;
        }
        Ok(out)
    }

    pub fn logits(&self) -> &[f32] {
        self.reg(Reg::Logits)
    }

    /// Audit hook: like `generate`, but also return a sha256 checksum of the full logit vector at each
    /// decode step. Proves tokens come from calculated logits (not hardcoded) and gives a multi-step
    /// signature that final-token parity alone cannot fake.
    pub fn decode_logit_shas(&mut self, prompt: &[u32], max_new: usize) -> Result<(Vec<u32>, Vec<String>)> {
        use sha2::{Digest, Sha256};
        let prompt_len = prompt.len();
        self.reset_cache(prompt_len + max_new);
        for (i, &t) in prompt.iter().enumerate() {
            self.forward(t, i)?;
        }
        let mut last_id = *prompt.last().unwrap();
        let mut ids = Vec::new();
        let mut shas = Vec::new();
        for step in 0..max_new {
            self.forward(last_id, prompt_len + step)?;
            let logits = self.reg(Reg::Logits);
            let mut h = Sha256::new();
            for &v in logits {
                h.update(v.to_le_bytes());
            }
            shas.push(format!("{:x}", h.finalize()));
            let next = self.next_token;
            ids.push(next);
            last_id = next;
        }
        Ok((ids, shas))
    }
}
