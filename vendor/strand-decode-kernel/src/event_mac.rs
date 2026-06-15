
#[derive(Clone, Copy, Debug)]
pub struct EventReport {
    
    pub fired: usize,
    
    pub total: usize,
}

impl EventReport {
    
    pub fn skip_fraction(&self) -> f64 {
        1.0 - self.fired as f64 / self.total.max(1) as f64
    }
}

pub fn salient_threshold(x: &[f32], tau: f32) -> Vec<usize> {
    x.iter()
        .enumerate()
        .filter(|(_, v)| v.abs() >= tau)
        .map(|(i, _)| i)
        .collect()
}

pub fn salient_topk(x: &[f32], k: usize) -> Vec<usize> {
    if k >= x.len() {
        return (0..x.len()).collect();
    }
    let mut idx: Vec<usize> = (0..x.len()).collect();
    
    idx.sort_by(|&a, &b| {
        x[b].abs()
            .partial_cmp(&x[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut s = idx[..k].to_vec();
    s.sort_unstable();
    s
}

pub struct EventMac {
    
    wt: Vec<f32>,
    pub out_features: usize,
    pub in_features: usize,
}

impl EventMac {
    
    pub fn from_dense(w: &[f32], out_features: usize, in_features: usize) -> Self {
        assert_eq!(w.len(), out_features * in_features, "w must be out x in");
        let mut wt = vec![0.0f32; w.len()];
        for o in 0..out_features {
            for i in 0..in_features {
                wt[i * out_features + o] = w[o * in_features + i];
            }
        }
        EventMac { wt, out_features, in_features }
    }

    pub fn from_q12(q12: &[i32], out_features: usize, in_features: usize) -> Self {
        assert_eq!(q12.len(), out_features * in_features, "q12 must be out x in");
        let inv = 1.0f32 / 4096.0;
        let mut wt = vec![0.0f32; q12.len()];
        for o in 0..out_features {
            for i in 0..in_features {
                wt[i * out_features + o] = (q12[o * in_features + i] as f32) * inv;
            }
        }
        EventMac { wt, out_features, in_features }
    }

    pub fn matvec_full(&self, x: &[f32]) -> Vec<f32> {
        let all: Vec<usize> = (0..self.in_features).collect();
        self.matvec_events(x, &all)
    }

    pub fn matvec_events(&self, x: &[f32], salient: &[usize]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_features, "x must have in_features entries");
        debug_assert!(salient.windows(2).all(|w| w[0] < w[1]), "salient must be ascending");
        let out = self.out_features;
        let mut y = vec![0.0f32; out];
        for &i in salient {
            let xv = x[i];
            let col = &self.wt[i * out..(i + 1) * out];
            for (a, &w) in y.iter_mut().zip(col.iter()) {
                *a += w * xv;
            }
        }
        y
    }

    pub fn matvec_threshold(&self, x: &[f32], tau: f32) -> (Vec<f32>, EventReport) {
        let s = salient_threshold(x, tau);
        let rep = EventReport { fired: s.len(), total: self.in_features };
        (self.matvec_events(x, &s), rep)
    }

    pub fn matvec_topk(&self, x: &[f32], k: usize) -> (Vec<f32>, EventReport) {
        let s = salient_topk(x, k);
        let rep = EventReport { fired: s.len(), total: self.in_features };
        (self.matvec_events(x, &s), rep)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DeltaReport {
    
    pub fired: usize,
    pub total: usize,
    
    pub refreshed: bool,
}

impl DeltaReport {
    
    pub fn skip_fraction(&self) -> f64 {
        1.0 - self.fired as f64 / self.total.max(1) as f64
    }
}

pub struct DeltaMac<'a> {
    mac: &'a EventMac,
    
    x_applied: Vec<f32>,
    y: Vec<f32>,
    steps_since_refresh: usize,
    started: bool,
}

impl<'a> DeltaMac<'a> {
    pub fn new(mac: &'a EventMac) -> Self {
        DeltaMac {
            mac,
            x_applied: vec![0.0f32; mac.in_features],
            y: vec![0.0f32; mac.out_features],
            steps_since_refresh: 0,
            started: false,
        }
    }

    pub fn refresh(&mut self, x: &[f32]) -> &[f32] {
        self.y = self.mac.matvec_full(x);
        self.x_applied.copy_from_slice(x);
        self.steps_since_refresh = 0;
        self.started = true;
        &self.y
    }

    pub fn step(&mut self, x: &[f32], tau: f32, refresh_every: usize) -> (&[f32], DeltaReport) {
        assert_eq!(x.len(), self.mac.in_features, "x must have in_features entries");
        assert!(refresh_every >= 1, "refresh_every must be >= 1");
        let total = self.mac.in_features;
        if !self.started || self.steps_since_refresh + 1 >= refresh_every {
            self.refresh(x);
            return (&self.y, DeltaReport { fired: total, total, refreshed: true });
        }
        let out = self.mac.out_features;
        let mut fired = 0usize;
        for i in 0..total {
            let dx = x[i] - self.x_applied[i];
            if dx.abs() >= tau {
                fired += 1;
                let col = &self.mac.wt[i * out..(i + 1) * out];
                for (a, &w) in self.y.iter_mut().zip(col.iter()) {
                    *a += w * dx;
                }
                self.x_applied[i] = x[i];
            }
        }
        self.steps_since_refresh += 1;
        (&self.y, DeltaReport { fired, total, refreshed: false })
    }

    pub fn output(&self) -> &[f32] {
        &self.y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strand_quant::encode::encode_tensor;
    use strand_quant::TrellisConfig;

    fn synth_x(n: usize, seed: f32) -> Vec<f32> {
        
        (0..n)
            .map(|i| {
                let c = ((i as f32 + seed) * 0.0713).cos();
                c * c * c
            })
            .collect()
    }

    #[test]
    fn full_set_bit_equals_reference_matvec() {
        let configs = [
            TrellisConfig::for_bpw(3.0),
            TrellisConfig::for_bpw_l(2.0, 12),
            TrellisConfig::for_bpw_l(2.0, 5), 
        ];
        for cfg in &configs {
            for &(rows, cols) in &[(8usize, 256usize), (37, 300), (5, 97)] {
                let w: Vec<f32> =
                    (0..rows * cols).map(|i| (i as f32 * 0.0137).sin() * 0.5).collect();
                let enc = encode_tensor(&w, cfg);
                let q12 = crate::decode_weights_q12(&enc, cfg, None);
                let em = EventMac::from_q12(&q12, rows, cols);
                let x = synth_x(cols, 1.0);
                let y_ref = crate::matvec(&enc, cfg, None, rows, cols, &x);
                for (label, y_ev) in [
                    ("matvec_full", em.matvec_full(&x)),
                    ("threshold tau=0", em.matvec_threshold(&x, 0.0).0),
                    ("topk k=n", em.matvec_topk(&x, cols).0),
                ] {
                    assert_eq!(y_ev.len(), y_ref.len());
                    for o in 0..rows {
                        assert_eq!(
                            y_ev[o].to_bits(),
                            y_ref[o].to_bits(),
                            "{label} y[{o}] != crate::matvec (L={} k={} {rows}x{cols})",
                            cfg.l_bits,
                            cfg.k_bits
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn partial_set_exact_over_s() {
        let (rows, cols) = (23usize, 300usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.0091).sin() * 0.4).collect();
        let em = EventMac::from_dense(&w, rows, cols);
        let x = synth_x(cols, 2.0);
        for tau in [0.05f32, 0.2, 0.5] {
            let (y, rep) = em.matvec_threshold(&x, tau);
            assert!(rep.fired < cols, "tau {tau} should skip something");
            
            for o in 0..rows {
                let mut acc = 0.0f32;
                for i in 0..cols {
                    if x[i].abs() >= tau {
                        acc += w[o * cols + i] * x[i];
                    }
                }
                assert_eq!(y[o].to_bits(), acc.to_bits(), "tau {tau} row {o} not exact over S");
            }
        }
    }

    #[test]
    fn topk_is_deterministic_and_ascending() {
        let x = synth_x(512, 3.3);
        for k in [1usize, 7, 64, 511, 512, 600] {
            let s = salient_topk(&x, k);
            assert_eq!(s.len(), k.min(512));
            assert!(s.windows(2).all(|w| w[0] < w[1]), "ascending");
            let s2 = salient_topk(&x, k);
            assert_eq!(s, s2, "deterministic");
        }
        
        let k = 32;
        let s = salient_topk(&x, k);
        let mut mags: Vec<f32> = x.iter().map(|v| v.abs()).collect();
        mags.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let cut = mags[k - 1];
        for &i in &s {
            assert!(x[i].abs() >= cut, "top-k member below the cut");
        }
    }

    #[test]
    fn delta_mode_refresh_identity_and_tau_bound() {
        let (rows, cols) = (16usize, 256usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.0113).sin() * 0.6).collect();
        let em = EventMac::from_dense(&w, rows, cols);

        let mut dm = DeltaMac::new(&em);
        for t in 0..5 {
            let x = synth_x(cols, t as f32 * 1.7);
            let (y, rep) = dm.step(&x, 0.1, 1);
            assert!(rep.refreshed);
            let y_full = em.matvec_full(&x);
            for o in 0..rows {
                assert_eq!(y[o].to_bits(), y_full[o].to_bits(), "refresh-every-1 step {t} row {o}");
            }
        }

        let tau = 0.05f32;
        let mut dm = DeltaMac::new(&em);
        let mut x = synth_x(cols, 0.0);
        dm.step(&x, tau, usize::MAX);
        for t in 1..20 {
            
            for (i, v) in x.iter_mut().enumerate() {
                *v += 0.01 * ((t * 31 + i) as f32 * 0.11).sin();
            }
            let (_y, rep) = dm.step(&x, tau, usize::MAX);
            assert!(!rep.refreshed);
            assert!(rep.fired < cols, "small drift must not fire everything");
            for i in 0..cols {
                assert!(
                    (x[i] - dm.x_applied[i]).abs() < tau,
                    "applied state escaped the tau bound at dim {i} step {t}"
                );
            }
        }
        
        let (y, rep) = dm.step(&x, tau, dm.steps_since_refresh + 1);
        assert!(rep.refreshed);
        let y_full = em.matvec_full(&x);
        for o in 0..rows {
            assert_eq!(y[o].to_bits(), y_full[o].to_bits(), "post-refresh row {o}");
        }
    }

    #[test]
    fn delta_mode_error_is_small_and_bounded() {
        let (rows, cols) = (16usize, 256usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.0177).cos() * 0.5).collect();
        let em = EventMac::from_dense(&w, rows, cols);
        let tau = 0.02f32;
        let mut dm = DeltaMac::new(&em);
        let mut x = synth_x(cols, 9.0);
        dm.step(&x, tau, usize::MAX);
        for t in 1..30 {
            for (i, v) in x.iter_mut().enumerate() {
                *v += 0.005 * ((t * 17 + i * 3) as f32 * 0.07).cos();
            }
            let (y, _rep) = dm.step(&x, tau, usize::MAX);
            let y_full = em.matvec_full(&x);
            let num: f32 = y.iter().zip(&y_full).map(|(a, b)| (a - b) * (a - b)).sum();
            let den: f32 = y_full.iter().map(|v| v * v).sum();
            let rel = (num / den.max(1e-30)).sqrt();
            
            assert!(rel < 0.05, "delta drift rel err {rel} too large at step {t}");
        }
    }
}
