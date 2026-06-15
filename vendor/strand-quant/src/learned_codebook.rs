
use crate::codebook::QUANTILE_SHIFT;

pub const RECON_CLAMP_Q12: i32 = 6 << QUANTILE_SHIFT;

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        let n = n as u64;
        
        let zone = u64::MAX - (u64::MAX % n);
        loop {
            let r = self.next_u64();
            if r < zone {
                return (r % n) as usize;
            }
        }
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainConfig {
    
    pub d: usize,
    
    pub k_centroids: usize,
    
    pub iters: usize,
    
    pub seed: u64,
}

impl TrainConfig {
    
    pub fn new(d: usize, k_centroids: usize) -> Self {
        TrainConfig {
            d,
            k_centroids,
            iters: 50,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Centroids {
    
    pub data: Vec<f64>,
    
    pub d: usize,
    
    pub k: usize,
}

impl Centroids {
    
    #[inline]
    pub fn centroid(&self, i: usize) -> &[f64] {
        &self.data[i * self.d..i * self.d + self.d]
    }

    pub fn freeze(&self) -> FrozenCodebook {
        freeze_centroids(self)
    }

    pub fn reconstruction_mse(&self, samples: &[f64]) -> f64 {
        let n = samples.len() / self.d;
        if n == 0 || self.k == 0 {
            return 0.0;
        }
        let mut acc = 0.0f64;
        for p in 0..n {
            let x = &samples[p * self.d..p * self.d + self.d];
            let (_, dist) = nearest_centroid(x, &self.data, self.k, self.d);
            acc += dist;
        }
        
        acc / (n as f64 * self.d as f64)
    }
}

#[inline]
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for j in 0..a.len() {
        let diff = a[j] - b[j];
        s += diff * diff;
    }
    s
}

#[inline]
fn nearest_centroid(x: &[f64], centroids: &[f64], k: usize, d: usize) -> (usize, f64) {
    let mut best = 0usize;
    let mut best_d = f64::INFINITY;
    for i in 0..k {
        let c = &centroids[i * d..i * d + d];
        let dist = sq_dist(x, c);
        if dist < best_d {
            best_d = dist;
            best = i;
        }
    }
    (best, best_d)
}

fn kmeans_pp_init(points: &[f64], n: usize, d: usize, k: usize, rng: &mut SplitMix64) -> Vec<f64> {
    let mut centroids = vec![0.0f64; k * d];
    
    let first = rng.next_below(n);
    centroids[0..d].copy_from_slice(&points[first * d..first * d + d]);

    let mut d2 = vec![f64::INFINITY; n];
    
    for chosen in 1..k {
        let last = &centroids[(chosen - 1) * d..(chosen - 1) * d + d];
        let mut total = 0.0f64;
        for p in 0..n {
            let x = &points[p * d..p * d + d];
            let dist = sq_dist(x, last);
            if dist < d2[p] {
                d2[p] = dist;
            }
            total += d2[p];
        }
        
        let pick = if total > 0.0 {
            let target = rng.next_f64() * total;
            let mut cum = 0.0f64;
            let mut idx = n - 1; 
            for p in 0..n {
                cum += d2[p];
                if cum > target {
                    idx = p;
                    break;
                }
            }
            idx
        } else {
            
            0
        };
        centroids[chosen * d..chosen * d + d].copy_from_slice(&points[pick * d..pick * d + d]);
    }
    centroids
}

pub fn train(samples: &[f32], cfg: &TrainConfig) -> Centroids {
    let d = cfg.d.max(1);
    let k = cfg.k_centroids.max(1);
    let n = samples.len() / d;

    if n == 0 {
        return Centroids {
            data: vec![0.0f64; k * d],
            d,
            k,
        };
    }

    let points: Vec<f64> = samples[..n * d].iter().map(|&w| w as f64).collect();

    let mut rng = SplitMix64::new(cfg.seed);
    let mut centroids = kmeans_pp_init(&points, n, d, k, &mut rng);

    let mut assign = vec![0usize; n];
    let mut sums = vec![0.0f64; k * d];
    let mut counts = vec![0usize; k];

    for _ in 0..cfg.iters {
        
        for p in 0..n {
            let x = &points[p * d..p * d + d];
            let (best, _) = nearest_centroid(x, &centroids, k, d);
            assign[p] = best;
        }

        for s in sums.iter_mut() {
            *s = 0.0;
        }
        for c in counts.iter_mut() {
            *c = 0;
        }
        for p in 0..n {
            let a = assign[p];
            counts[a] += 1;
            let base = a * d;
            let x = &points[p * d..p * d + d];
            for j in 0..d {
                sums[base + j] += x[j];
            }
        }

        for i in 0..k {
            if counts[i] > 0 {
                let inv = 1.0 / counts[i] as f64;
                let base = i * d;
                for j in 0..d {
                    centroids[base + j] = sums[base + j] * inv;
                }
            } else {
                
                let mut worst_p = 0usize;
                let mut worst_d = -1.0f64;
                for p in 0..n {
                    let a = assign[p];
                    let x = &points[p * d..p * d + d];
                    let c = &centroids[a * d..a * d + d];
                    let dist = sq_dist(x, c);
                    if dist > worst_d {
                        worst_d = dist;
                        worst_p = p;
                    }
                }
                let base = i * d;
                centroids[base..base + d]
                    .copy_from_slice(&points[worst_p * d..worst_p * d + d]);
                
            }
        }
    }

    Centroids {
        data: centroids,
        d,
        k,
    }
}

#[inline]
fn component_to_q12(x: f64) -> i32 {
    let scaled = (x * (1u32 << QUANTILE_SHIFT) as f64).round();
    let v = if scaled.is_finite() {
        scaled as i64
    } else if scaled > 0.0 {
        RECON_CLAMP_Q12 as i64
    } else {
        -(RECON_CLAMP_Q12 as i64)
    };
    v.clamp(-(RECON_CLAMP_Q12 as i64), RECON_CLAMP_Q12 as i64) as i32
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrozenCodebook {
    
    pub table: Vec<i32>,
    
    pub d: usize,
    
    pub k: usize,
}

impl FrozenCodebook {
    
    #[inline]
    pub fn reconstruct(&self, index: usize) -> &[i32] {
        &self.table[index * self.d..index * self.d + self.d]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.k
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.k == 0
    }
}

pub fn freeze_centroids(c: &Centroids) -> FrozenCodebook {
    let table: Vec<i32> = c.data.iter().map(|&x| component_to_q12(x)).collect();
    FrozenCodebook {
        table,
        d: c.d,
        k: c.k,
    }
}

pub fn train_state_vector_lut(
    samples: &[f32],
    l_bits: u32,
    d: usize,
    seed: u64,
    iters: usize,
) -> Vec<i32> {
    let k = 1usize << l_bits;
    let cfg = TrainConfig {
        d: d.max(1),
        k_centroids: k,
        iters,
        seed,
    };
    let centroids = train(samples, &cfg);
    freeze_centroids(&centroids).table
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv1a(table: &[i32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &v in table {
            for b in v.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    fn synth_gaussian(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        let mut out = Vec::with_capacity(n * d);
        for _ in 0..n * d {
            
            let u1 = rng.next_f64().max(1e-12);
            let u2 = rng.next_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            let z = r * (std::f64::consts::TAU * u2).cos();
            out.push(z as f32);
        }
        out
    }

    fn frozen_mse(fc: &FrozenCodebook, samples: &[f32]) -> f64 {
        let d = fc.d;
        let n = samples.len() / d;
        if n == 0 || fc.k == 0 {
            return 0.0;
        }
        let q_to_real = 1.0f64 / (1u32 << QUANTILE_SHIFT) as f64;
        let mut acc = 0.0f64;
        for p in 0..n {
            let x: Vec<f64> = samples[p * d..p * d + d].iter().map(|&w| w as f64).collect();
            
            let mut best_d = f64::INFINITY;
            for i in 0..fc.k {
                let c = fc.reconstruct(i);
                let mut dist = 0.0f64;
                for j in 0..d {
                    let cj = c[j] as f64 * q_to_real;
                    let diff = x[j] - cj;
                    dist += diff * diff;
                }
                if dist < best_d {
                    best_d = dist;
                }
            }
            acc += best_d;
        }
        acc / (n as f64 * d as f64)
    }

    #[test]
    fn d1_reduces_to_scalar_quantization() {
        
        let mut samples = Vec::new();
        for _ in 0..40 {
            samples.push(-2.0f32);
        }
        for _ in 0..40 {
            samples.push(0.0f32);
        }
        for _ in 0..40 {
            samples.push(3.0f32);
        }
        let cfg = TrainConfig {
            d: 1,
            k_centroids: 3,
            iters: 50,
            seed: 7,
        };
        let c = train(&samples, &cfg);
        assert_eq!(c.d, 1);
        assert_eq!(c.k, 3);
        
        let mut centres: Vec<f64> = (0..3).map(|i| c.centroid(i)[0]).collect();
        centres.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((centres[0] - (-2.0)).abs() < 1e-6, "got {centres:?}");
        assert!((centres[1] - 0.0).abs() < 1e-6, "got {centres:?}");
        assert!((centres[2] - 3.0).abs() < 1e-6, "got {centres:?}");

        let fc = c.freeze();
        assert_eq!(fc.d, 1);
        let q_to_real = 1.0f64 / (1u32 << QUANTILE_SHIFT) as f64;
        for i in 0..3 {
            let got = fc.reconstruct(i)[0] as f64 * q_to_real;
            assert!(
                (got - c.centroid(i)[0]).abs() <= q_to_real,
                "centroid {i}: frozen {got} vs float {}",
                c.centroid(i)[0]
            );
        }
    }

    #[test]
    fn training_reduces_mse_vs_random_init() {
        let d = 4;
        let k = 16;
        let samples = synth_gaussian(4000, d, 0xABCD_1234);
        let n = samples.len() / d;
        let points: Vec<f64> = samples.iter().map(|&w| w as f64).collect();

        let mut rng = SplitMix64::new(999);
        let mut random_book = vec![0.0f64; k * d];
        for i in 0..k {
            let p = rng.next_below(n);
            random_book[i * d..i * d + d].copy_from_slice(&points[p * d..p * d + d]);
        }
        let random = Centroids {
            data: random_book,
            d,
            k,
        };
        let random_mse = random.reconstruction_mse(&points);

        let cfg = TrainConfig {
            d,
            k_centroids: k,
            iters: 50,
            seed: 42,
        };
        let trained = train(&samples, &cfg);
        let trained_mse = trained.reconstruction_mse(&points);

        assert!(
            trained_mse < random_mse,
            "training did not reduce MSE: trained {trained_mse} vs random {random_mse}"
        );
        
        assert!(
            trained_mse < 0.9 * random_mse,
            "training reduced MSE only marginally: trained {trained_mse} vs random {random_mse}"
        );

        let frozen = freeze_centroids(&trained);
        let frozen_book_mse = frozen_mse(&frozen, &samples);
        assert!(
            frozen_book_mse < random_mse,
            "frozen book lost to random init: frozen {frozen_book_mse} vs random {random_mse}"
        );
    }

    #[test]
    fn freeze_lookup_within_q12_tolerance() {
        let d = 3;
        let k = 12;
        let samples = synth_gaussian(2000, d, 0x5151_2727);
        let cfg = TrainConfig {
            d,
            k_centroids: k,
            iters: 40,
            seed: 11,
        };
        let c = train(&samples, &cfg);
        let fc = freeze_centroids(&c);
        assert_eq!(fc.table.len(), k * d);
        assert_eq!(fc.len(), k);
        assert!(!fc.is_empty());

        let q_to_real = 1.0f64 / (1u32 << QUANTILE_SHIFT) as f64;
        for i in 0..k {
            let fl = c.centroid(i);
            let fz = fc.reconstruct(i);
            assert_eq!(fz.len(), d);
            for j in 0..d {
                
                let got = fz[j] as f64 * q_to_real;
                let want = fl[j].clamp(-6.0, 6.0);
                assert!(
                    (got - want).abs() <= q_to_real,
                    "centroid {i} comp {j}: frozen {got} vs float {want}"
                );
                
                assert!(fz[j].abs() <= RECON_CLAMP_Q12, "entry exceeds clamp: {}", fz[j]);
            }
        }

        let fc2 = freeze_centroids(&c);
        assert_eq!(fc.table, fc2.table);
    }

    #[test]
    fn two_runs_are_bit_identical() {
        let d = 5;
        let k = 32;
        
        let samples = synth_gaussian(6000, d, 0xDEAD_BEEF);
        let cfg = TrainConfig {
            d,
            k_centroids: k,
            iters: 60,
            seed: 2024,
        };

        let a = freeze_centroids(&train(&samples, &cfg));
        let b = freeze_centroids(&train(&samples, &cfg));

        assert_eq!(a.d, b.d);
        assert_eq!(a.k, b.k);
        assert_eq!(a.table, b.table, "frozen tables differ across identical runs");
        assert_eq!(fnv1a(&a.table), fnv1a(&b.table), "golden hash differs across runs");

        let ca = train(&samples, &cfg);
        let cb = train(&samples, &cfg);
        assert_eq!(ca.data, cb.data, "float centroids differ across identical runs");
    }

    #[test]
    fn splitmix64_stream_is_pinned() {
        let mut rng = SplitMix64::new(0);
        
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(rng.next_u64(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn empty_samples_yield_zero_book() {
        let cfg = TrainConfig {
            d: 4,
            k_centroids: 8,
            iters: 10,
            seed: 1,
        };
        let c = train(&[], &cfg);
        assert_eq!(c.k, 8);
        assert_eq!(c.d, 4);
        assert!(c.data.iter().all(|&x| x == 0.0));
        let fc = c.freeze();
        assert!(fc.table.iter().all(|&v| v == 0));
        assert_eq!(fc.table.len(), 32);
    }

    #[test]
    fn fewer_points_than_k_is_deterministic() {
        let d = 2;
        let k = 16;
        
        let samples: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let cfg = TrainConfig {
            d,
            k_centroids: k,
            iters: 20,
            seed: 3,
        };
        let a = train(&samples, &cfg);
        let b = train(&samples, &cfg);
        assert_eq!(a.data, b.data);
        assert_eq!(a.k, k);
    }
}
