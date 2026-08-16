//! Honest batch=1 Q80 decode-shape bandwidth control.
//!
//! Moves the same bytes, in the same access pattern, with NO model logic
//! (no codec, no matvec of packed weights). GPU time is only
//! `MTLCommandBuffer.GPUEndTime - GPUStartTime` after wait.
//!
//! Build:
//!   cargo build --profile release-fast -p hawking-core --example q80_decode_shape_bandwidth
//! Measure (GPU mutex required):
//!   ./tools/gpu_lane_lock.sh fs-occupancy \
//!     workspace/ops/build/rust/release-fast/examples/q80_decode_shape_bandwidth \
//!     --out receipts/ascent-2026-08-16/Q80_DECODE_SHAPE_BANDWIDTH.json

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("q80_decode_shape_bandwidth requires macOS").into())
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

#[cfg(target_os = "macos")]
mod macos {
    use serde_json::{json, Value};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    const SCHEMA: &str = "hawking.ascension.q80_decode_shape_bandwidth.v1";
    const PEAK_GBPS: f64 = 819.0;
    const WARMUP: usize = 2;
    const PAIRS: usize = 3;
    const FULL_THREADS: u32 = 256 * 60;
    const TG: u32 = 256;
    const PAGE: usize = 4096;

    const ACTIVE_WEIGHTS: u64 = 3_562_274_816;
    const ATTN_WEIGHTS: u64 = 1_539_833_856;
    const ROUTER_WEIGHTS: u64 = 50_331_648;
    const LM_HEAD_WEIGHTS: u64 = 311_164_928;
    const ORGAN_ELEMS: u64 = 512 * 2048;
    const Q80_LAYERS: u64 = 48;
    const Q80_TOP_K: u64 = 10;
    const Q80_EXPERTS: u32 = 512;
    const DELTANET_LAYERS: u64 = 36;
    const GQA_LAYERS: u64 = 12;
    const CBS_PER_TOKEN: u64 = 98;
    const DISPATCHES_PER_TOKEN: u64 = 1155;

    // Pack-report BPW. Storage complete_bpw is NOT active-decode BPW.
    const M15_GATE_BPW: f64 = 1.126922607421875;
    const M15_UP_BPW: f64 = 1.2918486054986715;
    const M15_DOWN_BPW: f64 = 1.2858703201636672;
    const M15_NE_BPW: f64 = 8.250600705299505;
    const S655_GATE_BPW: f64 = 0.14319598612686;
    const S655_UP_BPW: f64 = 0.3336358508095145;
    const S655_DOWN_BPW: f64 = 1.126922607421875;
    const S655_NE_BPW: f64 = 4.250595793629995;
    const Q4_BPW: f64 = 4.259241;
    const FS_PER_WEIGHT_AT_PEAK: f64 = 152.6252;

    // Measured Q4-vehicle token (TOKEN_NS_Q80.json). Not re-run here.
    const Q4_TOKEN_NS: u64 = 559_171_655;
    const Q4_GPU_NS: u64 = 231_968_522;
    const Q4_BYTES: u64 = 1_892_511_808;
    const Q4_FS: f64 = 156_970.3865879392;

    struct Spread {
        all_ns: Vec<u64>,
        all_gbps: Vec<f64>,
    }

    impl Spread {
        fn json(&self) -> Value {
            json!({
                "gpu_ns": self.all_ns,
                "gbps": self.all_gbps,
                "min_ns": self.all_ns.iter().copied().min(),
                "max_ns": self.all_ns.iter().copied().max(),
                "median_ns": median(&self.all_ns),
                "min_gbps": self.all_gbps.iter().copied().fold(f64::INFINITY, f64::min),
                "max_gbps": self.all_gbps.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                "median_gbps": gbps_of(median(&self.all_ns), &self.all_ns, &self.all_gbps),
            })
        }
    }

    fn gbps_of(med_ns: u64, ns: &[u64], gbps: &[f64]) -> f64 {
        ns.iter()
            .position(|&v| v == med_ns)
            .and_then(|i| gbps.get(i).copied())
            .unwrap_or(0.0)
    }

    fn set_u32(enc: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
        enc.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }

    fn gpu_ns(
        t: hawking_core::metal::MetalDispatchTiming,
        label: &str,
    ) -> Result<u64, Box<dyn Error>> {
        match (t.gpu_start_ns, t.gpu_end_ns) {
            (Some(s), Some(e)) if e > s => Ok(e - s),
            _ => Err(format!("{label}: MTLCommandBuffer GPUStartTime/GPUEndTime unavailable").into()),
        }
    }

    fn batch_gpu_ns(
        t: hawking_core::metal::MetalBatchTiming,
        label: &str,
    ) -> Result<u64, Box<dyn Error>> {
        match (t.gpu_start_ns, t.gpu_end_ns) {
            (Some(s), Some(e)) if e > s => Ok(e - s),
            _ => Err(format!("{label}: MTLCommandBuffer GPUStartTime/GPUEndTime unavailable").into()),
        }
    }

    fn median(values: &[u64]) -> u64 {
        let mut s = values.to_vec();
        s.sort_unstable();
        s[s.len() / 2]
    }

    fn gbps(bytes: u64, ns: u64) -> f64 {
        if ns == 0 {
            0.0
        } else {
            bytes as f64 / ns as f64
        }
    }

    fn align16(n: u64) -> u64 {
        (n + 15) & !15
    }

    fn bytes_from_bpw(weights: u64, bpw: f64) -> u64 {
        ((weights as f64) * bpw / 8.0).round() as u64
    }

    fn touch_pages(buf: &metal::Buffer, len: usize) {
        unsafe {
            let p = buf.contents() as *mut u8;
            let mut off = 0;
            while off < len {
                *p.add(off) = 0x5a;
                off += PAGE;
            }
            if len > 0 {
                *p.add(len - 1) = 0xa5;
            }
        }
    }

    struct VehicleBudget {
        name: &'static str,
        storage_complete_bpw: f64,
        expert_bpw: f64,
        nonexpert_bpw: f64,
        attn_bytes: u64,
        router_bytes: u64,
        routed_bytes: u64,
        shared_bytes: u64,
        lm_head_bytes: u64,
        kv_new_bytes: u64,
        activation_temp_bytes: u64,
        expert_table_meta_bytes: u64,
        gate_organ: u64,
        up_organ: u64,
        down_organ: u64,
        triplet: u64,
    }

    impl VehicleBudget {
        fn build(
            name: &'static str,
            storage: f64,
            gate: f64,
            up: f64,
            down: f64,
            ne: f64,
        ) -> Self {
            let gate_organ = align16(bytes_from_bpw(ORGAN_ELEMS, gate));
            let up_organ = align16(bytes_from_bpw(ORGAN_ELEMS, up));
            let down_organ = align16(bytes_from_bpw(ORGAN_ELEMS, down));
            let triplet = gate_organ + up_organ + down_organ;
            let expert_bpw = (gate + up + down) / 3.0;
            Self {
                name,
                storage_complete_bpw: storage,
                expert_bpw,
                nonexpert_bpw: ne,
                attn_bytes: align16(bytes_from_bpw(ATTN_WEIGHTS, ne)),
                router_bytes: align16(bytes_from_bpw(ROUTER_WEIGHTS, ne)),
                routed_bytes: Q80_LAYERS * Q80_TOP_K * triplet,
                shared_bytes: Q80_LAYERS * triplet,
                lm_head_bytes: align16(bytes_from_bpw(LM_HEAD_WEIGHTS, ne)),
                kv_new_bytes: GQA_LAYERS * 512 * 2 * 2,
                activation_temp_bytes: 16_711_680,
                expert_table_meta_bytes: u64::from(Q80_EXPERTS) * 64,
                gate_organ,
                up_organ,
                down_organ,
                triplet,
            }
        }

        fn weight_bytes(&self) -> u64 {
            self.attn_bytes
                + self.router_bytes
                + self.routed_bytes
                + self.shared_bytes
                + self.lm_head_bytes
        }

        fn total_moved(&self) -> u64 {
            self.weight_bytes()
                + self.kv_new_bytes
                + self.activation_temp_bytes
                + self.expert_table_meta_bytes
        }

        fn active_bpw(&self) -> f64 {
            self.weight_bytes() as f64 * 8.0 / ACTIVE_WEIGHTS as f64
        }

        fn json(&self) -> Value {
            let w = self.weight_bytes();
            let t = self.total_moved();
            json!({
                "name": self.name,
                "storage_complete_bpw": self.storage_complete_bpw,
                "expert_mean_bpw": self.expert_bpw,
                "nonexpert_bpw": self.nonexpert_bpw,
                "active_decode_bpw": self.active_bpw(),
                "storage_vs_active_note": "complete_physical_bpw averages unused experts (97% of stored mass). Decode only moves top-10+shared experts plus ALL nonexpert active weights.",
                "classes": {
                    "weights_attention": {"bytes": self.attn_bytes, "pct_of_weights": self.attn_bytes as f64 / w as f64},
                    "weights_router": {"bytes": self.router_bytes, "pct_of_weights": self.router_bytes as f64 / w as f64},
                    "weights_routed_experts": {"bytes": self.routed_bytes, "pct_of_weights": self.routed_bytes as f64 / w as f64},
                    "weights_shared_expert": {"bytes": self.shared_bytes, "pct_of_weights": self.shared_bytes as f64 / w as f64},
                    "weights_lm_head": {"bytes": self.lm_head_bytes, "pct_of_weights": self.lm_head_bytes as f64 / w as f64},
                    "kv_new_gqa_f16": {"bytes": self.kv_new_bytes, "note": "12 GQA layers × 512 dim × k+v × f16, seq increment = 1"},
                    "activations_temp": {"bytes": self.activation_temp_bytes, "source": "TOKEN_NS_Q80 TEMP_BYTES_PER_TOKEN"},
                    "expert_table_metadata": {"bytes": self.expert_table_meta_bytes, "note": "512-way address table; payloads billed under routed/shared"},
                },
                "weight_bytes_per_token": w,
                "total_bytes_per_token": t,
                "organ_bytes": {
                    "gate": self.gate_organ,
                    "up": self.up_organ,
                    "down": self.down_organ,
                    "triplet": self.triplet,
                },
                "vs_peak_819": {
                    "weight_floor_ns": (w as f64 / 819e9 * 1e9) as u64,
                    "total_floor_ns": (t as f64 / 819e9 * 1e9) as u64,
                    "fs_per_weight_at_unity": FS_PER_WEIGHT_AT_PEAK * self.active_bpw(),
                }
            })
        }
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let mut out = PathBuf::from("receipts/ascent-2026-08-16/Q80_DECODE_SHAPE_BANDWIDTH.json");
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--out" => out = PathBuf::from(args.next().ok_or("missing --out")?),
                other => return Err(format!("unsupported option {other}").into()),
            }
        }

        let m15 = VehicleBudget::build(
            "mixed-1p5-v1",
            1.4444457,
            M15_GATE_BPW,
            M15_UP_BPW,
            M15_DOWN_BPW,
            M15_NE_BPW,
        );
        let s655 = VehicleBudget::build(
            "mixed-sub655-v1",
            0.6462038524865996,
            S655_GATE_BPW,
            S655_UP_BPW,
            S655_DOWN_BPW,
            S655_NE_BPW,
        );
        let q4 = VehicleBudget::build(
            "uniform-q4-group64-v1 (DE-AUTHORISED vehicle; budget only, no Q4 kernel)",
            Q4_BPW,
            Q4_BPW,
            Q4_BPW,
            Q4_BPW,
            Q4_BPW,
        );

        eprintln!("compiling Metal library (first pipeline lookup)...");
        let ctx = hawking_core::metal::MetalContext::new()?;
        let device = ctx.device_name();
        eprintln!("device={device}");
        eprintln!(
            "active-decode BPW mixed-1p5={:.4} (storage {:.4}) mixed-sub655={:.4} (storage {:.4})",
            m15.active_bpw(),
            m15.storage_complete_bpw,
            s655.active_bpw(),
            s655.storage_complete_bpw
        );

        eprintln!("[1/8] reuse 64 MiB control");
        let reuse = measure_reuse_control(&ctx)?;
        eprintln!("[2/8] unique-once sweep");
        let unique = measure_unique_once_sweep(&ctx)?;
        eprintln!("[3/8] launch geometry");
        let geometry = measure_launch_geometry(&ctx, s655.triplet)?;
        eprintln!("[4/8] gather vs sequential");
        let gather = measure_gather(&ctx, &s655)?;
        eprintln!("[5/8] dispatch / CB tax");
        let dispatch = measure_dispatch_tax(&ctx)?;
        eprintln!("[6/8] arithmetic intensity");
        let intensity = measure_intensity(&ctx)?;
        eprintln!("[7/8] residency");
        let residency = measure_residency(&ctx, s655.routed_bytes.min(256 * 1024 * 1024))?;
        eprintln!("[8/8] token-shaped mixed-1p5 then mixed-sub655");
        let token_m15 = measure_token_shape(&ctx, &m15)?;
        let token_s655 = measure_token_shape(&ctx, &s655)?;

        let control_unique_med = unique
            .get("unique_once_512mib")
            .and_then(|v| v.get("median_gbps"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let control_reuse_lo = reuse
            .get("sequential")
            .and_then(|v| v.get("median_gbps"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let control_reuse_hi = reuse
            .get("conflict")
            .and_then(|v| v.get("median_gbps"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let honest_ceiling = control_unique_med.max(1.0);
        let req = restate_requirement(&m15, &s655, &q4, honest_ceiling, control_reuse_lo, control_reuse_hi);

        let caps = rank_caps(
            &dispatch,
            &geometry,
            &gather,
            &residency,
            &intensity,
            &token_s655,
            &s655,
            honest_ceiling,
        );

        let receipt = json!({
            "schema": SCHEMA,
            "lane": "fs-occupancy",
            "date": "2026-08-16",
            "measurement_label": "DIRTY_ENGINEERING",
            "device_name": device,
            "gpu_time_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
            "q4_kernels_benchmarked": false,
            "claim_boundary": {
                "no_model_logic": true,
                "no_q4_kernel": true,
                "dense_weight_not_materialized_as_decode": true,
                "unique_once_is_the_honest_decode_control": true,
                "reuse_64mib_x_iters_is_a_cache_friendly_roofline_not_decode": true,
                "storage_complete_bpw_is_not_active_decode_bpw": true,
            },
            "byte_budget": {
                "mixed_1p5_v1": m15.json(),
                "mixed_sub655_v1": s655.json(),
                "q4_vehicle_budget_only": q4.json(),
                "q4_ledger_cross_check_bytes": Q4_BYTES,
            },
            "controls": {
                "reuse_64mib_read_reduce": reuse,
                "unique_once_sweep": unique,
                "honest_decode_ceiling_gbps": honest_ceiling,
                "reuse_band_gbps": [control_reuse_lo, control_reuse_hi],
                "published_peak_gbps": PEAK_GBPS,
            },
            "launch_geometry": geometry,
            "gather_vs_sequential": gather,
            "dispatch_and_cb": dispatch,
            "arithmetic_intensity": intensity,
            "residency": residency,
            "token_shape": {
                "mixed_1p5_v1": token_m15,
                "mixed_sub655_v1": token_s655,
            },
            "ranked_occupancy_caps": caps,
            "sub_100_fs_requirement": req,
            "historical_0_135_pct": {
                "what_it_was": "212.53 fs mixed-storage floor / 156970 fs Q4-runtime",
                "why_it_is_a_category_error": "numerator used mixed-1p5 STORAGE complete_bpw (1.392467) as if it were active-decode BPW; denominator is the Q4 vehicle token. Those are different vehicles and different masses.",
                "q4_efficiency_vs_peak_pct": 100.0 * FS_PER_WEIGHT_AT_PEAK * Q4_BPW / Q4_FS,
                "q4_gpu_only_gbps": gbps(Q4_BYTES, Q4_GPU_NS),
                "q4_wall_gbps": gbps(Q4_BYTES, Q4_TOKEN_NS),
            }
        });

        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out, serde_json::to_vec_pretty(&receipt)?)?;
        eprintln!("wrote {}", out.display());
        Ok(())
    }

    fn measure_reuse_control(ctx: &hawking_core::metal::MetalContext) -> Result<Value, Box<dyn Error>> {
        let probe_bytes: usize = 64 * 1024 * 1024;
        let iters: u32 = 4096;
        let buf = ctx.new_buffer_checked(probe_bytes)?;
        touch_pages(&buf, probe_bytes);
        let outb = ctx.new_buffer_checked(FULL_THREADS as usize * 4)?;
        let moved = u64::from(FULL_THREADS) * u64::from(iters) * 16;
        let time = |stride: u32| -> Result<u64, Box<dyn Error>> {
            gpu_ns(
                ctx.dispatch_threads_timed(
                    "dram_row_locality_read_reduce",
                    (FULL_THREADS, 1, 1),
                    (TG, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&buf), 0);
                        enc.set_buffer(1, Some(&outb), 0);
                        set_u32(enc, 2, probe_bytes as u32);
                        set_u32(enc, 3, stride);
                        set_u32(enc, 4, iters);
                    },
                )?,
                "reuse",
            )
        };
        let seq_stride = FULL_THREADS * 16;
        let conflict_stride = 8192u32 + 64;
        for _ in 0..WARMUP {
            let _ = time(seq_stride)?;
            let _ = time(conflict_stride)?;
        }
        let mut seq = Vec::new();
        let mut cfl = Vec::new();
        for _ in 0..PAIRS {
            seq.push(time(seq_stride)?);
            cfl.push(time(conflict_stride)?);
        }
        Ok(json!({
            "working_set_bytes": probe_bytes,
            "bytes_moved_per_dispatch": moved,
            "note": "64 MiB reused 4096 times. Cache-friendly roofline, NOT decode traffic.",
            "sequential": spread_from(&seq, moved).json(),
            "conflict": spread_from(&cfl, moved).json(),
        }))
    }

    fn measure_unique_once_sweep(
        ctx: &hawking_core::metal::MetalContext,
    ) -> Result<Value, Box<dyn Error>> {
        let sizes_mib = [64u64, 256, 512, 1024];
        let mut map = serde_json::Map::new();
        let max = (*sizes_mib.iter().max().unwrap() as usize) * 1024 * 1024;
        let buf = ctx.new_buffer_checked(max)?;
        touch_pages(&buf, max);
        let outb = ctx.new_buffer_checked(FULL_THREADS as usize * 4)?;
        for mib in sizes_mib {
            let nbytes = (mib as usize) * 1024 * 1024;
            let time = || -> Result<u64, Box<dyn Error>> {
                gpu_ns(
                    ctx.dispatch_threads_timed(
                        "q80_decode_shape_unique_once",
                        (FULL_THREADS, 1, 1),
                        (TG, 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&buf), 0);
                            enc.set_buffer(1, Some(&outb), 0);
                            set_u32(enc, 2, nbytes as u32);
                        },
                    )?,
                    "unique",
                )
            };
            for _ in 0..WARMUP {
                let _ = time()?;
            }
            let mut ns = Vec::new();
            for _ in 0..PAIRS {
                ns.push(time()?);
            }
            map.insert(
                format!("unique_once_{mib}mib"),
                json!({
                    "bytes": nbytes,
                    "threads": FULL_THREADS,
                    "spread": spread_from(&ns, nbytes as u64).json(),
                    "median_gbps": gbps(nbytes as u64, median(&ns)),
                    "median_ns": median(&ns),
                    "all_ns": ns,
                }),
            );
        }
        Ok(Value::Object(map))
    }

    fn measure_launch_geometry(
        ctx: &hawking_core::metal::MetalContext,
        organ_bytes: u64,
    ) -> Result<Value, Box<dyn Error>> {
        let organ = align16(organ_bytes.max(16)) as usize;
        let n_organs = 32u32;
        let slab = organ * n_organs as usize;
        let buf = ctx.new_buffer_checked(slab)?;
        touch_pages(&buf, slab);
        let offsets: Vec<u32> = (0..n_organs).map(|i| i * organ as u32).collect();
        let off_buf = ctx.new_buffer_with_bytes_checked(bytemuck_u32(&offsets))?;
        let configs = [
            ("one_organ_512_threads", 1u32, 512u32),
            ("ten_organs_5120_threads", 10, 512),
            ("thirty_organs_15360_threads", 30, 512),
            ("one_organ_full_threads_capped_by_bytes", 1, FULL_THREADS),
        ];
        let mut out = serde_json::Map::new();
        for (name, norg, tpo) in configs {
            let grid = norg * tpo;
            let outb = ctx.new_buffer_checked(grid as usize * 4)?;
            let bytes = norg as u64 * organ as u64;
            let time = || -> Result<u64, Box<dyn Error>> {
                gpu_ns(
                    ctx.dispatch_threads_timed(
                        "q80_decode_shape_gather",
                        (grid, 1, 1),
                        (TG.min(grid).max(32), 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&buf), 0);
                            enc.set_buffer(1, Some(&off_buf), 0);
                            enc.set_buffer(2, Some(&outb), 0);
                            set_u32(enc, 3, norg);
                            set_u32(enc, 4, organ as u32);
                            set_u32(enc, 5, tpo);
                        },
                    )?,
                    name,
                )
            };
            for _ in 0..WARMUP {
                let _ = time()?;
            }
            let mut ns = Vec::new();
            for _ in 0..PAIRS {
                ns.push(time()?);
            }
            out.insert(
                name.to_string(),
                json!({
                    "n_organs": norg,
                    "threads_per_organ": tpo,
                    "grid": grid,
                    "bytes": bytes,
                    "occupancy_vs_15360": grid as f64 / FULL_THREADS as f64,
                    "spread": spread_from(&ns, bytes).json(),
                }),
            );
        }
        Ok(Value::Object(out))
    }

    fn measure_gather(
        ctx: &hawking_core::metal::MetalContext,
        v: &VehicleBudget,
    ) -> Result<Value, Box<dyn Error>> {
        let organ = v.triplet as usize;
        let n_table = Q80_EXPERTS as usize;
        let slab = organ * n_table;
        let buf = ctx.new_buffer_checked(slab)?;
        touch_pages(&buf, slab);
        let seq: Vec<u32> = (0..Q80_TOP_K as u32).map(|i| i * organ as u32).collect();
        let mut gather: Vec<u32> = Vec::new();
        // Deterministic 10-of-512 scatter, large stride (not a random draw).
        for i in 0..Q80_TOP_K as u32 {
            gather.push(((i * 47 + 3) % Q80_EXPERTS) * organ as u32);
        }
        let seq_buf = ctx.new_buffer_with_bytes_checked(bytemuck_u32(&seq))?;
        let gth_buf = ctx.new_buffer_with_bytes_checked(bytemuck_u32(&gather))?;
        let tpo = 512u32;
        let grid = Q80_TOP_K as u32 * tpo;
        let outb = ctx.new_buffer_checked(grid as usize * 4)?;
        let bytes = Q80_TOP_K * organ as u64;
        let time = |offs: &metal::Buffer| -> Result<u64, Box<dyn Error>> {
            gpu_ns(
                ctx.dispatch_threads_timed(
                    "q80_decode_shape_gather",
                    (grid, 1, 1),
                    (TG, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&buf), 0);
                        enc.set_buffer(1, Some(offs), 0);
                        enc.set_buffer(2, Some(&outb), 0);
                        set_u32(enc, 3, Q80_TOP_K as u32);
                        set_u32(enc, 4, organ as u32);
                        set_u32(enc, 5, tpo);
                    },
                )?,
                "gather",
            )
        };
        for _ in 0..WARMUP {
            let _ = time(&seq_buf)?;
            let _ = time(&gth_buf)?;
        }
        let mut a = Vec::new();
        let mut b = Vec::new();
        for _ in 0..PAIRS {
            a.push(time(&seq_buf)?);
            b.push(time(&gth_buf)?);
        }
        Ok(json!({
            "organ_bytes": organ,
            "table_experts": n_table,
            "top_k": Q80_TOP_K,
            "bytes": bytes,
            "sequential_10": spread_from(&a, bytes).json(),
            "scattered_10_of_512": spread_from(&b, bytes).json(),
        }))
    }

    fn measure_dispatch_tax(ctx: &hawking_core::metal::MetalContext) -> Result<Value, Box<dyn Error>> {
        let n_nop = 256usize;
        let outb = ctx.new_buffer_checked(n_nop * 4)?;
        let nop_one = || -> Result<(u64, u64), Box<dyn Error>> {
            let host = Instant::now();
            let t = ctx.dispatch_threads_timed(
                "q80_decode_shape_nop",
                (n_nop as u32, 1, 1),
                (TG, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&outb), 0);
                },
            )?;
            Ok((gpu_ns(t, "nop")?, host.elapsed().as_nanos() as u64))
        };
        for _ in 0..WARMUP {
            let _ = nop_one()?;
        }
        let mut one_gpu = Vec::new();
        let mut one_host = Vec::new();
        for _ in 0..PAIRS {
            let (g, h) = nop_one()?;
            one_gpu.push(g);
            one_host.push(h);
        }

        let nops_in_one_cb = |n: u64| -> Result<u64, Box<dyn Error>> {
            batch_gpu_ns(
                ctx.dispatch_batch_timed(|batch| {
                    for _ in 0..n {
                        batch.dispatch_threads(
                            "q80_decode_shape_nop",
                            (n_nop as u32, 1, 1),
                            (TG, 1, 1),
                            |enc| {
                                enc.set_buffer(0, Some(&outb), 0);
                            },
                        )?;
                    }
                    Ok(())
                })?,
                "nop-batch",
            )
        };
        for _ in 0..WARMUP {
            let _ = nops_in_one_cb(12)?;
        }
        let mut batch12 = Vec::new();
        let mut batch1155 = Vec::new();
        for _ in 0..PAIRS {
            batch12.push(nops_in_one_cb(12)?);
        }
        for _ in 0..WARMUP {
            let _ = nops_in_one_cb(DISPATCHES_PER_TOKEN)?;
        }
        for _ in 0..PAIRS {
            batch1155.push(nops_in_one_cb(DISPATCHES_PER_TOKEN)?);
        }

        let ninety_eight_cbs = || -> Result<(u64, u64), Box<dyn Error>> {
            let host = Instant::now();
            let mut gpu = 0u64;
            for _ in 0..CBS_PER_TOKEN {
                gpu = gpu.saturating_add(nops_in_one_cb(12)?);
            }
            Ok((gpu, host.elapsed().as_nanos() as u64))
        };
        for _ in 0..WARMUP {
            let _ = ninety_eight_cbs()?;
        }
        let mut cbs_gpu = Vec::new();
        let mut cbs_host = Vec::new();
        for _ in 0..PAIRS {
            let (g, h) = ninety_eight_cbs()?;
            cbs_gpu.push(g);
            cbs_host.push(h);
        }

        Ok(json!({
            "one_nop_cb": {
                "gpu_ns": one_gpu,
                "host_ns": one_host,
                "median_gpu_ns": median(&one_gpu),
                "median_host_ns": median(&one_host),
                "projected_1155_serial_cbs_gpu_ns": median(&one_gpu).saturating_mul(DISPATCHES_PER_TOKEN),
                "projected_1155_serial_cbs_host_ns": median(&one_host).saturating_mul(DISPATCHES_PER_TOKEN),
                "projected_98_serial_cbs_gpu_ns": median(&one_gpu).saturating_mul(CBS_PER_TOKEN),
                "projected_98_serial_cbs_host_ns": median(&one_host).saturating_mul(CBS_PER_TOKEN),
            },
            "twelve_nops_one_cb": {
                "gpu_ns": batch12,
                "median_gpu_ns": median(&batch12),
            },
            "eleven55_nops_one_cb": {
                "gpu_ns": batch1155,
                "median_gpu_ns": median(&batch1155),
            },
            "ninety_eight_cbs_x_12_nops": {
                "gpu_ns": cbs_gpu,
                "host_ns": cbs_host,
                "median_gpu_ns": median(&cbs_gpu),
                "median_host_ns": median(&cbs_host),
                "median_host_minus_gpu_ns": median(&cbs_host).saturating_sub(median(&cbs_gpu)),
            },
            "topology": {
                "production_cbs": CBS_PER_TOKEN,
                "production_dispatches": DISPATCHES_PER_TOKEN,
            }
        }))
    }

    fn measure_intensity(ctx: &hawking_core::metal::MetalContext) -> Result<Value, Box<dyn Error>> {
        let rows = 512u32;
        let cols = 2048u32;
        let w_bytes = rows as usize * cols as usize * 4;
        let w = ctx.new_buffer_checked(w_bytes)?;
        touch_pages(&w, w_bytes);
        let x = ctx.new_buffer_checked(cols as usize * 4)?;
        touch_pages(&x, cols as usize * 4);
        let out_l = ctx.new_buffer_checked(rows as usize * 4)?;
        let out_f = ctx.new_buffer_checked(rows as usize * 4)?;
        let offs = vec![0u32];
        let off_buf = ctx.new_buffer_with_bytes_checked(bytemuck_u32(&offs))?;
        let time_load = || -> Result<u64, Box<dyn Error>> {
            gpu_ns(
                ctx.dispatch_threads_timed(
                    "q80_decode_shape_gather",
                    (rows, 1, 1),
                    (TG, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&w), 0);
                        enc.set_buffer(1, Some(&off_buf), 0);
                        enc.set_buffer(2, Some(&out_l), 0);
                        set_u32(enc, 3, 1);
                        set_u32(enc, 4, w_bytes as u32);
                        set_u32(enc, 5, rows);
                    },
                )?,
                "load",
            )
        };
        let time_fma = || -> Result<u64, Box<dyn Error>> {
            gpu_ns(
                ctx.dispatch_threads_timed(
                    "q80_decode_shape_fma",
                    (rows, 1, 1),
                    (TG, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&w), 0);
                        enc.set_buffer(1, Some(&x), 0);
                        enc.set_buffer(2, Some(&out_f), 0);
                        set_u32(enc, 3, cols);
                    },
                )?,
                "fma",
            )
        };
        for _ in 0..WARMUP {
            let _ = time_load()?;
            let _ = time_fma()?;
        }
        let mut a = Vec::new();
        let mut b = Vec::new();
        for _ in 0..PAIRS {
            a.push(time_load()?);
            b.push(time_fma()?);
        }
        Ok(json!({
            "rows": rows,
            "cols": cols,
            "bytes": w_bytes,
            "launch": "512 threads, one thread per row",
            "load_only": spread_from(&a, w_bytes as u64).json(),
            "load_plus_fma": spread_from(&b, w_bytes as u64).json(),
            "fma_minus_load_median_ns": median(&b).saturating_sub(median(&a)),
        }))
    }

    fn measure_residency(
        ctx: &hawking_core::metal::MetalContext,
        bytes: u64,
    ) -> Result<Value, Box<dyn Error>> {
        let nbytes = bytes.max(16) as usize;
        let outb = ctx.new_buffer_checked(FULL_THREADS as usize * 4)?;
        let mut cold = Vec::new();
        let mut warm = Vec::new();
        let mut host_memcpy = Vec::new();
        for _ in 0..PAIRS {
            let cold_buf = ctx.new_buffer_checked(nbytes)?;
            let ns = gpu_ns(
                ctx.dispatch_threads_timed(
                    "q80_decode_shape_unique_once",
                    (FULL_THREADS, 1, 1),
                    (TG, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&cold_buf), 0);
                        enc.set_buffer(1, Some(&outb), 0);
                        set_u32(enc, 2, nbytes as u32);
                    },
                )?,
                "cold",
            )?;
            cold.push(ns);

            touch_pages(&cold_buf, nbytes);
            let mut wrep = Vec::new();
            for _ in 0..WARMUP {
                let _ = gpu_ns(
                    ctx.dispatch_threads_timed(
                        "q80_decode_shape_unique_once",
                        (FULL_THREADS, 1, 1),
                        (TG, 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&cold_buf), 0);
                            enc.set_buffer(1, Some(&outb), 0);
                            set_u32(enc, 2, nbytes as u32);
                        },
                    )?,
                    "warm-prep",
                )?;
            }
            wrep.push(gpu_ns(
                ctx.dispatch_threads_timed(
                    "q80_decode_shape_unique_once",
                    (FULL_THREADS, 1, 1),
                    (TG, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&cold_buf), 0);
                        enc.set_buffer(1, Some(&outb), 0);
                        set_u32(enc, 2, nbytes as u32);
                    },
                )?,
                "warm",
            )?);
            warm.push(wrep[0]);

            let src = vec![0x5au8; nbytes.min(64 * 1024 * 1024)];
            let mut dst = vec![0u8; src.len()];
            let t0 = Instant::now();
            dst.copy_from_slice(&src);
            std::hint::black_box(&dst);
            let took = t0.elapsed().as_nanos() as u64;
            let scaled = (took as f64 * nbytes as f64 / src.len() as f64) as u64;
            host_memcpy.push(scaled);
        }
        Ok(json!({
            "bytes": nbytes,
            "cold_first_gpu_read": spread_from(&cold, nbytes as u64).json(),
            "warm_after_page_touch": spread_from(&warm, nbytes as u64).json(),
            "host_memcpy_scaled_ns": host_memcpy,
            "host_memcpy_median_ns": median(&host_memcpy),
            "host_memcpy_median_gbps": gbps(nbytes as u64, median(&host_memcpy)),
            "note": "cold = new MTLBuffer, no CPU touch. warm = every 4 KiB page poked then GPU reread. host memcpy is a CPU Instant, labeled as such, analog of compact_expert_slab_pack.",
        }))
    }

    fn measure_token_shape(
        ctx: &hawking_core::metal::MetalContext,
        v: &VehicleBudget,
    ) -> Result<Value, Box<dyn Error>> {
        let attn_dn = align16(v.attn_bytes * DELTANET_LAYERS / (DELTANET_LAYERS + GQA_LAYERS)) as usize;
        let attn_gqa = align16(v.attn_bytes - attn_dn as u64) as usize;
        let dn_layer = align16((attn_dn as u64 / DELTANET_LAYERS) + (v.router_bytes / Q80_LAYERS)) as usize;
        let gqa_layer = align16((attn_gqa as u64 / GQA_LAYERS) + (v.router_bytes / Q80_LAYERS)) as usize;
        let lm = v.lm_head_bytes as usize;
        let organ = v.triplet as usize;
        let table = organ * Q80_EXPERTS as usize;
        // Unique working set: attention + router + one expert table + lm_head.
        // Layers walk this with advancing Metal buffer offsets so we do not
        // reread the same 64 MiB and credit a cache hit as DRAM bandwidth.
        let need = (attn_dn + attn_gqa + v.router_bytes as usize + table + lm + PAGE)
            .max(1024 * 1024 * 512);
        let buf = ctx.new_buffer_checked(need)?;
        touch_pages(&buf, need);
        let out_full = ctx.new_buffer_checked(FULL_THREADS as usize * 4)?;
        let tpo = 512u32;
        let grid_moe = Q80_TOP_K as u32 * tpo;
        let out_moe = ctx.new_buffer_checked(grid_moe as usize * 4)?;
        let seq: Vec<u32> = (0..Q80_TOP_K as u32)
            .map(|i| ((i * 47 + 3) % Q80_EXPERTS) * organ as u32)
            .collect();
        let off_buf = ctx.new_buffer_with_bytes_checked(bytemuck_u32(&seq))?;

        let unique_at = |offset: u64, nbytes: usize| -> Result<u64, Box<dyn Error>> {
            let off = offset.min((need.saturating_sub(nbytes.max(16))) as u64);
            gpu_ns(
                ctx.dispatch_threads_timed(
                    "q80_decode_shape_unique_once",
                    (FULL_THREADS, 1, 1),
                    (TG, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&buf), off);
                        enc.set_buffer(1, Some(&out_full), 0);
                        set_u32(enc, 2, nbytes as u32);
                    },
                )?,
                "token-unique",
            )
        };
        let prefix_cb = |offset: u64, nbytes: usize| -> Result<u64, Box<dyn Error>> {
            let off = offset.min((need.saturating_sub(nbytes.max(16))) as u64);
            batch_gpu_ns(
                ctx.dispatch_batch_timed(|batch| {
                    batch.dispatch_threads(
                        "q80_decode_shape_unique_once",
                        (FULL_THREADS, 1, 1),
                        (TG, 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&buf), off);
                            enc.set_buffer(1, Some(&out_full), 0);
                            set_u32(enc, 2, nbytes as u32);
                        },
                    )?;
                    Ok(())
                })?,
                "token-prefix",
            )
        };
        let suffix_cb = || -> Result<u64, Box<dyn Error>> {
            batch_gpu_ns(
                ctx.dispatch_batch_timed(|batch| {
                    batch.dispatch_threads(
                        "q80_decode_shape_gather",
                        (grid_moe, 1, 1),
                        (TG, 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&buf), 0);
                            enc.set_buffer(1, Some(&off_buf), 0);
                            enc.set_buffer(2, Some(&out_moe), 0);
                            set_u32(enc, 3, Q80_TOP_K as u32);
                            set_u32(enc, 4, organ as u32);
                            set_u32(enc, 5, tpo);
                        },
                    )?;
                    batch.dispatch_threads(
                        "q80_decode_shape_unique_once",
                        (FULL_THREADS, 1, 1),
                        (TG, 1, 1),
                        |enc| {
                            enc.set_buffer(0, Some(&buf), 0);
                            enc.set_buffer(1, Some(&out_full), 0);
                            set_u32(enc, 2, organ as u32);
                        },
                    )?;
                    Ok(())
                })?,
                "token-suffix",
            )
        };

        let run_token = || -> Result<Value, Box<dyn Error>> {
            let host = Instant::now();
            let mut gpu = 0u64;
            let mut cursor = 0u64;
            let mut dn = 0u64;
            for _ in 0..DELTANET_LAYERS {
                dn = dn.saturating_add(prefix_cb(cursor, dn_layer)?);
                cursor = cursor.saturating_add(dn_layer as u64);
            }
            let mut gqa = 0u64;
            for _ in 0..GQA_LAYERS {
                gqa = gqa.saturating_add(prefix_cb(cursor, gqa_layer)?);
                cursor = cursor.saturating_add(gqa_layer as u64);
            }
            let mut moe = 0u64;
            for _ in 0..Q80_LAYERS {
                moe = moe.saturating_add(suffix_cb()?);
            }
            let head = unique_at(cursor, lm.max(16))?;
            cursor = cursor.saturating_add(lm as u64);
            let embed = unique_at(cursor, 4096)?;
            gpu = gpu + dn + gqa + moe + head + embed;
            let host_ns = host.elapsed().as_nanos() as u64;
            let bytes = v.total_moved();
            Ok(json!({
                "gpu_ns": gpu,
                "host_ns": host_ns,
                "host_minus_gpu_ns": host_ns.saturating_sub(gpu),
                "bytes": bytes,
                "gbps_gpu": gbps(bytes, gpu),
                "gbps_host_wall": gbps(bytes, host_ns),
                "parts": {
                    "deltanet_prefix_gpu_ns": dn,
                    "gqa_prefix_gpu_ns": gqa,
                    "moe_suffix_gpu_ns": moe,
                    "lm_head_gpu_ns": head,
                    "embed_gpu_ns": embed,
                },
                "command_buffers": CBS_PER_TOKEN,
            }))
        };

        for _ in 0..WARMUP {
            let _ = run_token()?;
        }
        let mut reps = Vec::new();
        for _ in 0..PAIRS {
            reps.push(run_token()?);
        }
        let gpu_all: Vec<u64> = reps
            .iter()
            .filter_map(|r| r.get("gpu_ns").and_then(|v| v.as_u64()))
            .collect();
        let host_all: Vec<u64> = reps
            .iter()
            .filter_map(|r| r.get("host_ns").and_then(|v| v.as_u64()))
            .collect();
        let one_bytes = v.weight_bytes().min(need as u64) as usize;
        let one_shot = unique_at(0, one_bytes)?;
        Ok(json!({
            "vehicle": v.name,
            "active_decode_bpw": v.active_bpw(),
            "weight_bytes": v.weight_bytes(),
            "total_bytes": v.total_moved(),
            "working_set_bytes": need,
            "reps": reps,
            "median_gpu_ns": median(&gpu_all),
            "median_host_ns": median(&host_all),
            "median_gpu_gbps": gbps(v.total_moved(), median(&gpu_all)),
            "same_weight_bytes_one_dispatch_gpu_ns": one_shot,
            "same_weight_bytes_one_dispatch_gbps": gbps(one_bytes as u64, one_shot),
        }))
    }

    fn restate_requirement(
        m15: &VehicleBudget,
        s655: &VehicleBudget,
        q4: &VehicleBudget,
        unique_ceiling: f64,
        reuse_lo: f64,
        reuse_hi: f64,
    ) -> Value {
        let eff_unique = unique_ceiling / PEAK_GBPS;
        let eff_reuse_lo = reuse_lo / PEAK_GBPS;
        let eff_reuse_hi = reuse_hi / PEAK_GBPS;
        let bpw_for = |eff: f64| 100.0 / (FS_PER_WEIGHT_AT_PEAK / eff.max(1e-12));
        let fs_at = |bpw: f64, eff: f64| FS_PER_WEIGHT_AT_PEAK * bpw / eff.max(1e-12);
        json!({
            "law": "fs_per_weight = 152.6252 * ACTIVE_decode_BPW / efficiency",
            "efficiency_definition": "achieved_gbps / 819",
            "measured_unique_once_ceiling_gbps": unique_ceiling,
            "measured_unique_once_efficiency": eff_unique,
            "reuse_roofline_gbps": [reuse_lo, reuse_hi],
            "reuse_roofline_efficiency": [eff_reuse_lo, eff_reuse_hi],
            "bpw_required_for_sub_100_fs": {
                "at_unique_once_ceiling": bpw_for(eff_unique),
                "at_reuse_low": bpw_for(eff_reuse_lo),
                "at_reuse_high": bpw_for(eff_reuse_hi),
                "at_unity_819": 100.0 / FS_PER_WEIGHT_AT_PEAK,
                "note": "These are ACTIVE decode BPW, not storage complete_bpw.",
            },
            "vehicles_at_unique_once_ceiling": {
                "mixed_1p5_v1": {
                    "active_bpw": m15.active_bpw(),
                    "storage_bpw": m15.storage_complete_bpw,
                    "fs_at_unique_ceiling": fs_at(m15.active_bpw(), eff_unique),
                    "fs_at_unity": fs_at(m15.active_bpw(), 1.0),
                    "reaches_sub_100": fs_at(m15.active_bpw(), eff_unique) < 100.0,
                },
                "mixed_sub655_v1": {
                    "active_bpw": s655.active_bpw(),
                    "storage_bpw": s655.storage_complete_bpw,
                    "fs_at_unique_ceiling": fs_at(s655.active_bpw(), eff_unique),
                    "fs_at_unity": fs_at(s655.active_bpw(), 1.0),
                    "reaches_sub_100": fs_at(s655.active_bpw(), eff_unique) < 100.0,
                },
                "q4_budget_only": {
                    "active_bpw": q4.active_bpw(),
                    "fs_at_unique_ceiling": fs_at(q4.active_bpw(), eff_unique),
                }
            },
            "g013_correction": {
                "g013_claimed_sub655_best_fs": 124.85,
                "g013_assumed": "storage 0.6462 BPW at 647 GB/s",
                "actual": "decode moves nonexpert at 4.25 BPW and experts at 0.53 BPW; active mix is much higher than 0.6462",
            }
        })
    }

    fn rank_caps(
        dispatch: &Value,
        geometry: &Value,
        gather: &Value,
        residency: &Value,
        intensity: &Value,
        token: &Value,
        v: &VehicleBudget,
        unique_ceiling: f64,
    ) -> Value {
        let nop = dispatch
            .pointer("/one_nop_cb/median_gpu_ns")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        let cbs_host = dispatch
            .pointer("/ninety_eight_cbs_x_12_nops/median_host_ns")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        let cbs_gpu = dispatch
            .pointer("/ninety_eight_cbs_x_12_nops/median_gpu_ns")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        let one_organ = geometry
            .pointer("/one_organ_512_threads/spread/median_ns")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        let thirty = geometry
            .pointer("/thirty_organs_15360_threads/spread/median_ns")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        let gath = gather
            .pointer("/scattered_10_of_512/median_ns")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        let seq10 = gather
            .pointer("/sequential_10/median_ns")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        let cold = residency
            .pointer("/cold_first_gpu_read/median_ns")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        let warm = residency
            .pointer("/warm_after_page_touch/median_ns")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        let memcpy = residency
            .pointer("/host_memcpy_median_ns")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        let fma_extra = intensity
            .pointer("/fma_minus_load_median_ns")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        let token_gpu = token
            .get("median_gpu_ns")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);
        let token_host = token
            .get("median_host_ns")
            .and_then(|x| x.as_u64())
            .unwrap_or(0);

        let organs_per_token = Q80_LAYERS * (Q80_TOP_K + 1);
        let geometry_tax = one_organ.saturating_mul(organs_per_token).saturating_sub(
            if thirty > 0 {
                // 30 organs per dispatch → how many dispatches for all expert organs
                let disp = organs_per_token.div_ceil(30);
                thirty.saturating_mul(disp)
            } else {
                0
            },
        );
        let gather_tax = gath.saturating_sub(seq10).saturating_mul(Q80_LAYERS);
        let residency_tax = cold.saturating_sub(warm);
        let cb_tax = cbs_host.saturating_sub(cbs_gpu);
        let dispatch_tax = nop.saturating_mul(DISPATCHES_PER_TOKEN);
        // Do not scale the starved 512-thread fma extra across every organ —
        // that would invent a 50 ms ALU tax the token-shape load-only run
        // already refutes. Report the single-organ extra and the 2.3x ratio.
        // unique_ceiling is GB/s == bytes/ns from gbps().
        let bandwidth_floor = (v.total_moved() as f64 / unique_ceiling.max(1e-12)) as u64;

        let mut rows = vec![
            json!({"facet": "host_cb_serialization_98", "ns": cb_tax, "how": "98 CBs of 12 nops: host wall minus GPU busy"}),
            json!({"facet": "dispatch_count_1155_nops", "ns": dispatch_tax, "how": "1155 single-dispatch CBs of a live nop, projected from one-CB median GPU ns"}),
            json!({"facet": "launch_geometry_unbatched_organs", "ns": geometry_tax, "how": "512-thread one-organ vs 15360-thread 30-organ batch, scaled to 48*(10+shared) organs"}),
            json!({"facet": "moe_gather_10_of_512", "ns": gather_tax, "how": "scattered vs sequential top-10, times 48 layers"}),
            json!({"facet": "residency_cold_minus_warm", "ns": residency_tax, "how": "first GPU read of an untouched buffer minus warm reread"}),
            json!({"facet": "host_memcpy_expert_slab_analog", "ns": memcpy, "how": "CPU memcpy of routed-expert-sized payload; analog of compact_expert_slab_pack"}),
            json!({"facet": "arithmetic_intensity_fma_extra_one_organ", "ns": fma_extra, "how": "serial f32 MAC minus load-only on one 512x2048 organ at the same 512-thread launch; 2.3x, not a 50 ms token tax"}),
            json!({"facet": "unique_once_bandwidth_floor", "ns": bandwidth_floor, "how": "token bytes / measured unique-once ceiling"}),
            json!({"facet": "token_shape_gpu_busy", "ns": token_gpu, "how": "measured token-shaped control GPU ns"}),
            json!({"facet": "token_shape_host_wall", "ns": token_host, "how": "measured token-shaped control host wall"}),
        ];
        rows.sort_by(|a, b| {
            b.get("ns")
                .and_then(|v| v.as_u64())
                .cmp(&a.get("ns").and_then(|v| v.as_u64()))
        });
        json!({
            "order": "descending measured-or-projected ns",
            "rows": rows,
        })
    }

    fn spread_from(ns: &[u64], bytes: u64) -> Spread {
        Spread {
            all_gbps: ns.iter().map(|&t| gbps(bytes, t)).collect(),
            all_ns: ns.to_vec(),
        }
    }

    fn bytemuck_u32(values: &[u32]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4)
        }
    }
}
