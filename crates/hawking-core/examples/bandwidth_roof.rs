//! N017 streaming-read bandwidth roof. Pure traffic, no model math.
//!
//! Build:
//!   CARGO_TARGET_DIR=workspace/ops/build/rust cargo build --profile release-fast \
//!     -p hawking-core --example bandwidth_roof
//! Measure (GPU mutex required):
//!   ./tools/gpu_lane_lock.sh n017-roof \
//!     workspace/ops/build/rust/release-fast/examples/bandwidth_roof \
//!     --mode sweep --out receipts/headless/BANDWIDTH_ROOF.raw.json
//!
//! GPU time is only MTLCommandBuffer GPUStartTime/GPUEndTime after wait.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("bandwidth_roof requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

#[cfg(target_os = "macos")]
mod macos {
    use metal::objc::{msg_send, sel, sel_impl};
    use metal::{
        CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus, MTLDispatchType,
        MTLPurgeableState, MTLResourceOptions, MTLResourceUsage, MTLSize,
    };
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    const SHADER: &str = include_str!("../shaders/bandwidth_roof.metal");
    const SCHEMA: &str = "hawking.headless.bandwidth_roof.raw.v1";
    const PEAK_GB_S: f64 = 819.0;
    const TARGET_775: f64 = 775.0;
    const WARMUP: usize = 2;
    const WARM_REPS: usize = 5;
    const MAX_THREADS: u32 = 1_048_576;
    const FILL_TG: u32 = 256;
    const FILL_GROUPS: u32 = 4096;
    const EVICT_BYTES: u64 = 4 * 1024 * 1024 * 1024;
    const INDEXED_NIDX: u32 = 16 * 1024 * 1024; // 16M gathers, 64 MiB of uint indices

    type BoxErr = Box<dyn Error>;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RoofParams {
        nbytes: u64,
        nthreads: u32,
        iters: u32,
        stride_bytes: u32,
        nbufs: u32,
    }

    #[derive(Clone, Copy)]
    struct GpuSpan {
        start_ns: u64,
        end_ns: u64,
        dur_ns: u64,
    }

    struct Args {
        out: PathBuf,
        mode: String,
        kernel: String,
        tg: u32,
        groups: u32,
        bytes: u64,
        storage: String,
        max_bytes: u64,
        warm_reps: usize,
        stride: u32,
        iters: u32,
    }

    struct Pipes {
        map: BTreeMap<String, ComputePipelineState>,
    }

    struct Slab {
        priv_a: metal::Buffer,
        priv_b: metal::Buffer,
        priv_c: metal::Buffer,
        priv_d: metal::Buffer,
        shared_a: metal::Buffer,
        shared_b: metal::Buffer,
        out: metal::Buffer,
        indices: metal::Buffer,
        cap: u64,
        untracked: bool,
    }

    fn fail(msg: impl Into<String>) -> BoxErr {
        std::io::Error::new(std::io::ErrorKind::InvalidData, msg.into()).into()
    }

    fn parse_args() -> Result<Args, BoxErr> {
        let mut out = PathBuf::from("receipts/headless/BANDWIDTH_ROOF.raw.json");
        let mut mode = "sweep".to_string();
        let mut kernel = "roof_seq_f4".to_string();
        let mut tg = 256u32;
        let mut groups = 4096u32;
        let mut bytes = 4u64 << 30;
        let mut storage = "private".to_string();
        let mut max_bytes = 4u64 << 30;
        let mut warm_reps = WARM_REPS;
        let mut stride = 256u32;
        let mut iters = 0u32;
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--out" => out = PathBuf::from(args.next().ok_or("missing --out")?),
                "--mode" => mode = args.next().ok_or("missing --mode")?,
                "--kernel" => kernel = args.next().ok_or("missing --kernel")?,
                "--tg" => tg = args.next().ok_or("missing --tg")?.parse()?,
                "--groups" => groups = args.next().ok_or("missing --groups")?.parse()?,
                "--bytes" => bytes = args.next().ok_or("missing --bytes")?.parse()?,
                "--storage" => storage = args.next().ok_or("missing --storage")?,
                "--max-bytes" => max_bytes = args.next().ok_or("missing --max-bytes")?.parse()?,
                "--warm-reps" => warm_reps = args.next().ok_or("missing --warm-reps")?.parse()?,
                "--stride" => stride = args.next().ok_or("missing --stride")?.parse()?,
                "--iters" => iters = args.next().ok_or("missing --iters")?.parse()?,
                other => return Err(fail(format!("unsupported option {other}"))),
            }
        }
        Ok(Args {
            out,
            mode,
            kernel,
            tg,
            groups,
            bytes,
            storage,
            max_bytes,
            warm_reps,
            stride,
            iters,
        })
    }

    fn shader_sha256() -> String {
        let mut h = Sha256::new();
        h.update(SHADER.as_bytes());
        format!("{:x}", h.finalize())
    }

    fn gb_s(bytes: u64, ns: u64) -> f64 {
        if ns == 0 {
            0.0
        } else {
            bytes as f64 / ns as f64
        }
    }

    fn median_u64(xs: &[u64]) -> u64 {
        let mut s = xs.to_vec();
        s.sort_unstable();
        s[s.len() / 2]
    }

    fn median_f64(xs: &[f64]) -> f64 {
        let mut s = xs.to_vec();
        s.sort_by(f64::total_cmp);
        s[s.len() / 2]
    }

    fn spread_pct(xs: &[f64]) -> Option<f64> {
        if xs.is_empty() {
            return None;
        }
        let mut s = xs.to_vec();
        s.sort_by(f64::total_cmp);
        let med = s[s.len() / 2];
        if med == 0.0 {
            None
        } else {
            Some(100.0 * (s[s.len() - 1] - s[0]) / med)
        }
    }

    fn working_set_class(nbytes: u64) -> &'static str {
        if nbytes <= 16 * 1024 * 1024 {
            "below_cache"
        } else if nbytes <= 64 * 1024 * 1024 {
            "around_slc"
        } else if nbytes < 1024 * 1024 * 1024 {
            "above_cache"
        } else {
            "dram_streaming"
        }
    }

    fn gpu_span(cmd: &metal::CommandBufferRef, label: &str) -> Result<GpuSpan, BoxErr> {
        if cmd.status() != MTLCommandBufferStatus::Completed {
            return Err(fail(format!(
                "{label}: command buffer status {:?}",
                cmd.status()
            )));
        }
        let (start, end): (f64, f64) =
            unsafe { (msg_send![cmd, GPUStartTime], msg_send![cmd, GPUEndTime]) };
        if !(start.is_finite() && end.is_finite() && start > 0.0 && end > start) {
            return Err(fail(format!(
                "{label}: GPUStartTime/GPUEndTime unavailable start={start:?} end={end:?}"
            )));
        }
        let dur = ((end - start) * 1e9).round() as u64;
        if dur == 0 {
            return Err(fail(format!("{label}: GPU duration rounded to zero")));
        }
        Ok(GpuSpan {
            start_ns: (start * 1e9).round() as u64,
            end_ns: (end * 1e9).round() as u64,
            dur_ns: dur,
        })
    }

    fn compile_pipes(device: &metal::Device) -> Result<Pipes, BoxErr> {
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = device
            .new_library_with_source(SHADER, &opts)
            .map_err(|e| fail(format!("shader compile: {e}")))?;
        let names = [
            "roof_fill_u32",
            "roof_seq_f1",
            "roof_seq_f2",
            "roof_seq_f4",
            "roof_seq_f4x4",
            "roof_seq_f4x8",
            "roof_seq_simd8x8",
            "roof_stride_f4",
            "roof_gather_f4",
            "roof_gather_indexed_f4",
            "roof_multi_f4",
            "roof_write_f4",
            "roof_readwrite_f4",
            "roof_bad_control",
        ];
        let mut map = BTreeMap::new();
        for name in names {
            let f = lib
                .get_function(name, None)
                .map_err(|e| fail(format!("kernel {name}: {e}")))?;
            let p = device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| fail(format!("pipeline {name}: {e}")))?;
            map.insert(name.to_string(), p);
        }
        Ok(Pipes { map })
    }

    fn alloc(
        device: &metal::Device,
        len: u64,
        opts: MTLResourceOptions,
        label: &str,
    ) -> Result<metal::Buffer, BoxErr> {
        if len == 0 {
            return Err(fail(format!("{label}: zero-length buffer")));
        }
        let ceiling = device.recommended_max_working_set_size();
        if ceiling > 0 && len > ceiling {
            return Err(fail(format!(
                "{label}: {len} B exceeds recommendedMaxWorkingSetSize {ceiling}"
            )));
        }
        let buf = device.new_buffer(len, opts);
        buf.set_label(label);
        buf.set_purgeable_state(MTLPurgeableState::NonVolatile);
        Ok(buf)
    }

    fn make_slab(device: &metal::Device, cap: u64) -> Result<Slab, BoxErr> {
        let priv_opts = MTLResourceOptions::StorageModePrivate
            | MTLResourceOptions::HazardTrackingModeUntracked;
        let shared_opts = MTLResourceOptions::StorageModeShared;
        let idx_bytes = INDEXED_NIDX as u64 * 4;
        let out_bytes = MAX_THREADS as u64 * 4;
        let priv_a = alloc(device, cap, priv_opts, "n017.priv_a")?;
        let priv_b = alloc(device, cap, priv_opts, "n017.priv_b")?;
        let chunk = (cap / 4).max(16 * 1024 * 1024);
        let priv_c = alloc(device, chunk, priv_opts, "n017.priv_c")?;
        let priv_d = alloc(device, chunk, priv_opts, "n017.priv_d")?;
        let shared_a = alloc(device, cap, shared_opts, "n017.shared_a")?;
        // Shared dest is only a placeholder: no shared read+write config is launched
        // at full cap (that would double the unified-memory footprint).
        let shared_b = alloc(
            device,
            (64 * 1024 * 1024).min(cap),
            shared_opts,
            "n017.shared_b",
        )?;
        let out = alloc(device, out_bytes, shared_opts, "n017.out")?;
        let indices = alloc(device, idx_bytes, shared_opts, "n017.indices")?;
        unsafe {
            let n = INDEXED_NIDX as usize;
            let p = indices.contents() as *mut u32;
            let nvec = (cap / 16) as u32;
            for i in 0..n {
                let x = (i as u32)
                    .wrapping_mul(0x9E37_79B9)
                    .wrapping_add(0x85EB_CA6B);
                *p.add(i) = x % nvec.max(1);
            }
        }
        Ok(Slab {
            priv_a,
            priv_b,
            priv_c,
            priv_d,
            shared_a,
            shared_b,
            out,
            indices,
            cap,
            untracked: true,
        })
    }

    fn set_params(enc: &metal::ComputeCommandEncoderRef, index: u64, p: &RoofParams) {
        enc.set_bytes(
            index,
            std::mem::size_of::<RoofParams>() as u64,
            p as *const RoofParams as *const _,
        );
    }

    fn set_u64(enc: &metal::ComputeCommandEncoderRef, index: u64, v: u64) {
        enc.set_bytes(index, 8, &v as *const u64 as *const _);
    }

    fn maybe_use(
        enc: &metal::ComputeCommandEncoderRef,
        buf: &metal::Buffer,
        usage: MTLResourceUsage,
        untracked: bool,
    ) {
        if untracked {
            enc.use_resource(buf, usage);
        }
    }

    fn fill_buffer(
        queue: &metal::CommandQueue,
        pipes: &Pipes,
        buf: &metal::Buffer,
        nbytes: u64,
        untracked: bool,
        label: &str,
    ) -> Result<(), BoxErr> {
        let pipe = pipes.map.get("roof_fill_u32").ok_or("missing fill")?;
        let cmd = queue.new_command_buffer();
        cmd.set_label(label);
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(buf), 0);
        let nwords = nbytes / 4;
        set_u64(&enc, 1, nwords);
        maybe_use(&enc, buf, MTLResourceUsage::Write, untracked);
        enc.dispatch_thread_groups(
            MTLSize::new(FILL_GROUPS as u64, 1, 1),
            MTLSize::new(FILL_TG as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        if cmd.status() != MTLCommandBufferStatus::Completed {
            return Err(fail(format!("{label} fill failed: {:?}", cmd.status())));
        }
        Ok(())
    }

    fn evict(
        queue: &metal::CommandQueue,
        pipes: &Pipes,
        slab: &Slab,
        nthreads: u32,
        tg: u32,
    ) -> Result<(), BoxErr> {
        let nbytes = EVICT_BYTES.min(slab.cap);
        let groups = (nthreads / tg).max(1);
        let params = RoofParams {
            nbytes,
            nthreads: groups * tg,
            iters: 1,
            stride_bytes: 16,
            nbufs: 1,
        };
        let pipe = pipes.map.get("roof_seq_f4").ok_or("missing seq_f4")?;
        let cmd = queue.new_command_buffer();
        cmd.set_label("n017.evict");
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&slab.priv_a), 0);
        enc.set_buffer(1, Some(&slab.out), 0);
        set_params(&enc, 2, &params);
        maybe_use(&enc, &slab.priv_a, MTLResourceUsage::Read, slab.untracked);
        maybe_use(&enc, &slab.out, MTLResourceUsage::Write, false);
        enc.dispatch_thread_groups(
            MTLSize::new(groups as u64, 1, 1),
            MTLSize::new(tg as u64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    struct Launch {
        kernel: String,
        pattern: String,
        vec: String,
        rw: String,
        storage: String,
        tg: u32,
        groups: u32,
        nbytes: u64,
        iters: u32,
        stride: u32,
        nbufs: u32,
        n_queues: u32,
        concurrent: bool,
        blit: bool,
        bad: bool,
        evict_before_cold: bool,
    }

    impl Launch {
        fn nthreads(&self) -> u32 {
            self.groups.saturating_mul(self.tg).min(MAX_THREADS)
        }

        fn id(&self) -> String {
            format!(
                "{}|{}|tg{}|g{}|ws{}|it{}|st{}|{}|{}|nb{}|q{}{}{}{}",
                self.pattern,
                self.vec,
                self.tg,
                self.groups,
                self.nbytes,
                self.iters,
                self.stride,
                self.rw,
                self.storage,
                self.nbufs,
                self.n_queues,
                if self.concurrent { "|conc" } else { "" },
                if self.blit { "|blit" } else { "" },
                if self.bad { "|bad" } else { "" },
            )
        }

        fn bytes_moved_read(&self) -> u64 {
            if self.blit {
                return self.nbytes;
            }
            match self.rw.as_str() {
                "write" => 0,
                "readwrite" => self.nbytes.saturating_mul(self.iters as u64),
                _ if self.pattern == "multi" => self
                    .nbytes
                    .saturating_mul(self.nbufs as u64)
                    .saturating_mul(self.iters as u64),
                _ if self.pattern == "stride" || self.pattern == "gather" => {
                    self.nthreads() as u64 * self.iters as u64 * 16
                }
                _ if self.pattern == "gather_indexed" => {
                    INDEXED_NIDX as u64 * 16 * self.iters as u64
                }
                _ if self.bad => self.nthreads() as u64 * self.iters as u64 * 16,
                _ => self.nbytes.saturating_mul(self.iters as u64),
            }
        }

        fn bytes_moved_write(&self) -> u64 {
            if self.blit {
                return self.nbytes;
            }
            match self.rw.as_str() {
                "write" | "readwrite" => self.nbytes.saturating_mul(self.iters as u64),
                _ => 0,
            }
        }

        fn claimed_bytes(&self) -> u64 {
            self.nbytes.saturating_mul(self.iters.max(1) as u64)
        }
    }

    fn src_buf<'a>(launch: &Launch, slab: &'a Slab) -> &'a metal::Buffer {
        if launch.storage == "shared" {
            &slab.shared_a
        } else {
            &slab.priv_a
        }
    }

    fn dst_buf<'a>(launch: &Launch, slab: &'a Slab) -> &'a metal::Buffer {
        if launch.storage == "shared" {
            &slab.shared_b
        } else {
            &slab.priv_b
        }
    }

    fn encode_compute(
        enc: &metal::ComputeCommandEncoderRef,
        pipes: &Pipes,
        slab: &Slab,
        launch: &Launch,
        byte_offset: u64,
        nbytes: u64,
        nthreads: u32,
    ) -> Result<(), BoxErr> {
        let pipe = pipes
            .map
            .get(&launch.kernel)
            .ok_or_else(|| fail(format!("missing kernel {}", launch.kernel)))?;
        if (launch.tg as u64) > pipe.max_total_threads_per_threadgroup() {
            return Err(fail(format!(
                "{} tg {} exceeds pipeline max {}",
                launch.kernel,
                launch.tg,
                pipe.max_total_threads_per_threadgroup()
            )));
        }
        let params = RoofParams {
            nbytes,
            nthreads,
            iters: launch.iters.max(1),
            stride_bytes: launch.stride.max(16),
            nbufs: launch.nbufs.max(1),
        };
        enc.set_compute_pipeline_state(pipe);
        let untracked = launch.storage == "private";
        match launch.kernel.as_str() {
            "roof_multi_f4" => {
                enc.set_buffer(0, Some(&slab.priv_a), byte_offset);
                enc.set_buffer(1, Some(&slab.priv_b), byte_offset);
                enc.set_buffer(2, Some(&slab.priv_c), 0);
                enc.set_buffer(3, Some(&slab.priv_d), 0);
                enc.set_buffer(4, Some(&slab.out), 0);
                set_params(enc, 5, &params);
                maybe_use(enc, &slab.priv_a, MTLResourceUsage::Read, untracked);
                maybe_use(enc, &slab.priv_b, MTLResourceUsage::Read, untracked);
                maybe_use(enc, &slab.priv_c, MTLResourceUsage::Read, untracked);
                maybe_use(enc, &slab.priv_d, MTLResourceUsage::Read, untracked);
                maybe_use(enc, &slab.out, MTLResourceUsage::Write, false);
            }
            "roof_readwrite_f4" => {
                enc.set_buffer(0, Some(src_buf(launch, slab)), byte_offset);
                enc.set_buffer(1, Some(dst_buf(launch, slab)), byte_offset);
                enc.set_buffer(2, Some(&slab.out), 0);
                set_params(enc, 3, &params);
                maybe_use(
                    enc,
                    src_buf(launch, slab),
                    MTLResourceUsage::Read,
                    untracked,
                );
                maybe_use(
                    enc,
                    dst_buf(launch, slab),
                    MTLResourceUsage::Write,
                    untracked,
                );
                maybe_use(enc, &slab.out, MTLResourceUsage::Write, false);
            }
            "roof_gather_indexed_f4" => {
                enc.set_buffer(0, Some(src_buf(launch, slab)), 0);
                enc.set_buffer(1, Some(&slab.indices), 0);
                enc.set_buffer(2, Some(&slab.out), 0);
                let idx_params = RoofParams {
                    nbytes: INDEXED_NIDX as u64 * 4,
                    nthreads,
                    iters: launch.iters.max(1),
                    stride_bytes: 16,
                    nbufs: 1,
                };
                set_params(enc, 3, &idx_params);
                maybe_use(
                    enc,
                    src_buf(launch, slab),
                    MTLResourceUsage::Read,
                    untracked,
                );
                maybe_use(enc, &slab.out, MTLResourceUsage::Write, false);
            }
            "roof_write_f4" => {
                enc.set_buffer(0, Some(src_buf(launch, slab)), byte_offset);
                enc.set_buffer(1, Some(&slab.out), 0);
                set_params(enc, 2, &params);
                maybe_use(
                    enc,
                    src_buf(launch, slab),
                    MTLResourceUsage::Write,
                    untracked,
                );
                maybe_use(enc, &slab.out, MTLResourceUsage::Write, false);
            }
            _ => {
                enc.set_buffer(0, Some(src_buf(launch, slab)), byte_offset);
                enc.set_buffer(1, Some(&slab.out), 0);
                set_params(enc, 2, &params);
                maybe_use(
                    enc,
                    src_buf(launch, slab),
                    MTLResourceUsage::Read,
                    untracked,
                );
                maybe_use(enc, &slab.out, MTLResourceUsage::Write, false);
            }
        }
        let groups = (nthreads / launch.tg).max(1) as u64;
        enc.dispatch_thread_groups(
            MTLSize::new(groups, 1, 1),
            MTLSize::new(launch.tg as u64, 1, 1),
        );
        Ok(())
    }

    fn time_one(
        queues: &[metal::CommandQueue],
        pipes: &Pipes,
        slab: &Slab,
        launch: &Launch,
        label: &str,
    ) -> Result<(GpuSpan, u64), BoxErr> {
        if launch.blit {
            let q = &queues[0];
            let cmd = q.new_command_buffer();
            cmd.set_label(label);
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_buffer(
                src_buf(launch, slab),
                0,
                dst_buf(launch, slab),
                0,
                launch.nbytes,
            );
            blit.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            return Ok((gpu_span(&cmd, label)?, 1));
        }
        if launch.n_queues > 1 {
            let nq = launch.n_queues.min(queues.len() as u32) as usize;
            let chunk = launch.nbytes / nq as u64;
            let nthreads = launch.nthreads();
            let threads_each = (nthreads / nq as u32).max(launch.tg);
            let mut cmds: Vec<metal::CommandBuffer> = Vec::with_capacity(nq);
            for i in 0..nq {
                let cmd = queues[i].new_command_buffer().to_owned();
                cmd.set_label(&format!("{label}.q{i}"));
                let enc = cmd.new_compute_command_encoder();
                encode_compute(
                    &enc,
                    pipes,
                    slab,
                    launch,
                    i as u64 * chunk,
                    chunk,
                    threads_each,
                )?;
                enc.end_encoding();
                cmd.commit();
                cmds.push(cmd);
            }
            for cmd in &cmds {
                cmd.wait_until_completed();
            }
            let mut start = u64::MAX;
            let mut end = 0u64;
            for (i, cmd) in cmds.iter().enumerate() {
                let s = gpu_span(cmd, &format!("{label}.q{i}"))?;
                start = start.min(s.start_ns);
                end = end.max(s.end_ns);
            }
            if end <= start {
                return Err(fail(format!("{label}: concurrent GPU window empty")));
            }
            return Ok((
                GpuSpan {
                    start_ns: start,
                    end_ns: end,
                    dur_ns: end - start,
                },
                nq as u64,
            ));
        }
        let q = &queues[0];
        let cmd = q.new_command_buffer();
        cmd.set_label(label);
        if launch.concurrent {
            let half = launch.nbytes / 2;
            let nthreads = launch.nthreads();
            let half_threads = (nthreads / 2).max(launch.tg);
            let e1 = cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
            encode_compute(&e1, pipes, slab, launch, 0, half, half_threads)?;
            e1.end_encoding();
            let e2 = cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
            encode_compute(&e2, pipes, slab, launch, half, half, half_threads)?;
            e2.end_encoding();
        } else {
            let enc = cmd.new_compute_command_encoder();
            encode_compute(
                &enc,
                pipes,
                slab,
                launch,
                0,
                launch.nbytes,
                launch.nthreads(),
            )?;
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();
        Ok((gpu_span(&cmd, label)?, 1))
    }

    fn peek_out(slab: &Slab, n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        unsafe {
            let p = slab.out.contents() as *const f32;
            for i in 0..n {
                v[i] = *p.add(i);
            }
        }
        v
    }

    fn sample_json(span: &GpuSpan, bytes_read: u64, bytes_write: u64) -> Value {
        let read = gb_s(bytes_read, span.dur_ns);
        let write = gb_s(bytes_write, span.dur_ns);
        let traffic = gb_s(bytes_read + bytes_write, span.dur_ns);
        json!({
            "gpu_ns": span.dur_ns,
            "gpu_start_ns": span.start_ns,
            "gpu_end_ns": span.end_ns,
            "read_gb_s": read,
            "write_gb_s": write,
            "traffic_gb_s": traffic,
        })
    }

    fn run_launch(
        queues: &[metal::CommandQueue],
        pipes: &Pipes,
        slab: &Slab,
        launch: &Launch,
        warm_reps: usize,
    ) -> Result<Value, BoxErr> {
        if launch.nbytes > slab.cap && launch.pattern != "gather_indexed" {
            return Err(fail(format!(
                "{} nbytes {} exceeds slab {}",
                launch.id(),
                launch.nbytes,
                slab.cap
            )));
        }
        if launch.nthreads() == 0 || launch.nthreads() % launch.tg != 0 {
            return Err(fail(format!("{} bad geometry", launch.id())));
        }
        if launch.evict_before_cold {
            evict(&queues[0], pipes, slab, 4096 * 256, 256)?;
        }
        let bytes_read = launch.bytes_moved_read();
        let bytes_write = launch.bytes_moved_write();
        let cold = time_one(
            queues,
            pipes,
            slab,
            launch,
            &format!("cold.{}", launch.id()),
        )?;
        for i in 0..WARMUP {
            let _ = time_one(
                queues,
                pipes,
                slab,
                launch,
                &format!("warmup{i}.{}", launch.id()),
            )?;
        }
        let mut warm = Vec::with_capacity(warm_reps);
        for i in 0..warm_reps {
            warm.push(time_one(
                queues,
                pipes,
                slab,
                launch,
                &format!("warm{i}.{}", launch.id()),
            )?);
        }
        let warm_ns: Vec<u64> = warm.iter().map(|(s, _)| s.dur_ns).collect();
        let warm_read: Vec<f64> = warm_ns.iter().map(|&ns| gb_s(bytes_read, ns)).collect();
        let warm_traffic: Vec<f64> = warm_ns
            .iter()
            .map(|&ns| gb_s(bytes_read + bytes_write, ns))
            .collect();
        let med_ns = median_u64(&warm_ns);
        let med_read = median_f64(&warm_read);
        let sample = peek_out(slab, 8);
        let finite = sample.iter().all(|x| x.is_finite());
        let nonzero = sample.iter().any(|x| *x != 0.0);
        eprintln!(
            "  {:<72} cold={:7.1} warm_med={:7.1} GB/s  spread={:?}%  ns={}",
            launch.id(),
            gb_s(bytes_read, cold.0.dur_ns),
            med_read,
            spread_pct(&warm_read)
                .map(|p| format!("{p:.2}"))
                .unwrap_or_else(|| "n/a".into()),
            med_ns
        );
        Ok(json!({
            "id": launch.id(),
            "kernel": launch.kernel,
            "pattern": launch.pattern,
            "vec": launch.vec,
            "rw": launch.rw,
            "storage": launch.storage,
            "tg": launch.tg,
            "groups": launch.groups,
            "nthreads": launch.nthreads(),
            "nbytes": launch.nbytes,
            "iters": launch.iters,
            "stride_bytes": launch.stride,
            "nbufs": launch.nbufs,
            "n_queues": launch.n_queues,
            "concurrent_encoders": launch.concurrent,
            "blit": launch.blit,
            "bad_control": launch.bad,
            "working_set_class": working_set_class(launch.nbytes),
            "bytes_moved_read": bytes_read,
            "bytes_moved_write": bytes_write,
            "claimed_bytes": launch.claimed_bytes(),
            "actual_bytes_if_bad": if launch.bad { json!(bytes_read) } else { Value::Null },
            "command_buffers": launch.n_queues.max(1),
            "gpu_timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
            "cold": sample_json(&cold.0, bytes_read, bytes_write),
            "warm": {
                "n": warm_ns.len(),
                "gpu_ns": warm_ns,
                "read_gb_s": warm_read,
                "traffic_gb_s": warm_traffic,
                "median_gpu_ns": med_ns,
                "median_read_gb_s": med_read,
                "median_traffic_gb_s": median_f64(&warm_traffic),
                "min_read_gb_s": warm_read.iter().copied().fold(f64::INFINITY, f64::min),
                "max_read_gb_s": warm_read.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                "spread_pct": spread_pct(&warm_read),
            },
            "out_sample": sample,
            "out_finite": finite,
            "out_nonzero": nonzero,
            "q4_tile_shape_reused": false,
        }))
    }

    fn seq(
        kernel: &str,
        vec: &str,
        tg: u32,
        groups: u32,
        nbytes: u64,
        iters: u32,
        storage: &str,
    ) -> Launch {
        Launch {
            kernel: kernel.into(),
            pattern: "sequential".into(),
            vec: vec.into(),
            rw: "read".into(),
            storage: storage.into(),
            tg,
            groups,
            nbytes,
            iters,
            stride: 16,
            nbufs: 1,
            n_queues: 1,
            concurrent: false,
            blit: false,
            bad: false,
            evict_before_cold: working_set_class(nbytes) == "below_cache"
                || working_set_class(nbytes) == "around_slc",
        }
    }

    fn iters_for(nbytes: u64, target: u64) -> u32 {
        if nbytes >= 1024 * 1024 * 1024 {
            2
        } else {
            ((target / nbytes.max(1)).max(1)).min(1_000_000) as u32
        }
    }

    fn stride_iters(nthreads: u32, target: u64) -> u32 {
        let per = nthreads as u64 * 16;
        ((target / per.max(1)).max(64)).min(4_000_000) as u32
    }

    fn build_sweep(cap: u64, default_bytes: u64) -> Vec<Launch> {
        let mut v = Vec::new();
        let bytes4 = default_bytes.min(cap);
        let tg_list = [32u32, 64, 256, 512, 1024];
        let group_list = [240u32, 960, 3840, 4096, 8192];
        for tg in tg_list {
            for groups in group_list {
                let nthreads = groups.saturating_mul(tg);
                if nthreads == 0 || nthreads > MAX_THREADS {
                    continue;
                }
                if nthreads as u64 > bytes4 / 16 {
                    continue;
                }
                v.push(seq("roof_seq_f4", "f4", tg, groups, bytes4, 2, "private"));
            }
        }
        let g0 = 4096u32;
        let t0 = 256u32;
        for (k, vec) in [
            ("roof_seq_f1", "f1"),
            ("roof_seq_f2", "f2"),
            ("roof_seq_f4", "f4"),
            ("roof_seq_f4x4", "f4x4"),
            ("roof_seq_f4x8", "f4x8"),
            ("roof_seq_simd8x8", "simd8x8"),
        ] {
            v.push(seq(k, vec, t0, g0, bytes4, 2, "private"));
        }
        for stride in [16u32, 64, 256, 1024, 4096, 8192, 8256] {
            let nthreads = g0 * t0;
            v.push(Launch {
                kernel: "roof_stride_f4".into(),
                pattern: "strided".into(),
                vec: "f4".into(),
                rw: "read".into(),
                storage: "private".into(),
                tg: t0,
                groups: g0,
                nbytes: bytes4,
                iters: stride_iters(nthreads, bytes4),
                stride,
                nbufs: 1,
                n_queues: 1,
                concurrent: false,
                blit: false,
                bad: false,
                evict_before_cold: false,
            });
        }
        v.push(Launch {
            kernel: "roof_gather_f4".into(),
            pattern: "gather".into(),
            vec: "f4".into(),
            rw: "read".into(),
            storage: "private".into(),
            tg: t0,
            groups: g0,
            nbytes: bytes4,
            iters: stride_iters(g0 * t0, bytes4),
            stride: 16,
            nbufs: 1,
            n_queues: 1,
            concurrent: false,
            blit: false,
            bad: false,
            evict_before_cold: false,
        });
        v.push(Launch {
            kernel: "roof_gather_indexed_f4".into(),
            pattern: "gather_indexed".into(),
            vec: "f4".into(),
            rw: "read".into(),
            storage: "private".into(),
            tg: t0,
            groups: g0,
            nbytes: bytes4,
            iters: 1,
            stride: 16,
            nbufs: 1,
            n_queues: 1,
            concurrent: false,
            blit: false,
            bad: false,
            evict_before_cold: false,
        });
        let chunk = (bytes4 / 4).min(cap / 4);
        v.push(Launch {
            kernel: "roof_multi_f4".into(),
            pattern: "multi".into(),
            vec: "f4".into(),
            rw: "read".into(),
            storage: "private".into(),
            tg: t0,
            groups: g0,
            nbytes: chunk,
            iters: 2,
            stride: 16,
            nbufs: 4,
            n_queues: 1,
            concurrent: false,
            blit: false,
            bad: false,
            evict_before_cold: false,
        });
        for nbytes in [
            256 * 1024u64,
            1024 * 1024,
            16 * 1024 * 1024,
            64 * 1024 * 1024,
            256 * 1024 * 1024,
            1024 * 1024 * 1024,
            bytes4,
            cap,
        ] {
            if nbytes > cap {
                continue;
            }
            if nbytes < 16 * 1024 {
                continue;
            }
            let it = iters_for(nbytes, bytes4);
            v.push(seq("roof_seq_f4", "f4", t0, g0, nbytes, it, "private"));
        }
        for (rw, k) in [
            ("write", "roof_write_f4"),
            ("readwrite", "roof_readwrite_f4"),
        ] {
            v.push(Launch {
                kernel: k.into(),
                pattern: "sequential".into(),
                vec: "f4".into(),
                rw: rw.into(),
                storage: "private".into(),
                tg: t0,
                groups: g0,
                nbytes: bytes4,
                iters: 2,
                stride: 16,
                nbufs: 1,
                n_queues: 1,
                concurrent: false,
                blit: false,
                bad: false,
                evict_before_cold: false,
            });
        }
        v.push(seq("roof_seq_f4", "f4", t0, g0, bytes4, 2, "shared"));
        v.push(seq("roof_seq_f4x8", "f4x8", t0, g0, bytes4, 2, "shared"));
        for nq in [1u32, 2, 4] {
            v.push(Launch {
                kernel: "roof_seq_f4".into(),
                pattern: "sequential".into(),
                vec: "f4".into(),
                rw: "read".into(),
                storage: "private".into(),
                tg: t0,
                groups: g0,
                nbytes: bytes4,
                iters: 2,
                stride: 16,
                nbufs: 1,
                n_queues: nq,
                concurrent: false,
                blit: false,
                bad: false,
                evict_before_cold: false,
            });
        }
        v.push(Launch {
            kernel: "roof_seq_f4".into(),
            pattern: "sequential".into(),
            vec: "f4".into(),
            rw: "read".into(),
            storage: "private".into(),
            tg: t0,
            groups: g0,
            nbytes: bytes4,
            iters: 2,
            stride: 16,
            nbufs: 1,
            n_queues: 1,
            concurrent: true,
            blit: false,
            bad: false,
            evict_before_cold: false,
        });
        for b in [256 * 1024 * 1024u64, 1024 * 1024 * 1024, bytes4] {
            v.push(Launch {
                kernel: "MTLBlitCommandEncoder.copy".into(),
                pattern: "blit_copy".into(),
                vec: "blit".into(),
                rw: "readwrite".into(),
                storage: "private".into(),
                tg: 1,
                groups: 1,
                nbytes: b.min(cap),
                iters: 1,
                stride: 0,
                nbufs: 2,
                n_queues: 1,
                concurrent: false,
                blit: true,
                bad: false,
                evict_before_cold: false,
            });
        }
        v.push(Launch {
            kernel: "roof_bad_control".into(),
            pattern: "bad_control".into(),
            vec: "f4".into(),
            rw: "read".into(),
            storage: "private".into(),
            tg: t0,
            groups: g0,
            nbytes: bytes4,
            iters: 4096,
            stride: 16,
            nbufs: 1,
            n_queues: 1,
            concurrent: false,
            blit: false,
            bad: true,
            evict_before_cold: false,
        });
        v
    }

    fn build_headline(cap: u64, default_bytes: u64) -> Vec<Launch> {
        let b = default_bytes.min(cap);
        let t0 = 256u32;
        let g0 = 4096u32;
        vec![
            seq("roof_seq_f4", "f4", t0, g0, b, 2, "private"),
            seq("roof_seq_f4x8", "f4x8", t0, g0, b, 2, "private"),
            seq("roof_seq_f4", "f4", t0, g0, b, 2, "shared"),
            Launch {
                kernel: "roof_seq_f4".into(),
                pattern: "sequential".into(),
                vec: "f4".into(),
                rw: "read".into(),
                storage: "private".into(),
                tg: t0,
                groups: g0,
                nbytes: b,
                iters: 2,
                stride: 16,
                nbufs: 1,
                n_queues: 4,
                concurrent: false,
                blit: false,
                bad: false,
                evict_before_cold: false,
            },
            Launch {
                kernel: "roof_bad_control".into(),
                pattern: "bad_control".into(),
                vec: "f4".into(),
                rw: "read".into(),
                storage: "private".into(),
                tg: t0,
                groups: g0,
                nbytes: b,
                iters: 4096,
                stride: 16,
                nbufs: 1,
                n_queues: 1,
                concurrent: false,
                blit: false,
                bad: true,
                evict_before_cold: false,
            },
        ]
    }

    fn pick_best<'a>(rows: &'a [Value], dram_only: bool) -> Option<&'a Value> {
        let mut best: Option<(&Value, f64)> = None;
        for r in rows {
            if r.get("bad_control").and_then(|v| v.as_bool()) == Some(true) {
                continue;
            }
            if r.get("rw").and_then(|v| v.as_str()) != Some("read") {
                continue;
            }
            if dram_only
                && r.get("working_set_class").and_then(|v| v.as_str()) != Some("dram_streaming")
            {
                continue;
            }
            let gb = r
                .pointer("/warm/median_read_gb_s")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            if !gb.is_finite() || gb <= 0.0 {
                continue;
            }
            if best.map(|(_, b)| gb > b).unwrap_or(true) {
                best = Some((r, gb));
            }
        }
        best.map(|(r, _)| r)
    }

    pub fn run() -> Result<(), BoxErr> {
        let args = parse_args()?;
        let t0 = Instant::now();
        let device = Device::system_default().ok_or_else(|| fail("no Metal-capable GPU"))?;
        let device_name = device.name().to_string();
        if !device_name.contains("M3") {
            return Err(fail(format!(
                "restricted to Apple M3; found {device_name:?}"
            )));
        }
        let queues: Vec<metal::CommandQueue> = (0..4).map(|_| device.new_command_queue()).collect();
        eprintln!("n017 compile bandwidth_roof.metal on {device_name}");
        let pipes = compile_pipes(&device)?;
        let mut cap = args.max_bytes;
        let ceiling = device.recommended_max_working_set_size();
        // src+dst private + src+dst shared + 2 chunks ≈ 2.5 * cap. Keep a margin.
        let want = cap.saturating_mul(3);
        if ceiling > 0 && want > ceiling {
            cap = (ceiling / 4).min(args.max_bytes);
            eprintln!("n017 shrinking slab to {cap} B (working-set ceiling {ceiling})");
        }
        cap = cap.max(1024 * 1024 * 1024);
        eprintln!("n017 allocate slab cap={cap}");
        let slab = make_slab(&device, cap)?;
        eprintln!("n017 fill private+shared slabs");
        fill_buffer(
            &queues[0],
            &pipes,
            &slab.priv_a,
            cap,
            true,
            "n017.fill.priv_a",
        )?;
        fill_buffer(
            &queues[0],
            &pipes,
            &slab.priv_b,
            cap,
            true,
            "n017.fill.priv_b",
        )?;
        fill_buffer(
            &queues[0],
            &pipes,
            &slab.priv_c,
            slab.priv_c.length() as u64,
            true,
            "n017.fill.priv_c",
        )?;
        fill_buffer(
            &queues[0],
            &pipes,
            &slab.priv_d,
            slab.priv_d.length() as u64,
            true,
            "n017.fill.priv_d",
        )?;
        fill_buffer(
            &queues[0],
            &pipes,
            &slab.shared_a,
            cap,
            false,
            "n017.fill.shared_a",
        )?;
        fill_buffer(
            &queues[0],
            &pipes,
            &slab.shared_b,
            slab.shared_b.length() as u64,
            false,
            "n017.fill.shared_b",
        )?;

        let default_bytes = (4u64 << 30).min(cap);
        let launches = match args.mode.as_str() {
            "headline" => build_headline(cap, default_bytes),
            "confirm" => {
                let mut l = seq(
                    &args.kernel,
                    "f4",
                    args.tg,
                    args.groups,
                    args.bytes.min(cap),
                    if args.iters == 0 { 2 } else { args.iters },
                    &args.storage,
                );
                if args.kernel == "roof_bad_control" {
                    l.bad = true;
                    l.pattern = "bad_control".into();
                    l.iters = if args.iters == 0 { 4096 } else { args.iters };
                }
                if args.stride != 0 && args.kernel.contains("stride") {
                    l.stride = args.stride;
                    l.pattern = "strided".into();
                }
                vec![l]
            }
            _ => build_sweep(cap, default_bytes),
        };

        let mut rows = Vec::new();
        let mut errors = Vec::new();
        for launch in &launches {
            match run_launch(&queues, &pipes, &slab, launch, args.warm_reps) {
                Ok(row) => rows.push(row),
                Err(e) => {
                    eprintln!("  SKIP {}: {e}", launch.id());
                    errors.push(json!({"id": launch.id(), "error": e.to_string()}));
                }
            }
        }

        let best_dram = pick_best(&rows, true).cloned();
        let best_any = pick_best(&rows, false).cloned();
        let seq_dram = rows.iter().filter(|r| {
            r.get("pattern").and_then(|v| v.as_str()) == Some("sequential")
                && r.get("rw").and_then(|v| v.as_str()) == Some("read")
                && r.get("bad_control").and_then(|v| v.as_bool()) != Some(true)
                && r.get("working_set_class").and_then(|v| v.as_str()) == Some("dram_streaming")
                && r.get("n_queues").and_then(|v| v.as_u64()).unwrap_or(1) == 1
                && r.get("concurrent_encoders").and_then(|v| v.as_bool()) != Some(true)
        });
        let mut seq_best: Option<Value> = None;
        let mut seq_best_gb = 0.0;
        for r in seq_dram {
            let gb = r
                .pointer("/warm/median_read_gb_s")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            if gb > seq_best_gb {
                seq_best_gb = gb;
                seq_best = Some(r.clone());
            }
        }

        let bad = rows
            .iter()
            .find(|r| r.get("bad_control") == Some(&json!(true)));
        let bad_json = if let Some(b) = bad {
            let actual = b
                .pointer("/warm/median_read_gb_s")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let claimed_bytes = b.get("claimed_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
            let med_ns = b
                .pointer("/warm/median_gpu_ns")
                .and_then(|v| v.as_u64())
                .unwrap_or(1);
            let claimed_gb = gb_s(claimed_bytes, med_ns);
            json!({
                "present": true,
                "id": b.get("id"),
                "actual_read_gb_s": actual,
                "claimed_gb_s_if_naive": claimed_gb,
                "claimed_bytes": claimed_bytes,
                "actual_bytes": b.get("bytes_moved_read"),
                "rejected": true,
                "reason": "kernel reloads one 16-byte slot per thread; claimed nbytes/time is a no-op-passing fantasy and is not the roof",
            })
        } else {
            json!({"present": false, "rejected": false, "reason": "bad_control row missing"})
        };

        let doc = json!({
            "schema": SCHEMA,
            "device": device_name,
            "registry_id": device.registry_id(),
            "unified_memory": device.has_unified_memory(),
            "recommended_max_working_set_size": ceiling,
            "gpu_cores_campaign": 60,
            "cpu_cores": 28,
            "published_peak_gb_s": PEAK_GB_S,
            "target_775_gb_s": TARGET_775,
            "kernel_sha256": shader_sha256(),
            "shader": "crates/hawking-core/shaders/bandwidth_roof.metal",
            "mode": args.mode,
            "slab_bytes": cap,
            "max_threads": MAX_THREADS,
            "warm_reps": args.warm_reps,
            "warmup": WARMUP,
            "gpu_timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
            "did_not_compile_production_shader_library": true,
            "q4_tile_shape": "64 threads/row, 128-thread TG — not used",
            "configs": rows,
            "skipped": errors,
            "best_dram_read": best_dram,
            "best_any_read": best_any,
            "best_sequential_dram_read": seq_best,
            "bad_control": bad_json,
            "wall_s": t0.elapsed().as_secs_f64(),
        });

        if let Some(parent) = args.out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&args.out, serde_json::to_string_pretty(&doc)? + "\n")?;
        eprintln!(
            "n017 wrote {} ({} configs)",
            args.out.display(),
            doc["configs"].as_array().map(|a| a.len()).unwrap_or(0)
        );
        Ok(())
    }
}
