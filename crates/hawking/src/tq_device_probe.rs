//! Artifact-bound raw TQ device probe for the Appendix.
//!
//! This binary emits measurement evidence, not a promotion receipt. The Python
//! Appendix runner owns the shared heavy lease, resource/counter capture, strict
//! receipt validation, and atomic publication.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("hawking-tq-device-probe requires macOS Metal");
    std::process::exit(69);
}

#[cfg(target_os = "macos")]
mod process_joule;

#[cfg(target_os = "macos")]
mod macos {
    use crate::process_joule::{ProcessJouleCollector, Snapshot};
    use anyhow::{bail, Context, Result};
    use clap::Parser;
    use hawking_core::metal::{PhysicalTraceGuard, PhysicalTraceIdentity};
    use hawking_core::tq::{read_strand, RhtMode, StrandTensor};
    use hawking_core::{TqDeviceHarness, TqRuntimePath};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::collections::HashSet;
    use std::fs;
    use std::io::Write;
    use std::os::unix::fs::MetadataExt;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    const SCHEMA: &str = "hawking.tq_runtime_device_raw.v2";
    const MATRIX_IDENTITY_SCHEMA: &str = "hawking.tq_device_matrix_identity.v2";
    const FEATURE_IDENTITY_SCHEMA: &str = "hawking.tq_device_feature_identity.v1";
    const RESIDUAL_PROBE_SCHEMA: &str = "hawking.tq_device_residual_probe.v1";
    const HEAVY_LOCK: &str = "reports/cron/studio_heavy.lock";
    const PHASE_MARKERS_SCHEMA: &str = "hawking.physical_phase_markers.v1";
    const PHASE_INTERVAL_SCHEMA: &str = "hawking.physical_phase_interval.v1";
    const PHASE_PAIR_SCHEMA: &str = "hawking.physical_phase_pair.v1";
    const PHASE_INTERVAL_IDENTITY_SCHEMA: &str = "hawking.physical_phase_interval_identity.v1";

    #[repr(C)]
    struct MachTimebaseInfo {
        numer: u32,
        denom: u32,
    }

    extern "C" {
        fn mach_absolute_time() -> u64;
        fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
    }

    #[derive(Parser, Debug)]
    #[command(name = "hawking-tq-device-probe")]
    struct Args {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        tensor: Option<String>,
        /// Optional second STRAND artifact for the production residual
        /// accumulate pass.  This is never inferred from the base artifact.
        #[arg(long, requires = "residual_tensor")]
        residual_artifact: Option<PathBuf>,
        #[arg(long, requires = "residual_artifact")]
        residual_tensor: Option<String>,
        #[arg(long, default_value = "stored")]
        runtime_path: String,
        #[arg(long, default_value_t = 3)]
        warmups: usize,
        #[arg(long, default_value_t = 10)]
        trials: usize,
        #[arg(long)]
        source_commit: String,
        #[arg(long)]
        matrix_cell_id: String,
        #[arg(long)]
        matrix_cell_sha256: String,
        #[arg(long)]
        matrix_model: String,
        #[arg(long)]
        matrix_tensor_family: String,
        /// Hash-bound libproc/SDK provenance produced by the release authority.
        #[arg(long)]
        process_joule_provenance: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    }

    struct IntervalEnergySample {
        phase: String,
        role: String,
        batch: Option<usize>,
        iteration: usize,
        interval_sha256: String,
        wall_started_unix_ns: u64,
        wall_ended_unix_ns: u64,
        continuous_started_ns: u64,
        continuous_ended_ns: u64,
        before: Snapshot,
        after: Snapshot,
    }

    struct PendingEnergyRecord {
        batch: Option<usize>,
        iteration: usize,
        value: Value,
    }

    struct PhaseRecorder {
        run_nonce: String,
        probe_started_wall_unix_ns: u64,
        probe_started_continuous_ns: u64,
        intervals: Vec<Value>,
        pairs: Vec<Value>,
        process_joule: ProcessJouleCollector,
        energy_intervals: Vec<IntervalEnergySample>,
        energy_records: Vec<PendingEnergyRecord>,
        signpost_ids: HashSet<u64>,
    }

    fn wall_unix_ns() -> Result<u64> {
        Ok(SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("wall clock precedes UNIX epoch")?
            .as_nanos()
            .try_into()
            .context("wall clock nanoseconds overflow u64")?)
    }

    fn continuous_ns() -> Result<u64> {
        let mut info = MachTimebaseInfo { numer: 0, denom: 0 };
        if unsafe { mach_timebase_info(&mut info) } != 0 || info.denom == 0 {
            bail!("mach_timebase_info failed");
        }
        let ticks = unsafe { mach_absolute_time() } as u128;
        (ticks * info.numer as u128 / info.denom as u128)
            .try_into()
            .context("mach continuous nanoseconds overflow u64")
    }

    fn stamp_value(mut value: Value, field: &str) -> Result<Value> {
        let digest = canonical_sha256(&value)?;
        value
            .as_object_mut()
            .context("only JSON objects can be stamped")?
            .insert(field.to_owned(), Value::String(digest));
        Ok(value)
    }

    fn canonical_sha256(value: &Value) -> Result<String> {
        fn ordered(value: &Value) -> Value {
            match value {
                Value::Object(map) => {
                    let mut keys: Vec<&String> = map.keys().collect();
                    keys.sort();
                    let mut output = serde_json::Map::new();
                    for key in keys {
                        output.insert(key.clone(), ordered(&map[key]));
                    }
                    Value::Object(output)
                }
                Value::Array(rows) => Value::Array(rows.iter().map(ordered).collect()),
                scalar => scalar.clone(),
            }
        }
        Ok(sha256(&serde_json::to_vec(&ordered(value))?))
    }

    fn digest_field(value: &Value, field: &str) -> Result<String> {
        value
            .get(field)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .with_context(|| format!("stamped marker lacks {field}"))
    }

    impl PhaseRecorder {
        fn new(run_nonce: String, process_joule_provenance: &Path) -> Result<Self> {
            Ok(Self {
                run_nonce,
                probe_started_wall_unix_ns: wall_unix_ns()?,
                probe_started_continuous_ns: continuous_ns()?,
                intervals: Vec::new(),
                pairs: Vec::new(),
                process_joule: ProcessJouleCollector::from_provenance_file(
                    process_joule_provenance,
                )?,
                energy_intervals: Vec::new(),
                energy_records: Vec::new(),
                signpost_ids: HashSet::new(),
            })
        }

        fn measure<T, F>(
            &mut self,
            phase: &str,
            role: &str,
            batch: Option<usize>,
            iteration: usize,
            operation: F,
        ) -> Result<(T, u64, String)>
        where
            F: FnOnce() -> Result<T>,
        {
            let sequence = self.intervals.len();
            let interval_id = canonical_sha256(&json!({
                "schema": PHASE_INTERVAL_IDENTITY_SCHEMA,
                "run_nonce": self.run_nonce,
                "sequence": sequence,
                "phase": phase,
                "role": role,
                "batch": batch,
                "iteration": iteration,
            }))?;
            let mut signpost_id = u64::from_str_radix(&interval_id[..16], 16)
                .context("stable interval id cannot map to os_signpost_id_t")?;
            if signpost_id == 0 || signpost_id == u64::MAX {
                signpost_id = 1;
            }
            if !self.signpost_ids.insert(signpost_id) {
                bail!("stable phase intervals collide in os_signpost_id_t space");
            }
            let before = self.process_joule.snapshot()?;
            let wall_started_unix_ns = wall_unix_ns()?;
            let continuous_started_ns = continuous_ns()?;
            let trace_identity = PhysicalTraceIdentity::new(
                interval_id.clone(),
                self.run_nonce.clone(),
                phase.to_owned(),
                role.to_owned(),
                batch,
                iteration,
            )?;
            let trace_guard = PhysicalTraceGuard::begin(trace_identity)?;
            let output = operation();
            drop(trace_guard);
            let continuous_ended_ns = continuous_ns()?;
            let wall_ended_unix_ns = wall_unix_ns()?;
            let after = self.process_joule.snapshot()?;
            let output = output?;
            let elapsed_ns = continuous_ended_ns
                .checked_sub(continuous_started_ns)
                .context("continuous clock moved backwards")?;
            if elapsed_ns == 0 {
                bail!("phase interval has zero duration");
            }
            let interval = stamp_value(
                json!({
                    "schema": PHASE_INTERVAL_SCHEMA,
                    "run_nonce": self.run_nonce,
                    "sequence": sequence,
                    "interval_id": interval_id,
                    "signpost_id": signpost_id,
                    "phase": phase,
                    "role": role,
                    "batch": batch,
                    "iteration": iteration,
                    "wall_started_unix_ns": wall_started_unix_ns,
                    "wall_ended_unix_ns": wall_ended_unix_ns,
                    "continuous_started_ns": continuous_started_ns,
                    "continuous_ended_ns": continuous_ended_ns,
                    "elapsed_ns": elapsed_ns,
                }),
                "interval_sha256",
            )?;
            let digest = digest_field(&interval, "interval_sha256")?;
            self.intervals.push(interval);
            self.energy_intervals.push(IntervalEnergySample {
                phase: phase.to_owned(),
                role: role.to_owned(),
                batch,
                iteration,
                interval_sha256: digest.clone(),
                wall_started_unix_ns,
                wall_ended_unix_ns,
                continuous_started_ns,
                continuous_ended_ns,
                before,
                after,
            });
            Ok((output, elapsed_ns, digest))
        }

        fn pair(
            &mut self,
            phase: &str,
            batch: Option<usize>,
            iteration: usize,
            first_role: &str,
            baseline_interval_sha256: String,
            comparison_interval_sha256: String,
        ) -> Result<String> {
            let comparison_digest = comparison_interval_sha256.clone();
            let baseline_interval_id = self
                .intervals
                .iter()
                .find(|interval| {
                    interval.get("interval_sha256").and_then(Value::as_str)
                        == Some(baseline_interval_sha256.as_str())
                })
                .and_then(|interval| interval.get("interval_id"))
                .and_then(Value::as_str)
                .context("baseline phase interval lacks its stable interval id")?
                .to_owned();
            let candidate_interval_id = self
                .intervals
                .iter()
                .find(|interval| {
                    interval.get("interval_sha256").and_then(Value::as_str)
                        == Some(comparison_interval_sha256.as_str())
                })
                .and_then(|interval| interval.get("interval_id"))
                .and_then(Value::as_str)
                .context("candidate phase interval lacks its stable interval id")?
                .to_owned();
            let pair = stamp_value(
                json!({
                    "schema": PHASE_PAIR_SCHEMA,
                    "run_nonce": self.run_nonce,
                    "phase": phase,
                    "batch": batch,
                    "iteration": iteration,
                    "first_role": first_role,
                    "baseline_interval_sha256": baseline_interval_sha256,
                    "candidate_interval_sha256": comparison_interval_sha256,
                    "baseline_interval_id": baseline_interval_id,
                    "candidate_interval_id": candidate_interval_id,
                }),
                "phase_marker_sha256",
            )?;
            let digest = digest_field(&pair, "phase_marker_sha256")?;
            self.pairs.push(pair);
            if phase == "trial" {
                let sample = self
                    .energy_intervals
                    .iter()
                    .find(|sample| sample.interval_sha256 == comparison_digest)
                    .context("trial candidate interval lacks local process-energy samples")?;
                if sample.phase != phase
                    || sample.role != "candidate"
                    || sample.batch != batch
                    || sample.iteration != iteration
                {
                    bail!("trial process-energy sample is not the exact candidate interval");
                }
                let value = self.process_joule.phase_record(
                    &sample.before,
                    &sample.after,
                    &digest,
                    &sample.interval_sha256,
                    sample.wall_started_unix_ns,
                    sample.wall_ended_unix_ns,
                    sample.continuous_started_ns,
                    sample.continuous_ended_ns,
                )?;
                self.energy_records.push(PendingEnergyRecord {
                    batch,
                    iteration,
                    value,
                });
            }
            Ok(digest)
        }

        fn finish(mut self) -> Result<(Value, Value)> {
            let probe_ended_continuous_ns = continuous_ns()?;
            let probe_ended_wall_unix_ns = wall_unix_ns()?;
            self.energy_records
                .sort_by_key(|record| (record.batch.unwrap_or(0), record.iteration));
            let records = self
                .energy_records
                .into_iter()
                .map(|record| record.value)
                .collect();
            let process_energy_counters = self.process_joule.counter_block(records)?;
            let phase_markers = stamp_value(
                json!({
                    "schema": PHASE_MARKERS_SCHEMA,
                    "run_nonce": self.run_nonce,
                    "clock_source": "mach_absolute_time_plus_system_time_unix_epoch",
                    "probe_started_wall_unix_ns": self.probe_started_wall_unix_ns,
                    "probe_ended_wall_unix_ns": probe_ended_wall_unix_ns,
                    "probe_started_continuous_ns": self.probe_started_continuous_ns,
                    "probe_ended_continuous_ns": probe_ended_continuous_ns,
                    "intervals": self.intervals,
                    "pairs": self.pairs,
                }),
                "phase_markers_sha256",
            )?;
            Ok((phase_markers, process_energy_counters))
        }
    }

    fn validate_admission() -> Result<()> {
        if std::env::var("HAWKING_APPENDIX_DEVICE_ADMITTED").as_deref() != Ok("1") {
            bail!("device probe must be launched by the Appendix heavy-lease runner");
        }
        let raw = std::env::var("HAWKING_HEAVY_LEASE_FD")
            .context("HAWKING_HEAVY_LEASE_FD is required")?;
        let fd: i32 = raw.parse().context("invalid HAWKING_HEAVY_LEASE_FD")?;
        let fd_meta = fs::metadata(format!("/dev/fd/{fd}"))
            .context("inherited heavy-lease fd is not open")?;
        let lock_meta = fs::metadata(HEAVY_LOCK).context("canonical heavy lock is missing")?;
        if fd_meta.dev() != lock_meta.dev() || fd_meta.ino() != lock_meta.ino() {
            bail!("inherited fd does not identify the canonical Hawking heavy lock");
        }
        Ok(())
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn select_tensor<'a>(
        tensors: &'a [StrandTensor],
        requested: Option<&str>,
    ) -> Result<&'a StrandTensor> {
        if let Some(name) = requested {
            return tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .with_context(|| format!("tensor {name:?} is absent from artifact"));
        }
        tensors
            .iter()
            .find(|tensor| {
                tensor.is_tq_gpu_compatible()
                    && tensor.rht_mode != RhtMode::Rows
                    && (tensor.rht_mode != RhtMode::Cols || tensor.in_features % 256 == 0)
            })
            .context("artifact contains no currently admitted scalar TQ GPU projection")
    }

    fn activation(cols: usize) -> Vec<f32> {
        (0..cols)
            .map(|index| ((index as f32 + 0.5) * 0.013_671_875).sin())
            .collect()
    }

    fn run_projection(
        ctx: &hawking_core::metal::MetalContext,
        base: &TqDeviceHarness,
        residual: Option<&TqDeviceHarness>,
    ) -> Result<(Vec<f32>, usize)> {
        match residual {
            Some(residual) => base
                .run_gemv_two_pass(ctx, residual)
                .map_err(anyhow::Error::from),
            None => base.run_gemv(ctx).map_err(anyhow::Error::from),
        }
    }

    fn kernel_sequence(
        tensor: &StrandTensor,
        runtime_path: TqRuntimePath,
        accumulate: bool,
    ) -> Vec<String> {
        let mut sequence = Vec::new();
        if tensor.rht_mode == RhtMode::Cols {
            sequence.push("strand_rht_forward_cols".to_owned());
        }
        sequence.push(runtime_path.fused_kernel_name().to_owned());
        sequence.push(
            if accumulate {
                "strand_bitslice_reduce_rows_accum"
            } else {
                "strand_bitslice_reduce_rows"
            }
            .to_owned(),
        );
        if !tensor.outliers.is_empty() {
            sequence.push("strand_outlier_correct".to_owned());
        }
        sequence
    }

    fn float_diff(candidate: &[f32], reference: &[f32]) -> (usize, f64, f64) {
        let mut bit_mismatches = 0usize;
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        for (&got, &want) in candidate.iter().zip(reference) {
            if got.to_bits() != want.to_bits() {
                bit_mismatches += 1;
            }
            let abs = (got as f64 - want as f64).abs();
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(abs / (want as f64).abs().max(1e-9));
        }
        (bit_mismatches, max_abs, max_rel)
    }

    fn recipe(path: TqRuntimePath) -> Value {
        match path {
            TqRuntimePath::Stored => json!({"metadata": "expanded", "codebook": "stored"}),
            TqRuntimePath::CompactMetadata => json!({"metadata": "compact", "codebook": "stored"}),
            TqRuntimePath::HashedQuantile => {
                json!({"metadata": "expanded", "codebook": "hashed_quantile"})
            }
            TqRuntimePath::ComputedAcklam => {
                json!({"metadata": "expanded", "codebook": "computed_acklam"})
            }
        }
    }

    fn write_output(path: Option<&Path>, value: &Value) -> Result<()> {
        let mut bytes = serde_json::to_vec_pretty(value)?;
        bytes.push(b'\n');
        if let Some(path) = path {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
            let mut file = fs::File::create(&tmp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            fs::rename(tmp, path)?;
        } else {
            std::io::stdout().write_all(&bytes)?;
        }
        Ok(())
    }

    pub fn run() -> Result<()> {
        let args = Args::parse();
        validate_admission()?;
        if args.warmups < 3 || args.trials < 10 {
            bail!("Appendix contract requires warmups>=3 and trials>=10");
        }
        if !(7..=64).contains(&args.source_commit.len())
            || !args
                .source_commit
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            bail!("source_commit must be 7-64 lowercase hexadecimal characters");
        }
        let runtime_path = TqRuntimePath::parse(&args.runtime_path).map_err(anyhow::Error::msg)?;
        let run_nonce = std::env::var("HAWKING_PHYSICAL_RUN_NONCE")
            .context("HAWKING_PHYSICAL_RUN_NONCE is required")?;
        if run_nonce.len() != 64
            || !run_nonce
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            bail!("HAWKING_PHYSICAL_RUN_NONCE must be 64 lowercase hexadecimal characters");
        }
        if args.matrix_cell_id.trim().is_empty()
            || args.matrix_model.trim().is_empty()
            || args.matrix_tensor_family.trim().is_empty()
        {
            bail!("matrix cell/model/tensor-family bindings must be non-empty");
        }
        if args.matrix_cell_sha256.len() != 64
            || !args
                .matrix_cell_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            bail!("matrix_cell_sha256 must be 64 lowercase hexadecimal characters");
        }
        let bytes = fs::read(&args.artifact)
            .with_context(|| format!("read artifact {}", args.artifact.display()))?;
        let artifact_sha256 = sha256(&bytes);
        let tensors = read_strand(&bytes).map_err(anyhow::Error::msg)?;
        let tensor = select_tensor(&tensors, args.tensor.as_deref())?;
        if tensor.name != args.matrix_tensor_family {
            bail!("selected artifact tensor does not match matrix tensor family");
        }
        let residual_source = match (&args.residual_artifact, &args.residual_tensor) {
            (Some(path), Some(name)) => {
                let residual_bytes = fs::read(path)
                    .with_context(|| format!("read residual artifact {}", path.display()))?;
                let residual_sha256 = sha256(&residual_bytes);
                if residual_sha256 == artifact_sha256 {
                    bail!("residual artifact must be independently bound from the base artifact");
                }
                let residual_tensors = read_strand(&residual_bytes).map_err(anyhow::Error::msg)?;
                let residual_tensor = select_tensor(&residual_tensors, Some(name))?;
                if residual_tensor.out_features != tensor.out_features
                    || residual_tensor.in_features != tensor.in_features
                {
                    bail!("residual tensor geometry must exactly match the base projection");
                }
                Some((residual_bytes, residual_sha256, residual_tensors))
            }
            (None, None) => None,
            _ => bail!("residual artifact and tensor must be supplied together"),
        };
        let residual_tensor = residual_source.as_ref().map(|(_, _, residual_tensors)| {
            select_tensor(residual_tensors, args.residual_tensor.as_deref())
                .expect("validated residual tensor must remain present")
        });

        let x = activation(tensor.in_features);
        let cpu_q12 = tensor.decode_q12_raw();
        let mut cpu_y = tensor.matvec(&x);
        if let Some(residual_tensor) = residual_tensor {
            let residual_y = residual_tensor.matvec(&x);
            for (base, residual) in cpu_y.iter_mut().zip(residual_y) {
                *base += residual;
            }
        }
        let ctx = hawking_core::metal::MetalContext::new()?;

        let baseline = TqDeviceHarness::prepare(&ctx, tensor, TqRuntimePath::Stored, &x)?;
        let candidate = TqDeviceHarness::prepare(&ctx, tensor, runtime_path, &x)?;
        let residual_baseline = residual_tensor
            .map(|tensor| TqDeviceHarness::prepare(&ctx, tensor, TqRuntimePath::Stored, &x))
            .transpose()?;
        let residual_candidate = residual_tensor
            .map(|tensor| TqDeviceHarness::prepare(&ctx, tensor, runtime_path, &x))
            .transpose()?;
        let mut phase_recorder = PhaseRecorder::new(run_nonce, &args.process_joule_provenance)?;
        let (gpu_q12, _, q12_phase_marker_sha256) =
            phase_recorder.measure("parity", "candidate_q12", None, 0, || {
                candidate.decode_q12(&ctx).map_err(anyhow::Error::from)
            })?;
        let q12_mismatches = gpu_q12
            .iter()
            .zip(&cpu_q12)
            .filter(|(got, want)| got != want)
            .count();

        let residual_q12_evidence = if let (Some(residual_tensor), Some(residual_candidate)) =
            (residual_tensor, residual_candidate.as_ref())
        {
            let cpu_residual_q12 = residual_tensor.decode_q12_raw();
            let (gpu_residual_q12, _, marker) =
                phase_recorder.measure("parity", "candidate_residual_q12", None, 1, || {
                    residual_candidate
                        .decode_q12(&ctx)
                        .map_err(anyhow::Error::from)
                })?;
            let mismatches = gpu_residual_q12
                .iter()
                .zip(&cpu_residual_q12)
                .filter(|(got, want)| got != want)
                .count();
            Some((mismatches, cpu_residual_q12.len(), marker))
        } else {
            None
        };

        let (baseline_run, _, baseline_parity_interval) =
            phase_recorder.measure("parity", "baseline", None, 0, || {
                run_projection(&ctx, &baseline, residual_baseline.as_ref())
            })?;
        let (candidate_run, _, candidate_parity_interval) =
            phase_recorder.measure("parity", "candidate", None, 0, || {
                run_projection(&ctx, &candidate, residual_candidate.as_ref())
            })?;
        let fused_phase_marker_sha256 = phase_recorder.pair(
            "parity",
            None,
            0,
            "baseline",
            baseline_parity_interval,
            candidate_parity_interval,
        )?;
        let (baseline_y, _) = baseline_run;
        let (candidate_y, dispatches) = candidate_run;
        let (fused_bit_mismatches, _, _) = float_diff(&candidate_y, &baseline_y);
        let (_, cpu_max_abs_error, cpu_max_rel_error) = float_diff(&candidate_y, &cpu_y);

        let mut warmup_phase_marker_sha256 = Vec::with_capacity(args.warmups);
        for warmup in 0..args.warmups {
            let baseline_first = warmup % 2 == 0;
            let (baseline_interval, candidate_interval) = if baseline_first {
                let (_, _, baseline_interval) =
                    phase_recorder.measure("warmup", "baseline", None, warmup, || {
                        run_projection(&ctx, &baseline, residual_baseline.as_ref()).map(|_| ())
                    })?;
                let (_, _, candidate_interval) =
                    phase_recorder.measure("warmup", "candidate", None, warmup, || {
                        run_projection(&ctx, &candidate, residual_candidate.as_ref()).map(|_| ())
                    })?;
                (baseline_interval, candidate_interval)
            } else {
                let (_, _, candidate_interval) =
                    phase_recorder.measure("warmup", "candidate", None, warmup, || {
                        run_projection(&ctx, &candidate, residual_candidate.as_ref()).map(|_| ())
                    })?;
                let (_, _, baseline_interval) =
                    phase_recorder.measure("warmup", "baseline", None, warmup, || {
                        run_projection(&ctx, &baseline, residual_baseline.as_ref()).map(|_| ())
                    })?;
                (baseline_interval, candidate_interval)
            };
            warmup_phase_marker_sha256.push(phase_recorder.pair(
                "warmup",
                None,
                warmup,
                if baseline_first {
                    "baseline"
                } else {
                    "candidate"
                },
                baseline_interval,
                candidate_interval,
            )?);
        }
        let mut baseline_wall_ns = Vec::with_capacity(args.trials);
        let mut candidate_wall_ns = Vec::with_capacity(args.trials);
        let mut trial_phase_marker_sha256 = Vec::with_capacity(args.trials);
        for trial in 0..args.trials {
            if trial % 2 == 0 {
                let (_, wall, baseline_interval) =
                    phase_recorder.measure("trial", "baseline", None, trial, || {
                        run_projection(&ctx, &baseline, residual_baseline.as_ref()).map(|_| ())
                    })?;
                baseline_wall_ns.push(wall);
                let (_, wall, candidate_interval) =
                    phase_recorder.measure("trial", "candidate", None, trial, || {
                        run_projection(&ctx, &candidate, residual_candidate.as_ref()).map(|_| ())
                    })?;
                candidate_wall_ns.push(wall);
                trial_phase_marker_sha256.push(phase_recorder.pair(
                    "trial",
                    None,
                    trial,
                    "baseline",
                    baseline_interval,
                    candidate_interval,
                )?);
            } else {
                let (_, wall, candidate_interval) =
                    phase_recorder.measure("trial", "candidate", None, trial, || {
                        run_projection(&ctx, &candidate, residual_candidate.as_ref()).map(|_| ())
                    })?;
                candidate_wall_ns.push(wall);
                let (_, wall, baseline_interval) =
                    phase_recorder.measure("trial", "baseline", None, trial, || {
                        run_projection(&ctx, &baseline, residual_baseline.as_ref()).map(|_| ())
                    })?;
                baseline_wall_ns.push(wall);
                trial_phase_marker_sha256.push(phase_recorder.pair(
                    "trial",
                    None,
                    trial,
                    "candidate",
                    baseline_interval,
                    candidate_interval,
                )?);
            }
        }

        let rht_mode = match tensor.rht_mode {
            RhtMode::None => "none",
            RhtMode::Cols => "cols",
            RhtMode::Rows => unreachable!("row-RHT is excluded by tensor admission"),
        };
        let rht_blocks = if tensor.rht_mode == RhtMode::Cols {
            tensor.in_features / 256
        } else {
            0
        };
        let base_kernel_sequence = kernel_sequence(tensor, runtime_path, false);
        let residual_kernel_sequence =
            residual_tensor.map(|tensor| kernel_sequence(tensor, runtime_path, true));
        let mut kernel_sequence = base_kernel_sequence.clone();
        if let Some(sequence) = &residual_kernel_sequence {
            kernel_sequence.extend(sequence.iter().cloned());
        }
        if dispatches != kernel_sequence.len() {
            bail!(
                "observed dispatch count {dispatches} differs from exact pass sequence {}",
                kernel_sequence.len()
            );
        }
        let matrix_identity = stamp_value(
            json!({
                "schema": MATRIX_IDENTITY_SCHEMA,
                "cell_id": args.matrix_cell_id,
                "matrix_cell_sha256": args.matrix_cell_sha256,
                "model": args.matrix_model,
                "tensor_family": args.matrix_tensor_family,
                "shape": {"rows": candidate.rows, "cols": candidate.cols},
                "k_bits": candidate.k_bits,
                "l_bits": candidate.l_bits,
                "runtime_path": runtime_path.as_str(),
                "artifact_sha256": artifact_sha256,
                "artifact_tensor_name": tensor.name,
            }),
            "identity_sha256",
        )?;
        let matrix_identity_sha256 = digest_field(&matrix_identity, "identity_sha256")?;
        let mut pass_sequence = vec![json!({
            "ordinal": 0,
            "role": "base_overwrite",
            "artifact_sha256": artifact_sha256,
            "tensor_name": tensor.name,
            "runtime_path": runtime_path.as_str(),
            "rht_mode": rht_mode,
            "rht_blocks": rht_blocks,
            "outlier_count": tensor.outliers.len(),
            "reduce_kernel": "strand_bitslice_reduce_rows",
            "kernel_sequence": base_kernel_sequence,
        })];
        if let (Some((_, residual_sha256, _)), Some(residual_tensor), Some(sequence)) = (
            residual_source.as_ref(),
            residual_tensor,
            residual_kernel_sequence.as_ref(),
        ) {
            let residual_rht_mode = match residual_tensor.rht_mode {
                RhtMode::None => "none",
                RhtMode::Cols => "cols",
                RhtMode::Rows => unreachable!("row-RHT is excluded by tensor admission"),
            };
            let residual_rht_blocks = if residual_tensor.rht_mode == RhtMode::Cols {
                residual_tensor.in_features / 256
            } else {
                0
            };
            pass_sequence.push(json!({
                "ordinal": 1,
                "role": "residual_accumulate",
                "artifact_sha256": residual_sha256,
                "tensor_name": residual_tensor.name,
                "runtime_path": runtime_path.as_str(),
                "rht_mode": residual_rht_mode,
                "rht_blocks": residual_rht_blocks,
                "outlier_count": residual_tensor.outliers.len(),
                "reduce_kernel": "strand_bitslice_reduce_rows_accum",
                "kernel_sequence": sequence,
            }));
        }
        let rht_cols_passes = pass_sequence
            .iter()
            .filter(|pass| pass.get("rht_mode").and_then(Value::as_str) == Some("cols"))
            .count();
        let outlier_corrected_passes = pass_sequence
            .iter()
            .filter(|pass| {
                pass.get("outlier_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
                    > 0
            })
            .count();
        let residual_passes = usize::from(residual_tensor.is_some());
        let feature_identity = stamp_value(
            json!({
                "schema": FEATURE_IDENTITY_SCHEMA,
                "matrix_identity_sha256": matrix_identity_sha256,
                "matrix_cell_sha256": args.matrix_cell_sha256,
                "projection_recipe": if residual_passes == 1 {
                    "two_pass_residual_accumulate"
                } else {
                    "single_pass_overwrite"
                },
                "projection_passes": pass_sequence.len(),
                "pass_sequence": pass_sequence,
                "feature_counts": {
                    "rht_cols_passes": rht_cols_passes,
                    "outlier_corrected_passes": outlier_corrected_passes,
                    "residual_accumulate_passes": residual_passes,
                    "dispatches_per_invocation": dispatches,
                },
            }),
            "feature_identity_sha256",
        )?;
        let feature_identity_sha256 = digest_field(&feature_identity, "feature_identity_sha256")?;
        let dispatch_geometry_sha256 = canonical_sha256(&json!({
            "rows": candidate.rows,
            "cols": candidate.cols,
            "blocks": candidate.blocks,
            "rht_blocks": rht_blocks,
            "outlier_count": tensor.outliers.len(),
            "projection_passes": pass_sequence.len(),
            "residual_passes": residual_passes,
            "dispatches_per_invocation": dispatches,
            "kernel_sequence": kernel_sequence,
            "feature_identity_sha256": feature_identity_sha256,
        }))?;
        let (phase_markers, process_energy_counters) = phase_recorder.finish()?;

        let traffic = candidate.traffic;
        let residual_traffic = residual_candidate.as_ref().map(|harness| harness.traffic);
        let payload = traffic.payload_bytes
            + residual_traffic
                .map(|value| value.payload_bytes)
                .unwrap_or(0);
        let metadata = traffic.metadata_bytes(runtime_path)
            + residual_traffic
                .map(|value| value.metadata_bytes(runtime_path))
                .unwrap_or(0);
        let codebook_staging = traffic.staged_codebook_bytes(runtime_path)
            + residual_traffic
                .map(|value| value.staged_codebook_bytes(runtime_path))
                .unwrap_or(0);
        let partial_roundtrip = traffic.partial_roundtrip_bytes
            + residual_traffic
                .map(|value| value.partial_roundtrip_bytes)
                .unwrap_or(0);
        let total = payload + metadata + codebook_staging + partial_roundtrip;
        // A residual projection is a second, independently bound tensor with
        // the same geometry.  Normalize combined traffic by every weight that
        // was actually decoded; using only the base tensor here would double
        // the reported bpw for a two-pass probe while silently changing the
        // denominator between otherwise comparable receipts.
        let total_projection_weights =
            traffic.weights + residual_traffic.map(|value| value.weights).unwrap_or(0);
        let residual_probe = if let (
            Some((residual_bytes, residual_sha256, _)),
            Some(residual_tensor),
            Some(residual_candidate),
            Some((residual_mismatches, residual_values, residual_marker)),
        ) = (
            residual_source.as_ref(),
            residual_tensor,
            residual_candidate.as_ref(),
            residual_q12_evidence.as_ref(),
        ) {
            json!({
                "schema": RESIDUAL_PROBE_SCHEMA,
                "enabled": true,
                "artifact": {
                    "path": args.residual_artifact,
                    "sha256": residual_sha256,
                    "size_bytes": residual_bytes.len(),
                },
                "tensor": {
                    "name": residual_tensor.name,
                    "rows": residual_candidate.rows,
                    "cols": residual_candidate.cols,
                    "weights": residual_candidate.rows * residual_candidate.cols,
                    "blocks": residual_candidate.blocks,
                    "k_bits": residual_candidate.k_bits,
                    "l_bits": residual_candidate.l_bits,
                    "rht_mode": if residual_tensor.rht_mode == RhtMode::Cols {
                        "cols"
                    } else {
                        "none"
                    },
                    "rht_blocks": if residual_tensor.rht_mode == RhtMode::Cols {
                        residual_tensor.in_features / 256
                    } else {
                        0
                    },
                    "outlier_count": residual_tensor.outliers.len(),
                },
                "runtime_path": runtime_path.as_str(),
                "recipe": recipe(runtime_path),
                "metal": {
                    "compiled": true,
                    "kernel": runtime_path.fused_kernel_name(),
                    "reduce_kernel": "strand_bitslice_reduce_rows_accum",
                    "host_entry_bytes": residual_candidate.host_entry_bytes,
                    "gpu_entry_bytes": residual_candidate.gpu_entry_bytes,
                },
                "q12_parity": {
                    "exact": *residual_mismatches == 0,
                    "mismatches": residual_mismatches,
                    "values_compared": residual_values,
                    "phase_marker_sha256": residual_marker,
                },
            })
        } else {
            json!({"schema": RESIDUAL_PROBE_SCHEMA, "enabled": false})
        };
        let result = json!({
            "schema": SCHEMA,
            "source_commit": args.source_commit,
            "artifact": {
                "path": args.artifact,
                "sha256": artifact_sha256,
                "size_bytes": bytes.len(),
            },
            "matrix_identity": matrix_identity,
            "feature_identity": feature_identity,
            "residual_probe": residual_probe,
            "device": {"name": ctx.device_name()},
            "runtime_path": runtime_path.as_str(),
            "recipe": recipe(runtime_path),
            "tensor": {
                "name": tensor.name,
                "rows": candidate.rows,
                "cols": candidate.cols,
                "weights": candidate.rows * candidate.cols,
                "blocks": candidate.blocks,
                "k_bits": candidate.k_bits,
                "l_bits": candidate.l_bits,
            },
            "admission": {"eligible": candidate.admission.eligible, "reason": Value::Null},
            "metal": {
                "compiled": true,
                "kernel": runtime_path.fused_kernel_name(),
                "host_entry_bytes": candidate.host_entry_bytes,
                "gpu_entry_bytes": candidate.gpu_entry_bytes,
            },
            "parity": {
                "projection_recipe": if residual_passes == 1 {
                    "two_pass_residual_accumulate"
                } else {
                    "single_pass_overwrite"
                },
                "projection_passes": pass_sequence.len(),
                "feature_identity_sha256": feature_identity_sha256,
                "exact_q12": q12_mismatches == 0,
                "q12_mismatches": q12_mismatches,
                "q12_values_compared": cpu_q12.len(),
                "exact_fused_vs_stored_gpu": fused_bit_mismatches == 0,
                "fused_bit_mismatches": fused_bit_mismatches,
                "fused_values_compared": candidate_y.len(),
                "cpu_reference_max_abs_error": cpu_max_abs_error,
                "cpu_reference_max_rel_error": cpu_max_rel_error,
                "q12_phase_marker_sha256": q12_phase_marker_sha256,
                "fused_phase_marker_sha256": fused_phase_marker_sha256,
            },
            "feature_census": {
                "schema": "hawking.tq_device_feature_census.v2",
                "rht_mode": rht_mode,
                "rht_blocks": rht_blocks,
                "rht_exercised": rht_blocks > 0,
                "outlier_count": tensor.outliers.len(),
                "outlier_exercised": !tensor.outliers.is_empty(),
                "projection_passes": pass_sequence.len(),
                "residual_passes": residual_passes,
                "residual_exercised": residual_passes == 1,
                "dispatches_per_invocation": dispatches,
                "dispatch_geometry_sha256": dispatch_geometry_sha256,
                "kernel_sequence": kernel_sequence,
                "feature_identity_sha256": feature_identity_sha256,
            },
            "logical_traffic": {
                "payload": payload,
                "metadata": metadata,
                "codebook_staging": codebook_staging,
                "partial_roundtrip": partial_roundtrip,
                "compressed_runtime_total": total,
                "compressed_runtime_bpw": total as f64 * 8.0
                    / total_projection_weights as f64,
            },
            "benchmark": {
                "projection_recipe": if residual_passes == 1 {
                    "two_pass_residual_accumulate"
                } else {
                    "single_pass_overwrite"
                },
                "feature_identity_sha256": feature_identity_sha256,
                "warmups": args.warmups,
                "trials": args.trials,
                "baseline_wall_ns": baseline_wall_ns,
                "candidate_wall_ns": candidate_wall_ns,
                "dispatches_per_invocation": dispatches,
                "order": "paired_interleaved_alternating",
                "warmup_phase_marker_sha256": warmup_phase_marker_sha256,
                "trial_phase_marker_sha256": trial_phase_marker_sha256,
            },
            "phase_markers": phase_markers,
            "process_energy_counters": process_energy_counters,
            "physical_counters": {"measured": false},
            "default_change_requested": false,
        });
        write_output(args.output.as_deref(), &result)
    }
}

#[cfg(target_os = "macos")]
fn main() {
    if let Err(error) = macos::run() {
        eprintln!("hawking-tq-device-probe: {error:#}");
        std::process::exit(1);
    }
}
