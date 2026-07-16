//! Raw TQ-native speculative-verifier parity and cost probe.
//!
//! This binary is intentionally lease-admitted and receipt-incomplete. The
//! Python Appendix runner binds resource state, validates the raw evidence, and
//! emits canonical parity/curve receipts. No proposer runs here: it isolates the
//! target hardware interaction before speculative policy can hide its cost.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("hawking-tq-spec-probe requires macOS Metal");
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
    use hawking_core::model::qwen_dense::QwenDense;
    use hawking_core::{Engine, EngineConfig};
    use serde::Deserialize;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::collections::HashSet;
    use std::fs;
    use std::io::{Read, Write};
    use std::os::unix::fs::MetadataExt;
    use std::path::{Path, PathBuf};
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    const SCHEMA: &str = "hawking.spec_tq_batched_raw.v1";
    const PROMPT_SCHEMA: &str = "hawking.spec_token_prompts.v1";
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
    #[command(name = "hawking-tq-spec-probe")]
    struct Args {
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        prompts: PathBuf,
        #[arg(long, default_value = "stored")]
        runtime_path: String,
        #[arg(long, default_value_t = 256)]
        generated_tokens: usize,
        #[arg(long, default_value_t = 3)]
        warmups_per_batch: usize,
        #[arg(long, default_value_t = 5)]
        repeats_per_batch: usize,
        #[arg(long)]
        source_commit: String,
        #[arg(long)]
        parity_cell_id: String,
        #[arg(long)]
        curve_cell_id: String,
        /// Hash-bound libproc/SDK provenance produced by the release authority.
        #[arg(long)]
        process_joule_provenance: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PromptSet {
        schema: String,
        tokenizer_sha256: String,
        prompts: Vec<TokenPrompt>,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct TokenPrompt {
        id: String,
        text: Option<String>,
        token_ids: Option<Vec<u32>>,
    }

    struct ResolvedPrompt {
        token_ids: Vec<u32>,
    }

    struct IntervalEnergySample {
        phase: String,
        role: String,
        batch: usize,
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
        batch: usize,
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
        Ok(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&ordered(value))?)
        ))
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
            batch: usize,
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
                Some(batch),
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
            batch: usize,
            iteration: usize,
            first_role: &str,
            baseline_interval_sha256: String,
            verifier_interval_sha256: String,
        ) -> Result<String> {
            let verifier_digest = verifier_interval_sha256.clone();
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
            let verifier_interval_id = self
                .intervals
                .iter()
                .find(|interval| {
                    interval.get("interval_sha256").and_then(Value::as_str)
                        == Some(verifier_interval_sha256.as_str())
                })
                .and_then(|interval| interval.get("interval_id"))
                .and_then(Value::as_str)
                .context("verifier phase interval lacks its stable interval id")?
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
                    "verifier_interval_sha256": verifier_interval_sha256,
                    "baseline_interval_id": baseline_interval_id,
                    "verifier_interval_id": verifier_interval_id,
                }),
                "phase_marker_sha256",
            )?;
            let digest = digest_field(&pair, "phase_marker_sha256")?;
            self.pairs.push(pair);
            if phase == "trial" {
                let sample = self
                    .energy_intervals
                    .iter()
                    .find(|sample| sample.interval_sha256 == verifier_digest)
                    .context("trial verifier interval lacks local process-energy samples")?;
                if sample.phase != phase
                    || sample.role != "verifier"
                    || sample.batch != batch
                    || sample.iteration != iteration
                {
                    bail!("trial process-energy sample is not the exact verifier interval");
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
                .sort_by_key(|record| (record.batch, record.iteration));
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
        if std::env::var("HAWKING_APPENDIX_SPEC_ADMITTED").as_deref() != Ok("1") {
            bail!("spec probe must be launched by the Appendix heavy-lease runner");
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

    fn sha256_file(path: &Path) -> Result<(String, u64)> {
        let mut file = fs::File::open(path)?;
        let size = file.metadata()?.len();
        let mut digest = Sha256::new();
        let mut chunk = [0u8; 1024 * 1024];
        loop {
            let count = file.read(&mut chunk)?;
            if count == 0 {
                break;
            }
            digest.update(&chunk[..count]);
        }
        Ok((format!("{:x}", digest.finalize()), size))
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

    fn prefill_prefix(model: &mut QwenDense, prompt: &[u32]) -> Result<()> {
        model.kv.reset();
        for (position, &token) in prompt[..prompt.len() - 1].iter().enumerate() {
            model.forward_token_greedy_tcb(token, position)?;
        }
        Ok(())
    }

    fn greedy_continuation(
        model: &mut QwenDense,
        prompt: &[u32],
        count: usize,
    ) -> Result<(Vec<u32>, Vec<u64>)> {
        prefill_prefix(model, prompt)?;
        let mut token = *prompt.last().expect("validated non-empty prompt");
        let mut position = prompt.len() - 1;
        let mut generated = Vec::with_capacity(count);
        let mut wall_ns = Vec::with_capacity(count);
        for _ in 0..count {
            let start = Instant::now();
            let next = model.forward_token_greedy_tcb(token, position)?;
            wall_ns.push(start.elapsed().as_nanos() as u64);
            generated.push(next);
            token = next;
            position += 1;
        }
        Ok((generated, wall_ns))
    }

    fn verify_continuation(
        model: &mut QwenDense,
        prompt: &[u32],
        generated: &[u32],
        batch: usize,
        count: usize,
    ) -> Result<(usize, u64)> {
        debug_assert_eq!(count % batch, 0);
        prefill_prefix(model, prompt)?;
        let mut inputs = Vec::with_capacity(count);
        inputs.push(*prompt.last().expect("validated non-empty prompt"));
        inputs.extend_from_slice(&generated[..count - 1]);
        let mut mismatches = 0usize;
        let mut total_ns = 0u64;
        let position0 = prompt.len() - 1;
        for (group, input) in inputs.chunks_exact(batch).enumerate() {
            let offset = group * batch;
            let positions: Vec<usize> =
                (0..batch).map(|index| position0 + offset + index).collect();
            let start = Instant::now();
            let (predictions, _residuals) = model.forward_tokens_verify(input, &positions)?;
            total_ns = total_ns.saturating_add(start.elapsed().as_nanos() as u64);
            if predictions.len() != batch {
                bail!(
                    "B={batch} verifier returned {} predictions for {batch} inputs",
                    predictions.len()
                );
            }
            mismatches += predictions
                .iter()
                .zip(&generated[offset..offset + batch])
                .filter(|(got, want)| got != want)
                .count();
        }
        Ok((mismatches, total_ns))
    }

    fn baseline_corpus(
        model: &mut QwenDense,
        prompts: &[ResolvedPrompt],
        count: usize,
    ) -> Result<Vec<Vec<u32>>> {
        prompts
            .iter()
            .map(|prompt| {
                greedy_continuation(model, &prompt.token_ids, count).map(|(generated, _)| generated)
            })
            .collect()
    }

    fn baseline_mismatches(observed: &[Vec<u32>], canonical: &[Vec<u32>], count: usize) -> usize {
        observed
            .iter()
            .zip(canonical)
            .map(|(got, want)| {
                got.iter()
                    .zip(&want[..count])
                    .filter(|(left, right)| left != right)
                    .count()
            })
            .sum()
    }

    fn verifier_corpus(
        model: &mut QwenDense,
        prompts: &[ResolvedPrompt],
        canonical: &[Vec<u32>],
        batch: usize,
        count: usize,
    ) -> Result<usize> {
        let mut mismatches = 0usize;
        for (prompt, generated) in prompts.iter().zip(canonical) {
            mismatches += verify_continuation(model, &prompt.token_ids, generated, batch, count)?.0;
        }
        Ok(mismatches)
    }

    fn balanced_batch_order(run_nonce: &str, phase: &str, round: usize) -> Vec<usize> {
        let mut batches: Vec<usize> = (1..=8).collect();
        batches.sort_by_key(|batch| {
            Sha256::digest(format!("{run_nonce}:{phase}:{round}:{batch}").as_bytes()).to_vec()
        });
        batches
    }

    fn configure_runtime(args: &Args) {
        std::env::set_var("HAWKING_QWEN_TQ", "1");
        std::env::set_var("HAWKING_QWEN_TQ_PATH", &args.artifact);
        std::env::set_var("HAWKING_QWEN_TQ_STRICT", "1");
        std::env::set_var("HAWKING_QWEN_TQ_REQUIRE_ALL_LINEAR", "1");
        std::env::set_var("HAWKING_QWEN_TQ_REQUIRE_GPU", "1");
        std::env::set_var("HAWKING_TQ_RUNTIME_PATH", &args.runtime_path);
        std::env::set_var("HAWKING_QWEN_TCB", "1");
        std::env::set_var("HAWKING_QWEN_PREFIX_CACHE", "0");
        std::env::set_var("HAWKING_QWEN_VOCAB_PRUNE", "32000");
        std::env::set_var("HAWKING_QWEN_Q4K_LMHEAD", "1");
        std::env::set_var("HAWKING_QWEN_Q4K_PREDEC", "1");
        std::env::remove_var("HAWKING_QWEN_TQ_CPU");
    }

    pub fn run() -> Result<()> {
        let args = Args::parse();
        validate_admission()?;
        if args.generated_tokens < 256 {
            bail!("parity contract requires generated_tokens>=256");
        }
        if args.warmups_per_batch < 3 || args.repeats_per_batch < 5 {
            bail!("measurement contract requires warmups-per-batch>=3 and repeats-per-batch>=5");
        }
        if !(7..=64).contains(&args.source_commit.len())
            || !args
                .source_commit
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            bail!("source_commit must be 7-64 lowercase hexadecimal characters");
        }
        let runtime_path =
            hawking_core::TqRuntimePath::parse(&args.runtime_path).map_err(anyhow::Error::msg)?;
        let run_nonce = std::env::var("HAWKING_PHYSICAL_RUN_NONCE")
            .context("HAWKING_PHYSICAL_RUN_NONCE is required")?;
        if run_nonce.len() != 64
            || !run_nonce
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            bail!("HAWKING_PHYSICAL_RUN_NONCE must be 64 lowercase hexadecimal characters");
        }
        if args.parity_cell_id.trim().is_empty() || args.curve_cell_id.trim().is_empty() {
            bail!("spec matrix cell bindings must be non-empty");
        }
        if !args.weights.is_file() || !args.artifact.is_file() || !args.prompts.is_file() {
            bail!("weights, artifact, and prompt-set paths must be regular files");
        }
        let prompt_bytes = fs::read(&args.prompts)?;
        let prompt_set: PromptSet = serde_json::from_slice(&prompt_bytes)?;
        if prompt_set.schema != PROMPT_SCHEMA || prompt_set.prompts.len() < 20 {
            bail!("prompt set must use {PROMPT_SCHEMA} and contain at least 20 prompts");
        }
        if prompt_set.tokenizer_sha256.len() != 64
            || !prompt_set
                .tokenizer_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            bail!("prompt set tokenizer_sha256 must be lowercase SHA-256");
        }
        let mut ids = std::collections::HashSet::new();
        for prompt in &prompt_set.prompts {
            if prompt.id.trim().is_empty() || !ids.insert(prompt.id.as_str()) {
                bail!("prompt ids must be non-empty and unique");
            }
            if prompt.text.is_some() == prompt.token_ids.is_some() {
                bail!(
                    "prompt {:?} must contain exactly one of text or token_ids",
                    prompt.id
                );
            }
        }

        // Bind the tokenizer source the Qwen loader will actually consume. A
        // sidecar tokenizer.json wins; otherwise the full GGUF hash binds the
        // embedded tokenizer metadata together with the target model.
        let (weights_sha256, weights_size) = sha256_file(&args.weights)?;
        let (artifact_sha256, artifact_size) = sha256_file(&args.artifact)?;
        let tokenizer_path = args
            .weights
            .parent()
            .map(|parent| parent.join("tokenizer.json"))
            .filter(|path| path.is_file());
        let (tokenizer_sha256, tokenizer_source, tokenizer_size) =
            if let Some(path) = tokenizer_path.as_ref() {
                let (sha256, size) = sha256_file(path)?;
                (sha256, "tokenizer_json", size)
            } else {
                (weights_sha256.clone(), "gguf_embedded", weights_size)
            };
        if prompt_set.tokenizer_sha256 != tokenizer_sha256 {
            bail!("prompt set tokenizer hash does not match the tokenizer the model will use");
        }

        configure_runtime(&args);
        let mut model = <QwenDense as Engine>::load(&args.weights, EngineConfig::default())?;
        if model.metal_ctx.is_none() {
            bail!("TQ speculative probe requires a Metal device");
        }
        let mut resolved_prompts = Vec::with_capacity(prompt_set.prompts.len());
        for prompt in &prompt_set.prompts {
            let token_ids = match (&prompt.text, &prompt.token_ids) {
                (Some(text), None) => model.tokenizer.encode(text, true)?,
                (None, Some(token_ids)) => token_ids.clone(),
                _ => unreachable!("validated exactly one prompt representation"),
            };
            if token_ids.len() < 2 {
                bail!(
                    "prompt {:?} resolves to fewer than two token ids",
                    prompt.id
                );
            }
            if token_ids
                .iter()
                .any(|&token| token as usize >= model.config.vocab_size)
            {
                bail!("prompt {:?} contains an out-of-vocabulary token", prompt.id);
            }
            if token_ids.len() + args.generated_tokens + 7 > model.kv.max_seq {
                bail!(
                    "prompt {:?} plus verification window exceeds max_seq",
                    prompt.id
                );
            }
            resolved_prompts.push(ResolvedPrompt { token_ids });
        }
        let coverage = model.tq_gpu_proof_coverage()?;
        if coverage.runtime_path != runtime_path
            || coverage.mapped != coverage.expected_all_linear
            || coverage.gpu_resident != coverage.expected_all_linear
        {
            bail!(
                "TQ proof coverage is incomplete or uses the wrong runtime: {:?}",
                coverage
            );
        }

        let max_count = (1usize..=8)
            .map(|batch| args.generated_tokens.div_ceil(batch) * batch)
            .max()
            .unwrap();
        let mut canonical = Vec::with_capacity(resolved_prompts.len());
        for prompt in &resolved_prompts {
            canonical.push(greedy_continuation(&mut model, &prompt.token_ids, max_count)?.0);
        }

        let mut phase_recorder =
            PhaseRecorder::new(run_nonce.clone(), &args.process_joule_provenance)?;
        let mut warmup_markers: Vec<Vec<String>> = (0..8).map(|_| Vec::new()).collect();
        for warmup in 0..args.warmups_per_batch {
            for batch in balanced_batch_order(&run_nonce, "warmup", warmup) {
                let count = args.generated_tokens.div_ceil(batch) * batch;
                let baseline_first = (warmup + batch) % 2 == 0;
                let (baseline_mismatch, verifier_mismatch, baseline_marker, verifier_marker) =
                    if baseline_first {
                        let (outputs, _, baseline_marker) =
                            phase_recorder.measure("warmup", "baseline", batch, warmup, || {
                                baseline_corpus(&mut model, &resolved_prompts, count)
                            })?;
                        let baseline_mismatch = baseline_mismatches(&outputs, &canonical, count);
                        let (verifier_mismatch, _, verifier_marker) =
                            phase_recorder.measure("warmup", "verifier", batch, warmup, || {
                                verifier_corpus(
                                    &mut model,
                                    &resolved_prompts,
                                    &canonical,
                                    batch,
                                    count,
                                )
                            })?;
                        (
                            baseline_mismatch,
                            verifier_mismatch,
                            baseline_marker,
                            verifier_marker,
                        )
                    } else {
                        let (verifier_mismatch, _, verifier_marker) =
                            phase_recorder.measure("warmup", "verifier", batch, warmup, || {
                                verifier_corpus(
                                    &mut model,
                                    &resolved_prompts,
                                    &canonical,
                                    batch,
                                    count,
                                )
                            })?;
                        let (outputs, _, baseline_marker) =
                            phase_recorder.measure("warmup", "baseline", batch, warmup, || {
                                baseline_corpus(&mut model, &resolved_prompts, count)
                            })?;
                        (
                            baseline_mismatches(&outputs, &canonical, count),
                            verifier_mismatch,
                            baseline_marker,
                            verifier_marker,
                        )
                    };
                if baseline_mismatch != 0 || verifier_mismatch != 0 {
                    bail!("B={batch} warmup {warmup} failed exact parity");
                }
                warmup_markers[batch - 1].push(phase_recorder.pair(
                    "warmup",
                    batch,
                    warmup,
                    if baseline_first {
                        "baseline"
                    } else {
                        "verifier"
                    },
                    baseline_marker,
                    verifier_marker,
                )?);
            }
        }

        let mut repeat_rows: Vec<Vec<Value>> = (0..8).map(|_| Vec::new()).collect();
        for repeat in 0..args.repeats_per_batch {
            for batch in balanced_batch_order(&run_nonce, "trial", repeat) {
                let count = args.generated_tokens.div_ceil(batch) * batch;
                let baseline_first = (repeat + batch) % 2 == 0;
                let (
                    baseline_mismatch,
                    verifier_mismatch,
                    baseline_wall_ns,
                    verifier_wall_ns,
                    baseline_marker,
                    verifier_marker,
                ) = if baseline_first {
                    let (outputs, baseline_wall_ns, baseline_marker) =
                        phase_recorder.measure("trial", "baseline", batch, repeat, || {
                            baseline_corpus(&mut model, &resolved_prompts, count)
                        })?;
                    let baseline_mismatch = baseline_mismatches(&outputs, &canonical, count);
                    let (verifier_mismatch, verifier_wall_ns, verifier_marker) = phase_recorder
                        .measure("trial", "verifier", batch, repeat, || {
                            verifier_corpus(&mut model, &resolved_prompts, &canonical, batch, count)
                        })?;
                    (
                        baseline_mismatch,
                        verifier_mismatch,
                        baseline_wall_ns,
                        verifier_wall_ns,
                        baseline_marker,
                        verifier_marker,
                    )
                } else {
                    let (verifier_mismatch, verifier_wall_ns, verifier_marker) = phase_recorder
                        .measure("trial", "verifier", batch, repeat, || {
                            verifier_corpus(&mut model, &resolved_prompts, &canonical, batch, count)
                        })?;
                    let (outputs, baseline_wall_ns, baseline_marker) =
                        phase_recorder.measure("trial", "baseline", batch, repeat, || {
                            baseline_corpus(&mut model, &resolved_prompts, count)
                        })?;
                    (
                        baseline_mismatches(&outputs, &canonical, count),
                        verifier_mismatch,
                        baseline_wall_ns,
                        verifier_wall_ns,
                        baseline_marker,
                        verifier_marker,
                    )
                };
                let mismatches = baseline_mismatch + verifier_mismatch;
                let phase_marker_sha256 = phase_recorder.pair(
                    "trial",
                    batch,
                    repeat,
                    if baseline_first {
                        "baseline"
                    } else {
                        "verifier"
                    },
                    baseline_marker,
                    verifier_marker,
                )?;
                repeat_rows[batch - 1].push(json!({
                    "repeat": repeat,
                    "baseline_wall_ns": baseline_wall_ns,
                    "verifier_wall_ns": verifier_wall_ns,
                    "phase_marker_sha256": phase_marker_sha256,
                    "exact_token_match": mismatches == 0,
                    "mismatches": mismatches,
                    "skipped": 0,
                }));
            }
        }
        let (phase_markers, process_energy_counters) = phase_recorder.finish()?;
        let phase_markers_sha256 = digest_field(&phase_markers, "phase_markers_sha256")?;
        let mut batch_rows = Vec::with_capacity(8);
        let mut protocol_batches = Vec::with_capacity(8);
        for batch in 1usize..=8 {
            let count = args.generated_tokens.div_ceil(batch) * batch;
            let rows = &repeat_rows[batch - 1];
            let baseline_wall_ns: Vec<u64> = rows
                .iter()
                .map(|row| row["baseline_wall_ns"].as_u64().unwrap())
                .collect();
            let verifier_wall_ns: Vec<u64> = rows
                .iter()
                .map(|row| row["verifier_wall_ns"].as_u64().unwrap())
                .collect();
            let mismatches: u64 = rows
                .iter()
                .map(|row| row["mismatches"].as_u64().unwrap())
                .sum();
            batch_rows.push(json!({
                "b": batch,
                "prompts": resolved_prompts.len(),
                "generated_tokens_per_prompt": count,
                "values_compared": count * resolved_prompts.len() * args.repeats_per_batch * 2,
                "exact_token_match": mismatches == 0,
                "mismatches": mismatches,
                "skipped": 0,
                "baseline_greedy_wall_ns": baseline_wall_ns,
                "verifier_wall_ns": verifier_wall_ns,
            }));
            protocol_batches.push(json!({
                "b": batch,
                "repeats": rows,
            }));
        }

        let prompt_sha256 = format!("{:x}", Sha256::digest(&prompt_bytes));
        let output = json!({
            "schema": SCHEMA,
            "source_commit": args.source_commit,
            "model": {
                "path": args.weights,
                "sha256": weights_sha256,
                "size_bytes": weights_size,
                "family": "qwen_dense",
            },
            "artifact": {
                "path": args.artifact,
                "sha256": artifact_sha256,
                "size_bytes": artifact_size,
            },
            "matrix_identity": {
                "runtime_path": runtime_path.as_str(),
                "parity_cell_id": args.parity_cell_id,
                "curve_cell_id": args.curve_cell_id,
                "model_sha256": weights_sha256,
                "artifact_sha256": artifact_sha256,
                "tokenizer_sha256": tokenizer_sha256,
                "prompt_set_sha256": prompt_sha256,
            },
            "prompt_set": {
                "path": args.prompts,
                "sha256": prompt_sha256,
                "schema": PROMPT_SCHEMA,
                "prompts": resolved_prompts.len(),
                "tokenizer_sha256": tokenizer_sha256,
            },
            "tokenizer": {
                "source": tokenizer_source,
                "path": tokenizer_path,
                "sha256": tokenizer_sha256,
                "size_bytes": tokenizer_size,
            },
            "device": {
                "name": model.metal_ctx.as_ref().unwrap().device_name(),
                "profile": "Studio-M3Ultra-96",
            },
            "runtime_path": runtime_path.as_str(),
            "kernel": runtime_path.small_batch_kernel_name(),
            "coverage": {
                "expected_all_linear": coverage.expected_all_linear,
                "mapped": coverage.mapped,
                "gpu_resident": coverage.gpu_resident,
                "residual_gpu_resident": coverage.residual_gpu_resident,
            },
            "target_identity": {
                "reference": "tq_single_token_greedy",
                "verifier": "tq_batch_major_b1_b8",
                "greedy_tie_break": "canonical_qwen_argmax",
                "all_owned_projections_tq_native": true,
            },
            "profile_flags": {
                "vocab_prune": 32000,
                "q4k_lmhead": true,
                "q4k_predec": true,
                "prefix_cache": false,
            },
            "batches": batch_rows,
            "measurement_protocol": {
                "warmups_per_batch": args.warmups_per_batch,
                "independent_repeats_per_batch": args.repeats_per_batch,
                "randomized_balanced_batch_order": true,
                "paired_interleaved_baseline": true,
                "baseline_reused_across_batches": false,
                "phase_marker_schema": PHASE_MARKERS_SCHEMA,
                "phase_markers_sha256": phase_markers_sha256,
                "monotone_transform_applied": false,
                "batches": protocol_batches,
            },
            "phase_markers": phase_markers,
            "process_energy_counters": process_energy_counters,
            "physical_counters": {"measured": false},
            "default_change_requested": false,
        });
        write_output(args.output.as_deref(), &output)
    }
}

#[cfg(target_os = "macos")]
fn main() {
    if let Err(error) = macos::run() {
        eprintln!("hawking-tq-spec-probe: {error:#}");
        std::process::exit(1);
    }
}
