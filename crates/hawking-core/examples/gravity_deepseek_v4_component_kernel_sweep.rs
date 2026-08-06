//! Bounded M3 Metal threadgroup sweep for the two source-native DeepSeek-V4
//! component authorities.
//!
//! This example deliberately performs **no** full-model load, decode loop,
//! routing, generation, HCLI call, or BASE_TRUE_TPS measurement.  It only
//! rebinds two real tensors from the sealed 43-layer Gravity stream and sweeps
//! the threadgroup dimension of their already-proven direct matvec kernels:
//!
//! * FP8 control: `layers.0.attn.wq_a.weight [1024,4096]` E4M3FN + E8M0FNU.
//! * FP4 routed-expert gate: `layers.0.ffn.experts.0.w1.weight [2048,2048]`
//!   packed E2M1FNx2 + E8M0FNU.
//!
//! The 2048-thread rung is intentionally retained in the ladder even though a
//! normal Apple GPU pipeline cannot admit it.  It is recorded as unsupported
//! rather than silently deleted, so the deepest stable rung is evidence rather
//! than a hand-selected endpoint.
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_component_kernel_sweep -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --out /absolute/path/to/DSV4F_COMPONENT_KERNEL_SWEEP.json
//! ```

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(
        std::io::Error::other("gravity_deepseek_v4_component_kernel_sweep requires macOS Metal")
            .into(),
    )
}

#[cfg(target_os = "macos")]
#[path = "gravity_deepseek_v4_fp8_metal_probe.rs"]
mod fp8_probe;

#[cfg(target_os = "macos")]
#[path = "gravity_deepseek_v4_fp4_metal_probe.rs"]
mod fp4_probe;

#[cfg(target_os = "macos")]
mod macos {
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::error::Error;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;
    use std::path::{Path, PathBuf};

    const THREADGROUP_LADDER: &[u32] = &[32, 64, 128, 256, 512, 1024, 2048];
    const DEFAULT_WARMUPS: usize = 3;
    const DEFAULT_TRIALS: usize = 9;

    type SweepResult<T> = Result<T, Box<dyn Error>>;

    struct Args {
        artifact: PathBuf,
        out: PathBuf,
        warmups: usize,
        trials: usize,
    }

    fn failure(message: impl Into<String>) -> Box<dyn Error> {
        std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
    }

    fn parse_positive(value: Option<String>, flag: &str) -> SweepResult<usize> {
        let value = value.ok_or_else(|| failure(format!("{flag} needs a value")))?;
        let parsed = value
            .parse::<usize>()
            .map_err(|_| failure(format!("{flag} must be a positive integer")))?;
        if parsed == 0 {
            return Err(failure(format!("{flag} must be positive")));
        }
        Ok(parsed)
    }

    fn parse_args() -> SweepResult<Args> {
        let mut artifact = None::<PathBuf>;
        let mut out = None::<PathBuf>;
        let mut warmups = DEFAULT_WARMUPS;
        let mut trials = DEFAULT_TRIALS;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--artifact" => artifact = args.next().map(PathBuf::from),
                "--out" => out = args.next().map(PathBuf::from),
                "--warmups" => warmups = parse_positive(args.next(), "--warmups")?,
                "--trials" => trials = parse_positive(args.next(), "--trials")?,
                "--help" | "-h" => {
                    println!(
                        "usage: gravity_deepseek_v4_component_kernel_sweep --artifact <absolute full Gravity dir> --out <absolute receipt.json> [--warmups N] [--trials N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(failure(format!("unknown argument {other:?}"))),
            }
        }
        let artifact = artifact.ok_or_else(|| failure("--artifact is required"))?;
        let out = out.ok_or_else(|| failure("--out is required"))?;
        if !artifact.is_absolute() || !out.is_absolute() {
            return Err(failure("--artifact and --out must be absolute paths"));
        }
        Ok(Args {
            artifact,
            out,
            warmups,
            trials,
        })
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn sealed_receipt(mut receipt: Value) -> SweepResult<(Value, String)> {
        if !receipt.is_object() {
            return Err(failure("sweep receipt root must be an object"));
        }
        if receipt.get("seal_sha256").is_some() {
            return Err(failure("sweep receipt already contains a seal"));
        }
        let seal = sha256(&serde_json::to_vec(&receipt)?);
        receipt
            .as_object_mut()
            .expect("receipt object was checked")
            .insert("seal_sha256".to_owned(), Value::String(seal.clone()));
        Ok((receipt, seal))
    }

    fn write_new_receipt(path: &Path, receipt: &Value) -> SweepResult<()> {
        if path.exists() {
            return Err(failure(format!(
                "refusing to overwrite existing component sweep receipt {}",
                path.display()
            )));
        }
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .ok_or_else(|| failure("--out needs a parent directory"))?;
        fs::create_dir_all(parent)?;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| failure("--out filename must be UTF-8"))?;
        let temporary = parent.join(format!(
            ".{name}.{}.component-sweep.tmp",
            std::process::id()
        ));
        let bytes = serde_json::to_vec_pretty(receipt)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| {
                failure(format!(
                    "cannot create sweep receipt temporary file: {error}"
                ))
            })?;
        if let Err(error) = file
            .write_all(&bytes)
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.sync_all())
        {
            let _ = fs::remove_file(&temporary);
            return Err(Box::new(error));
        }
        if let Err(error) = fs::rename(&temporary, path) {
            let _ = fs::remove_file(&temporary);
            return Err(Box::new(error));
        }
        File::open(parent)?.sync_all()?;
        Ok(())
    }

    fn string_at<'a>(value: &'a Value, pointer: &str, label: &str) -> SweepResult<&'a str> {
        value
            .pointer(pointer)
            .and_then(Value::as_str)
            .ok_or_else(|| failure(format!("component result lacks {label}")))
    }

    fn u64_at(value: &Value, pointer: &str, label: &str) -> SweepResult<u64> {
        value
            .pointer(pointer)
            .and_then(Value::as_u64)
            .ok_or_else(|| failure(format!("component result lacks {label}")))
    }

    fn clone_at(value: &Value, pointer: &str, label: &str) -> SweepResult<Value> {
        value
            .pointer(pointer)
            .cloned()
            .ok_or_else(|| failure(format!("component result lacks {label}")))
    }

    fn nonpassing_rungs(component: &Value) -> SweepResult<Vec<Value>> {
        let candidates = component
            .pointer("/ladder/candidates")
            .and_then(Value::as_array)
            .ok_or_else(|| failure("component result lacks ladder candidates"))?;
        Ok(candidates
            .iter()
            .filter(|candidate| {
                candidate.get("status").and_then(Value::as_str)
                    != Some("PASS_GPU_TIMESTAMPED_CPU_PARITY")
            })
            .cloned()
            .collect())
    }

    pub fn run() -> SweepResult<()> {
        let args = parse_args()?;
        let fp8 = super::fp8_probe::macos::sweep_component(
            &args.artifact,
            args.warmups,
            args.trials,
            THREADGROUP_LADDER,
        )?;
        let fp4 = super::fp4_probe::macos::sweep_component(
            &args.artifact,
            args.warmups,
            args.trials,
            THREADGROUP_LADDER,
        )?;
        let fp8_device = string_at(&fp8, "/metal/device", "FP8 Metal device")?.to_owned();
        let fp4_device = string_at(&fp4, "/metal/device", "FP4 Metal device")?.to_owned();
        if fp8_device != fp4_device {
            return Err(failure(
                "FP8 and FP4 component sweeps used different Metal devices",
            ));
        }
        if !fp8_device.contains("M3") {
            return Err(failure(format!(
                "this bounded campaign requires an Apple M3 Metal run, found {fp8_device:?}"
            )));
        }
        let fp8_manifest =
            string_at(&fp8, "/artifact/manifest_seal_sha256", "FP8 manifest seal")?.to_owned();
        let fp4_manifest =
            string_at(&fp4, "/artifact/manifest_seal_sha256", "FP4 manifest seal")?.to_owned();
        if fp8_manifest != fp4_manifest {
            return Err(failure(
                "FP8 and FP4 component sweeps bound different artifacts",
            ));
        }
        let fp8_dispatches = u64_at(&fp8, "/metal/observed_gpu_dispatches", "FP8 dispatches")?;
        let fp4_dispatches = u64_at(&fp4, "/metal/observed_gpu_dispatches", "FP4 dispatches")?;
        let fp8_waits = u64_at(&fp8, "/metal/observed_cpu_visible_waits", "FP8 waits")?;
        let fp4_waits = u64_at(&fp4, "/metal/observed_cpu_visible_waits", "FP4 waits")?;
        let fp8_cbs = u64_at(
            &fp8,
            "/metal/physical_trace_command_buffers",
            "FP8 physical command buffers",
        )?;
        let fp4_cbs = u64_at(
            &fp4,
            "/metal/physical_trace_command_buffers",
            "FP4 physical command buffers",
        )?;
        let fp8_encoders = u64_at(
            &fp8,
            "/metal/physical_trace_compute_encoders",
            "FP8 physical encoders",
        )?;
        let fp4_encoders = u64_at(
            &fp4,
            "/metal/physical_trace_compute_encoders",
            "FP4 physical encoders",
        )?;
        if fp8_dispatches != fp8_waits
            || fp4_dispatches != fp4_waits
            || fp8_dispatches != fp8_cbs
            || fp4_dispatches != fp4_cbs
            || fp8_dispatches != fp8_encoders
            || fp4_dispatches != fp4_encoders
        {
            return Err(failure(
                "component sweep command-buffer/encoder/dispatch/wait accounting did not reconcile",
            ));
        }
        let fp8_deepest = clone_at(
            &fp8,
            "/ladder/deepest_stable_threadgroup_rung",
            "FP8 deepest stable rung",
        )?;
        let fp4_deepest = clone_at(
            &fp4,
            "/ladder/deepest_stable_threadgroup_rung",
            "FP4 deepest stable rung",
        )?;
        let fp8_winner = clone_at(&fp8, "/ladder/winner", "FP8 winner")?;
        let fp4_winner = clone_at(&fp4, "/ladder/winner", "FP4 winner")?;
        let fp8_nonpassing = nonpassing_rungs(&fp8)?;
        let fp4_nonpassing = nonpassing_rungs(&fp4)?;

        let unsigned = json!({
            "schema": "hawking.gravity.deepseek_v4.m3_component_kernel_sweep.v1",
            "status": "PASS_REAL_M3_METAL_COMPONENT_SWEEP_NOT_FULL_RUNTIME",
            "scope": {
                "component_microbenchmark_only": true,
                "source_hash_bound_full_gravity_artifact": true,
                "not_a_full_model_load": true,
                "not_a_full_token_or_generation": true,
                "not_a_full_43_layer_runtime_adapter": true,
                "not_a_route_or_MoE_execution_claim": true,
                "not_a_HCLI_measurement": true,
                "not_a_BASE_TRUE_TPS_measurement": true,
                "not_a_kernel_promotion_into_a_runtime": true,
            },
            "reproduction": {
                "example": "cargo run --release -p hawking-core --example gravity_deepseek_v4_component_kernel_sweep",
                "warmup_dispatches_per_supported_geometry": args.warmups,
                "measured_gpu_timestamped_dispatches_per_supported_geometry": args.trials,
                "threadgroup_ladder": THREADGROUP_LADDER,
                "timing_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime only; host times are separately reported",
                "parity_authority": "exact source-native CPU byte decoder and per-supported-geometry output comparison",
            },
            "artifact_binding": {
                "path": args.artifact,
                "manifest_seal_sha256": fp8_manifest,
                "manifest_seal_verified_by_each_component": true,
            },
            "metal": {
                "device": fp8_device,
                "fallback": false,
                "fallback_count": 0,
                "aggregate_gpu_dispatches": fp8_dispatches + fp4_dispatches,
                "aggregate_command_buffers": fp8_cbs + fp4_cbs,
                "aggregate_compute_encoders": fp8_encoders + fp4_encoders,
                "aggregate_cpu_visible_waits": fp8_waits + fp4_waits,
                "empty_command_buffers": 0,
                "accounting_reconciled": true,
            },
            "deepest_stable_threadgroup_rung": {
                "fp8_control_matvec": fp8_deepest,
                "fp4_routed_expert_matvec": fp4_deepest,
            },
            "winner_per_component": {
                "fp8_control_matvec": fp8_winner,
                "fp4_routed_expert_matvec": fp4_winner,
            },
            "unsupported_or_failed_rungs_preserved": {
                "fp8_control_matvec": fp8_nonpassing,
                "fp4_routed_expert_matvec": fp4_nonpassing,
            },
            "components": {
                "fp8_control_matvec": fp8,
                "fp4_routed_expert_matvec": fp4,
            },
            "next_boundary": "The winning geometries are component-only evidence. A registered content-addressed 43-layer V4 adapter, native full-graph parity, actual routing/attention/state execution, and eligible 8K/512-token BASE_TRUE_TPS remain required before any runtime or 100-TPS claim.",
        });
        let (receipt, seal) = sealed_receipt(unsigned)?;
        write_new_receipt(&args.out, &receipt)?;
        println!(
            "{}",
            serde_json::to_string(&json!({
                "status": "PASS_REAL_M3_METAL_COMPONENT_SWEEP_NOT_FULL_RUNTIME",
                "receipt": args.out,
                "seal_sha256": seal,
                "aggregate_gpu_dispatches": fp8_dispatches + fp4_dispatches,
            }))?
        );
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}
