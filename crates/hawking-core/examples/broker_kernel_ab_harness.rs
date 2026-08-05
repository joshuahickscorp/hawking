//! Broker kernel A/B harness scaffold (dry-run + simulated trials).
//!
//! This example does **not** load a Gravity artifact, does **not** dispatch
//! DeepSeek-V4 forward-lane modules, and **cannot** promote a candidate into
//! the serve path. It exists so later tuning work has a single gated entry
//! point that always scores parity before speed.
//!
//! ```sh
//! cargo run -p hawking-core --example broker_kernel_ab_harness -- --dry-run
//! cargo run -p hawking-core --example broker_kernel_ab_harness -- \
//!   --kernel act_quant --simulate-parity pass --simulate-speed-ratio 0.8
//! ```
//!
//! Real Metal A/B should call into existing sealed oracles / sweep examples
//! and feed their results into [`hawking_core::broker_kernel_ab::decide_promotion`].

use hawking_core::broker_kernel_ab::{
    broker_kernel_ranking, decide_promotion, layer_cost_snapshot, AbTrialInput, KernelFamily,
    PromotionVerdict, HARNESS_SCHEMA,
};
use serde_json::json;
use std::env;
use std::process::ExitCode;

fn usage() -> ! {
    eprintln!(
        "usage:
  broker_kernel_ab_harness --dry-run
  broker_kernel_ab_harness --list
  broker_kernel_ab_harness --kernel <family> --simulate-parity pass|fail \
      [--simulate-speed-ratio R] [--authority-us U] [--candidate-us U]

families: {}",
        KernelFamily::all()
            .iter()
            .map(|f| f.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    std::process::exit(2);
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1).peekable();
    if args.peek().is_none() {
        usage();
    }

    let mut dry_run = false;
    let mut list = false;
    let mut kernel: Option<KernelFamily> = None;
    let mut sim_parity: Option<bool> = None;
    let mut speed_ratio: Option<f64> = None;
    let mut authority_us: Option<u64> = None;
    let mut candidate_us: Option<u64> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dry-run" => dry_run = true,
            "--list" => list = true,
            "--kernel" => {
                let v = args.next().unwrap_or_else(|| usage());
                kernel = Some(KernelFamily::parse(&v).unwrap_or_else(|| {
                    eprintln!("unknown kernel family {v:?}");
                    usage();
                }));
            }
            "--simulate-parity" => {
                let v = args.next().unwrap_or_else(|| usage());
                sim_parity = Some(match v.as_str() {
                    "pass" | "true" | "1" => true,
                    "fail" | "false" | "0" => false,
                    _ => {
                        eprintln!("--simulate-parity expects pass|fail");
                        usage();
                    }
                });
            }
            "--simulate-speed-ratio" => {
                let v = args.next().unwrap_or_else(|| usage());
                speed_ratio = Some(v.parse().unwrap_or_else(|_| {
                    eprintln!("invalid --simulate-speed-ratio");
                    usage();
                }));
            }
            "--authority-us" => {
                let v = args.next().unwrap_or_else(|| usage());
                authority_us = Some(v.parse().unwrap_or_else(|_| usage()));
            }
            "--candidate-us" => {
                let v = args.next().unwrap_or_else(|| usage());
                candidate_us = Some(v.parse().unwrap_or_else(|_| usage()));
            }
            "--help" | "-h" => usage(),
            other => {
                eprintln!("unknown argument {other}");
                usage();
            }
        }
    }

    let costs = layer_cost_snapshot();
    if dry_run || list {
        let ranking: Vec<_> = broker_kernel_ranking()
            .iter()
            .map(|e| {
                json!({
                    "rank": e.rank,
                    "family": e.family.as_str(),
                    "scope": e.scope,
                    "authority_kernel": e.authority_kernel,
                    "candidate_kernels": e.candidate_kernels,
                    "current_cost": e.current_cost,
                    "tuning_levers": e.tuning_levers,
                    "parity_oracle": e.parity_oracle,
                })
            })
            .collect();
        let out = json!({
            "schema": HARNESS_SCHEMA,
            "mode": if list { "list" } else { "dry_run" },
            "serve_promoted": false,
            "layer_cost_snapshot": costs,
            "ranking": ranking,
            "gate_policy": {
                "parity_required": true,
                "auto_serve_promote": false,
                "verdicts": ["reject_parity", "reject_no_win", "candidate_ready"],
            },
            "note": "Dry-run only. Wire sealed oracles for real A/B; never flip serve from this tool.",
        });
        println!("{}", serde_json::to_string_pretty(&out).expect("json"));
        return ExitCode::SUCCESS;
    }

    let family = kernel.unwrap_or_else(|| {
        eprintln!("--kernel is required unless --dry-run / --list");
        usage();
    });
    let parity_pass = sim_parity.unwrap_or_else(|| {
        eprintln!("--simulate-parity is required for non-dry runs in this scaffold");
        usage();
    });

    let entry = broker_kernel_ranking()
        .iter()
        .find(|e| e.family == family)
        .expect("ranking covers all families");

    let (auth_us, cand_us) = match (authority_us, candidate_us, speed_ratio) {
        (Some(a), Some(c), _) => (Some(a), Some(c)),
        (None, None, Some(r)) => {
            let a = 10_000u64;
            let c = ((a as f64) * r).max(1.0).round() as u64;
            (Some(a), Some(c))
        }
        (Some(a), None, Some(r)) => (Some(a), Some(((a as f64) * r).max(1.0).round() as u64)),
        _ => (authority_us, candidate_us),
    };

    let decision = decide_promotion(AbTrialInput {
        family,
        authority_kernel: entry.authority_kernel.to_owned(),
        candidate_kernel: entry
            .candidate_kernels
            .first()
            .copied()
            .unwrap_or("unspecified_candidate")
            .to_owned(),
        parity_pass,
        parity_detail: if parity_pass {
            None
        } else {
            Some("simulated parity failure".into())
        },
        authority_gpu_p50_us: auth_us,
        candidate_gpu_p50_us: cand_us,
    });

    let out = json!({
        "schema": HARNESS_SCHEMA,
        "mode": "simulate",
        "layer_cost_snapshot": costs,
        "decision": decision,
        "forward_lane_touched": false,
        "metal_dispatched": false,
    });
    println!("{}", serde_json::to_string_pretty(&out).expect("json"));

    match decision.verdict {
        PromotionVerdict::RejectParity => ExitCode::from(1),
        PromotionVerdict::RejectNoWin => ExitCode::from(3),
        PromotionVerdict::CandidateReady => ExitCode::SUCCESS,
    }
}
