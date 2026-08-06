//! CLI wrapper for the evidence-oriented headless HIDE runner.
//!
//! This binary deliberately requires an explicit `--model-url` for a live
//! agent run. It never treats the offline/stub kernel as a model benchmark.

use hide_backend::headless::{run_headless_audit, write_sealed_receipt, HeadlessRunConfig};
use hide_backend::BackendHost;
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    workspace: PathBuf,
    goal: String,
    model_url: Option<String>,
    receipt: Option<PathBuf>,
    max_transitions: u32,
    allow_incomplete: bool,
}

fn usage() -> &'static str {
    "Usage: hide-headless --goal <text> [options]\n\
     \n\
     Options:\n\
       --workspace <path>        Workspace to open (default: current directory)\n\
       --goal <text>             Literal agent objective\n\
       --goal-file <path>        Read the objective from a UTF-8 file\n\
       --model-url <url>         Running hawking serve base URL (required for a live run)\n\
       --receipt <path>          Receipt target (default: <workspace>/.hide/receipts/)\n\
       --max-transitions <n>     Driver cap; kernel governor remains authoritative (default: 200)\n\
       --allow-incomplete        Exit 0 for blocked/paused/step-limit receipts\n\
       --help                    Show this message\n"
}

fn parse_args() -> anyhow::Result<Args> {
    let mut workspace = std::env::current_dir()?;
    let mut goal = None;
    let mut goal_file = None;
    let mut model_url = None;
    let mut receipt = None;
    let mut max_transitions = 200u32;
    let mut allow_incomplete = false;
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        match flag.as_str() {
            "--workspace" => workspace = PathBuf::from(required_value(&mut argv, "--workspace")?),
            "--goal" => goal = Some(required_value(&mut argv, "--goal")?),
            "--goal-file" => {
                goal_file = Some(PathBuf::from(required_value(&mut argv, "--goal-file")?))
            }
            "--model-url" => model_url = Some(required_value(&mut argv, "--model-url")?),
            "--receipt" => receipt = Some(PathBuf::from(required_value(&mut argv, "--receipt")?)),
            "--max-transitions" => {
                max_transitions = required_value(&mut argv, "--max-transitions")?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--max-transitions must be an integer"))?
            }
            "--allow-incomplete" => allow_incomplete = true,
            "--help" | "-h" => {
                print!("{}", usage());
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument {other:?}\n\n{}", usage()),
        }
    }
    if goal.is_some() && goal_file.is_some() {
        anyhow::bail!("use exactly one of --goal or --goal-file")
    }
    let goal = match (goal, goal_file) {
        (Some(goal), None) => goal,
        (None, Some(path)) => std::fs::read_to_string(&path).map_err(|error| {
            anyhow::anyhow!("could not read --goal-file {}: {error}", path.display())
        })?,
        (None, None) => anyhow::bail!("--goal or --goal-file is required\n\n{}", usage()),
        (Some(_), Some(_)) => unreachable!("checked above"),
    };
    Ok(Args {
        workspace,
        goal,
        model_url,
        receipt,
        max_transitions,
        allow_incomplete,
    })
}

fn required_value(argv: &mut impl Iterator<Item = String>, flag: &str) -> anyhow::Result<String> {
    argv.next()
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = parse_args()?;
    let workspace = args.workspace.canonicalize().map_err(|error| {
        anyhow::anyhow!(
            "could not resolve workspace {}: {error}",
            args.workspace.display()
        )
    })?;
    let receipt_path = args.receipt.unwrap_or_else(|| {
        workspace
            .join(".hide")
            .join("receipts")
            .join(format!("headless-{}.json", hide_core::ids::now_ms()))
    });
    let host = BackendHost::open_workspace(&workspace)?;
    let result = run_headless_audit(
        &host,
        HeadlessRunConfig {
            goal: args.goal,
            model_url: args.model_url,
            max_transitions: args.max_transitions,
            ..HeadlessRunConfig::default()
        },
    )
    .await?;
    write_sealed_receipt(&receipt_path, &result.receipt)?;
    println!(
        "hide-headless status={} receipt={}",
        result.status.as_str(),
        receipt_path.display()
    );
    if !result.status.is_complete() && !args.allow_incomplete {
        anyhow::bail!(
            "headless run did not complete (status={}); receipt was written",
            result.status.as_str()
        );
    }
    Ok(())
}
