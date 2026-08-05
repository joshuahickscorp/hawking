//! HCLI — Hawking's headless local-model product surface.
//!
//! This is intentionally a CLI-first shell over the real HIDE/Hawking backend:
//! durable contextual sessions, local model calls, agent receipts, evidence
//! procurement, safe tool invocation, throughput measurement, and a parallel
//! analysis swarm.  It does not claim a graphical interface, cloud proxy,
//! full-source V4 parity, unlimited uploads, or an isolated concurrent write
//! swarm.

use anyhow::{anyhow, bail, Context, Result};
use hide_backend::hcli_bridge::{
    hcli_model_optimization_identity, AgentRealization, AgentResponse, CapabilitiesResponse,
    ComponentState, ComponentStatus, DecodeTelemetry, DurableTurnReceipt, GenerateResponse,
    HcliAutonomy, HcliError, HcliErrorCode, HcliMethod, HcliOperation, HcliRequest, HcliResponse,
    HcliResult, MethodCapability, OperationStatus, SessionAction, SessionRequest, SessionResponse,
    SessionSummary, StatusArea, StatusResponse, HCLI_MODEL_OPTIMIZATION_NOT_AVAILABLE_CODE,
};
use hide_backend::hcli_profile::HcliProfile;
use hide_backend::hcli_research::{run_hcli_research, HcliResearchConfig};
use hide_backend::hcli_sources::{HcliSourceContext, HcliSourceStore};
use hide_backend::hcli_swarm::{run_parallel_analysis_swarm, HcliSwarmConfig};
use hide_backend::headless::{run_headless_audit, write_sealed_receipt, HeadlessRunConfig};
use hide_backend::{BackendHost, HttpModelProvider};
use hide_core::runtime::{InferenceRequest, ModelProvider, SamplerProfile, StreamChunk};
use hide_core::tool::ToolCall;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const HCLI_SCHEMA: &str = "hcli.command.v1";

#[derive(Debug, Default)]
struct Options {
    values: BTreeMap<String, String>,
    attachments: Vec<String>,
    flags: BTreeSet<String>,
    positional: Vec<String>,
}

impl Options {
    fn value(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(String::as_str)
    }

    fn required(&self, name: &str) -> Result<String> {
        self.value(name)
            .map(str::to_string)
            .ok_or_else(|| anyhow!("--{name} is required"))
    }

    fn flag(&self, name: &str) -> bool {
        self.flags.contains(name)
    }

    /// `--attach` is intentionally repeatable: each value must identify an
    /// existing local HCLI evidence ref/hash and is validated by the object
    /// store before a model call begins.
    fn attachments(&self) -> &[String] {
        &self.attachments
    }

    fn number<T>(&self, name: &str, default: T) -> Result<T>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        match self.value(name) {
            Some(value) => value
                .parse()
                .map_err(|error| anyhow!("--{name} must be valid: {error}")),
            None => Ok(default),
        }
    }
}

fn usage() -> &'static str {
    "HCLI — Hawking's headless local-model shell\n\
     \n\
     Usage:\n\
       hcli capabilities [--workspace PATH] [--model-url URL] [--json]\n\
       hcli model inspect [--workspace PATH] [--model-url URL] [--json]\n\
       hcli model optimize [--json]\n\
       hcli run --prompt TEXT [--session NAME] --model-url URL [--max-output-tokens N]
                [--attach OREF_OR_BLAKE3]...
                [--workspace PATH] [--json]\n\
       hcli repl --model-url URL [--session NAME] [--workspace PATH]\n\
       hcli agent --goal TEXT --model-url URL [--profile balanced|power|maximum]\n\
                  [--max-transitions N] [--attach OREF_OR_BLAKE3]... [--receipt PATH] [--allow-incomplete]\n\
       hcli swarm --goal TEXT --model-url URL [--profile balanced|power|maximum]\n\
                  [--lanes N] [--max-concurrency N] [--max-transitions N]\n\
                  [--no-synthesis] [--receipt PATH] [--allow-incomplete]\n\
       hcli research --topic TEXT --model-url URL [--per-query-limit N]\n\
                     [--read-budget N] [--max-rounds N] [--receipt PATH] [--allow-incomplete]\n\
       hcli source ingest (--file PATH | --stdin) [--label NAME] [--mime TYPE]\n\
                          [--workspace PATH] [--json]\n\
       hcli source list [--workspace PATH] [--json]\n\
       hcli source show (--ref OREF | --hash BLAKE3) [--workspace PATH] [--json]\n\
       hcli source context --attach OREF_OR_BLAKE3 [--attach OREF_OR_BLAKE3]... [--workspace PATH] [--json]\n\
       hcli bench --prompt TEXT --model-url URL [--warmup N] [--runs N]\n\
                  [--max-output-tokens N] [--receipt PATH]\n\
       hcli bridge jsonl [--workspace PATH] [--model-url URL]\n\
       hcli session show [--session NAME] [--workspace PATH]\n\
       hcli session fork --session NAME --at-seq N [--workspace PATH]\n\
       hcli tool list [--workspace PATH]\n\
       hcli tool run --name TOOL --args JSON [--session NAME] [--workspace PATH]\n\
       hcli serve (--weights PATH | --gravity PATH) [--addr HOST:PORT]\n\
                  [--hawking-bin PATH] [--profile fast|exact|race|efficient]\n\
     \n\
     Environment:\n\
       HCLI_MODEL_URL     Default explicit local model URL\n\
       HCLI_HAWKING_BIN   Default Hawking executable for `hcli serve`\n\
     \n\
     `agent` has a real model planner/act/verify loop and sealed receipt.\n\
     `swarm` runs real parallel analysis kernels in shared workspace sessions;\n\
     it is not an isolated concurrent write swarm. `research` currently uses\n\
     arXiv title/abstract procurement with CAS-pinned evidence.\n\
     \n\
     Gravity is the public model-optimization identity; Condense is its engine\n\
     operation. HCLI reports that taxonomy but does not run optimization or\n\
     promote a live V4 diagnostic into a full-model, numeric-parity, Metal, or\n\
     TPS claim. `hcli serve --gravity` is a compatibility launcher for\n\
     an already-built Gravity artifact; inspect `model_optimization_identity`\n\
     for the canonical Hawking command and compatibility scope. `hcli model\n\
     optimize` and JSONL `model_optimization` return the stable\n\
     `model_optimization_not_available` refusal rather than accepting partial\n\
     optimization controls.\n"
}

fn parse_options(args: &[String]) -> Result<Options> {
    const BOOL_FLAGS: &[&str] = &["json", "allow-incomplete", "help", "stdin", "no-synthesis"];
    let mut parsed = Options::default();
    let mut index = 0usize;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--" {
            parsed.positional.extend(args[index + 1..].iter().cloned());
            break;
        }
        if let Some(name) = arg.strip_prefix("--") {
            if name.is_empty() {
                bail!("empty option name");
            }
            if BOOL_FLAGS.contains(&name) {
                parsed.flags.insert(name.to_string());
            } else {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("--{name} requires a value"))?;
                if value.starts_with("--") {
                    bail!("--{name} requires a value");
                }
                if name == "attach" {
                    if value.trim().is_empty() {
                        bail!("--attach requires a non-empty source reference or content hash");
                    }
                    parsed.attachments.push(value.to_string());
                } else if parsed
                    .values
                    .insert(name.to_string(), value.to_string())
                    .is_some()
                {
                    bail!("--{name} was provided more than once");
                }
            }
        } else {
            parsed.positional.push(arg.to_string());
        }
        index += 1;
    }
    Ok(parsed)
}

fn workspace(options: &Options) -> Result<PathBuf> {
    let raw = options
        .value("workspace")
        .map(PathBuf::from)
        .unwrap_or(std::env::current_dir()?);
    raw.canonicalize()
        .with_context(|| format!("could not resolve workspace {}", raw.display()))
}

fn model_url(options: &Options) -> Option<String> {
    options
        .value("model-url")
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .map(str::to_string)
        .or_else(|| {
            std::env::var("HCLI_MODEL_URL")
                .ok()
                .map(|url| url.trim().to_string())
                .filter(|url| !url.is_empty())
        })
}

fn named_session(host: &BackendHost, options: &Options) -> hide_core::ids::SessionId {
    match options.value("session") {
        Some(name) => host.services.session_named(name),
        None => host.services.session_named("hcli"),
    }
}

fn profile(options: &Options, default: HcliProfile) -> Result<HcliProfile> {
    match options.value("profile") {
        Some(name) => HcliProfile::parse(name).ok_or_else(|| {
            anyhow!("unknown --profile {name:?}; expected balanced, power, or maximum")
        }),
        None => Ok(default),
    }
}

fn path_or_default(root: &std::path::Path, options: &Options, prefix: &str) -> PathBuf {
    options
        .value("receipt")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            root.join(".hide")
                .join("receipts")
                .join(format!("{prefix}-{}.json", hide_core::ids::now_ms()))
        })
}

/// Resolve the caller's repeatable explicit evidence selection before a model
/// invocation. This performs no network fetch and returns only derivatives the
/// object store explicitly allows a model-facing reader to see.
fn selected_source_context(
    root: &std::path::Path,
    options: &Options,
) -> Result<Option<HcliSourceContext>> {
    if options.attachments().is_empty() {
        return Ok(None);
    }
    let identifiers = options
        .attachments()
        .iter()
        .flat_map(|raw| raw.split(','))
        .map(str::trim)
        .filter(|identifier| !identifier.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if identifiers.is_empty() {
        bail!("--attach requires at least one non-empty source reference or content hash");
    }
    HcliSourceStore::open(root)?
        .select_context(&identifiers)
        .map(Some)
}

fn command_envelope(command: &str, value: Value) -> Value {
    json!({
        "schema": HCLI_SCHEMA,
        "command": command,
        "result": value,
    })
}

fn emit(value: Value, compact: bool) -> Result<()> {
    if compact {
        println!("{}", serde_json::to_string(&value)?);
    } else {
        println!("{}", serde_json::to_string_pretty(&value)?);
    }
    Ok(())
}

async fn runtime_preflight(url: &str) -> Value {
    let health_url = format!("{}/healthz", url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_default();
    let health = match client.get(&health_url).send().await {
        Ok(response) => json!({
            "status": if response.status().is_success() { "ready" } else { "unhealthy" },
            "ready": response.status().is_success(),
            "http_status": response.status().as_u16(),
            "url": health_url,
        }),
        Err(error) => json!({
            "status": "unreachable",
            "ready": false,
            "url": health_url,
            "error": error.to_string(),
        }),
    };
    let context = HttpModelProvider::new(url.to_string())
        .get_context_info()
        .await
        .map(|info| {
            json!({
                "status": "available",
                "model_id": info.model_id,
                "arch": info.arch,
                "ctx_len_native": info.ctx_len_native,
                "ctx_len_effective": info.ctx_len_effective,
                "tq_multiplier": info.tq_multiplier,
                "tq_estimated": info.tq_estimated,
                "recurrent_state_bytes": info.recurrent_state_bytes,
                "active_slots": info.active_slots,
                "free_slots": info.free_slots,
                "max_batch": info.max_batch,
                "max_output_tokens": info.max_output_tokens,
                "artifact_seal_sha256": info.artifact_seal_sha256,
                "capability_status": info.capability_status,
                "metal_dispatches": info.metal_dispatches,
                "chat_template": info.chat_template,
            })
        })
        .unwrap_or_else(|| {
            json!({
                "status": "unavailable",
                "note": "runtime did not expose identifiable context facts",
            })
        });
    json!({ "health": health, "context": context })
}

async fn cmd_capabilities(options: &Options) -> Result<Value> {
    let root = workspace(options)?;
    let host = BackendHost::open_workspace(&root)?;
    let status = host.status().await;
    let endpoint = model_url(options);
    let runtime = match endpoint.as_deref() {
        Some(url) => runtime_preflight(url).await,
        None => json!({
            "status": "not_requested",
            "note": "pass --model-url (or HCLI_MODEL_URL) to inspect a local endpoint",
        }),
    };
    Ok(command_envelope(
        "capabilities",
        json!({
            "product": "HCLI",
            "version": env!("CARGO_PKG_VERSION"),
            "model_optimization_identity": hcli_model_optimization_identity(),
            "workspace": root,
            "backend": status,
            "runtime": runtime,
            "surfaces": {
                "durable_contextual_sessions": true,
                "compiled_repo_and_claude_instructions_for_run": true,
                "headless_agent_kernel": true,
                "agent_context_injection": "code-index pack injected on selected agent steps; receipt records manifest/counts",
                "parallel_analysis_swarm": true,
                "evidence_first_research": "arXiv title/abstract adapter with CAS-pinned evidence",
                "local_evidence_ingest": "bounded local-file/stdin ingestion with durable object refs, safe derivatives, and sealed receipts",
                "explicit_source_to_context": "run/chat/agent may opt in to existing local evidence refs with repeatable --attach; only bounded model-facing derivatives are injected",
                "permission_gated_tools_and_mcp": true,
                "machine_bridge": "available through hcli bridge jsonl",
            },
            "honesty_limits": {
                "model_roles": "one explicit local endpoint per HCLI run; no realized multi-model role routing",
                "model_optimization": "Gravity is the canonical public identity and Condense is its engine operation; HCLI reports the shared taxonomy but does not execute optimization. `hcli model optimize` and JSONL `model_optimization` return the stable model_optimization_not_available refusal. See model_optimization_identity.public_cli for the executable Hawking surface.",
                "uploads_to_context": "no HTTP upload route; explicit existing local evidence refs can be attached to one HCLI run/chat/agent invocation",
                "source_ingest_to_context": "opt-in bounded local derivative selection only; no implicit session carry-forward, URL fetch, or unlimited storage",
                "write_swarm_isolation": false,
                "deepseek_v4_gravity": "an explicit --model-url may expose a live V4 diagnostic; HCLI never promotes that into a full-model, numeric-parity, Metal, or TPS claim",
                "tps": "only claimed when runtime supplies completed_decode_forwards and decode_ms",
            },
        }),
    ))
}

/// Read-only, machine-consumable acknowledgement of HCLI's optimization
/// boundary.  It deliberately accepts no execution controls: any values that
/// reached the generic option parser are reported only by name and are never
/// passed to a model, packer, artifact writer, or Hawking subprocess.
fn cmd_model_optimization_status(options: &Options) -> Value {
    let identity = hcli_model_optimization_identity();
    let canonical_hawking_plan_command = identity.public_cli.plan_command.clone();
    let mut received_control_names = options
        .values
        .keys()
        .map(|name| format!("--{name}"))
        .collect::<Vec<_>>();
    received_control_names.extend(
        options
            .flags
            .iter()
            .filter(|name| name.as_str() != "json")
            .map(|name| format!("--{name}")),
    );
    if !options.attachments.is_empty() {
        received_control_names.push("--attach".to_string());
    }
    if !options.positional.is_empty() {
        received_control_names.push("positional".to_string());
    }
    received_control_names.sort();
    received_control_names.dedup();

    command_envelope(
        "model.optimize",
        json!({
            "status": "not_available",
            "model_optimization_identity": identity,
            "refusal": HcliError::model_optimization_not_available(),
            "request": {
                "controls_accepted": false,
                "received_control_names": received_control_names,
                "execution_side_effects": false,
            },
            "next": {
                "canonical_hawking_plan_command": canonical_hawking_plan_command,
                "note": "The named Hawking command is the canonical planning surface; this HCLI response does not assert a produced artifact, runnable V4 model, or benchmark result.",
            },
        }),
    )
}

async fn cmd_run(options: &Options) -> Result<Value> {
    let root = workspace(options)?;
    let prompt = prompt(options)?;
    let endpoint =
        model_url(options).ok_or_else(|| anyhow!("--model-url is required for hcli run"))?;
    let host = BackendHost::open_workspace(&root)?;
    let session = named_session(&host, options);
    let source_context = selected_source_context(&root, options)?;
    let requested_output_cap = match options.value("max-output-tokens") {
        Some(value) => {
            let cap = value
                .parse::<usize>()
                .map_err(|_| anyhow!("--max-output-tokens must be a positive integer"))?;
            if cap == 0 {
                bail!("--max-output-tokens must be greater than zero");
            }
            Some(cap)
        }
        None => None,
    };
    let turn = host
        .hcli_turn_with_output_cap_and_source_context(
            session,
            endpoint.clone(),
            prompt,
            requested_output_cap,
            source_context,
        )
        .await?;
    Ok(command_envelope(
        "run",
        json!({
            "workspace": root,
            "model_url": endpoint,
            "turn": turn,
            "tps_note": "complete_forward_tps is null unless the runtime exposed both completed_decode_forwards and decode_ms",
        }),
    ))
}

async fn cmd_agent(options: &Options) -> Result<Value> {
    let root = workspace(options)?;
    let goal = goal(options)?;
    let profile = profile(options, HcliProfile::Power)?;
    let max_transitions = options.number("max-transitions", profile.budget().max_steps)?;
    let endpoint = model_url(options);
    let host = BackendHost::open_workspace(&root)?;
    let source_context = selected_source_context(&root, options)?;
    let result = run_headless_audit(
        &host,
        HeadlessRunConfig {
            goal,
            model_url: endpoint.clone(),
            session_id: None,
            max_transitions,
            profile,
            source_context,
        },
    )
    .await?;
    let receipt_path = path_or_default(&root, options, "hcli-agent");
    write_sealed_receipt(&receipt_path, &result.receipt)?;
    let complete = result.status.is_complete();
    let value = command_envelope(
        "agent",
        json!({
            "workspace": root,
            "model_url": endpoint,
            "status": result.status.as_str(),
            "receipt_path": receipt_path,
            "receipt": result.receipt,
        }),
    );
    if !complete && !options.flag("allow-incomplete") {
        emit(value, options.flag("json"))?;
        bail!("agent run did not complete; sealed receipt was written (pass --allow-incomplete to accept the receipt exit status)");
    }
    Ok(value)
}

async fn cmd_swarm(options: &Options) -> Result<Value> {
    let root = workspace(options)?;
    let goal = goal(options)?;
    let profile = profile(options, HcliProfile::Power)?;
    let default_lanes = profile.budget().search_breadth as usize;
    let lanes = options.number("lanes", default_lanes)?;
    if lanes == 0 {
        bail!("--lanes must be greater than zero");
    }
    let max_concurrency = options.number("max-concurrency", lanes)?;
    if max_concurrency == 0 {
        bail!("--max-concurrency must be greater than zero");
    }
    let max_transitions = options.number("max-transitions", profile.budget().max_steps)?;
    let endpoint = model_url(options);
    let host = BackendHost::open_workspace(&root)?;
    let result = run_parallel_analysis_swarm(
        &host,
        HcliSwarmConfig {
            goal,
            model_url: endpoint.clone(),
            profile,
            lanes,
            max_concurrency,
            max_transitions,
            synthesize: !options.flag("no-synthesis"),
        },
    )
    .await?;
    let receipt_path = path_or_default(&root, options, "hcli-swarm");
    write_sealed_receipt(&receipt_path, &result.receipt)?;
    let value = command_envelope(
        "swarm",
        json!({
            "workspace": root,
            "model_url": endpoint,
            "complete": result.complete,
            "synthesis": result.synthesis,
            "receipt_path": receipt_path,
            "receipt": result.receipt,
        }),
    );
    if !result.complete && !options.flag("allow-incomplete") {
        emit(value, options.flag("json"))?;
        bail!("parallel analysis swarm did not fully complete; sealed receipt was written (pass --allow-incomplete to accept the receipt exit status)");
    }
    Ok(value)
}

async fn cmd_research(options: &Options) -> Result<Value> {
    let root = workspace(options)?;
    let topic = options
        .value("topic")
        .map(str::to_string)
        .or_else(|| (!options.positional.is_empty()).then(|| options.positional.join(" ")))
        .ok_or_else(|| anyhow!("--topic is required"))?;
    if topic.trim().is_empty() {
        bail!("--topic must not be empty");
    }
    let endpoint = model_url(options);
    let host = BackendHost::open_workspace(&root)?;
    let result = run_hcli_research(
        &host,
        HcliResearchConfig {
            topic,
            model_url: endpoint.clone(),
            per_query_limit: options.number("per-query-limit", 12usize)?,
            read_budget: options.number("read-budget", 24usize)?,
            max_rounds: options.number("max-rounds", 3u32)?,
        },
    )
    .await?;
    let receipt_path = path_or_default(&root, options, "hcli-research");
    write_sealed_receipt(&receipt_path, &result.receipt)?;
    let value = command_envelope(
        "research",
        json!({
            "workspace": root,
            "model_url": endpoint,
            "complete": result.complete,
            "receipt_path": receipt_path,
            "receipt": result.receipt,
        }),
    );
    if !result.complete && !options.flag("allow-incomplete") {
        emit(value, options.flag("json"))?;
        bail!("research pipeline did not complete; sealed receipt was written (pass --allow-incomplete to accept the receipt exit status)");
    }
    Ok(value)
}

async fn cmd_session(subcommand: Option<&str>, options: &Options) -> Result<Value> {
    let root = workspace(options)?;
    let host = BackendHost::open_workspace(&root)?;
    let session = named_session(&host, options);
    match subcommand.unwrap_or("show") {
        "show" | "new" => {
            let projection = host.rebuild_session_projection(session.clone()).await?;
            Ok(command_envelope(
                "session.show",
                json!({
                    "workspace": root,
                    "session_name": options.value("session").unwrap_or("hcli"),
                    "session_id": session,
                    "projection": projection,
                }),
            ))
        }
        "fork" => {
            let at_seq: u64 = options
                .required("at-seq")?
                .parse()
                .map_err(|_| anyhow!("--at-seq must be an integer"))?;
            let (forked, projection) = host.fork_session(session.clone(), at_seq).await?;
            Ok(command_envelope(
                "session.fork",
                json!({
                    "workspace": root,
                    "source_session": session,
                    "at_seq": at_seq,
                    "forked_session": forked,
                    "projection": projection,
                }),
            ))
        }
        other => bail!("unknown session subcommand {other:?}; use show, new, or fork"),
    }
}

async fn cmd_tool(subcommand: Option<&str>, options: &Options) -> Result<Value> {
    let root = workspace(options)?;
    let host = BackendHost::open_workspace(&root)?;
    match subcommand.unwrap_or("list") {
        "list" => Ok(command_envelope(
            "tool.list",
            json!({
                "workspace": root,
                "tools": host.tools.specs(),
                "note": "Tools execute through the real permission and sandbox policy. Configured MCP tools appear in this registry when available.",
            }),
        )),
        "run" => {
            let name = options.required("name")?;
            let raw_args = options.required("args")?;
            let args: Value =
                serde_json::from_str(&raw_args).with_context(|| "--args must be a JSON object")?;
            if !args.is_object() {
                bail!("--args must be a JSON object");
            }
            let session = named_session(&host, options);
            let result = host
                .dispatch_tool(session.clone(), None, ToolCall::new(name.clone(), args))
                .await?;
            Ok(command_envelope(
                "tool.run",
                json!({
                    "workspace": root,
                    "session_id": session,
                    "tool": name,
                    "result": result,
                }),
            ))
        }
        other => bail!("unknown tool subcommand {other:?}; use list or run"),
    }
}

/// Bounded local evidence procurement. Sources are durable and inspectable
/// first; `source context` previews a metadata-only explicit selection before
/// `run`, `chat`, or `agent --attach` injects its bounded derivatives.
async fn cmd_source(subcommand: Option<&str>, options: &Options) -> Result<Value> {
    let root = workspace(options)?;
    let sources = HcliSourceStore::open(&root)?;
    match subcommand.unwrap_or("list") {
        "ingest" => {
            let local_file = options.value("file");
            let from_stdin = options.flag("stdin");
            if local_file.is_some() == from_stdin {
                bail!("source ingest requires exactly one of --file PATH or --stdin");
            }
            let label = options.value("label").map(str::to_string);
            let mime = options.value("mime").map(str::to_string);
            let result = match local_file {
                Some(path) => sources.ingest_file(path, label, mime)?,
                None => {
                    let stdin = io::stdin();
                    let mut locked = stdin.lock();
                    sources.ingest_stdin(&mut locked, label, mime)?
                }
            };
            Ok(command_envelope(
                "source.ingest",
                json!({
                    "workspace": root,
                    "evidence_root": sources.root(),
                    "receipt_path": result.receipt_path,
                    "source": result.receipt["object"].clone(),
                    "receipt": result.receipt,
                    "context_note": "Ingestion never auto-attaches. Use repeatable --attach on hcli run/chat/agent to inject this source's bounded model-facing derivatives for one invocation.",
                }),
            ))
        }
        "list" => {
            let entries = sources.list()?;
            Ok(command_envelope(
                "source.list",
                json!({
                    "workspace": root,
                    "evidence_root": sources.root(),
                    "source_references": entries,
                    "raw_body_included": false,
                    "context_note": "Listing evidence does not make it available to a model turn. Pass one or more listed refs to --attach on hcli run/chat/agent, or inspect the metadata-only pack with hcli source context.",
                }),
            ))
        }
        "show" => {
            let by_ref = options.value("ref");
            let by_hash = options.value("hash");
            if by_ref.is_some() && by_hash.is_some() {
                bail!("source show accepts one identifier: --ref OREF or --hash BLAKE3");
            }
            let identifier = by_ref
                .or(by_hash)
                .map(str::to_string)
                .or_else(|| options.positional.first().cloned())
                .ok_or_else(|| anyhow!("source show requires --ref OREF or --hash BLAKE3"))?;
            Ok(command_envelope(
                "source.show",
                json!({
                    "workspace": root,
                    "evidence_root": sources.root(),
                    "source": sources.show(&identifier)?,
                }),
            ))
        }
        "context" => {
            let context = selected_source_context(&root, options)?.ok_or_else(|| {
                anyhow!("source context requires one or more --attach OREF_OR_BLAKE3 values")
            })?;
            Ok(command_envelope(
                "source.context",
                json!({
                    "workspace": root,
                    "evidence_root": sources.root(),
                    "source_context": context.receipt_json(),
                    "prompt_text_included": false,
                    "note": "This validates the exact bounded derivative selection without echoing source text. Pass the same --attach values to run/chat/agent for actual model-prompt injection.",
                }),
            ))
        }
        other => bail!("unknown source subcommand {other:?}; use ingest, list, show, or context"),
    }
}

async fn cmd_bench(options: &Options) -> Result<Value> {
    let root = workspace(options)?;
    let endpoint =
        model_url(options).ok_or_else(|| anyhow!("--model-url is required for hcli bench"))?;
    let prompt = prompt(options)?;
    let warmup: usize = options.number("warmup", 1usize)?;
    let runs: usize = options.number("runs", 6usize)?;
    if runs == 0 || runs > 100 {
        bail!("--runs must be between 1 and 100");
    }
    let max_output_tokens: usize = options.number("max-output-tokens", 512usize)?;
    if max_output_tokens == 0 || max_output_tokens > 32_768 {
        bail!("--max-output-tokens must be between 1 and 32768");
    }
    let provider = HttpModelProvider::new(endpoint.clone());
    let preflight = runtime_preflight(&endpoint).await;
    if preflight.pointer("/health/ready").and_then(Value::as_bool) != Some(true) {
        let mut blocked = json!({
            "schema": "hcli.model_benchmark.v1",
            "status": "blocked_runtime_unreachable",
            "runtime": preflight,
            "requested": { "warmup": warmup, "runs": runs, "max_output_tokens": max_output_tokens },
        });
        seal(&mut blocked)?;
        let receipt_path = path_or_default(&root, options, "hcli-bench");
        write_sealed_receipt(&receipt_path, &blocked)?;
        return Ok(command_envelope(
            "bench",
            json!({ "receipt_path": receipt_path, "receipt": blocked }),
        ));
    }

    for _ in 0..warmup {
        let _ = one_bench_call(&provider, &prompt, max_output_tokens).await?;
    }
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        samples.push(one_bench_call(&provider, &prompt, max_output_tokens).await?);
    }
    let total_wall_ms: f64 = samples.iter().map(|sample| sample.wall_ms).sum();
    let total_output_tokens: usize = samples.iter().map(|sample| sample.output_tokens).sum();
    let total_forwards: usize = samples
        .iter()
        .filter_map(|sample| sample.completed_decode_forwards)
        .sum();
    let complete_metric_samples: Vec<&BenchSample> = samples
        .iter()
        .filter(|sample| {
            sample
                .decode_ms
                .zip(sample.completed_decode_forwards)
                .is_some_and(|(ms, forwards)| ms > 0.0 && forwards > 0)
        })
        .collect();
    let total_decode_ms: f64 = complete_metric_samples
        .iter()
        .filter_map(|sample| sample.decode_ms)
        .sum();
    let complete_forward_tps = (complete_metric_samples.len() == samples.len()
        && total_decode_ms > 0.0
        && total_forwards > 0)
        .then(|| total_forwards as f64 / (total_decode_ms / 1_000.0));
    let per_sample_complete_tps: Vec<f64> = samples
        .iter()
        .filter_map(|sample| {
            sample
                .decode_ms
                .zip(sample.completed_decode_forwards)
                .and_then(|(ms, forwards)| {
                    (ms > 0.0 && forwards > 0).then(|| forwards as f64 / (ms / 1_000.0))
                })
        })
        .collect();
    let runtime_reported_decode_tps: Vec<f64> = samples
        .iter()
        .filter_map(|sample| sample.decode_tokens_per_second.map(f64::from))
        .collect();
    let e2e_emitted_token_tps = (total_wall_ms > 0.0 && total_output_tokens > 0)
        .then(|| total_output_tokens as f64 / (total_wall_ms / 1_000.0));
    let mut receipt = json!({
        "schema": "hcli.model_benchmark.v1",
        "status": "completed",
        "runtime": preflight,
        "requested": {
            "warmup": warmup,
            "measured_runs": runs,
            "max_output_tokens": max_output_tokens,
            "prompt_blake3": blake3::hash(prompt.as_bytes()).to_hex().to_string(),
        },
        "aggregate": {
            "output_tokens": total_output_tokens,
            "wall_ms": total_wall_ms,
            "e2e_emitted_token_tps": e2e_emitted_token_tps,
            "completed_decode_forwards": total_forwards,
            "decode_ms": if complete_metric_samples.is_empty() { None } else { Some(total_decode_ms) },
            "complete_forward_tps": complete_forward_tps,
            "complete_forward_tps_quantiles": quantiles(&per_sample_complete_tps),
            "runtime_reported_decode_token_tps_quantiles": quantiles(&runtime_reported_decode_tps),
            "tps_authority": if complete_forward_tps.is_some() {
                "sum(completed_decode_forwards) / sum(decode_ms), from real runtime requests"
            } else {
                "complete-forward TPS unavailable: every measured request must expose both completed_decode_forwards and decode_ms"
            },
            "wall_time_note": "e2e_emitted_token_tps includes request, scheduler, prefill, decode, and streaming overhead; it is not a kernel decode-TPS claim",
        },
        "samples": samples,
    });
    seal(&mut receipt)?;
    let receipt_path = path_or_default(&root, options, "hcli-bench");
    write_sealed_receipt(&receipt_path, &receipt)?;
    Ok(command_envelope(
        "bench",
        json!({ "workspace": root, "receipt_path": receipt_path, "receipt": receipt }),
    ))
}

#[derive(Debug)]
struct BridgeDispatchError {
    code: HcliErrorCode,
    message: String,
    retryable: bool,
}

impl BridgeDispatchError {
    fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: HcliErrorCode::InvalidRequest,
            message: message.into(),
            retryable: false,
        }
    }

    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            code: HcliErrorCode::Unavailable,
            message: message.into(),
            retryable: true,
        }
    }

    fn model_optimization_not_available() -> Self {
        let refusal = HcliError::model_optimization_not_available();
        Self {
            code: refusal.code,
            message: refusal.message,
            retryable: refusal.retryable,
        }
    }

    fn unsupported(message: impl Into<String>) -> Self {
        Self {
            code: HcliErrorCode::Unsupported,
            message: message.into(),
            retryable: false,
        }
    }

    fn internal(error: impl std::fmt::Display) -> Self {
        Self {
            code: HcliErrorCode::Internal,
            message: compact_bridge_message(error.to_string()),
            retryable: false,
        }
    }

    fn into_wire(self) -> HcliError {
        HcliError {
            code: self.code,
            message: compact_bridge_message(self.message),
            retryable: self.retryable,
        }
    }
}

fn compact_bridge_message(message: impl Into<String>) -> String {
    let message = message.into();
    const MAX: usize = 480;
    if message.len() <= MAX {
        return message;
    }
    let mut end = MAX;
    while !message.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    format!("{}…", &message[..end])
}

fn bridge_runtime_url(request_url: Option<&str>, options: &Options) -> Option<String> {
    request_url
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .map(str::to_string)
        .or_else(|| model_url(options))
}

fn bridge_decode_telemetry(stats: &hide_core::runtime::GenerationStats) -> DecodeTelemetry {
    let (decode_ms, completed_decode_forwards, decode_forwards_per_second) =
        match (stats.decode_ms, stats.completed_decode_forwards) {
            (Some(milliseconds), Some(forwards)) if milliseconds > 0.0 && forwards > 0 => (
                Some(milliseconds),
                Some(forwards as u64),
                Some(forwards as f64 * 1_000.0 / milliseconds),
            ),
            _ => (None, None, None),
        };
    DecodeTelemetry {
        input_tokens: Some(stats.input_tokens as u64),
        output_tokens: Some(stats.output_tokens as u64),
        decode_ms,
        completed_decode_forwards,
        decode_forwards_per_second,
    }
}

fn bridge_requires_suggest_only(
    autonomy: HcliAutonomy,
) -> std::result::Result<(), BridgeDispatchError> {
    if matches!(autonomy, HcliAutonomy::SuggestOnly) {
        Ok(())
    } else {
        Err(BridgeDispatchError::unsupported(
            "this HCLI adapter exposes SuggestOnly agent autonomy only; ReadOnly and FullAuto are not silently remapped",
        ))
    }
}

fn bridge_sampler(request: &hide_backend::hcli_bridge::SamplingRequest) -> Option<SamplerProfile> {
    (request.temperature.is_some() || request.top_p.is_some() || request.seed.is_some()).then(
        || SamplerProfile {
            temperature: request.temperature.unwrap_or(0.0),
            top_k: None,
            top_p: request.top_p,
            repetition_penalty: None,
            seed: request.seed,
            deterministic: request.temperature.unwrap_or(0.0) == 0.0,
        },
    )
}

async fn bridge_direct_generate(
    request: hide_backend::hcli_bridge::GenerateRequest,
    endpoint: String,
) -> std::result::Result<GenerateResponse, BridgeDispatchError> {
    if request.session_id.is_some() {
        return Err(BridgeDispatchError::unsupported(
            "an ephemeral generate request cannot name a session; set durable=true to append contextual history",
        ));
    }
    if request.model.is_some() {
        return Err(BridgeDispatchError::unsupported(
            "runtime model selection is not exposed by the current Hawking HTTP endpoint",
        ));
    }
    if request.max_context_tokens.is_some() {
        return Err(BridgeDispatchError::unsupported(
            "per-request context-window overrides are not exposed by the current Hawking HTTP endpoint",
        ));
    }
    let provider = HttpModelProvider::new(endpoint);
    let inference = InferenceRequest {
        task_kind: "hcli.machine.generate".to_string(),
        prompt: request.prompt,
        messages: Vec::new(),
        max_output_tokens: request.max_output_tokens as usize,
        sampler: bridge_sampler(&request.sampling),
        grammar: None,
        want_logprobs: false,
        metadata: request.metadata,
    };
    let mut output = String::new();
    let mut sink = |chunk: StreamChunk| match chunk {
        StreamChunk::Token { text, .. } => {
            output.push_str(&text);
            Ok(())
        }
        StreamChunk::Done { .. } => Ok(()),
        StreamChunk::Error { message } => {
            Err(hide_core::error::HideError::RuntimeUnavailable(message))
        }
    };
    let stats = provider
        .generate(inference, &mut sink)
        .await
        .map_err(BridgeDispatchError::internal)?;
    let mut warnings = vec![
        "This JSONL adapter returns one terminal record; native token streaming remains available through the host event bus, not as JSONL token frames.".to_string(),
    ];
    if !stats
        .decode_ms
        .zip(stats.completed_decode_forwards)
        .is_some_and(|(milliseconds, forwards)| milliseconds > 0.0 && forwards > 0)
    {
        warnings.push("The runtime omitted completed_decode_forwards and/or decode_ms, so no complete-forward TPS is reported.".to_string());
    }
    Ok(GenerateResponse {
        status: OperationStatus::Completed,
        session_id: None,
        model: None,
        output: Some(output),
        stream_id: request
            .stream
            .then(|| format!("ephemeral-{}", hide_core::ids::now_ms())),
        durable_turn: None,
        telemetry: Some(bridge_decode_telemetry(&stats)),
        warnings,
    })
}

async fn bridge_generate(
    host: &BackendHost,
    options: &Options,
    request: hide_backend::hcli_bridge::GenerateRequest,
) -> std::result::Result<GenerateResponse, BridgeDispatchError> {
    let endpoint =
        bridge_runtime_url(request.runtime_url.as_deref(), options).ok_or_else(|| {
            BridgeDispatchError::unavailable(
                "generate requires params.runtime_url or --model-url/HCLI_MODEL_URL",
            )
        })?;
    if !request.durable {
        return bridge_direct_generate(request, endpoint).await;
    }
    if request.model.is_some() {
        return Err(BridgeDispatchError::unsupported(
            "runtime model selection is not exposed by the current Hawking HTTP endpoint",
        ));
    }
    if request.max_context_tokens.is_some() {
        return Err(BridgeDispatchError::unsupported(
            "per-request context-window overrides are not exposed by the current Hawking HTTP endpoint",
        ));
    }
    if bridge_sampler(&request.sampling).is_some() {
        return Err(BridgeDispatchError::unsupported(
            "durable contextual turns currently use the host's deterministic target sampler; use durable=false for explicit sampling controls",
        ));
    }
    let session_id = request
        .session_id
        .as_deref()
        .map(hide_core::ids::SessionId::from)
        .unwrap_or_else(|| host.services.session_named("hcli-bridge"));
    let turn = host
        .hcli_turn_with_output_cap(
            session_id,
            endpoint,
            request.prompt,
            Some(request.max_output_tokens as usize),
        )
        .await
        .map_err(BridgeDispatchError::internal)?;
    let mut warnings = vec![
        "The durable turn uses real compiled repository context, resolved CLAUDE instructions, and durable verified history; the output cap may be lowered by available model-window space.".to_string(),
    ];
    if request.stream {
        warnings.push("This JSONL adapter returns the terminal completion in one record; correlate native streamed tokens with stream_id through the host event bus.".to_string());
    }
    if !request.metadata.is_empty() {
        warnings.push(
            "Request metadata is not yet persisted into the durable turn event schema.".to_string(),
        );
    }
    if turn.complete_forward_tps.is_none() {
        warnings.push("The runtime omitted completed_decode_forwards and/or decode_ms, so no complete-forward TPS is reported.".to_string());
    }
    Ok(GenerateResponse {
        status: OperationStatus::Completed,
        session_id: Some(turn.session_id.as_str().to_string()),
        model: None,
        output: Some(turn.completion),
        stream_id: request.stream.then_some(turn.stream_id),
        durable_turn: Some(DurableTurnReceipt {
            user_event_id: turn.intent_event_id.as_str().to_string(),
            assistant_event_id: Some(turn.assistant_event_id.as_str().to_string()),
        }),
        telemetry: Some(bridge_decode_telemetry(&turn.generation_stats)),
        warnings,
    })
}

fn bridge_agent_realization(receipt: &Value) -> AgentRealization {
    let get_u64 = |pointer: &str| receipt.pointer(pointer).and_then(Value::as_u64);
    AgentRealization {
        transitions: get_u64("/agent/transitions_executed"),
        model_calls: get_u64("/agent/model_metrics/recorded_call_count"),
        tool_calls: get_u64("/agent/tool_activity/durable_tool_call_events"),
        subagents_total: get_u64("/agent/agent_topology/actual_subagents_total"),
        input_tokens: get_u64("/agent/model_metrics/recorded_input_tokens"),
        output_tokens: get_u64("/agent/model_metrics/recorded_output_tokens"),
        wall_elapsed_ms: get_u64("/wall_elapsed_ms"),
    }
}

fn bridge_agent_status(status: &str) -> OperationStatus {
    match status {
        "completed" => OperationStatus::Completed,
        "paused" => OperationStatus::Paused,
        "blocked_no_model_url" | "blocked_runtime_unreachable" => OperationStatus::Blocked,
        "failed" | "step_limit" => OperationStatus::Failed,
        _ => OperationStatus::Failed,
    }
}

/// Resolve JSONL agent attachment refs through the same bounded, local-only
/// evidence path used by the direct HCLI command.  A machine request carries
/// only reference identities; no source body is accepted over JSONL, and no
/// selected derivative is echoed in the bridge response.
fn bridge_agent_source_context(
    root: &std::path::Path,
    attachments: &[String],
) -> std::result::Result<Option<HcliSourceContext>, BridgeDispatchError> {
    if attachments.is_empty() {
        return Ok(None);
    }
    let sources = HcliSourceStore::open(root).map_err(BridgeDispatchError::internal)?;
    sources
        .select_context(attachments)
        .map(Some)
        .map_err(|error| BridgeDispatchError::invalid_request(error.to_string()))
}

async fn bridge_agent(
    host: &BackendHost,
    root: &std::path::Path,
    options: &Options,
    request: hide_backend::hcli_bridge::AgentRequest,
) -> std::result::Result<AgentResponse, BridgeDispatchError> {
    bridge_requires_suggest_only(request.autonomy)?;
    if request.dry_run {
        return Err(BridgeDispatchError::unsupported(
            "agent dry_run is not implemented; HCLI refuses to present a model-free receipt as an executed agent run",
        ));
    }
    if request.model.is_some() {
        return Err(BridgeDispatchError::unsupported(
            "runtime model selection is not exposed by the current Hawking HTTP endpoint",
        ));
    }
    if request.budget.max_wallclock_ms.is_some()
        || request.budget.max_tool_calls.is_some()
        || request.budget.max_subagents.is_some()
        || request.budget.search_breadth.is_some()
        || request.budget.self_consistency_k.is_some()
        || request.budget.max_context_tokens.is_some()
        || request.budget.token_budget_hint.is_some()
    {
        return Err(BridgeDispatchError::unsupported(
            "this adapter currently supports named profile plus budget.max_transitions only; other per-run budget overrides are rejected rather than silently ignored",
        ));
    }
    // Resolve before runtime admission so a stale/missing evidence ref is a
    // deterministic invalid request, never a model-free success or a surprise
    // model call. `select_context` enforces the derivative and aggregate
    // context bounds and produces receipt metadata without retaining raw bytes.
    let source_context = bridge_agent_source_context(root, &request.attachments)?;
    let endpoint = bridge_runtime_url(request.runtime_url.as_deref(), options);
    let session_id = request
        .session_id
        .as_deref()
        .map(hide_core::ids::SessionId::from)
        .or_else(|| Some(host.services.session_named("hcli-bridge-agent")));
    let profile_budget = request.profile.budget();
    let max_transitions = request
        .budget
        .max_transitions
        .unwrap_or(profile_budget.max_steps);
    let result = run_headless_audit(
        host,
        HeadlessRunConfig {
            goal: request.goal,
            model_url: endpoint,
            session_id: session_id.clone(),
            max_transitions,
            profile: request.profile,
            source_context,
        },
    )
    .await
    .map_err(BridgeDispatchError::internal)?;
    let receipt_path = root.join(".hide").join("receipts").join(format!(
        "hcli-bridge-agent-{}.json",
        hide_core::ids::now_ms()
    ));
    write_sealed_receipt(&receipt_path, &result.receipt).map_err(BridgeDispatchError::internal)?;
    let actual_session_id = result
        .receipt
        .pointer("/event_window/session_id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| session_id.map(|id| id.as_str().to_string()))
        .ok_or_else(|| {
            BridgeDispatchError::internal("agent receipt did not record a session id")
        })?;
    let run_id = result
        .receipt
        .pointer("/agent/run_id")
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut warnings = vec![
        "The executed profile remains SuggestOnly: high local compute does not grant raw tool effects.".to_string(),
        "Only budget.max_transitions was applied as a per-run override; other budget fields are rejected by this adapter.".to_string(),
    ];
    if !request.metadata.is_empty() {
        warnings.push(
            "Request metadata is not yet persisted into the durable agent run schema.".to_string(),
        );
    }
    if !request.attachments.is_empty() {
        warnings.push(format!(
            "{} explicit local evidence attachment reference(s) were selected through the bounded derivative path; identities and injection outcome are sealed in the agent receipt.",
            request.attachments.len()
        ));
    }
    Ok(AgentResponse {
        status: bridge_agent_status(result.status.as_str()),
        session_id: actual_session_id,
        run_id,
        requested_profile: request.profile,
        effective_profile: Some(request.profile),
        realized: bridge_agent_realization(&result.receipt),
        receipt_path: Some(receipt_path.display().to_string()),
        warnings,
    })
}

fn bridge_session_summary(
    session_id: hide_core::ids::SessionId,
    label: Option<String>,
) -> SessionSummary {
    SessionSummary {
        session_id: session_id.as_str().to_string(),
        label,
        state: OperationStatus::Completed,
        updated_ms: None,
    }
}

async fn bridge_session(
    host: &BackendHost,
    request: SessionRequest,
) -> std::result::Result<SessionResponse, BridgeDispatchError> {
    match request {
        SessionRequest::Create { label } => {
            let label = label.unwrap_or_else(|| format!("hcli-{}", hide_core::ids::now_ms()));
            let session_id = host.services.session_named(&label);
            Ok(SessionResponse {
                status: OperationStatus::Completed,
                action: SessionAction::Create,
                session: Some(bridge_session_summary(session_id, Some(label))),
                sessions: Vec::new(),
                warnings: Vec::new(),
            })
        }
        SessionRequest::Resume { session_id } => {
            let session = hide_core::ids::SessionId::from(session_id.clone());
            let known_by_name = host
                .services
                .key_value_store
                .list("sessions")
                .map_err(BridgeDispatchError::internal)?
                .into_iter()
                .any(|(_, value)| value.get("session_id").and_then(Value::as_str) == Some(session_id.as_str()));
            let has_events = !host
                .services
                .event_log
                .scan(Some(session.clone()), None, Some(1))
                .await
                .map_err(BridgeDispatchError::internal)?
                .is_empty();
            if !known_by_name && !has_events {
                return Err(BridgeDispatchError::unavailable(
                    "the requested session is not present in this workspace's durable registry or event log",
                ));
            }
            Ok(SessionResponse {
                status: OperationStatus::Completed,
                action: SessionAction::Resume,
                session: Some(bridge_session_summary(session, None)),
                sessions: Vec::new(),
                warnings: Vec::new(),
            })
        }
        SessionRequest::List { limit } => {
            let mut sessions: Vec<SessionSummary> = host
                .services
                .key_value_store
                .list("sessions")
                .map_err(BridgeDispatchError::internal)?
                .into_iter()
                .filter_map(|(label, value)| {
                    value
                        .get("session_id")
                        .and_then(Value::as_str)
                        .map(|id| bridge_session_summary(hide_core::ids::SessionId::from(id), Some(label)))
                })
                .collect();
            sessions.sort_by(|left, right| left.session_id.cmp(&right.session_id));
            sessions.truncate(limit as usize);
            Ok(SessionResponse {
                status: OperationStatus::Completed,
                action: SessionAction::List,
                session: None,
                sessions,
                warnings: vec!["Only named sessions are enumerable today; ephemeral headless sessions remain recoverable by id from their receipts.".to_string()],
            })
        }
        SessionRequest::Close { .. } => Err(BridgeDispatchError::unsupported(
            "HCLI does not expose session deletion/closure; durable history is intentionally retained",
        )),
    }
}

async fn bridge_status(
    host: &BackendHost,
    options: &Options,
    request: hide_backend::hcli_bridge::StatusRequest,
) -> std::result::Result<StatusResponse, BridgeDispatchError> {
    let include = |area: StatusArea| request.include.is_empty() || request.include.contains(&area);
    let host_status = host.status().await;
    let mut components = BTreeMap::new();
    if include(StatusArea::Host) {
        components.insert(
            "host".to_string(),
            ComponentStatus {
                state: ComponentState::Ready,
                detail: Some("durable workspace backend is open".to_string()),
                data: Some(
                    serde_json::to_value(&host_status).map_err(BridgeDispatchError::internal)?,
                ),
            },
        );
    }
    if include(StatusArea::Runtime) {
        let runtime = model_url(options);
        let (state, detail, data) = match runtime {
            Some(endpoint) if request.refresh => {
                let data = runtime_preflight(&endpoint).await;
                let ready = data.pointer("/health/ready").and_then(Value::as_bool) == Some(true);
                (
                    if ready { ComponentState::Ready } else { ComponentState::Down },
                    if ready { "local endpoint accepted health check".to_string() } else { "local endpoint did not pass health check".to_string() },
                    Some(data),
                )
            }
            Some(endpoint) => (
                ComponentState::Unknown,
                "runtime endpoint configured but not refreshed; pass refresh=true for a health probe".to_string(),
                Some(json!({ "endpoint": endpoint })),
            ),
            None => (
                ComponentState::Unavailable,
                "no explicit HCLI runtime endpoint configured".to_string(),
                None,
            ),
        };
        components.insert(
            "runtime".to_string(),
            ComponentStatus {
                state,
                detail: Some(detail),
                data,
            },
        );
    }
    if include(StatusArea::Models) {
        components.insert(
            "models".to_string(),
            ComponentStatus {
                state: ComponentState::Ready,
                detail: Some("registered local roles are configuration, not proof of a loaded runtime artifact".to_string()),
                data: Some(json!({
                    "roles": host_status.model_roles,
                    "model_optimization_identity": hcli_model_optimization_identity(),
                })),
            },
        );
    }
    if include(StatusArea::Agents) {
        components.insert(
            "agents".to_string(),
            ComponentStatus {
                state: ComponentState::Ready,
                detail: Some(
                    "headless AgentKernel with code-index grounding and SuggestOnly effects"
                        .to_string(),
                ),
                data: Some(json!({ "profiles": HcliProfile::ALL })),
            },
        );
    }
    if include(StatusArea::Swarms) {
        components.insert(
            "swarms".to_string(),
            ComponentStatus {
                state: ComponentState::Degraded,
                detail: Some("parallel independent analysis lanes work; graph coordinator, evidence partitioning, and worktree write isolation do not".to_string()),
                data: None,
            },
        );
    }
    if include(StatusArea::Storage) {
        components.insert(
            "storage".to_string(),
            ComponentStatus {
                state: ComponentState::Ready,
                detail: Some("workspace blobs, event log, receipts, and research ledger are durable; arbitrary uploads are not context inputs yet".to_string()),
                data: Some(json!({ "workspace": host.services.config.workspace_root })),
            },
        );
    }
    let session = request
        .session_id
        .map(|id| bridge_session_summary(hide_core::ids::SessionId::from(id), None));
    Ok(StatusResponse {
        components,
        session,
        warnings: Vec::new(),
    })
}

async fn bridge_dispatch(
    host: &BackendHost,
    root: &std::path::Path,
    options: &Options,
    operation: HcliOperation,
) -> std::result::Result<HcliResult, BridgeDispatchError> {
    match operation {
        HcliOperation::Capabilities(_) => Ok(HcliResult::Capabilities(CapabilitiesResponse {
            operations: vec![
                MethodCapability { method: HcliMethod::Capabilities, available: true, detail: Some("protocol and effective limits".to_string()) },
                MethodCapability { method: HcliMethod::ModelOptimization, available: false, detail: Some(format!("always refuses with stable code {HCLI_MODEL_OPTIMIZATION_NOT_AVAILABLE_CODE}; HCLI reports the taxonomy but never executes Condense/Gravity optimization")) },
                MethodCapability { method: HcliMethod::Generate, available: true, detail: Some("direct local generation and durable contextual turns".to_string()) },
                MethodCapability { method: HcliMethod::Agent, available: true, detail: Some("SuggestOnly headless AgentKernel; profile plus transition cap".to_string()) },
                MethodCapability { method: HcliMethod::Swarm, available: false, detail: Some("direct CLI analysis swarm exists, but this machine contract lacks a graph coordinator and hard shared budget governor".to_string()) },
                MethodCapability { method: HcliMethod::Session, available: true, detail: Some("create, resume, and list; close is intentionally unavailable".to_string()) },
                MethodCapability { method: HcliMethod::Status, available: true, detail: Some("host/runtime/storage status; optional runtime refresh".to_string()) },
            ],
            model_optimization_identity: hcli_model_optimization_identity(),
            limits: Default::default(),
            notes: vec![
                "HCLI is local-endpoint oriented and does not proxy model traffic.".to_string(),
                "Gravity is the canonical public model-optimization identity; Condense is its engine operation. This bridge reports identity only and does not claim optimization execution or promote a live V4 diagnostic into a full-model, numeric-parity, Metal, or TPS result.".to_string(),
                "Machine requests report effective behavior and reject unsupported controls instead of silently downgrading them.".to_string(),
            ],
        })),
        HcliOperation::ModelOptimization(_) => {
            Err(BridgeDispatchError::model_optimization_not_available())
        }
        HcliOperation::Generate(request) => bridge_generate(host, options, request).await.map(HcliResult::Generate),
        HcliOperation::Agent(request) => bridge_agent(host, root, options, request).await.map(HcliResult::Agent),
        HcliOperation::Swarm(_) => Err(BridgeDispatchError::unsupported(
            "the JSONL swarm method is withheld until HCLI has a real coordinator, evidence manifest, and hard shared model-call/concurrency governor; use `hcli swarm` only for the explicitly labeled independent analysis swarm",
        )),
        HcliOperation::Session(request) => bridge_session(host, request).await.map(HcliResult::Session),
        HcliOperation::Status(request) => bridge_status(host, options, request).await.map(HcliResult::Status),
    }
}

async fn cmd_bridge(subcommand: Option<&str>, options: &Options) -> Result<()> {
    match subcommand.unwrap_or("jsonl") {
        "jsonl" | "stdio" => {}
        other => bail!("unknown bridge transport {other:?}; use jsonl"),
    }
    let root = workspace(options)?;
    let host = BackendHost::open_workspace(&root)?;
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        let response = match HcliRequest::parse_jsonl_line(&line) {
            Ok(request) => {
                let id = request.id.clone();
                match request.into_validated() {
                    Ok(validated) => {
                        match bridge_dispatch(&host, &root, options, validated.operation).await {
                            Ok(result) => {
                                HcliResponse::success(id, result).unwrap_or_else(|error| {
                                    HcliResponse::failure(
                                        "invalid",
                                        HcliError {
                                            code: HcliErrorCode::Internal,
                                            message: compact_bridge_message(format!(
                                                "adapter produced invalid response: {error}"
                                            )),
                                            retryable: false,
                                        },
                                    )
                                    .expect("internal HCLI error response must validate")
                                })
                            }
                            Err(error) => HcliResponse::failure(id, error.into_wire())
                                .expect("bridge error response must validate"),
                        }
                    }
                    Err(error) => HcliResponse::failure(
                        "invalid",
                        HcliError {
                            code: HcliErrorCode::InvalidRequest,
                            message: compact_bridge_message(error.to_string()),
                            retryable: false,
                        },
                    )
                    .expect("invalid-request response must validate"),
                }
            }
            Err(error) => HcliResponse::failure(
                "invalid",
                HcliError {
                    code: HcliErrorCode::ParseError,
                    message: compact_bridge_message(error.to_string()),
                    retryable: false,
                },
            )
            .expect("parse-error response must validate"),
        };
        writeln!(
            stdout,
            "{}",
            response.to_jsonl_line().expect("response was validated")
        )?;
        stdout.flush()?;
    }
    Ok(())
}

#[derive(Debug, serde::Serialize)]
struct BenchSample {
    wall_ms: f64,
    input_tokens: usize,
    output_tokens: usize,
    decode_ms: Option<f64>,
    completed_decode_forwards: Option<usize>,
    decode_tokens_per_second: Option<f32>,
    emitted_utf8_bytes: usize,
}

async fn one_bench_call(
    provider: &HttpModelProvider,
    prompt: &str,
    max_output_tokens: usize,
) -> Result<BenchSample> {
    let request = InferenceRequest {
        task_kind: "benchmark".to_string(),
        prompt: prompt.to_string(),
        messages: Vec::new(),
        max_output_tokens,
        sampler: None,
        grammar: None,
        want_logprobs: false,
        metadata: Default::default(),
    };
    let started = Instant::now();
    let mut emitted_utf8_bytes = 0usize;
    let mut sink = |chunk: StreamChunk| {
        if let StreamChunk::Token { text, .. } = chunk {
            emitted_utf8_bytes = emitted_utf8_bytes.saturating_add(text.len());
        }
        Ok(())
    };
    let stats = provider.generate(request, &mut sink).await?;
    Ok(BenchSample {
        wall_ms: started.elapsed().as_secs_f64() * 1_000.0,
        input_tokens: stats.input_tokens,
        output_tokens: stats.output_tokens,
        decode_ms: stats.decode_ms,
        completed_decode_forwards: stats.completed_decode_forwards,
        decode_tokens_per_second: stats.decode_tokens_per_second,
        emitted_utf8_bytes,
    })
}

fn quantiles(values: &[f64]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let pick = |fraction: f64| {
        let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
        sorted[index]
    };
    json!({ "count": sorted.len(), "p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99) })
}

/// A deliberately small, HCLI-specific slash-command surface for the
/// interactive shell.  This is not a second command registry: each variant
/// below is wired to an operation the CLI already performs.  The broader HIDE
/// protocol command catalog includes UI commands with no equivalent safe HCLI
/// operation, so those commands are intentionally absent here.
#[derive(Debug, Clone, PartialEq)]
enum ReplSlashCommand {
    Help(Option<String>),
    Quit,
    Session(ReplSessionCommand),
    Status,
    Capabilities,
    Model,
    /// A read-only boundary report.  It never accepts optimization controls
    /// and never dispatches an optimizer.
    ModelOptimization,
    Agent(ReplAgentCommand),
    Swarm(ReplSwarmCommand),
    Research(ReplResearchCommand),
    Source(ReplSourceCommand),
    Context {
        attachments: Vec<String>,
    },
    Tool(ReplToolCommand),
    Chat(ReplChatCommand),
}

#[derive(Debug, Clone, PartialEq)]
enum ReplSessionCommand {
    Show,
    Select(String),
    Fork { at_seq: u64 },
}

#[derive(Debug, Clone, PartialEq)]
struct ReplAgentCommand {
    goal: String,
    profile: HcliProfile,
    max_transitions: u32,
    attachments: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
struct ReplSwarmCommand {
    goal: String,
    profile: HcliProfile,
    lanes: usize,
    max_concurrency: usize,
    max_transitions: u32,
    synthesize: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct ReplResearchCommand {
    topic: String,
    per_query_limit: usize,
    read_budget: usize,
    max_rounds: u32,
}

#[derive(Debug, Clone, PartialEq)]
struct ReplChatCommand {
    prompt: String,
    attachments: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
enum ReplSourceCommand {
    List,
    Ingest {
        path: String,
        label: Option<String>,
        mime: Option<String>,
    },
    Show(String),
}

#[derive(Debug, Clone, PartialEq)]
enum ReplToolCommand {
    List,
    Run { name: String, args: Value },
}

/// The REPL normally uses a stable named session, but a real session fork is
/// identified by its newly minted id.  Keeping the id separately avoids
/// resolving that fork through `session_named` and accidentally creating a
/// different session with the fork id as its label.
#[derive(Debug, Clone)]
struct ReplSession {
    label: String,
    id: hide_core::ids::SessionId,
}

impl ReplSession {
    fn named(host: &BackendHost, label: impl Into<String>) -> Self {
        let label = label.into();
        Self {
            id: host.services.session_named(&label),
            label,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplControl {
    Continue,
    Quit,
}

fn repl_help(topic: Option<&str>) -> String {
    const GENERAL: &str = "\
HCLI slash commands (only real HCLI operations are listed):
  /help [COMMAND]                 Show this help or details for one command
  /status                          Show current session, backend, and runtime health
  /capabilities                    Show the implemented REPL/HCLI surface and limits
  /model                           Show Gravity/Condense identity and local runtime facts
  /optimize                        Show the stable non-execution boundary for Gravity/Condense
  /session [NAME]                  Show the current session or open/select a named session
  /session fork SEQ                Fork the current durable session at event sequence SEQ
  /chat [--attach REF]... TEXT      Send an explicit contextual model turn (plain text also does this)
  /agent [--profile P] [--max-transitions N] [--attach REF]... GOAL
  /swarm [--profile P] [--lanes N] [--max-concurrency N] [--max-transitions N] [--no-synthesis] GOAL
  /research [--per-query-limit N] [--read-budget N] [--max-rounds N] TOPIC
  /source list | show ID | ingest PATH [--label NAME] [--mime TYPE]
  /context [--attach REF]...        Report the current turn/context boundary without mutating it
  /tool list | run NAME JSON       Inspect or invoke a registered, permission-gated tool
  /quit                            Leave the REPL

Profiles are balanced, power, or maximum.  They expand bounded exploration,
not tool permissions.  The wider HIDE protocol command catalog contains
surface-specific UI commands that are not silently exposed through HCLI.
Legacy colon commands remain available unchanged; use :help for their list.";

    match topic.map(|value| value.trim().to_ascii_lowercase()) {
        None => GENERAL.to_string(),
        Some(value) if value.is_empty() || value == "help" || value == "commands" => {
            GENERAL.to_string()
        }
        Some(value) if value == "session" => "\
/session
  /session                 show the active durable session projection
  /session NAME            open-or-create and select a stable named session
  /session fork SEQ        create and select a real fork at durable event sequence SEQ

Forks keep their actual SessionId in the REPL; they are not re-resolved as a
named session."
            .to_string(),
        Some(value) if value == "model" => "\
/model

Reports the shared model-optimization identity and the current local endpoint
facts. Gravity is the canonical public identity; Condense is its engine
operation. This HCLI build does not execute optimization or assert a V4
runtime. The machine-readable `public_cli` field names the live Hawking
artifact-serving surface."
            .to_string(),
        Some(value) if value == "optimize" || value == "optimization" => "\
/optimize

Reports HCLI's explicit, machine-readable non-execution boundary for
Gravity/Condense. It returns `model_optimization_not_available`; no model
weights, target, or artifact controls are accepted or applied. Use the
canonical Hawking planning command in the response only when you intend to
leave HCLI's execution boundary."
            .to_string(),
        Some(value) if value == "agent" => "\
/agent [--profile balanced|power|maximum] [--max-transitions N] [--attach REF]... GOAL

Runs the real SuggestOnly AgentKernel and writes a sealed receipt.  Larger
profiles increase finite governor budgets; they do not grant direct effects."
            .to_string(),
        Some(value) if value == "swarm" => "\
/swarm [--profile P] [--lanes N] [--max-concurrency N] [--max-transitions N] [--no-synthesis] GOAL

Runs the real parallel independent-analysis swarm and writes a sealed receipt.
It is not a worktree-isolated concurrent write swarm or a graph coordinator."
            .to_string(),
        Some(value) if value == "research" => "\
/research [--per-query-limit N] [--read-budget N] [--max-rounds N] TOPIC

Runs the existing evidence-first research pipeline.  Its current source
adapter is arXiv Atom title/abstract procurement, not general web/PDF search."
            .to_string(),
        Some(value) if value == "source" => "\
/source list
/source show OREF_OR_BLAKE3
/source ingest PATH [--label NAME] [--mime TYPE]

Ingest is bounded local-file storage.  REPL stdin is reserved for commands, so
use `hcli source ingest --stdin` outside the REPL when that input mode is needed."
            .to_string(),
        Some(value) if value == "context" => "\
/context [--attach REF]... reports the active durable session projection and
runtime context facts.  With --attach, it validates and previews the bounded
metadata-only source pack but does not inject it.  It cannot clear, compact, or
otherwise mutate session context.  /chat and /agent are the real invocation
paths that can inject an explicit attachment."
            .to_string(),
        Some(value) if value == "tool" || value == "tools" => "\
/tool list
/tool run NAME JSON_OBJECT

Tool execution goes through HCLI's real registry, permission, schema, and
sandbox policy.  Listing a tool is not an authorization grant."
            .to_string(),
        Some(value) if value == "chat" || value == "run" => "\
/chat [--attach REF]... TEXT sends one durable contextual turn in the current REPL session.
Plain text does the same thing; /chat is useful when the intended model prompt
would otherwise begin with a slash.  --attach selects bounded local evidence
derivatives only for this invocation and records the selection in its receipt."
            .to_string(),
        Some(value) => format!("No HCLI REPL command named {value:?}.\n\n{GENERAL}"),
    }
}

/// Tokenize command options without invoking a shell.  Quotes and backslash
/// escapes are only a convenience for HCLI command arguments; no expansion,
/// interpolation, or process execution occurs here.
fn split_repl_words(input: &str) -> Result<Vec<String>> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    let mut escaped = false;
    let mut token_started = false;

    for character in input.chars() {
        if escaped {
            current.push(character);
            token_started = true;
            escaped = false;
            continue;
        }
        match quote {
            Some(delimiter) if character == delimiter => quote = None,
            Some(_) if character == '\\' => escaped = true,
            Some(_) => {
                current.push(character);
                token_started = true;
            }
            None if character == '\\' => {
                escaped = true;
                token_started = true;
            }
            None if character == '\'' || character == '"' => {
                quote = Some(character);
                token_started = true;
            }
            None if character.is_whitespace() => {
                if token_started {
                    words.push(std::mem::take(&mut current));
                    token_started = false;
                }
            }
            None => {
                current.push(character);
                token_started = true;
            }
        }
    }

    if escaped {
        bail!("unterminated escape in slash command");
    }
    if quote.is_some() {
        bail!("unterminated quote in slash command");
    }
    if token_started {
        words.push(current);
    }
    Ok(words)
}

fn repl_head(input: &str) -> Option<(&str, &str)> {
    let input = input.trim();
    if input.is_empty() {
        return None;
    }
    let split_at = input
        .char_indices()
        .find_map(|(index, character)| character.is_whitespace().then_some(index))
        .unwrap_or(input.len());
    Some((&input[..split_at], input[split_at..].trim()))
}

fn parse_repl_options(arguments: &str) -> Result<Options> {
    parse_options(&split_repl_words(arguments)?)
}

fn ensure_repl_options(
    options: &Options,
    command: &str,
    allowed_values: &[&str],
    allowed_flags: &[&str],
    allow_attachments: bool,
) -> Result<()> {
    for name in options.values.keys() {
        if !allowed_values
            .iter()
            .any(|allowed| *allowed == name.as_str())
        {
            bail!("{command} does not support --{name}; use /help {command}");
        }
    }
    for name in &options.flags {
        if !allowed_flags
            .iter()
            .any(|allowed| *allowed == name.as_str())
        {
            bail!("{command} does not support --{name}; use /help {command}");
        }
    }
    if !allow_attachments && !options.attachments().is_empty() {
        bail!("{command} does not support --attach; use /help {command}");
    }
    Ok(())
}

fn repl_required_text(options: &Options, usage: &str) -> Result<String> {
    if options.positional.is_empty() {
        bail!("usage: {usage}");
    }
    let text = options.positional.join(" ");
    if text.trim().is_empty() {
        bail!("usage: {usage}");
    }
    Ok(text)
}

fn parse_repl_session(arguments: &str) -> Result<ReplSessionCommand> {
    let words = split_repl_words(arguments)?;
    match words.as_slice() {
        [] => Ok(ReplSessionCommand::Show),
        [command] if command.eq_ignore_ascii_case("show") => Ok(ReplSessionCommand::Show),
        [command, at_seq] if command.eq_ignore_ascii_case("fork") => Ok(ReplSessionCommand::Fork {
            at_seq: at_seq
                .parse()
                .map_err(|_| anyhow!("usage: /session fork EVENT_SEQUENCE"))?,
        }),
        [command, ..]
            if command.eq_ignore_ascii_case("fork")
                || command.eq_ignore_ascii_case("new")
                || command.eq_ignore_ascii_case("switch") =>
        {
            if command.eq_ignore_ascii_case("fork") {
                bail!("usage: /session fork EVENT_SEQUENCE");
            }
            let name = words[1..].join(" ");
            if name.trim().is_empty() {
                bail!("usage: /session NAME");
            }
            Ok(ReplSessionCommand::Select(name))
        }
        _ => Ok(ReplSessionCommand::Select(words.join(" "))),
    }
}

fn parse_repl_agent(arguments: &str) -> Result<ReplAgentCommand> {
    let options = parse_repl_options(arguments)?;
    ensure_repl_options(
        &options,
        "agent",
        &["profile", "max-transitions"],
        &[],
        true,
    )?;
    let profile = profile(&options, HcliProfile::Power)?;
    let max_transitions: u32 = options.number("max-transitions", profile.budget().max_steps)?;
    if max_transitions == 0 {
        bail!("/agent --max-transitions must be greater than zero");
    }
    Ok(ReplAgentCommand {
        goal: repl_required_text(
            &options,
            "/agent [--profile PROFILE] [--max-transitions N] GOAL",
        )?,
        profile,
        max_transitions,
        attachments: options.attachments().to_vec(),
    })
}

fn parse_repl_swarm(arguments: &str) -> Result<ReplSwarmCommand> {
    let options = parse_repl_options(arguments)?;
    ensure_repl_options(
        &options,
        "swarm",
        &["profile", "lanes", "max-concurrency", "max-transitions"],
        &["no-synthesis"],
        false,
    )?;
    let profile = profile(&options, HcliProfile::Power)?;
    let lanes: usize = options.number("lanes", profile.budget().search_breadth as usize)?;
    let max_concurrency: usize = options.number("max-concurrency", lanes)?;
    let max_transitions: u32 = options.number("max-transitions", profile.budget().max_steps)?;
    if lanes == 0 || max_concurrency == 0 || max_transitions == 0 {
        bail!("/swarm --lanes, --max-concurrency, and --max-transitions must be greater than zero");
    }
    Ok(ReplSwarmCommand {
        goal: repl_required_text(
            &options,
            "/swarm [--profile PROFILE] [--lanes N] [--max-concurrency N] [--max-transitions N] [--no-synthesis] GOAL",
        )?,
        profile,
        lanes,
        max_concurrency,
        max_transitions,
        synthesize: !options.flag("no-synthesis"),
    })
}

fn parse_repl_research(arguments: &str) -> Result<ReplResearchCommand> {
    let options = parse_repl_options(arguments)?;
    ensure_repl_options(
        &options,
        "research",
        &["per-query-limit", "read-budget", "max-rounds"],
        &[],
        false,
    )?;
    let per_query_limit: usize = options.number("per-query-limit", 12usize)?;
    let read_budget: usize = options.number("read-budget", 24usize)?;
    let max_rounds: u32 = options.number("max-rounds", 3u32)?;
    if per_query_limit == 0 || read_budget == 0 || max_rounds == 0 {
        bail!("/research limits must be greater than zero");
    }
    Ok(ReplResearchCommand {
        topic: repl_required_text(
            &options,
            "/research [--per-query-limit N] [--read-budget N] [--max-rounds N] TOPIC",
        )?,
        per_query_limit,
        read_budget,
        max_rounds,
    })
}

fn parse_repl_context_attachments(arguments: &str) -> Result<Vec<String>> {
    let options = parse_repl_options(arguments)?;
    ensure_repl_options(&options, "context", &[], &[], true)?;
    if !options.positional.is_empty() {
        bail!("usage: /context [--attach OREF_OR_BLAKE3]...");
    }
    Ok(options.attachments().to_vec())
}

fn parse_repl_chat(arguments: &str) -> Result<ReplChatCommand> {
    let options = parse_repl_options(arguments)?;
    ensure_repl_options(&options, "chat", &[], &[], true)?;
    Ok(ReplChatCommand {
        prompt: repl_required_text(&options, "/chat [--attach OREF_OR_BLAKE3]... TEXT")?,
        attachments: options.attachments().to_vec(),
    })
}

fn parse_repl_source(arguments: &str) -> Result<ReplSourceCommand> {
    let options = parse_repl_options(arguments)?;
    let Some(subcommand) = options.positional.first().map(String::as_str) else {
        ensure_repl_options(&options, "source", &[], &[], false)?;
        return Ok(ReplSourceCommand::List);
    };
    match subcommand.to_ascii_lowercase().as_str() {
        "list" => {
            ensure_repl_options(&options, "source list", &[], &[], false)?;
            if options.positional.len() != 1 {
                bail!("usage: /source list");
            }
            Ok(ReplSourceCommand::List)
        }
        "ingest" => {
            ensure_repl_options(
                &options,
                "source ingest",
                &["label", "mime"],
                &["stdin"],
                false,
            )?;
            if options.flag("stdin") {
                bail!("/source ingest --stdin is unavailable in the REPL because stdin carries commands; run `hcli source ingest --stdin` outside the REPL");
            }
            if options.positional.len() != 2 {
                bail!("usage: /source ingest PATH [--label NAME] [--mime TYPE]");
            }
            Ok(ReplSourceCommand::Ingest {
                path: options.positional[1].clone(),
                label: options.value("label").map(str::to_string),
                mime: options.value("mime").map(str::to_string),
            })
        }
        "show" => {
            ensure_repl_options(&options, "source show", &["ref", "hash"], &[], false)?;
            let by_ref = options.value("ref");
            let by_hash = options.value("hash");
            if by_ref.is_some() && by_hash.is_some() {
                bail!("/source show accepts one identifier: --ref OREF or --hash BLAKE3");
            }
            let positional_identifier =
                (options.positional.len() == 2).then(|| options.positional[1].clone());
            if options.positional.len() > 2
                || (positional_identifier.is_some() && by_ref.or(by_hash).is_some())
            {
                bail!("/source show accepts exactly one identifier");
            }
            let identifier = by_ref
                .or(by_hash)
                .map(str::to_string)
                .or(positional_identifier)
                .ok_or_else(|| anyhow!("usage: /source show OREF_OR_BLAKE3"))?;
            Ok(ReplSourceCommand::Show(identifier))
        }
        _ => bail!("unknown /source subcommand {subcommand:?}; use list, show, or ingest"),
    }
}

fn parse_repl_tool(arguments: &str) -> Result<ReplToolCommand> {
    let Some((subcommand, remainder)) = repl_head(arguments) else {
        return Ok(ReplToolCommand::List);
    };
    match subcommand.to_ascii_lowercase().as_str() {
        "list" => {
            if !remainder.is_empty() {
                bail!("usage: /tool list");
            }
            Ok(ReplToolCommand::List)
        }
        "run" => {
            let Some((name, raw_args)) = repl_head(remainder) else {
                bail!("usage: /tool run NAME JSON_OBJECT");
            };
            if raw_args.is_empty() {
                bail!("usage: /tool run NAME JSON_OBJECT");
            }
            let args: Value = serde_json::from_str(raw_args)
                .with_context(|| "/tool run JSON_OBJECT must be valid JSON")?;
            if !args.is_object() {
                bail!("/tool run JSON_OBJECT must be a JSON object");
            }
            Ok(ReplToolCommand::Run {
                name: name.to_string(),
                args,
            })
        }
        _ => bail!("unknown /tool subcommand {subcommand:?}; use list or run"),
    }
}

fn parse_repl_slash_command(line: &str) -> Result<ReplSlashCommand> {
    let Some((name, arguments)) = line.strip_prefix('/').and_then(repl_head) else {
        bail!("slash commands must start with /COMMAND");
    };
    match name.to_ascii_lowercase().as_str() {
        "help" | "h" | "?" | "commands" => {
            let topics = split_repl_words(arguments)?;
            if topics.len() > 1 {
                bail!("usage: /help [COMMAND]");
            }
            Ok(ReplSlashCommand::Help(topics.into_iter().next()))
        }
        "quit" | "exit" | "q" => {
            if !arguments.is_empty() {
                bail!("usage: /quit");
            }
            Ok(ReplSlashCommand::Quit)
        }
        "session" => Ok(ReplSlashCommand::Session(parse_repl_session(arguments)?)),
        "status" => {
            if !arguments.is_empty() {
                bail!("usage: /status");
            }
            Ok(ReplSlashCommand::Status)
        }
        "capabilities" | "caps" => {
            if !arguments.is_empty() {
                bail!("usage: /capabilities");
            }
            Ok(ReplSlashCommand::Capabilities)
        }
        "model" => {
            if !arguments.is_empty() {
                bail!("usage: /model");
            }
            Ok(ReplSlashCommand::Model)
        }
        "optimize" | "optimization" => {
            if !arguments.is_empty() {
                bail!("usage: /optimize");
            }
            Ok(ReplSlashCommand::ModelOptimization)
        }
        "agent" => Ok(ReplSlashCommand::Agent(parse_repl_agent(arguments)?)),
        "swarm" => Ok(ReplSlashCommand::Swarm(parse_repl_swarm(arguments)?)),
        "research" => Ok(ReplSlashCommand::Research(parse_repl_research(arguments)?)),
        "source" => Ok(ReplSlashCommand::Source(parse_repl_source(arguments)?)),
        "context" => Ok(ReplSlashCommand::Context {
            attachments: parse_repl_context_attachments(arguments)?,
        }),
        "tool" | "tools" => Ok(ReplSlashCommand::Tool(parse_repl_tool(arguments)?)),
        "chat" | "run" => Ok(ReplSlashCommand::Chat(parse_repl_chat(arguments)?)),
        _ => bail!("unknown slash command /{name}; use /help"),
    }
}

async fn emit_repl_context(
    host: &BackendHost,
    endpoint: &str,
    session: &ReplSession,
    source_context: Option<&HcliSourceContext>,
) -> Result<()> {
    let projection = host.rebuild_session_projection(session.id.clone()).await?;
    let runtime = runtime_preflight(endpoint).await;
    emit(
        json!({
            "command": "context.show",
            "session": { "label": session.label, "session_id": session.id },
            "projection": projection,
            "runtime_context": runtime["context"].clone(),
            "explicit_source_context": source_context
                .map(HcliSourceContext::receipt_json)
                .unwrap_or_else(|| json!({ "status": "not_requested" })),
            "scope": {
                "durable_contextual_turns": true,
                "repl_mutates_source_attachments": false,
                "note": "This is an inspection command. Selected source metadata is not injected here. A run must explicitly opt into the bounded source pack."
            },
        }),
        false,
    )
}

async fn emit_repl_status(host: &BackendHost, endpoint: &str, session: &ReplSession) -> Result<()> {
    let projection = host.rebuild_session_projection(session.id.clone()).await?;
    let backend = host.status().await;
    let runtime = runtime_preflight(endpoint).await;
    emit(
        json!({
            "command": "status",
            "session": { "label": session.label, "session_id": session.id, "projection": projection },
            "backend": backend,
            "runtime": runtime,
        }),
        false,
    )
}

fn selected_repl_source_context(
    root: &std::path::Path,
    attachments: &[String],
) -> Result<Option<HcliSourceContext>> {
    let options = Options {
        attachments: attachments.to_vec(),
        ..Options::default()
    };
    selected_source_context(root, &options)
}

async fn repl_send_turn(
    host: &BackendHost,
    endpoint: &str,
    session: &ReplSession,
    prompt: &str,
    source_context: Option<HcliSourceContext>,
) -> Result<()> {
    let turn = host
        .hcli_turn_with_output_cap_and_source_context(
            session.id.clone(),
            endpoint.to_string(),
            prompt,
            None,
            source_context,
        )
        .await?;
    println!("{}", turn.completion);
    if let Some(tps) = turn.complete_forward_tps {
        eprintln!("complete-forward TPS: {:.2}", tps);
    } else {
        eprintln!("complete-forward TPS: unavailable (runtime did not report completed forwards + decode ms)");
    }
    Ok(())
}

async fn execute_repl_slash_command(
    command: ReplSlashCommand,
    root: &std::path::Path,
    host: &BackendHost,
    endpoint: &str,
    session: &mut ReplSession,
) -> Result<ReplControl> {
    match command {
        ReplSlashCommand::Help(topic) => {
            eprintln!("{}", repl_help(topic.as_deref()));
        }
        ReplSlashCommand::Quit => return Ok(ReplControl::Quit),
        ReplSlashCommand::Session(command) => match command {
            ReplSessionCommand::Show => {
                let projection = host.rebuild_session_projection(session.id.clone()).await?;
                emit(
                    json!({
                        "command": "session.show",
                        "session": { "label": session.label, "session_id": session.id },
                        "projection": projection,
                    }),
                    false,
                )?;
            }
            ReplSessionCommand::Select(label) => {
                *session = ReplSession::named(host, label);
                emit(
                    json!({
                        "command": "session.select",
                        "session": { "label": session.label, "session_id": session.id },
                        "note": "Named sessions are open-or-create and durable in this workspace.",
                    }),
                    false,
                )?;
            }
            ReplSessionCommand::Fork { at_seq } => {
                let source_session = session.id.clone();
                let (forked_session, projection) =
                    host.fork_session(source_session.clone(), at_seq).await?;
                *session = ReplSession {
                    label: format!("fork:{}", forked_session.as_str()),
                    id: forked_session.clone(),
                };
                emit(
                    json!({
                        "command": "session.fork",
                        "source_session": source_session,
                        "at_seq": at_seq,
                        "forked_session": forked_session,
                        "selected_session": { "label": session.label, "session_id": session.id },
                        "projection": projection,
                    }),
                    false,
                )?;
            }
        },
        ReplSlashCommand::Status => emit_repl_status(host, endpoint, session).await?,
        ReplSlashCommand::Capabilities => {
            emit(
                json!({
                    "command": "capabilities",
                    "backend": host.status().await,
                    "runtime": runtime_preflight(endpoint).await,
                    "model_optimization_identity": hcli_model_optimization_identity(),
                    "repl_realized_commands": [
                        "help", "status", "capabilities", "model", "optimize", "session", "chat", "agent", "swarm",
                        "research", "source", "context", "tool", "quit"
                    ],
                    "catalog_scope_note": "The protocol command catalog also covers GUI and protocol surfaces. This REPL exposes only operations with an implemented HCLI mapping.",
                }),
                false,
            )?;
        }
        ReplSlashCommand::Model => {
            emit(
                json!({
                    "command": "model.inspect",
                    "model_optimization_identity": hcli_model_optimization_identity(),
                    "runtime": runtime_preflight(endpoint).await,
                    "note": "Gravity is the canonical public model-optimization identity and Condense is its engine operation. This HCLI surface reports naming and live endpoint facts only; it does not execute optimization or promote a live V4 diagnostic into a full-model, numeric-parity, Metal, or TPS result.",
                }),
                false,
            )?;
        }
        ReplSlashCommand::ModelOptimization => {
            emit(cmd_model_optimization_status(&Options::default()), false)?;
        }
        ReplSlashCommand::Agent(config) => {
            let attachments = config.attachments.clone();
            let source_context = selected_repl_source_context(root, &attachments)?;
            let result = run_headless_audit(
                host,
                HeadlessRunConfig {
                    goal: config.goal,
                    model_url: Some(endpoint.to_string()),
                    session_id: None,
                    max_transitions: config.max_transitions,
                    profile: config.profile,
                    source_context,
                },
            )
            .await?;
            let receipt_path = root
                .join(".hide")
                .join("receipts")
                .join(format!("hcli-repl-agent-{}.json", hide_core::ids::now_ms()));
            write_sealed_receipt(&receipt_path, &result.receipt)?;
            emit(
                json!({
                    "command": "agent",
                    "status": result.status.as_str(),
                    "profile": config.profile,
                    "max_transitions": config.max_transitions,
                    "attachments_requested": attachments,
                    "receipt_path": receipt_path,
                    "receipt": result.receipt,
                    "effect_policy": "SuggestOnly; compute profile does not grant raw effects.",
                }),
                false,
            )?;
        }
        ReplSlashCommand::Swarm(config) => {
            let result = run_parallel_analysis_swarm(
                host,
                HcliSwarmConfig {
                    goal: config.goal,
                    model_url: Some(endpoint.to_string()),
                    profile: config.profile,
                    lanes: config.lanes,
                    max_concurrency: config.max_concurrency,
                    max_transitions: config.max_transitions,
                    synthesize: config.synthesize,
                },
            )
            .await?;
            let receipt_path = root
                .join(".hide")
                .join("receipts")
                .join(format!("hcli-repl-swarm-{}.json", hide_core::ids::now_ms()));
            write_sealed_receipt(&receipt_path, &result.receipt)?;
            emit(
                json!({
                    "command": "swarm",
                    "complete": result.complete,
                    "profile": config.profile,
                    "lanes": config.lanes,
                    "max_concurrency": config.max_concurrency,
                    "max_transitions": config.max_transitions,
                    "synthesis": result.synthesis,
                    "receipt_path": receipt_path,
                    "receipt": result.receipt,
                    "limitation": "This is the real independent analysis swarm, not an isolated concurrent write swarm or full graph coordinator.",
                }),
                false,
            )?;
        }
        ReplSlashCommand::Research(config) => {
            let result = run_hcli_research(
                host,
                HcliResearchConfig {
                    topic: config.topic,
                    model_url: Some(endpoint.to_string()),
                    per_query_limit: config.per_query_limit,
                    read_budget: config.read_budget,
                    max_rounds: config.max_rounds,
                },
            )
            .await?;
            let receipt_path = root.join(".hide").join("receipts").join(format!(
                "hcli-repl-research-{}.json",
                hide_core::ids::now_ms()
            ));
            write_sealed_receipt(&receipt_path, &result.receipt)?;
            emit(
                json!({
                    "command": "research",
                    "complete": result.complete,
                    "receipt_path": receipt_path,
                    "receipt": result.receipt,
                }),
                false,
            )?;
        }
        ReplSlashCommand::Source(command) => {
            let sources = HcliSourceStore::open(root)?;
            match command {
                ReplSourceCommand::List => emit(
                    json!({
                        "command": "source.list",
                        "evidence_root": sources.root(),
                        "source_references": sources.list()?,
                        "raw_body_included": false,
                    }),
                    false,
                )?,
                ReplSourceCommand::Ingest { path, label, mime } => {
                    let result = sources.ingest_file(&path, label, mime)?;
                    emit(
                        json!({
                            "command": "source.ingest",
                            "evidence_root": sources.root(),
                            "receipt_path": result.receipt_path,
                            "source": result.receipt["object"].clone(),
                            "receipt": result.receipt,
                            "context_note": "This REPL command stores bounded local evidence. It does not itself attach the source to a model run.",
                        }),
                        false,
                    )?;
                }
                ReplSourceCommand::Show(identifier) => emit(
                    json!({
                        "command": "source.show",
                        "evidence_root": sources.root(),
                        "source": sources.show(&identifier)?,
                    }),
                    false,
                )?,
            }
        }
        ReplSlashCommand::Context { attachments } => {
            let source_context = selected_repl_source_context(root, &attachments)?;
            emit_repl_context(host, endpoint, session, source_context.as_ref()).await?
        }
        ReplSlashCommand::Tool(command) => match command {
            ReplToolCommand::List => emit(
                json!({
                    "command": "tool.list",
                    "tools": host.tools.specs(),
                    "note": "Tools remain subject to the real permission and sandbox policy.",
                }),
                false,
            )?,
            ReplToolCommand::Run { name, args } => {
                let result = host
                    .dispatch_tool(session.id.clone(), None, ToolCall::new(name.clone(), args))
                    .await?;
                emit(
                    json!({
                        "command": "tool.run",
                        "session_id": session.id,
                        "tool": name,
                        "result": result,
                    }),
                    false,
                )?;
            }
        },
        ReplSlashCommand::Chat(command) => {
            let source_context = selected_repl_source_context(root, &command.attachments)?;
            repl_send_turn(host, endpoint, session, &command.prompt, source_context).await?
        }
    }
    Ok(ReplControl::Continue)
}

async fn cmd_repl(options: &Options) -> Result<()> {
    let root = workspace(options)?;
    let endpoint =
        model_url(options).ok_or_else(|| anyhow!("--model-url is required for hcli repl"))?;
    let host = BackendHost::open_workspace(&root)?;
    let mut session = ReplSession::named(&host, options.value("session").unwrap_or("hcli"));
    eprintln!(
        "HCLI REPL — session={}; /help for slash commands, :help for legacy commands, /quit to leave.",
        session.label
    );
    let stdin = io::stdin();
    loop {
        eprint!("hcli[{}]> ", session.label);
        io::stderr().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('/') {
            match parse_repl_slash_command(line) {
                Ok(command) => {
                    match execute_repl_slash_command(command, &root, &host, &endpoint, &mut session)
                        .await
                    {
                        Ok(ReplControl::Continue) => {}
                        Ok(ReplControl::Quit) => break,
                        Err(error) => eprintln!(
                            "slash command failed: {error}\nTry /help for supported operations."
                        ),
                    }
                }
                Err(error) => {
                    eprintln!("invalid slash command: {error}\nTry /help for supported operations.")
                }
            }
            continue;
        }
        if let Some(command) = line.strip_prefix(':') {
            match command.trim() {
                "quit" | "q" | "exit" => break,
                "help" => {
                    eprintln!(":help  :quit  :session NAME  :status  :agent GOAL  :swarm GOAL  :research TOPIC\nplain text sends a contextual model turn");
                }
                "status" => {
                    let projection = host.rebuild_session_projection(session.id.clone()).await?;
                    emit(
                        json!({ "session": session.label, "session_id": session.id, "projection": projection }),
                        false,
                    )?;
                }
                other if other.starts_with("session ") => {
                    let name = other.trim_start_matches("session ").trim();
                    if name.is_empty() {
                        eprintln!("usage: :session NAME");
                    } else {
                        session = ReplSession::named(&host, name);
                    }
                }
                other if other.starts_with("agent ") => {
                    let goal = other.trim_start_matches("agent ").trim();
                    let result = run_headless_audit(
                        &host,
                        HeadlessRunConfig {
                            goal: goal.to_string(),
                            model_url: Some(endpoint.clone()),
                            session_id: None,
                            max_transitions: HcliProfile::Power.budget().max_steps,
                            profile: HcliProfile::Power,
                            source_context: None,
                        },
                    )
                    .await?;
                    emit(
                        json!({ "status": result.status.as_str(), "receipt": result.receipt }),
                        false,
                    )?;
                }
                other if other.starts_with("swarm ") => {
                    let goal = other.trim_start_matches("swarm ").trim();
                    let power = HcliProfile::Power;
                    let result = run_parallel_analysis_swarm(
                        &host,
                        HcliSwarmConfig {
                            goal: goal.to_string(),
                            model_url: Some(endpoint.clone()),
                            profile: power,
                            lanes: power.budget().search_breadth as usize,
                            max_concurrency: power.budget().search_breadth as usize,
                            max_transitions: power.budget().max_steps,
                            synthesize: true,
                        },
                    )
                    .await?;
                    emit(
                        json!({ "complete": result.complete, "receipt": result.receipt }),
                        false,
                    )?;
                }
                other if other.starts_with("research ") => {
                    let topic = other.trim_start_matches("research ").trim();
                    let result = run_hcli_research(
                        &host,
                        HcliResearchConfig {
                            topic: topic.to_string(),
                            model_url: Some(endpoint.clone()),
                            ..HcliResearchConfig::default()
                        },
                    )
                    .await?;
                    emit(
                        json!({ "complete": result.complete, "receipt": result.receipt }),
                        false,
                    )?;
                }
                _ => eprintln!("unknown command; :help"),
            }
            continue;
        }
        repl_send_turn(&host, &endpoint, &session, line, None).await?;
    }
    Ok(())
}

fn cmd_serve(options: &Options) -> Result<()> {
    let weights = options.value("weights");
    let gravity = options.value("gravity");
    if weights.is_some() == gravity.is_some() {
        bail!("hcli serve requires exactly one of --weights or --gravity");
    }
    let bin = options
        .value("hawking-bin")
        .map(str::to_string)
        .or_else(|| std::env::var("HCLI_HAWKING_BIN").ok())
        .unwrap_or_else(|| "hawking".to_string());
    let addr = options.value("addr").unwrap_or("127.0.0.1:8080");
    let mut command = std::process::Command::new(&bin);
    if let Some(profile) = options.value("profile") {
        command.args(["--profile", profile]);
    }
    command.arg("serve").args(["--addr", addr]);
    if let Some(path) = weights {
        command.args(["--weights", path]);
    }
    if let Some(path) = gravity {
        command.args(["--gravity", path]);
    }
    eprintln!(
        "Launching local Hawking runtime via {bin:?}; HCLI does not proxy or alter model bytes. \
         --gravity selects an already-built artifact only and never runs model optimization."
    );
    let status = command
        .status()
        .with_context(|| format!("could not start {bin:?}"))?;
    if !status.success() {
        bail!("hawking serve exited with {status}");
    }
    Ok(())
}

fn prompt(options: &Options) -> Result<String> {
    if options.value("prompt").is_some() && options.value("prompt-file").is_some() {
        bail!("use exactly one of --prompt or --prompt-file");
    }
    let prompt = if let Some(value) = options.value("prompt") {
        value.to_string()
    } else if let Some(path) = options.value("prompt-file") {
        std::fs::read_to_string(path)
            .with_context(|| format!("could not read --prompt-file {path}"))?
    } else if !options.positional.is_empty() {
        options.positional.join(" ")
    } else {
        bail!("--prompt or --prompt-file is required");
    };
    if prompt.trim().is_empty() {
        bail!("prompt must not be empty");
    }
    Ok(prompt)
}

fn goal(options: &Options) -> Result<String> {
    if options.value("goal").is_some() && options.value("goal-file").is_some() {
        bail!("use exactly one of --goal or --goal-file");
    }
    let goal = if let Some(value) = options.value("goal") {
        value.to_string()
    } else if let Some(path) = options.value("goal-file") {
        std::fs::read_to_string(path)
            .with_context(|| format!("could not read --goal-file {path}"))?
    } else if !options.positional.is_empty() {
        options.positional.join(" ")
    } else {
        bail!("--goal or --goal-file is required");
    };
    if goal.trim().is_empty() {
        bail!("goal must not be empty");
    }
    Ok(goal)
}

fn seal(receipt: &mut Value) -> Result<()> {
    let bytes = serde_json::to_vec(receipt)?;
    receipt["content_blake3"] = json!(blake3::hash(&bytes).to_hex().to_string());
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty()
        || matches!(
            argv.first().map(String::as_str),
            Some("--help" | "-h" | "help")
        )
    {
        print!("{}", usage());
        return Ok(());
    }
    let command = argv[0].as_str();
    let (subcommand, option_start) = match command {
        "model" | "session" | "source" | "tool" | "bridge" | "goal" => {
            (argv.get(1).map(String::as_str), 2)
        }
        _ => (None, 1),
    };
    let options = parse_options(&argv[option_start..])?;
    if options.flag("help") {
        print!("{}", usage());
        return Ok(());
    }
    if command == "serve" {
        return cmd_serve(&options);
    }
    if command == "repl" {
        return cmd_repl(&options).await;
    }
    if command == "bridge" {
        return cmd_bridge(subcommand, &options).await;
    }
    let value = match command {
        "capabilities" => cmd_capabilities(&options).await?,
        "model" if matches!(subcommand, Some("inspect" | "status")) => {
            cmd_capabilities(&options).await?
        }
        "model" if matches!(subcommand, Some("optimize" | "optimization")) => {
            cmd_model_optimization_status(&options)
        }
        "run" | "chat" => cmd_run(&options).await?,
        "agent" => cmd_agent(&options).await?,
        "goal" if matches!(subcommand, Some("run") | None) => cmd_agent(&options).await?,
        "swarm" => cmd_swarm(&options).await?,
        "research" => cmd_research(&options).await?,
        "bench" => cmd_bench(&options).await?,
        "session" => cmd_session(subcommand, &options).await?,
        "source" => cmd_source(subcommand, &options).await?,
        "tool" => cmd_tool(subcommand, &options).await?,
        other => bail!("unknown HCLI command {other:?}\n\n{}", usage()),
    };
    emit(value, options.flag("json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn option_parser_keeps_flags_values_and_positionals_distinct() {
        let args = vec![
            "--model-url".to_string(),
            "http://127.0.0.1:8080".to_string(),
            "--json".to_string(),
            "review this".to_string(),
        ];
        let parsed = parse_options(&args).unwrap();
        assert_eq!(parsed.value("model-url"), Some("http://127.0.0.1:8080"));
        assert!(parsed.flag("json"));
        assert_eq!(parsed.positional, vec!["review this"]);
    }

    #[test]
    fn option_parser_accepts_repeatable_explicit_source_attachments() {
        let args = vec![
            "--attach".to_string(),
            "oref_first".to_string(),
            "--attach".to_string(),
            "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
        ];
        let parsed = parse_options(&args).unwrap();
        assert_eq!(
            parsed.attachments(),
            &[
                "oref_first".to_string(),
                "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_string(),
            ]
        );
    }

    #[test]
    fn quantiles_are_null_when_runtime_reported_no_samples() {
        assert!(quantiles(&[]).is_null());
        let value = quantiles(&[10.0, 20.0, 30.0]);
        assert_eq!(value["p50"].as_f64(), Some(20.0));
        assert_eq!(value["p95"].as_f64(), Some(30.0));
    }

    #[test]
    fn model_slash_command_and_help_use_the_shared_gravity_condense_identity() {
        assert!(matches!(
            parse_repl_slash_command("/model").unwrap(),
            ReplSlashCommand::Model
        ));
        let error = parse_repl_slash_command("/model now").unwrap_err();
        assert!(error.to_string().contains("usage: /model"));
        let help = repl_help(Some("model"));
        assert!(help.contains("Gravity is the canonical public identity"));
        assert!(help.contains("Condense is its engine"));

        let identity = hcli_model_optimization_identity();
        assert_eq!(identity.canonical_identity, "gravity");
        assert_eq!(identity.condense_engine_operation.id, "condense");
        assert!(!identity.claim_limits.hcli_executes_optimization);
        assert!(!identity.claim_limits.v4_runtime_claim);
        assert!(usage().contains("Gravity is the public model-optimization identity"));
    }

    #[test]
    fn optimize_surfaces_are_explicit_machine_readable_non_execution_boundaries() {
        assert!(matches!(
            parse_repl_slash_command("/optimize").unwrap(),
            ReplSlashCommand::ModelOptimization
        ));
        let error = parse_repl_slash_command("/optimize --weights model.safetensors").unwrap_err();
        assert!(error.to_string().contains("usage: /optimize"));
        assert!(repl_help(Some("optimize")).contains(HCLI_MODEL_OPTIMIZATION_NOT_AVAILABLE_CODE));

        let options = parse_options(&[
            "--weights".to_string(),
            "model.safetensors".to_string(),
            "--json".to_string(),
            "unapplied-target".to_string(),
        ])
        .unwrap();
        let status = cmd_model_optimization_status(&options);
        assert_eq!(status["command"], "model.optimize");
        assert_eq!(status["result"]["status"], "not_available");
        assert_eq!(
            status["result"]["refusal"]["code"],
            HCLI_MODEL_OPTIMIZATION_NOT_AVAILABLE_CODE
        );
        assert_eq!(
            status["result"]["model_optimization_identity"]["execution_boundary"]["available"],
            false
        );
        let legacy_compatibility =
            &status["result"]["model_optimization_identity"]["legacy_compatibility"];
        assert!(legacy_compatibility["field_tag_mapping_id"]
            .as_str()
            .is_some());
        assert!(legacy_compatibility["schema_aliases"]
            .as_array()
            .unwrap()
            .iter()
            .all(|alias| {
                alias["deprecated"] == true
                    && alias["superseded_by"].as_str().is_some()
                    && alias["field_tag_mapping_id"].as_str().is_some()
            }));
        assert!(legacy_compatibility["command_aliases"]
            .as_array()
            .unwrap()
            .iter()
            .all(|alias| {
                alias["deprecated"] == true
                    && alias["superseded_by"].as_str().is_some()
                    && alias["maps_to_engine_operation"].as_str().is_some()
            }));
        assert_eq!(status["result"]["request"]["controls_accepted"], false);
        assert!(status["result"]["request"]["received_control_names"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "--weights"));
        assert!(usage().contains("hcli model optimize"));
    }

    #[tokio::test]
    async fn bridge_model_optimization_dispatch_uses_the_stable_refusal_code() {
        let directory = tempfile::tempdir().unwrap();
        let host = BackendHost::open_workspace(directory.path()).unwrap();
        let error = bridge_dispatch(
            &host,
            directory.path(),
            &Options::default(),
            HcliOperation::ModelOptimization(Default::default()),
        )
        .await
        .unwrap_err()
        .into_wire();
        assert_eq!(error.code, HcliErrorCode::ModelOptimizationNotAvailable);
        assert_eq!(
            serde_json::to_value(error).unwrap()["code"],
            HCLI_MODEL_OPTIMIZATION_NOT_AVAILABLE_CODE
        );
    }

    #[test]
    fn bridge_agent_source_attachments_select_bounded_provenance_without_source_text() {
        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("bridge-agent-evidence.txt");
        let selected_fact = "HCLI_BRIDGE_SELECTED_FACT_5c31";
        std::fs::write(&source_path, selected_fact).unwrap();
        let sources = HcliSourceStore::open(directory.path()).unwrap();
        let ingested = sources.ingest_file(&source_path, None, None).unwrap();
        let source_context = bridge_agent_source_context(
            directory.path(),
            &[ingested.reference.id.as_str().to_string()],
        )
        .unwrap()
        .expect("selected bridge source context");
        let evidence = source_context.receipt_json();
        assert_eq!(evidence["status"], "selected");
        assert_eq!(
            evidence["selected_sources"][0]["reference_id"],
            ingested.reference.id.as_str()
        );
        assert!(
            !evidence.to_string().contains(selected_fact),
            "the bridge context receipt must preserve identity/provenance but never echo selected derivative text"
        );
    }

    #[tokio::test]
    async fn bridge_agent_rejects_missing_local_attachment_before_runtime_work() {
        let directory = tempfile::tempdir().unwrap();
        let host = BackendHost::open_workspace(directory.path()).unwrap();
        let error = bridge_agent(
            &host,
            directory.path(),
            &Options::default(),
            hide_backend::hcli_bridge::AgentRequest {
                goal: "do not start an agent for a missing evidence reference".to_string(),
                session_id: None,
                runtime_url: Some("http://127.0.0.1:9".to_string()),
                model: None,
                profile: HcliProfile::Balanced,
                budget: Default::default(),
                attachments: vec!["oref_missing_evidence".to_string()],
                autonomy: HcliAutonomy::SuggestOnly,
                acknowledge_effects: false,
                dry_run: false,
                metadata: BTreeMap::new(),
            },
        )
        .await
        .unwrap_err()
        .into_wire();
        assert_eq!(error.code, HcliErrorCode::InvalidRequest);
        assert!(!directory.path().join(".hide").join("receipts").exists());
    }

    #[test]
    fn slash_agent_preserves_profile_budget_and_explicit_attachments() {
        let command = parse_repl_slash_command(
            "/agent --profile maximum --max-transitions 17 --attach oref_design review \"the source pipeline\"",
        )
        .unwrap();
        let ReplSlashCommand::Agent(agent) = command else {
            panic!("expected an agent slash command");
        };
        assert_eq!(agent.profile, HcliProfile::Maximum);
        assert_eq!(agent.max_transitions, 17);
        assert_eq!(agent.attachments, vec!["oref_design"]);
        assert_eq!(agent.goal, "review the source pipeline");
    }

    #[test]
    fn slash_chat_and_context_accept_only_explicit_attachment_controls() {
        let command = parse_repl_slash_command(
            "/chat --attach oref_first --attach blake3:second summarize the evidence",
        )
        .unwrap();
        let ReplSlashCommand::Chat(chat) = command else {
            panic!("expected a chat slash command");
        };
        assert_eq!(chat.prompt, "summarize the evidence");
        assert_eq!(chat.attachments, vec!["oref_first", "blake3:second"]);

        let command = parse_repl_slash_command("/context --attach oref_first").unwrap();
        assert!(matches!(
            command,
            ReplSlashCommand::Context { attachments }
                if attachments == vec!["oref_first"]
        ));
        let error = parse_repl_slash_command("/context clear").unwrap_err();
        assert!(error.to_string().contains("usage: /context"));
    }

    #[test]
    fn slash_source_handles_quoted_path_without_shell_expansion() {
        let command = parse_repl_slash_command(
            "/source ingest \"/tmp/design notes.md\" --label \"design notes\" --mime text/markdown",
        )
        .unwrap();
        assert!(matches!(
            command,
            ReplSlashCommand::Source(ReplSourceCommand::Ingest { path, label, mime })
                if path == "/tmp/design notes.md"
                    && label.as_deref() == Some("design notes")
                    && mime.as_deref() == Some("text/markdown")
        ));
        let error = parse_repl_slash_command("/source ingest --stdin").unwrap_err();
        assert!(error.to_string().contains("unavailable in the REPL"));
        assert!(split_repl_words("unterminated \"quote").is_err());
    }

    #[test]
    fn slash_session_fork_and_tool_json_map_to_real_operations() {
        let session = parse_repl_slash_command("/session fork 42").unwrap();
        assert_eq!(
            session,
            ReplSlashCommand::Session(ReplSessionCommand::Fork { at_seq: 42 })
        );

        let tool =
            parse_repl_slash_command("/tool run read_file {\"path\":\"Cargo.toml\"}").unwrap();
        assert!(matches!(
            tool,
            ReplSlashCommand::Tool(ReplToolCommand::Run { name, args })
                if name == "read_file" && args["path"] == "Cargo.toml"
        ));
    }

    #[test]
    fn slash_commands_reject_controls_that_would_be_silently_ignored() {
        let error =
            parse_repl_slash_command("/swarm --attach oref_one inspect the repo").unwrap_err();
        assert!(error.to_string().contains("does not support --attach"));

        let error = parse_repl_slash_command("/agent --not-real yes audit").unwrap_err();
        assert!(error.to_string().contains("does not support --not-real"));

        let help = repl_help(Some("context"));
        assert!(help.contains("metadata-only"));
    }
}
